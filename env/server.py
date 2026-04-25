"""FastAPI server — OpenEnv-compatible HTTP interface + replay-mode demo API.

OpenEnv endpoints (used by the trainer and any external client):
  GET  /health             → {"status": "ok", "defense_mode": str, "mode": str}
  POST /reset              → InjectObservation
  POST /step               → StepResult
  GET  /state              → current episode state dict

Demo endpoints (used by the public frontend on Hugging Face Spaces):
  GET  /api/attack-types   → list of attack types + step options
  POST /api/attack         → request a trace; body {attack_type, steps}
  GET  /api/stream/{key}   → Server-Sent Events stream of the trace timeline
  GET  /api/highlight      → pre-computed highlight reel for the homepage
  GET  /api/stats          → aggregate bypass-rate stats

Static frontend (when present):
  GET  /                   → frontend/index.html
  GET  /static/*           → frontend/* (CSS, JS, assets)

Environment variables
---------------------
  USE_STUB_DEFENSES=true     Use in-process stub defenses (no GPU). Default in
                             the Dockerfile so HF Spaces boots without GPUs.
  INJECTARENA_MODE=replay    Serve pre-recorded traces from data/traces/ via
                             the /api/* endpoints. Use ``live`` on a paid GPU
                             Space to run real attacks.
  HF_TOKEN                   Required only for real defense loading (live mode).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .environment import InjectArenaEnv
from .models import InjectAction, InjectObservation, StepResult
from .replay import TraceStore
from .scenarios import ScenarioBank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub defenses (no GPU — used in HF Space and Mac tests via USE_STUB_DEFENSES)
# ---------------------------------------------------------------------------

class _StubVerdict(BaseModel):
    flagged: bool = False
    score: float = 0.0
    reason: Optional[str] = "stub"


class _StubPG2:
    name = "stub_pg2"

    def scan(self, text: str) -> _StubVerdict:
        return _StubVerdict()


class _StubSecAlign:
    name = "stub_secalign"
    mode = "stub"

    def run(self, system: str, user: str, untrusted: Any) -> str:
        slot = next(iter(untrusted), "input")
        content = next(iter(untrusted.values()), "")
        return f"[STUB] Processed {slot}: {str(content)[:80]}"


class _StubFirewall:
    name = "stub_firewall"

    def scan(self, user_query: str, agent_output: str, tool_call: Any = None) -> _StubVerdict:
        return _StubVerdict()


def _build_stub_env(bank: ScenarioBank) -> InjectArenaEnv:
    return InjectArenaEnv(
        pg2=_StubPG2(),
        secalign=_StubSecAlign(),
        firewall=_StubFirewall(),
        bank=bank,
    )


def _build_real_env(bank: ScenarioBank) -> InjectArenaEnv:
    from .defenses.prompt_guard import PromptGuard
    from .defenses.secalign_agent import SecAlignAgent
    from .defenses.llama_firewall import FirewallWrapper
    from .utils.embedding_cache import EmbeddingCache

    pg2 = PromptGuard()
    secalign = SecAlignAgent()
    firewall = FirewallWrapper(prompt_guard_fallback=pg2)
    embedder = EmbeddingCache()
    return InjectArenaEnv(pg2=pg2, secalign=secalign, firewall=firewall, bank=bank, embedder=embedder)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_env: Optional[InjectArenaEnv] = None
_defense_mode: str = "unknown"
_serve_mode: str = "live"   # "replay" | "live"
_trace_store: Optional[TraceStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env, _defense_mode, _serve_mode, _trace_store

    _serve_mode = os.environ.get("INJECTARENA_MODE", "live").strip().lower()
    if _serve_mode not in ("live", "replay"):
        logger.warning("Unknown INJECTARENA_MODE=%s — defaulting to live.", _serve_mode)
        _serve_mode = "live"

    # Trace store is needed in both modes (highlight reel is always replay-driven).
    _trace_store = TraceStore()

    if _serve_mode == "live":
        bank = ScenarioBank()
        use_stub = os.environ.get("USE_STUB_DEFENSES", "").strip().lower() in ("1", "true", "yes")
        if use_stub:
            _env = _build_stub_env(bank)
            _defense_mode = "stub"
            logger.info("InjectArena server: live mode, STUB defenses.")
        else:
            _env = _build_real_env(bank)
            _defense_mode = "real"
            logger.info("InjectArena server: live mode, REAL defenses.")
    else:
        _defense_mode = "n/a"
        logger.info("InjectArena server: replay mode (no defenses loaded).")

    yield

    if _env is not None:
        _env.close()


app = FastAPI(title="InjectArena", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# OpenEnv request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    scenario_id: Optional[str] = None
    seed: Optional[int] = None
    split: str = "train"


class AttackRequest(BaseModel):
    attack_type: str
    steps: int


# ---------------------------------------------------------------------------
# Health + OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "defense_mode": _defense_mode,
        "mode": _serve_mode,
    }


@app.post("/reset", response_model=InjectObservation)
def reset(req: ResetRequest = ResetRequest()):
    if _env is None:
        raise HTTPException(status_code=503, detail="Live mode disabled. Use /api/attack instead.")
    return _env.reset(scenario_id=req.scenario_id, seed=req.seed, split=req.split)


@app.post("/step", response_model=StepResult)
def step(action: InjectAction):
    if _env is None:
        raise HTTPException(status_code=503, detail="Live mode disabled. Use /api/attack instead.")
    try:
        return _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=503, detail="Live mode disabled.")
    return _env.state


# ---------------------------------------------------------------------------
# Demo API (replay mode)
# ---------------------------------------------------------------------------

@app.get("/api/attack-types")
def api_attack_types():
    if _trace_store is None:
        raise HTTPException(status_code=503, detail="Trace store not initialised.")
    return {
        "attack_types": _trace_store.attack_types(),
        "step_options": _trace_store.step_options(),
    }


@app.post("/api/attack")
def api_attack(req: AttackRequest):
    if _trace_store is None:
        raise HTTPException(status_code=503, detail="Trace store not initialised.")
    trace = _trace_store.get(req.attack_type, req.steps)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trace available for attack_type={req.attack_type} steps={req.steps}",
        )
    # Returns the full trace immediately for clients that don't want streaming.
    # The streaming endpoint below paces the events out over time for animation.
    return trace


@app.get("/api/stream/{attack_type}/{steps}")
async def api_stream(attack_type: str, steps: int):
    if _trace_store is None:
        raise HTTPException(status_code=503, detail="Trace store not initialised.")
    trace = _trace_store.get(attack_type, steps)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trace available for attack_type={attack_type} steps={steps}",
        )

    async def event_stream() -> AsyncIterator[bytes]:
        # First event: trace metadata
        meta = {
            "type": "meta",
            "attack_type": trace.get("attack_type"),
            "steps": trace.get("steps"),
            "scenario_id": trace.get("scenario_id"),
        }
        yield _sse(meta)

        prev_t = 0.0
        for ev in trace.get("timeline", []):
            t = float(ev.get("t", prev_t))
            await asyncio.sleep(max(0.0, t - prev_t))
            prev_t = t
            yield _sse({"type": "event", **ev})

        # Final event: outcome
        yield _sse({"type": "outcome", **trace.get("outcome", {})})
        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/highlight")
def api_highlight():
    if _trace_store is None:
        raise HTTPException(status_code=503, detail="Trace store not initialised.")
    trace = _trace_store.highlight()
    if trace is None:
        raise HTTPException(status_code=404, detail="No highlight reel available yet.")
    return trace


@app.get("/api/stats")
def api_stats():
    if _trace_store is None:
        raise HTTPException(status_code=503, detail="Trace store not initialised.")
    return _trace_store.aggregate_stats()


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api/* takes precedence)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_PLOTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "plots"

if _PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=_PLOTS_DIR), name="plots")

if _FRONTEND_DIR.exists() and (_FRONTEND_DIR / "index.html").exists():
    # Static assets (CSS/JS) live under /static/...
    assets_dir = _FRONTEND_DIR
    app.mount("/static", StaticFiles(directory=assets_dir), name="static")

    @app.get("/")
    def root():
        return FileResponse(_FRONTEND_DIR / "index.html")
else:
    @app.get("/")
    def root():
        return {
            "service": "InjectArena",
            "version": "1.0.0",
            "mode": _serve_mode,
            "docs": "/docs",
            "endpoints": [
                "/health", "/reset", "/step", "/state",
                "/api/attack-types", "/api/attack", "/api/stream/{type}/{steps}",
                "/api/highlight", "/api/stats",
            ],
        }
