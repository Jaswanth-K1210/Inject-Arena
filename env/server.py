"""FastAPI server — OpenEnv-compatible HTTP interface for InjectArena.

Endpoints:
  GET  /health         → {"status": "ok", "defense_mode": str}
  POST /reset          → InjectObservation
  POST /step           → StepResult
  GET  /state          → current episode state dict

Environment variables:
  USE_STUB_DEFENSES=true  — use in-process stub defenses (no GPU needed).
                            Set automatically in the Dockerfile for HF Spaces.
  HF_TOKEN                — required for real defense loading.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .environment import InjectArenaEnv
from .models import InjectAction, InjectObservation, StepResult
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env, _defense_mode
    bank = ScenarioBank()
    use_stub = os.environ.get("USE_STUB_DEFENSES", "").strip().lower() in ("1", "true", "yes")
    if use_stub:
        _env = _build_stub_env(bank)
        _defense_mode = "stub"
        logger.info("InjectArena server started with STUB defenses.")
    else:
        _env = _build_real_env(bank)
        _defense_mode = "real"
        logger.info("InjectArena server started with REAL defenses.")
    yield
    if _env is not None:
        _env.close()


app = FastAPI(title="InjectArena", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    scenario_id: Optional[str] = None
    seed: Optional[int] = None
    split: str = "train"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "defense_mode": _defense_mode}


@app.post("/reset", response_model=InjectObservation)
def reset(req: ResetRequest = ResetRequest()):
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")
    obs = _env.reset(scenario_id=req.scenario_id, seed=req.seed, split=req.split)
    return obs


@app.post("/step", response_model=StepResult)
def step(action: InjectAction):
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")
    try:
        result = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")
    return _env.state
