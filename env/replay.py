"""Replay-mode trace loader for the public demo.

Pre-recorded attack traces live in ``data/traces/`` as JSON files. Each trace
captures one (attack_type, steps) combination of: payload, stage-by-stage
defense verdicts, agent output, and final outcome — with realistic timings so
the frontend can stream them back as if the attack were running live.

This module provides read-only access; the trace files are produced offline
by ``scripts/generate_traces.py`` running on Colab against the trained
checkpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_TRACES_DIR = Path(__file__).resolve().parent.parent / "data" / "traces"
DEFAULT_HIGHLIGHT_DIR = Path(__file__).resolve().parent.parent / "data" / "highlights"


# ---------------------------------------------------------------------------
# Attack-type catalogue (driven by scenario target_category)
# ---------------------------------------------------------------------------

ATTACK_TYPES: List[Dict[str, str]] = [
    {
        "id": "email_exfiltration",
        "label": "Email Exfiltration",
        "description": "Trick the email agent into forwarding session data to an attacker address.",
        "icon": "📧",
        "scenario_id": "email_exfil_001",
    },
    {
        "id": "forbidden_tool",
        "label": "Forbidden Tool Call",
        "description": "Force the agent to invoke a tool that is explicitly off-limits.",
        "icon": "🔧",
        "scenario_id": "email_forbidden_001",
    },
    {
        "id": "prompt_leak",
        "label": "System-Prompt Leak",
        "description": "Get the agent to reveal its hidden system prompt verbatim.",
        "icon": "🔓",
        "scenario_id": "email_leak_001",
    },
    {
        "id": "rag_injection",
        "label": "RAG Document Injection",
        "description": "Hide an injection inside a retrieved document the agent has to summarise.",
        "icon": "📄",
        "scenario_id": "rag_exfil_001",
    },
]

STEP_OPTIONS: List[int] = [50, 100, 300, 500, 1000, 1500]


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

class TraceStore:
    """Loads every trace JSON under traces_dir into memory at startup."""

    def __init__(
        self,
        traces_dir: Path | str = DEFAULT_TRACES_DIR,
        highlight_dir: Path | str = DEFAULT_HIGHLIGHT_DIR,
    ) -> None:
        self.traces_dir = Path(traces_dir)
        self.highlight_dir = Path(highlight_dir)
        self._traces: Dict[str, Dict[str, Any]] = {}
        self._highlight: Optional[Dict[str, Any]] = None
        self._load()

    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.traces_dir.exists():
            logger.warning("Traces dir not found: %s — replay mode will return 404s.", self.traces_dir)
            return

        for path in sorted(self.traces_dir.glob("*.json")):
            try:
                with path.open() as f:
                    trace = json.load(f)
                key = self._key(trace["attack_type"], trace["steps"])
                self._traces[key] = trace
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed trace %s: %s", path.name, exc)

        logger.info("Loaded %d traces from %s", len(self._traces), self.traces_dir)

        # Highlight reel — first match wins.
        if self.highlight_dir.exists():
            for path in sorted(self.highlight_dir.glob("*.json")):
                try:
                    with path.open() as f:
                        self._highlight = json.load(f)
                    logger.info("Loaded highlight reel from %s", path.name)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping malformed highlight %s: %s", path.name, exc)

    @staticmethod
    def _key(attack_type: str, steps: int) -> str:
        return f"{attack_type}__{steps}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attack_types(self) -> List[Dict[str, str]]:
        return ATTACK_TYPES

    def step_options(self) -> List[int]:
        return STEP_OPTIONS

    def get(self, attack_type: str, steps: int) -> Optional[Dict[str, Any]]:
        # Exact match first
        trace = self._traces.get(self._key(attack_type, steps))
        if trace is not None:
            return trace
        # Fallback: nearest available step count for that attack type
        candidates = [
            (s, t) for s, t in (
                (int(k.rsplit("__", 1)[1]), v) for k, v in self._traces.items()
                if k.startswith(f"{attack_type}__")
            )
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: abs(x[0] - steps))
        chosen = candidates[0][1]
        # Mark that we substituted, so the frontend can show a hint if it likes
        return {**chosen, "step_substituted": True, "requested_steps": steps}

    def highlight(self) -> Optional[Dict[str, Any]]:
        return self._highlight

    def aggregate_stats(self) -> Dict[str, Any]:
        """Roll-up across all loaded traces — feeds the homepage stats banner."""
        if not self._traces:
            return {
                "trace_count": 0,
                "pg2_bypass_rate": None,
                "fw_bypass_rate": None,
                "composed_bypass_rate": None,
            }

        n = len(self._traces)
        pg2_pass = sum(1 for t in self._traces.values() if t.get("outcome", {}).get("broke_pg2"))
        fw_pass = sum(1 for t in self._traces.values() if t.get("outcome", {}).get("broke_fw"))
        composed = sum(
            1 for t in self._traces.values()
            if t.get("outcome", {}).get("broke_pg2")
            and t.get("outcome", {}).get("broke_fw")
            and t.get("outcome", {}).get("task_succeeded")
        )
        return {
            "trace_count": n,
            "pg2_bypass_rate": pg2_pass / n,
            "fw_bypass_rate": fw_pass / n,
            "composed_bypass_rate": composed / n,
        }
