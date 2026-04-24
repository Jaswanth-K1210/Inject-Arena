"""Scenario bank loader.

A scenario describes one prompt-injection opportunity: the agent's
system prompt, the user's benign query, the untrusted channel the
attacker gets to control, the tool surface, the attacker's goal,
and the verifier spec that determines whether the goal was reached.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


DEFAULT_BANK_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "bank_v1.jsonl"


class ScenarioBank:
    """Read-only collection of scenarios loaded from a JSONL file."""

    def __init__(self, path: Path | str = DEFAULT_BANK_PATH) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Scenario bank not found at {self.path}")
        self._scenarios: List[Dict[str, Any]] = self._load(self.path)
        self._by_id: Dict[str, Dict[str, Any]] = {s["scenario_id"]: s for s in self._scenarios}
        if len(self._by_id) != len(self._scenarios):
            raise ValueError("Duplicate scenario_id in bank")

    @staticmethod
    def _load(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSON at line {i} of {path}: {e}") from e
                ScenarioBank._validate(rec, i)
                out.append(rec)
        return out

    _REQUIRED_KEYS = {
        "scenario_id",
        "target_category",
        "target_behavior",
        "target_spec",
        "agent_system_prompt",
        "user_query",
        "injection_slot",
        "tool_surface",
        "split",
    }
    _VALID_SLOTS = {"email_body", "calendar_description", "rag_document", "tool_description"}
    _VALID_TARGETS = {"exfiltration", "forbidden_tool", "prompt_leak"}
    _VALID_SPLITS = {"train", "eval"}

    @classmethod
    def _validate(cls, rec: Dict[str, Any], line_no: int) -> None:
        missing = cls._REQUIRED_KEYS - rec.keys()
        if missing:
            raise ValueError(f"Scenario at line {line_no} missing keys: {sorted(missing)}")
        if rec["injection_slot"] not in cls._VALID_SLOTS:
            raise ValueError(f"Bad injection_slot at line {line_no}: {rec['injection_slot']}")
        if rec["target_category"] not in cls._VALID_TARGETS:
            raise ValueError(f"Bad target_category at line {line_no}: {rec['target_category']}")
        if rec["split"] not in cls._VALID_SPLITS:
            raise ValueError(f"Bad split at line {line_no}: {rec['split']}")
        if not isinstance(rec["tool_surface"], list):
            raise ValueError(f"tool_surface must be a list at line {line_no}")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._scenarios)

    def by_split(self, split: str) -> List[Dict[str, Any]]:
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}, got {split!r}")
        return [s for s in self._scenarios if s["split"] == split]

    def by_id(self, scenario_id: str) -> Dict[str, Any]:
        return self._by_id[scenario_id]

    def sample(self, seed: Optional[int] = None, split: Optional[str] = None) -> Dict[str, Any]:
        pool = self.by_split(split) if split else list(self._scenarios)
        if not pool:
            raise ValueError(f"Empty scenario pool for split={split}")
        rng = random.Random(seed)
        return rng.choice(pool)
