"""Frozen defense interface. Used by Phase 2 reward + Phase 3 wrappers.

This file is a contract: every defense (Llama Prompt Guard 2, Meta-SecAlign
agent, LlamaFirewall) returns a ``DefenseVerdict``. The reward function
only depends on ``.flagged`` and ``.score``; per-defense reasons are kept
for logging and error analysis.
"""

from __future__ import annotations

from typing import Optional, Protocol

from pydantic import BaseModel


class DefenseVerdict(BaseModel):
    flagged: bool
    score: float
    reason: Optional[str] = None


class Defense(Protocol):
    name: str

    def scan(self, **kwargs) -> DefenseVerdict: ...
