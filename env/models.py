"""Frozen Pydantic schemas for InjectArena.

These types are the public contract between the attacker, the environment,
the defense wrappers, and any OpenEnv client. Do not change fields without
updating Section 5.1 of CLAUDE.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


InjectionSlot = Literal[
    "email_body",
    "calendar_description",
    "rag_document",
    "tool_description",
]

TargetCategory = Literal["exfiltration", "forbidden_tool", "prompt_leak"]

PG2Label = Literal["benign", "injection"]


class AttemptRecord(BaseModel):
    payload: str
    pg2_label: PG2Label
    firewall_flagged: bool
    task_success: bool


class InjectObservation(BaseModel):
    scenario_id: str
    target_behavior: str
    target_category: TargetCategory
    agent_system_prompt: str
    user_query: str
    injection_slot: InjectionSlot
    tool_surface: List[str]
    canary_string: Optional[str] = None
    previous_attempts: List[AttemptRecord] = Field(default_factory=list)
    attempts_remaining: int
    max_payload_tokens: int = 512


class InjectAction(BaseModel):
    payload: str
    strategy_tag: Optional[str] = None


class StepResult(BaseModel):
    observation: InjectObservation
    reward: float
    done: bool
    info: Dict[str, Any]
