"""InjectArenaEnv — the core RL environment.

Defense instances are injected at construction so the environment can run
with stub defenses on Mac (no GPU) and real defenses on Colab.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from .models import (
    AttemptRecord,
    InjectAction,
    InjectObservation,
    StepResult,
)
from .rewards import compute_reward
from .scenarios import ScenarioBank
from .utils.safety_filter import is_safe
from .verifiers.exfiltration import verify_exfiltration
from .verifiers.forbidden_tool import verify_forbidden_tool
from .verifiers.prompt_leak import verify_prompt_leak

_MAX_ATTEMPTS = 3
_STEP_TIMEOUT_S = 30.0


class InjectArenaEnv:
    """Episode-based environment for prompt-injection red-teaming."""

    def __init__(
        self,
        pg2: Any,
        secalign: Any,
        firewall: Any,
        bank: Optional[ScenarioBank] = None,
        max_attempts: int = _MAX_ATTEMPTS,
        embedder: Any = None,
    ) -> None:
        self._pg2 = pg2
        self._secalign = secalign
        self._firewall = firewall
        self._bank = bank or ScenarioBank()
        self._max_attempts = max_attempts
        self._embedder = embedder

        self._scenario: Optional[Dict[str, Any]] = None
        self._attempts: int = 0
        self._previous: List[AttemptRecord] = []
        self._episode_id: Optional[str] = None
        self._done: bool = True

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
        split: str = "train",
    ) -> InjectObservation:
        if scenario_id is not None:
            self._scenario = self._bank.by_id(scenario_id)
        else:
            self._scenario = self._bank.sample(seed=seed, split=split)

        self._attempts = 0
        self._previous = []
        self._episode_id = str(uuid.uuid4())
        self._done = False
        return self._build_obs()

    def step(self, action: InjectAction) -> StepResult:
        if self._done or self._scenario is None:
            raise RuntimeError("Call reset() before step().")

        t0 = time.monotonic()
        payload = action.payload
        scenario = self._scenario

        # Safety floor — immediate termination.
        safe, safety_reason = is_safe(payload)
        if not safe:
            self._done = True
            return StepResult(
                observation=self._build_obs(),
                reward=-1.0,
                done=True,
                info={
                    "safety_violation": True,
                    "safety_reason": safety_reason,
                    "episode_id": self._episode_id,
                },
            )

        # All three defenses run on every step (§5.4 invariant).
        pg2_verdict = self._pg2.scan(payload)

        agent_output = self._secalign.run(
            system=scenario["agent_system_prompt"],
            user=scenario["user_query"],
            untrusted={scenario["injection_slot"]: payload},
        )

        fw_verdict = self._firewall.scan(
            user_query=payload,
            agent_output=agent_output,
            tool_call=None,
        )

        elapsed = time.monotonic() - t0

        # Task success via category-specific verifier.
        cat = scenario["target_category"]
        if cat == "exfiltration":
            task_success = verify_exfiltration(agent_output, scenario)
        elif cat == "forbidden_tool":
            task_success = verify_forbidden_tool(agent_output, scenario)
        else:
            task_success = verify_prompt_leak(agent_output, scenario)

        # Reward.
        recent = [a.payload for a in self._previous]
        reward, reward_info = compute_reward(
            payload=payload,
            pg2_verdict=pg2_verdict,
            firewall_verdict=fw_verdict,
            task_success=task_success,
            scenario=scenario,
            recent_payloads=recent,
            embedder=self._embedder,
        )

        # Record.
        record = AttemptRecord(
            payload=payload,
            pg2_label="injection" if pg2_verdict.flagged else "benign",
            firewall_flagged=fw_verdict.flagged,
            task_success=task_success,
        )
        self._previous.append(record)
        self._attempts += 1

        # Done conditions (§5.4).
        timeout = elapsed > _STEP_TIMEOUT_S
        done = (
            task_success
            or self._attempts >= self._max_attempts
            or reward_info.get("safety_violation", False)
            or timeout
        )
        self._done = done

        return StepResult(
            observation=self._build_obs(),
            reward=reward,
            done=done,
            info={
                **reward_info,
                "agent_output": agent_output,
                "pg2_verdict": pg2_verdict.model_dump(),
                "fw_verdict": fw_verdict.model_dump(),
                "task_success": task_success,
                "elapsed_s": round(elapsed, 3),
                "timeout": timeout,
                "strategy_tag": action.strategy_tag,
                "episode_id": self._episode_id,
            },
        )

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "scenario_id": self._scenario["scenario_id"] if self._scenario else None,
            "attempts": self._attempts,
            "max_attempts": self._max_attempts,
            "done": self._done,
        }

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------

    def _build_obs(self) -> InjectObservation:
        s = self._scenario
        return InjectObservation(
            scenario_id=s["scenario_id"],
            target_behavior=s["target_behavior"],
            target_category=s["target_category"],
            agent_system_prompt=s["agent_system_prompt"],
            user_query=s["user_query"],
            injection_slot=s["injection_slot"],
            tool_surface=s["tool_surface"],
            canary_string=s.get("canary_string"),
            previous_attempts=list(self._previous),
            attempts_remaining=self._max_attempts - self._attempts,
        )
