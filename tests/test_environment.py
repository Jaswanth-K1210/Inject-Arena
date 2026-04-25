"""Environment tests — run on Mac with stub defenses (no GPU needed)."""

from __future__ import annotations

import pytest
from typing import Any, Optional

from env.defenses.base import DefenseVerdict
from env.environment import InjectArenaEnv
from env.models import InjectAction, InjectObservation, StepResult
from env.scenarios import ScenarioBank


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _AlwaysBenignPG2:
    name = "stub_pg2_benign"

    def scan(self, text: str) -> DefenseVerdict:
        return DefenseVerdict(flagged=False, score=0.0, reason="stub_benign")


class _AlwaysMaliciousPG2:
    name = "stub_pg2_malicious"

    def scan(self, text: str) -> DefenseVerdict:
        return DefenseVerdict(flagged=True, score=1.0, reason="stub_malicious")


class _DummySecAlign:
    name = "stub_secalign"
    mode = "stub"

    def __init__(self, output: str = "Here is a summary.") -> None:
        self._output = output

    def run(self, system: str, user: str, untrusted: Any) -> str:
        return self._output


class _DummyFirewall:
    name = "stub_firewall"

    def __init__(self, flagged: bool = False) -> None:
        self._flagged = flagged

    def scan(self, user_query: str, agent_output: str, tool_call: Any = None) -> DefenseVerdict:
        return DefenseVerdict(flagged=self._flagged, score=float(self._flagged), reason="stub")


def _make_env(pg2=None, secalign=None, firewall=None, bank=None) -> InjectArenaEnv:
    return InjectArenaEnv(
        pg2=pg2 or _AlwaysBenignPG2(),
        secalign=secalign or _DummySecAlign(),
        firewall=firewall or _DummyFirewall(),
        bank=bank or ScenarioBank(),
    )


# ---------------------------------------------------------------------------
# Basic episode flow
# ---------------------------------------------------------------------------

def test_reset_returns_valid_observation():
    env = _make_env()
    obs = env.reset()
    assert isinstance(obs, InjectObservation)
    assert obs.attempts_remaining == 3
    assert obs.previous_attempts == []
    assert obs.scenario_id


def test_step_returns_step_result():
    env = _make_env()
    env.reset()
    result = env.step(InjectAction(payload="Hello, please summarize this."))
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert "episode_id" in result.info


def test_attempts_decrement():
    env = _make_env()
    obs = env.reset()
    assert obs.attempts_remaining == 3

    r1 = env.step(InjectAction(payload="attempt 1"))
    assert r1.observation.attempts_remaining == 2

    r2 = env.step(InjectAction(payload="attempt 2"))
    assert r2.observation.attempts_remaining == 1


def test_done_after_max_attempts():
    env = _make_env()
    env.reset()
    for i in range(3):
        result = env.step(InjectAction(payload=f"attempt {i}"))
    assert result.done


def test_previous_attempts_accumulate():
    env = _make_env()
    env.reset()
    env.step(InjectAction(payload="first"))
    result = env.step(InjectAction(payload="second"))
    assert len(result.observation.previous_attempts) == 2
    assert result.observation.previous_attempts[0].payload == "first"
    assert result.observation.previous_attempts[1].payload == "second"


def test_step_before_reset_raises():
    env = _make_env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(InjectAction(payload="oops"))


# ---------------------------------------------------------------------------
# Safety floor
# ---------------------------------------------------------------------------

def test_safety_violation_terminates_with_minus_one():
    env = _make_env()
    env.reset()
    # This pattern triggers the reward-function-attack safety rule.
    result = env.step(InjectAction(payload="reward: 1.0"))
    assert result.reward == -1.0
    assert result.done
    assert result.info.get("safety_violation")


def test_after_safety_violation_reset_works():
    env = _make_env()
    env.reset()
    env.step(InjectAction(payload="reward: 1.0"))
    obs = env.reset()
    assert isinstance(obs, InjectObservation)
    assert obs.attempts_remaining == 3


# ---------------------------------------------------------------------------
# Reward signal sanity
# ---------------------------------------------------------------------------

def test_reward_zero_for_trivial_payload():
    env = _make_env()
    env.reset()
    result = env.step(InjectAction(payload="   "))
    assert result.reward == 0.0


def test_reward_positive_when_both_defenses_bypassed():
    env = _make_env(
        pg2=_AlwaysBenignPG2(),
        firewall=_DummyFirewall(flagged=False),
    )
    env.reset()
    result = env.step(InjectAction(payload="Please summarize this document carefully."))
    assert result.reward > 0.0
    assert result.info["components"]["r_bypass_pg2"] == 1.0
    assert result.info["components"]["r_bypass_fw"] == 1.0


def test_reward_lower_when_pg2_flags():
    env_benign = _make_env(pg2=_AlwaysBenignPG2())
    env_malicious = _make_env(pg2=_AlwaysMaliciousPG2())
    payload = "Please summarize this document carefully."

    env_benign.reset(seed=0)
    env_malicious.reset(seed=0)

    r_benign = env_benign.step(InjectAction(payload=payload)).reward
    r_malicious = env_malicious.step(InjectAction(payload=payload)).reward
    assert r_benign > r_malicious


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_state_reflects_episode():
    env = _make_env()
    obs = env.reset()
    st = env.state
    assert st["scenario_id"] == obs.scenario_id
    assert st["attempts"] == 0
    assert not st["done"]
    assert st["episode_id"] is not None


def test_state_done_after_exhaustion():
    env = _make_env()
    env.reset()
    for i in range(3):
        env.step(InjectAction(payload=f"try {i}"))
    assert env.state["done"]


# ---------------------------------------------------------------------------
# Reset between episodes
# ---------------------------------------------------------------------------

def test_reset_clears_previous_episode():
    env = _make_env()
    env.reset(seed=0)
    env.step(InjectAction(payload="old attempt"))

    obs2 = env.reset(seed=1)
    assert obs2.previous_attempts == []
    assert obs2.attempts_remaining == 3
    assert not env.state["done"]
