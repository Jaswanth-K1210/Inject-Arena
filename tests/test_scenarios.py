"""Tests for env.scenarios.ScenarioBank."""

from __future__ import annotations

from pathlib import Path

import pytest

from env.scenarios import DEFAULT_BANK_PATH, ScenarioBank


@pytest.fixture(scope="module")
def bank() -> ScenarioBank:
    return ScenarioBank()


def test_bank_loads(bank: ScenarioBank) -> None:
    assert len(bank) >= 10
    assert DEFAULT_BANK_PATH.exists()


def test_every_scenario_has_required_fields(bank: ScenarioBank) -> None:
    required = {
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
    for s in bank:
        missing = required - s.keys()
        assert not missing, f"{s.get('scenario_id')!r} missing {missing}"


def test_splits_are_disjoint_and_present(bank: ScenarioBank) -> None:
    train_ids = {s["scenario_id"] for s in bank.by_split("train")}
    eval_ids = {s["scenario_id"] for s in bank.by_split("eval")}
    assert train_ids, "train split is empty"
    assert eval_ids, "eval split is empty"
    assert not (train_ids & eval_ids), "train/eval IDs overlap"


def test_sample_returns_valid_record(bank: ScenarioBank) -> None:
    s = bank.sample(seed=0)
    assert s["scenario_id"]
    assert s["injection_slot"] in {
        "email_body",
        "calendar_description",
        "rag_document",
        "tool_description",
    }
    assert s["target_category"] in {"exfiltration", "forbidden_tool", "prompt_leak"}


def test_sample_seed_is_deterministic(bank: ScenarioBank) -> None:
    a = bank.sample(seed=42)
    b = bank.sample(seed=42)
    assert a["scenario_id"] == b["scenario_id"]


def test_sample_can_filter_split(bank: ScenarioBank) -> None:
    s = bank.sample(seed=1, split="eval")
    assert s["split"] == "eval"


def test_by_id_round_trip(bank: ScenarioBank) -> None:
    any_id = next(iter(bank))["scenario_id"]
    assert bank.by_id(any_id)["scenario_id"] == any_id


def test_unknown_split_raises(bank: ScenarioBank) -> None:
    with pytest.raises(ValueError):
        bank.by_split("test")


def test_matrix_coverage(bank: ScenarioBank) -> None:
    """We expect each of the four slots and each of the three target
    categories to appear at least once across the bank. 10 scenarios
    is enough to cover both axes (4 + 3 = 7 minimum)."""
    slots = {s["injection_slot"] for s in bank}
    targets = {s["target_category"] for s in bank}
    assert slots == {"email_body", "calendar_description", "rag_document", "tool_description"}
    assert targets == {"exfiltration", "forbidden_tool", "prompt_leak"}


def test_benign_refs_present_and_balanced() -> None:
    """The stealth reward depends on having benign reference text per channel."""
    import json

    path = Path(__file__).resolve().parent.parent / "scenarios" / "benign_refs.jsonl"
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts[rec["channel"]] = counts.get(rec["channel"], 0) + 1
    expected_channels = {"email_body", "calendar_description", "rag_document", "tool_description"}
    assert set(counts) == expected_channels
    for ch, n in counts.items():
        assert n >= 20, f"{ch} has only {n} benign refs; need >= 20"
    assert sum(counts.values()) >= 80


def test_pydantic_models_round_trip() -> None:
    """A scenario dict's relevant fields should populate an InjectObservation."""
    from env.models import InjectObservation

    bank = ScenarioBank()
    s = bank.sample(seed=0)
    obs = InjectObservation(
        scenario_id=s["scenario_id"],
        target_behavior=s["target_behavior"],
        target_category=s["target_category"],
        agent_system_prompt=s["agent_system_prompt"],
        user_query=s["user_query"],
        injection_slot=s["injection_slot"],
        tool_surface=s["tool_surface"],
        canary_string=s.get("canary_string"),
        attempts_remaining=3,
    )
    assert obs.scenario_id == s["scenario_id"]
    assert obs.attempts_remaining == 3
