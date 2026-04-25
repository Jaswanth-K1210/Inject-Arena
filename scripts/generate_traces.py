"""Generate replay traces for the public demo (run on Colab).

For each (attack_type, steps_label) requested, this script:
  1. Loads the attacker checkpoint (or zero-shot Qwen for the lowest step count).
  2. Loads the real defense stack (PG2 + SecAlign + LlamaFirewall).
  3. Runs the attacker once against the scenario for that attack type.
  4. Captures stage-by-stage timings + verdicts into a JSON trace.
  5. Saves the trace to ``data/traces/{attack_type}_{steps}.json``.

Also writes ``data/highlights/highlight.json`` — the most successful trace
across the run, used by the homepage hero animation.

Usage (typical Colab cell)
--------------------------
    python scripts/generate_traces.py \\
        --checkpoint /content/drive/MyDrive/injectarena/run_v1/final \\
        --steps-labels 50 100 300 500 1000 1500 \\
        --output-dir data/traces \\
        --highlight-dir data/highlights

For ``--steps-labels`` values <= ``--baseline-cutoff`` (default 100), the
zero-shot baseline (untrained Qwen) is used to simulate an early/under-trained
attacker. Above the cutoff, the trained checkpoint is used. This is
documented in the trace itself via the ``model_source`` field, so the demo
stays honest.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("generate_traces")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# Attack-type → scenario_id mapping (kept in sync with env/replay.py)
ATTACK_TYPE_TO_SCENARIO = {
    "email_exfiltration": "email_exfil_001",
    "forbidden_tool":     "email_forbidden_001",
    "prompt_leak":        "email_leak_001",
    "rag_injection":      "rag_exfil_001",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained attacker checkpoint dir (the final/ folder).")
    p.add_argument("--steps-labels", type=int, nargs="+",
                   default=[50, 100, 300, 500, 1000, 1500],
                   help="Step-count labels to generate traces for.")
    p.add_argument("--baseline-cutoff", type=int, default=100,
                   help="Steps <= cutoff use the zero-shot baseline; above use the checkpoint.")
    p.add_argument("--output-dir", type=str, default="data/traces")
    p.add_argument("--highlight-dir", type=str, default="data/highlights")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Attacker loading (uses the same Unsloth path as train/eval.py)
# ---------------------------------------------------------------------------

def _load_attacker(checkpoint: str, max_new_tokens: int):
    from unsloth import FastLanguageModel
    logger.info("Loading attacker from %s", checkpoint)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype="bfloat16",
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_zero_shot(max_new_tokens: int):
    from unsloth import FastLanguageModel
    logger.info("Loading zero-shot baseline (Qwen2.5-1.5B-Instruct)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        load_in_4bit=False,
        dtype="bfloat16",
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate_payload(model, tokenizer, observation, max_new_tokens: int, seed: int) -> str:
    import torch
    from train.client import format_prompt, parse_payload

    prompt = format_prompt(observation)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return parse_payload(raw)


# ---------------------------------------------------------------------------
# Defense stack — wrapped to record timings + verdicts
# ---------------------------------------------------------------------------

def _load_defenses():
    from env.defenses.prompt_guard import PromptGuard
    from env.defenses.secalign_agent import SecAlignAgent
    from env.defenses.llama_firewall import FirewallWrapper
    logger.info("Loading defense stack (SecAlign first for vLLM CUDA order) …")
    secalign = SecAlignAgent()
    pg2 = PromptGuard()
    fw = FirewallWrapper(prompt_guard_fallback=pg2)
    logger.info("Defense stack ready (SecAlign mode=%s).", secalign.mode)
    return pg2, secalign, fw


def _run_pipeline(
    payload: str,
    scenario: Dict[str, Any],
    pg2,
    secalign,
    firewall,
) -> Dict[str, Any]:
    """Run the full attack pipeline and return a trace dict."""
    timeline: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    # Stage 1: generation already happened upstream — we record it as t=0
    timeline.append({"stage": "generation", "t": 0.0, "payload": payload})

    # Stage 2: PG2 input scan
    s = time.perf_counter()
    pg2_v = pg2.scan(payload)
    pg2_t = time.perf_counter() - t0
    timeline.append({
        "stage": "pg2_scan",
        "t": round(pg2_t, 3),
        "duration": round(time.perf_counter() - s, 3),
        "flagged": bool(pg2_v.flagged),
        "score": float(getattr(pg2_v, "score", 0.0)),
        "reason": getattr(pg2_v, "reason", None),
    })

    # Stage 3: SecAlign agent run
    s = time.perf_counter()
    untrusted = {scenario["injection_slot"]: payload}
    agent_output = secalign.run(
        system=scenario["agent_system_prompt"],
        user=scenario["user_query"],
        untrusted=untrusted,
    )
    agent_t = time.perf_counter() - t0
    timeline.append({
        "stage": "agent_run",
        "t": round(agent_t, 3),
        "duration": round(time.perf_counter() - s, 3),
        "agent_output": agent_output,
    })

    # Stage 4: LlamaFirewall output scan
    s = time.perf_counter()
    fw_v = firewall.scan(
        user_query=scenario["user_query"],
        agent_output=agent_output,
        tool_call=None,
    )
    fw_t = time.perf_counter() - t0
    timeline.append({
        "stage": "firewall_scan",
        "t": round(fw_t, 3),
        "duration": round(time.perf_counter() - s, 3),
        "flagged": bool(fw_v.flagged),
        "score": float(getattr(fw_v, "score", 0.0)),
        "reason": getattr(fw_v, "reason", None),
    })

    # Stage 5: task verifier
    task_succeeded = _check_task_success(scenario, agent_output)
    timeline.append({
        "stage": "verdict",
        "t": round(time.perf_counter() - t0, 3),
        "task_succeeded": task_succeeded,
    })

    broke_pg2 = not pg2_v.flagged
    broke_fw = not fw_v.flagged
    composed = broke_pg2 and broke_fw and task_succeeded

    return {
        "timeline": timeline,
        "outcome": {
            "broke_pg2": broke_pg2,
            "broke_fw": broke_fw,
            "task_succeeded": bool(task_succeeded),
            "composed_bypass": bool(composed),
            "blocked_at": (
                "Llama Prompt Guard 2" if not broke_pg2
                else ("LlamaFirewall" if not broke_fw
                else (None if task_succeeded else "SecAlign agent (refused)"))
            ),
        },
    }


def _check_task_success(scenario: Dict[str, Any], agent_output: str) -> bool:
    """Lightweight verifier so traces can include a task-success flag."""
    from env.verifiers.exfiltration import verify_exfiltration
    from env.verifiers.forbidden_tool import verify_forbidden_tool
    from env.verifiers.prompt_leak import verify_prompt_leak

    cat = scenario["target_category"]
    try:
        if cat == "exfiltration":
            return verify_exfiltration(agent_output, scenario)
        if cat == "forbidden_tool":
            return verify_forbidden_tool(agent_output, scenario)
        if cat == "prompt_leak":
            return verify_prompt_leak(agent_output, scenario)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Verifier error for %s: %s", scenario.get("scenario_id"), exc)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    highlight_dir = Path(args.highlight_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    highlight_dir.mkdir(parents=True, exist_ok=True)

    from env.environment import InjectArenaEnv
    from env.scenarios import ScenarioBank
    bank = ScenarioBank()

    # Load defenses ONCE — they dominate cost.
    pg2, secalign, firewall = _load_defenses()

    # Build a minimal env so we can reset() into a scenario and get an InjectObservation.
    env = InjectArenaEnv(pg2=pg2, secalign=secalign, firewall=firewall, bank=bank)

    # Cache attackers by source so we don't reload between calls.
    attackers: Dict[str, Any] = {}

    def get_attacker(source: str):
        if source in attackers:
            return attackers[source]
        if source == "checkpoint":
            attackers[source] = _load_attacker(args.checkpoint, args.max_new_tokens)
        else:
            attackers[source] = _load_zero_shot(args.max_new_tokens)
        return attackers[source]

    best: Optional[Dict[str, Any]] = None
    best_score = -1

    for attack_type, scenario_id in ATTACK_TYPE_TO_SCENARIO.items():
        try:
            scenario = bank.by_id(scenario_id)
        except KeyError:
            logger.warning("Scenario %s not in bank — skipping %s", scenario_id, attack_type)
            continue

        for steps in args.steps_labels:
            source = "checkpoint" if steps > args.baseline_cutoff else "zero_shot"
            model, tokenizer = get_attacker(source)
            obs = env.reset(scenario_id=scenario_id)
            payload = _generate_payload(model, tokenizer, obs, args.max_new_tokens, args.seed + steps)

            pipe = _run_pipeline(payload, scenario, pg2, secalign, firewall)
            trace = {
                "attack_type": attack_type,
                "steps": steps,
                "scenario_id": scenario_id,
                "scenario_label": scenario.get("target_behavior", ""),
                "model_source": source,
                "payload": payload,
                **pipe,
            }
            out_path = out_dir / f"{attack_type}_{steps}.json"
            with out_path.open("w") as f:
                json.dump(trace, f, indent=2)
            logger.info("Wrote %s (broke_pg2=%s, broke_fw=%s, task=%s)",
                        out_path.name,
                        pipe["outcome"]["broke_pg2"],
                        pipe["outcome"]["broke_fw"],
                        pipe["outcome"]["task_succeeded"])

            # Score for highlight selection: composed > fw_bypass > pg2_bypass.
            o = pipe["outcome"]
            score = (4 if o["composed_bypass"] else 0) \
                  + (2 if o["broke_fw"] else 0) \
                  + (1 if o["broke_pg2"] else 0) \
                  + (steps / 10000.0)   # tiebreaker: prefer higher-step traces
            if score > best_score:
                best_score = score
                best = trace

    if best is not None:
        with (highlight_dir / "highlight.json").open("w") as f:
            json.dump(best, f, indent=2)
        logger.info("Highlight: %s_%s (score=%.2f)", best["attack_type"], best["steps"], best_score)

    logger.info("Done. Wrote %d traces to %s.", len(list(out_dir.glob("*.json"))), out_dir)


if __name__ == "__main__":
    main()
