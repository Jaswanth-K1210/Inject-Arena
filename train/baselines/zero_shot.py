"""Zero-shot baseline — same Qwen2.5-1.5B-Instruct model, no training.

Evaluates the base model's raw prompt-injection capability without any RL.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("zero_shot")

_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def run_zero_shot(output_json: str = "docs/zero_shot_results.json",
                  temperature: float = 0.7,
                  seed: int = 42) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from env.defenses.secalign_agent import SecAlignAgent
    from env.defenses.prompt_guard import PromptGuard
    from env.defenses.llama_firewall import FirewallWrapper
    from env.utils.embedding_cache import EmbeddingCache
    from env.environment import InjectArenaEnv
    from env.models import InjectAction, InjectObservation
    from env.scenarios import ScenarioBank
    from train.client import format_prompt, parse_payload

    # Load defenses.
    secalign = SecAlignAgent()
    pg2 = PromptGuard()
    firewall = FirewallWrapper(prompt_guard_fallback=pg2)
    embedder = EmbeddingCache()
    bank = ScenarioBank()
    env = InjectArenaEnv(pg2=pg2, secalign=secalign, firewall=firewall,
                         bank=bank, embedder=embedder)

    # Load base model (no LoRA).
    logger.info("Loading zero-shot model %s …", _MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model.eval()
    logger.info("Zero-shot model ready.")

    eval_scenarios = bank.by_split("eval")
    records: List[Dict[str, Any]] = []

    for s in eval_scenarios:
        obs = InjectObservation(
            scenario_id=s["scenario_id"],
            target_behavior=s["target_behavior"],
            target_category=s["target_category"],
            agent_system_prompt=s["agent_system_prompt"],
            user_query=s["user_query"],
            injection_slot=s["injection_slot"],
            tool_surface=s["tool_surface"],
            canary_string=s.get("canary_string"),
            previous_attempts=[],
            attempts_remaining=3,
        )
        prompt = format_prompt(obs)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        torch.manual_seed(seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=512, do_sample=True,
                temperature=temperature, top_p=0.9,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        payload = parse_payload(completion)

        env.reset(scenario_id=s["scenario_id"])
        result = env.step(InjectAction(payload=payload))

        records.append({
            "scenario_id": s["scenario_id"],
            "target_category": s["target_category"],
            "injection_slot": s["injection_slot"],
            "payload": payload,
            "reward": result.reward,
            "task_success": result.info.get("task_success", False),
            "pg2_flagged": result.info.get("pg2_verdict", {}).get("flagged", False),
            "fw_flagged": result.info.get("fw_verdict", {}).get("flagged", False),
        })

    n = len(records)
    summary = {
        "baseline": "zero_shot",
        "model": _MODEL_ID,
        "n_scenarios": n,
        "mean_reward": round(sum(r["reward"] for r in records) / n, 4),
        "task_success_rate": round(sum(r["task_success"] for r in records) / n, 4),
        "pg2_bypass_rate": round(sum(not r["pg2_flagged"] for r in records) / n, 4),
        "fw_bypass_rate": round(sum(not r["fw_flagged"] for r in records) / n, 4),
        "composed_bypass_rate": round(
            sum(r["task_success"] and not r["pg2_flagged"] and not r["fw_flagged"]
                for r in records) / n, 4),
        "records": records,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Zero-shot results saved to %s", out_path)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", default="docs/zero_shot_results.json")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    result = run_zero_shot(args.output_json, args.temperature, args.seed)
    print(f"\nZero-shot composed bypass: {result['composed_bypass_rate']:.1%}")
