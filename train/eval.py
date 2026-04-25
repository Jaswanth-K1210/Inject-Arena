"""Evaluation script — runs a checkpoint against the eval split.

Usage:
    python train/eval.py --checkpoint /path/to/checkpoint --output-json docs/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-json", type=str, default="docs/eval_results.json")
    p.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_model(checkpoint: str, model_base: str):
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        logger.info("Eval model loaded via Unsloth from %s", checkpoint)
        return model, tokenizer
    except Exception as exc:
        logger.warning("Unsloth load failed (%s) — using transformers.", exc)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_base, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, seed: int) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    args = _parse_args()

    # Load defenses.
    from env.defenses.secalign_agent import SecAlignAgent
    from env.defenses.prompt_guard import PromptGuard
    from env.defenses.llama_firewall import FirewallWrapper
    from env.utils.embedding_cache import EmbeddingCache
    from env.environment import InjectArenaEnv
    from env.models import InjectAction
    from env.scenarios import ScenarioBank
    from train.client import format_prompt, parse_payload
    from env.models import InjectObservation

    secalign = SecAlignAgent()
    pg2 = PromptGuard()
    firewall = FirewallWrapper(prompt_guard_fallback=pg2)
    embedder = EmbeddingCache()
    bank = ScenarioBank()
    env = InjectArenaEnv(pg2=pg2, secalign=secalign, firewall=firewall,
                         bank=bank, embedder=embedder)

    # Load attacker model.
    model, tokenizer = _load_model(args.checkpoint, args.model_base)

    eval_scenarios = bank.by_split("eval")
    logger.info("Evaluating %d scenarios …", len(eval_scenarios))

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
        completion = _generate(model, tokenizer, prompt,
                                args.max_new_tokens, args.temperature, args.seed)
        payload = parse_payload(completion)

        env.reset(scenario_id=s["scenario_id"])
        result = env.step(InjectAction(payload=payload))

        rec = {
            "scenario_id": s["scenario_id"],
            "target_category": s["target_category"],
            "injection_slot": s["injection_slot"],
            "payload": payload,
            "reward": result.reward,
            "task_success": result.info.get("task_success", False),
            "pg2_flagged": result.info.get("pg2_verdict", {}).get("flagged", False),
            "fw_flagged": result.info.get("fw_verdict", {}).get("flagged", False),
            "components": result.info.get("components", {}),
        }
        records.append(rec)
        logger.info("  %s  reward=%.3f  task=%s  pg2=%s",
                    s["scenario_id"], rec["reward"],
                    rec["task_success"], rec["pg2_flagged"])

    # Aggregate metrics.
    n = len(records)
    task_success_rate = sum(r["task_success"] for r in records) / n
    pg2_bypass_rate = sum(not r["pg2_flagged"] for r in records) / n
    fw_bypass_rate = sum(not r["fw_flagged"] for r in records) / n
    composed_bypass = sum(
        r["task_success"] and not r["pg2_flagged"] and not r["fw_flagged"]
        for r in records
    ) / n
    mean_reward = sum(r["reward"] for r in records) / n

    # Per-category breakdown.
    categories = {r["target_category"] for r in records}
    per_category = {}
    for cat in categories:
        cat_recs = [r for r in records if r["target_category"] == cat]
        per_category[cat] = {
            "n": len(cat_recs),
            "task_success": sum(r["task_success"] for r in cat_recs) / len(cat_recs),
            "composed_bypass": sum(
                r["task_success"] and not r["pg2_flagged"] and not r["fw_flagged"]
                for r in cat_recs
            ) / len(cat_recs),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "n_scenarios": n,
        "mean_reward": round(mean_reward, 4),
        "task_success_rate": round(task_success_rate, 4),
        "pg2_bypass_rate": round(pg2_bypass_rate, 4),
        "fw_bypass_rate": round(fw_bypass_rate, 4),
        "composed_bypass_rate": round(composed_bypass, 4),
        "per_category": per_category,
        "records": records,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Eval Results ===")
    print(f"Scenarios:        {n}")
    print(f"Mean reward:      {mean_reward:.4f}")
    print(f"Task success:     {task_success_rate:.1%}")
    print(f"PG2 bypass:       {pg2_bypass_rate:.1%}")
    print(f"FW bypass:        {fw_bypass_rate:.1%}")
    print(f"Composed bypass:  {composed_bypass:.1%}  (all three)")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
