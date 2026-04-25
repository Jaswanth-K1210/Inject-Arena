"""GRPO training entrypoint for the InjectArena attacker.

Attacker: Qwen/Qwen2.5-1.5B-Instruct + LoRA (r=16), trained with GRPO.
Reward: InjectArenaEnv.step() reward signal from the full defense stack.

Usage (from /content/injectarena on Colab):
    python train/grpo_train.py --steps 50 --output-dir /tmp/inject_smoke
    python train/grpo_train.py --steps 500 --output-dir /content/drive/MyDrive/injectarena/run_v1 --log-to wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# vLLM 0.19+ moved GuidedDecodingParams; TRL's grpo_trainer imports it at load
# time and crashes. We don't use vLLM-guided decoding, so stub it out before
# importing anything from trl.
try:
    import vllm.sampling_params as _vsp
    if not hasattr(_vsp, "GuidedDecodingParams"):
        _vsp.GuidedDecodingParams = type("GuidedDecodingParams", (), {})
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grpo_train")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO attacker training")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--output-dir", type=str, default="/tmp/injectarena_run")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-to", choices=["jsonl", "wandb"], default="jsonl")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_dataset(split: str = "train") -> Any:
    from datasets import Dataset
    from env.scenarios import ScenarioBank
    from train.client import format_prompt

    bank = ScenarioBank()
    rows = []
    for s in bank.by_split(split):
        # Build a fresh observation with no previous attempts for prompt formatting.
        from env.models import InjectObservation
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
        rows.append({
            "prompt": format_prompt(obs),
            "scenario_id": s["scenario_id"],
        })

    # GRPO needs many prompts — repeat to fill out training steps.
    if len(rows) < 100:
        rows = rows * (100 // len(rows) + 1)

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model loading — standard transformers + PEFT (no Unsloth)
#
# Unsloth 2025.11.x patches TRL's GRPOTrainer with an incompatible
# grpo_accumulated_loss signature, crashing at trainer.train(). Since the
# reward evaluation (3.5 s/step) dominates, Unsloth's generation speedup
# is not worth the breakage. Standard BitsAndBytes 4-bit is sufficient.
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_id: str, seed: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    logger.info("Loading %s via transformers + PEFT (4-bit LoRA) …", model_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Cast non-quantized params (lm_head, embeddings, norms) to bfloat16 so they
    # match the compute dtype during generation.
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None and module.weight.dtype == torch.float32:
            module.to(torch.bfloat16)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    logger.info("Attacker model ready (peft).")
    return model, tokenizer, "peft"


# ---------------------------------------------------------------------------
# Defense stack loader
# ---------------------------------------------------------------------------

def _load_defenses():
    from env.defenses.secalign_agent import SecAlignAgent
    from env.defenses.prompt_guard import PromptGuard
    from env.defenses.llama_firewall import FirewallWrapper
    from env.utils.embedding_cache import EmbeddingCache

    logger.info("Loading defense stack …")
    secalign = SecAlignAgent()
    pg2 = PromptGuard()
    firewall = FirewallWrapper(prompt_guard_fallback=pg2)
    embedder = EmbeddingCache()
    logger.info("Defense stack ready (SecAlign mode=%s).", secalign.mode)
    return pg2, secalign, firewall, embedder


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _make_reward_fn(pg2, secalign, firewall, embedder):
    from env.environment import InjectArenaEnv
    from env.models import InjectAction
    from env.scenarios import ScenarioBank

    bank = ScenarioBank()
    env = InjectArenaEnv(pg2=pg2, secalign=secalign, firewall=firewall,
                         bank=bank, embedder=embedder)

    def inject_reward(completions: list[str], scenario_id: list[str], **kwargs) -> list[float]:
        rewards = []
        for payload_raw, sid in zip(completions, scenario_id):
            from train.client import parse_payload
            payload = parse_payload(payload_raw)
            try:
                env.reset(scenario_id=sid)
                result = env.step(InjectAction(payload=payload))
                rewards.append(result.reward)
            except Exception as exc:
                logger.warning("Reward eval error (scenario=%s): %s", sid, exc)
                rewards.append(0.0)
        return rewards

    inject_reward.__name__ = "inject_reward"
    return inject_reward


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_to: str, output_dir: str) -> Any:
    if log_to == "wandb":
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            import wandb
            wandb.init(project="injectarena", dir=output_dir)
            return "wandb"
        else:
            logger.warning("WANDB_API_KEY not set — falling back to JSONL logging.")

    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return str(logs_dir / "train.jsonl")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== InjectArena GRPO Training ===")
    logger.info("steps=%d  batch=%d  gen=%d  output=%s",
                args.steps, args.batch_size, args.num_generations, args.output_dir)

    # Load defenses first (before model — keeps CUDA init order safe).
    pg2, secalign, firewall, embedder = _load_defenses()

    # Load attacker model.
    model, tokenizer, model_mode = _load_model_and_tokenizer(args.model, args.seed)

    # Dataset.
    train_dataset = _build_dataset("train")
    logger.info("Train dataset: %d prompts", len(train_dataset))

    # Reward function.
    reward_fn = _make_reward_fn(pg2, secalign, firewall, embedder)

    # Log setup.
    log_target = _setup_logging(args.log_to, args.output_dir)

    # GRPO config.
    from trl import GRPOConfig, GRPOTrainer

    report_to = "wandb" if log_target == "wandb" else "none"
    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=512,
        beta=0.04,                   # KL coefficient (CLAUDE.md §5.2)
        learning_rate=5e-6,
        save_steps=args.save_every,
        logging_steps=10,
        eval_strategy="no",          # eval handled externally by eval.py
        seed=args.seed,
        report_to=report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training (model_mode=%s) …", model_mode)
    trainer.train()

    # Save final checkpoint.
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info("Checkpoint saved to %s/final", output_dir)

    # Write run metadata.
    meta = {
        "steps": args.steps,
        "model": args.model,
        "model_mode": model_mode,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Done.")


if __name__ == "__main__":
    main()
