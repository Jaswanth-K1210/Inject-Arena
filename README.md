# InjectArena

**OpenEnv-compliant RL environment for training an adaptive prompt-injection attacker against a frozen Meta agent-safety stack.**

*Build in progress — targeting the Meta / Scaler / Hugging Face OpenEnv Hackathon India finale (Bangalore, April 25–26, 2026).*

The frozen defense stack:

1. **Llama Prompt Guard 2 (86M)** — input classifier
2. **Meta-SecAlign-8B** — LoRA on Llama-3.1-8B-Instruct (structured-prompt defense)
3. **LlamaFirewall** — Meta's scanner pipeline (PromptGuard + AgentAlignment)

The attacker is a small Qwen model trained with GRPO via TRL + Unsloth. Rewards jointly credit per-defense bypass and end-to-end task success, with a hard safety floor.

## How to run on Colab

This repo uses a split workflow: code lives on the Mac, GPU runs on Colab. The bridge is [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb). Open it in Colab, set `HF_TOKEN` (and optionally `GH_TOKEN`) as Colab secrets, then run cells in order.

See [`CLAUDE.md`](CLAUDE.md) for the phased build plan. Current phase: **bootstrap (Phase 0)**.

## Local install (Mac, no GPU)

```bash
pip install -e ".[demo]"
pytest tests/ -x
```

## License

Apache-2.0.
