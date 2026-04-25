# InjectArena — Demo Video Script

**Target length:** under 2 minutes (≈ 200-240 words spoken)

---

## Setup before recording

- Browser open at: `https://huggingface.co/spaces/Jaswanth-K/Inject-Arena`
- VS Code open on `README.md` (for architecture diagram shot)
- Terminal ready with `git log --oneline -8` output visible

---

## Script

**[0:00 — 0:20] Hook**

> "Prompt injection is the top threat to LLM agents. Meta has deployed three production defenses to stop it: Llama Prompt Guard 2, SecAlign-8B, and LlamaFirewall. In this demo I'll show what happens when you train an RL attacker directly against all three."

**[0:20 — 0:40] Problem in 10 seconds**

> "Static attack corpora — fixed injections from published papers — only test what defenders have already seen. InjectArena is an OpenEnv-compliant environment where an attacker model gets reward signal directly from the live defense stack, so it learns to find the gaps."

**[0:40 — 1:15] Live demo** ← screen record the Space

> "Here's the Gradio demo. I'll pick the Email Exfiltration scenario — the target agent is an email assistant, and the attack goal is to make it forward a session key to an attacker-controlled address."
>
> *(Click "Email Exfiltration", click "Run Attack")*
>
> "Left column: the best attack from a published handcrafted corpus. Prompt Guard 2 flags it — attack fails."
>
> "Right column: the RL-trained attacker. Prompt Guard passes. LlamaFirewall passes. And the task succeeded — the agent called `send_email` with the target address. Same defense stack, completely different outcome."

**[1:15 — 1:40] Architecture + results**

> "The environment exposes OpenEnv-compatible `/reset` and `/step` endpoints. Reward is a weighted sum across all three defenses plus task success and stealth regularisers. The attacker is Qwen2.5-1.5B with GRPO — just 1.5 billion parameters finding strategies that a 86M classifier and an 8B SecAlign model both miss."

**[1:40 — 2:00] Close**

> "The environment, scenario bank, reward design, and training pipeline are fully open-source. You can reproduce the training run on a free Colab A100 with one notebook. Link in the description."

---

## Recording checklist

- [ ] Record at 1080p
- [ ] Microphone check (no background noise)
- [ ] Browser zoom at 110% for legibility
- [ ] Demo runs in stub mode if real defenses are slow — that's fine, the chips still show the right colors
- [ ] Upload to YouTube as unlisted, then set to public after submission
- [ ] Paste YouTube URL into README Links section and hackathon form
