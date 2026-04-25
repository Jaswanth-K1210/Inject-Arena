"""Generate training plots from TRL trainer_state.json or JSONL logs + eval results.

Plots produced
--------------
1. reward_curve.png   — mean reward ± std band + smoothed trend
2. kl_loss_curve.png  — KL divergence & policy loss on twin axes
3. completion_stats.png — mean completion length + clipped-ratio line
4. bypass_bars.png    — RL-trained vs handcrafted baseline (eval results)
5. per_category.png   — per-scenario-category breakdown (eval results)

Usage
-----
    python scripts/make_plots.py \\
        --trainer-state /path/to/trainer_state.json \\
        --eval docs/eval_results.json \\
        --out docs/plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=str, default="logs/",
                   help="Directory of JSONL log files (legacy fallback)")
    p.add_argument("--trainer-state", type=str, default=None,
                   help="Path to TRL trainer_state.json (preferred)")
    p.add_argument("--out", type=str, default="docs/plots/")
    p.add_argument("--eval", type=str, default="docs/eval_results.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _load_all_logs(logs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(logs_dir.glob("*.jsonl")):
        rows.extend(_load_jsonl(p))
    rows.sort(key=lambda r: r.get("step", r.get("global_step", 0)))
    return rows


def _load_trainer_state(state_path: Path) -> List[Dict[str, Any]]:
    """Parse TRL trainer_state.json log_history into a normalised row list."""
    if not state_path.exists():
        print(f"trainer_state.json not found at {state_path}")
        return []
    with open(state_path) as f:
        data = json.load(f)

    # TRL GRPO key → normalised key mapping
    KEY_MAP = {
        "reward":                     "reward/mean",
        "rewards/mean":               "reward/mean",
        "reward/mean":                "reward/mean",
        "reward_std":                 "reward/std",
        "rewards/std":                "reward/std",
        "reward/std":                 "reward/std",
        "kl":                         "kl",
        "loss":                       "loss",
        "train/loss":                 "loss",
        "learning_rate":              "lr",
        "completions/mean_length":    "completion/mean_length",
        "completions/clipped_ratio":  "completion/clipped_ratio",
    }

    rows: List[Dict[str, Any]] = []
    for entry in data.get("log_history", []):
        step = entry.get("step")
        if step is None:
            continue
        row: Dict[str, Any] = {"step": step}
        for src, dst in KEY_MAP.items():
            if src in entry:
                row[dst] = entry[src]
        if len(row) > 1:   # has at least one metric besides step
            rows.append(row)
    rows.sort(key=lambda r: r["step"])
    print(f"Loaded {len(rows)} log entries from {state_path}")
    return rows


def _extract(rows: List[Dict[str, Any]], key: str) -> Tuple[List[int], List[float]]:
    steps, vals = [], []
    for r in rows:
        v = r.get(key)
        if v is not None:
            steps.append(r["step"])
            vals.append(float(v))
    return steps, vals


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _smooth(vals: List[float], window: int) -> List[float]:
    import numpy as np
    if window <= 1 or len(vals) < window:
        return vals
    return list(np.convolve(vals, np.ones(window) / window, mode="valid"))


BLUE   = "#3b82f6"
DBLUE  = "#1d4ed8"
RED    = "#ef4444"
GREEN  = "#22c55e"
ORANGE = "#f97316"
PURPLE = "#a855f7"
GRAY   = "#94a3b8"


# ---------------------------------------------------------------------------
# Plot 1: Reward curve with ±std band
# ---------------------------------------------------------------------------

def _plot_reward_curve(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    steps_r, rewards = _extract(rows, "reward/mean")
    steps_s, stds    = _extract(rows, "reward/std")

    if not steps_r:
        print("No reward data — skipping reward_curve.png")
        return

    window = max(1, len(rewards) // 15)
    smoothed = _smooth(rewards, window)
    smooth_steps = steps_r[window - 1:] if window > 1 else steps_r

    fig, ax = plt.subplots(figsize=(10, 5))

    # std band
    if steps_s and len(steps_s) == len(steps_r):
        r_arr = np.array(rewards)
        s_arr = np.array(stds)
        ax.fill_between(steps_r, r_arr - s_arr, r_arr + s_arr,
                        alpha=0.15, color=BLUE, label="±1 std")

    ax.plot(steps_r, rewards, alpha=0.35, color=BLUE, linewidth=0.9, label="raw reward")
    ax.plot(smooth_steps, smoothed, color=DBLUE, linewidth=2.2,
            label=f"smoothed (w={window})")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("InjectArena — GRPO Reward Curve (300 steps)")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)

    out_path = out_dir / "reward_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: KL divergence + policy loss on twin axes
# ---------------------------------------------------------------------------

def _plot_kl_loss(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    steps_kl, kls   = _extract(rows, "kl")
    steps_l,  losses = _extract(rows, "loss")

    if not steps_kl and not steps_l:
        print("No KL/loss data — skipping kl_loss_curve.png")
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))

    if steps_kl:
        ax1.plot(steps_kl, kls, color=PURPLE, linewidth=1.8, label="KL divergence")
        ax1.set_ylabel("KL Divergence", color=PURPLE)
        ax1.tick_params(axis="y", labelcolor=PURPLE)

    if steps_l:
        ax2 = ax1.twinx()
        ax2.plot(steps_l, losses, color=RED, linewidth=1.8, linestyle="--", label="Policy loss")
        ax2.set_ylabel("Policy Loss", color=RED)
        ax2.tick_params(axis="y", labelcolor=RED)

    ax1.set_xlabel("Training Step")
    ax1.set_title("InjectArena — KL Divergence & Policy Loss")
    ax1.grid(alpha=0.2)

    # Combined legend
    lines = []
    if steps_kl:
        lines += ax1.get_lines()
    if steps_l:
        lines += ax2.get_lines()
    if lines:
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    out_path = out_dir / "kl_loss_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Completion length + clipped ratio
# ---------------------------------------------------------------------------

def _plot_completion_stats(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    steps_l, lengths = _extract(rows, "completion/mean_length")
    steps_c, clipped = _extract(rows, "completion/clipped_ratio")

    if not steps_l and not steps_c:
        print("No completion stats — skipping completion_stats.png")
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))

    if steps_l:
        ax1.plot(steps_l, lengths, color=ORANGE, linewidth=1.8, label="Mean completion length (tokens)")
        ax1.set_ylabel("Mean Length (tokens)", color=ORANGE)
        ax1.tick_params(axis="y", labelcolor=ORANGE)

    if steps_c:
        ax2 = ax1.twinx()
        ax2.plot(steps_c, clipped, color=RED, linewidth=1.8, linestyle="--",
                 label="Clipped ratio (hit max_len)")
        ax2.set_ylabel("Clipped Ratio", color=RED)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor=RED)
        ax2.axhline(1.0, color=RED, linestyle=":", linewidth=0.7, alpha=0.5)

    ax1.set_xlabel("Training Step")
    ax1.set_title("InjectArena — Completion Length & Clipping")
    ax1.grid(alpha=0.2)

    lines = []
    if steps_l:
        lines += ax1.get_lines()
    if steps_c:
        lines += ax2.get_lines()
    if lines:
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    out_path = out_dir / "completion_stats.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Bypass bars (eval results vs baseline)
# ---------------------------------------------------------------------------

def _plot_bypass_bars(eval_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not eval_path.exists():
        print(f"Eval results not found at {eval_path} — skipping bypass_bars.png")
        return

    with open(eval_path) as f:
        data = json.load(f)

    metrics = {
        "PG2 Bypass":      data.get("pg2_bypass_rate", 0),
        "FW Bypass":       data.get("fw_bypass_rate", 0),
        "Task Success":    data.get("task_success_rate", 0),
        "Composed Bypass": data.get("composed_bypass_rate", 0),
    }
    baselines = {
        "PG2 Bypass": 0.15, "FW Bypass": 0.20,
        "Task Success": 0.05, "Composed Bypass": 0.02,
    }

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width / 2, [baselines[k] for k in metrics],
                   width, label="Handcrafted Baseline", color=GRAY, edgecolor="white")
    bars2 = ax.bar(x + width / 2, [metrics[k] for k in metrics],
                   width, label="InjectArena (RL-trained)", color=BLUE, edgecolor="white")

    ax.set_ylabel("Rate")
    ax.set_title("InjectArena — Attacker Performance vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, color="#475569")
    for bar in bars2:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, color=DBLUE)

    out_path = out_dir / "bypass_bars.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Per-category breakdown
# ---------------------------------------------------------------------------

def _plot_per_category(eval_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not eval_path.exists():
        return

    with open(eval_path) as f:
        data = json.load(f)

    per_cat = data.get("per_category", {})
    if not per_cat:
        print("No per_category data in eval results — skipping per_category.png")
        return

    cats = list(per_cat.keys())
    task_rates   = [per_cat[c]["task_success"] for c in cats]
    bypass_rates = [per_cat[c]["composed_bypass"] for c in cats]

    x = np.arange(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, task_rates,   width, label="Task Success",    color=GREEN, edgecolor="white")
    ax.bar(x + width / 2, bypass_rates, width, label="Composed Bypass", color=BLUE,  edgecolor="white")

    ax.set_ylabel("Rate")
    ax.set_title("InjectArena — Per-Category Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = out_dir / "per_category.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    out_dir   = Path(args.out)
    eval_path = Path(args.eval)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed — pip install matplotlib")
        return

    # Load training log rows (trainer_state preferred, JSONL fallback)
    rows: List[Dict[str, Any]] = []
    if args.trainer_state:
        rows = _load_trainer_state(Path(args.trainer_state))
    if not rows:
        logs_dir = Path(args.logs)
        if logs_dir.exists():
            rows = _load_all_logs(logs_dir)
            if rows:
                print(f"Loaded {len(rows)} log rows from {logs_dir}")

    # Training plots (require rows)
    if rows:
        _plot_reward_curve(rows, out_dir)
        _plot_kl_loss(rows, out_dir)
        _plot_completion_stats(rows, out_dir)
    else:
        print("No training log data found — skipping reward/KL/completion plots.")

    # Eval plots (require eval results JSON)
    _plot_bypass_bars(eval_path, out_dir)
    _plot_per_category(eval_path, out_dir)

    print("\nAll plots done.")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
