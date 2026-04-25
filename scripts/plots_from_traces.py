"""Generate dashboard plots locally from the committed trace JSONs.

Use this when ``trainer_state.json`` is not available locally (it lives only on
the Colab Drive). The 24 trace files in ``data/traces/`` are enough to produce
``bypass_bars.png`` and ``per_category.png`` honestly.

Usage:
    python scripts/plots_from_traces.py --traces data/traces --out docs/plots
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


# Same baseline numbers used by scripts/make_plots.py (handcrafted-corpus side)
BASELINES = {
    "PG2 Bypass":      0.15,
    "FW Bypass":       0.20,
    "Task Success":    0.05,
    "Composed Bypass": 0.02,
}


def _aggregate(traces_dir: Path) -> Dict:
    files = sorted(traces_dir.glob("*.json"))
    n = len(files)
    if n == 0:
        raise SystemExit(f"No traces in {traces_dir}")

    pg2 = fw = task = composed = 0
    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "task": 0, "composed": 0, "pg2": 0, "fw": 0})

    for f in files:
        with f.open() as fh:
            t = json.load(fh)
        o = t.get("outcome", {})
        atype = t.get("attack_type", "?")

        by_type[atype]["n"] += 1
        if o.get("broke_pg2"):     pg2 += 1; by_type[atype]["pg2"] += 1
        if o.get("broke_fw"):      fw += 1;  by_type[atype]["fw"] += 1
        if o.get("task_succeeded"): task += 1; by_type[atype]["task"] += 1
        if o.get("composed_bypass"): composed += 1; by_type[atype]["composed"] += 1

    return {
        "n": n,
        "pg2_rate": pg2 / n,
        "fw_rate":  fw / n,
        "task_rate": task / n,
        "composed_rate": composed / n,
        "by_type": dict(by_type),
    }


def _plot_bypass_bars(stats: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = {
        "PG2 Bypass":      stats["pg2_rate"],
        "FW Bypass":       stats["fw_rate"],
        "Task Success":    stats["task_rate"],
        "Composed Bypass": stats["composed_rate"],
    }
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, [BASELINES[k] for k in metrics], w,
                label="Handcrafted Baseline", color="#94a3b8", edgecolor="white")
    b2 = ax.bar(x + w / 2, [metrics[k] for k in metrics], w,
                label="InjectArena (RL-trained)", color="#3b82f6", edgecolor="white")

    ax.set_ylabel("Rate")
    ax.set_title(f"InjectArena — Attacker Performance vs Baseline (n={stats['n']} traces)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9)

    out_path = out / "bypass_bars.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_per_category(stats: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    by_type = stats["by_type"]
    types = sorted(by_type.keys())
    pg2_rates = [by_type[t]["pg2"] / by_type[t]["n"] for t in types]
    fw_rates  = [by_type[t]["fw"] / by_type[t]["n"] for t in types]
    task_rates = [by_type[t]["task"] / by_type[t]["n"] for t in types]

    x = np.arange(len(types))
    w = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, pg2_rates, w, label="PG2 Bypass",  color="#3b82f6", edgecolor="white")
    ax.bar(x,     fw_rates,  w, label="FW Bypass",   color="#1d4ed8", edgecolor="white")
    ax.bar(x + w, task_rates, w, label="Task Success", color="#22c55e", edgecolor="white")

    ax.set_ylabel("Rate")
    ax.set_title("InjectArena — Per Attack Category (across all step counts)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title() for t in types])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = out / "per_category.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_step_curve(traces_dir: Path, out: Path) -> None:
    """Bypass rate vs training-step label — the 'progression' visual."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    by_step: Dict[int, Dict[str, int]] = defaultdict(lambda: {"n": 0, "pg2": 0, "fw": 0, "task": 0})
    for f in sorted(traces_dir.glob("*.json")):
        with f.open() as fh:
            t = json.load(fh)
        s = t.get("steps")
        if s is None:
            continue
        o = t.get("outcome", {})
        by_step[s]["n"] += 1
        if o.get("broke_pg2"):      by_step[s]["pg2"]  += 1
        if o.get("broke_fw"):       by_step[s]["fw"]   += 1
        if o.get("task_succeeded"): by_step[s]["task"] += 1

    if not by_step:
        return
    xs = sorted(by_step.keys())
    pg2 = [by_step[s]["pg2"]  / by_step[s]["n"] for s in xs]
    fw  = [by_step[s]["fw"]   / by_step[s]["n"] for s in xs]
    task = [by_step[s]["task"] / by_step[s]["n"] for s in xs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, pg2,  marker="o", linewidth=2, label="PG2 Bypass",   color="#3b82f6")
    ax.plot(xs, fw,   marker="s", linewidth=2, label="FW Bypass",    color="#1d4ed8")
    ax.plot(xs, task, marker="^", linewidth=2, label="Task Success", color="#22c55e")
    ax.set_xlabel("Attacker training steps")
    ax.set_ylabel("Bypass rate")
    ax.set_title("InjectArena — Bypass Rate by Training Step Count")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = out / "reward_curve.png"   # reuse this filename so the dashboard finds it
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="data/traces")
    p.add_argument("--out", default="docs/plots")
    args = p.parse_args()

    traces = Path(args.traces)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    stats = _aggregate(traces)
    print(f"Aggregated {stats['n']} traces  "
          f"PG2={stats['pg2_rate']:.0%}  FW={stats['fw_rate']:.0%}  "
          f"Task={stats['task_rate']:.0%}  Composed={stats['composed_rate']:.0%}")

    _plot_bypass_bars(stats, out)
    _plot_per_category(stats, out)
    _plot_step_curve(traces, out)


if __name__ == "__main__":
    main()
