"""Summarize tshi_sim runs: best designs vs the paper, learning curves, figures.

    python tools/summarize_tshi_sim.py results/tshi_sim/s1/case3/seed1 [more run dirs] [--paper 453648]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BASELINE = 630350.0


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(run: Path, paper_tac: float):
    s = json.loads((run / "summary.json").read_text(encoding="utf-8")) if (run / "summary.json").exists() else {}
    evals = read_csv(run / "evals.csv") if (run / "evals.csv").exists() else []
    curve = read_csv(run / "plant_curve.csv") if (run / "plant_curve.csv").exists() else []
    out = {"run": str(run), "summary": s}
    if evals:
        tac = np.array([float(r["TAC"]) for r in evals])
        strict = np.array([r["strictly_feasible"] == "1" for r in evals])
        n = np.arange(1, len(evals) + 1)
        best = np.minimum.accumulate(np.where(strict, tac, np.inf))
        target = BASELINE - 0.99 * (BASELINE - paper_tac)
        hit = n[best <= target]
        out["evals"] = len(evals)
        out["best_TAC"] = float(best[-1]) if np.isfinite(best[-1]) else None
        out["evals_to_99pct"] = int(hit[0]) if len(hit) else None
        out["strict_fraction"] = float(strict.mean())
        policy = [float(r["TAC"]) for r in evals if r["phase"] == "policy" and r["strictly_feasible"] == "1"]
        out["policy_probe_best"] = min(policy) if policy else None
        out["final_policy_TAC"] = s.get("final_policy", {}).get("TAC")
        out["final_policy_strict"] = s.get("final_policy", {}).get("strictly_feasible")
    if curve:
        last = {}
        for r in curve:
            last[r["plant"]] = r
        out["phase1_last"] = {p: {k: r[k] for k in ("iteration", "reward_mean", "strict_fraction", "tac_clean_min_strict")}
                              for p, r in last.items()}
    return out, evals, curve


def plot(run: Path, evals, curve, paper_tac):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"P1": "#1f5fbf", "P2": "#c25a00", "P3": "#2a8a4a"}
    ax = axes[0]
    for p in sorted({r["plant"] for r in curve}):
        rows = [r for r in curve if r["plant"] == p]
        x = [int(r["iteration"]) for r in rows]
        y = [float(r["reward_mean"]) if r["reward_mean"] else np.nan for r in rows]
        ax.plot(x, y, lw=1.6, color=colors.get(p, "#555"), label=p)
        ax.annotate(p, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=9, color=colors.get(p, "#555"))
    ax.set_xlabel("phase-1 iteration (batch of random interfaces)")
    ax.set_ylabel("mean plant reward (own TAC saving / site baseline)")
    ax.set_title("Phase 1: plant agents")
    ax.grid(alpha=0.25)
    ax = axes[1]
    if evals:
        tac = np.array([float(r["TAC"]) for r in evals])
        strict = np.array([r["strictly_feasible"] == "1" for r in evals])
        n = np.arange(1, len(evals) + 1)
        best = np.minimum.accumulate(np.where(strict, tac, np.inf))
        ax.scatter(n[strict], tac[strict], s=5, color="#9aa7b8", label="strictly feasible evaluation")
        ax.plot(n, best, lw=1.8, color="#1f5fbf", label="best so far")
        pol = [(int(r["eval"]), float(r["TAC"])) for r in evals if r["phase"] == "policy" and r["strictly_feasible"] == "1"]
        if pol:
            ax.plot(*zip(*pol), lw=1.2, color="#c25a00", label="deterministic policy")
        ax.axhline(paper_tac, color="#2a8a4a", lw=1.2, ls="--")
        ax.annotate(f"paper {paper_tac:,.0f}", (n[-1], paper_tac), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=9, color="#2a8a4a")
        lo = min(paper_tac, np.nanmin(best[np.isfinite(best)]) if np.isfinite(best).any() else paper_tac)
        ax.set_ylim(lo - 10000, min(BASELINE, lo + 120000))
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_xlabel("site evaluation")
    ax.set_ylabel("site TAC (USD/yr)")
    ax.set_title("Phase 2: site agent")
    ax.grid(alpha=0.25)
    fig.savefig(run / "curves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--paper", type=float, default=453648.0)
    args = ap.parse_args()
    print("| Run | Evals | Best strict TAC | vs paper | Evals to 99% | Final policy TAC | Strict frac |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for run in args.runs:
        out, evals, curve = summarize(run, args.paper)
        if evals or curve:
            plot(run, evals, curve, args.paper)
        b = out.get("best_TAC")
        fp = out.get("final_policy_TAC")
        print(f"| {run} | {out.get('evals', 0)} | {b:,.0f} | {(b - args.paper) / args.paper * 100:+.2f}% | "
              f"{out.get('evals_to_99pct')} | {fp:,.0f}{'' if out.get('final_policy_strict') else ' (infeasible)'} | "
              f"{out.get('strict_fraction', 0):.2f} |" if b else f"| {run} | {out.get('evals', 0)} | - | - | - | - | - |")
        if "phase1_last" in out:
            print("  phase 1 (last logged):", json.dumps(out["phase1_last"]))


if __name__ == "__main__":
    main()
