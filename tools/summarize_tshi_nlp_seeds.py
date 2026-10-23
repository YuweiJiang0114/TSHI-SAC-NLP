"""Table of all seeds of a tshi_nlp configuration and the overall best compliant design.

    python tools/summarize_tshi_nlp_seeds.py results/tshi_nlp/s1/case3 upper_shaped10_30k_seed
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PAPER = 453648.0


def main():
    root, prefix = Path(sys.argv[1]), sys.argv[2]
    runs = sorted(root.glob(prefix + "*"), key=lambda p: int(p.name[len(prefix):]) if p.name[len(prefix):].isdigit() else 0)
    print("| Seed | Evals | Best strict TAC | vs paper | Trimmed | Final policy | P3 loop stage |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    best = None
    for run in runs:
        seed = run.name[len(prefix):]
        if not (run / "evals.csv").exists():
            continue
        n = sum(1 for _ in open(run / "evals.csv", encoding="utf-8")) - 1
        if not (run / "best_design.json").exists():
            print(f"| {seed} | {n} | (running) | | | | |")
            continue
        x = json.loads((run / "best_design.json").read_text(encoding="utf-8"))
        fp = json.loads((run / "final_policy.json").read_text(encoding="utf-8")) if (run / "final_policy.json").exists() else {}
        p3 = x["plants"].get("P3", {})
        loop_stages = sorted({u["stage"] for u in p3.get("units", []) if u["kind"] == "loop"})
        print(f"| {seed} | {n} | {x['TAC']:,.0f} | {(x['TAC'] - PAPER) / PAPER * 100:+.1f}% | {x['TAC_trimmed']:,.0f} | "
              f"{fp.get('TAC', float('nan')):,.0f}{'' if fp.get('strictly_feasible') else ' (infeasible)'} | {loop_stages} |")
        if best is None or x["TAC"] < best[1]["TAC"]:
            best = (seed, x)
    if best:
        seed, x = best
        print(f"\nBest: seed {seed}, TAC {x['TAC']:,.0f} ({(x['TAC'] - PAPER) / PAPER * 100:+.2f}% vs paper), trimmed {x['TAC_trimmed']:,.0f}, "
              f"T_r {x['T_return']:.1f} T_s {x['T_supply']:.1f} flow {x['loop_flow']:.2f}")
        for p, d in x["plants"].items():
            print(f"  {p}: TAC {d['TAC']:,.0f} | " + ", ".join(f"{u['id']}@{u['stage']} {u['duty']:.0f} kW" for u in d["units"]))


if __name__ == "__main__":
    main()
