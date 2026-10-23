"""Run the decomposed-SAC experiment set in parallel lanes, skipping finished runs.

    python tools/run_tshi_decomp_queue.py --lanes 2 --workers 10
    python tools/run_tshi_decomp_queue.py --config configs/tshi_decomp_paper_rules.yaml --study paper_rules

Case-3 study (paper's best role case, P1 sink / P2, P3 sources): SAC, DE and random
search, three seeds, 3,000 evaluations each.  All-case study: the 12 role cases of
Lo et al. Tables 3-5 with one contextual SAC policy and equal-budget baselines.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ([("case3", "sac", s, 3000, "-++") for s in (1, 2, 3)]
        + [("case3", m, s, 3000, "-++") for s in (1, 2, 3) for m in ("de", "random")]
        + [("all_cases", m, 1, 6000, None) for m in ("sac", "de", "random")])


def lane(jobs, workers, lock, config, prefix):
    for study, method, seed, budget, cases in jobs:
        out = ROOT / "results" / "tshi_decomp" / prefix / study / f"{method}_seed{seed}"
        if (out / "summary.json").exists():
            continue
        if out.exists() and any(out.iterdir()):
            with lock:
                print(f"skip {out}: incomplete earlier run present", flush=True)
            continue
        cmd = [sys.executable, str(ROOT / "run_tshi_decomp.py"), "--config", str(config),
               "--method", method, "--seed", str(seed),
               "--budget", str(budget), "--workers", str(workers), "--out", str(out), "--no-plot"]
        if cases:
            cmd.append(f"--cases={cases}")
        with lock:
            print("start", " ".join(cmd[2:]), flush=True)
        log = out.with_suffix(".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        with lock:
            print(f"done {study}/{method}_seed{seed} exit {code}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "tshi_decomp.yaml")
    ap.add_argument("--study", default="", help="results subfolder, e.g. paper_rules")
    ap.add_argument("--runs", help="comma list of study:method:seed:budget[:cases] (default: the full study)")
    args = ap.parse_args()
    runs = RUNS
    if args.runs:
        runs = []
        for item in args.runs.split(","):
            study, method, seed, budget, *cases = item.split(":")
            runs.append((study, method, int(seed), int(budget), cases[0] if cases else None))
    lock = threading.Lock()
    threads = [threading.Thread(target=lane, args=(runs[i::args.lanes], args.workers, lock, args.config, args.study))
               for i in range(args.lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for study in sorted({r[0] for r in runs}):
        subprocess.run([sys.executable, str(ROOT / "plot_tshi_decomp.py"),
                        str(ROOT / "results" / "tshi_decomp" / args.study / study)],
                       cwd=ROOT)


if __name__ == "__main__":
    main()
