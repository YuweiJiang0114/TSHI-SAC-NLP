"""SAC coordination of independently optimized plant retrofits (Lo et al. 2021 case).

    python run_tshi_decomp.py --method sac    --seed 1
    python run_tshi_decomp.py --method random --seed 1
    python run_tshi_decomp.py --method de     --seed 1
    python run_tshi_decomp.py --validate        # paper interface + zero-loop checks

Decomposition.  The SAC agent decides only the intermediate-fluid interface of each
plant (return temperature, per-plant duty, source outlet temperatures).  Each plant
then optimizes its own Strategy-I retrofit at that interface (``tshi_decomp.plant``),
in parallel and independently of the other plants; site TAC is the sum of plant TACs
plus the intermediate-fluid cost.

Learning problem.  The retrofit is a static design problem, so every episode is one
absolute interface proposal (gamma = 0, done = 1).  The observation is the
source/sink role case (Lo et al. Tables 3-5 enumerate the same 12 cases), so one
contextual policy learns the optimal interface of every coalition and role
assignment; the twin critics learn each case's savings surface and the
maximum-entropy actor concentrates on its optimum.  Random search and differential
evolution (the outer-layer method of Tian et al. 2020a) use the identical evaluator
and the same number of evaluations per case.

Only original-plant data enter the search.  The published design is used solely by
``--validate`` as a post hoc check.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from tshi_decomp.plant import PlantSolverConfig
from tshi_decomp.site import RoleCase, SiteConfig, SiteEvaluator

ROOT = Path(__file__).resolve().parent
PAPER = {"strategy_I_TAC": 453648.0, "strategy_I_TACS": 176702.0,
         "strategy_III_TAC": 451648.0, "strategy_III_TACS": 178702.0,
         "strategy_I_shapley": {"P1": 85660.0, "P2": 24486.0, "P3": 66557.0}}
ROLE_CODE = {"sink": "-", "source": "+", "inactive": "0"}


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=1), encoding="utf-8")
    tmp.replace(path)


def role_code(roles: dict) -> str:
    """Executed roles: '-' sink, '+' source, '0' not taking part."""
    return "".join(ROLE_CODE[v] for v in roles.values())


def compact(r: dict) -> dict:
    """One evaluation without the full plant exports."""
    keys = ("case", "action", "reward", "TAC", "TACS", "TAC_without_penalty", "strictly_feasible", "slack_kW",
            "T_return", "T_supply", "loop_flow", "roles", "interfaces", "plant_TAC")
    return {**{k: r[k] for k in keys},
            "structures": {p: x["structure"]["label"] for p, x in r["plants"].items()},
            "plant_wall_s": {p: x.get("solver", {}).get("wall_s", 0.0) for p, x in r["plants"].items()}}


class Recorder:
    """Append-only evaluation log with per-case best-so-far tracking.

    The best design of a case is the strictly feasible evaluation (zero slack in
    every plant) with the largest saving.  Each case starts from its coalition's
    original networks (TACS = 0).
    """

    def __init__(self, out: Path, cases, baseline_of):
        self.out = out
        self.n = 0
        self.best = {c: {"TACS": 0.0, "TAC": baseline_of(c), "eval": 0, "result": None} for c in cases}
        self.started = time.perf_counter()
        self.trace = (out / "evaluations.jsonl").open("w", encoding="utf-8")
        self.csv = (out / "log.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv)
        self.writer.writerow(["eval", "phase", "case", "TAC", "TACS", "reward", "strictly_feasible", "slack_kW",
                              "roles", "best_case_TACS", "wall_s"])

    def add(self, results, phase, iteration):
        for r in results:
            self.n += 1
            c = r["case"]
            if r["strictly_feasible"] and r["TACS"] > self.best[c]["TACS"]:
                self.best[c] = {"TACS": r["TACS"], "TAC": r["TAC"], "eval": self.n, "phase": phase, "result": r}
                dump(self.out / "best_designs.json", self.best)
            self.writer.writerow([self.n, phase, c, f"{r['TAC']:.4f}", f"{r['TACS']:.4f}", f"{r['reward']:.6f}",
                                  int(r["strictly_feasible"]), f"{r['slack_kW']:.6f}", role_code(r["roles"]),
                                  f"{self.best[c]['TACS']:.4f}", f"{time.perf_counter() - self.started:.2f}"])
            self.trace.write(json.dumps(jsonable({"eval": self.n, "phase": phase, "iteration": iteration,
                                                  **compact(r), "best_case_TACS": self.best[c]["TACS"]})) + "\n")
        self.csv.flush()
        self.trace.flush()

    def close(self):
        self.trace.close()
        self.csv.close()


def build(cfg):
    site = SiteConfig(**{k: tuple(v) if k == "members" and v else v for k, v in cfg.get("site", {}).items()})
    solver = PlantSolverConfig(**cfg.get("plant_solver", {}))
    return SiteEvaluator(site, solver, workers=cfg.get("workers"))


def select_cases(cfg, model):
    spec = cfg.get("cases", "all")
    if spec == "all":
        return list(model.cases)
    return [RoleCase.from_code(c) for c in spec]


def baseline_check(ev):
    zero = np.zeros(ev.model.action_dim)
    r = ev.evaluate([zero])[0]  # role-free zero request: every plant keeps its original network
    if not r["strictly_feasible"] or abs(r["TAC"] - ev.model.site_baseline) > 1.0:  # USD/yr
        raise RuntimeError(f"zero-loop site does not reproduce the original utility cost: {r['TAC']}")
    return r


def round_robin(cases, start, n):
    return [cases[(start + i) % len(cases)] for i in range(n)]


def run_sac(cfg, ev, rec, out, cases):
    import torch
    from total_site.central_sac import CentralSAC, SACConfig
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))
    seed = int(cfg["seed"])
    sac = dict(cfg.get("sac", {}))
    sac["hidden_dims"] = tuple(sac.get("hidden_dims", (256, 256)))
    per_dim = sac.pop("target_entropy_per_dim", None)
    if per_dim is not None:
        sac["target_entropy"] = float(per_dim) * ev.model.action_dim
    d = ev.model.action_dim
    agent = CentralSAC(ev.model.observation_dim, -np.ones(d), np.ones(d), SACConfig(seed=seed, **sac))
    rng = np.random.default_rng(seed)
    budget, batch = int(cfg["budget"]), int(cfg.get("batch", len(cases)))
    warmup, utd = int(cfg.get("warmup", 256)), float(cfg.get("utd", 4))
    probe_every = int(cfg.get("policy_eval_every", 10))
    updates, it, cursor = [], 0, 0
    pcsv = (out / "policy_curve.csv").open("w", newline="", encoding="utf-8")
    pw = csv.writer(pcsv)
    pw.writerow(["eval", "iteration", "case", "TAC", "TACS", "strictly_feasible", "slack_kW", "roles", "alpha"]
                + ev.model.labels)
    last_probe = {}
    while rec.n < budget:
        it += 1
        n = min(batch, budget - rec.n)
        batch_cases = round_robin(cases, cursor, n)
        cursor += n
        phase = "warmup" if rec.n < warmup else "sac"
        if phase == "warmup":
            actions = rng.uniform(-1, 1, (n, d))
        else:
            actions = np.stack([agent.act(c.observation()) for c in batch_cases])
        results = ev.evaluate(actions, batch_cases)
        for c, a, r in zip(batch_cases, actions, results):
            agent.remember(c.observation(), a, r["reward"], c.observation(), True)
        rec.add(results, phase, it)
        # Deterministic-policy probes of every case; counted in the budget and replayed.
        if phase == "sac" and it % probe_every == 0 and rec.n + len(cases) <= budget:
            probes = np.stack([agent.act(c.observation(), deterministic=True) for c in cases])
            presults = ev.evaluate(probes, cases)
            for c, a, r in zip(cases, probes, presults):
                agent.remember(c.observation(), a, r["reward"], c.observation(), True)
                last_probe[c.code] = r["TACS"]
                pw.writerow([rec.n + 1, it, c.code, f"{r['TAC']:.4f}", f"{r['TACS']:.4f}",
                             int(r["strictly_feasible"]), f"{r['slack_kW']:.6f}", role_code(r["roles"]),
                             f"{float(agent.alpha):.6g}"] + [f"{x:.6f}" for x in r["action"]])
                rec.add([r], "policy", it)
            pcsv.flush()
        if rec.n >= warmup:
            for _ in range(int(round(n * utd))):
                u = agent.update()
                if u is not None and agent.gradient_updates % 25 == 0:
                    updates.append({"eval": rec.n, **u})
        if it % 5 == 0:
            g = cases[0].code
            print(f"[sac] eval {rec.n}/{budget} best[{g}] TACS {rec.best[g]['TACS']:.0f} "
                  f"policy[{g}] {last_probe.get(g, float('nan')):.0f} alpha {float(agent.alpha):.4g} "
                  f"updates {agent.gradient_updates} {time.perf_counter() - rec.started:.0f}s", flush=True)
    pcsv.close()
    agent.save(out / "policy.pt")
    final_actions = np.stack([agent.act(c.observation(), deterministic=True) for c in cases])
    final = ev.evaluate(final_actions, cases)
    dump(out / "final_policy.json", {r["case"]: r for r in final})
    with (out / "updates.csv").open("w", newline="", encoding="utf-8") as f:
        if updates:
            w = csv.DictWriter(f, fieldnames=list(updates[0]))
            w.writeheader()
            w.writerows(updates)
    return {"final_policy": {r["case"]: {"TACS": r["TACS"], "TAC": r["TAC"], "strict": r["strictly_feasible"],
                                         "roles": role_code(r["roles"])} for r in final},
            "gradient_updates": agent.gradient_updates, "final_alpha": float(agent.alpha)}


def run_random(cfg, ev, rec, out, cases):
    rng = np.random.default_rng(int(cfg["seed"]))
    budget, batch, it, cursor = int(cfg["budget"]), int(cfg.get("batch", len(cases))), 0, 0
    while rec.n < budget:
        it += 1
        n = min(batch, budget - rec.n)
        batch_cases = round_robin(cases, cursor, n)
        cursor += n
        rec.add(ev.evaluate(rng.uniform(-1, 1, (n, ev.model.action_dim)), batch_cases), "random", it)
        if it % 10 == 0:
            g = cases[0].code
            print(f"[random] eval {rec.n}/{budget} best[{g}] TACS {rec.best[g]['TACS']:.0f}", flush=True)
    return {}


def run_de(cfg, ev, rec, out, cases):
    """DE/rand/1/bin, one population per case, same evaluator and per-case budget."""
    from scipy.stats import qmc
    rng = np.random.default_rng(int(cfg["seed"]))
    d = ev.model.action_dim
    de = cfg.get("de", {})
    pop_n, f_w, cr = int(de.get("population", 24)), float(de.get("F", 0.6)), float(de.get("CR", 0.9))
    per_case = int(cfg["budget"]) // len(cases)
    pops = {c: qmc.LatinHypercube(d, seed=int(rng.integers(2**31))).random(pop_n) * 2 - 1 for c in cases}
    flat = [(c, x) for c in cases for x in pops[c]]
    res = ev.evaluate([x for _, x in flat], [c for c, _ in flat])
    rec.add(res, "de", 0)
    fit = {c: np.array([-r["reward"] for r in res[k * pop_n:(k + 1) * pop_n]]) for k, c in enumerate(cases)}
    used, it = pop_n, 0
    while used < per_case:
        it += 1
        m = min(pop_n, per_case - used)
        trials = {}
        for c in cases:
            pop, rows = pops[c], []
            for i in range(m):
                a, b, cc = pop[rng.choice([j for j in range(pop_n) if j != i], 3, replace=False)]
                mutant = np.clip(a + f_w * (b - cc), -1, 1)
                mask = rng.random(d) < cr
                mask[rng.integers(d)] = True
                rows.append(np.where(mask, mutant, pop[i]))
            trials[c] = rows
        flat = [(c, x) for c in cases for x in trials[c]]
        res = ev.evaluate([x for _, x in flat], [c for c, _ in flat])
        rec.add(res, "de", it)
        for k, (c, x) in enumerate(flat):
            i = k % m
            if -res[k]["reward"] <= fit[c][i]:
                pops[c][i], fit[c][i] = x, -res[k]["reward"]
        used += m
        if it % 5 == 0:
            g = cases[0].code
            print(f"[de] eval {rec.n} best[{g}] TACS {rec.best[g]['TACS']:.0f}", flush=True)
    return {"de_population": pop_n, "de_F": f_w, "de_CR": cr, "de_evaluations_per_case": used}


def shapley(values: dict, players) -> dict:
    """Shapley value of a characteristic function given on frozensets of players."""
    n = len(players)
    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for k in range(n):
            for S in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                total += w * (values[frozenset(S) | {i}] - values[frozenset(S)])
        phi[i] = total
    return phi


def characteristic(case_tacs: dict, players) -> dict:
    """v(S): best saving of any role case whose coalition is contained in S (>= 0)."""
    v = {}
    for k in range(len(players) + 1):
        for S in itertools.combinations(players, k):
            S = frozenset(S)
            v[S] = max([0.0] + [t for code, t in case_tacs.items()
                                if frozenset(p for p, ch in zip(players, code) if ch != "0") <= S])
    return v


def experiment(cfg, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {out}")
    started = time.perf_counter()
    with build(cfg) as ev:
        baseline_check(ev)
        model = ev.model
        cases = select_cases(cfg, model)
        codes = [c.code for c in cases]

        def baseline_of(code):
            return sum(model.baseline[p] for p, ch in zip(model.ids, code) if ch != "0")

        meta = {"config": cfg, "action_labels": model.labels, "cases": codes,
                "return_windows": {c.code: model.return_window(c) for c in cases},
                "outlet_high": model.outlet_high, "duty_max": model.duty_max,
                "site_baseline_TAC": model.site_baseline, "plant_solver": asdict(ev.solver_cfg),
                "site": asdict(model.cfg), "reference_inputs": "none (original plant data only)",
                "paper_reference": PAPER}
        dump(out / "config.json", meta)
        rec = Recorder(out, codes, baseline_of)
        try:
            extra = {"sac": run_sac, "random": run_random, "de": run_de}[cfg["method"]](cfg, ev, rec, out, cases)
        finally:
            rec.close()
        rechecks = {}
        for code, b in rec.best.items():
            if b["result"] is not None:
                r = ev.evaluate([b["result"]["action"]], [RoleCase.from_code(code)])[0]
                rechecks[code] = {"TACS": r["TACS"], "strict": r["strictly_feasible"]}
    per_case = {code: {"best_TACS": b["TACS"], "best_TAC": b["TAC"], "best_eval": b["eval"],
                       "best_roles_executed": role_code(b["result"]["roles"]) if b["result"] else None,
                       "recheck": rechecks.get(code)} for code, b in rec.best.items()}
    players = list(model.ids)
    v_best = characteristic({c: x["best_TACS"] for c, x in per_case.items()}, players)
    summary = {"method": cfg["method"], "seed": cfg["seed"], "budget": cfg["budget"], "evaluations": rec.n,
               "cases": per_case,
               "characteristic_function_best": {"".join(sorted(S)) or "{}": val for S, val in v_best.items()},
               "wall_s": time.perf_counter() - started, **extra}
    if len(cases) == len(model.cases):
        summary["shapley_best"] = shapley(v_best, players)
        if "final_policy" in extra:
            v_pol = characteristic({c: x["TACS"] if x["strict"] else 0.0 for c, x in extra["final_policy"].items()},
                                   players)
            summary["shapley_final_policy"] = shapley(v_pol, players)
    grand = max((c for c in codes if "0" not in c), key=lambda c: per_case[c]["best_TACS"], default=None)
    if grand:
        b = rec.best[grand]
        summary["grand_coalition_best_case"] = grand
        summary["grand_coalition_best_TAC"] = b["TAC"]
        summary["grand_coalition_best_TACS"] = b["TACS"]
        summary["gap_to_paper_strategy_I_pct"] = 100 * (b["TAC"] / PAPER["strategy_I_TAC"] - 1)
        if b["result"] is not None:
            dump(out / "best_design.json", {"eval": b["eval"], **b["result"]})
    dump(out / "summary.json", summary)
    print(json.dumps(jsonable({k: summary.get(k) for k in ("method", "seed", "grand_coalition_best_case",
                                                           "grand_coalition_best_TAC", "gap_to_paper_strategy_I_pct",
                                                           "wall_s")})), flush=True)
    return summary


def validate(cfg, out: Path):
    """Post hoc: original network and the published Fig. 3 interface through the pipeline."""
    out.mkdir(parents=True, exist_ok=True)
    with build(cfg) as ev:
        base = baseline_check(ev)
        case = RoleCase.from_code("-++")
        a = ev.model.encode(100.7, {"P1": -395.9, "P2": 110.5, "P3": 285.4}, {"P2": 140.0, "P3": 153.0}, case)
        paper = ev.evaluate([a], [case])[0]
    report = {"zero_loop_TAC": base["TAC"], "paper_interface_TAC": paper["TAC"],
              "paper_interface_TACS": paper["TACS"], "paper_interface_strict": paper["strictly_feasible"],
              "paper_reported_strategy_I_TAC": PAPER["strategy_I_TAC"],
              "paper_interface_structures": {p: x["structure"]["label"] for p, x in paper["plants"].items()},
              "paper_interface_evaluation": paper}
    dump(out / "validation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "paper_interface_evaluation"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "tshi_decomp.yaml")
    ap.add_argument("--method", choices=["sac", "random", "de"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--budget", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--cases", help="comma-separated role codes, e.g. -++,-+0 (default: config)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("method", "seed", "budget", "workers"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.cases:
        cfg["cases"] = args.cases.split(",")
    if args.validate:
        validate(cfg, args.out or ROOT / "results" / "tshi_decomp" / "validation")
        return
    out = args.out or ROOT / "results" / "tshi_decomp" / f"{cfg['method']}_seed{cfg['seed']}"
    experiment(cfg, out)
    plotter = ROOT / "plot_tshi_decomp.py"
    if not args.no_plot and plotter.exists():
        subprocess.run([sys.executable, str(plotter), str(out.parent)], check=False)


if __name__ == "__main__":
    main()
