"""One SAC agent sets every area, stage and loop flow; a site NLP solves the operation.

    python run_tshi_nlp.py configs/tshi_nlp_s1.yaml --out results/tshi_nlp/s1/case3/seed1

Each evaluation = decode the action (definite areas, stages, loop flows) -> site NLP
(IPOPT, simulator-seeded) -> TAC, rules, reward.  Evaluations run in parallel processes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from tshi_decomp.site import RoleCase
from tshi_nlp.evaluate import Rules, SiteNLP
from tshi_nlp.model import NLPConfig

_W = {}


def _init_worker(cfg, rules):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["site"] = SiteNLP(cfg=NLPConfig(**cfg), rules=Rules(**rules))


def _eval(a, code, hindsight=False):
    site = _W["site"]
    case = RoleCase.from_code(code)
    r = site.evaluate(np.asarray(a, float), case)
    if hindsight and r["solver"]["ok"]:
        a_h = site.relabel_action(a, r, case)
        if a_h is not None:
            r["hindsight"] = site.evaluate_relabeled(a_h, case)
    return r


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def make_agent(obs_dim, dim, sac_cfg, seed):
    from total_site.central_sac import CentralSAC, SACConfig
    cfg = dict(sac_cfg)
    cfg["hidden_dims"] = tuple(cfg.get("hidden_dims", (256, 256)))
    per_dim = cfg.pop("target_entropy_per_dim", None)
    if per_dim is not None:
        cfg["target_entropy"] = float(per_dim) * dim
    return CentralSAC(obs_dim, -np.ones(dim), np.ones(dim), SACConfig(seed=seed, **cfg))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 1))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))
    logf = (out / "log.txt").open("a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    nlp_cfg, rules = dict(cfg.get("nlp", {})), dict(cfg.get("rules", {}))
    site = SiteNLP(cfg=NLPConfig(**nlp_cfg), rules=Rules(**rules))
    case = RoleCase.from_code(cfg.get("case", "-++"))
    dim = site.action_dim(case)
    layout = site.layout(case)
    obs = case.observation()
    log(f"case {case.code}: action dim {dim}: " + ", ".join(f"{k}:{p}:{key}" for k, p, key in layout))
    agent = make_agent(len(site.plants), dim, cfg.get("sac", {}), seed)
    rng = np.random.default_rng(seed)
    budget, batch = int(cfg["budget"]), int(cfg.get("batch", 24))
    warmup, utd = int(cfg.get("warmup", 256)), float(cfg.get("utd", 4))
    probe_every = int(cfg.get("policy_eval_every", 5))
    random_fraction = float(cfg.get("random_fraction", 0.0))
    # Reward shaping (training only): the annualized capital of area the operation does not
    # use is charged ``unused_area_weight`` extra times.  Reported TACs exclude it.
    unused_w = float(cfg.get("unused_area_weight", 0.0))
    # Exploration of the discrete stage choices: in training evaluations every "stage" entry
    # of a policy action is re-drawn uniformly with this probability.
    stage_explore = float(cfg.get("stage_explore", 0.0))
    stage_cols = [i for i, (k, _, _) in enumerate(layout) if k == "stage"]
    # Hindsight relabeling: besides the executed action, store the action that buys exactly
    # the areas the NLP operation used, rewarded with the trimmed TAC (same operation).
    hindsight = bool(cfg.get("hindsight", False))
    workers = int(cfg.get("workers", max(1, min(18, (os.cpu_count() or 2) - 2))))
    pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(nlp_cfg, rules))
    f = (out / "evals.csv").open("w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["eval", "iteration", "phase", "TAC", "TACS", "TAC_with_penalty", "strictly_feasible", "violation_kW",
                "T_return", "T_supply", "loop_flow", "solver_ok", "solver_seconds", "TAC_trimmed", "unused_area_annual_cost"]
               + [f"TAC:{p}" for p in site.plant_roles(case)]
               + [f"{k}:{p}:{key}" for k, p, key in layout])
    best = {"tac": np.inf, "record": None}
    n, it, last_probe, started = 0, 0, float("nan"), time.perf_counter()

    def run(actions, phase):
        nonlocal n
        futures = [pool.submit(_eval, a, case.code, hindsight) for a in actions]
        results = [fu.result() for fu in futures]
        for a, r in zip(actions, results):
            n += 1
            reward = r["reward"]
            if unused_w > 0 and np.isfinite(r.get("unused_area_annual_cost", np.nan)):
                baseline = r["TAC"] + r["TACS"]
                s = (baseline - r["TAC"] - r["penalty"] - unused_w * r["unused_area_annual_cost"]) / site.site_baseline
                reward = float(s if s >= 0 else s / (1.0 - s))
            agent.remember(obs, a, reward, obs, True)
            rh = r.get("hindsight")
            if rh is not None and np.isfinite(rh["reward"]):
                # The relabeled action was actually evaluated (same operation, trimmed areas):
                # its own reward, including the unused-area term, which is ~0 by construction.
                reward_h = rh["reward"]
                if unused_w > 0 and np.isfinite(rh.get("unused_area_annual_cost", np.nan)):
                    baseline = rh["TAC"] + rh["TACS"]
                    s = (baseline - rh["TAC"] - rh["penalty"] - unused_w * rh["unused_area_annual_cost"]) / site.site_baseline
                    reward_h = float(s if s >= 0 else s / (1.0 - s))
                agent.remember(obs, np.asarray(rh["action"], float), reward_h, obs, True)
                if rh["strictly_feasible"] and rh["TAC"] < best["tac"]:
                    best["tac"] = rh["TAC"]
                    best["record"] = {"eval": n, "phase": phase + "+hindsight", **{k: v for k, v in rh.items()}}
            if r["strictly_feasible"] and r["TAC"] < best["tac"]:
                best["tac"] = r["TAC"]
                best["record"] = {"eval": n, "phase": phase, **r}
            w.writerow([n, it, phase, f"{r['TAC']:.4f}", f"{r['TACS']:.4f}", f"{r['TAC_with_penalty']:.4f}",
                        int(r["strictly_feasible"]), f"{r['violation_kW']:.6g}", f"{r['T_return']:.4f}",
                        f"{r['T_supply']:.4f}", f"{r['loop_flow']:.6f}", int(r["solver"]["ok"]),
                        f"{r['solver']['seconds']:.2f}", f"{r.get('TAC_trimmed', float('nan')):.4f}",
                        f"{r.get('unused_area_annual_cost', float('nan')):.4f}"]
                       + [f"{r['plants'][p]['TAC']:.2f}" for p in site.plant_roles(case)]
                       + [f"{x:.6f}" for x in a])
        return results

    try:
        while n < budget:
            it += 1
            nb = min(batch, budget - n)
            phase = "warmup" if n < warmup else "sac"
            if phase == "warmup":
                actions = rng.uniform(-1.0, 1.0, (nb, dim))
            else:
                actions = np.stack([agent.act(obs) for _ in range(nb)])
                k = int(round(random_fraction * nb))
                if k:
                    actions[:k] = rng.uniform(-1.0, 1.0, (k, dim))
                if stage_explore > 0 and stage_cols:
                    redraw = rng.uniform(size=(nb, len(stage_cols))) < stage_explore
                    fresh = rng.uniform(-1.0, 1.0, (nb, len(stage_cols)))
                    sub = actions[:, stage_cols]
                    actions[:, stage_cols] = np.where(redraw, fresh, sub)
            res = run(actions, phase)
            if phase == "sac" and it % probe_every == 0 and n < budget:
                pres = run(np.stack([agent.act(obs, deterministic=True)]), "policy")
                last_probe = pres[0]["TAC"] if pres[0]["strictly_feasible"] else float("nan")
            if n >= warmup:
                for _ in range(int(round(nb * utd))):
                    agent.update()
            if it % 2 == 0:
                f.flush()
                log(f"[{phase}] eval {n}/{budget} batch best TAC {min(r['TAC'] for r in res):.0f} strict "
                    f"{np.mean([r['strictly_feasible'] for r in res]):.2f} best {best['tac']:.0f} policy {last_probe:.0f} "
                    f"alpha {float(agent.alpha):.3g} solver ok {np.mean([r['solver']['ok'] for r in res]):.2f} "
                    f"{np.mean([r['solver']['seconds'] for r in res]):.1f}s/eval {time.perf_counter() - started:.0f}s")
        final = run(np.stack([agent.act(obs, deterministic=True)]), "final")[0]
    finally:
        f.close()
        pool.shutdown(cancel_futures=True)
    agent.save(out / "policy.pt", include_replay=False)
    if best["record"] is not None:
        (out / "best_design.json").write_text(json.dumps(jsonable(best["record"]), indent=1), encoding="utf-8")
    (out / "final_policy.json").write_text(json.dumps(jsonable(final), indent=1), encoding="utf-8")
    summary = {"case": case.code, "seed": seed, "action_dim": dim, "evaluations": n, "seconds": time.perf_counter() - started,
               "best": {k: best["record"][k] for k in ("TAC", "TACS", "eval")} if best["record"] else None,
               "final_policy": {k: final[k] for k in ("TAC", "TACS", "strictly_feasible", "violation_kW")}}
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1), encoding="utf-8")
    log(f"done: best strict TAC {best['tac']:.1f}, final policy TAC {final['TAC']:.1f} (strict {final['strictly_feasible']}), "
        f"{time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
