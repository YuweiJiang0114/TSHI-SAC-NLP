"""SAC on the simulation model (tshi_sim): plant agents and the site agent together.

Joint training (default, ``phase1.iterations: 0``): every iteration the site agent
proposes interfaces (random during the warmup), every plant agent designs its plant at
its interface, the site is simulated, and all agents learn from the same evaluation with
the same reward: the total TAC saving of the site (``plant_reward: total``; ``own`` gives
a plant agent its own plant's saving instead).

Optional phase 1 (``phase1.iterations > 0``): plant agents are first trained alone on
random interfaces before the joint phase.

    python run_tshi_sim.py configs/tshi_sim_s1.yaml --out results/tshi_sim/s1/case3/seed1
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from tshi_decomp.site import RoleCase, SiteConfig
from tshi_sim.plant import SimConfig
from tshi_sim.site import SiteSim, stages_for_case


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


def act_batch(agent, obs, deterministic=False):
    import torch
    o = torch.as_tensor(np.asarray(obs, np.float32), device=agent.device)
    with torch.no_grad():
        a = agent.actor.deterministic(o) if deterministic else agent.actor.sample(o)[0]
    return a.cpu().numpy().astype(np.float64)


PLANT_OBS = {"mode": "own"}


def plant_obs(site, pid, decoded):
    """A plant agent's observation: its own interface, or the whole site's (``plant_obs: site``)."""
    if PLANT_OBS["mode"] == "site":
        return site.site_observation(pid, decoded)
    return site.observation(decoded["_interfaces"][pid])


def make_agent(obs_dim, dim, sac_cfg, seed):
    from total_site.central_sac import CentralSAC, SACConfig
    cfg = dict(sac_cfg)
    cfg["hidden_dims"] = tuple(cfg.get("hidden_dims", (256, 256)))
    per_dim = cfg.pop("target_entropy_per_dim", None)
    if per_dim is not None:
        cfg["target_entropy"] = float(per_dim) * dim
    return CentralSAC(obs_dim, -np.ones(dim), np.ones(dim), SACConfig(seed=seed, **cfg))


class Best:
    """Best strictly feasible site design seen so far."""

    def __init__(self):
        self.tac = np.inf
        self.record = None

    def offer(self, r, site_action, plant_actions, n_eval, phase):
        if r["strictly_feasible"] and r["TAC"] < self.tac:
            self.tac = r["TAC"]
            self.record = {"TAC": r["TAC"], "TACS": r["TACS"], "eval": n_eval, "phase": phase,
                           "site_action": np.array(site_action, float), "plant_actions": {p: np.array(a, float) for p, a in plant_actions.items()}}
            return True
        return False


def phase_one(cfg, site, case, agents, rng, out, log):
    p1 = cfg.get("phase1", {})
    iterations, batch = int(p1.get("iterations", 1000)), int(p1.get("batch", 64))
    warmup, utd = int(p1.get("warmup_iterations", 20)), float(p1.get("utd", 0.5))
    roles = site.plant_roles(case)
    f = (out / "plant_curve.csv").open("w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["iteration", "plant", "reward_mean", "reward_max", "tac_clean_min_strict", "strict_fraction",
                "violation_mean", "slack_mean", "alpha", "seconds"])
    started = time.perf_counter()
    for it in range(1, iterations + 1):
        sa = rng.uniform(-1.0, 1.0, (batch, site.model.action_dim))
        decoded = [site.model.decode(a, case) for a in sa]
        obs = {p: np.stack([plant_obs(site, p, d) for d in decoded]) for p in roles}
        acts = {}
        for p, agent in agents.items():
            acts[p] = rng.uniform(-1.0, 1.0, (batch, agent.action_dim)) if it <= warmup else act_batch(agent, obs[p])
        res = site.evaluate(sa, case, acts)
        for p, agent in agents.items():
            rewards, active = [], 0
            for i, r in enumerate(res):
                pr = r["plant"][p]
                if not pr["active"] or not pr["feasible_interface"]:
                    continue  # a plant learns only at interfaces it can possibly serve
                agent.remember(obs[p][i], acts[p][i], pr["reward"], obs[p][i], True)
                rewards.append(pr["reward"])
                active += 1
            if it > warmup:
                for _ in range(int(round(utd * active))):
                    agent.update()
            if it % 10 == 0 or it == iterations:
                pr = [r["plant"][p] for r in res if r["plant"][p]["active"] and r["plant"][p]["feasible_interface"]]
                strict = [x["TAC_clean"] for x in pr if x["strict"]]
                w.writerow([it, p, f"{np.mean(rewards):.6f}" if rewards else "", f"{np.max(rewards):.6f}" if rewards else "",
                            f"{min(strict):.1f}" if strict else "", f"{len(strict) / max(len(pr), 1):.3f}",
                            f"{np.mean([x['violation'] for x in pr]):.4g}" if pr else "",
                            f"{np.mean([x['slack'] for x in pr]):.4g}" if pr else "",
                            f"{float(agent.alpha):.4g}", f"{time.perf_counter() - started:.0f}"])
        if it % 10 == 0 or it == iterations:
            f.flush()
            parts = []
            for p in roles:
                pr = [r["plant"][p] for r in res if r["plant"][p]["active"] and r["plant"][p]["feasible_interface"]]
                parts.append(f"{p}:{np.mean([x['reward'] for x in pr]) if pr else float('nan'):+.3f}/"
                             f"{np.mean([x['strict'] for x in pr]) if pr else float('nan'):.2f}/{len(pr)}/a{float(agents[p].alpha):.2g}")
            log(f"[phase1] it {it}/{iterations} reward/strict/n/alpha {' '.join(parts)} "
                f"site strict {np.mean([r['strictly_feasible'] for r in res]):.2f} {time.perf_counter() - started:.0f}s")
    f.close()


def phase_two(cfg, site, case, master, agents, rng, out, log, best):
    p2 = cfg.get("phase2", {})
    budget, batch = int(p2.get("budget", 3000)), int(p2.get("batch", 24))
    warmup, utd = int(p2.get("warmup", 240)), float(p2.get("utd", 4))
    plant_utd, explore = float(p2.get("plant_utd", 0.5)), bool(p2.get("plant_explore", True))
    probe_every = int(p2.get("policy_eval_every", 5))
    random_fraction = float(p2.get("random_fraction", 0.0))  # share of random site actions after warmup
    p1_iterations = int(cfg.get("phase1", {}).get("iterations", 0))
    # Adaptive warmup: random site actions until this many strictly feasible designs beat the baseline.
    warmup_until_strict = int(p2.get("warmup_until_strict", 0))
    # Rule curriculum: [{until_eval, sim: {...SimConfig overrides}}, ...]; the last entry (or the
    # base ``sim`` config) applies afterwards.  Only designs under the final rules are reported.
    curriculum = list(cfg.get("curriculum", []))
    base_sim = dict(cfg.get("sim", {}))

    def rules_at(n_eval):
        for stage in curriculum:
            if n_eval < int(stage["until_eval"]):
                return dict(base_sim, **stage.get("sim", {}))
        return base_sim

    current_rules = {"value": None}

    def apply_rules(n_eval):
        rules = rules_at(n_eval)
        if rules != current_rules["value"]:
            current_rules["value"] = rules
            site.sim_cfg = SimConfig(**rules)
            site._sims.clear()
            log(f"[rules] from eval {n_eval}: SimConfig overrides {rules or 'none (paper rules)'}")
        return rules == base_sim
    # "total": every agent is rewarded with the site's TAC saving; "own": a plant agent with
    # its own plant's saving (equivalent in expectation given the interface, lower variance).
    plant_reward = str(cfg.get("plant_reward", "total"))
    if plant_reward not in ("total", "own"):
        raise ValueError("plant_reward must be 'total' or 'own'")
    roles = site.plant_roles(case)
    obs_case = case.observation()
    f = (out / "evals.csv").open("w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["eval", "iteration", "phase", "TAC", "TACS", "TAC_without_penalty", "strictly_feasible", "slack_kW",
                "violation_kW", "T_return", "T_supply", "loop_flow"] + [f"TAC:{p}" for p in roles] + site.model.labels)
    started = time.perf_counter()
    n, it, last_probe = 0, 0, float("nan")

    good = 0  # strictly feasible designs better than the baseline, under the final rules

    def run(sa, phase, deterministic_plants):
        nonlocal n, good
        final_rules = apply_rules(n)
        decoded = [site.model.decode(a, case) for a in sa]
        obs = {p: np.stack([plant_obs(site, p, d) for d in decoded]) for p in roles}
        acts = {p: act_batch(agents[p], obs[p], deterministic=deterministic_plants) for p in roles}
        res = site.evaluate(sa, case, acts)
        for i, r in enumerate(res):
            n += 1
            master.remember(obs_case, sa[i], r["reward"], obs_case, True)
            for p in roles:
                pr = r["plant"][p]
                if pr["active"] and pr["feasible_interface"]:
                    reward = r["reward"] if plant_reward == "total" else pr["reward"]
                    agents[p].remember(obs[p][i], acts[p][i], reward, obs[p][i], True)
            if final_rules:
                good += bool(r["strictly_feasible"] and r["TACS"] > 0)
                best.offer(r, sa[i], {p: acts[p][i] for p in roles}, n, phase)
            w.writerow([n, it, phase, f"{r['TAC']:.4f}", f"{r['TACS']:.4f}", f"{r['TAC_without_penalty']:.4f}",
                        int(r["strictly_feasible"]), f"{r['slack_kW']:.6g}", f"{r['violation_kW']:.6g}",
                        f"{r['T_return']:.4f}", f"{r['T_supply']:.4f}", f"{r['loop_flow']:.6f}"]
                       + [f"{r['plant_TAC'][p]:.2f}" for p in roles] + [f"{x:.6f}" for x in r["action"]])
        return res

    while n < budget:
        it += 1
        nb = min(batch, budget - n)
        phase = "warmup" if (n < warmup or good < warmup_until_strict) else "sac"
        if phase == "warmup":
            sa = rng.uniform(-1.0, 1.0, (nb, site.model.action_dim))
        else:
            sa = np.stack([master.act(obs_case) for _ in range(nb)])
            k = int(round(random_fraction * nb))
            if k:
                sa[:k] = rng.uniform(-1.0, 1.0, (k, site.model.action_dim))
        res = run(sa, phase, deterministic_plants=not explore)
        if phase == "sac" and it % probe_every == 0 and n < budget:
            pres = run(np.stack([master.act(obs_case, deterministic=True)]), "policy", deterministic_plants=True)
            last_probe = pres[0]["TAC"]
        if n >= warmup:
            for _ in range(int(round(nb * utd))):
                master.update()
            for p in roles:
                for _ in range(int(round(nb * plant_utd))):
                    agents[p].update()
        if it % 5 == 0:
            f.flush()
            pl = " ".join(f"{p}:{np.mean([r['plant'][p]['reward'] for r in res if r['plant'][p]['active']] or [np.nan]):+.3f}"
                          for p in roles)
            log(f"[{'joint' if not p1_iterations else 'phase2'}/{phase}] eval {n}/{budget} batch best TAC "
                f"{min(r['TAC'] for r in res):.0f} strict {np.mean([r['strictly_feasible'] for r in res]):.2f} "
                f"good {good} best {best.tac:.0f} policy {last_probe:.0f} alpha {float(master.alpha):.3g} plants {pl} "
                f"{time.perf_counter() - started:.0f}s")
    final_sa = master.act(obs_case, deterministic=True)
    final = run(np.stack([final_sa]), "final", deterministic_plants=True)[0]
    f.close()
    return final


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

    site = SiteSim(SiteConfig(**cfg.get("site", {})), SimConfig(**cfg.get("sim", {})))
    PLANT_OBS["mode"] = str(cfg.get("plant_obs", "own"))
    case = RoleCase.from_code(cfg.get("case", "-++"))
    dims = site.plant_dims(case)
    sac = cfg.get("sac", {})
    plant_sac = dict(sac, **cfg.get("plant_sac", {}))
    obs_dim = site.site_observation_dim if cfg.get("plant_obs", "own") == "site" else 4
    agents = {p: make_agent(obs_dim, d, plant_sac, seed + 100 + k) for k, (p, d) in enumerate(dims.items())}
    master = make_agent(len(site.plants), site.model.action_dim, dict(sac, **cfg.get("site_sac", {})), seed)
    rng = np.random.default_rng(seed)
    log(f"case {case.code} stages {stages_for_case(site.plants, case)} plant action dims {dims} "
        f"site action dim {site.model.action_dim}")
    best = Best()
    t0 = time.perf_counter()
    if int(cfg.get("phase1", {}).get("iterations", 0)) > 0:
        phase_one(cfg, site, case, agents, rng, out, log)
    else:
        log("joint training: plant agents and the site agent learn together from the first evaluation")
    t1 = time.perf_counter()
    final = phase_two(cfg, site, case, master, agents, rng, out, log, best)
    t2 = time.perf_counter()
    for p, agent in agents.items():
        agent.save(out / f"policy_{p}.pt", include_replay=False)
    master.save(out / "policy_site.pt", include_replay=False)
    summary = {"case": case.code, "seed": seed, "plant_action_dims": dims, "phase1_seconds": t1 - t0,
               "phase2_seconds": t2 - t1, "final_policy": {k: final[k] for k in ("TAC", "TACS", "strictly_feasible", "slack_kW", "violation_kW")}}
    if best.record is not None:
        x = site.export(best.record["site_action"], case, best.record["plant_actions"])
        x["eval"], x["phase"] = best.record["eval"], best.record["phase"]
        (out / "best_design.json").write_text(json.dumps(jsonable(x), indent=1), encoding="utf-8")
        summary["best"] = {"TAC": best.record["TAC"], "TACS": best.record["TACS"], "eval": best.record["eval"]}
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1), encoding="utf-8")
    log(f"done: best strict TAC {best.tac:.1f} final policy TAC {final['TAC']:.1f} (strict {final['strictly_feasible']}) "
        f"phase1 {t1 - t0:.0f}s phase2 {t2 - t1:.0f}s")


if __name__ == "__main__":
    main()
