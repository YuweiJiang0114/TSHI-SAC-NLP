"""Evaluate trained tshi_sim plant policies at the published Fig. 3 interfaces.

For every plant policy of a run: the deterministic design at the paper's interface and
its TAC against the reference optimum of the old ``tshi_decomp`` solver at that
interface (reference only; the method itself uses no solver).

    python tools/eval_tshi_sim_plants.py results/tshi_sim/s1/case3/seed1b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tshi_decomp.case import load_case
from tshi_decomp.plant import Interface, PlantSolver, paper_rules
from tshi_decomp.site import RoleCase
from tshi_sim.site import SiteSim, stages_for_case

FIG3 = {"P1": Interface(8.260891, 148.5996, 100.7), "P2": Interface(2.81170, 100.7, 140.0),
        "P3": Interface(5.457, 100.7, 153.0)}


def act(agent, obs, deterministic=True):
    import torch
    o = torch.as_tensor(np.asarray(obs, np.float32), device=agent.device)
    with torch.no_grad():
        a = agent.actor.deterministic(o) if deterministic else agent.actor.sample(o)[0]
    return a.cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--samples", type=int, default=32, help="stochastic policy samples per plant")
    args = ap.parse_args()
    from total_site.central_sac import CentralSAC
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    case = RoleCase.from_code(summary["case"])
    site = SiteSim()
    K = stages_for_case(site.plants, case)
    plants, costs = load_case()
    print(f"| Plant | Reference optimum (tshi_decomp) | SAC policy (deterministic) | Best of {args.samples} policy samples |")
    print("|---|---:|---:|---:|")
    for pid, role in site.plant_roles(case).items():
        itf = FIG3[pid]
        sim = site.simulator(pid, role, K)
        ref = PlantSolver(plants[pid], costs, paper_rules()).solve(itf)
        agent = CentralSAC.load(args.run / f"policy_{pid}.pt", restore_rng=False)
        obs = site.observation(itf)
        a = act(agent, obs)
        d = sim.export(a, itf)
        samples = np.stack([act(agent, obs, deterministic=False) for _ in range(args.samples)])
        ev = sim.evaluate(samples, itf)
        best_s = float(ev["tac"].min())
        print(f"| {pid} | {ref['TAC']:,.0f} | {d['TAC']:,.0f}{'' if d['strictly_feasible'] else ' (infeasible)'} | {best_s:,.0f} |")
        print(f"   policy design: slack {d['slack_kW']:.3g} kW, violation {d['violation_kW']:.3g} kW, "
              f"units {[(e['id'], e['stage'], round(e['duty'], 1)) for e in d['exchangers']]}")


if __name__ == "__main__":
    main()
