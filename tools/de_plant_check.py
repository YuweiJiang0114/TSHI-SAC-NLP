"""Can a black-box optimizer find the known plant optimum in the tshi_sim action space?

Differential evolution on one plant's action vector at a fixed interface (default: the
published Fig. 3 interface), objective = penalized plant TAC.  Compares with the
enumeration solver's optimum at the same interface.

    python tools/de_plant_check.py --plant P3 --generations 150 --popsize 2
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import differential_evolution

from tshi_decomp.case import load_case
from tshi_decomp.plant import Interface, PlantSolver, paper_rules
from tshi_sim.plant import PlantLayout, PlantSimulator

FIG3 = {"P1": ("sink", Interface(8.260891, 148.5996, 100.7)), "P2": ("source", Interface(2.812, 100.7, 140.0)),
        "P3": ("source", Interface(5.4570, 100.7, 153.0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plant", default="P3")
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--popsize", type=int, default=2, help="population = popsize x dim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--flow", type=float)
    ap.add_argument("--t-in", type=float)
    ap.add_argument("--t-out", type=float)
    args = ap.parse_args()
    plants, costs = load_case()
    role, itf = FIG3[args.plant]
    if args.flow is not None:
        itf = Interface(args.flow, args.t_in, args.t_out)
    sim = PlantSimulator(PlantLayout(plants[args.plant], costs, role, args.stages))
    t = time.perf_counter()
    ref = PlantSolver(plants[args.plant], costs, paper_rules()).solve(itf)
    print(f"{args.plant} {role} interface {itf}: enumeration solver TAC {ref['TAC']:.1f} "
          f"(strict {ref['strictly_feasible']}, {time.perf_counter() - t:.1f}s); action dim {sim.L.dim}")
    calls = [0]

    def objective(X):
        calls[0] += X.shape[1]
        return sim.evaluate(X.T, itf)["tac"]

    t = time.perf_counter()
    res = differential_evolution(objective, [(-1, 1)] * sim.L.dim, maxiter=args.generations, popsize=args.popsize,
                                 seed=args.seed, vectorized=True, updating="deferred", polish=False, tol=0, init="sobol")
    d = sim.export(res.x, itf)
    print(f"DE: {calls[0]} evaluations in {time.perf_counter() - t:.1f}s -> TAC {d['TAC']:.1f} "
          f"(clean {d['costs']['TAC_without_penalty']:.1f}, strict {d['strictly_feasible']}, slack {d['slack_kW']:.3g}, "
          f"viol {d['violation_kW']:.3g}); gap to solver {d['TAC'] - ref['TAC']:+.1f}")
    for e in d["exchangers"]:
        print(f"   {e['id']:12s} stage {e['stage']} q {e['duty']:8.2f} A {e['area_required']:8.2f} "
              f"aug {e.get('augmentation_area', 0):7.2f} f {e['hot_fraction']:.3f}/{e['cold_fraction']:.3f}")
    for u in d["utilities"]:
        print(f"   util {u['stream']:4s} q {u['duty']:8.2f} A {u['area_required']:7.2f} X {u['augmentation_area']:6.2f} r {u['bypass']:.3f}")
    print("   bypasses", d["bypasses"])


if __name__ == "__main__":
    main()
