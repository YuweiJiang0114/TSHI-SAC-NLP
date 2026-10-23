"""Action vectors from explicit designs (validation and seeding; never used by the search).

A design *spec* names, per plant: the stage of every existing unit, its augmentation,
the area of every new unit, split fractions and bypasses.  ``from_decomp_export`` builds
one from a ``tshi_decomp`` plant export (its series/parallel chains are placed on the
stages by longest-path ordering from the hot end).
"""
from __future__ import annotations

import numpy as np

from tshi_decomp.plant import LOOP

from .plant import PlantSimulator


def actions_from_spec(sim: PlantSimulator, interface, spec: dict) -> np.ndarray:
    """spec keys: stage {unit: k (1-based)}, aug {unit: X m2}, area {new unit id: m2},
    split {(stream, k): {unit id: fraction}}, bypass {(stream, k or 'end'): r}."""
    L, cfg, c = sim.L, sim.cfg, sim.costs
    a = -np.ones(L.dim)
    F = sim.F.copy()
    T_in, T_out = sim.T_in.copy(), sim.T_out.copy()
    if L.role != "inactive":
        li = L.sidx[LOOP]
        F[li], T_in[li], T_out[li] = interface.flow, interface.t_in, interface.t_out
    load = F * np.abs(T_in - T_out)
    for i in L.existing:
        e = L.ex[i]
        k = spec["stage"][e["unit"]] - 1
        a[L.col[("stage", i)]] = -1.0 + (k + 0.5) / L.K * 2.0
        x = spec.get("aug", {}).get(e["unit"], 0.0)
        if x > 0:
            qmax = min(load[L.ex_hot[i]], load[L.ex_cold[i]])
            xmax = max(c.min_augmentation, qmax / (L.U[i] * cfg.approach_scale) - sim.area_ex[i])
            a[L.col[("aug", i)]] = min((x - c.min_augmentation) / max(xmax - c.min_augmentation, 1e-12), 1.0)
    for i in L.loop_units:
        area = spec.get("area", {}).get(L.ex[i]["id"], 0.0)
        if area > 0:
            qmax = min(load[L.ex_hot[i]], load[L.ex_cold[i]])
            amax = qmax / (L.U[i] * cfg.approach_scale)
            u = cfg.area_dead_zone + (1.0 - cfg.area_dead_zone) * min(np.sqrt(area / amax), 1.0)
            a[L.col[("area", i)]] = 2.0 * u - 1.0
    stage_of = {}
    for i in L.existing:
        stage_of[i] = spec["stage"][L.ex[i]["unit"]] - 1
    for i in L.loop_units:
        stage_of[i] = L.ex[i]["stage"]
    lb = cfg.min_fraction
    for g, (si, k, cands, cols) in enumerate(L.groups):
        s = L.streams[si]
        given = spec.get("split", {}).get((s, k + 1))
        if not given:
            continue
        r = spec.get("bypass", {}).get((s, k + 1), 0.0)
        act = [i for i in cands if stage_of[i] == k and L.ex[i]["id"] in given]
        if len(act) < 2:
            continue
        R = 1.0 - r - len(act) * lb
        for n, i in enumerate(cands[:-1]):
            if i in act and i != act[-1]:
                f = given[L.ex[i]["id"]]
                u = (f - lb) / R if R > 1e-12 else 0.0
                a[cols[n]] = 2.0 * min(max(u, 0.0), 1.0) - 1.0
                R -= f - lb
    for (si, k), col in L.bypass_col.items():
        key = (L.streams[si], "end" if k == "end" else k + 1)
        r = spec.get("bypass", {}).get(key, 0.0)
        if r > 0:
            a[col] = (r - cfg.min_bypass) / (0.9 - cfg.min_bypass)
    return a


def stages_from_chains(chains: dict, hot: dict, K: int) -> dict:
    """Stage (1-based) of every unit of a ``tshi_decomp`` chain description: units of a
    parallel group share a stage; stages increase along hot streams and decrease along
    cold streams (longest path from the hot end)."""
    node_of, groups = {}, []
    for sid, chain in chains.items():
        for grp in chain:
            ids = [g for g in groups if set(g) & set(grp)]
            merged = set(grp)
            for g in ids:
                merged |= set(g)
                groups.remove(g)
            groups.append(frozenset(merged))
    for n, g in enumerate(groups):
        for u in g:
            node_of[u] = n
    succ = {n: set() for n in range(len(groups))}
    for sid, chain in chains.items():
        order = [node_of[grp[0]] for grp in chain]
        if not hot[sid]:
            order = order[::-1]
        for u, v in zip(order, order[1:]):
            if u != v:
                succ[u].add(v)
    depth = {}

    def visit(n, seen=()):
        if n in depth:
            return depth[n]
        if n in seen:
            raise ValueError("cyclic chains cannot be staged")
        depth[n] = 1 + max([visit(m, seen + (n,)) for m in succ[n]], default=0)
        return depth[n]

    for n in succ:
        visit(n)
    longest = max(depth.values(), default=0)
    if longest > K:
        raise ValueError(f"chains need {longest} stages, superstructure has {K}")
    # depth counts from the cold end; stage = longest - depth + 1 puts sources at the hot end
    return {u: longest - depth[node_of[u]] + 1 for u in node_of}


def from_decomp_export(x: dict, plant, K: int) -> dict:
    """Design spec of a ``tshi_decomp`` plant export (Strategy I)."""
    hot = {s.id: s.hot for s in plant.streams.values()}
    role = x["interface"]["role"]
    hot[LOOP] = role == "sink"
    chains = {s: c for s, c in x["chains"].items() if c}
    stage = stages_from_chains(chains, hot, K)
    spec = {"stage": {}, "aug": {}, "area": {}, "split": {}, "bypass": {}}
    for e in x["exchangers"]:
        k = stage[e["id"]]
        if e["kind"] == "existing":
            spec["stage"][e["id"]] = k
            if e["augmented"]:
                spec["aug"][e["id"]] = e["augmentation_area"]
            new_id = e["id"]
        else:
            s = e["cold"] if role == "sink" else e["hot"]
            new_id = f"LOOP-{s}@{k}"
            spec["area"][new_id] = e["area_required"]
        for side in ("hot", "cold"):
            s = e[side]
            spec["split"].setdefault((s, k), {})[new_id] = e[f"{side}_fraction"]
    return spec
