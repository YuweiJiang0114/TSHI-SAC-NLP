"""One plant's retrofit at a fixed intermediate-fluid interface (Strategy I).

The site fixes the plant's intermediate stream: hot ("sink" plant) or cold
("source" plant), its heat-capacity flowrate and its inlet/outlet temperatures.
This module then designs the plant's retrofit, independently of other plants:

* every existing process match stays in its original unit and in the original
  series/parallel order of Fig. 2; its duty is free and its area may be augmented;
* one or two purchased intermediate-fluid exchangers are inserted on distinct
  compatible process streams, either in series at any gap of the stream's
  original chain or as a new parallel branch of an existing unit;
* existing coolers/heaters keep their utility, sit at the stream end and may be
  turned down, switched off or augmented; a new utility unit is purchased if a
  stream without one needs utility.

Physical model: countercurrent units, Chen's LMTD approximation, minimum approach
``dt_min`` at both ends, exact stream energy balances, non-isothermal mixing of
parallel branches.  Simplifications relative to the paper's MINLP (documented
modelling choices): an existing unit whose required area is below its installed
area is throttled by a bypass whose $500 repiping cost is neglected; the paper's
30 kW / 0.1 lower bounds are applied only to new intermediate-fluid exchangers
(duty) and to split fractions (0.05).

Optimisation: enumerate structures, rank them by Sobol samples evaluated in
one vectorized pass, then polish the best samples of the best structures with
SLSQP.  A priced slack on the intermediate stream (``slack_price``) keeps every
interface evaluable; a design is *strictly feasible* only when the slack is zero.
The solution is deterministic for a given interface.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .case import Costs, Plant

LOOP = "LOOP"


@dataclass(frozen=True)
class Interface:
    """Intermediate-fluid stream seen by one plant.  ``t_in > t_out``: plant is a sink."""
    flow: float = 0.0  # kW/K
    t_in: float = 0.0
    t_out: float = 0.0

    @property
    def duty(self) -> float:
        return self.flow * abs(self.t_in - self.t_out)

    @property
    def active(self) -> bool:
        return self.flow > 0 and self.duty > 1e-9

    @property
    def role(self) -> str:
        if not self.active:
            return "inactive"
        return "sink" if self.t_in > self.t_out else "source"


@dataclass(frozen=True)
class Placement:
    stream: str
    mode: str  # "series" | "parallel"
    position: int  # gap index for series, group index for parallel


@dataclass(frozen=True)
class NewMatch:
    """Strategy III: a new process-process match at series gaps of both streams.

    Gaps index the stream chains after the intermediate exchangers are inserted.
    """
    hot: str
    cold: str
    hot_gap: int
    cold_gap: int


@dataclass(frozen=True)
class Structure:
    placements: tuple = ()
    loop_arrangement: str = "series"  # order of placements along the loop, or "parallel"
    # Strategy II: (match, housing unit) for every existing match not in its own unit.
    # A match is named by the id of the unit that housed it in Fig. 2.
    assignment: tuple = ()
    new_matches: tuple = ()  # Strategy III

    def label(self) -> str:
        if not self.placements:
            text = "no intermediate exchanger"
        else:
            parts = [f"{p.stream}:{p.mode}{p.position}" for p in self.placements]
            text = (" | " if self.loop_arrangement == "parallel" else " -> ").join(parts)
        if self.assignment:
            text += " ; " + ", ".join(f"{m} in {u}" for m, u in self.assignment)
        for n in self.new_matches:
            text += f" + new {n.hot}-{n.cold} @{n.hot_gap}/{n.cold_gap}"
        return text


@dataclass(frozen=True)
class PlantSolverConfig:
    dt_min: float = 1.0
    max_loop_exchangers: int = 2
    min_loop_duty: float = 30.0
    min_fraction: float = 0.05
    # USD/(kW yr) on unmet intermediate-fluid duty.  It exceeds any saving a kW of
    # loop heat can earn (dearest hot utility plus dearest cold utility, 1250), so
    # slack is never part of an optimum; it only grades infeasible interfaces.
    slack_price: float = 2000.0
    area_tolerance: float = 0.05  # m2: half digit of Fig. 2 printed areas
    # Paper-rule options (Lo et al. Sec. 6); the defaults give the relaxed model.
    free_area: float | None = None  # m2 of excess area left uncharged; None: area_tolerance
    min_unit_duty: float = 0.0  # kW; an active existing exchanger or utility needs at least this
    bypass_cost: float = 0.0  # USD per existing unit throttled below its installed area
    duty_tolerance: float = 1e-3  # kW: smaller utility residuals count as zero
    feasibility_tolerance: float = 1e-4  # K
    samples: int = 512
    # Share of samples whose existing-unit duties stay near the original operating
    # point (factor range below); retrofits rarely move them far.
    local_sample_fraction: float = 0.5
    local_duty_range: tuple = (0.5, 1.05)
    top_structures: int = 5
    exhaustive_doubles: int = 12
    polish_per_structure: int = 2
    polish_maxiter: int = 100
    polish_ftol: float = 1e-7  # objective is scaled by 1e-5: about USD 0.01/yr
    polish_start_pass: bool = True  # SLSQP with the start point's fixed-cost decisions
    allow_new_utility_units: bool = True
    # Revamp strategy of Lo et al. Sec. 3: 1 = matches stay in their own units;
    # 2 = an existing unit may house another existing match of the same type;
    # 3 = new process matches may be added and existing matches dropped; every match
    #     sits in an existing unit (each existing unit houses exactly one match) or a
    #     purchased one, chosen optimally for every evaluated point.
    strategy: int = 1
    max_new_matches: int = 1  # Strategy III: new process-process matches per plant
    new_match_top_structures: int = 3  # loop structures given new-match variants
    new_match_polish: int = 3  # new-match structures refined
    # Strategies I/II keep every existing match (at least ``min_unit_duty``).
    require_existing_matches: bool = False
    assignment_top_structures: int = 1  # structures refined per non-identity assignment
    seed: int = 12345


def paper_rules(**overrides) -> PlantSolverConfig:
    """Lo et al. Sec. 6 bounds: 30 kW minimum duty of every active unit, 0.1 minimum
    split fraction, $500 per (throttling) bypass, and no uncharged excess area.
    Strategies I/II also keep every existing match and buy no utility units."""
    strategy = overrides.get("strategy", 1)
    rules = dict(min_fraction=0.1, free_area=1e-6, min_unit_duty=30.0, bypass_cost=500.0,
                 require_existing_matches=strategy < 3, allow_new_utility_units=strategy >= 3)
    rules.update(overrides)
    return PlantSolverConfig(**rules)


def chen(a, b):
    return np.cbrt(a * b * (a + b) / 2.0)


class Network:
    """A fixed retrofit structure with a vectorized evaluator over decision vectors."""

    def __init__(self, plant: Plant, costs: Costs, cfg: PlantSolverConfig,
                 structure: Structure, interface: Interface, area_existing: dict):
        self.plant, self.costs, self.cfg = plant, costs, cfg
        self.structure, self.interface = structure, interface
        self.free_area = cfg.area_tolerance if cfg.free_area is None else cfg.free_area
        self.active = interface.active
        streams = {sid: s for sid, s in plant.streams.items()}
        self.loop_kind = None
        if self.active:
            self.loop_kind = "hot" if interface.role == "sink" else "cold"
        # Exchangers: existing first, then loop units in placement order.
        self.ex_ids, self.ex_hot, self.ex_cold, self.ex_kind = [], [], [], []
        for e in plant.exchangers.values():
            self.ex_ids.append(e.id)
            self.ex_hot.append(e.hot)
            self.ex_cold.append(e.cold)
            self.ex_kind.append("existing")
        chains = {sid: [list(g) for g in chain] for sid, chain in plant.chains.items()}
        loop_ids = []
        for p in structure.placements:
            lid = f"{LOOP}-{p.stream}"
            loop_ids.append(lid)
            self.ex_ids.append(lid)
            self.ex_kind.append("loop")
            if self.loop_kind == "hot":
                self.ex_hot.append(LOOP)
                self.ex_cold.append(p.stream)
            else:
                self.ex_hot.append(p.stream)
                self.ex_cold.append(LOOP)
            chain = chains[p.stream]
            if p.mode == "series":
                chain.insert(p.position, [lid])
            else:
                chain[p.position].append(lid)
        if loop_ids:
            chains[LOOP] = [loop_ids] if structure.loop_arrangement == "parallel" else [[i] for i in loop_ids]
        for k, nm in enumerate(structure.new_matches):
            nid = f"NEW{k + 1}-{nm.hot}-{nm.cold}"
            self.ex_ids.append(nid)
            self.ex_hot.append(nm.hot)
            self.ex_cold.append(nm.cold)
            self.ex_kind.append("new")
            chains[nm.hot].insert(nm.hot_gap, [nid])
            chains[nm.cold].insert(nm.cold_gap, [nid])
        self.chains = chains
        self.n_ex = len(self.ex_ids)
        self.index = {eid: i for i, eid in enumerate(self.ex_ids)}
        self.existing_idx = np.array([i for i, k in enumerate(self.ex_kind) if k == "existing"], dtype=int)
        self.loop_idx = np.array([i for i, k in enumerate(self.ex_kind) if k == "loop"], dtype=int)
        self.new_idx = np.array([i for i, k in enumerate(self.ex_kind) if k == "new"], dtype=int)
        self.s3 = cfg.strategy >= 3
        # Strategy III: process matches (existing and new) and the injective maps
        # unit -> match; every existing unit houses exactly one match (Eq. 54).
        self.match_idx = np.concatenate([self.existing_idx, self.new_idx]).astype(int)
        self.units = list(plant.exchangers)
        self.unit_maps = list(itertools.permutations(range(len(self.match_idx)), len(self.units))) if self.s3 else []
        self.stream_data = {sid: (s.hot, s.t_in, s.t_out, s.fcp, s.h) for sid, s in streams.items()}
        if self.active:
            self.stream_data[LOOP] = (self.loop_kind == "hot", interface.t_in, interface.t_out,
                                      interface.flow, costs.loop_h)
        h = {sid: d[4] for sid, d in self.stream_data.items()}
        self.U = np.array([1.0 / (1.0 / h[a] + 1.0 / h[b]) for a, b in zip(self.ex_hot, self.ex_cold)])
        housing = dict(structure.assignment)
        self.housing = [housing.get(eid, eid) if k == "existing" else None
                        for eid, k in zip(self.ex_ids, self.ex_kind)]
        self.area_existing = np.array([area_existing.get(u, 0.0) if u else 0.0 for u in self.housing])
        moved = 0  # units housing a match with another stream pair (Eq. 76)
        for eid, u in zip(self.ex_ids, self.housing):
            if u and u != eid:
                a, b = plant.exchangers[eid], plant.exchangers[u]
                moved += (a.hot, a.cold) != (b.hot, b.cold)
        self.ca = np.array([costs.ca_loop if k == "loop" else costs.ca_exchanger for k in self.ex_kind])
        if self.s3:
            self.unit_area = np.array([area_existing[u] for u in self.units])
            pair = {u: (plant.exchangers[u].hot, plant.exchangers[u].cold) for u in self.units}
            self.unit_cm = np.array([[costs.cm if (self.ex_hot[e], self.ex_cold[e]) != pair[u] else 0.0
                                      for u in self.units] for e in self.match_idx])

        # ---- decision vector layout -------------------------------------------------
        lo, hi, names = [], [], []
        self.col_q = {}
        for i in np.concatenate([self.existing_idx, self.new_idx]).astype(int):
            hs, cs = streams[self.ex_hot[i]], streams[self.ex_cold[i]]
            self.col_q[i] = len(lo)
            keep = cfg.require_existing_matches and self.ex_kind[i] == "existing" and not self.s3
            lo.append(cfg.min_unit_duty if keep else 0.0)
            hi.append(min(hs.load, cs.load))
            names.append(f"q:{self.ex_ids[i]}")
        self.col_phi = None
        if len(self.loop_idx) == 2:
            self.col_phi = len(lo)
            lo.append(0.0)
            hi.append(1.0)
            names.append("loop_share")
        elif len(self.loop_idx) > 2:
            raise ValueError("at most two intermediate exchangers are supported")
        self.split_cols = {}  # (stream, group index) -> list of columns (len 1: two-way split)
        for sid, chain in self.chains.items():
            for g, group in enumerate(chain):
                if len(group) < 2:
                    continue
                cols = []
                for _ in range(1 if len(group) == 2 else len(group)):
                    cols.append(len(lo))
                    lo.append(cfg.min_fraction if len(group) == 2 else 0.1)
                    hi.append(1.0 - cfg.min_fraction if len(group) == 2 else 1.0)
                    names.append(f"split:{sid}:{g}")
                self.split_cols[(sid, g)] = cols
        self.col_slack = None
        if self.active:
            self.col_slack = len(lo)
            lo.append(0.0)
            hi.append(interface.duty)
            names.append("slack")
        self.lo, self.hi, self.names = np.array(lo, float), np.array(hi, float), names
        self.dim = len(lo)

        # ---- utilities at stream ends ------------------------------------------------------
        self.utility_rows = []
        for sid, s in streams.items():
            unit = next((u for u in plant.utility_units.values() if u.stream == sid), None)
            kind = "cooler" if s.hot else "heater"
            if unit is not None:
                util = plant.utilities[unit.utility]
            else:
                options = [u for u in plant.utilities.values() if u.kind == ("cold" if s.hot else "hot") and
                           ((u.t_in + cfg.dt_min <= s.t_out) if s.hot else (u.t_in >= s.t_out + cfg.dt_min))]
                util = min(options, key=lambda u: u.price) if options else None
            if util is None:
                raise ValueError(f"{plant.id}/{sid}: no thermally feasible utility")
            home = housing.get(unit.id, unit.id) if unit else None
            if unit is not None and home != unit.id:
                moved += plant.utility_units[home].stream != sid
            self.utility_rows.append({
                "stream": sid, "kind": kind, "utility": util, "unit": unit, "housing": home,
                "U": 1.0 / (1.0 / s.h + 1.0 / util.h),
                "area_existing": area_existing.get(home, 0.0) if unit else 0.0,
                "ca": costs.ca_cooler if s.hot else costs.ca_heater})
        self.reassigned_units = moved
        self.reassign_cost = costs.cm * moved

    # ------------------------------------------------------------------------------------------
    def fractions(self, X, sid, g, size):
        cols = self.split_cols[(sid, g)]
        if size == 2:
            r = X[:, cols[0]]
            return [r, 1.0 - r]
        w = X[:, cols]
        w = w / w.sum(axis=1, keepdims=True)
        return [w[:, j] for j in range(size)]

    def evaluate(self, X, modes=None):
        """Evaluate N decision vectors.

        ``modes=None`` gives the exact discontinuous cost.  Otherwise ``modes`` fixes
        which augmentations / new utility units exist, giving a smooth cost and the
        extra constraints SLSQP needs; ``out['g']`` then holds all inequality
        constraints in ``g >= 0`` form.
        """
        X = np.atleast_2d(np.asarray(X, float))
        N = X.shape[0]
        c, cfg = self.costs, self.cfg
        q = np.zeros((N, self.n_ex))
        for i, col in self.col_q.items():
            q[:, i] = X[:, col]
        slack = X[:, self.col_slack] if self.col_slack is not None else np.zeros(N)
        if len(self.loop_idx):
            q_loop = self.interface.duty - slack
            if len(self.loop_idx) == 1:
                q[:, self.loop_idx[0]] = q_loop
            else:
                phi = X[:, self.col_phi]
                q[:, self.loop_idx[0]] = phi * q_loop
                q[:, self.loop_idx[1]] = (1.0 - phi) * q_loop
        hot_in = np.zeros((N, self.n_ex))
        hot_out = np.zeros((N, self.n_ex))
        cold_in = np.zeros((N, self.n_ex))
        cold_out = np.zeros((N, self.n_ex))
        frac_hot = np.ones((N, self.n_ex))
        frac_cold = np.ones((N, self.n_ex))
        t_end = {}
        frac_bounds = []
        for sid, chain in self.chains.items():
            hot, t_in, _, fcp, _ = self.stream_data[sid]
            sign = -1.0 if hot else 1.0
            T = np.full(N, t_in, float)
            for g, group in enumerate(chain):
                idx = [self.index[e] for e in group]
                fr = [np.ones(N)] if len(idx) == 1 else self.fractions(X, sid, g, len(idx))
                if len(idx) > 2:
                    frac_bounds += [f - cfg.min_fraction for f in fr]
                for i, f in zip(idx, fr):
                    out = T + sign * q[:, i] / (f * fcp)
                    if hot:
                        hot_in[:, i], hot_out[:, i], frac_hot[:, i] = T, out, f
                    else:
                        cold_in[:, i], cold_out[:, i], frac_cold[:, i] = T, out, f
                T = T + sign * q[:, idx].sum(axis=1) / fcp
            t_end[sid] = T
        dt1 = hot_in - cold_out
        dt2 = hot_out - cold_in
        smooth = modes is not None
        approach1, approach2 = dt1 - cfg.dt_min, dt2 - cfg.dt_min
        if self.s3:
            # A dropped match (zero duty) is not a unit: no approach constraint.
            M = self.match_idx
            if smooth:
                off = np.array([modes["use"].get(self.ex_ids[i]) is False for i in M])[None, :]
            else:
                off = q[:, M] <= cfg.duty_tolerance
            approach1[:, M] = np.where(off, 0.0, approach1[:, M])
            approach2[:, M] = np.where(off, 0.0, approach2[:, M])
        g_list = [approach1, approach2] + [f[:, None] for f in frac_bounds]
        area = q / (self.U * chen(np.maximum(dt1, 1e-3), np.maximum(dt2, 1e-3)))
        cap_aug = np.zeros(N)
        cap_loop = np.zeros(N)
        cap_util = np.zeros(N)
        cap_bypass = np.zeros(N)
        beta, lx = c.beta, c.min_augmentation
        tol_q = cfg.duty_tolerance

        def min_duty(duty, key, scale):
            # Paper bound: an active unit carries at least ``min_unit_duty``.
            if cfg.min_unit_duty <= 0:
                return
            if not smooth:
                g_list.append(np.where(duty > tol_q, (duty - cfg.min_unit_duty) / scale, 0.0)[:, None])
            elif modes["use"].get(key) is True:
                g_list.append(((duty - cfg.min_unit_duty) / scale)[:, None])
            elif modes["use"].get(key) is False:
                g_list.append((-duty / scale)[:, None])

        cap_new = np.zeros(N)
        cap_cm = np.zeros(N)
        self._last_map = None
        if self.s3:
            cap_aug, cap_new, cap_cm, cap_b = self._process_units(q, area, modes, g_list, min_duty)
            cap_bypass += cap_b
        for i in ([] if self.s3 else self.existing_idx):
            x = area[:, i] - self.area_existing[i]
            if not smooth:
                cap_aug += np.where(x > self.free_area, c.cf_unit + self.ca[i] * np.maximum(x, lx) ** beta, 0.0)
                cap_bypass += np.where((q[:, i] > tol_q) & (x < -self.free_area), cfg.bypass_cost, 0.0)
            elif modes["aug"][self.ex_ids[i]]:
                cap_aug += c.cf_unit + self.ca[i] * _softmax1(x, lx) ** beta
            else:
                g_list.append(((self.area_existing[i] + self.free_area - area[:, i])
                               / max(self.area_existing[i], 1.0))[:, None])
            min_duty(q[:, i], self.ex_ids[i], 10.0)
        if len(self.loop_idx):
            cap_loop = (c.cf_loop + self.ca[self.loop_idx] * np.maximum(area[:, self.loop_idx], 0.0) ** beta).sum(axis=1)
            g_list.append((q[:, self.loop_idx] - cfg.min_loop_duty) / self.interface.flow)
        util_hot = np.zeros(N)
        util_cold = np.zeros(N)
        utility_out = []
        for row in self.utility_rows:
            hot, t_in, t_out, fcp, _ = self.stream_data[row["stream"]]
            u = row["utility"]
            T = t_end[row["stream"]]
            duty = fcp * ((T - t_out) if hot else (t_out - T))
            g_list.append((duty / fcp)[:, None])  # no stream may overshoot its target
            if hot:
                d1, d2 = T - u.t_out, t_out - u.t_in
            else:
                d1, d2 = u.t_in - t_out, u.t_out - T
            d1 = d1 if isinstance(d1, np.ndarray) else np.full(N, float(d1))
            d2 = d2 if isinstance(d2, np.ndarray) else np.full(N, float(d2))
            on = duty > cfg.duty_tolerance
            g_list.append(np.where(on, d1 - cfg.dt_min, 0.0)[:, None])
            g_list.append(np.where(on, d2 - cfg.dt_min, 0.0)[:, None])
            # Exact costing treats a residual within ``duty_tolerance`` as numerical zero:
            # no utility, no unit (the stream then ends within 1e-3 kW of its target).
            pos = np.maximum(duty, 0.0) if smooth else np.where(on, duty, 0.0)
            a_req = pos / (row["U"] * chen(np.maximum(d1, 1e-3), np.maximum(d2, 1e-3)))
            if hot:
                util_cold += pos * u.price
            else:
                util_hot += pos * u.price
            key = row["unit"].id if row["unit"] else f"new:{row['stream']}"
            if row["unit"] is not None:
                x = a_req - row["area_existing"]
                if not smooth:
                    cap_util += np.where(x > self.free_area, c.cf_unit + row["ca"] * np.maximum(x, lx) ** beta, 0.0)
                    cap_bypass += np.where(on & (x < -self.free_area), cfg.bypass_cost, 0.0)
                elif modes["aug"][key]:
                    cap_util += c.cf_unit + row["ca"] * _softmax1(x, lx) ** beta
                else:
                    g_list.append(((row["area_existing"] + self.free_area - a_req)
                                   / max(row["area_existing"], 1.0))[:, None])
            else:
                if not smooth:
                    cap_util += np.where(on, c.cf_unit + row["ca"] * a_req ** beta, 0.0)
                    if not cfg.allow_new_utility_units:
                        g_list.append((-duty / fcp)[:, None])
                elif modes["aug"][key] and cfg.allow_new_utility_units:
                    cap_util += c.cf_unit + row["ca"] * a_req ** beta
                else:
                    g_list.append((-duty / fcp)[:, None])
            min_duty(duty, key, fcp)
            utility_out.append({"key": key, "duty": duty, "area": a_req, "T_before": T, "dt": (d1, d2)})
        g = np.concatenate([np.atleast_2d(x).reshape(N, -1) for x in g_list], axis=1)
        violation = np.maximum(-g, 0.0).sum(axis=1)
        capital = cap_aug + cap_loop + cap_util + cap_bypass + self.reassign_cost + cap_new + cap_cm
        penalty = cfg.slack_price * slack
        tac = util_hot + util_cold + c.af * capital + penalty
        return {"tac": tac, "violation": violation, "g": g, "q": q, "area": area,
                "hot_in": hot_in, "hot_out": hot_out, "cold_in": cold_in, "cold_out": cold_out,
                "frac_hot": frac_hot, "frac_cold": frac_cold, "t_end": t_end, "slack": slack,
                "util_hot": util_hot, "util_cold": util_cold, "cap_aug": cap_aug, "cap_loop": cap_loop,
                "cap_util": cap_util, "cap_bypass": cap_bypass, "cap_new": cap_new, "cap_cm": cap_cm,
                "unit_map": self._last_map, "capital": capital, "penalty": penalty, "utilities": utility_out}

    def _process_units(self, q, area, modes, g_list, min_duty):
        """Strategy III capital of process matches, minimised over unit assignments.

        Exact mode: for every sample, the cheapest injective map of existing units to
        active matches (each unit houses exactly one match; other active matches buy a
        unit).  Smooth mode: the map in ``modes['map']`` with smooth costs.
        """
        c, cfg = self.costs, self.cfg
        N, beta, lx, tol_q = q.shape[0], c.beta, c.min_augmentation, cfg.duty_tolerance
        M = self.match_idx
        A = area[:, M]  # (N, matches)
        Q = q[:, M]
        if modes is None:
            active = Q > tol_q
            if cfg.min_unit_duty > 0:
                g_list.append(np.where(active, (Q - cfg.min_unit_duty) / 10.0, 0.0))
            x = A[:, :, None] - self.unit_area[None, None, :]  # (N, matches, units)
            aug = np.where(x > self.free_area, c.cf_unit + c.ca_exchanger * np.maximum(x, lx) ** beta, 0.0)
            byp = np.where(active[:, :, None] & (x < -self.free_area), cfg.bypass_cost, 0.0)
            housed = aug + byp + self.unit_cm[None, :, :]
            housed = np.where(active[:, :, None], housed, np.inf)  # a unit cannot be idle
            new = np.where(active, c.cf_unit + c.ca_exchanger * np.maximum(A, 0.0) ** beta, 0.0)
            totals, parts = [], []
            for mp in self.unit_maps:
                used = np.zeros(len(M), bool)
                h = np.zeros(N)
                a_part = np.zeros(N)
                cm_part = np.zeros(N)
                b_part = np.zeros(N)
                for u, j in enumerate(mp):
                    used[j] = True
                    h = h + housed[:, j, u]
                    a_part += aug[:, j, u]
                    cm_part += self.unit_cm[j, u]
                    b_part += byp[:, j, u]
                n_part = new[:, ~used].sum(axis=1)
                totals.append(h + n_part)
                parts.append((a_part, n_part, cm_part, b_part))
            totals = np.array(totals)  # (maps, N)
            if not len(totals):
                inf = np.full(N, np.inf)
                return inf, np.zeros(N), np.zeros(N), np.zeros(N)
            best = np.argmin(totals, axis=0)
            feasible = np.isfinite(totals[best, np.arange(N)])
            # Infeasible samples (an existing unit would be idle) get a large penalty
            # through the violation measure rather than an infinite cost.
            g_list.append(np.where(feasible, 0.0, -1.0)[:, None])
            cols = np.arange(N)
            stacked = [np.array([pt[k] for pt in parts]) for k in range(4)]  # (maps, N) each
            pick = lambda k: np.where(feasible, stacked[k][best, cols], 0.0)
            self._last_map = best
            return pick(0), pick(1), pick(2), pick(3)
        mp = modes["map"]
        cap_aug, cap_new, cap_cm = np.zeros(N), np.zeros(N), np.zeros(N)
        housed = set()
        for u, j in enumerate(mp):
            housed.add(j)
            i = M[j]
            key = self.ex_ids[i]
            g_list.append(((Q[:, j] - max(cfg.min_unit_duty, tol_q)) / 10.0)[:, None])  # housed => active
            x = A[:, j] - self.unit_area[u]
            if modes["aug"].get(key, False):
                cap_aug += c.cf_unit + c.ca_exchanger * _softmax1(x, lx) ** beta
            else:
                g_list.append(((self.unit_area[u] + self.free_area - A[:, j]) / max(self.unit_area[u], 1.0))[:, None])
            cap_cm += self.unit_cm[j, u]
        for j, i in enumerate(M):
            if j in housed:
                continue
            key = self.ex_ids[i]
            use = modes["use"].get(key)
            if use is False:
                g_list.append((-Q[:, j] / 10.0)[:, None])
                continue
            if use is True:
                g_list.append(((Q[:, j] - cfg.min_unit_duty) / 10.0)[:, None])
            cap_new += c.cf_unit * (1.0 if use else 0.0) + c.ca_exchanger * (np.maximum(A[:, j], 0.0) + 1e-6) ** beta
        return cap_aug, cap_new, cap_cm, np.zeros(N)

    # ------------------------------------------------------------------------------------------
    def modes_at(self, ev, n=0, threshold=None, use_threshold=None):
        """Fixed-cost and on/off decisions active at sample ``n`` of an exact evaluation.

        An augmentation counts as active when the excess area exceeds ``threshold``
        (default: the uncharged excess area); a unit counts as used when its duty
        exceeds ``use_threshold`` (default: the duty tolerance).
        """
        tol = self.free_area if threshold is None else threshold
        use_tol = self.cfg.duty_tolerance if use_threshold is None else use_threshold
        modes = {"aug": {}, "use": {}}
        if self.s3:
            best = ev.get("unit_map")
            mp = self.unit_maps[int(best[n])] if best is not None and len(self.unit_maps) else ()
            modes["map"] = mp
            for u, j in enumerate(mp):
                i = self.match_idx[j]
                modes["aug"][self.ex_ids[i]] = bool(ev["area"][n, i] - self.unit_area[u] > tol)
            for i in self.match_idx:
                modes["use"][self.ex_ids[i]] = bool(ev["q"][n, i] > use_tol)
        for i in ([] if self.s3 else self.existing_idx):
            modes["aug"][self.ex_ids[i]] = bool(ev["area"][n, i] - self.area_existing[i] > tol)
            modes["use"][self.ex_ids[i]] = bool(ev["q"][n, i] > use_tol)
        for row, u in zip(self.utility_rows, ev["utilities"]):
            if row["unit"] is not None:
                modes["aug"][u["key"]] = bool(u["area"][n] - row["area_existing"] > tol)
            else:
                modes["aug"][u["key"]] = bool(u["duty"][n] > self.cfg.duty_tolerance)
            modes["use"][u["key"]] = bool(u["duty"][n] > use_tol)
        return modes

    def baseline_point(self):
        """Original duties and duty-proportional (isothermal) split fractions."""
        x = 0.5 * (self.lo + self.hi)
        for i, col in self.col_q.items():
            # A new Strategy-III match starts switched off (its base structure).
            x[col] = self.plant.exchangers[self.ex_ids[i]].duty if self.ex_kind[i] == "existing" else 0.0
        for (sid, g), cols in self.split_cols.items():
            members = self.chains[sid][g]
            duties = np.array([self.plant.exchangers[m].duty if m in self.plant.exchangers else self.interface.duty / 2
                               for m in members])
            w = duties / duties.sum()
            if len(cols) == 1:
                x[cols[0]] = w[0]
            else:
                x[cols] = np.clip(w, 0.1, 1.0)
        if self.col_slack is not None:
            x[self.col_slack] = 0.0
        return np.clip(x, self.lo, self.hi)

    def samples(self, n, seed):
        """Scrambled Sobol samples plus the original operating point.

        Three quarters of the samples have zero slack (the region of interest).  In a
        share of them (``local_sample_fraction``) the existing units' duties are drawn
        near their original duties, the rest of the vector over its full range.
        """
        if self.dim == 0:
            return np.zeros((1, 0))
        m = int(math.ceil(math.log2(max(n, 2))))
        u = qmc.Sobol(self.dim, scramble=True, seed=seed).random_base2(m)[:n]
        X = self.lo + u * (self.hi - self.lo)
        k = int(round(self.cfg.local_sample_fraction * n))
        lo_f, hi_f = self.cfg.local_duty_range
        for i, col in self.col_q.items():
            if self.ex_kind[i] != "existing":
                continue  # new matches are sampled over their full range
            q0 = self.plant.exchangers[self.ex_ids[i]].duty
            X[:k, col] = np.clip(q0 * (lo_f + (hi_f - lo_f) * u[:k, col]), self.lo[col], self.hi[col])
        if self.col_slack is not None:
            X[: (3 * n) // 4, self.col_slack] = 0.0
        return np.vstack([self.baseline_point(), X])

    def polish(self, x0, maxiter):
        """Local refinement of ``x0``; returns every refined point for exact comparison.

        1. SLSQP with the fixed-cost modes found at ``x0``.
        2. SLSQP with every augmentation / new utility unit switched on.  Its fixed
           costs are constants there, so this is the relaxation in which area may
           grow wherever useful.
        3. A final pass keeping only the augmentations of 2 that need at least the
           minimum augmentation area; a smaller excess is cheaper to back off than
           to buy (fixed cost plus the 1 m2 minimum).
        """
        if self.dim == 0:
            return [x0]
        start = self.modes_at(self.evaluate(x0[None, :]))
        x1 = self._slsqp(x0, start, maxiter) if self.cfg.polish_start_pass else x0
        # Relaxation: every augmentation allowed, no on/off decision imposed.
        x2 = self._slsqp(x0, {"aug": {k: True for k in start["aug"]}, "use": {},
                              **({"map": start["map"]} if "map" in start else {})}, maxiter)
        use = self.cfg.min_unit_duty / 2 if self.cfg.min_unit_duty > 0 else None
        x3 = self._slsqp(x2, self.modes_at(self.evaluate(x2[None, :]), threshold=self.costs.min_augmentation,
                                           use_threshold=use), maxiter)
        return [x1, x2, x3]

    def _slsqp(self, x0, modes, maxiter, step=1e-7):
        """SLSQP with forward-difference derivatives from one batched evaluation."""
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        scale = 1e-5
        cache = {}

        def ev(z):
            key = z.tobytes()
            if key not in cache:
                cache.clear()
                Z = np.repeat(z[None, :], self.dim + 1, axis=0)
                # Step away from the upper bound so all points stay in the box.
                h = np.where(z + step <= 1.0, step, -step)
                Z[1:] += np.diag(h)
                out = self.evaluate(self.lo + Z * span, modes)
                f, g = out["tac"] * scale, out["g"]
                cache[key] = (f[0], (f[1:] - f[0]) / h, g[0], ((g[1:] - g[0]) / h[:, None]).T)
            return cache[key]

        res = minimize(lambda z: ev(z)[0], (x0 - self.lo) / span, jac=lambda z: ev(z)[1],
                       method="SLSQP", bounds=[(0.0, 1.0)] * self.dim,
                       constraints=[{"type": "ineq", "fun": lambda z: ev(z)[2], "jac": lambda z: ev(z)[3]}],
                       options={"maxiter": maxiter, "ftol": self.cfg.polish_ftol})
        return self.lo + np.clip(res.x, 0.0, 1.0) * span

    def export(self, x):
        """Physical description of one design, independent of the evaluator's arrays."""
        ev = self.evaluate(x[None, :])
        cfg, c = self.cfg, self.costs
        exchangers = []
        house = {}  # Strategy III: match index -> housing unit position
        if self.s3 and ev.get("unit_map") is not None and len(self.unit_maps):
            for u, j in enumerate(self.unit_maps[int(ev["unit_map"][0])]):
                house[int(self.match_idx[j])] = u
        for i, eid in enumerate(self.ex_ids):
            ex = {"id": eid, "kind": self.ex_kind[i], "hot": self.ex_hot[i], "cold": self.ex_cold[i],
                  "duty": float(ev["q"][0, i]), "hot_in": float(ev["hot_in"][0, i]),
                  "hot_out": float(ev["hot_out"][0, i]), "cold_in": float(ev["cold_in"][0, i]),
                  "cold_out": float(ev["cold_out"][0, i]), "hot_fraction": float(ev["frac_hot"][0, i]),
                  "cold_fraction": float(ev["frac_cold"][0, i]), "U": float(self.U[i]),
                  "area_required": float(ev["area"][0, i]), "area_existing": float(self.area_existing[i])}
            if self.s3 and ex["kind"] in ("existing", "new"):
                active = ex["duty"] > cfg.duty_tolerance
                u = house.get(i)
                ex["housing_unit"] = self.units[u] if u is not None else ("new" if active else None)
                ex["area_existing"] = float(self.unit_area[u]) if u is not None else 0.0
                ex["reassignment_capital"] = float(self.unit_cm[list(self.match_idx).index(i), u]) if u is not None else 0.0
                if u is not None:
                    aug = ex["area_required"] - ex["area_existing"]
                    ex["augmented"] = bool(aug > self.free_area)
                    ex["bypass"] = bool(active and aug < -self.free_area and cfg.bypass_cost > 0)
                    ex["augmentation_area"] = max(aug, c.min_augmentation) if ex["augmented"] else 0.0
                    ex["capital"] = (c.cf_unit + self.ca[i] * ex["augmentation_area"] ** c.beta) if ex["augmented"] else 0.0
                else:
                    ex["augmented"], ex["augmentation_area"], ex["bypass"] = False, 0.0, False
                    ex["capital"] = (c.cf_unit + self.ca[i] * ex["area_required"] ** c.beta) if active else 0.0
            elif ex["kind"] == "existing":
                ex["housing_unit"] = self.housing[i]
                aug = ex["area_required"] - ex["area_existing"]
                ex["augmented"] = bool(aug > self.free_area)
                ex["bypass"] = bool(ex["duty"] > cfg.duty_tolerance and aug < -self.free_area and cfg.bypass_cost > 0)
                ex["augmentation_area"] = max(aug, c.min_augmentation) if ex["augmented"] else 0.0
                ex["capital"] = (c.cf_unit + self.ca[i] * ex["augmentation_area"] ** c.beta) if ex["augmented"] else 0.0
            else:
                ex["augmented"], ex["augmentation_area"], ex["bypass"] = False, 0.0, False
                ex["capital"] = c.cf_loop + self.ca[i] * ex["area_required"] ** c.beta
            exchangers.append(ex)
        utilities = []
        for row, u in zip(self.utility_rows, ev["utilities"]):
            # Exported exactly as costed: a residual within ``duty_tolerance`` is zero.
            duty = float(u["duty"][0])
            on = duty > cfg.duty_tolerance
            duty = duty if on else 0.0
            entry = {"stream": row["stream"], "kind": row["kind"], "utility": row["utility"].id,
                     "unit": row["unit"].id if row["unit"] else None, "housing_unit": row["housing"],
                     "duty": duty,
                     "T_in": float(u["T_before"][0]), "T_out": self.stream_data[row["stream"]][2],
                     "U": row["U"], "area_required": float(u["area"][0]) if on else 0.0,
                     "area_existing": row["area_existing"], "annual_cost": duty * row["utility"].price}
            if row["unit"] is not None:
                aug = entry["area_required"] - row["area_existing"]
                entry["augmented"] = bool(aug > self.free_area)
                entry["bypass"] = bool(on and aug < -self.free_area and cfg.bypass_cost > 0)
                entry["augmentation_area"] = max(aug, c.min_augmentation) if entry["augmented"] else 0.0
                entry["capital"] = (c.cf_unit + row["ca"] * entry["augmentation_area"] ** c.beta) if entry["augmented"] else 0.0
            else:
                entry["augmented"], entry["augmentation_area"], entry["bypass"] = False, 0.0, False
                entry["capital"] = (c.cf_unit + row["ca"] * entry["area_required"] ** c.beta) if on else 0.0
            utilities.append(entry)
        costs = {"hot_utility": float(ev["util_hot"][0]), "cold_utility": float(ev["util_cold"][0]),
                 "augmentation_capital": float(ev["cap_aug"][0]) + sum(
                     u["capital"] for u in utilities if u["unit"] is not None),
                 "loop_exchanger_capital": float(ev["cap_loop"][0]),
                 "new_utility_capital": sum(u["capital"] for u in utilities if u["unit"] is None),
                 "bypass_capital": float(ev["cap_bypass"][0]),
                 "reassignment_capital": self.reassign_cost + float(ev["cap_cm"][0]),
                 "new_process_capital": float(ev["cap_new"][0]),
                 "capital": float(ev["capital"][0]), "annualized_capital": c.af * float(ev["capital"][0]),
                 "slack_penalty": float(ev["penalty"][0]), "TAC": float(ev["tac"][0])}
        slack = float(ev["slack"][0])
        return {"plant": self.plant.id, "structure": {
                    "label": self.structure.label(),
                    "placements": [vars(p) for p in self.structure.placements],
                    "new_matches": [vars(n) for n in self.structure.new_matches],
                    "assignment": [list(a) for a in self.structure.assignment],
                    "loop_arrangement": self.structure.loop_arrangement},
                "interface": {"role": self.interface.role, "flow": self.interface.flow,
                              "T_in": self.interface.t_in, "T_out": self.interface.t_out,
                              "duty": self.interface.duty},
                "chains": self.chains, "decision_vector": dict(zip(self.names, map(float, x))),
                "slack_kW": slack, "max_violation": float(ev["violation"][0]),
                "feasible": bool(ev["violation"][0] <= cfg.feasibility_tolerance),
                "strictly_feasible": bool(ev["violation"][0] <= cfg.feasibility_tolerance and slack <= 1e-6),
                "exchangers": exchangers, "utilities": utilities,
                "stream_end_temperatures": {k: float(v[0]) for k, v in ev["t_end"].items()},
                "costs": costs, "TAC": costs["TAC"],
                "rules": {"dt_min": cfg.dt_min, "min_fraction": cfg.min_fraction, "free_area": self.free_area,
                          "strategy": cfg.strategy,
                          "min_unit_duty": cfg.min_unit_duty, "min_loop_duty": cfg.min_loop_duty,
                          "bypass_cost": cfg.bypass_cost, "slack_price": cfg.slack_price,
                          "duty_tolerance": cfg.duty_tolerance, "area_tolerance": cfg.area_tolerance}}


def _softmax1(x, floor):
    """Smooth max(x, floor) for the augmentation-area lower bound."""
    d = x - floor
    return floor + 0.5 * (d + np.sqrt(d * d + 1e-4))


class PlantSolver:
    """Independent local retrofit optimizer of one plant."""

    def __init__(self, plant: Plant, costs: Costs, cfg: PlantSolverConfig | None = None):
        self.plant, self.costs = plant, costs
        self.cfg = cfg or PlantSolverConfig()
        self.area_existing = self._reconcile_areas()

    def _reconcile_areas(self):
        """Reconcile printed Fig. 2 areas with the original duties.

        Fig. 2 prints areas to 0.1 m2.  When the area the original duties require
        lies within half a printed digit of the printed area, it is taken as the
        installed area, so the original network operates exactly (no augmentation,
        no throttling bypass).  A unit printed larger than that is kept oversized; a
        unit printed smaller is a data error.
        """
        areas = {e.id: e.area for e in self.plant.exchangers.values()}
        areas.update({u.id: u.area for u in self.plant.utility_units.values()})
        net = Network(self.plant, self.costs, self.cfg, Structure(), Interface(), areas)
        ev = net.evaluate(net.baseline_point()[None, :])
        if ev["violation"][0] > self.cfg.feasibility_tolerance:
            raise ValueError(f"{self.plant.id}: original network violates the approach temperature")
        self.reconciliation = {}
        for i, eid in enumerate(net.ex_ids):
            self._reconcile(areas, eid, float(ev["area"][0, i]))
        for row, u in zip(net.utility_rows, ev["utilities"]):
            if row["unit"] is not None:
                self._reconcile(areas, row["unit"].id, float(u["area"][0]))
        return areas

    def _reconcile(self, areas, uid, required):
        printed = areas[uid]
        if required > printed + self.cfg.area_tolerance:
            raise ValueError(f"{self.plant.id}/{uid}: original duty needs {required:.3f} m2 > printed {printed}")
        areas[uid] = required if abs(required - printed) <= self.cfg.area_tolerance else printed
        self.reconciliation[uid] = {"printed": printed, "required_by_original_duty": required,
                                    "used": areas[uid]}

    def structures(self, interface: Interface):
        """Every structure of the strategy: loop placements x unit assignments."""
        return [replace(st, assignment=a) for a in self.assignments() for st in self.loop_structures(interface)]

    def new_match_variants(self, base: Structure, interface: Interface):
        """Strategy III: ``base`` plus one new process match at every pair of series gaps."""
        if self.cfg.strategy < 3 or self.cfg.max_new_matches < 1:
            return []
        net = self.network(base, interface)
        hots = [sid for sid, st in self.plant.streams.items() if st.hot]
        colds = [sid for sid, st in self.plant.streams.items() if not st.hot]
        out = []
        for h in hots:
            for cname in colds:
                for gh in range(len(net.chains[h]) + 1):
                    for gc in range(len(net.chains[cname]) + 1):
                        out.append(replace(base, new_matches=(NewMatch(h, cname, gh, gc),)))
        return out

    def loop_structures(self, interface: Interface):
        """Placements of the intermediate-fluid exchangers (units in their own matches)."""
        if not interface.active:
            return [Structure()]
        hot_loop = interface.role == "sink"
        compatible = [sid for sid, s in self.plant.streams.items() if s.hot != hot_loop]
        options = {}
        for sid in compatible:
            n = len(self.plant.chains[sid])
            options[sid] = ([Placement(sid, "series", g) for g in range(n + 1)] +
                            [Placement(sid, "parallel", g) for g in range(n)])
        result = [Structure((p,)) for sid in compatible for p in options[sid]]
        if self.cfg.max_loop_exchangers >= 2 and interface.duty >= 2 * self.cfg.min_loop_duty:
            for a, b in itertools.combinations(compatible, 2):
                for pa, pb in itertools.product(options[a], options[b]):
                    result += [Structure((pa, pb), "series"), Structure((pb, pa), "series"),
                               Structure((pa, pb), "parallel")]
        return result

    def assignments(self):
        """Unit-to-match assignments of the strategy, identity first.

        Strategy II: each existing unit houses exactly one existing match of the same
        type (process exchanger, or a heater/cooler with the same utility).
        """
        if self.cfg.strategy < 2:
            return [()]
        groups = [[e.id for e in self.plant.exchangers.values()]]
        by_type = {}
        for u in self.plant.utility_units.values():
            by_type.setdefault((u.kind, u.utility), []).append(u.id)
        groups += list(by_type.values())
        per_group = [[tuple((m, h) for m, h in zip(g, perm) if m != h) for perm in itertools.permutations(g)]
                     for g in groups]
        combos = [tuple(sorted(sum(c, ()))) for c in itertools.product(*per_group)]
        return sorted(set(combos), key=lambda a: (len(a), a))

    def network(self, structure: Structure, interface: Interface) -> Network:
        return Network(self.plant, self.costs, self.cfg, structure, interface, self.area_existing)

    def solve(self, interface: Interface) -> dict:
        """Best retrofit found for this interface (deterministic)."""
        started = time.perf_counter()
        cfg = self.cfg
        def screen(k, st, variant=False):
            net = self.network(st, interface)
            X = net.samples(cfg.samples, seed=cfg.seed + 7919 * k)
            ev = net.evaluate(X)
            score = ev["tac"] + 1e6 * ev["violation"]
            if variant:  # Strategy II/III variants: refine their best sample only
                order = np.argsort(score)[:1]
            else:
                order = np.argsort(score)[: cfg.polish_per_structure]
                if 0 not in order:  # always refine from the original operating point too
                    order = np.append(order, 0)
            return (float(score[order[0]]), k, net, X[order])

        loops = self.loop_structures(interface)
        candidates = [screen(k, st) for k, st in enumerate(loops)]
        # Strategy III: new-match variants of the best-ranked loop structures.
        new_cands = []
        if cfg.strategy >= 3:
            top = [loops[c[1]] for c in sorted(candidates, key=lambda c: c[0])][: cfg.new_match_top_structures]
            variants = [v for st in top for v in self.new_match_variants(st, interface)]
            new_cands = sorted((screen(5000 + k, v, True) for k, v in enumerate(variants)), key=lambda c: c[0])
        # Strategy II: a reassignment changes only installed areas and the CM charge, so
        # it is screened on the structures that rank best with units in their own matches.
        ranked = [loops[c[1]] for c in sorted(candidates, key=lambda c: c[0])][: cfg.assignment_top_structures + 2]
        for j, a in enumerate(self.assignments()[1:], 1):
            candidates += [screen(1000 * j + k, replace(st, assignment=a), True) for k, st in enumerate(ranked)]
        # Every structure with at most one intermediate exchanger is refined (few,
        # cheap and often optimal).  Two-exchanger structures are all refined when
        # there are at most ``exhaustive_doubles`` of them, otherwise they are ranked
        # by their best sample and only the leading ``top_structures`` are refined.
        candidates.sort(key=lambda c: c[0])
        own = [c for c in candidates if not c[2].structure.assignment and not c[2].structure.new_matches]
        singles = [c for c in own if len(c[2].structure.placements) <= 1]
        doubles = [c for c in own if len(c[2].structure.placements) > 1]
        chosen = singles + (doubles if len(doubles) <= cfg.exhaustive_doubles else doubles[: cfg.top_structures])
        # Strategy II: every other unit assignment refines its best-ranked structures.
        moved = {}
        for c in candidates:
            if c[2].structure.assignment:
                moved.setdefault(c[2].structure.assignment, []).append(c)
        for group in moved.values():
            chosen += group[: cfg.assignment_top_structures]
        chosen += new_cands[: cfg.new_match_polish]
        candidates += new_cands
        best, polished = None, 0
        for _, k, net, starts in chosen:
            for x0 in starts:
                trials = [x0]
                if net.dim:
                    trials += net.polish(x0, cfg.polish_maxiter)
                    polished += 1
                for x in trials:
                    ev = net.evaluate(x[None, :])
                    feasible = ev["violation"][0] <= cfg.feasibility_tolerance
                    key = (not feasible, float(ev["tac"][0]) if feasible else float(ev["violation"][0]))
                    if best is None or key < best[0]:
                        best = (key, net, x)
        result = best[1].export(best[2])
        if not result["feasible"]:
            # No design meets every constraint: charge the whole interface duty as slack so
            # an infeasible interface can never earn a reward from a violating design.
            penalty = cfg.slack_price * interface.duty
            result["costs"]["violation_penalty"] = penalty
            result["TAC"] = result["costs"]["TAC"] = result["costs"]["TAC"] + penalty
        result["solver"] = {"structures_enumerated": len(candidates), "structures_polished": len(chosen),
                            "polish_starts": polished, "wall_s": time.perf_counter() - started}
        return result
