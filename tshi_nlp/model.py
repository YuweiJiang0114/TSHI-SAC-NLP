"""Site NLP: the paper's stage-wise equations for every plant plus the loop closure.

Given per plant the active units (existing units at their stages with definite areas,
intermediate-fluid exchangers with definite areas) and the loop flows, the model solves
for duties, branch temperatures, split fractions, stage/end bypass fractions, utility
duties and the loop temperatures, minimizing utility cost.  Constraints that the given
areas may make impossible (approach temperatures, stream targets, a stream without a
utility reaching its target) are elastic: their slacks are priced in the objective and
reported as violations.

Variables of unit e between hot stream h and cold stream c at stage k:
    q[e]            duty
    tb[h,k,e]       hot branch outlet, tb[c,k,e] cold branch outlet          (A17-A22)
    f[h,k,e], f[c,k,e]  flow fractions, with the stage bypass r[s,k]         (A25-A28)
    T[s,k]          stage boundary temperatures, mixing                      (A31-A34)
    th[e]           Chen mean: th^3 = d1 d2 (d1 + d2)/2, q = U A th          (A76)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tshi_decomp.case import Costs, Plant

LOOP = "LOOP"


@dataclass(frozen=True)
class NLPConfig:
    dt_min: float = 1.0
    min_fraction: float = 0.1  # LRH/LRC...
    min_duty: float = 30.0  # LQ: utility units are off or at least this (smooth penalty)
    max_bypass: float = 0.9
    # "definite": the agent's area is used exactly (A76 as an equality);
    # "upper": the agent's area is bought and paid for, the NLP may use any part of it
    # (A_used <= A_bought), so an oversized unit wastes capital instead of violating dt_min.
    area_mode: str = "definite"
    area_tolerance: float = 0.05  # m2: half a printed digit of the Fig. 2 areas (Eq. 10-11)
    penalty_k: float = 50.0  # objective weight per K / kW of elastic slack (final value)
    penalty_schedule: tuple = (2.0, 50.0)  # continuation: solve with each weight in turn
    bypass_weight: float = 0.01  # small cost on bypass fractions (ties)
    ipopt_tol: float = 1e-8
    ipopt_max_iter: int = 600  # a design that needs more is a poor one; its penalties tell
    ipopt_max_cpu: float = 15.0  # seconds per IPOPT call
    ipopt_print: bool = False
    # Starting point of the free solve: split/bypass settings sampled with the rating
    # simulator at the loop temperatures of the fixed-fraction solve (0 = equal splits).
    sim_samples: int = 32


@dataclass
class UnitSpec:
    id: str
    kind: str  # "existing" | "loop"
    hot: str
    cold: str
    stage: int  # 0-based
    area: float  # heat-transfer area bought (installed + augmentation for an existing unit)
    U: float
    unit: str | None = None  # existing unit id
    aug: float = 0.0  # augmentation part of ``area`` for an existing unit


@dataclass
class PlantSpec:
    plant: Plant
    role: str  # "sink" | "source"
    flow: float  # intermediate-fluid branch flow (kW/K)
    units: list  # UnitSpec


def build_site_model(specs: dict, costs: Costs, K: int, cfg: NLPConfig):
    """Pyomo model of the site: ``specs`` maps plant id -> PlantSpec."""
    import pyomo.environ as pe

    m = pe.ConcreteModel()
    m.cfg = cfg
    dt, lb = cfg.dt_min, cfg.min_fraction
    scale = 1e-3  # objective in kUSD
    m.pk = pe.Param(initialize=cfg.penalty_k, mutable=True)  # elastic-slack weight (continuation)
    m.T_r = pe.Var(bounds=(0.0, 500.0))
    m.T_s = pe.Var(bounds=(0.0, 500.0))
    m.slack = pe.VarList(domain=pe.NonNegativeReals)
    m.obj_terms = []
    m.eqs = pe.ConstraintList()
    m.info = {}
    sources = [p for p, s in specs.items() if s.role == "source"]
    sinks = [p for p, s in specs.items() if s.role == "sink"]
    total_src = sum(specs[p].flow for p in sources)
    # Supply header: flow-weighted mix of the source outlets (added after plant blocks).
    for pid, sp in specs.items():
        b = pe.Block()
        setattr(m, f"plant_{pid}", b)
        plant = sp.plant
        streams = {s.id: (s.hot, s.t_in, s.t_out, s.fcp, s.h) for s in plant.streams.values()}
        loop_hot = sp.role == "sink"
        streams[LOOP] = (loop_hot, None, None, sp.flow, costs.loop_h)
        sids = list(streams)
        b.sids = sids
        units = {u.id: u for u in sp.units}
        b.units = units
        members = {(s, k): [u.id for u in sp.units if u.stage == k and (u.hot == s or u.cold == s)]
                   for s in sids for k in range(K)}
        b.members = members
        # stage boundary temperatures T[s, k], k = 0..K  (hot: k=0 inlet; cold: k=K inlet)
        b.T = pe.Var(sids, range(K + 1), bounds=(0.0, 600.0))
        b.q = pe.Var(list(units), domain=pe.NonNegativeReals)
        b.tb = pe.Var([(s, k, e) for (s, k), es in members.items() for e in es], bounds=(0.0, 600.0))
        b.f = pe.Var([(s, k, e) for (s, k), es in members.items() for e in es], bounds=(0.0, 1.0))
        b.r = pe.Var([(s, k) for (s, k), es in members.items() if es and s != LOOP], bounds=(0.0, cfg.max_bypass))
        b.th = pe.Var(list(units), bounds=(1e-3, 600.0))
        b.A_used = pe.Var(list(units), bounds=(0.0, None))
        b.eqs = pe.ConstraintList()
        for s in sids:
            hot, t_in, t_out, F, h = streams[s]
            if s == LOOP:
                if loop_hot:
                    b.eqs.add(b.T[s, 0] == m.T_s)
                    b.eqs.add(b.T[s, K] == m.T_r)  # A52: sink returns the fluid at T_r
                else:
                    b.eqs.add(b.T[s, K] == m.T_r)  # A53: source receives it at T_r
            else:
                b.eqs.add(b.T[s, 0 if hot else K] == t_in)  # A44 / A47
            for k in range(K):
                es = members[(s, k)]
                k_in, k_out = (k, k + 1) if hot else (k + 1, k)
                if not es:
                    b.eqs.add(b.T[s, k_out] == b.T[s, k_in])
                    continue
                r = b.r[s, k] if s != LOOP else 0.0
                b.eqs.add(sum(b.f[s, k, e] for e in es) + r == 1.0)  # A25-A28
                mix = r * b.T[s, k_in] + sum(b.f[s, k, e] * b.tb[s, k, e] for e in es)
                b.eqs.add(mix == b.T[s, k_out])  # A31-A34
                for e in es:
                    b.f[s, k, e].setlb(lb)
                    sign = 1.0 if hot else -1.0
                    b.eqs.add(b.q[e] == b.f[s, k, e] * F * sign * (b.T[s, k_in] - b.tb[s, k, e]))  # A17-A22
                    if hot:
                        b.eqs.add(b.tb[s, k, e] <= b.T[s, k_in])
                    else:
                        b.eqs.add(b.tb[s, k, e] >= b.T[s, k_in])
        # Units: approaches (elastic) and Chen area equation.
        b.d1 = pe.Var(list(units), bounds=(0.0, 600.0))
        b.d2 = pe.Var(list(units), bounds=(0.0, 600.0))
        for e, u in units.items():
            k = u.stage
            hin, cin = b.T[u.hot, k], b.T[u.cold, k + 1]
            hout, cout = b.tb[u.hot, k, e], b.tb[u.cold, k, e]
            s1, s2 = m.slack.add(), m.slack.add()
            b.eqs.add(b.d1[e] == hin - cout)
            b.eqs.add(b.d2[e] == hout - cin)
            b.eqs.add(b.d1[e] + s1 >= dt)  # A56-A67, elastic
            b.eqs.add(b.d2[e] + s2 >= dt)
            m.obj_terms.append(m.pk * (s1 + s2))
            b.eqs.add(b.th[e] ** 3 == b.d1[e] * b.d2[e] * (b.d1[e] + b.d2[e]) / 2.0)
            b.A_used[e].setub(u.area)
            b.A_used[e].value = u.area
            if cfg.area_mode == "upper":
                b.eqs.add(b.q[e] == u.U * b.A_used[e] * b.th[e])  # A76 with the used area
                if u.kind == "existing":  # Eq. 9: an existing unit is used at least at its installed area
                    b.A_used[e].setlb(max(u.area - (u.aug or 0.0), 0.0))
            else:
                b.A_used[e].fix(u.area)
                b.eqs.add(b.q[e] == u.U * u.area * b.th[e])  # A76 with the definite area
        # Utilities at the process stream ends (A15-A16, A23-A24, A35-A36, A68-A75).
        b.util = {}
        for s in sids:
            if s == LOOP:
                continue
            hot, t_in, t_out, F, h = streams[s]
            unit = next((x for x in plant.utility_units.values() if x.stream == s), None)
            T_end = b.T[s, K] if hot else b.T[s, 0]
            Q = pe.Var(domain=pe.NonNegativeReals, bounds=(0.0, F * abs(t_in - t_out)))
            setattr(b, f"Q_{s}", Q)
            over = m.slack.add()  # overshoot of the target (A46/A49)
            b.eqs.add(Q == F * ((T_end - t_out) if hot else (t_out - T_end)) + over)
            m.obj_terms.append(m.pk * 10.0 * over)
            if unit is None:
                short = m.slack.add()  # Strategy I: no new utility unit; stream must hit its target
                b.eqs.add(Q <= short)
                m.obj_terms.append(m.pk * 10.0 * short)
                b.util[s] = {"Q": Q, "unit": None, "over": over, "short": short}
                continue
            util = plant.utilities[unit.utility]
            Uu = 1.0 / (1.0 / h + 1.0 / util.h)
            re = pe.Var(bounds=(0.0, cfg.max_bypass))  # end bypass fraction
            tb_u = pe.Var(bounds=(0.0, 600.0))  # branch outlet of the utility unit
            A = pe.Var(bounds=(0.0, 1e4))
            thu = pe.Var(bounds=(1e-3, 600.0))
            setattr(b, f"re_{s}", re)
            setattr(b, f"tbu_{s}", tb_u)
            setattr(b, f"A_{s}", A)
            setattr(b, f"thu_{s}", thu)
            b.eqs.add((1.0 - re) * tb_u + re * T_end == t_out)  # A35/A36
            if hot:
                da, db = T_end - util.t_out, tb_u - util.t_in
            else:
                da, db = util.t_in - tb_u, util.t_out - T_end
            sa_, sb_ = m.slack.add(), m.slack.add()
            b.eqs.add(da + sa_ >= dt)
            b.eqs.add(db + sb_ >= dt)
            m.obj_terms.append(m.pk * (sa_ + sb_))
            b.eqs.add(thu ** 3 == da * db * (da + db) / 2.0)
            b.eqs.add(Q == Uu * A * thu)
            # Eq. 10-11: when the unit is used, its required area Q/(U th) is at least the
            # installed one; the end bypass deepens the branch and raises it.  Written in kW
            # with a smooth on/off factor Q/(Q + 1), elastic in ``under``.
            under = m.slack.add()
            b.eqs.add(Q + under >= Uu * max(unit.area - cfg.area_tolerance, 0.0) * thu * (Q / (Q + 1.0)))
            m.obj_terms.append(m.pk * under)
            b.util[s] = {"Q": Q, "unit": unit, "utility": util, "U": Uu, "re": re, "tb": tb_u, "A": A,
                         "over": over, "under": under}
            price = util.price
            m.obj_terms.append(scale * price * Q)
            m.obj_terms.append(cfg.bypass_weight * re)
            # Sec. 6: a utility unit is either off or at least ``min_duty`` (LQ).  Smooth
            # tent penalty on the forbidden band (0, min_duty): zero at both ends, peak at
            # the middle, slope 0.04 pk kUSD/kW (= 2 kUSD/kW at the final weight, above any
            # utility price), so the solver settles at 0 or at min_duty.
            if cfg.min_duty > 0:
                half = 0.5 * cfg.min_duty
                tent = half - 0.5 * pe.sqrt((Q - half) ** 2 * 4.0 + 1e-2)
                tent = 0.5 * (tent + pe.sqrt(tent * tent + 1e-2))
                m.obj_terms.append(0.04 * m.pk * tent)
        for (s, k) in b.r:
            m.obj_terms.append(cfg.bypass_weight * b.r[s, k])
        m.info[pid] = {"streams": streams, "members": members}
    # Loop headers: supply mix of source outlets, flows given.
    m.eqs.add(m.T_s * total_src == sum(specs[p].flow * getattr(m, f"plant_{p}").T[LOOP, 0] for p in sources))
    m.eqs.add(m.T_s >= m.T_r + 1e-3)
    m.obj = pe.Objective(expr=sum(m.obj_terms), sense=pe.minimize)
    return m
