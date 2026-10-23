"""Site coordinator: action -> intermediate-fluid interfaces -> independent plant solves.

Loop topology (two headers).  Source plants heat parallel branches of the loop from
the common return temperature ``T_r`` to their own outlet temperatures; the
branches mix into the supply header ``T_s``.  Sink plants cool parallel branches
from ``T_s`` back to ``T_r``.  Mass and energy balances hold by construction.
With one sink and two sources this is the paper's Fig. 3/4 arrangement.

Role cases.  As in Lo et al. (Tables 3-5), every coalition of two or more plants
is studied for every assignment of source/sink roles with at least one of each:
12 cases for three plants.  The case is the SAC observation; one policy serves
all cases.  Action (all in [-1, 1]), plants p = 1..n in case-file order:

    a[0]        return temperature T_r, inside the case's thermodynamic window
    a[1..n]     duty request of plant p as a fraction of its own utility:
                hot utility for a sink, cold utility for a source
    a[n+1..2n]  outlet temperature of source p's branch (unused for sinks)

Importing more than one's own hot-utility demand, or exporting more than one's own
cold-utility rejection, can only move heat across that plant's own pinch.  The
larger of total export/import is scaled down to the smaller one so the loop
balances; plants below the 30 kW minimum exchanger duty then drop out.

``case=None`` keeps a role-free variant: the sign of a[1..n] picks the role and
|a| < ``dead_zone`` leaves the plant out.  It is retained as an ablation; about
half of its action space has no loop at all, and SAC collapses onto that
zero-saving plateau.
"""
from __future__ import annotations

import copy
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from .case import CASE_FILE, load_case
from .plant import Interface, PlantSolver, PlantSolverConfig

SIGN = {"source": 1.0, "sink": -1.0, None: 0.0}
CODE = {"source": "+", "sink": "-", None: "0"}


@dataclass(frozen=True)
class RoleCase:
    """Source/sink role of every plant; ``None`` = not in the coalition."""
    roles: tuple

    @property
    def members(self):
        return tuple(i for i, r in enumerate(self.roles) if r is not None)

    @property
    def code(self) -> str:
        return "".join(CODE[r] for r in self.roles)

    def observation(self) -> np.ndarray:
        return np.array([SIGN[r] for r in self.roles], np.float32)

    @classmethod
    def from_code(cls, code: str) -> "RoleCase":
        inverse = {v: k for k, v in CODE.items()}
        return cls(tuple(inverse[c] for c in code))


def all_role_cases(n: int) -> list:
    """Every coalition of >= 2 plants with a role assignment containing both roles."""
    cases = [RoleCase(r) for r in itertools.product((None, "sink", "source"), repeat=n)
             if "sink" in r and "source" in r]
    return sorted(cases, key=lambda c: (-len(c.members), c.code))


@dataclass(frozen=True)
class SiteConfig:
    members: tuple | None = None  # coalition for case=None; None = all plants
    dt_min: float = 1.0
    dead_zone: float = 0.05
    min_plant_duty: float = 30.0
    min_lift: float = 2.0  # K between return temperature and a source outlet
    t_return_low: float | None = None
    t_return_high: float | None = None
    duty_multiplier: float = 1.0
    # Share of the duty action range at its top that maps exactly onto the plant's
    # maximum duty, so a policy with tanh-squashed actions can reach the bound.
    duty_saturation: float = 0.0
    # "min": the larger of total export/import is scaled down to the smaller one.
    # "sources": every source exports exactly its request; sinks share the total in
    # proportion to their requests, each capped at its maximum import.
    balance: str = "min"
    # Utility-elimination snap: the top ``duty_snap`` share of a duty action means
    # "replace all of my utility"; below it the duty spans 0 .. max - min_residual_utility.
    # With min_residual_utility = 30 kW no action leaves a forbidden 0-30 kW utility.
    duty_snap: float = 0.0
    min_residual_utility: float = 0.0


class SiteModel:
    """Pure (no solver) action decoding and cost assembly."""

    def __init__(self, cfg: SiteConfig | None = None, case_file=CASE_FILE):
        self.cfg = cfg or SiteConfig()
        self.plants, self.costs = load_case(case_file)
        self.ids = tuple(self.plants)
        self.members = tuple(self.cfg.members or self.ids)
        if not set(self.members) <= set(self.ids) or not self.members:
            raise ValueError("coalition must be a non-empty subset of the plants")
        dt = self.cfg.dt_min
        self.cold_low = {pid: min(s.t_in for s in p.streams.values() if not s.hot)
                         for pid, p in self.plants.items()}
        # Hottest outlet a source plant can reach: its hottest hot stream minus approach.
        self.outlet_high = {pid: max(s.t_in for s in p.streams.values() if s.hot) - dt
                            for pid, p in self.plants.items()}
        self.t_low = self.cfg.t_return_low if self.cfg.t_return_low is not None else \
            min(self.cold_low.values()) + dt
        self.t_high = self.cfg.t_return_high if self.cfg.t_return_high is not None else \
            max(self.outlet_high.values()) - self.cfg.min_lift
        self.duty_max = {pid: {"sink": self.cfg.duty_multiplier * p.utility_duty("heater"),
                               "source": self.cfg.duty_multiplier * p.utility_duty("cooler")}
                         for pid, p in self.plants.items()}
        self.baseline = {pid: p.baseline_utility_cost for pid, p in self.plants.items()}
        self.site_baseline = sum(self.baseline.values())
        self.baseline_total = sum(self.baseline[p] for p in self.members)
        self.action_dim = 1 + 2 * len(self.ids)
        self.observation_dim = len(self.ids)
        self.labels = (["T_return"] + [f"duty:{p}" for p in self.ids] + [f"T_out:{p}" for p in self.ids])
        self.cases = all_role_cases(len(self.ids))

    def return_window(self, case: RoleCase | None):
        """Return-temperature range: every sink can cool to it, every source heat from it."""
        if case is None:
            return self.t_low, self.t_high
        dt = self.cfg.dt_min
        sinks = [self.ids[i] for i, r in enumerate(case.roles) if r == "sink"]
        sources = [self.ids[i] for i, r in enumerate(case.roles) if r == "source"]
        low = max(self.cold_low[p] for p in sinks) + dt
        high = min(self.outlet_high[p] for p in sources) - self.cfg.min_lift
        if self.cfg.t_return_low is not None:
            low = max(low, self.cfg.t_return_low)
        if self.cfg.t_return_high is not None:
            high = min(high, self.cfg.t_return_high)
        if not high > low:
            raise ValueError(f"role case {case.code} has an empty return-temperature window")
        return low, high

    def _duty(self, a, qmax):
        """Duty requested by one role-case action coordinate."""
        cfg = self.cfg
        u = (a + 1.0) / 2.0
        if cfg.duty_snap > 0:
            if u >= 1.0 - cfg.duty_snap:
                return qmax
            return u / (1.0 - cfg.duty_snap) * max(qmax - cfg.min_residual_utility, 0.0)
        return min(1.0, u * (1.0 + cfg.duty_saturation)) * qmax

    def _duty_inverse(self, q, qmax):
        cfg = self.cfg
        if cfg.duty_snap > 0:
            if q >= qmax - 1e-9:
                return 2.0 * (1.0 - cfg.duty_snap / 2.0) - 1.0
            return 2.0 * q / max(qmax - cfg.min_residual_utility, 1e-12) * (1.0 - cfg.duty_snap) - 1.0
        return 2.0 * q / qmax / (1.0 + cfg.duty_saturation) - 1.0

    def _balance_sources(self, duty, role):
        """Sources export exactly their requests (scaled only if the sinks cannot
        absorb the total); sinks share the total in proportion to their requests,
        each capped at its maximum import (water-filling)."""
        sources = [p for p in self.ids if role[p] == "source"]
        sinks = [p for p in self.ids if role[p] == "sink"]
        out = dict.fromkeys(self.ids, 0.0)
        if not sources or not sinks:
            return out
        cap = {p: self.duty_max[p]["sink"] for p in sinks}
        total = sum(duty[p] for p in sources)
        scale = min(1.0, sum(cap.values()) / total) if total > 0 else 0.0
        for p in sources:
            out[p] = duty[p] * scale
        remaining, open_ = total * scale, list(sinks)
        weight = {p: duty[p] if sum(duty[q] for q in sinks) > 0 else 1.0 for p in sinks}
        while remaining > 1e-12 and open_:
            w = sum(weight[p] for p in open_) or float(len(open_))
            share = {p: remaining * (weight[p] if w else 1.0) / w for p in open_}
            full = [p for p in open_ if out[p] + share[p] >= cap[p]]
            if not full:
                for p in open_:
                    out[p] += share[p]
                break
            for p in full:
                remaining -= cap[p] - out[p]
                out[p] = cap[p]
                open_.remove(p)
        return out

    def decode(self, action, case: RoleCase | None = None) -> dict:
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        if a.shape != (self.action_dim,):
            raise ValueError(f"action must have {self.action_dim} entries")
        n, cfg = len(self.ids), self.cfg
        t_low, t_high = self.return_window(case)
        t_r = t_low + (a[0] + 1.0) / 2.0 * (t_high - t_low)
        members = self.members if case is None else tuple(self.ids[i] for i in case.members)
        request, outlet = {}, {}
        for k, pid in enumerate(self.ids):
            if pid not in members:
                request[pid] = ("inactive", 0.0)
            elif case is None:
                frac = max(0.0, (abs(a[1 + k]) - cfg.dead_zone) / (1.0 - cfg.dead_zone))
                role = "source" if a[1 + k] > 0 else "sink"
                request[pid] = (role, frac * self.duty_max[pid][role]) if frac > 0 else ("inactive", 0.0)
            else:
                role = case.roles[k]
                request[pid] = (role, self._duty(a[1 + k], self.duty_max[pid][role]))
            lo = t_r + cfg.min_lift
            hi = max(lo, self.outlet_high[pid])
            outlet[pid] = lo + (a[1 + n + k] + 1.0) / 2.0 * (hi - lo)
        duty = {pid: q for pid, (r, q) in request.items()}
        role = {pid: r for pid, (r, q) in request.items()}
        for _ in range(len(self.ids) + 1):
            if case is not None and cfg.balance == "sources":
                scaled = self._balance_sources(duty, role)
            else:
                src = sum(duty[p] for p in self.ids if role[p] == "source")
                snk = sum(duty[p] for p in self.ids if role[p] == "sink")
                loop = min(src, snk)
                scaled = {p: duty[p] * (loop / src if role[p] == "source" else loop / snk)
                          if role[p] != "inactive" and loop > 0 else 0.0 for p in self.ids}
            drop = [p for p in self.ids if role[p] != "inactive" and scaled[p] < cfg.min_plant_duty]
            if not drop:
                break
            for p in drop:
                role[p], duty[p] = "inactive", 0.0
        active = {p: scaled[p] for p in self.ids if role[p] != "inactive" and scaled[p] > 0}
        interfaces = {p: Interface() for p in self.ids}
        flows = {}
        if active:
            sources = [p for p in active if role[p] == "source"]
            sinks = [p for p in active if role[p] == "sink"]
            for p in sources:
                flows[p] = active[p] / (outlet[p] - t_r)
                interfaces[p] = Interface(flows[p], t_r, outlet[p])
            total = sum(flows[p] for p in sources)
            t_s = sum(flows[p] * outlet[p] for p in sources) / total
            for p in sinks:
                flows[p] = active[p] / (t_s - t_r)
                interfaces[p] = Interface(flows[p], t_s, t_r)
        else:
            t_s, total = t_r, 0.0
        roles = {p: interfaces[p].role for p in self.ids}
        return {"action": a.tolist(), "case": case.code if case else None, "members": list(members),
                "T_return": t_r, "T_return_window": [t_low, t_high], "T_supply": t_s, "loop_flow": total,
                "roles": roles, "requested": {p: {"role": r, "duty": q} for p, (r, q) in request.items()},
                "interfaces": {p: asdict(interfaces[p]) | {"duty": interfaces[p].duty, "role": roles[p]}
                               for p in self.ids},
                "_interfaces": interfaces,
                "balance_kW": sum((1 if roles[p] == "source" else -1) * interfaces[p].duty
                                  for p in self.ids if roles[p] != "inactive")}

    def encode(self, t_return: float, duties: dict, outlets: dict, case: RoleCase | None = None) -> np.ndarray:
        """Inverse of :meth:`decode` for validation only (signed duties: + export).

        Used to re-evaluate a published design; never used by the search.
        """
        n, cfg = len(self.ids), self.cfg
        t_low, t_high = self.return_window(case)
        a = np.zeros(self.action_dim) if case is None else -np.ones(self.action_dim)
        a[0] = 2.0 * (t_return - t_low) / (t_high - t_low) - 1.0
        for k, pid in enumerate(self.ids):
            q = duties.get(pid, 0.0)
            if q:
                frac = abs(q) / self.duty_max[pid]["source" if q > 0 else "sink"]
                qmax = self.duty_max[pid]["source" if q > 0 else "sink"]
                a[1 + k] = (math.copysign(cfg.dead_zone + frac * (1.0 - cfg.dead_zone), q)
                            if case is None else self._duty_inverse(abs(q), qmax))
            if pid in outlets:
                lo = t_return + cfg.min_lift
                hi = max(lo, self.outlet_high[pid])
                a[1 + n + k] = 2.0 * (outlets[pid] - lo) / (hi - lo) - 1.0
        if np.any(np.abs(a) > 1.0 + 1e-9):
            raise ValueError("design lies outside the action domain")
        return np.clip(a, -1.0, 1.0)

    def assemble(self, decoded: dict, plant_results: dict) -> dict:
        c = self.costs
        members = decoded["members"]
        baseline = sum(self.baseline[p] for p in members)
        loop_capital = c.cim * decoded["loop_flow"]
        plant_tac = {p: plant_results[p]["TAC"] for p in members}
        tac = sum(plant_tac.values()) + c.af * loop_capital
        slack = sum(plant_results[p]["slack_kW"] for p in members)
        strict = all(plant_results[p]["strictly_feasible"] for p in members)
        penalty = sum(plant_results[p]["costs"]["slack_penalty"] for p in members)
        out = {k: v for k, v in decoded.items() if not k.startswith("_")}
        # Savings are normalised by the whole site's original utility cost so that
        # rewards of different coalitions share one scale.
        out.update({"TAC": tac, "TACS": baseline - tac, "coalition_baseline_TAC": baseline,
                    "site_TAC": tac + self.site_baseline - baseline, "TAC_without_penalty": tac - penalty,
                    "slack_kW": slack, "strictly_feasible": strict, "loop_capital": loop_capital,
                    "loop_annual_cost": c.af * loop_capital, "plant_TAC": plant_tac,
                    "plants": {p: plant_results[p] for p in members},
                    "reward": reward_from_savings((baseline - tac) / self.site_baseline)})
        return out


def reward_from_savings(s: float) -> float:
    """Fractional TAC saving; losses are compressed smoothly into (-1, 0)."""
    return float(s if s >= 0 else s / (1.0 - s))


_WORKER = {}


def _init_worker(case_file, solver_cfg):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    plants, costs = load_case(case_file)
    _WORKER["solvers"] = {p: PlantSolver(plants[p], costs, solver_cfg) for p in plants}


def _solve(pid, flow, t_in, t_out):
    return _WORKER["solvers"][pid].solve(Interface(flow, t_in, t_out))


class SiteEvaluator:
    """Evaluate batches of site actions; plant problems run in parallel processes."""

    def __init__(self, site_cfg: SiteConfig | None = None, solver_cfg: PlantSolverConfig | None = None,
                 workers: int | None = None, case_file=CASE_FILE):
        self.model = SiteModel(site_cfg, case_file)
        self.solver_cfg = solver_cfg or PlantSolverConfig(dt_min=self.model.cfg.dt_min)
        self.case_file = case_file
        self.workers = workers if workers is not None else max(1, min(18, (os.cpu_count() or 2) - 2))
        self._pool = None
        self._local = None
        self._inactive = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def _ensure(self):
        if self.workers > 1 and self._pool is None:
            self._pool = ProcessPoolExecutor(self.workers, initializer=_init_worker,
                                             initargs=(str(self.case_file), self.solver_cfg))
        if self.workers <= 1 and self._local is None:
            plants, costs = load_case(self.case_file)
            self._local = {p: PlantSolver(plants[p], costs, self.solver_cfg) for p in plants}

    def _inactive_result(self, pid):
        if pid not in self._inactive:
            self._ensure()
            if self._pool is not None:
                self._inactive[pid] = self._pool.submit(_solve, pid, 0.0, 0.0, 0.0).result()
            else:
                self._inactive[pid] = self._local[pid].solve(Interface())
        return copy.deepcopy(self._inactive[pid])

    def evaluate(self, actions, cases=None) -> list:
        """Evaluate actions; ``cases`` holds one RoleCase per action (None: role-free)."""
        actions = np.atleast_2d(np.asarray(actions, float))
        self._ensure()
        cases = [None] * len(actions) if cases is None else list(cases)
        decoded = [self.model.decode(a, c) for a, c in zip(actions, cases)]
        jobs = {}
        for i, d in enumerate(decoded):
            for p in d["members"]:
                itf = d["_interfaces"][p]
                if itf.active:
                    args = (p, itf.flow, itf.t_in, itf.t_out)
                    jobs[(i, p)] = self._pool.submit(_solve, *args) if self._pool else args
        results = []
        for i, d in enumerate(decoded):
            plant_results = {}
            for p in d["members"]:
                job = jobs.get((i, p))
                if job is None:
                    plant_results[p] = self._inactive_result(p)
                elif self._pool is not None:
                    plant_results[p] = job.result()
                else:
                    plant_results[p] = self._local[p].solve(Interface(*job[1:]))
            results.append(self.model.assemble(d, plant_results))
        return results
