"""Agent action -> definite areas and loop flows -> site NLP -> costs and rule checks."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from tshi_decomp.case import CASE_FILE, load_case
from tshi_decomp.site import RoleCase, all_role_cases
from tshi_sim.plant import chen, reconciled_areas
from tshi_sim.site import stages_for_case

from .model import LOOP, NLPConfig, PlantSpec, UnitSpec, build_site_model


@dataclass(frozen=True)
class Rules:
    dt_min: float = 1.0
    min_duty: float = 30.0
    min_bypass: float = 0.1
    bypass_cost: float = 500.0
    # USD/(kW yr) on rule residuals in the reward (ours).  Five times the loop slack price of
    # the earlier studies: with definite areas an oversized unit can only meet dt_min by
    # violating it, and at 2,000 the agent found that worth paying.
    slack_price: float = 10000.0
    strict_tolerance: float = 1.0  # kW-equivalent of rule residuals a reported design may carry
    area_tolerance: float = 0.05
    area_dead_zone: float = 0.15
    approach_scale: float = 1.2
    flow_min: float = 0.5  # kW/K
    flow_max: float = 40.0


class SiteNLP:
    """One action vector describes the whole site; the NLP solves its operation."""

    def __init__(self, case_file=CASE_FILE, cfg: NLPConfig | None = None, rules: Rules | None = None):
        self.plants, self.costs = load_case(case_file)
        self.rules = rules or Rules()
        self.cfg = cfg or NLPConfig(dt_min=self.rules.dt_min, area_tolerance=self.rules.area_tolerance,
                                    min_duty=self.rules.min_duty)
        self.installed = {pid: reconciled_areas(p, self.costs, self.rules.dt_min, self.rules.area_tolerance)
                          for pid, p in self.plants.items()}
        self.baseline = {pid: p.baseline_utility_cost for pid, p in self.plants.items()}
        self.site_baseline = sum(self.baseline.values())
        self.cases = all_role_cases(len(self.plants))
        self._layouts = {}
        self._sims = {}

    # -- action layout ----------------------------------------------------------------
    def layout(self, case: RoleCase) -> list:
        """Action entries: (kind, plant, key).  kinds: stage/area of an existing unit, loop
        (area of a candidate intermediate exchanger at a stage), flow, split."""
        if case.code in self._layouts:
            return self._layouts[case.code]
        K = stages_for_case(self.plants, case)
        entries = []
        roles = {pid: r for pid, r in zip(self.plants, case.roles) if r is not None}
        for pid, role in roles.items():
            p = self.plants[pid]
            for e in p.exchangers.values():
                entries.append(("stage", pid, e.id))
                entries.append(("area", pid, e.id))
            streams = [s.id for s in p.streams.values() if (s.hot if role == "source" else not s.hot)]
            for s in streams:
                for k in range(K):
                    entries.append(("loop", pid, (s, k)))
        entries.append(("flow", None, None))
        sources = [pid for pid, r in roles.items() if r == "source"]
        sinks = [pid for pid, r in roles.items() if r == "sink"]
        for pid in sources[:-1]:
            entries.append(("split", pid, "source"))
        for pid in sinks[:-1]:
            entries.append(("split", pid, "sink"))
        self._layouts[case.code] = entries
        return entries

    def action_dim(self, case: RoleCase) -> int:
        return len(self.layout(case))

    def plant_roles(self, case: RoleCase) -> dict:
        return {pid: r for pid, r in zip(self.plants, case.roles) if r is not None}

    def decode(self, a, case: RoleCase) -> tuple[dict, float]:
        """Action -> {pid: PlantSpec}, total loop flow."""
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        entries = self.layout(case)
        if a.shape != (len(entries),):
            raise ValueError(f"action must have {len(entries)} entries")
        K = stages_for_case(self.plants, case)
        R, c = self.rules, self.costs
        roles = {pid: r for pid, r in zip(self.plants, case.roles) if r is not None}
        vals = dict(zip(range(len(entries)), a))
        stage, area_x, loop_area, splits = {}, {}, {}, {}
        flow = R.flow_min
        for i, (kind, pid, key) in enumerate(entries):
            u = (a[i] + 1.0) / 2.0
            if kind == "stage":
                stage[(pid, key)] = min(int(u * K), K - 1)
            elif kind == "area":
                area_x[(pid, key)] = a[i]
            elif kind == "loop":
                loop_area[(pid, key)] = u
            elif kind == "flow":
                flow = math.exp(math.log(R.flow_min) + u * (math.log(R.flow_max) - math.log(R.flow_min)))
            elif kind == "split":
                splits[(pid, key)] = u
        # Eq. 7: one existing unit of a match per stage.
        for pid in roles:
            pairs = {}
            for e in self.plants[pid].exchangers.values():
                pairs.setdefault((e.hot, e.cold), []).append(e.id)
            for ids in pairs.values():
                used = set()
                for uid in ids:
                    k = stage[(pid, uid)]
                    while k in used:
                        k = (k + 1) % K
                    stage[(pid, uid)] = k
                    used.add(k)
        # Loop flows: stick-breaking among sources and among sinks (each side sums to the total).
        def share(group):
            """Stick-breaking shares of the loop flow (each branch keeps at least 5 %)."""
            out, rest = {}, 1.0
            for pid in group[:-1]:
                s = splits[(pid, roles[pid])]
                take = rest * (0.05 + 0.9 * s)
                out[pid] = take
                rest -= take
            if group:
                out[group[-1]] = rest
            return out
        sources = [pid for pid, r in roles.items() if r == "source"]
        sinks = [pid for pid, r in roles.items() if r == "sink"]
        shares = {}
        shares.update(share(sources))
        shares.update(share(sinks))
        specs = {}
        for pid, role in roles.items():
            p = self.plants[pid]
            units = []
            h = {s.id: s.h for s in p.streams.values()}
            h[LOOP] = c.loop_h
            for e in p.exchangers.values():
                inst = self.installed[pid][e.id]
                qmax = min(p.streams[e.hot].load, p.streams[e.cold].load)
                U = 1.0 / (1.0 / h[e.hot] + 1.0 / h[e.cold])
                xmax = max(c.min_augmentation, qmax / (U * R.approach_scale) - inst)
                x = area_x[(pid, e.id)]
                X = c.min_augmentation + x * (xmax - c.min_augmentation) if x > 0 else 0.0
                units.append(UnitSpec(e.id, "existing", e.hot, e.cold, stage[(pid, e.id)], inst + X, U, e.id, X))
            for (q_pid, (s, k)), u in loop_area.items():
                if q_pid != pid or u <= R.area_dead_zone:
                    continue
                st = p.streams[s]
                U = 1.0 / (1.0 / st.h + 1.0 / c.loop_h)
                amax = st.load / (U * R.approach_scale)
                A = amax * ((u - R.area_dead_zone) / (1.0 - R.area_dead_zone)) ** 2
                hot, cold = (s, LOOP) if role == "source" else (LOOP, s)
                units.append(UnitSpec(f"LOOP-{s}@{k + 1}", "loop", hot, cold, k, A, U))
            specs[pid] = PlantSpec(p, role, flow * shares[pid], units)
        return specs, flow

    # -- NLP ---------------------------------------------------------------------------
    def solve(self, specs: dict, case: RoleCase, flow: float) -> dict:
        """Three stages: (A) IPOPT with equal splits and no bypass fixed, which settles the
        loop temperatures; (B) the fast rating simulator samples split/bypass settings at
        those interfaces and the best becomes the starting point; (C) IPOPT with everything
        free and the elastic-penalty continuation.  Returns the costed solution."""
        import pyomo.environ as pe
        from pyomo.opt import SolverStatus, TerminationCondition as TC

        K = stages_for_case(self.plants, case)
        started = time.perf_counter()
        m = build_site_model(specs, self.costs, K, self.cfg)
        self._initialize(m, specs, K)
        solver = pe.SolverFactory("ipopt")
        solver.options.update({"tol": self.cfg.ipopt_tol, "max_iter": self.cfg.ipopt_max_iter,
                               "max_cpu_time": self.cfg.ipopt_max_cpu,
                               "print_level": 5 if self.cfg.ipopt_print else 0, "mu_strategy": "adaptive",
                               "bound_relax_factor": 0.0, "honor_original_bounds": "yes"})
        fixed = []
        for pid in specs:
            b = getattr(m, f"plant_{pid}")
            for idx in b.f:
                b.f[idx].fix(b.f[idx].value)
                fixed.append(b.f[idx])
            for idx in b.r:
                b.r[idx].fix(0.0)
                fixed.append(b.r[idx])
            for s, uu in b.util.items():
                if uu["unit"] is not None:
                    uu["re"].fix(0.0)
                    fixed.append(uu["re"])
        schedule = tuple(self.cfg.penalty_schedule) or (self.cfg.penalty_k,)
        ok, term, sim_info = False, "not solved", None

        def run(weight):
            m.pk.set_value(weight)
            res = solver.solve(m, tee=self.cfg.ipopt_print, load_solutions=False)
            good = res.solver.status == SolverStatus.ok and res.solver.termination_condition in (TC.optimal, TC.locallyOptimal)
            if res.solver.termination_condition != TC.infeasible:
                m.solutions.load_from(res)
            return good, str(res.solver.termination_condition)

        try:
            ok, term = run(schedule[0])  # (A): loop temperatures with fixed fractions
            for v in fixed:
                v.unfix()
            if self.cfg.sim_samples > 0:
                sim_info = self._simulator_start(m, specs, K)  # (B)
            for weight in schedule:  # (C)
                ok, term = run(weight)
            # (D) Eq. 25-28: a bypass is 0 or at least 0.1.  Bypasses the free solve left
            # below 0.1 are snapped (all to 0, or all to 0.1) and the operation re-solved
            # from the free solution; the better compliant re-solve is reported.
            small = []
            for pid in specs:
                b = getattr(m, f"plant_{pid}")
                for idx in b.r:
                    if 1e-6 < float(b.r[idx].value or 0.0) < self.cfg.min_fraction:
                        small.append(b.r[idx])
                for s, uu in b.util.items():
                    if uu["unit"] is not None and 1e-6 < float(uu["re"].value or 0.0) < self.cfg.min_fraction:
                        small.append(uu["re"])
            if small:
                allvars = list(m.component_data_objects(pe.Var))
                free_point = [v.value for v in allvars]
                best_point, best_val, best_ok, best_term = None, np.inf, ok, term
                for snap in (0.0, self.cfg.min_fraction):
                    for v, val in zip(allvars, free_point):
                        v.value = val
                    for v in small:
                        v.fix(snap)
                    ok_s, term_s = run(schedule[-1])
                    for v in small:
                        v.unfix()
                    score = self._extract(m, specs, case, flow, K)["TAC_with_penalty"]
                    if ok_s and score < best_val:
                        best_point, best_val, best_ok, best_term = [v.value for v in allvars], score, ok_s, term_s
                if best_point is not None:
                    for v, val in zip(allvars, best_point):
                        v.value = val
                    ok, term = best_ok, best_term
                else:
                    for v, val in zip(allvars, free_point):
                        v.value = val
        except Exception as exc:  # solver crash: report as failed
            ok, term = False, f"error: {exc}"
        out = self._extract(m, specs, case, flow, K)
        out["solver"] = {"ok": ok, "termination": term, "seconds": time.perf_counter() - started, "start": sim_info}
        self._last = (m, specs)
        if not ok:
            out["strictly_feasible"] = False
        if not all(np.isfinite(out[k]) for k in ("TAC", "TAC_with_penalty", "violation_kW")):
            # No usable solution: the coalition keeps its original networks and pays a
            # failure penalty equal to the largest stream load of the site.
            baseline = sum(self.baseline[p] for p in specs)
            load = max(s.load for p in specs for s in self.plants[p].streams.values())
            out.update({"TAC": baseline, "TACS": 0.0, "violation_kW": load, "penalty": self.rules.slack_price * load,
                        "TAC_with_penalty": baseline + self.rules.slack_price * load, "strictly_feasible": False})
            s = -self.rules.slack_price * load / self.site_baseline
            out["reward"] = float(s / (1.0 - s))
        return out

    def _simulator(self, pid, role, K):
        from tshi_sim.plant import PlantLayout, PlantSimulator, SimConfig
        key = (pid, role, K)
        if key not in self._sims:
            # a rough fixed point is enough for a starting point
            cfg = SimConfig(dt_min=self.rules.dt_min, area_tolerance=self.rules.area_tolerance, fp_max_iter=60, fp_tol=1e-5)
            self._sims[key] = PlantSimulator(PlantLayout(self.plants[pid], self.costs, role, K, cfg))
        return self._sims[key]

    def _sim_spec(self, pid, sp):
        """``tshi_sim`` design spec (stages, augmentations, new-unit areas, equal splits)."""
        spec = {"stage": {}, "aug": {}, "area": {}, "split": {}}
        for u in sp.units:
            if u.kind == "existing":
                spec["stage"][u.unit] = u.stage + 1
                x = u.area - self.installed[pid][u.unit]
                if x > 1e-9:
                    spec["aug"][u.unit] = x
            else:
                spec["area"][u.id] = u.area
        groups = {}
        for u in sp.units:
            for s in (u.hot, u.cold):
                groups.setdefault((s, u.stage + 1), []).append(u.id)
        for key, ids in groups.items():
            if len(ids) >= 2:
                spec["split"][key] = {i: 1.0 / len(ids) for i in ids}
        return spec

    def _loop_pass(self, specs, K, actions, iters=4):
        """Consistent loop temperatures from the rating simulator alone: sources heat
        their branches from T_r as far as their areas allow, the branches mix into T_s,
        the sinks cool from T_s and return the fluid; iterate T_r (damped).  ``actions``
        holds each plant's simulator action (its split/bypass settings)."""
        from tshi_decomp.plant import Interface
        dt = self.rules.dt_min
        sources = [p for p, s in specs.items() if s.role == "source"]
        sinks = [p for p, s in specs.items() if s.role == "sink"]
        T_r = min(min(s.t_in for s in self.plants[p].streams.values() if not s.hot) for p in sinks) + 2.0 * dt
        itfs = {}
        for _ in range(iters):
            outs = {}
            for p in sources:
                sp = specs[p]
                sim = self._simulator(p, sp.role, K)
                t_max = max(s.t_in for s in self.plants[p].streams.values() if s.hot) - dt
                itf = Interface(sp.flow, T_r, max(t_max, T_r + 1.0))
                ev = sim.evaluate(actions[p][None, :], itf)
                outs[p] = float(ev["t_end"][0, sim.L.sidx[LOOP]])
                itfs[p] = Interface(sp.flow, T_r, max(outs[p], T_r + 1.0))
            T_s = sum(specs[p].flow * outs[p] for p in sources) / sum(specs[p].flow for p in sources)
            rets = []
            for p in sinks:
                sp = specs[p]
                sim = self._simulator(p, sp.role, K)
                t_min = min(s.t_in for s in self.plants[p].streams.values() if not s.hot) + dt
                itf = Interface(sp.flow, T_s, min(t_min, T_s - 1.0))
                ev = sim.evaluate(actions[p][None, :], itf)
                ret = float(ev["t_end"][0, sim.L.sidx[LOOP]])
                itfs[p] = Interface(sp.flow, T_s, min(ret, T_s - 1.0))
                rets.append(ret)
            T_new = sum(specs[p].flow * r for p, r in zip(sinks, rets)) / sum(specs[p].flow for p in sinks)
            T_r = 0.5 * (T_r + T_new)
        return itfs, T_r, T_s

    def _simulator_start(self, m, specs, K):
        """Sample split/bypass settings with the rating simulator at the loop temperatures
        of the fixed-fraction solve (open-ended loop outlet targets); load the best point
        of every plant into the NLP as the start of the free solve."""
        from tshi_decomp.plant import Interface
        from tshi_sim.encode import actions_from_spec
        rng = np.random.default_rng(0)
        n = self.cfg.sim_samples
        dt = self.rules.dt_min
        T_r, T_s = float(m.T_r.value), float(m.T_s.value)
        info = {"T_r": T_r, "T_s": T_s}
        for pid, sp in specs.items():
            sim = self._simulator(pid, sp.role, K)
            L = sim.L
            streams = self.plants[pid].streams.values()
            if sp.role == "source":
                itf = Interface(sp.flow, T_r, max(max(s.t_in for s in streams if s.hot) - dt, T_r + 1.0))
            else:
                itf = Interface(sp.flow, T_s, min(min(s.t_in for s in streams if not s.hot) + dt, T_s - 1.0))
            base = actions_from_spec(sim, itf, self._sim_spec(pid, sp))
            cols_split = [c for g in L.groups for c in g[3]]
            cols_byp = list(L.bypass_col.values())
            A = np.tile(base, (n, 1))
            if n > 1 and cols_split:
                A[1:, cols_split] = rng.uniform(-1.0, 1.0, (n - 1, len(cols_split)))
            if n > 1 and cols_byp:
                use = rng.uniform(size=(n - 1, len(cols_byp))) < 0.3
                A[1:, cols_byp] = np.where(use, rng.uniform(-1.0, 1.0, (n - 1, len(cols_byp))), -1.0)
            ev = sim.evaluate(A, itf)
            # Score without the loop-duty slack (open-ended targets) and with rule violations
            # weighted mildly: definite areas leave residuals such as a 0.01 kW utility that
            # the 30 kW rule would otherwise price at 60 kUSD and steer the start away.
            score = ev["tac_clean"] + 50.0 * ev["violation"]
            i = int(np.argmin(score))
            x = sim.export(A[i], itf)
            self._load_from_sim(getattr(m, f"plant_{pid}"), x, m.info[pid]["streams"], K, itf)
            info[pid] = {"sample": i, "sim_TAC": float(ev["tac_clean"][i]), "violation": float(ev["violation"][i]),
                         "interface": (itf.flow, itf.t_in, itf.t_out)}
        return info

    @staticmethod
    def _load_from_sim(b, x, streams, K, itf):
        """NLP starting values from a ``tshi_sim`` export of the same plant."""
        ex = {e["id"]: e for e in x["exchangers"]}
        byp = {(bb["stream"], bb["stage"]): bb["fraction"] for bb in x["bypasses"]}
        for s, (hot, t_in, t_out, F, h) in streams.items():
            if s == LOOP:
                t_in = itf.t_in
            qk = np.zeros(K)
            for e in x["exchangers"]:
                if (e["hot"] if hot else e["cold"]) == s:
                    qk[e["stage"] - 1] += e["duty"]
            T = t_in
            if hot:
                b.T[s, 0].value = T
                for k in range(K):
                    T -= qk[k] / F
                    b.T[s, k + 1].value = T
            else:
                b.T[s, K].value = T
                for k in range(K - 1, -1, -1):
                    T += qk[k] / F
                    b.T[s, k].value = T
        for (s, k), es in b.members.items():
            for e in es:
                if e in ex:
                    hot = streams[s][0]
                    b.tb[s, k, e].value = ex[e]["hot_out"] if hot else ex[e]["cold_out"]
                    b.f[s, k, e].value = min(max(ex[e]["hot_fraction"] if hot else ex[e]["cold_fraction"], 0.1), 1.0)
        for (s, k) in b.r:
            b.r[s, k].value = min(byp.get((s, k + 1), 0.0), 0.9)
        for e, u in b.units.items():
            if e not in ex:  # a unit the simulator dropped (no duty): keep the NLP's own start
                continue
            b.q[e].value = ex[e]["duty"]
            b.d1[e].value = max(ex[e]["hot_in"] - ex[e]["cold_out"], 1e-3)
            b.d2[e].value = max(ex[e]["hot_out"] - ex[e]["cold_in"], 1e-3)
            b.th[e].value = float(chen(b.d1[e].value, b.d2[e].value))
        for u in x["utilities"]:
            s = u["stream"]
            uu = b.util[s]
            uu["Q"].value = u["duty"]
            if uu["unit"] is not None:
                hot, t_in, t_out, F, h = streams[s]
                uu["re"].value = min(u["bypass"], 0.9)
                uu["tb"].value = u["branch_out"] if u["duty"] > 0 else t_out
                uu["A"].value = max(u["area_required"], 1e-3)
                util = uu["utility"]
                T_end = x["stream_end_temperatures"][s]
                da, db = ((T_end - util.t_out, uu["tb"].value - util.t_in) if hot
                          else (util.t_in - uu["tb"].value, util.t_out - T_end))
                getattr(b, "thu_" + s).value = float(chen(max(da, 1e-3), max(db, 1e-3)))

    def _initialize(self, m, specs, K, mode="equal"):
        m.T_r.value, m.T_s.value = 100.0, 150.0
        for pid, sp in specs.items():
            b = getattr(m, f"plant_{pid}")
            streams = m.info[pid]["streams"]
            for s, (hot, t_in, t_out, F, h) in streams.items():
                if s == LOOP:
                    lo, hi = (100.0, 150.0)
                    t_in, t_out = (hi, lo) if hot else (lo, hi)
                for k in range(K + 1):
                    frac = k / K
                    b.T[s, k].value = t_in + (t_out - t_in) * (frac if hot else 1.0 - frac)
            for (s, k), es in b.members.items():
                if not es:
                    continue
                lbf = self.cfg.min_fraction
                if mode == "ua":
                    w = np.array([b.units[e].U * b.units[e].area for e in es])
                elif mode.startswith("random"):
                    w = np.random.default_rng(int(mode[6:] or 0) + len(es)).uniform(0.1, 1.0, len(es))
                else:
                    w = np.ones(len(es))
                w = np.clip(lbf + (1.0 - lbf * len(es)) * w / w.sum(), lbf, 1.0)
                for e, wi in zip(es, w):
                    b.f[s, k, e].value = float(wi)
                    hot = streams[s][0]
                    b.tb[s, k, e].value = b.T[s, k + 1].value if hot else b.T[s, k].value
            for (s, k) in b.r:
                b.r[s, k].value = 0.0
            for e, u in b.units.items():
                k = u.stage
                d1 = max(b.T[u.hot, k].value - b.tb[u.cold, k, e].value, 1.0)
                d2 = max(b.tb[u.hot, k, e].value - b.T[u.cold, k + 1].value, 1.0)
                b.d1[e].value, b.d2[e].value = d1, d2
                b.th[e].value = float(chen(d1, d2))
                b.q[e].value = u.U * u.area * b.th[e].value
            for s, uu in b.util.items():
                hot, t_in, t_out, F, h = streams[s]
                uu["Q"].value = 0.3 * F * abs(t_in - t_out)
                if uu["unit"] is not None:
                    uu["re"].value = 0.0
                    uu["tb"].value = t_out
                    uu["A"].value = max(uu["unit"].area, 1.0)
                    thu = getattr(b, f"thu_{s}")
                    thu.value = 20.0

    def _extract(self, m, specs, case, flow, K) -> dict:
        R, c = self.rules, self.costs
        val = lambda v: float(v.value) if v.value is not None else float("nan")
        plants_out = {}
        total_util = total_capital = 0.0
        total_trim = unused = 0.0  # capital with areas trimmed to what is used; capital of unused area
        violation = 0.0
        T_r, T_s = val(m.T_r), val(m.T_s)
        for pid, sp in specs.items():
            b = getattr(m, f"plant_{pid}")
            streams = m.info[pid]["streams"]
            units_out, cap, cap_trim = [], 0.0, 0.0
            n_bypass = 0
            for e, u in b.units.items():
                q = val(b.q[e])
                k = u.stage
                hin, cin = val(b.T[u.hot, k]), val(b.T[u.cold, k + 1])
                hout, cout = val(b.tb[u.hot, k, e]), val(b.tb[u.cold, k, e])
                fh, fc = val(b.f[u.hot, k, e]), val(b.f[u.cold, k, e])
                d1, d2 = hin - cout, hout - cin
                violation += max(R.dt_min - d1, 0.0) * min(fh * streams[u.hot][3], fc * streams[u.cold][3])
                violation += max(R.dt_min - d2, 0.0) * min(fh * streams[u.hot][3], fc * streams[u.cold][3])
                violation += max(R.min_duty - q, 0.0)  # every unit the agent built is active (Eq. 8 / A37)
                a_used = val(b.A_used[e])
                if u.kind == "existing":
                    inst = self.installed[pid][u.unit]
                    X = u.area - inst
                    capital = (c.cf_unit + c.ca_exchanger * X ** c.beta) if X > 1e-9 else 0.0
                    x_used = max(a_used - inst, 0.0)
                    trimmed = (c.cf_unit + c.ca_exchanger * max(x_used, c.min_augmentation) ** c.beta) if x_used > R.area_tolerance else 0.0
                else:
                    capital = c.cf_loop + c.ca_loop * u.area ** c.beta
                    trimmed = c.cf_loop + c.ca_loop * max(a_used, 0.0) ** c.beta
                cap += capital
                cap_trim += min(trimmed, capital)
                unused += max(capital - trimmed, 0.0)
                units_out.append({"id": e, "kind": u.kind, "hot": u.hot, "cold": u.cold, "stage": k + 1, "duty": q,
                                  "area": u.area, "area_used": val(b.A_used[e]), "hot_in": hin, "hot_out": hout,
                                  "cold_in": cin, "cold_out": cout, "hot_fraction": fh, "cold_fraction": fc,
                                  "capital": capital})
            bypasses = []
            for (s, k) in b.r:
                r = val(b.r[s, k])
                if r > 1e-4:  # below: numerical zero, no bypass
                    n_bypass += 1
                    bypasses.append({"stream": s, "stage": k + 1, "fraction": r})
                    violation += max(R.min_bypass - r, 0.0) * streams[s][3] * R.dt_min  # Eq. 25-26
            util_cost = 0.0
            utilities = []
            for s, uu in b.util.items():
                hot, t_in, t_out, F, h = streams[s]
                Q = val(uu["Q"])
                violation += val(uu["over"]) + val(uu.get("short", uu["over"])) * (uu["unit"] is None)
                entry = {"stream": s, "duty": Q, "T_end": val(b.T[s, K] if hot else b.T[s, 0])}
                if uu["unit"] is None:
                    utilities.append(entry)
                    continue
                price = uu["utility"].price
                if Q > 1e-6:
                    violation += max(R.min_duty - Q, 0.0)
                    util_cost += Q * price
                    A = val(uu["A"])
                    re = val(uu["re"])
                    inst = uu["unit"].area
                    violation += val(uu["under"])
                    X = max(A - inst, c.min_augmentation) if A > inst + R.area_tolerance else 0.0
                    capital = (c.cf_unit + (c.ca_cooler if hot else c.ca_heater) * X ** c.beta) if X > 0 else 0.0
                    cap += capital
                    if re > 1e-4:
                        n_bypass += 1
                        bypasses.append({"stream": s, "stage": "end", "fraction": re})
                        violation += max(R.min_bypass - re, 0.0) * F * R.dt_min
                    entry.update({"area": A, "bypass": re, "augmentation_area": X, "capital": capital,
                                  "annual_cost": Q * price})
                else:
                    entry.update({"area": 0.0, "bypass": 0.0, "augmentation_area": 0.0, "capital": 0.0, "annual_cost": 0.0})
                utilities.append(entry)
            cap_trim += cap - sum(u["capital"] for u in units_out)  # utilities and bypasses as they are
            cap += R.bypass_cost * n_bypass
            cap_trim += R.bypass_cost * n_bypass
            tac = util_cost + c.af * cap
            total_util += util_cost
            total_capital += cap
            total_trim += cap_trim
            lt = {k: val(b.T[LOOP, k]) for k in range(K + 1)}
            plants_out[pid] = {"plant": pid, "role": sp.role, "flow": sp.flow, "loop_in": lt[0] if sp.role == "sink" else lt[K],
                               "loop_out": lt[K] if sp.role == "sink" else lt[0], "TAC": tac, "utility_cost": util_cost,
                               "capital": cap, "capital_trimmed": cap_trim, "TAC_trimmed": util_cost + c.af * cap_trim,
                               "units": units_out, "utilities": utilities, "bypasses": bypasses}
            plants_out[pid]["duty"] = sp.flow * abs(plants_out[pid]["loop_in"] - plants_out[pid]["loop_out"])
        loop_capital = c.cim * flow
        total_capital += loop_capital
        total_trim += loop_capital
        tac = total_util + c.af * total_capital
        baseline = sum(self.baseline[p] for p in specs)
        penalty = R.slack_price * violation
        strict = violation <= R.strict_tolerance
        s = (baseline - tac - penalty) / self.site_baseline
        if not np.isfinite(s):
            s = -1e6
        return {"case": case.code, "TAC": tac, "TACS": baseline - tac, "penalty": penalty, "violation_kW": violation,
                # areas trimmed to what the operation uses (the paper sizes every unit exactly)
                "TAC_trimmed": total_util + c.af * total_trim, "unused_area_annual_cost": c.af * unused,
                "TAC_with_penalty": tac + penalty, "strictly_feasible": strict, "T_return": T_r, "T_supply": T_s,
                "loop_flow": flow, "loop_capital": loop_capital, "loop_annual_cost": c.af * loop_capital,
                "plants": plants_out, "reward": float(s if s >= 0 else s / (1.0 - s))}

    def evaluate(self, a, case: RoleCase) -> dict:
        specs, flow = self.decode(a, case)
        out = self.solve(specs, case, flow)
        out["action"] = list(map(float, np.asarray(a, float)))
        return out

    def evaluate_relabeled(self, a_h, case: RoleCase) -> dict:
        """Evaluate a hindsight action (same units, areas trimmed to what the previous
        evaluation used) starting from that evaluation's operating state, so the NLP
        polishes the same operation instead of searching for a new one."""
        import pyomo.environ as pe
        from pyomo.opt import SolverStatus, TerminationCondition as TC

        m0, specs0 = self._last
        specs, flow = self.decode(a_h, case)
        K = stages_for_case(self.plants, case)
        started = time.perf_counter()
        m = build_site_model(specs, self.costs, K, self.cfg)
        self._initialize(m, specs, K)
        old = {v.name: v.value for v in m0.component_data_objects(pe.Var)}
        for v in m.component_data_objects(pe.Var):
            if v.name in old and old[v.name] is not None:
                val = old[v.name]
                if v.lb is not None:
                    val = max(val, v.lb)
                if v.ub is not None:
                    val = min(val, v.ub)
                v.value = val
        solver = pe.SolverFactory("ipopt")
        solver.options.update({"tol": self.cfg.ipopt_tol, "max_iter": self.cfg.ipopt_max_iter,
                               "max_cpu_time": self.cfg.ipopt_max_cpu, "print_level": 0, "mu_strategy": "adaptive",
                               "bound_relax_factor": 0.0, "honor_original_bounds": "yes"})
        ok, term = False, "not solved"
        try:
            m.pk.set_value((tuple(self.cfg.penalty_schedule) or (self.cfg.penalty_k,))[-1])
            res = solver.solve(m, tee=False, load_solutions=False)
            ok = res.solver.status == SolverStatus.ok and res.solver.termination_condition in (TC.optimal, TC.locallyOptimal)
            if res.solver.termination_condition != TC.infeasible:
                m.solutions.load_from(res)
            term = str(res.solver.termination_condition)
        except Exception as exc:
            ok, term = False, f"error: {exc}"
        out = self._extract(m, specs, case, flow, K)
        out["solver"] = {"ok": ok, "termination": term, "seconds": time.perf_counter() - started, "start": "hindsight"}
        out["action"] = list(map(float, np.asarray(a_h, float)))
        if not ok:
            out["strictly_feasible"] = False
        if not all(np.isfinite(out[k]) for k in ("TAC", "TAC_with_penalty", "violation_kW")):
            out["strictly_feasible"] = False
            out["reward"] = -1.0
        return out

    def relabel_action(self, a, result: dict, case: RoleCase):
        """Hindsight action: the same action with every area entry set to the area the
        NLP operation actually used (``area_used``), so that buying it reproduces the
        operation at the trimmed cost.  Returns None when some used area cannot be
        represented (a new unit used below the dead zone, or a partial augmentation
        below the 1 m2 minimum)."""
        a = np.array(a, float)
        R, c = self.rules, self.costs
        entries = self.layout(case)
        index = {(k, p, key): i for i, (k, p, key) in enumerate(entries)}
        for pid, d in result["plants"].items():
            if "units" not in d:
                continue
            p = self.plants[pid]
            h = {s.id: s.h for s in p.streams.values()}
            h[LOOP] = c.loop_h
            used = {u["id"]: u for u in d["units"]}
            for e in p.exchangers.values():
                u = used.get(e.id)
                if u is None:
                    return None
                inst = self.installed[pid][e.id]
                x_used = u["area_used"] - inst
                i = index[("area", pid, e.id)]
                if x_used <= R.area_tolerance:
                    a[i] = -1.0
                else:
                    if x_used < c.min_augmentation:
                        return None
                    qmax = min(p.streams[e.hot].load, p.streams[e.cold].load)
                    U = 1.0 / (1.0 / h[e.hot] + 1.0 / h[e.cold])
                    xmax = max(c.min_augmentation, qmax / (U * R.approach_scale) - inst)
                    a[i] = min((x_used - c.min_augmentation) / max(xmax - c.min_augmentation, 1e-12), 1.0)
            for (k, q, key), i in index.items():
                if k != "loop" or q != pid:
                    continue
                s, stage = key
                u = used.get(f"LOOP-{s}@{stage + 1}")
                if u is None:
                    a[i] = -1.0  # not built (or dropped as useless)
                    continue
                st = p.streams[s]
                U = 1.0 / (1.0 / st.h + 1.0 / c.loop_h)
                amax = st.load / (U * R.approach_scale)
                if u["area_used"] <= 0.5:
                    return None
                frac = min(np.sqrt(u["area_used"] / amax), 1.0)
                a[i] = 2.0 * (R.area_dead_zone + (1.0 - R.area_dead_zone) * frac) - 1.0
        return a
