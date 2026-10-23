"""One plant of the Lo, Hsu & Chang (2021) retrofit as a *simulation* (Strategy I).

Superstructure (SI Part A, Sec. 2): ``K`` stages; hot streams pass stages 1..K, cold
streams K..1; every stage holds all candidate matches.  Strategy I (Sec. 4.1): the
process matches are the existing ones, each existing unit sits in exactly one stage
(Eq. 7-8) and keeps at least its installed area (Eq. 9); new units are the matches
between process streams and the intermediate stream (SI A19-A22).  Every process
stream has a bypass at every stage and at its end (A25-A36); intermediate streams
have none (A27-A28).  Coolers/heaters sit at the stream ends (A15-A16, A23-A24).

Variables of the agent (all in [-1, 1]):
    stage and augmentation of every existing unit; area of every candidate new unit
    (0 = absent); split fractions of every stream at every stage; bypass fractions.

Simulation (no optimizer): given the areas, the duties follow from the paper's own
area equation (A76, Chen's mean) solved as a rating problem: q = U A Chen(dt1, dt2).
The stage-wise network is countercurrent, so the duties are a fixed point, solved by
damped successive substitution.  Two deterministic rules keep every design feasible
and let it hit equality targets exactly, both things the paper's MINLP does by choice
of continuous variables:

* a unit never violates ``dt_min``: its duty is capped where an approach reaches
  ``dt_min``;
* a stream never overshoots its target (process streams: A46/A49; the intermediate
  stream: A52/A55): the duties of its last stages are clipped.

A capped new unit is simply bought smaller (its area is the area its duty needs); a
capped existing unit keeps its installed area and is throttled by the paper's bypass
on one of its streams (Eq. 9 then holds with equality), charged 500 USD like every
bypass (Eq. 18-21), with the 0.1 lower bound of Eq. 25-28.

Unmet intermediate duty (slack) and rule violations (30 kW minimum duty, streams
without a utility unit missing their target, approach deficits) are priced at
``slack_price`` USD/(kW yr), which exceeds any saving a kW can earn, so they never
belong to an optimum; only designs without them are reported as feasible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tshi_decomp.case import Costs, Plant
from tshi_decomp.plant import LOOP, Interface  # data types only; no solver is used

INACTIVE = "inactive"


def chen(a, b):
    """Chen's approximation of the log-mean temperature difference (SI A76)."""
    return np.cbrt(a * b * (a + b) / 2.0)


def reconciled_areas(plant: Plant, costs: Costs, dt_min: float, tolerance: float) -> dict:
    """Installed areas: the printed Fig. 2 area, replaced by the area the original duties
    require when the two agree within half a printed digit (``tolerance``), so that the
    original network operates exactly.  The original temperatures are walked along the
    Fig. 2 chains with duty-proportional branch fractions."""
    hot_in, hot_out, cold_in, cold_out = {}, {}, {}, {}
    ends = {}
    for sid, s in plant.streams.items():
        T, sign = s.t_in, (-1.0 if s.hot else 1.0)
        for group in plant.chains[sid]:
            total = sum(plant.exchangers[u].duty for u in group)
            for u in group:
                q = plant.exchangers[u].duty
                f = q / total if total > 0 else 1.0
                (hot_in if s.hot else cold_in)[u] = T
                (hot_out if s.hot else cold_out)[u] = T + sign * q / (f * s.fcp)
            T += sign * total / s.fcp
        ends[sid] = T
    areas = {}
    for e in plant.exchangers.values():
        U = 1.0 / (1.0 / plant.streams[e.hot].h + 1.0 / plant.streams[e.cold].h)
        d1, d2 = hot_in[e.id] - cold_out[e.id], hot_out[e.id] - cold_in[e.id]
        if min(d1, d2) < dt_min - 1e-6:
            raise ValueError(f"{plant.id}/{e.id}: original network violates the approach temperature")
        areas[e.id] = e.duty / (U * chen(d1, d2))
    for u in plant.utility_units.values():
        s, util = plant.streams[u.stream], plant.utilities[u.utility]
        T = ends[u.stream]
        d1, d2 = ((T - util.t_out, s.t_out - util.t_in) if s.hot else (util.t_in - s.t_out, util.t_out - T))
        U = 1.0 / (1.0 / s.h + 1.0 / util.h)
        areas[u.id] = u.duty / (U * chen(d1, d2))
    out = {}
    for uid, required in areas.items():
        printed = plant.exchangers[uid].area if uid in plant.exchangers else plant.utility_units[uid].area
        if required > printed + tolerance:
            raise ValueError(f"{plant.id}/{uid}: original duty needs {required:.3f} m2 > printed {printed}")
        out[uid] = required if abs(required - printed) <= tolerance else printed
    return out


@dataclass(frozen=True)
class SimConfig:
    # Lo et al. Sec. 6 bounds
    dt_min: float = 1.0  # K; not printed in the paper, inferred from Fig. 2 (case/data.json)
    min_duty: float = 30.0  # kW, LQ of every unit
    min_fraction: float = 0.1  # LRH/LRC/LRHM/LRCM...
    min_bypass: float = 0.1  # LRRH/LRRC/LRRHU/LRRCU
    bypass_cost: float = 500.0
    allow_new_utility_units: bool = False  # Strategy I/II: Eq. 2-3, 5-6
    # Ours
    slack_price: float = 2000.0  # USD/(kW yr) on unmet loop duty and rule violations
    area_tolerance: float = 0.05  # m2: half a printed digit of Fig. 2 areas
    area_dead_zone: float = 0.15  # share of a new unit's action range that means "absent"
    approach_scale: float = 1.2  # K: an area action of 1 is the full load at this mean approach
    duty_tolerance: float = 1e-3  # kW: smaller duties count as zero
    fp_tol: float = 1e-7  # fixed-point tolerance, relative to the largest stream load
    fp_max_iter: int = 100  # continuous phase; the snap phase gets half of it
    fp_damping: float = 0.5
    anderson_memory: int = 3  # Anderson acceleration depth (0 = plain damped iteration)
    rating_iters: int = 12  # Illinois steps of the rating (1e-12 kW against bisection)


def rating(UA, Thi, Tci, Ch, Cc, iters=16, q0=None):
    """Duty of countercurrent units: q = UA chen(dt1, dt2) with dt from the branch balances.

    Vectorized Illinois (modified regula falsi) on [0, Cmin (Thi - Tci)]; the bracket is
    tightened around a warm start ``q0`` when that still brackets the root.  0 where the
    inlets cross or UA = 0.  g is concave and decreasing in q, so the root is unique.
    """
    dT = Thi - Tci
    Ch = np.maximum(Ch, 1e-12)
    Cc = np.maximum(Cc, 1e-12)
    qmax = np.maximum(np.minimum(Ch, Cc) * dT, 0.0) * (UA > 0)

    def g(q):
        return UA * chen(np.maximum(dT - q / Cc, 0.0), np.maximum(dT - q / Ch, 0.0)) - q

    a = np.zeros_like(qmax)
    b = qmax.copy()
    fa = UA * dT
    fb = -qmax
    if q0 is not None:
        q0 = np.clip(q0, 0.0, qmax)
        delta = 0.02 * qmax
        a1, b1 = np.maximum(q0 - delta, 0.0), np.minimum(q0 + delta, qmax)
        fa1, fb1 = g(a1), g(b1)
        ok = (fa1 >= 0) & (fb1 <= 0)
        a, b = np.where(ok, a1, a), np.where(ok, b1, b)
        fa, fb = np.where(ok, fa1, fa), np.where(ok, fb1, fb)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iters):
            c = b - fb * (b - a) / (fb - fa)  # (a, b) is an unordered bracket
            c = np.where(np.isfinite(c), np.clip(c, np.minimum(a, b), np.maximum(a, b)), 0.5 * (a + b))
            fc = g(c)
            same = (fc < 0) == (fb < 0)  # c is on b's side: b <- c, Illinois halves fa
            a, fa = np.where(same, a, b), np.where(same, 0.5 * fa, fb)
            b, fb = c, fc
    return np.where(qmax > 0, b, 0.0)


def _bisect(f, lo, hi, iters=28):
    """Root of the vectorized decreasing function f on [lo, hi]; clipped to the bounds."""
    lo, hi = lo.copy(), hi.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = f(mid) > 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


class PlantLayout:
    """Candidate units, split groups and bypass slots of one plant in one role."""

    def __init__(self, plant: Plant, costs: Costs, role: str, stages: int, cfg: SimConfig | None = None):
        if role not in ("sink", "source", INACTIVE):
            raise ValueError(role)
        self.plant, self.costs, self.role, self.K = plant, costs, role, int(stages)
        self.cfg = cfg or SimConfig()
        self.loop_hot = role == "sink"
        self.streams = list(plant.streams) + ([LOOP] if role != INACTIVE else [])
        self.sidx = {s: i for i, s in enumerate(self.streams)}
        self.is_hot = np.array([plant.streams[s].hot if s != LOOP else self.loop_hot for s in self.streams])
        self.process = np.array([s != LOOP for s in self.streams])
        self.S = len(self.streams)
        ex = []
        for e in plant.exchangers.values():
            ex.append({"id": e.id, "kind": "existing", "hot": e.hot, "cold": e.cold, "stage": None, "unit": e.id})
        if role == "sink":
            for j, s in plant.streams.items():
                if not s.hot:
                    for k in range(self.K):
                        ex.append({"id": f"LOOP-{j}@{k + 1}", "kind": "loop", "hot": LOOP, "cold": j, "stage": k})
        elif role == "source":
            for i, s in plant.streams.items():
                if s.hot:
                    for k in range(self.K):
                        ex.append({"id": f"LOOP-{i}@{k + 1}", "kind": "loop", "hot": i, "cold": LOOP, "stage": k})
        self.ex = ex
        self.E = len(ex)
        self.ex_hot = np.array([self.sidx[e["hot"]] for e in ex], int)
        self.ex_cold = np.array([self.sidx[e["cold"]] for e in ex], int)
        self.existing = [i for i, e in enumerate(ex) if e["kind"] == "existing"]
        self.loop_units = [i for i, e in enumerate(ex) if e["kind"] == "loop"]
        self.fixed_stage = np.array([e["stage"] if e["stage"] is not None else -1 for e in ex], int)
        h = {s: plant.streams[s].h for s in plant.streams}
        h[LOOP] = costs.loop_h
        self.U = np.array([1.0 / (1.0 / h[e["hot"]] + 1.0 / h[e["cold"]]) for e in ex])
        # ---- action layout -----------------------------------------------------------
        names, col = [], {}
        for i in self.existing:
            col[("stage", i)] = len(names)
            names.append(f"stage:{ex[i]['id']}")
            col[("aug", i)] = len(names)
            names.append(f"aug:{ex[i]['id']}")
        for i in self.loop_units:
            col[("area", i)] = len(names)
            names.append(f"area:{ex[i]['id']}")
        # Split groups: candidates of a stream at a stage are its existing units (any stage)
        # and the new units of that stage; n - 1 stick-breaking actions for n candidates.
        self.groups = []
        for s in self.streams:
            si = self.sidx[s]
            for k in range(self.K):
                cands = [i for i, e in enumerate(ex) if (e["hot"] == s or e["cold"] == s)
                         and (e["kind"] == "existing" or e["stage"] == k)]
                if len(cands) >= 2:
                    cols = []
                    for c in cands[:-1]:
                        cols.append(len(names))
                        names.append(f"split:{s}@{k + 1}:{ex[c]['id']}")
                    self.groups.append((si, k, cands, cols))
        self.bypass_col = {}
        for s in self.streams:
            if s == LOOP:
                continue
            si = self.sidx[s]
            for k in range(self.K):
                self.bypass_col[(si, k)] = len(names)
                names.append(f"bypass:{s}@{k + 1}")
            self.bypass_col[(si, "end")] = len(names)
            names.append(f"bypass:{s}@end")
        self.names, self.col, self.dim = names, col, len(names)


class PlantSimulator:
    """Vectorized rating and costing of one plant's designs at a fixed interface."""

    def __init__(self, layout: PlantLayout):
        self.L = layout
        plant, costs, cfg = layout.plant, layout.costs, layout.cfg
        self.plant, self.costs, self.cfg = plant, costs, cfg
        # Installed areas reconciled with the printed Fig. 2 values.
        self.installed_area = reconciled_areas(plant, costs, cfg.dt_min, cfg.area_tolerance)
        L = layout
        self.F = np.array([plant.streams[s].fcp if s != LOOP else 0.0 for s in L.streams])
        self.T_in = np.array([plant.streams[s].t_in if s != LOOP else 0.0 for s in L.streams])
        self.T_out = np.array([plant.streams[s].t_out if s != LOOP else 0.0 for s in L.streams])
        self.area_ex = np.array([self.installed_area[e["unit"]] if e["kind"] == "existing" else 0.0 for e in L.ex])
        self.ca = np.array([costs.ca_loop if e["kind"] == "loop" else costs.ca_exchanger for e in L.ex])
        self.cf = np.array([costs.cf_loop if e["kind"] == "loop" else costs.cf_unit for e in L.ex])
        # Utility at every process stream end: the existing unit's utility, else (if allowed)
        # the cheapest feasible one.
        self.utility = []
        for s in L.streams:
            if s == LOOP:
                continue
            st = plant.streams[s]
            unit = next((u for u in plant.utility_units.values() if u.stream == s), None)
            if unit is not None:
                util = plant.utilities[unit.utility]
            else:
                options = [u for u in plant.utilities.values() if u.kind == ("cold" if st.hot else "hot") and
                           ((u.t_in + cfg.dt_min <= st.t_out) if st.hot else (u.t_in >= st.t_out + cfg.dt_min))]
                util = min(options, key=lambda u: u.price) if options else None
            self.utility.append({
                "stream": s, "si": L.sidx[s], "hot": st.hot, "unit": unit, "utility": util,
                "U": 1.0 / (1.0 / st.h + 1.0 / util.h) if util else 0.0,
                "area_existing": self.installed_area[unit.id] if unit else 0.0,
                "ca": costs.ca_cooler if st.hot else costs.ca_heater})

    # ---------------------------------------------------------------------------------
    def interfaces(self, interface, N):
        """Loop flow and temperatures per design: one ``Interface`` or a sequence of N."""
        if isinstance(interface, Interface):
            seq = [interface] * N
        else:
            seq = list(interface)
            if len(seq) != N:
                raise ValueError(f"{len(seq)} interfaces for {N} designs")
        for itf in seq:
            if itf.role != self.L.role:
                raise ValueError(f"interface role {itf.role} != layout role {self.L.role}")
        return (np.array([i.flow for i in seq]), np.array([i.t_in for i in seq]), np.array([i.t_out for i in seq]))

    def decode(self, actions, interface) -> dict:
        """Physical design variables of N action vectors (interface sets the loop stream)."""
        L, cfg, c = self.L, self.cfg, self.costs
        a = np.clip(np.atleast_2d(np.asarray(actions, float)), -1.0, 1.0)
        N = a.shape[0]
        if a.shape[1] != L.dim:
            raise ValueError(f"{self.plant.id}/{L.role}: action must have {L.dim} entries, got {a.shape[1]}")
        F, T_in, T_out = (np.tile(v, (N, 1)) for v in (self.F, self.T_in, self.T_out))
        if L.role != INACTIVE:
            li = L.sidx[LOOP]
            F[:, li], T_in[:, li], T_out[:, li] = self.interfaces(interface, N)
        load = F * np.abs(T_in - T_out)  # (N, S)
        stage = np.tile(L.fixed_stage, (N, 1))
        area = np.zeros((N, L.E))
        aug = np.zeros((N, L.E))
        active = np.zeros((N, L.E), bool)
        for i in L.existing:
            u = (a[:, L.col[("stage", i)]] + 1.0) / 2.0
            stage[:, i] = np.minimum((u * L.K).astype(int), L.K - 1)
            qmax = np.minimum(load[:, L.ex_hot[i]], load[:, L.ex_cold[i]])
            xmax = np.maximum(c.min_augmentation, qmax / (L.U[i] * cfg.approach_scale) - self.area_ex[i])
            x = a[:, L.col[("aug", i)]]
            aug[:, i] = np.where(x > 0, c.min_augmentation + x * (xmax - c.min_augmentation), 0.0)
            area[:, i] = self.area_ex[i] + aug[:, i]
            active[:, i] = True
        # Eq. 7: one existing unit of a match per stage; a colliding unit takes the next stage.
        pairs = {}
        for i in L.existing:
            pairs.setdefault((L.ex[i]["hot"], L.ex[i]["cold"]), []).append(i)
        for units in pairs.values():
            for n, i in enumerate(units[1:], 1):
                for _ in range(L.K):
                    clash = np.zeros(N, bool)
                    for j in units[:n]:
                        clash |= stage[:, j] == stage[:, i]
                    if not clash.any():
                        break
                    stage[:, i] = np.where(clash, (stage[:, i] + 1) % L.K, stage[:, i])
        for i in L.loop_units:
            u = (a[:, L.col[("area", i)]] + 1.0) / 2.0
            on = u > cfg.area_dead_zone
            qmax = np.minimum(load[:, L.ex_hot[i]], load[:, L.ex_cold[i]])
            amax = qmax / (L.U[i] * cfg.approach_scale)
            area[:, i] = np.where(on, amax * ((u - cfg.area_dead_zone) / (1.0 - cfg.area_dead_zone)) ** 2, 0.0)
            active[:, i] = on & (area[:, i] > 0)
        bypass = np.zeros((N, L.S, L.K + 1))  # agent's bypass fractions; [:, :, K] = stream end
        for (si, k), col in L.bypass_col.items():
            kk = L.K if k == "end" else k
            x = a[:, col]
            bypass[:, si, kk] = np.where(x > 0, cfg.min_bypass + x * (0.9 - cfg.min_bypass), 0.0)
        split = {g: (a[:, cols] + 1.0) / 2.0 for g, (_, _, _, cols) in enumerate(L.groups)}
        return {"N": N, "F": F, "T_in": T_in, "T_out": T_out, "load": load, "stage": stage, "area": area,
                "aug": aug, "active": active, "bypass": bypass, "split": split, "interface": interface}

    # ---------------------------------------------------------------------------------
    def _fractions(self, D):
        """Flow fraction of every stream through every unit (A25-A28), from the agent's
        split and bypass actions; the stage bypass is capped so that every active match
        keeps its 0.1 minimum.  Returns f_hot, f_cold (N, E) and the stage bypass (N, S, K)."""
        L, cfg = self.L, self.cfg
        N = D["N"]
        stage, active = D["stage"], D["active"]
        lb = cfg.min_fraction
        f_hot = np.zeros((N, L.E))
        f_cold = np.zeros((N, L.E))
        r_stage = np.zeros((N, L.S, L.K))
        grouped = {(si, k): (cands, g) for g, (si, k, cands, _) in enumerate(L.groups)}
        for si in range(L.S):
            on_side = (L.ex_hot == si) if L.is_hot[si] else (L.ex_cold == si)
            members = [i for i in range(L.E) if on_side[i]]
            for k in range(L.K):
                here = {i: active[:, i] & (stage[:, i] == k) for i in members}
                m = sum(here[i].astype(int) for i in members) if members else np.zeros(N, int)
                r = D["bypass"][:, si, k] if L.process[si] else np.zeros(N)
                r = np.where(m > 0, np.minimum(r, np.maximum(1.0 - m * lb, 0.0)), 0.0)
                r_stage[:, si, k] = r
                rest = 1.0 - r
                lbm = np.where(m >= 2, lb, 0.0)
                R = rest - m * lbm  # length of the stick after the lower bounds
                if (si, k) in grouped:
                    cands, g = grouped[(si, k)]
                    u = D["split"][g]
                    # Stick-breaking over the active candidates in order; the last active one
                    # takes the remainder, so the last candidate needs no action.
                    later = [np.zeros(N, bool)]
                    for n in range(len(cands) - 1, 0, -1):
                        later.append(later[-1] | here[cands[n]])
                    later = later[::-1]  # later[n]: an active candidate follows candidate n
                    for n, i in enumerate(cands):
                        is_last = here[i] & ~later[n]
                        take = np.where(is_last, R, R * u[:, n]) if n < len(cands) - 1 else R
                        self._set(f_hot, f_cold, i, si, here[i], lbm + take)
                        R = np.where(here[i] & ~is_last, R - take, R)
                else:
                    for i in members:
                        self._set(f_hot, f_cold, i, si, here[i], rest)
        return f_hot, f_cold, r_stage

    def _set(self, f_hot, f_cold, i, si, mask, f):
        """Fraction of stream si through unit i, for the designs where i sits at this stage."""
        target = f_hot if self.L.ex_hot[i] == si else f_cold
        target[:, i] = np.where(mask, f, target[:, i])

    def _temps(self, q, D):
        """Stage inlet temperatures of every unit and stream end temperatures from the duties."""
        L = self.L
        N = D["N"]
        stage = D["stage"]
        Thi = np.zeros((N, L.E))
        Tci = np.zeros((N, L.E))
        t_end = np.zeros((N, L.S))
        F = np.maximum(D["F"], 1e-12)
        rows = np.arange(N)
        for si in range(L.S):
            hot = L.is_hot[si]
            on = (L.ex_hot == si) if hot else (L.ex_cold == si)
            idx = np.flatnonzero(on)
            Qk = np.zeros((N, L.K))
            for k in range(L.K):
                for i in idx:
                    Qk[:, k] += np.where(stage[:, i] == k, q[:, i], 0.0)
            before = np.cumsum(Qk, axis=1) - Qk  # duty at stages < k
            total = Qk.sum(axis=1)
            after = total[:, None] - before - Qk  # duty at stages > k
            for i in idx:
                k = stage[:, i]
                if hot:
                    Thi[:, i] = D["T_in"][:, si] - before[rows, k] / F[:, si]
                else:
                    Tci[:, i] = D["T_in"][:, si] + after[rows, k] / F[:, si]
            t_end[:, si] = D["T_in"][:, si] + (-1.0 if hot else 1.0) * total / F[:, si]
        return Thi, Tci, t_end

    def _step(self, q, D, fr, snap=True, warm=True):
        """One evaluation of the fixed-point map: duties -> temperatures -> rated, capped,
        clipped and realized duties.  Returns the new duties and the realization.

        ``snap=False`` leaves out the 0.1 lower bound of a throttling bypass (Eq. 25-26),
        which makes the map continuous; the rule is applied from the continuous solution.
        """
        L, cfg = self.L, self.cfg
        N = D["N"]
        f_hot, f_cold, r_stage = fr
        dt = cfg.dt_min
        Thi, Tci, t_end = self._temps(q, D)
        Ch = f_hot * D["F"][:, L.ex_hot]
        Cc = f_cold * D["F"][:, L.ex_cold]
        UA = L.U * D["area"] * D["active"]
        q_rate = rating(UA, Thi, Tci, Ch, Cc, cfg.rating_iters, q if warm else None)
        cap_dt = np.maximum(np.minimum(Ch, Cc) * (Thi - Tci - dt), 0.0)
        q_c = np.minimum(q_rate, cap_dt)
        # Stream targets (A46, A49, A52, A55): clip the last stages, new units first.
        for si in list(np.flatnonzero(L.process)) + list(np.flatnonzero(~L.process)):
            hot = L.is_hot[si]
            on = (L.ex_hot == si) if hot else (L.ex_cold == si)
            idx = np.flatnonzero(on)
            if not len(idx):
                continue
            excess = np.maximum(q_c[:, idx].sum(axis=1) - D["load"][:, si], 0.0)
            order = range(L.K - 1, -1, -1) if hot else range(L.K)
            for k in order:
                for kind in ("loop", "existing"):
                    for i in idx:
                        if L.ex[i]["kind"] != kind:
                            continue
                        here = D["stage"][:, i] == k
                        red = np.where(here, np.minimum(q_c[:, i], excess), 0.0)
                        q_c[:, i] -= red
                        excess -= red
        # Realization of existing units below their rating: throttle by a stream bypass.
        q_new = q_c.copy()
        f_real_hot, f_real_cold = f_hot.copy(), f_cold.copy()
        deficit = np.zeros((N, L.E))  # K of approach missing after the best bypass
        side = np.full((N, L.E), -1, int)  # 0: hot-side bypass, 1: cold-side, -1: none
        ex_idx = np.array(L.existing, int)
        if len(ex_idx):
            # An existing unit below its rating is throttled, unless the area its duty needs
            # is within the printed-area tolerance of the installed area (an exact fit).
            qe = q_c[:, ex_idx]
            Che, Cce = Ch[:, ex_idx], Cc[:, ex_idx]
            d1 = Thi[:, ex_idx] - (Tci[:, ex_idx] + qe / np.maximum(Cce, 1e-12))
            d2 = (Thi[:, ex_idx] - qe / np.maximum(Che, 1e-12)) - Tci[:, ex_idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                a_fit = np.where(qe > 0, qe / (L.U[ex_idx] * chen(np.maximum(d1, 1e-9), np.maximum(d2, 1e-9))), 0.0)
            need = (qe < q_rate[:, ex_idx] - cfg.duty_tolerance) & (a_fit < D["area"][:, ex_idx] - cfg.area_tolerance)
            rows, cols = np.nonzero(need)
            if len(rows):
                units = ex_idx[cols]
                q_t = q_c[rows, units]
                UAe = UA[rows, units]
                Th, Tc = Thi[rows, units], Tci[rows, units]
                Fh, Fc = D["F"][rows, L.ex_hot[units]], D["F"][rows, L.ex_cold[units]]
                fh0, fc0 = f_hot[rows, units], f_cold[rows, units]
                kk = D["stage"][rows, units]
                lb = cfg.min_fraction
                results = []
                for s in (0, 1):
                    def g(f, s=s):
                        """q_t minus what the installed area transfers at duty q_t and fraction f
                        (decreasing in f: a deeper bypassed branch lowers the mean approach)."""
                        ch = (f if s == 0 else fh0) * Fh
                        cc = (fc0 if s == 0 else f) * Fc
                        d1 = Th - (Tc + q_t / np.maximum(cc, 1e-12))
                        d2 = (Th - q_t / np.maximum(ch, 1e-12)) - Tc
                        return q_t - UAe * chen(np.maximum(d1, 0.0), np.maximum(d2, 0.0))
                    f0 = fh0 if s == 0 else fc0
                    f = np.minimum(_bisect(g, np.full_like(f0, lb), f0), f0)
                    # Eq. 25-26: a bypass is 0 or at least 0.1 of the stream.
                    si = (L.ex_hot if s == 0 else L.ex_cold)[units]
                    r0 = r_stage[rows, si, kk]
                    r_tot = r0 + (f0 - f)
                    short = (r_tot > 1e-12) & (r_tot < cfg.min_bypass) & snap
                    f = np.where(short, np.maximum(f0 - (cfg.min_bypass - r0), lb), f)
                    ch = (f if s == 0 else fh0) * Fh
                    cc = (fc0 if s == 0 else f) * Fc
                    qs = rating(UAe, Th, Tc, ch, cc, cfg.rating_iters, q_t)
                    d1 = Th - (Tc + qs / np.maximum(cc, 1e-12))
                    d2 = (Th - qs / np.maximum(ch, 1e-12)) - Tc
                    results.append((f, qs, np.maximum(dt - np.minimum(d1, d2), 0.0)))
                (fh1, qh, dh), (fc1, qc_, dc) = results
                use_hot = dh <= dc
                q_new[rows, units] = np.where(use_hot, qh, qc_)
                f_real_hot[rows, units] = np.where(use_hot, fh1, fh0)
                f_real_cold[rows, units] = np.where(use_hot, fc0, fc1)
                deficit[rows, units] = np.where(use_hot, dh, dc)
                side[rows, units] = np.where(use_hot, 0, 1)
        return q_new, {"Thi": Thi, "Tci": Tci, "t_end": t_end, "q_rate": q_rate, "cap_dt": cap_dt,
                       "f_hot": f_real_hot, "f_cold": f_real_cold, "deficit": deficit, "side": side,
                       "r_stage": r_stage}

    def solve(self, D):
        """Fixed point of the duties: Anderson-accelerated successive substitution with a
        damped fallback for designs whose residual grows (the map has discrete rules)."""
        cfg = self.cfg
        fr = self._fractions(D)
        N, E = D["N"], self.L.E
        q = np.zeros((N, E))
        scale = max(float(D["load"].max()), 1.0)
        total = 0
        m = cfg.anderson_memory
        # Phase 1: the continuous map (no 0.1 bypass snap); phase 2: the paper's rule from there.
        for snap, budget in ((False, cfg.fp_max_iter), (True, cfg.fp_max_iter // 2)):
            converged = np.zeros(N, bool)
            omega = np.full(N, cfg.fp_damping)  # per-design damping, halved when the residual grows
            last = np.full(N, np.inf)
            dR = np.zeros((N, E, m))
            dG = np.zeros((N, E, m))
            hlen = np.zeros(N, int)
            accelerate = np.full(N, m > 0)  # a design whose residual ever grows falls back for good
            qcap = np.minimum(D["load"][:, self.L.ex_hot], D["load"][:, self.L.ex_cold])
            r_prev = G_prev = None
            growths = np.zeros(N, int)
            for it in range(budget):
                G, aux = self._step(q, D, fr, snap, warm=it > 0)
                r = G - q
                res = np.abs(r).max(axis=1)
                converged = res <= cfg.fp_tol * scale
                total += 1
                if converged.all():
                    break
                grew = res > last
                accelerate &= ~grew
                growths = np.where(grew, growths + 1, 0)
                omega = np.where(growths >= 2, np.maximum(0.5 * omega, 0.1), omega)
                q_ss = q + omega[:, None] * r
                if r_prev is not None and m > 0:
                    dR = np.roll(dR, 1, axis=2)
                    dG = np.roll(dG, 1, axis=2)
                    dR[:, :, 0] = r - r_prev
                    dG[:, :, 0] = G - G_prev
                    hlen = np.minimum(hlen + 1, m)
                    mask = (np.arange(m)[None, :] < hlen[:, None]).astype(float)  # (N, m)
                    Rm = dR * mask[:, None, :]
                    Gm = dG * mask[:, None, :]
                    A = np.einsum("nem,nek->nmk", Rm, Rm)
                    reg = 1e-10 * np.maximum(np.einsum("nmm->n", A), 1e-12)
                    A = A + reg[:, None, None] * np.eye(m)[None]
                    b = np.einsum("nem,ne->nm", Rm, r)
                    gamma = np.linalg.solve(A, b[:, :, None])[:, :, 0]
                    q_and = np.clip(G - np.einsum("nek,nk->ne", Gm, gamma), 0.0, qcap)
                    use = accelerate & np.isfinite(q_and).all(axis=1)
                    q = np.where(use[:, None], q_and, q_ss)
                else:
                    q = q_ss
                q = np.where(converged[:, None], G, q)  # converged designs stay put
                r_prev, G_prev, last = r, G, res
        q_new, aux = self._step(q, D, fr)  # realization consistent with the final duties
        aux["iterations"] = total
        aux["residual"] = np.abs(q_new - q).max(axis=1)
        aux["converged"] = aux["residual"] <= cfg.fp_tol * scale
        aux["f_agent"] = fr[:2]
        return q_new, aux

    # ---------------------------------------------------------------------------------
    def evaluate(self, actions, interface: Interface) -> dict:
        """Rate and cost N designs.  All arrays are per design."""
        L, cfg, c = self.L, self.cfg, self.costs
        D = self.decode(actions, interface)
        N = D["N"]
        q, aux = self.solve(D)
        # A new unit that ends up without duty (crossed inlets, clipped away) is not bought
        # and carries no flow: drop it and re-solve with the fractions of the remaining units.
        is_loop_unit = np.array([e["kind"] == "loop" for e in L.ex])
        for _ in range(3):
            drop = D["active"] & is_loop_unit & (q <= cfg.duty_tolerance)
            if not drop.any():
                break
            D["active"] &= ~drop
            D["area"] = np.where(drop, 0.0, D["area"])
            q, aux = self.solve(D)
        Thi, Tci, t_end = self._temps(q, D)
        f_hot, f_cold = aux["f_hot"], aux["f_cold"]
        Ch = np.maximum(f_hot * D["F"][:, L.ex_hot], 1e-12)
        Cc = np.maximum(f_cold * D["F"][:, L.ex_cold], 1e-12)
        on = q > cfg.duty_tolerance
        hot_out = np.where(on, Thi - q / Ch, Thi)
        cold_out = np.where(on, Tci + q / Cc, Tci)
        d1 = np.where(on, Thi - cold_out, cfg.dt_min)
        d2 = np.where(on, hot_out - Tci, cfg.dt_min)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_req = np.where(on, q / (L.U * chen(np.maximum(d1, 1e-9), np.maximum(d2, 1e-9))), 0.0)
        violation = np.zeros(N)
        # Units: new units are bought at their required area; existing ones keep their
        # installed (+ augmented) area, with the bypass that realizes their duty.
        cap_unit = np.zeros(N)
        cap_bypass = np.zeros(N)
        is_loop = np.array([e["kind"] == "loop" for e in L.ex])
        area_bought = np.where(is_loop, a_req, 0.0)
        cap_unit += np.where(on & is_loop, c.cf_loop + c.ca_loop * np.maximum(area_bought, 0.0) ** c.beta, 0.0).sum(axis=1)
        violation += np.where(on & is_loop & (q < cfg.min_duty), cfg.min_duty - q, 0.0).sum(axis=1)
        ex_idx = np.array(L.existing, int)
        if len(ex_idx):
            x = D["aug"][:, ex_idx]
            cap_unit += np.where(x > 0, c.cf_unit + c.ca_exchanger * x ** c.beta, 0.0).sum(axis=1)
            violation += np.maximum(cfg.min_duty - q[:, ex_idx], 0.0).sum(axis=1)  # every existing match is kept
            # approach deficits a bypass could not remove, as kW at the unit's smaller capacity
            violation += (aux["deficit"][:, ex_idx] * np.minimum(Ch, Cc)[:, ex_idx]).sum(axis=1)
            # Eq. 9: the duty must use at least the installed area.
            short = np.maximum(D["area"][:, ex_idx] - a_req[:, ex_idx] - cfg.area_tolerance, 0.0)
            violation += (short * L.U[ex_idx] * cfg.dt_min).sum(axis=1)
        # Bypasses at the stages: the agent's plus those throttling existing units.
        r_stage = aux["r_stage"].copy()
        fa_hot, fa_cold = aux["f_agent"]
        rows = np.arange(N)
        for i in ex_idx:
            kk = D["stage"][:, i]
            r_stage[rows, L.ex_hot[i], kk] += np.where(aux["side"][:, i] == 0, fa_hot[:, i] - f_hot[:, i], 0.0)
            r_stage[rows, L.ex_cold[i], kk] += np.where(aux["side"][:, i] == 1, fa_cold[:, i] - f_cold[:, i], 0.0)
        cap_bypass += cfg.bypass_cost * (r_stage[:, L.process, :] > 1e-12).sum(axis=(1, 2))
        # Utilities at the stream ends (A15-A16, A23-A24, A35-A36, A68-A75).
        util_hot = np.zeros(N)
        util_cold = np.zeros(N)
        cap_util = np.zeros(N)
        utilities = []
        for row in self.utility:
            si = row["si"]
            hot = row["hot"]
            T = t_end[:, si]
            Fs = D["F"][:, si]
            R = Fs * ((T - D["T_out"][:, si]) if hot else (D["T_out"][:, si] - T))
            violation += np.maximum(-R, 0.0)  # overshoot of the target (A46/A49)
            R = np.maximum(R, 0.0)
            on_u = R > cfg.duty_tolerance
            u = row["utility"]
            entry = {"stream": row["stream"], "duty": np.where(on_u, R, 0.0)}
            if u is None or (row["unit"] is None and not cfg.allow_new_utility_units):
                violation += np.where(on_u, R, 0.0)  # no unit may serve this stream (Eq. 2-6)
                entry.update({"area": np.zeros(N), "bypass": np.zeros(N), "aug": np.zeros(N),
                              "capital": np.zeros(N), "branch_out": T, "unit": None})
                utilities.append(entry)
                continue
            violation += np.where(on_u & (R < cfg.min_duty), cfg.min_duty - R, 0.0)
            if hot:
                util_cold += np.where(on_u, R, 0.0) * u.price
            else:
                util_hot += np.where(on_u, R, 0.0) * u.price
            r_end = D["bypass"][:, si, L.K]
            # Branch outlet with a bypass r (A35/A36) and the approach that bounds r.
            t_target = D["T_out"][:, si]
            limit = (u.t_in + cfg.dt_min) if hot else (u.t_in - cfg.dt_min)

            def area_at(r, T=T, hot=hot, R=R, u=u, t_target=t_target, row=row):
                branch = (t_target - r * T) / np.maximum(1.0 - r, 1e-9)
                da = (T - u.t_out) if hot else (u.t_in - branch)
                db = (branch - u.t_in) if hot else (u.t_out - T)
                return R / (row["U"] * chen(np.maximum(da, 1e-9), np.maximum(db, 1e-9))), np.minimum(da, db)

            reach = (T > t_target + 1e-9) if hot else (T < t_target - 1e-9)
            r_max = np.where(reach, np.clip((t_target - limit) / np.where(reach, T - limit, 1.0), 0.0, 0.9), 0.0)
            r_max = np.where(on_u, r_max, 0.0)
            a0, _ = area_at(np.minimum(r_end, r_max))
            if row["unit"] is not None:
                a_ex = row["area_existing"]
                # Eq. 10-11: an existing unit needs at least its area; a bypass deepens the branch.
                under = on_u & (a0 < a_ex - cfg.area_tolerance)
                lo = np.maximum(np.where(r_end > 0, r_end, cfg.min_bypass), 0.0)
                r_fix = _bisect(lambda r: a_ex - area_at(r)[0], np.minimum(lo, r_max), r_max)
                r = np.where(under, np.maximum(r_fix, np.minimum(lo, r_max)), np.minimum(r_end, r_max))
                a_fin, d_fin = area_at(r)
                a_fin = np.where(on_u, a_fin, 0.0)
                violation += np.where(on_u, np.maximum(cfg.dt_min - d_fin, 0.0) * Fs, 0.0)
                violation += np.where(on_u, np.maximum(a_ex - a_fin - cfg.area_tolerance, 0.0) * row["U"] * cfg.dt_min, 0.0)
                x = np.where(on_u & (a_fin > a_ex + cfg.area_tolerance), np.maximum(a_fin - a_ex, c.min_augmentation), 0.0)
                capital = np.where(x > 0, c.cf_unit + row["ca"] * x ** c.beta, 0.0)
                cap_util += capital
                byp = on_u & (r > 1e-12)
                cap_bypass += cfg.bypass_cost * byp
                entry.update({"area": a_fin, "bypass": np.where(byp, r, 0.0), "aug": x, "capital": capital,
                              "branch_out": (t_target - r * T) / np.maximum(1.0 - r, 1e-9), "unit": row["unit"].id})
            else:
                r = np.minimum(r_end, r_max)
                a_fin, d_fin = area_at(r)
                a_fin = np.where(on_u, a_fin, 0.0)
                violation += np.where(on_u, np.maximum(cfg.dt_min - d_fin, 0.0) * Fs, 0.0)
                capital = np.where(on_u, c.cf_unit + row["ca"] * a_fin ** c.beta, 0.0)
                cap_util += capital
                byp = on_u & (r > 1e-12)
                cap_bypass += cfg.bypass_cost * byp
                entry.update({"area": a_fin, "bypass": np.where(byp, r, 0.0), "aug": np.zeros(N), "capital": capital,
                              "branch_out": (t_target - r * T) / np.maximum(1.0 - r, 1e-9), "unit": None})
            utilities.append(entry)
        slack = np.zeros(N)
        if L.role != INACTIVE:
            li = L.sidx[LOOP]
            loop_on = (L.ex_hot == li) | (L.ex_cold == li)
            slack = np.maximum(D["load"][:, li] - q[:, loop_on].sum(axis=1), 0.0)
        # An unconverged fixed point is inconsistent by its residual (kW).
        violation += np.where(aux["converged"], 0.0, aux["residual"])
        capital = cap_unit + cap_util + cap_bypass
        # Defensive bounds: a violation beyond three times the largest load is already a
        # hopeless design; keep the penalty finite and comparable.
        violation = np.minimum(np.nan_to_num(violation, nan=np.inf, posinf=np.inf), 3.0 * D["load"].max(axis=1))
        penalty = cfg.slack_price * (slack + violation)
        tac = np.nan_to_num(util_hot + util_cold + c.af * capital + penalty, nan=1e9, posinf=1e9)
        return {"tac": tac, "tac_clean": tac - penalty, "penalty": penalty, "slack": slack, "violation": violation,
                "strict": (slack <= 1e-6) & (violation <= 1e-6 * max(float(D["load"].max()), 1.0)) & aux["converged"],
                "q": q, "area_required": a_req, "area_installed": D["area"], "aug": D["aug"], "stage": D["stage"],
                "hot_in": Thi, "hot_out": hot_out, "cold_in": Tci, "cold_out": cold_out, "f_hot": f_hot,
                "f_cold": f_cold, "side": aux["side"], "t_end": t_end, "r_stage": r_stage, "utilities": utilities,
                "util_hot": util_hot, "util_cold": util_cold, "cap_unit": cap_unit, "cap_util": cap_util,
                "cap_bypass": cap_bypass, "capital": capital, "iterations": aux["iterations"],
                "converged": aux["converged"], "D": D}

    # ---------------------------------------------------------------------------------
    def export(self, action, interface: Interface) -> dict:
        """Physical description of one design."""
        L, cfg, c = self.L, self.cfg, self.costs
        ev = self.evaluate(np.asarray(action, float)[None, :], [interface])
        D = ev["D"]
        exchangers = []
        for i, e in enumerate(L.ex):
            q = float(ev["q"][0, i])
            on = q > cfg.duty_tolerance
            if e["kind"] == "loop" and not on:
                continue
            ex = {"id": e["id"], "kind": e["kind"], "hot": e["hot"], "cold": e["cold"], "stage": int(D["stage"][0, i]) + 1,
                  "duty": q, "hot_in": float(ev["hot_in"][0, i]), "hot_out": float(ev["hot_out"][0, i]),
                  "cold_in": float(ev["cold_in"][0, i]), "cold_out": float(ev["cold_out"][0, i]),
                  "hot_fraction": float(ev["f_hot"][0, i]), "cold_fraction": float(ev["f_cold"][0, i]),
                  "U": float(L.U[i]), "area_required": float(ev["area_required"][0, i]),
                  "area_existing": float(self.area_ex[i])}
            if e["kind"] == "existing":
                x = float(D["aug"][0, i])
                ex.update({"housing_unit": e["unit"], "augmented": x > 0, "augmentation_area": x,
                           "area_installed": float(D["area"][0, i]),
                           "throttled_side": {0: "hot", 1: "cold"}.get(int(ev["side"][0, i])),
                           "capital": (c.cf_unit + c.ca_exchanger * x ** c.beta) if x > 0 else 0.0})
            else:
                ex.update({"augmented": False, "augmentation_area": 0.0,
                           "capital": c.cf_loop + c.ca_loop * ex["area_required"] ** c.beta})
            exchangers.append(ex)
        utilities = []
        for row, u in zip(self.utility, ev["utilities"]):
            duty = float(u["duty"][0])
            utilities.append({"stream": row["stream"], "kind": "cooler" if row["hot"] else "heater",
                              "utility": row["utility"].id if row["utility"] else None, "unit": u["unit"],
                              "duty": duty, "T_in": float(ev["t_end"][0, row["si"]]), "T_out": float(D["T_out"][0, row["si"]]),
                              "branch_out": float(u["branch_out"][0]), "bypass": float(u["bypass"][0]),
                              "area_required": float(u["area"][0]), "area_existing": row["area_existing"],
                              "augmentation_area": float(u["aug"][0]), "capital": float(u["capital"][0]),
                              "annual_cost": duty * (row["utility"].price if row["utility"] else 0.0)})
        bypasses = []
        for si in np.flatnonzero(L.process):
            for k in range(L.K):
                r = float(ev["r_stage"][0, si, k])
                if r > 1e-12:
                    bypasses.append({"stream": L.streams[si], "stage": k + 1, "fraction": r})
        for u in utilities:
            if u["bypass"] > 1e-12:
                bypasses.append({"stream": u["stream"], "stage": "end", "fraction": u["bypass"]})
        costs = {"hot_utility": float(ev["util_hot"][0]), "cold_utility": float(ev["util_cold"][0]),
                 "unit_capital": float(ev["cap_unit"][0]), "utility_capital": float(ev["cap_util"][0]),
                 "bypass_capital": float(ev["cap_bypass"][0]), "capital": float(ev["capital"][0]),
                 "annualized_capital": c.af * float(ev["capital"][0]), "penalty": float(ev["penalty"][0]),
                 "TAC": float(ev["tac"][0]), "TAC_without_penalty": float(ev["tac_clean"][0])}
        return {"plant": self.plant.id, "role": L.role, "stages": L.K,
                "interface": {"role": interface.role, "flow": interface.flow, "T_in": interface.t_in,
                              "T_out": interface.t_out, "duty": interface.duty},
                "action": dict(zip(L.names, map(float, np.asarray(action, float)))),
                "slack_kW": float(ev["slack"][0]), "violation_kW": float(ev["violation"][0]),
                "strictly_feasible": bool(ev["strict"][0]), "converged": bool(ev["converged"][0]),
                "iterations": int(ev["iterations"]), "exchangers": exchangers, "utilities": utilities,
                "bypasses": bypasses, "stream_end_temperatures": {s: float(ev["t_end"][0, L.sidx[s]]) for s in L.streams},
                "costs": costs, "TAC": costs["TAC"]}
