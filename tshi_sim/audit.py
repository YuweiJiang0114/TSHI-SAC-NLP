"""Independent audit of ``tshi_sim`` plant and site exports against the paper's constraints.

Only case inputs are imported; nothing from the simulator.  Every stream is re-walked
through the stages (A17–A36), every unit's approaches and area are recomputed (A56–A76),
the Strategy I equipment rules (Eq. 7–12, 22, 25–28; Sec. 6 bounds) and every cost term
are checked.  Temperatures and duties: 1e-6 absolute; areas and costs: 1e-6 relative.

    python -m tshi_sim.audit results/tshi_sim/s1/case3/seed1/best_design.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from tshi_decomp.case import load_case

LOOP = "LOOP"
RULES = {"dt_min": 1.0, "min_duty": 30.0, "min_fraction": 0.1, "min_bypass": 0.1, "bypass_cost": 500.0,
         "slack_price": 2000.0, "area_tolerance": 0.05, "duty_tolerance": 1e-3}


class Report:
    def __init__(self):
        self.failures = []
        self.residuals = {}

    def check(self, name, residual, allowed, detail):
        v = abs(float(residual))
        v = v if math.isfinite(v) else math.inf
        self.residuals[name] = max(self.residuals.get(name, 0.0), v)
        if v > allowed:
            self.failures.append(f"{name}: {detail} (residual={v:.9g}, allowed={allowed:.3g})")

    def rule(self, name, ok, detail):
        if not ok:
            self.failures.append(f"{name}: {detail}")

    def equal(self, name, a, b, detail, relative=False):
        tol = 1e-6 * max(1.0, abs(float(b))) if relative else 1e-6
        self.check(name, float(a) - float(b), tol, detail)

    def lower(self, name, value, bound, detail, allowance=1e-6):
        self.check(name, min(float(value) - float(bound), 0.0), allowance, detail)

    def output(self):
        return {"passed": not self.failures, "failures": self.failures, "residuals": self.residuals}


def chen(a, b):
    return (a * b * (a + b) / 2.0) ** (1.0 / 3.0)


def audit_plant(x: dict, plant, costs, rules: dict | None = None) -> dict:
    r = Report()
    R = dict(RULES, **(rules or {}))
    try:
        _audit_plant(x, plant, costs, r, R)
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as exc:
        r.rule("export.schema", False, f"{type(exc).__name__}: {exc}")
    return r.output()


def _audit_plant(x, plant, costs, r, R):
    dt, tol_a, tol_q = R["dt_min"], R["area_tolerance"], R["duty_tolerance"]
    r.rule("plant.id", x["plant"] == plant.id, "plant identity")
    if x.get("role") == "inactive" and "exchangers" not in x:
        r.equal("cost.inactive_TAC", x["TAC"], plant.baseline_utility_cost, "original network kept", relative=True)
        return
    K = int(x["stages"])
    itf = x["interface"]
    flow, tin, tout = float(itf["flow"]), float(itf["T_in"]), float(itf["T_out"])
    duty_itf = flow * abs(tin - tout)
    active = flow > 0 and duty_itf > 1e-9
    role = ("sink" if tin > tout else "source") if active else "inactive"
    r.rule("interface.role", itf["role"] == role == x["role"], f"{itf['role']} / {x['role']} / {role}")
    r.equal("interface.duty", itf["duty"], duty_itf, plant.id)
    streams = {s.id: (s.hot, s.t_in, s.t_out, s.fcp, s.h) for s in plant.streams.values()}
    if active:
        streams[LOOP] = (role == "sink", tin, tout, flow, costs.loop_h)
    ex = x["exchangers"]
    ids = [e["id"] for e in ex]
    r.rule("equipment.unique_ids", len(ids) == len(set(ids)), "exchanger IDs")
    counts = Counter(e.get("housing_unit") for e in ex if e["kind"] == "existing")
    for uid in plant.exchangers:  # Eq. 7-8: every existing unit houses its match in one stage
        r.rule("equipment.existing_unit_once", counts[uid] == 1, uid)
    bypass = {}
    for b in x["bypasses"]:
        key = (b["stream"], b["stage"])
        r.rule("bypass.unique", key not in bypass, str(key))
        bypass[key] = float(b["fraction"])
        r.rule("bypass.process_stream", b["stream"] in plant.streams, f"{key}: no bypass on the intermediate stream (A27-A28)")
        r.lower("rule.min_bypass", b["fraction"], R["min_bypass"], str(key))  # Eq. 25-28
        r.rule("bypass.range", 0 < b["fraction"] < 1, str(key))
    # ---- units -----------------------------------------------------------------------
    for e in ex:
        eid = e["id"]
        hot, cold = e["hot"], e["cold"]
        r.rule("equipment.streams", hot in streams and cold in streams, eid)
        if hot not in streams or cold not in streams:
            continue
        r.rule("equipment.sides", streams[hot][0] and not streams[cold][0], f"{eid}: hot/cold sides")
        r.rule("equipment.stage", 1 <= int(e["stage"]) <= K, eid)
        q = float(e["duty"])
        if e["kind"] == "existing":
            o = plant.exchangers.get(e.get("housing_unit"))
            r.rule("equipment.existing_match", o is not None and (o.hot, o.cold) == (hot, cold), eid)
            r.lower("rule.min_duty", q, R["min_duty"], eid)  # Strategy I keeps every match (A37, Eq. 8)
        elif e["kind"] == "loop":
            r.rule("equipment.loop_sides", active and (LOOP in (hot, cold)) and
                   ((hot == LOOP) == (role == "sink")), eid)
            r.lower("rule.min_duty", q, R["min_duty"], eid)
        else:
            r.rule("equipment.strategy_one", False, f"{eid}: new process match not allowed in Strategy I (Eq. 4)")
        U = 1.0 / (1.0 / streams[hot][4] + 1.0 / streams[cold][4])
        r.equal("unit.U", e["U"], U, eid, relative=True)
        d1 = float(e["hot_in"]) - float(e["cold_out"])
        d2 = float(e["hot_out"]) - float(e["cold_in"])
        r.lower("approach.hot_end", d1, dt, eid, 1e-4)  # A56-A67 (1e-4 K: fixed-point tolerance)
        r.lower("approach.cold_end", d2, dt, eid, 1e-4)
        area = q / (U * chen(max(d1, 1e-12), max(d2, 1e-12))) if q > tol_q else 0.0
        r.equal("area.required", e["area_required"], area, eid, relative=True)  # A76
        if e["kind"] == "existing":
            a_ex = float(e["area_existing"])
            r.check("area.installed", a_ex - plant.exchangers[e["housing_unit"]].area, tol_a, eid)
            X = float(e["augmentation_area"])
            r.rule("rule.augmentation_min", X == 0 or X >= costs.min_augmentation - 1e-9, f"{eid}: X={X}")  # Eq. 22
            if q > tol_q:  # an idle unit already fails rule.min_duty
                r.lower("area.at_least_installed", area + tol_a, a_ex, eid)  # Eq. 9
                r.check("area.installed_plus_augmentation", area - (a_ex + X), tol_a, eid)  # Eq. 12
            cap = (costs.cf_unit + costs.ca_exchanger * X ** costs.beta) if X > 0 else 0.0
        else:
            cap = costs.cf_loop + costs.ca_loop * area ** costs.beta
        r.equal("cost.unit_capital", e["capital"], cap, eid, relative=True)
    # ---- stream walk (A17-A36) ---------------------------------------------------------
    ends = {}
    for sid, (hot, t_in, t_out, F, _) in streams.items():
        T = t_in
        order = range(1, K + 1) if hot else range(K, 0, -1)
        for k in order:
            here = [e for e in ex if int(e["stage"]) == k and e["hot" if hot else "cold"] == sid]
            rb = bypass.get((sid, k), 0.0)
            if not here:
                r.rule("bypass.no_unit", rb == 0.0, f"{sid}@{k}: bypass without a unit")
                continue
            total_f, heat, mixed = rb, 0.0, rb * T
            for e in here:
                side = "hot" if hot else "cold"
                f = float(e[f"{side}_fraction"])
                q = float(e["duty"])
                tin_b, tout_b = float(e[f"{side}_in"]), float(e[f"{side}_out"])
                r.equal("temperature.branch_inlet", tin_b, T, f"{sid}@{k}/{e['id']}")
                r.equal("energy.branch", q, f * F * abs(tout_b - tin_b), f"{sid}@{k}/{e['id']}")  # A17-A22
                r.rule("energy.direction", (tout_b <= tin_b + 1e-9) if hot else (tout_b >= tin_b - 1e-9), f"{sid}@{k}/{e['id']}")
                if q > tol_q:
                    r.lower("rule.min_fraction", f, R["min_fraction"], f"{sid}@{k}/{e['id']}")  # LRH/LRC...
                total_f += f
                heat += q
                mixed += f * tout_b
            r.equal("fraction.sum", total_f, 1.0, f"{sid}@{k}")  # A25-A28
            T_next = T + (-1.0 if hot else 1.0) * heat / F
            r.equal("temperature.mixing", mixed, T_next, f"{sid}@{k}")  # A31-A34
            T = T_next
        ends[sid] = T
        r.equal("temperature.stream_end", x["stream_end_temperatures"][sid], T, sid)
    # ---- utilities (A15-A16, A23-A24, A35-A36, A68-A75; Eq. 2-6, 10-11, 13-14, 23-24) ------
    utils = x["utilities"]
    r.rule("utility.coverage", Counter(u["stream"] for u in utils) == Counter(plant.streams.keys()),
           "one entry per process stream")
    hot_cost = cold_cost = cap_util = 0.0
    n_bypass = len([k for k in bypass if k[1] != "end"])
    for u in utils:
        sid = u["stream"]
        hot, t_in, t_out, F, h = streams[sid]
        T = ends[sid]
        q = F * ((T - t_out) if hot else (t_out - T))
        ctx = u["unit"] or f"new:{sid}"
        r.lower("utility.no_overshoot", q, 0.0, ctx)  # A46, A49
        on = q > tol_q
        r.equal("energy.utility", u["duty"], q if on else 0.0, ctx)  # A15-A16
        r.equal("temperature.utility_inlet", u["T_in"], T, ctx)
        r.equal("temperature.utility_target", u["T_out"], t_out, ctx)
        rb = float(u["bypass"])
        r.rule("bypass.end_recorded", (rb > 0) == ((sid, "end") in bypass) and (rb == 0 or abs(bypass[(sid, "end")] - rb) < 1e-12), ctx)
        if not on:
            r.equal("area.utility_off", u["area_required"], 0.0, ctx)
            r.equal("cost.utility_off", u["capital"], 0.0, ctx)
            r.equal("cost.utility_annual_off", u["annual_cost"], 0.0, ctx)
            r.rule("bypass.end_off", rb == 0.0, ctx)
            continue
        if u["utility"] is None or u["unit"] is None:
            r.rule("equipment.utility_allowed", False, f"{ctx}: Strategy I allows no new cooler/heater (Eq. 2-3, 5-6)")
            continue
        util = plant.utilities[u["utility"]]
        orig = plant.utility_units[u["unit"]]
        r.rule("equipment.utility_match", (orig.stream, orig.utility) == (sid, util.id), ctx)
        r.lower("rule.min_duty", q, R["min_duty"], ctx)
        n_bypass += rb > 0
        branch = (t_out - rb * T) / (1.0 - rb)  # A35/A36 mixing to the target
        r.equal("temperature.utility_branch", u["branch_out"], branch, ctx)
        if hot:
            da, db = T - util.t_out, branch - util.t_in
        else:
            da, db = util.t_in - branch, util.t_out - T
        r.lower("approach.utility", min(da, db), dt, ctx, 1e-4)  # A68-A75
        U = 1.0 / (1.0 / h + 1.0 / util.h)
        area = q / (U * chen(max(da, 1e-12), max(db, 1e-12)))
        r.equal("area.utility_required", u["area_required"], area, ctx, relative=True)
        a_ex = float(u["area_existing"])
        r.check("area.utility_installed", a_ex - orig.area, tol_a, ctx)
        r.lower("area.utility_at_least_installed", area + tol_a, a_ex, ctx)  # Eq. 10-11
        X = float(u["augmentation_area"])
        expected_X = max(area - a_ex, costs.min_augmentation) if area > a_ex + tol_a else 0.0
        r.equal("area.utility_augmentation", X, expected_X, ctx, relative=True)  # Eq. 13-14, 23-24
        ca = costs.ca_cooler if hot else costs.ca_heater
        cap = (costs.cf_unit + ca * X ** costs.beta) if X > 0 else 0.0
        r.equal("cost.utility_capital", u["capital"], cap, ctx, relative=True)
        cap_util += cap
        r.equal("cost.utility_annual", u["annual_cost"], q * util.price, ctx, relative=True)
        if hot:
            cold_cost += q * util.price
        else:
            hot_cost += q * util.price
    # ---- loop and costs ----------------------------------------------------------------
    loop_duty = sum(float(e["duty"]) for e in ex if e["kind"] == "loop")
    slack = max(duty_itf - loop_duty, 0.0)
    r.equal("energy.loop", x["slack_kW"], slack, plant.id)
    r.lower("energy.loop_not_exceeded", duty_itf + 1e-6, loop_duty, plant.id)  # A52/A55
    c = x["costs"]
    unit_cap = sum(float(e["capital"]) for e in ex)
    r.equal("cost.hot_utility", c["hot_utility"], hot_cost, plant.id, relative=True)
    r.equal("cost.cold_utility", c["cold_utility"], cold_cost, plant.id, relative=True)
    r.equal("cost.unit_capital_sum", c["unit_capital"], unit_cap, plant.id, relative=True)
    r.equal("cost.utility_capital_sum", c["utility_capital"], cap_util, plant.id, relative=True)
    r.equal("cost.bypass_capital", c["bypass_capital"], R["bypass_cost"] * n_bypass, plant.id, relative=True)  # Eq. 18-21
    capital = unit_cap + cap_util + R["bypass_cost"] * n_bypass
    r.equal("cost.capital", c["capital"], capital, plant.id, relative=True)
    tac_clean = hot_cost + cold_cost + costs.af * capital
    r.equal("cost.TAC_without_penalty", c["TAC_without_penalty"], tac_clean, plant.id, relative=True)
    r.equal("cost.TAC", c["TAC"], tac_clean + float(c["penalty"]), plant.id, relative=True)
    r.lower("cost.penalty_nonnegative", c["penalty"], 0.0, plant.id)
    r.equal("cost.penalty_slack", c["penalty"], R["slack_price"] * (slack + float(x["violation_kW"])), plant.id, relative=True)
    strict = not r.failures and slack <= 1e-6 and float(x["violation_kW"]) <= 1e-6
    r.rule("plant.strictly_feasible", bool(x["strictly_feasible"]) == strict,
           f"exported={x['strictly_feasible']}, audited={strict}")


def audit_site(x: dict, case_file=None, rules: dict | None = None) -> dict:
    """Audit a ``SiteSim.export``: every plant, the loop headers and the site costs."""
    r = Report()
    reports = {}
    try:
        plants, costs = load_case() if case_file is None else load_case(case_file)
        members = list(x["plants"])
        for pid in members:
            rep = audit_plant(x["plants"][pid], plants[pid], costs, rules)
            reports[pid] = rep
            r.failures.extend(f"{pid}: {f}" for f in rep["failures"])
        tr, ts, loop = float(x["T_return"]), float(x["T_supply"]), float(x["loop_flow"])
        src_flow = snk_flow = src_duty = snk_duty = mix = 0.0
        for pid, itf in x["interfaces"].items():
            flow, tin, tout = float(itf["flow"]), float(itf["t_in"]), float(itf["t_out"])
            q = flow * abs(tin - tout)
            role = ("sink" if tin > tout else "source") if flow > 0 and q > 1e-9 else "inactive"
            r.rule("site.role", itf["role"] == role == x["roles"][pid], pid)
            if pid in members and "interface" in x["plants"][pid]:
                e = x["plants"][pid]["interface"]
                for a, b in (("flow", "flow"), ("T_in", "t_in"), ("T_out", "t_out")):
                    r.equal("site.plant_interface", e[a], float(itf[b]), f"{pid}/{a}")
            if role == "source":
                r.equal("site.source_inlet", tin, tr, pid)
                src_flow += flow
                src_duty += q
                mix += flow * tout
            elif role == "sink":
                r.equal("site.sink_inlet", tin, ts, pid)
                r.equal("site.sink_outlet", tout, tr, pid)
                snk_flow += flow
                snk_duty += q
        r.equal("site.source_flow", src_flow, loop, "header")
        r.equal("site.sink_flow", snk_flow, loop, "header")
        r.equal("site.supply_mix", ts, mix / src_flow if src_flow > 0 else tr, "header")
        r.equal("site.energy_balance", src_duty, snk_duty, "loop")
        loop_cap = costs.cim * loop
        r.equal("site.loop_capital", x["loop_capital"], loop_cap, "Eq. 74", relative=True)
        plant_tac = sum(float(x["plants"][p]["TAC"]) for p in members)
        r.equal("site.TAC", x["TAC"], plant_tac + costs.af * loop_cap, "site", relative=True)
        baseline = sum(plants[p].baseline_utility_cost for p in members)
        r.equal("site.TACS", x["TACS"], baseline - float(x["TAC"]), "site", relative=True)
        strict = not r.failures and all(bool(x["plants"][p]["strictly_feasible"]) for p in members)
        r.rule("site.strictly_feasible", bool(x["strictly_feasible"]) == strict, f"exported={x['strictly_feasible']}")
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as exc:
        r.rule("export.schema", False, f"{type(exc).__name__}: {exc}")
    return r.output() | {"plants": reports}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)
    for f in args.files:
        x = json.loads(f.read_text(encoding="utf-8"))
        rep = audit_site(x) if "plants" in x else None
        if rep is None:
            plants, costs = load_case()
            rep = audit_plant(x, plants[x["plant"]], costs)
        print(f"{f}: {'PASS' if rep['passed'] else 'FAIL'}")
        for fail in rep["failures"][:30]:
            print("   ", fail)


if __name__ == "__main__":
    main()
