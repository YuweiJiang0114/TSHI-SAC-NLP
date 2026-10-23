"""Independent physical and cost audit of exported Strategy-I designs.

Only case inputs are imported. Temperatures and duties have 1e-6 absolute
accuracy; areas and costs have 1e-6 relative accuracy with an absolute floor.
Approach inequalities use tol. Residuals retain their original units.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .case import load_case


class _Report:
    def __init__(self):
        self.residuals = {}
        self.failures = []

    def check(self, name, residual, allowed=0.0, detail=""):
        value = abs(float(residual))
        if not math.isfinite(value):
            value = math.inf
        self.residuals[name] = max(self.residuals.get(name, 0.0), value)
        if value > allowed:
            self.failures.append(f"{name}: {detail} (residual={value:.12g}, allowed={allowed:.12g})")

    def rule(self, name, condition, detail):
        self.check(name, 0 if condition else 1, detail=detail)

    def equal(self, name, actual, expected, detail, *, relative=False):
        allowance = 1e-6 * max(1.0, abs(expected)) if relative else 1e-6
        self.check(name, _number(actual) - expected, allowance, detail)

    def lower(self, name, actual, limit, allowance, detail):
        self.check(name, max(0.0, limit - actual), allowance, detail)

    def output(self):
        return {"passed": not self.failures,
                "max_residual": max(self.residuals.values(), default=0.0),
                "residuals": self.residuals, "failures": self.failures}


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric value")
    return value


def _chen(a, b):
    return (a * b * (a + b) / 2.0) ** (1.0 / 3.0)


def _area(r, name, duty, conductance, a, b, dt_min, tol, context):
    r.lower(name + ".approach", a, dt_min, tol, context + " terminal 1")
    r.lower(name + ".approach", b, dt_min, tol, context + " terminal 2")
    if a <= 0 or b <= 0 or conductance <= 0:
        r.rule(name + ".area_denominator", False, context)
        return 0.0
    return max(duty, 0.0) / (conductance * _chen(a, b))


def _bypass(r, entry, required, duty, free_area, bypass_cost, duty_tolerance, context):
    """A used existing unit with surplus installed area is throttled by a bypass."""
    needed = bypass_cost > 0 and duty > duty_tolerance and required < _number(entry["area_existing"]) - free_area
    if "bypass" in entry:
        r.rule("equipment.bypass_flag", bool(entry["bypass"]) == needed, context)
    return bypass_cost if needed else 0.0


def _augmentation(r, entry, required, ca, costs, area_tolerance, context):
    excess = required - _number(entry["area_existing"])
    augmented = excess > area_tolerance
    # At the discontinuity, independently evaluated Chen expressions can differ
    # by a few floating-point ulps. Both decisions are valid only in that band.
    if abs(excess - area_tolerance) <= 1e-10:
        augmented = entry["augmented"]
    r.rule("equipment.augmentation_flag", isinstance(entry["augmented"], bool) and
           entry["augmented"] == augmented, context)
    addition = max(excess, costs.min_augmentation) if augmented else 0.0
    r.equal("area.augmentation", entry["augmentation_area"], addition, context, relative=True)
    return costs.cf_unit + ca * addition ** costs.beta if augmented else 0.0


RELAXED_RULES = {"free_area": None, "min_unit_duty": 0.0, "bypass_cost": 0.0, "min_fraction": 0.0,
                 "strategy": 1}


def audit_plant(result: dict, plant, costs, *, dt_min=1.0, slack_price=2000.0, duty_tolerance=1e-3,
                area_tolerance=0.05, tol=1e-4, rules=None) -> dict:
    """Audit one plant export without importing or invoking an optimizer.

    ``rules`` (free_area, min_unit_duty, bypass_cost, min_fraction) default to the
    rules the export declares, else to the relaxed model; pass them explicitly to
    audit a design against other rules (e.g. Lo et al. Sec. 6 bounds).
    Malformed exports fail with a diagnostic. max_residual is unnormalised:
    use passed instead of comparing residuals of different units with one limit.
    """
    r = _Report()
    declared = dict(RELAXED_RULES)
    declared.update({k: v for k, v in (result.get("rules") or {}).items() if k in RELAXED_RULES})
    declared.update(rules or {})
    if declared["free_area"] is None:
        declared["free_area"] = area_tolerance
    try:
        _audit_plant(result, plant, costs, r, dt_min, slack_price, area_tolerance, tol, duty_tolerance, declared)
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError, OverflowError) as exc:
        r.rule("export.schema", False, f"{type(exc).__name__}: {exc}")
    return r.output()


def _audit_plant(result, plant, costs, r, dt_min, slack_price, area_tolerance, tol, duty_tolerance=1e-3,
                 rules=None):
    rules = rules or dict(RELAXED_RULES, free_area=area_tolerance)
    strategy = int(rules.get("strategy", 1))
    free_area, min_duty = rules["free_area"], rules["min_unit_duty"]
    bypass_cost, min_fraction = rules["bypass_cost"], rules["min_fraction"]
    r.rule("equipment.plant", result["plant"] == plant.id, "plant identity")
    itf = result["interface"]
    flow, tin, tout = (_number(itf[k]) for k in ("flow", "T_in", "T_out"))
    interface_duty = flow * abs(tin - tout)
    active = flow > 0 and interface_duty > 1e-9
    role = ("sink" if tin > tout else "source") if active else "inactive"
    r.lower("interface.flow", flow, 0, 0, plant.id)
    r.rule("interface.role", itf["role"] == role, plant.id)
    r.equal("interface.duty", itf["duty"], interface_duty, plant.id)
    slack = _number(result["slack_kW"])
    r.lower("interface.slack", slack, 0, 1e-6, plant.id)
    r.lower("interface.slack", interface_duty - slack, 0, 1e-6, plant.id)

    rows = result["exchangers"]
    ids = [e["id"] for e in rows]
    counts = Counter(ids)
    r.rule("equipment.unique_ids", len(ids) == len(counts), "exchanger IDs")
    for eid in plant.exchangers:
        r.rule("equipment.existing_count", counts[eid] == 1, eid)
    ex = {e["id"]: e for e in rows}
    streams = {sid: (s.hot, s.t_in, s.fcp, s.h) for sid, s in plant.streams.items()}
    if active:
        streams["LOOP"] = (role == "sink", tin, flow, costs.loop_h)
    loop_streams = []
    housed_by = Counter()

    def check_housing(e, eid):
        """Unit housing a process match: its own (I), another existing one (II/III), new (III)."""
        unit = e.get("housing_unit", eid if e["kind"] == "existing" else "new")
        active = _number(e["duty"]) > duty_tolerance
        if unit is None:
            r.rule("equipment.dropped_match", strategy >= 3 and not active, eid)
            r.equal("area.dropped_installed", e["area_existing"], 0, eid, relative=True)
        elif unit == "new":
            r.rule("equipment.new_unit", strategy >= 3, eid)
            r.equal("area.new_installed", e["area_existing"], 0, eid, relative=True)
        else:
            r.rule("equipment.housing_unit", unit in plant.exchangers and (unit == eid or strategy >= 2), eid)
            if unit in plant.exchangers:
                r.check("area.installed", _number(e["area_existing"]) - plant.exchangers[unit].area,
                        area_tolerance, eid)
                housed_by[unit] += 1
                r.rule("equipment.housed_match_active", active or strategy < 3, eid)

    for e in rows:
        eid = e["id"]
        if eid in plant.exchangers:
            original = plant.exchangers[eid]
            r.rule("equipment.existing_match", e["kind"] == "existing" and
                   (e["hot"], e["cold"]) == (original.hot, original.cold), eid)
            check_housing(e, eid)
        elif e["kind"] == "new":
            r.rule("equipment.new_match", strategy >= 3 and e["hot"] in plant.streams and e["cold"] in plant.streams
                   and plant.streams[e["hot"]].hot and not plant.streams[e["cold"]].hot, eid)
            check_housing(e, eid)
        else:
            r.rule("equipment.loop_kind", e["kind"] == "loop", eid)
            loop_hot = role == "sink"
            sid = e["cold"] if loop_hot else e["hot"]
            valid = active and sid in plant.streams and (
                e["hot"] == "LOOP" if loop_hot else e["cold"] == "LOOP")
            if sid in plant.streams:
                valid = valid and plant.streams[sid].hot != loop_hot
            r.rule("equipment.loop_side", valid, eid)
            loop_streams.append(sid)
            r.equal("area.loop_installed", e["area_existing"], 0, eid, relative=True)
            r.rule("equipment.loop_augmentation", e["augmented"] is False, eid)
            r.equal("area.loop_augmentation", e["augmentation_area"], 0, eid, relative=True)
        r.lower("exchanger.duty_nonnegative", _number(e["duty"]), 0, 1e-6, eid)
        r.lower("area.installed_nonnegative", _number(e["area_existing"]), 0, 0, eid)
    for unit in plant.exchangers:  # Eq. 7-8 / 54: every existing unit houses exactly one match
        r.rule("equipment.unit_houses_one_match", housed_by[unit] == 1, unit)
    r.rule("equipment.loop_per_stream", len(loop_streams) == len(set(loop_streams)),
           "at most one loop exchanger per process stream")
    r.rule("equipment.inactive_loop", active or not loop_streams, "inactive interface")

    chains = result["chains"]
    expected_chains = set(plant.streams) | ({"LOOP"} if loop_streams else set())
    r.rule("topology.streams", set(chains) == expected_chains, "chain stream IDs")
    ends, incidence = {}, Counter()
    for sid, chain in chains.items():
        if sid not in streams:
            r.rule("topology.stream", False, sid)
            continue
        hot, supply, fcp, _ = streams[sid]
        sign = -1 if hot else 1
        temperature, original_groups = supply, []
        for gi, group in enumerate(chain):
            context = f"{sid}/group {gi}"
            r.rule("topology.nonempty_group", bool(group), context)
            retained = [eid for eid in group if eid in plant.exchangers]
            if retained:
                original_groups.append(set(retained))
            fractions, heat, mixed = [], 0.0, 0.0
            for eid in group:
                incidence[(sid, eid)] += 1
                if eid not in ex:
                    r.rule("topology.member", False, context + "/" + eid)
                    continue
                e = ex[eid]
                side = "hot" if hot else "cold"
                r.rule("topology.side", e[side] == sid, context + "/" + eid)
                inlet, outlet = _number(e[side + "_in"]), _number(e[side + "_out"])
                fraction, q = _number(e[side + "_fraction"]), _number(e["duty"])
                r.rule("topology.fraction_range", 0 < fraction <= 1, context + "/" + eid)
                if len(group) > 1:
                    r.lower("rule.min_fraction", fraction, min_fraction, 1e-9, context + "/" + eid)
                r.equal("temperature.branch_inlet", inlet, temperature, context + "/" + eid)
                r.equal("energy.branch", q, fraction * fcp * abs(inlet - outlet), context + "/" + eid)
                r.equal("energy.branch_direction", fraction * fcp * (outlet - inlet), sign * q,
                        context + "/" + eid)
                fractions.append(fraction)
                heat += q
                mixed += fraction * outlet
            r.equal("topology.fraction_sum", sum(fractions), 1, context)
            next_temperature = temperature + sign * heat / fcp
            r.equal("temperature.mixed_outlet", mixed, next_temperature, context)
            temperature = next_temperature
        if sid in plant.streams:
            r.rule("topology.original_order", original_groups == [set(g) for g in plant.chains[sid]], sid)
        ends[sid] = temperature
    for e in rows:
        for side in ("hot", "cold"):
            r.rule("topology.incidence", incidence[(e[side], e["id"])] == 1, e["id"] + "/" + side)
    r.rule("temperature.end_streams", set(result["stream_end_temperatures"]) == set(ends), "stream end IDs")
    for sid, temperature in ends.items():
        r.equal("temperature.stream_end", result["stream_end_temperatures"][sid], temperature, sid)

    augmentation_capital = loop_capital = new_capital = bypass_capital = 0.0
    reassignment_capital = new_process_capital = 0.0
    for e in rows:
        eid = e["id"]
        if e["hot"] not in streams or e["cold"] not in streams:
            r.rule("equipment.stream_reference", False, eid)
            continue
        hh, hc = streams[e["hot"]][3], streams[e["cold"]][3]
        conductance = 1 / (1 / hh + 1 / hc)
        r.equal("exchanger.U", e["U"], conductance, eid, relative=True)
        duty = _number(e["duty"])
        dropped = strategy >= 3 and e["kind"] in ("existing", "new") and duty <= duty_tolerance
        if dropped:  # a dropped match is not a unit: no approach or area
            required = 0.0
        else:
            required = _area(r, "exchanger", duty, conductance,
                             _number(e["hot_in"]) - _number(e["cold_out"]),
                             _number(e["hot_out"]) - _number(e["cold_in"]), dt_min, tol, eid)
        r.equal("area.exchanger_required", e["area_required"], required, eid, relative=True)
        unit = e.get("housing_unit", eid if e["kind"] == "existing" else "new")
        if e["kind"] in ("existing", "new") and unit in plant.exchangers and unit != eid:
            moved = (plant.exchangers[unit].hot, plant.exchangers[unit].cold) != (e["hot"], e["cold"])
            reassignment_capital += costs.cm if moved else 0.0  # Eq. 76/78: changed stream match
        if e["kind"] in ("existing", "new") and unit in (None, "new"):
            capital = (costs.cf_unit + costs.ca_exchanger * required ** costs.beta) if not dropped else 0.0
            new_process_capital += capital
            if not dropped:
                r.lower("rule.min_unit_duty", duty, min_duty, 1e-6, eid)
        elif e["kind"] in ("existing", "new"):
            capital = _augmentation(r, e, required, costs.ca_exchanger, costs, free_area, eid)
            augmentation_capital += capital
            duty = _number(e["duty"])
            bypass_capital += _bypass(r, e, required, duty, free_area, bypass_cost, duty_tolerance, eid)
            if duty > duty_tolerance:
                r.lower("rule.min_unit_duty", duty, min_duty, 1e-6, eid)
        else:
            capital = costs.cf_loop + costs.ca_loop * required ** costs.beta
            loop_capital += capital
        r.equal("cost.exchanger_capital", e["capital"], capital, eid, relative=True)

    utilities = result["utilities"]
    utility_counts = Counter(u["stream"] for u in utilities)
    r.rule("equipment.utility_per_stream", set(utility_counts) == set(plant.streams) and
           all(n == 1 for n in utility_counts.values()), "utility coverage and uniqueness")
    unit_counts = Counter(u["unit"] for u in utilities if u["unit"] is not None)
    for uid in plant.utility_units:
        r.rule("equipment.existing_utility_count", unit_counts[uid] == 1, uid)
    hot_cost = cold_cost = 0.0
    for entry in utilities:
        sid, uid = entry["stream"], entry["unit"]
        s, utility = plant.streams[sid], plant.utilities[entry["utility"]]
        context = uid or f"new:{sid}"
        r.lower("area.utility_installed_nonnegative", _number(entry["area_existing"]), 0, 0, context)
        r.rule("equipment.utility_kind", entry["kind"] == ("cooler" if s.hot else "heater") and
               utility.kind == ("cold" if s.hot else "hot"), context)
        if uid is not None:
            original = plant.utility_units[uid]
            r.rule("equipment.utility_match", (sid, entry["kind"], entry["utility"]) ==
                   (original.stream, original.kind, original.utility), context)
            if (entry.get("housing_unit") or uid) == uid:  # a reassigned unit is checked below
                r.check("area.utility_installed", _number(entry["area_existing"]) - original.area,
                        area_tolerance, context)
        else:
            r.rule("equipment.new_utility", not any(u.stream == sid for u in plant.utility_units.values()), context)
            r.equal("area.new_utility_installed", entry["area_existing"], 0, context, relative=True)
        before = ends[sid]
        q = s.fcp * (before - s.t_out if s.hot else s.t_out - before)
        r.lower("utility.duty_nonnegative", q, 0, 1e-6, context)
        r.lower("utility.exported_duty_nonnegative", _number(entry["duty"]), 0, 0, context)
        # A residual within the optimizer's duty tolerance (1e-3 kW) is numerical zero:
        # exported as 0 kW, no utility and no unit.  The stream then misses its target
        # by at most duty_tolerance / FCp.
        on = q > duty_tolerance
        if on:
            r.equal("energy.utility", entry["duty"], q, context)
        else:
            r.check("energy.utility_switched_off", _number(entry["duty"]), 0.0, context)
            r.check("energy.utility_residual_within_tolerance", max(abs(q) - duty_tolerance, 0.0), 0.0, context)
        r.equal("temperature.utility_inlet", entry["T_in"], before, context)
        r.equal("temperature.utility_target", entry["T_out"], s.t_out, context)
        conductance = 1 / (1 / s.h + 1 / utility.h)
        r.equal("utility.U", entry["U"], conductance, context, relative=True)
        a, b = ((before - utility.t_out, s.t_out - utility.t_in) if s.hot else
                (utility.t_in - s.t_out, utility.t_out - before))
        required = _area(r, "utility", q, conductance, a, b, dt_min, tol, context) if on else 0.0
        r.equal("area.utility_required", entry["area_required"], required, context, relative=True)
        ca = costs.ca_cooler if s.hot else costs.ca_heater
        if on:
            r.lower("rule.min_unit_duty", q, min_duty, 1e-6, context)
        if uid is not None:
            home = entry.get("housing_unit") or uid
            if home != uid:  # Strategy II/III: another existing utility unit of the same type
                r.rule("equipment.utility_housing", strategy >= 2 and home in plant.utility_units and
                       plant.utility_units[home].kind == original.kind and
                       plant.utility_units[home].utility == original.utility, context)
                if home in plant.utility_units:
                    r.check("area.utility_housing", _number(entry["area_existing"]) - plant.utility_units[home].area,
                            area_tolerance, context)
                    reassignment_capital += costs.cm if plant.utility_units[home].stream != sid else 0.0
            capital = _augmentation(r, entry, required, ca, costs, free_area, context)
            augmentation_capital += capital
            bypass_capital += _bypass(r, entry, required, q if on else 0.0, free_area, bypass_cost,
                                      duty_tolerance, context)
        else:
            r.rule("equipment.new_utility_augmentation", entry["augmented"] is False, context)
            r.equal("area.new_utility_augmentation", entry["augmentation_area"], 0, context, relative=True)
            capital = costs.cf_unit + ca * required ** costs.beta if on else 0.0
            new_capital += capital
        r.equal("cost.utility_capital", entry["capital"], capital, context, relative=True)
        annual = q * utility.price if on else 0.0
        r.equal("cost.utility_annual", entry["annual_cost"], annual, context, relative=True)
        if s.hot:
            cold_cost += annual
        else:
            hot_cost += annual

    loop_duty = sum(_number(e["duty"]) for e in rows if e["kind"] == "loop")
    r.equal("energy.loop", loop_duty + slack, interface_duty, plant.id)
    capital = (augmentation_capital + loop_capital + new_capital + bypass_capital + reassignment_capital
               + new_process_capital)
    penalty = slack_price * slack
    tac = hot_cost + cold_cost + costs.af * capital + penalty
    expected = {"hot_utility": hot_cost, "cold_utility": cold_cost,
                "augmentation_capital": augmentation_capital, "loop_exchanger_capital": loop_capital,
                "new_utility_capital": new_capital, "capital": capital,
                **({"bypass_capital": bypass_capital} if "bypass_capital" in result["costs"] or bypass_capital else {}),
                **({"reassignment_capital": reassignment_capital}
                   if "reassignment_capital" in result["costs"] or reassignment_capital else {}),
                **({"new_process_capital": new_process_capital}
                   if "new_process_capital" in result["costs"] or new_process_capital else {}),
                "annualized_capital": costs.af * capital, "slack_penalty": penalty, "TAC": tac}
    for key, value in expected.items():
        r.equal("cost." + key, result["costs"].get(key, 0.0), value, plant.id, relative=True)
    r.equal("cost.plant_TAC", result["TAC"], tac, plant.id, relative=True)
    strict = not r.failures and abs(slack) <= 1e-6
    r.rule("plant.strictly_feasible", result["strictly_feasible"] is strict, plant.id)


PAPER_RULES = {"free_area": 1e-6, "min_unit_duty": 30.0, "bypass_cost": 500.0, "min_fraction": 0.1}


def audit_site(site_result: dict, *, case_file=None, dt_min=1.0, slack_price=2000.0, rules=None) -> dict:
    """Audit a SiteEvaluator export, including coalition membership and headers.

    ``rules=None`` audits against the rules each plant export declares; pass
    ``PAPER_RULES`` to check a design against Lo et al. Sec. 6 bounds.
    """
    r, reports = _Report(), {}
    try:
        plants, costs = load_case() if case_file is None else load_case(case_file)
        _audit_site(site_result, plants, costs, r, reports, dt_min, slack_price, rules)
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError, OverflowError) as exc:
        r.rule("export.schema", False, f"{type(exc).__name__}: {exc}")
    return r.output() | {"plants": reports}


def _audit_site(result, plants, costs, r, reports, dt_min, slack_price, rules=None):
    members = set(result["plants"])
    r.rule("site.members", bool(members) and members <= set(plants), "coalition IDs")
    for pid, design in result["plants"].items():
        if pid not in plants:
            continue
        report = audit_plant(design, plants[pid], costs, dt_min=dt_min, slack_price=slack_price, rules=rules)
        reports[pid] = report
        for name, value in report["residuals"].items():
            r.residuals[f"{pid}.{name}"] = value
        r.failures.extend(f"{pid}: {failure}" for failure in report["failures"])
    interfaces = result["interfaces"]
    r.rule("site.interface_ids", set(interfaces) == set(plants), "interface IDs")
    r.rule("site.role_ids", set(result["roles"]) == set(interfaces), "role IDs")
    tr, ts, loop = (_number(result[k]) for k in ("T_return", "T_supply", "loop_flow"))
    r.lower("site.loop_flow_nonnegative", loop, 0, 0, "loop flow")
    source_flow = sink_flow = source_duty = sink_duty = outlet_sum = 0.0
    for pid, itf in interfaces.items():
        flow, tin, tout = (_number(itf[k]) for k in ("flow", "t_in", "t_out"))
        q = flow * abs(tin - tout)
        active = flow > 0 and q > 1e-9
        role = ("sink" if tin > tout else "source") if active else "inactive"
        r.lower("site.interface_flow", flow, 0, 0, pid)
        r.rule("site.interface_role", itf["role"] == role and result["roles"][pid] == role, pid)
        r.equal("site.interface_duty", itf["duty"], q, pid)
        r.rule("site.coalition_interface", pid in members or not active, pid)
        if pid in members:
            exported = result["plants"][pid]["interface"]
            for local, site in (("flow", "flow"), ("T_in", "t_in"), ("T_out", "t_out"), ("duty", "duty")):
                r.equal("site.plant_interface", exported[local], _number(itf[site]), pid + "/" + local)
            r.rule("site.plant_role", exported["role"] == role, pid)
        if role == "source":
            r.equal("site.source_inlet", tin, tr, pid)
            source_flow += flow
            source_duty += q
            outlet_sum += flow * tout
        elif role == "sink":
            r.equal("site.sink_inlet", tin, ts, pid)
            r.equal("site.sink_outlet", tout, tr, pid)
            sink_flow += flow
            sink_duty += q
    r.equal("site.source_flow", source_flow, loop, "source header")
    r.equal("site.sink_flow", sink_flow, loop, "sink header")
    r.equal("site.supply_mix", ts, outlet_sum / source_flow if source_flow > 0 else tr, "supply header")
    r.equal("site.energy_balance", source_duty, sink_duty, "loop duties")
    if "balance_kW" in result:
        r.equal("site.exported_balance", result["balance_kW"], source_duty - sink_duty, "loop duties")
    r.lower("site.temperature_lift", ts - tr, 0, 1e-6, "headers")
    loop_capital = costs.cim * loop
    annual = costs.af * loop_capital
    plant_tac = sum(_number(d["TAC"]) for d in result["plants"].values())
    tac = plant_tac + annual
    r.equal("site.loop_capital", result["loop_capital"], loop_capital, "loop", relative=True)
    r.equal("site.loop_annual_cost", result["loop_annual_cost"], annual, "loop", relative=True)
    r.equal("site.TAC", result["TAC"], tac, "site", relative=True)
    baseline = sum(plants[pid].baseline_utility_cost for pid in members)
    r.equal("site.TACS", result["TACS"], baseline - tac, "coalition savings", relative=True)
    r.rule("site.plant_TAC_ids", set(result["plant_TAC"]) == members, "plant TAC IDs")
    for pid in members:
        r.equal("site.plant_TAC", result["plant_TAC"][pid], _number(result["plants"][pid]["TAC"]),
                pid, relative=True)
    slack = sum(_number(d["slack_kW"]) for d in result["plants"].values())
    r.equal("site.slack", result["slack_kW"], slack, "site slack")
    r.equal("site.TAC_without_penalty", result["TAC_without_penalty"], tac - slack_price * slack,
            "site unpenalised cost", relative=True)
    strict = not r.failures and all(abs(_number(d["slack_kW"])) <= 1e-6
                                   for d in result["plants"].values())
    r.rule("site.strictly_feasible", result["strictly_feasible"] is strict,
           f"exported={result['strictly_feasible']}, independently computed={strict}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="exported site design JSON files")
    parser.add_argument("--case-file", type=Path)
    parser.add_argument("--dt-min", type=float, default=1.0)
    parser.add_argument("--slack-price", type=float, default=2000.0)
    parser.add_argument("--paper-rules", action="store_true",
                        help="check Lo et al. Sec. 6 bounds instead of the rules the design declares")
    args = parser.parse_args(argv)
    failed = False
    for path in args.files:
        try:
            design = json.loads(path.read_text(encoding="utf-8"))
            report = audit_site(design, case_file=args.case_file, dt_min=args.dt_min,
                                slack_price=args.slack_price, rules=PAPER_RULES if args.paper_rules else None)
        except (OSError, ValueError) as exc:
            print(f"FAIL {path}: {exc}")
            failed = True
            continue
        print(f"{'PASS' if report['passed'] else 'FAIL'} {path} "
              f"(max residual {report['max_residual']:.6g})")
        for pid, plant_report in report["plants"].items():
            print(f"  {pid}: {'PASS' if plant_report['passed'] else 'FAIL'}")
        for failure in report["failures"][:10]:
            print("  " + failure)
        if len(report["failures"]) > 10:
            print(f"  ... {len(report['failures']) - 10} more failures")
        failed |= not report["passed"]
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
