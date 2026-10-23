"""Problem-table check of case/data.json against the stand-alone utilities in Lo 2021 Fig. 2."""

import json
import sys
from pathlib import Path

DATA = Path(__file__).with_name("data.json")


def problem_table(streams, dt_min):
    shifted = []
    for s in streams:
        shift = -dt_min / 2 if s["type"] == "hot" else dt_min / 2
        sign = 1 if s["type"] == "hot" else -1
        shifted.append((s["Tin_C"] + shift, s["Tout_C"] + shift, s["FCp_kW_per_C"], sign))
    temps = sorted({t for s in shifted for t in s[:2]}, reverse=True)
    cascade = [0.0]
    for hi, lo in zip(temps, temps[1:]):
        net = sum(
            sign * f * (hi - lo)
            for a, b, f, sign in shifted
            if max(a, b) >= hi and min(a, b) <= lo
        )
        cascade.append(cascade[-1] + net)
    qh = max(0.0, -min(cascade))
    qc = cascade[-1] + qh
    pinches = [temps[i] for i, v in enumerate(cascade) if abs(v + qh) < 1e-9]
    return qh, qc, pinches


def existing_utilities(plant):
    hu = sum(u["duty_kW"] for u in plant["existing_hen"]["units"] if u["kind"] == "heater")
    cu = sum(u["duty_kW"] for u in plant["existing_hen"]["units"] if u["kind"] == "cooler")
    return hu, cu


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    status = 0
    print("Stream energy balance (sum hot - sum cold) vs existing (HU - CU):")
    for name, plant in data["plants"].items():
        h = sum(s["FCp_kW_per_C"] * (s["Tin_C"] - s["Tout_C"]) for s in plant["streams"] if s["type"] == "hot")
        c = sum(s["FCp_kW_per_C"] * (s["Tout_C"] - s["Tin_C"]) for s in plant["streams"] if s["type"] == "cold")
        hu, cu = existing_utilities(plant)
        ok = abs((c - h) - (hu - cu)) < 1e-6
        status |= not ok
        print(f"  {name}: hot={h:.1f} cold={c:.1f} cold-hot={c - h:.1f} HU-CU={hu - cu:.1f} {'OK' if ok else 'MISMATCH'}")

    print("\nExchanger duty closure per stream (existing HEN):")
    for name, plant in data["plants"].items():
        for s in plant["streams"]:
            load = s["FCp_kW_per_C"] * abs(s["Tin_C"] - s["Tout_C"])
            side = "hot" if s["type"] == "hot" else "cold"
            served = sum(u["duty_kW"] for u in plant["existing_hen"]["units"] if u[side] == s["id"])
            ok = abs(load - served) < 1e-6
            status |= not ok
            print(f"  {name}-{s['id']}: load={load:.1f} served={served:.1f} {'OK' if ok else 'MISMATCH'}")

    print("\nProblem-table targets vs existing stand-alone utilities (Lo 2021 Fig. 2):")
    for dt in data["min_approach_temperature"]["candidates_checked_K"]:
        print(f"  dTmin = {dt} K")
        for name, plant in data["plants"].items():
            qh, qc, pinches = problem_table(plant["streams"], dt)
            hu, cu = existing_utilities(plant)
            flag = "MATCH" if abs(qh - hu) < 0.05 and abs(qc - cu) < 0.05 else "differs"
            print(f"    {name}: QHmin={qh:7.1f} QCmin={qc:7.1f} shifted pinch={pinches} | existing HU={hu:.1f} CU={cu:.1f} -> {flag}")

    print("\nZhou 2024 Fig. 8 stand-alone targets (same streams, HRAT = 10 K):")
    zhou = data["same_streams_elsewhere"]["zhou2024_case2"]["published_standalone_targets_kW_Fig8"]
    for name, plant in data["plants"].items():
        qh, qc, _ = problem_table(plant["streams"], 10.0)
        ref = zhou[name]
        ref_qh = next(v for k, v in ref.items() if k.startswith("QH"))
        ok = abs(qh - ref_qh) < 0.05 and abs(qc - ref["QC_CW"]) < 0.05
        status |= not ok
        print(f"  {name}: QHmin={qh:.1f} QCmin={qc:.1f} | Zhou QH={ref_qh:.1f} QC={ref['QC_CW']:.1f} -> {'MATCH' if ok else 'MISMATCH'}")

    print("\nInstalled area vs Q/(U*LMTD_Chen) from Fig. 2 temperatures:")
    for name, plant in data["plants"].items():
        h = {s["id"]: s["h_kW_per_m2C"] for s in plant["streams"]}
        util = data["utilities"][name]
        for u in plant["existing_hen"]["units"]:
            if u["kind"] == "exchanger":
                hh, hc = h[u["hot"]], h[u["cold"]]
                d1, d2 = u["hot_in_C"] - u["cold_out_C"], u["hot_out_C"] - u["cold_in_C"]
            elif u["kind"] == "cooler":
                hh, hc = h[u["hot"]], util["CW"]["h_kW_per_m2C"]
                d1, d2 = u["hot_in_C"] - util["CW"]["Tout_C"], u["hot_out_C"] - util["CW"]["Tin_C"]
            else:
                hh, hc = util[u["hot"]]["h_kW_per_m2C"], h[u["cold"]]
                d1, d2 = util[u["hot"]]["Tin_C"] - u["cold_out_C"], util[u["hot"]]["Tout_C"] - u["cold_in_C"]
            lmtd = (d1 * d2 * 0.5 * (d1 + d2)) ** (1 / 3)
            area = u["duty_kW"] / (hh * hc / (hh + hc) * lmtd)
            print(f"  {name}-{u['id']}: printed={u['area_m2']:7.1f} recomputed={area:7.1f}")

    print("\nStand-alone utility cost of the existing HENs (Lo 2021 Table 2 prices):")
    total = 0.0
    for name, plant in data["plants"].items():
        cost = 0.0
        for u in plant["existing_hen"]["units"]:
            if u["kind"] in ("heater", "cooler"):
                util = u["hot"] if u["kind"] == "heater" else u["cold"]
                cost += u["duty_kW"] * data["utilities"][name][util]["cost_usd_per_kW_yr"]
        total += cost
        print(f"  {name}: {cost:,.1f} $/yr")
    print(f"  site: {total:,.1f} $/yr")
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
