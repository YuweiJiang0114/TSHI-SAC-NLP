#!/usr/bin/env python
"""Plot exported total-site retrofit networks without importing training code."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

TEXT = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e6e5e1"
HOT = "#c8534a"
COLD = "#528bac"
LOOP = "#8673a8"


def number(*values):
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            result = float(value)
            if math.isfinite(result):
                return result
        except (TypeError, ValueError):
            pass
    return float("nan")


def pick(mapping, *keys, default=None):
    if isinstance(mapping, dict):
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return default


def num(mapping, *keys):
    return number(*(mapping.get(k) for k in keys)) if isinstance(mapping, dict) else number()


def fmt(value, digits=2):
    value = number(value)
    return f"{value:,.{digits}f}" if math.isfinite(value) else "—"


def records(value):
    if isinstance(value, list):
        return [dict(v) for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            if isinstance(item, dict):
                result.append(dict(item, id=item.get("id", key)))
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict):
                        result.append(dict(child, stream=child.get("stream", key)))
        return result
    return []


def stream_id(value):
    if isinstance(value, dict):
        value = pick(value, "id", "name", "stream", default="?")
    text = str(value)
    if re.match(r"^(loop|if|intermediate(?:_fluid)?)(?:$|[_:\s])", text, re.I):
        return "LOOP"
    return text


def style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.edgecolor": SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "lines.linewidth": 1.44,  # approximately 2 px at 100 dpi
        "svg.fonttype": "none",
    })


def save_figure(fig, basename):
    basename = Path(basename)
    if basename.suffix.lower() in {".png", ".svg"}:
        basename = basename.with_suffix("")
    basename.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(str(basename) + suffix, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plant_items(design):
    plants = design.get("plants", {})
    if isinstance(plants, dict):
        return list(plants.items())
    return [
        (str(pick(p, "name", "plant", "id", default=f"P{i + 1}")), p)
        for i, p in enumerate(plants) if isinstance(p, dict)
    ] if isinstance(plants, list) else []


def role_for(design, name):
    roles = design.get("roles", {})
    if isinstance(roles, dict) and name in roles:
        value = str(roles[name])
    elif isinstance(roles, list) and re.fullmatch(r"P[123]", name):
        index = int(name[1:]) - 1
        value = str(roles[index]) if index < len(roles) else ""
    else:
        case = str(design.get("case", ""))
        index = int(name[1:]) - 1 if re.fullmatch(r"P[123]", name) else -1
        value = case[index] if 0 <= index < len(case) else ""
    return {
        "+": "source", "-": "sink", "0": "outside coalition",
        "None": "outside coalition",
    }.get(value, value or "unknown")


def interface_for(design, name):
    interfaces = design.get("interfaces", {})
    if isinstance(interfaces, dict):
        result = interfaces.get(name, {})
        return result if isinstance(result, dict) else {}
    return {}


def normalize_plant(design, name, raw):
    """Return a plotting view; retain unknown quantities as NaN."""
    exchangers = []
    for index, item in enumerate(records(raw.get("exchangers", [])), 1):
        base = num(item, "area_existing", "existing_area", "base_area", "original_area", "A_existing")
        extra = num(item, "augmentation_area", "added_area", "area_added", "A_aug")
        installed = num(item, "installed_area", "area_installed", "A_installed")
        if not math.isfinite(installed) and math.isfinite(base) and math.isfinite(extra):
            installed = base + extra
        required = num(item, "area_required", "required_area", "A_required", "A_req")
        housing = item.get("housing_unit")
        if "LOOP" in (str(item.get("hot")), str(item.get("cold"))) or housing == "new":
            installed = required  # purchased unit (intermediate-fluid or Strategy III new unit)
        generic_area = num(item, "area", "A", "area_m2")
        installed = number(installed, generic_area)
        required = number(required, generic_area)
        if not math.isfinite(extra) and math.isfinite(installed) and math.isfinite(base):
            extra = max(0.0, installed - base)
        kind = str(pick(item, "kind", "type", default="exchanger"))
        hot = stream_id(pick(item, "hot", "hot_stream", "hot_id", default="?"))
        cold = stream_id(pick(item, "cold", "cold_stream", "cold_id", default="?"))
        exchangers.append({
            "id": str(pick(item, "id", "name", default=f"E{index}")),
            "kind": kind, "hot": hot, "cold": cold,
            "duty": num(item, "duty", "duty_kW", "Q", "q", "Q_kW"),
            "required": required, "installed": installed, "augmentation": extra,
            "housing": housing if housing not in (None,) and housing != str(pick(item, "id", "name", default="")) else None,
            "intermediate": "LOOP" in (hot, cold)
                            or "intermediate" in kind.lower() or "loop" in kind.lower(),
            "augmented": (math.isfinite(extra) and extra > 1e-8)
                         or "augment" in kind.lower(),
        })

    utilities = []
    for item in records(raw.get("utilities", [])):
        utilities.append({
            "stream": stream_id(pick(item, "stream", "stream_id", "id", default="?")),
            "utility": str(pick(item, "utility", "kind", "type", "name", default="?")),
            "duty": num(item, "duty", "duty_kW", "Q", "q", "Q_kW"),
            "cost": num(item, "annual_cost", "cost_annual", "annual_utility_cost", "cost"),
        })

    chains = {}
    for stream, groups in (raw.get("chains", {}) or {}).items():
        parsed = []
        if isinstance(groups, dict):
            groups = pick(groups, "groups", "chain", "exchangers", default=[])
        if isinstance(groups, str):
            groups = [groups]
        for group in groups if isinstance(groups, (list, tuple)) else []:
            if isinstance(group, dict):
                group = pick(group, "exchangers", "ids", "parallel", "id", default=[])
            if isinstance(group, (str, int)):
                group = [group]
            if isinstance(group, (list, tuple)):
                ids = [str(pick(v, "id", "name", default="?")) if isinstance(v, dict)
                       else str(v) for v in group]
                if ids:
                    parsed.append(ids)
        chains[stream_id(stream)] = parsed

    hot_streams = list(dict.fromkeys(e["hot"] for e in exchangers if e["hot"] != "LOOP"))
    cold_streams = list(dict.fromkeys(e["cold"] for e in exchangers if e["cold"] != "LOOP"))
    for utility in utilities:
        stream = utility["stream"]
        if stream in hot_streams + cold_streams or stream == "LOOP":
            continue
        label = utility["utility"].lower()
        if stream.upper().startswith("H") or label in {"cw", "cold", "cold utility"}:
            hot_streams.append(stream)
        else:
            cold_streams.append(stream)

    loop_present = any("LOOP" in (e["hot"], e["cold"]) for e in exchangers)
    streams = hot_streams + (["LOOP"] if loop_present else []) + cold_streams
    temperature_data = {}
    for item in records(raw.get("streams", [])):
        temperature_data[stream_id(pick(item, "id", "name", "stream", default="?"))] = item
    ends = raw.get("stream_end_temperatures", {}) or {}
    for stream, data in ends.items() if isinstance(ends, dict) else []:
        stream = stream_id(stream)
        merged = dict(temperature_data.get(stream, {}))
        if isinstance(data, dict):
            merged.update(data)
        else:
            merged["actual_end"] = data
        temperature_data[stream] = merged
    # Supply/target temperatures are case data (the export holds only end temperatures).
    try:
        from tshi_decomp.case import load_case
        case_streams = load_case()[0][name].streams
    except (ImportError, KeyError, OSError, ValueError):
        case_streams = {}
    for stream, s in case_streams.items():
        temperature_data.setdefault(stream, {}).update({"supply": s.t_in, "target": s.t_out})
    interface = interface_for(design, name)
    if loop_present:
        temperature_data["LOOP"] = {
            "supply": pick(interface, "t_in", "T_in", "Tin"),
            "target": pick(interface, "t_out", "T_out", "Tout"),
        }

    costs = raw.get("costs", {}) or {}
    hot_cost = num(costs, "hot_utility", "hot_utility_cost", "hot_utility_annual_cost",
                   "annual_hot_utility", "HU")
    cold_cost = num(costs, "cold_utility", "cold_utility_cost", "cold_utility_annual_cost",
                    "annual_cold_utility", "CU")
    # Prefer explicit costs. Otherwise classify utilities using their stream/type.
    hot_rows, cold_rows = [], []
    for utility in utilities:
        label = utility["utility"].lower()
        if (utility["stream"] in cold_streams
                or any(s in label for s in ("hot", "steam", "heating"))
                or label in {"hu", "h"}):
            hot_rows.append(utility["cost"])
        elif (utility["stream"] in hot_streams
              or any(s in label for s in ("cold", "cool", "water"))
              or label in {"cu", "cw", "c"}):
            cold_rows.append(utility["cost"])

    def known_sum(values):
        return sum(values) if all(math.isfinite(v) for v in values) else number()

    if "utilities" in raw:
        hot_cost = number(hot_cost, known_sum(hot_rows))
        cold_cost = number(cold_cost, known_sum(cold_rows))
    site_tac = design.get("plant_TAC", {})
    tac = number(num(costs, "TAC", "tac", "total", "annual_total", "total_annual_cost"),
                 num(raw, "TAC", "tac"),
                 site_tac.get(name) if isinstance(site_tac, dict) else None)
    capital = num(costs, "annualised_capital", "annualized_capital",
                  "annual_capital", "capital_annual", "annual_capital_cost")
    if not math.isfinite(capital) and all(map(math.isfinite, (tac, hot_cost, cold_cost))):
        capital = tac - hot_cost - cold_cost
    structure = raw.get("structure", "—")
    if isinstance(structure, dict):
        structure = pick(structure, "label", "name", "id", default=str(structure))
    return {
        "name": name, "role": role_for(design, name), "structure": str(structure),
        "exchangers": exchangers, "utilities": utilities, "chains": chains,
        "hot_streams": hot_streams, "cold_streams": cold_streams,
        "streams": streams, "temperatures": temperature_data,
        "hot_cost": hot_cost, "cold_cost": cold_cost, "capital": capital, "TAC": tac,
    }


def exchanger_order(plant):
    """Topologically order hot chains; resolve cycles with stable first occurrence."""
    ids = [e["id"] for e in plant["exchangers"]]
    priority, predecessors = [], {eid: set() for eid in ids}
    for stream in plant["hot_streams"] + ["LOOP"]:
        if not any(e["hot"] == stream for e in plant["exchangers"]):
            continue
        previous = []
        for group in plant["chains"].get(stream, []):
            current = [eid for eid in group if eid in predecessors]
            for eid in current:
                if eid not in priority:
                    priority.append(eid)
                predecessors[eid].update(previous)
            if current:
                previous = current
    for stream in plant["cold_streams"]:
        for group in reversed(plant["chains"].get(stream, [])):
            for eid in group:
                if eid in predecessors and eid not in priority:
                    priority.append(eid)
    priority += [eid for eid in ids if eid not in priority]
    result = []
    while len(result) < len(ids):
        remaining = [eid for eid in priority if eid not in result]
        ready = [eid for eid in remaining if predecessors[eid].issubset(result)]
        result.append((ready or remaining)[0])
    return result


def draw_plant(ax, plant):
    order = exchanger_order(plant)
    positions = {eid: 1.8 + i * 2.3 for i, eid in enumerate(order)}
    left, right = 0.0, max(3.6, 3.6 + (len(order) - 1) * 2.3)
    ys = {stream: float(len(plant["streams"]) - i) for i, stream in enumerate(plant["streams"])}
    endpoint_y = {}
    utility_counts = {}
    for utility in plant["utilities"]:
        utility_counts[utility["stream"]] = utility_counts.get(utility["stream"], 0) + 1
    margin = 1.3 + 0.65 * max(utility_counts.values(), default=0)

    for stream in plant["streams"]:
        y = ys[stream]
        is_hot = stream in plant["hot_streams"] or any(
            e["hot"] == stream for e in plant["exchangers"])
        direction = 1 if is_hot else -1
        color = LOOP if stream == "LOOP" else HOT if is_hot else COLD
        linestyle = "--" if stream == "LOOP" else "-"
        intervals = []
        for group in plant["chains"].get(stream, []):
            group = [eid for eid in group if eid in positions]
            if len(group) <= 1:
                continue
            start = min(positions[eid] for eid in group) - 0.55
            end = max(positions[eid] for eid in group) + 0.55
            intervals.append((start, end))
            offsets = np.linspace(-0.27, 0.27, len(group))
            for eid, offset in zip(group, offsets):
                by = y + offset
                endpoint_y[(stream, eid)] = by
                ax.plot([start, start + 0.25, end - 0.25, end],
                        [y, by, by, y], color=color, ls=linestyle, lw=1.44)
        cursor = left
        for start, end in sorted(intervals):
            if start > cursor:
                ax.plot([cursor, start], [y, y], color=color, ls=linestyle, lw=1.44)
            cursor = max(cursor, end)
        ax.plot([cursor, right], [y, y], color=color, ls=linestyle, lw=1.44)
        end_x = right if direction == 1 else left
        ax.annotate("", xy=(end_x, y), xytext=(end_x - direction * 0.4, y),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.44})
        ax.text(left - margin - 0.7, y, stream, va="center", weight="bold", fontsize=10)

        data = plant["temperatures"].get(stream, {})
        supply = num(data, "supply", "T_supply", "Ts", "T_s", "T_in", "Tin")
        target = num(data, "target", "T_target", "Tt", "T_t", "T_out", "Tout")
        actual = num(data, "actual_end", "end", "T_end", "actual")
        supply_x, target_x = (left, right) if direction == 1 else (right, left)
        ax.text(supply_x, y + 0.35, f"{fmt(supply, 1)} °C",
                ha="center", fontsize=8, color=SECONDARY)
        label = f"{fmt(target, 1)} °C"
        if not math.isfinite(target) and math.isfinite(actual):
            label = f"end {fmt(actual, 1)} °C"
        ax.text(target_x, y + 0.35, label, ha="center", fontsize=8, color=SECONDARY)

    for exchanger in plant["exchangers"]:
        if exchanger["hot"] not in ys or exchanger["cold"] not in ys:
            continue
        x = positions[exchanger["id"]]
        yh = endpoint_y.get((exchanger["hot"], exchanger["id"]), ys[exchanger["hot"]])
        yc = endpoint_y.get((exchanger["cold"], exchanger["id"]), ys[exchanger["cold"]])
        ax.plot([x, x], [yc, yh], color=SECONDARY, lw=1.2, zorder=3)
        face = "#d3d2ce" if exchanger["intermediate"] else "white"
        for y in (yh, yc):
            ax.add_patch(Circle((x, y), 0.12, facecolor=face,
                                edgecolor=TEXT, lw=1.2, zorder=5))
            if exchanger["augmented"]:
                ax.add_patch(Circle((x, y), 0.18, facecolor="none",
                                    edgecolor=TEXT, lw=1, zorder=5))
        ax.text(x, max(yh, yc) + 0.24, exchanger["id"], ha="center", fontsize=8)
        ax.text(x + 0.18, (yh + yc) / 2,
                f'{fmt(exchanger["duty"], 1)} kW\n'
                f'{fmt(exchanger["required"], 1)} m² req.\n'
                f'{fmt(exchanger["installed"], 1)} m² inst.'
                + (("\nnew unit" if exchanger["housing"] == "new" else f'\nin unit {exchanger["housing"]}')
                   if exchanger.get("housing") else ""),
                va="center", ha="left", fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2}, zorder=6)

    counts = {}
    for utility in plant["utilities"]:
        stream = utility["stream"]
        if stream not in ys:
            continue
        is_hot = stream in plant["hot_streams"] or any(
            e["hot"] == stream for e in plant["exchangers"])
        direction = 1 if is_hot else -1
        counts[stream] = counts.get(stream, 0) + 1
        x = (right if direction == 1 else left) + direction * 0.65 * counts[stream]
        y = ys[stream]
        off = not (utility["duty"] > 1e-6)  # unused: original unit switched off
        edge = GRID if off else SECONDARY
        ax.plot([right if direction == 1 else left, x], [y, y],
                color=edge, lw=1.2, ls=":" if off else "-")
        ax.add_patch(Circle((x, y), 0.18, facecolor="white",
                            edgecolor=edge, lw=1.2, zorder=5, ls=":" if off else "-"))
        ax.text(x, y, utility["utility"], ha="center", va="center", fontsize=7,
                color=SECONDARY if off else TEXT, zorder=6)
        ax.text(x, y - 0.35, "off" if off else f'{fmt(utility["duty"], 1)} kW',
                ha="center", fontsize=7.5, color=SECONDARY if off else TEXT)

    ax.set_xlim(left - margin - 1.2, right + margin + 0.5)
    ax.set_ylim(0.25, max(2, len(plant["streams"])) + 1)
    ax.set_title(f'{plant["name"]}: {plant["role"]}   |   {plant["structure"]}',
                 loc="left", fontsize=12, pad=14)
    ax.set_axis_off()
    if not plant["streams"]:
        ax.text(0.5, 0.5, "No stream/exchanger data exported",
                transform=ax.transAxes, ha="center", color=SECONDARY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with args.design.open(encoding="utf-8-sig") as handle:
        design = json.load(handle)
    if isinstance(design.get("result"), dict):
        design = design["result"]
    plants = [normalize_plant(design, name, raw)
              for name, raw in plant_items(design) if isinstance(raw, dict)]
    plants = [p for p in plants if p["role"] not in {"0", "outside coalition", "inactive"}]
    style()
    if not plants:
        raise SystemExit("No participating plants found in the design.")
    heights = [max(3.4, 0.95 * len(p["streams"]) + 1.8) for p in plants]
    width = max(13.5, 2.0 * max(len(p["exchangers"]) for p in plants) + 6)
    fig, axes = plt.subplots(len(plants), 1, figsize=(width, sum(heights)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, plant in zip(axes[:, 0], plants):
        draw_plant(ax, plant)
    fig.suptitle(
        f'Case {design.get("case", "—")}   |   '
        f'T_return {fmt(design.get("T_return"), 1)} °C   |   '
        f'T_supply {fmt(design.get("T_supply"), 1)} °C   |   '
        f'Loop flow {fmt(design.get("loop_flow"), 3)}',
        fontsize=14, y=0.995)
    handles = [
        Line2D([], [], color=HOT, label="Hot process stream"),
        Line2D([], [], color=COLD, label="Cold process stream"),
        Line2D([], [], color=LOOP, ls="--", label="Intermediate fluid"),
        Line2D([], [], color=TEXT, marker="o", mfc="#d3d2ce", ls="none",
               label="Intermediate exchanger"),
        Line2D([], [], color=TEXT, marker=r"$\odot$", markersize=11, ls="none",
               label="Augmented existing exchanger"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.text(0.99, 0.006, "Areas: required / installed after retrofit. — = not in export.", ha="right",
             fontsize=8, color=SECONDARY)
    fig.tight_layout(rect=(0, 0.055, 1, 0.97), h_pad=2.5)
    output = args.out or args.design.with_name(args.design.stem + "_network")
    save_figure(fig, output)
    print(f"Wrote {output}.png and {output}.svg")


if __name__ == "__main__":
    main()
