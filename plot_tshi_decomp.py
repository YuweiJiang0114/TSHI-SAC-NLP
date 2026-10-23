#!/usr/bin/env python
"""Plot decomposed TSHI study results and produce a Markdown report."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# This sibling module imports only plotting libraries and the standard library.
from plot_tshi_network import (
    TEXT, SECONDARY, GRID, HOT, COLD, LOOP, number, num, pick, fmt,
    style, save_figure, normalize_plant, plant_items, interface_for,
)

CASES = ["-+0", "+-0", "-0+", "+0-", "0-+", "0+-",
         "--+", "-+-", "-++", "+--", "+-+", "++-"]
SI = dict(zip(CASES, [53876, 0, 138019, 0, 15669, 0,
                      138019, 53876, 176702, 0, 15669, 0]))
SIII = dict(SI, **{"-0+": 140159, "--+": 140159, "-++": 178702})
METHODS = ["sac", "de", "random"]
COLORS = {"sac": "#2a78d6", "de": "#eb6834", "random": "#1baf7a"}
LABELS = {"sac": "SAC", "de": "DE", "random": "Random"}
PAPER_TAC = 453648.0
PAPER_UTILITY = 330453.0 + 81021.0
PAPER_CAPITAL = (79401.0 + 149614.0 + 1000.0 + 82609.0) * 0.1349
PAPER_SHAPLEY = np.array([85660.0, 24486.0, 66557.0])
BASELINE = 630350.0


def feasible(row):
    value = row.get("strictly_feasible")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def load_json(path, notes):
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError) as error:
        notes.append(f"{path}: {error}")
        return {}


def load_csv(path, notes):
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return sorted(rows, key=lambda row: number(row.get("eval"), 0))
    except (OSError, ValueError, csv.Error) as error:
        notes.append(f"{path}: {error}")
        return []


def scalar(value, *keys):
    if isinstance(value, dict):
        return num(value, *keys)
    return number(value)


def valid(values):
    return np.array([v for v in values if math.isfinite(number(v))], dtype=float)


def stats(values):
    values = valid(values)
    if not len(values):
        return (number(),) * 4
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            float(np.min(values)), float(np.max(values)))


def row_total(run):
    return max([number(row.get("eval"), 0) for row in run["log"] + run["policy"]] + [0])


def case_value(run, case):
    value = run["summary"].get("cases", {}).get(case)
    result = scalar(value, "TACS", "best_TACS", "best_case_TACS", "tacs")
    if math.isfinite(result):
        return result
    value = run["bests"].get(case)
    result = scalar(value, "TACS", "best_TACS", "tacs")
    if math.isfinite(result):
        return result
    values = [num(row, "TACS") for row in run["log"]
              if row.get("case") == case and feasible(row)]
    return max(valid(values), default=number())


def policy_value(run, case):
    value = run["final"].get(case)
    if not isinstance(value, dict):
        value = run["summary"].get("final_policy", {}).get(case, {})
    return num(value, "TACS", "tacs")


def covered_cases(run):
    return (set(run["summary"].get("cases", {}))
            | set(run["bests"])
            | {row.get("case") for row in run["log"]})


def shapley_values(run):
    value = run["summary"].get("shapley_best", {})
    if isinstance(value, dict):
        return np.array([scalar(value.get(p), "value", "TACS", "allocation")
                         for p in ("P1", "P2", "P3")])
    if isinstance(value, list) and len(value) == 3:
        return np.array([number(v) for v in value])
    return np.full(3, np.nan)


def full_shapley(run):
    return set(CASES).issubset(covered_cases(run)) and np.isfinite(shapley_values(run)).all()


def grand_info(run):
    summary = run["summary"]
    design = run["design"]
    case = pick(summary, "grand_coalition_best_case", default=design.get("case", ""))
    tac = number(summary.get("grand_coalition_best_TAC"), design.get("TAC"))
    tacs = number(summary.get("grand_coalition_best_TACS"), design.get("TACS"))
    rows = [row for row in run["log"]
            if len(str(row.get("case", ""))) == 3
            and "0" not in str(row.get("case", "")) and feasible(row)
            and math.isfinite(num(row, "TACS"))]
    best_row = max(rows, key=lambda row: num(row, "TACS"), default={})
    if not math.isfinite(tacs):
        tacs = num(best_row, "TACS")
        tac = number(tac, num(best_row, "TAC"))
        case = case or best_row.get("case", "")
    if not math.isfinite(tac) and math.isfinite(tacs):
        tac = BASELINE - tacs

    entry = run["bests"].get(case, {})
    best_eval = num(entry, "eval") if isinstance(entry, dict) else number()
    if not math.isfinite(best_eval) and math.isfinite(tacs):
        matches = [row for row in rows if row.get("case") == case
                   and math.isclose(num(row, "TACS"), tacs, rel_tol=1e-7, abs_tol=0.05)]
        if matches:
            best_eval = num(matches[0], "eval")
    gap = number(summary.get("gap_to_paper_strategy_I_pct"))
    if not math.isfinite(gap) and math.isfinite(tac):
        gap = 100 * (tac - PAPER_TAC) / PAPER_TAC
    wall = number(summary.get("wall_s"),
                  max(valid([num(row, "wall_s") for row in run["log"]]), default=number()))
    return {"case": case or "—", "TAC": tac, "TACS": tacs,
            "gap": gap, "eval": best_eval, "wall": wall}


def discover(root, notes):
    candidates = [root] if (root / "log.csv").exists() else [
        path for path in root.iterdir() if path.is_dir()]
    runs = []
    for folder in sorted(candidates):
        match = re.fullmatch(r"(sac|de|random)_seed(.+)", folder.name, re.I)
        if not match:
            continue
        method, seed = match.group(1).lower(), match.group(2)
        run = {
            "folder": folder, "method": method, "seed": seed,
            "summary": load_json(folder / "summary.json", notes),
            "design": load_json(folder / "best_design.json", notes),
            "bests": load_json(folder / "best_designs.json", notes),
            "final": load_json(folder / "final_policy.json", notes),
            "config": load_json(folder / "config.json", notes),
            "log": load_csv(folder / "log.csv", notes),
            "policy": load_csv(folder / "policy_curve.csv", notes),
        }
        if not (folder / "log.csv").exists():
            notes.append(f"{folder.name}: log.csv missing; convergence/samples unavailable.")
        run["grand"] = grand_info(run)
        runs.append(run)
    return runs


def clean_axes(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def empty(ax, message):
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", color=SECONDARY)


def method_list(runs):
    return [method for method in METHODS if any(r["method"] == method for r in runs)]


def step_curve(run):
    x, y, best = [], [], -np.inf
    for row in run["log"]:
        if row.get("case") != "-++" or not feasible(row):
            continue
        value, evaluation = num(row, "TACS"), num(row, "eval")
        if not all(map(math.isfinite, (value, evaluation))):
            continue
        best = max(best, value / 1000)
        if x and evaluation == x[-1]:
            y[-1] = best
        else:
            x.append(evaluation)
            y.append(best)
    if x and row_total(run) > x[-1]:
        x.append(row_total(run))
        y.append(best)
    return np.array(x), np.array(y)


def stepped_values(x, y, grid):
    result = np.full(len(grid), np.nan)
    if len(x):
        indices = np.searchsorted(x, grid, side="right") - 1
        mask = (indices >= 0) & (grid <= x[-1])
        result[mask] = y[indices[mask]]
    return result


def convergence(runs, output):
    curves = [(run, *step_curve(run)) for run in runs]
    available = [(run, x, y) for run, x, y in curves if len(x)]
    needs_full = any(np.min(y) < 100 for _, _, y in available)
    fig, axes = plt.subplots(2 if needs_full else 1, 1,
                             figsize=(10, 7.5 if needs_full else 5),
                             sharex=needs_full, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]} if needs_full else None)
    main = axes[0, 0]
    for ax in axes[:, 0]:
        for method in method_list(runs):
            selected = [(run, x, y) for run, x, y in available if run["method"] == method]
            if not selected:
                continue
            for _, x, y in selected:
                ax.step(x, y, where="post", color=COLORS[method], alpha=0.25, lw=0.9)
            grid = np.unique(np.concatenate([x for _, x, _ in selected]))
            matrix = np.vstack([stepped_values(x, y, grid) for _, x, y in selected])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(matrix, axis=0)
            ax.step(grid, median, where="post", color=COLORS[method],
                    lw=2.3, label=f"{LABELS[method]} median")
        ax.axhline(SI["-++"] / 1000, color=SECONDARY, ls="--", lw=1.1)
        ax.axhline(SIII["-++"] / 1000, color=SECONDARY, ls=":", lw=1.1)
        clean_axes(ax)
        ax.set_ylabel("Best feasible TACS (k$/yr)")
        ax.margins(x=0.015)
    ymax = max([185.0] + [float(np.max(y)) + 3 for _, _, y in available])
    if available:
        best_seen = max(float(np.max(y)) for _, _, y in available)
        ymin = min(100.0, best_seen - 20) if best_seen < 100 else 100.0
        main.set_ylim(ymin, ymax)
    else:
        main.set_ylim(100, 185)
        empty(main, 'No strictly feasible "-++" evaluations')
    main.annotate("Paper Strategy I: 176.702",
                  xy=(1, 176.702), xycoords=("axes fraction", "data"),
                  xytext=(5, -9), textcoords="offset points",
                  fontsize=8, color=SECONDARY, va="top")
    main.annotate("Paper Strategy III: 178.702",
                  xy=(1, 178.702), xycoords=("axes fraction", "data"),
                  xytext=(5, 4), textcoords="offset points",
                  fontsize=8, color=SECONDARY, va="bottom")
    if main.get_legend_handles_labels()[0]:
        main.legend(frameon=False, loc="lower right")
    main.set_title('Grand-coalition case "-++": best-so-far convergence', loc="left")
    axes[-1, 0].set_xlabel("Total evaluations")
    if needs_full:
        axes[-1, 0].set_title("Full range", loc="left", fontsize=9, color=SECONDARY)
    fig.tight_layout(rect=(0, 0, 0.83, 1))
    save_figure(fig, output / "convergence_grand")


def case_label(case):
    words = {"-": "sink", "+": "source", "0": "outside coalition"}
    return ", ".join(f"P{i + 1} {words[c]}" for i, c in enumerate(case)) + f"  ({case})"


def cases_plot(runs, output):
    methods = method_list(runs)
    fig, ax = plt.subplots(figsize=(13.5, 10))
    ys = np.arange(len(CASES))
    height = 0.72 / max(1, len(methods))
    for i, method in enumerate(methods):
        positions = ys + (i - (len(methods) - 1) / 2) * height
        first = True
        for case, y in zip(CASES, positions):
            mean, _, low, high = stats([case_value(r, case) / 1000
                                       for r in runs if r["method"] == method])
            if not math.isfinite(mean):
                continue
            ax.barh(y, mean, height=height * 0.85, color=COLORS[method],
                    label=LABELS[method] if first else None)
            ax.errorbar(mean, y, xerr=[[max(0, mean - low)], [max(0, high - mean)]],
                        fmt="none", ecolor=TEXT, capsize=3, lw=1)
            first = False
    ax.scatter([SI[c] / 1000 for c in CASES], ys, marker="D", s=26,
               color=TEXT, label="Paper Strategy I", zorder=5)
    different = [i for i, c in enumerate(CASES) if SIII[c] != SI[c]]
    ax.scatter([SIII[CASES[i]] / 1000 for i in different], ys[different],
               marker="D", s=38, facecolors="white", edgecolors=TEXT,
               label="Paper Strategy III (different values)", zorder=6)
    ax.set_yticks(ys, [case_label(c) for c in CASES], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Best TACS (k$/yr); mean and seed min–max")
    ax.set_title("Role-case savings", loc="left")
    clean_axes(ax, "x")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output / "cases")


def policy_rows(run):
    if run["policy"]:
        return run["policy"]
    return [row for row in run["log"] if row.get("phase") == "policy"]


def sac_policy(runs, output):
    selected = [r for r in runs if r["method"] == "sac"]
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), squeeze=False)
    handles = {}
    any_infeasible = False
    for case, ax in zip(CASES, axes.flat):
        found = False
        for run in selected:
            rows = [row for row in policy_rows(run)
                    if row.get("case") == case and math.isfinite(num(row, "eval"))
                    and math.isfinite(num(row, "TACS"))]
            if not rows:
                continue
            x = np.array([num(row, "eval") for row in rows])
            y = np.array([num(row, "TACS") / 1000 for row in rows])
            line, = ax.plot(x, y, color=COLORS["sac"], alpha=0.65, lw=1.2,
                            label=f'Seed {run["seed"]}')
            handles.setdefault(run["seed"], line)
            bad = np.array([not feasible(row) for row in rows])
            if bad.any():
                ax.scatter(x[bad], y[bad], facecolors="white",
                           edgecolors=COLORS["sac"], s=25, zorder=4)
                any_infeasible = True
            found = True
        ax.axhline(SI[case] / 1000, color=SECONDARY, ls="--", lw=1)
        ax.set_title(case, loc="left")
        clean_axes(ax)
        if not found:
            empty(ax, "No policy probes")
    for ax in axes[-1, :]:
        ax.set_xlabel("Total evaluations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Policy TACS (k$/yr)")
    legend = list(handles.values())
    labels = [f"Seed {seed}" for seed in handles]
    legend.append(Line2D([], [], color=SECONDARY, ls="--"))
    labels.append("Paper Strategy I")
    if any_infeasible:
        legend.append(Line2D([], [], color=COLORS["sac"], marker="o",
                             mfc="white", ls="none"))
        labels.append("Not strictly feasible")
    fig.legend(legend, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("SAC deterministic-policy probes", x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    save_figure(fig, output / "sac_policy")


def sac_samples(runs, output):
    phase_colors = {"warmup": "#aaa9a5", "sac": COLORS["sac"], "policy": "#183b63"}
    fig, ax = plt.subplots(figsize=(10, 5))
    found = False
    for phase, color in phase_colors.items():
        rows = [row for run in runs if run["method"] == "sac" for row in run["log"]
                if row.get("case") == "-++" and row.get("phase") == phase
                and math.isfinite(num(row, "eval")) and math.isfinite(num(row, "TACS"))]
        if rows:
            ax.scatter([num(row, "eval") for row in rows],
                       np.clip([num(row, "TACS") / 1000 for row in rows], -100, 185),
                       color=color, s=10 if phase != "policy" else 22,
                       alpha=0.38 if phase != "policy" else 0.85,
                       label={"warmup": "Warmup", "sac": "SAC samples",
                              "policy": "Policy probes"}[phase], rasterized=False)
            found = True
    if not found:
        empty(ax, 'No SAC evaluations for "-++"')
    else:
        ax.legend(frameon=False)
    ax.set_ylim(-103, 188)
    ax.set_xlabel("Total evaluations")
    ax.set_ylabel("TACS (k$/yr), clipped to [−100, 185]")
    ax.set_title('SAC evaluations of "-++" across seeds', loc="left")
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, output / "sac_samples_grand")


def overall_design(runs):
    candidates = []
    for run in runs:
        designs = [run["design"]]
        designs += [entry.get("result") for entry in run["bests"].values()
                    if isinstance(entry, dict)]
        for design in designs:
            if not isinstance(design, dict) or not design.get("plants"):
                continue
            case = str(design.get("case", ""))
            if len(case) != 3 or "0" in case:
                continue
            if "strictly_feasible" in design and not feasible(design):
                continue
            value = number(design.get("TACS"))
            if not math.isfinite(value):
                value = BASELINE - num(design, "TAC")
            if math.isfinite(value):
                candidates.append((value, run, design))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def design_costs(best, output, notes):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    if best is None:
        for ax in axes:
            empty(ax, "No exported grand-coalition design")
        save_figure(fig, output / "best_design_costs")
        return
    _, run, design = best
    plants = [normalize_plant(design, name, raw)
              for name, raw in plant_items(design) if isinstance(raw, dict)]
    categories = [("hot_cost", "Hot utility", HOT), ("cold_cost", "Cold utility", COLD),
                  ("capital", "Annualised plant capital", "#bcbab3")]
    ys = np.arange(len(plants) + 1)
    left = np.zeros(len(plants))
    for key, label, color in categories:
        values = np.array([p[key] for p in plants]) / 1000
        axes[0].barh(ys[:-1], np.nan_to_num(values, nan=0), left=left,
                     color=color, label=label, height=0.65)
        left += np.nan_to_num(values, nan=0)
    loop_cost = num(design, "loop_annual_cost")
    if math.isfinite(loop_cost):
        axes[0].barh(ys[-1], loop_cost / 1000, color=LOOP,
                     label="Intermediate fluid", height=0.65)
    incomplete = []
    for i, plant in enumerate(plants):
        if any(not math.isfinite(plant[key]) for key, _, _ in categories):
            incomplete.append(plant["name"])
            axes[0].text(left[i] + 2, i, "incomplete costs", va="center",
                         fontsize=8, color=SECONDARY)
    if not math.isfinite(loop_cost):
        axes[0].text(2, ys[-1], "cost unavailable", va="center", color=SECONDARY)
    if incomplete:
        notes.append("Cost export incomplete for: " + ", ".join(incomplete))
    axes[0].set_yticks(ys, [p["name"] for p in plants] + ["Intermediate fluid"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Annual cost (k$/yr)")
    axes[0].set_title("Best exported design: cost allocation", loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    clean_axes(axes[0], "x")

    utility = sum(p["hot_cost"] + p["cold_cost"] for p in plants)
    capital = sum(p["capital"] for p in plants) + loop_cost
    total = num(design, "TAC")
    actual = [utility, capital, total]
    paper = [PAPER_UTILITY, PAPER_CAPITAL, PAPER_TAC]
    x = np.arange(3)
    axes[1].bar(x - 0.18, np.array(paper) / 1000, width=0.34,
                color="#bcbab3", label="Paper Strategy I")
    for i, value in enumerate(actual):
        if math.isfinite(value):
            axes[1].bar(i + 0.18, value / 1000, width=0.34,
                        color=COLORS[run["method"]],
                        label="Best exported design" if i == next(
                            j for j, v in enumerate(actual) if math.isfinite(v)) else None)
            axes[1].text(i + 0.18, value / 1000 + 5, fmt(value / 1000, 1),
                        ha="center", fontsize=8)
        else:
            axes[1].text(i + 0.18, 5, "—", ha="center")
        axes[1].text(i - 0.18, paper[i] / 1000 + 5, fmt(paper[i] / 1000, 1),
                    ha="center", fontsize=8)
    axes[1].set_xticks(x, ["Utility", "Annualised\ncapital", "TAC"])
    axes[1].set_ylabel("Annual cost (k$/yr)")
    axes[1].set_title("Comparison with Table C1", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].margins(y=0.15)
    clean_axes(axes[1])
    fig.suptitle(f'{run["folder"].name}, case {design.get("case", "—")}', fontsize=11)
    fig.tight_layout()
    save_figure(fig, output / "best_design_costs")


def shapley_plot(runs, output):
    eligible = [run for run in runs if full_shapley(run)]
    methods = method_list(eligible)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.75 / (1 + len(methods))
    x = np.arange(3)
    offset = -0.375 + width / 2
    ax.bar(x + offset, PAPER_SHAPLEY / 1000, width=width * 0.9,
           color="#bcbab3", label="Paper Strategy I")
    for i, method in enumerate(methods, 1):
        matrix = np.vstack([shapley_values(run) for run in eligible
                            if run["method"] == method]) / 1000
        means = np.mean(matrix, axis=0)
        lows, highs = np.min(matrix, axis=0), np.max(matrix, axis=0)
        positions = x + offset + i * width
        ax.bar(positions, means, width=width * 0.9, color=COLORS[method],
               label=LABELS[method])
        ax.errorbar(positions, means,
                    yerr=np.vstack((np.maximum(0, means - lows),
                                    np.maximum(0, highs - means))),
                    fmt="none", ecolor=TEXT, capsize=3, lw=1)
    if not eligible:
        ax.text(0.5, 0.94, "No runs cover all 12 cases with exported Shapley values",
                transform=ax.transAxes, ha="center", color=SECONDARY, fontsize=9)
    ax.set_xticks(x, ["P1", "P2", "P3"])
    ax.set_ylabel("Shapley allocation (k$/yr)")
    ax.set_title("Coalition savings allocation; seed mean and min–max", loc="left")
    ax.legend(frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, output / "shapley")


def mdcell(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def table(headers, rows):
    lines = ["| " + " | ".join(map(mdcell, headers)) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(map(mdcell, row)) + " |" for row in rows]
    return "\n".join(lines)


def report(root, runs, best, notes):
    methods = method_list(runs)
    lines = [
        "# Decomposed total-site heat-integration results",
        "",
        "Costs and savings in the tables are $/yr. Figure axes use k$/yr. "
        "Seed standard deviations use the sample definition; a single seed has "
        "standard deviation zero. Whiskers show seed min–max. Missing values are "
        "shown as — and excluded from aggregates.",
        "",
        "The grand-best columns describe the best full-coalition role case. "
        'The convergence and SAC figures specifically describe "-++". '
        "Convergence includes only strictly feasible evaluations. "
        "Per-case values prefer the recorder's exported bests.",
        "",
        "## Per-run results", "",
    ]
    rows = []
    for run in runs:
        g = run["grand"]
        rows.append([LABELS[run["method"]], run["seed"], fmt(row_total(run), 0),
                     g["case"], fmt(g["TAC"]), fmt(g["TACS"]), fmt(g["gap"], 3),
                     fmt(g["eval"], 0), fmt(policy_value(run, "-++")),
                     fmt(g["wall"] / 60, 2)])
    lines.append(table(
        ["Method", "Seed", "Evaluations", "Grand best case", "Grand best TAC",
         "Grand best TACS", "Gap to SI (%)", "Best eval",
         'SAC final-policy TACS "-++"', "Wall time (min)"], rows))
    lines += ["", "## Method aggregates", ""]
    rows = []
    for method in methods:
        selected = [r for r in runs if r["method"] == method]
        tac_mean, tac_std, _, _ = stats([r["grand"]["TAC"] for r in selected])
        gap_mean, gap_std, _, _ = stats([r["grand"]["gap"] for r in selected])
        rows.append([LABELS[method], len(selected),
                     f"{fmt(tac_mean)} ± {fmt(tac_std)}",
                     f"{fmt(gap_mean, 3)} ± {fmt(gap_std, 3)}"])
    lines.append(table(["Method", "Runs", "Grand best TAC: mean ± SD",
                        "Gap to SI (%): mean ± SD"], rows))
    lines += ["", "## Per-case savings", ""]
    rows = []
    for case in CASES:
        row = [case, case_label(case), fmt(SI[case], 0), fmt(SIII[case], 0)]
        row += [fmt(stats([case_value(r, case) for r in runs
                           if r["method"] == method])[0]) for method in methods]
        rows.append(row)
    lines.append(table(["Case", "Roles", "Paper SI", "Paper SIII"]
                       + [LABELS[m] + " mean" for m in methods], rows))
    lines += ["", "## Shapley allocations", "",
              "Only runs covering all 12 cases and exporting three finite "
              "`shapley_best` values are included.", ""]
    eligible = [r for r in runs if full_shapley(r)]
    rows = [["Paper Strategy I", "—"] + [fmt(v, 0) for v in PAPER_SHAPLEY]]
    for method in methods:
        selected = [r for r in eligible if r["method"] == method]
        cells = []
        for i in range(3):
            mean, std, low, high = stats([shapley_values(r)[i] for r in selected])
            cells.append(f"{fmt(mean)} ± {fmt(std)} [{fmt(low)}, {fmt(high)}]"
                         if selected else "—")
        rows.append([LABELS[method], len(selected)] + cells)
    lines.append(table(["Method", "Eligible runs", "P1: mean ± SD [min, max]",
                        "P2: mean ± SD [min, max]", "P3: mean ± SD [min, max]"], rows))
    lines += ["", "## Overall best exported grand-coalition design", ""]
    if best is None:
        lines.append("No full grand-coalition design was exported.")
    else:
        _, run, design = best
        lines += [
            f"Run: `{run['folder'].name}`. Case: `{design.get('case', '—')}`. "
            f"TAC: {fmt(design.get('TAC'))}; TACS: {fmt(design.get('TACS'))}.",
            "",
            f"T_return: {fmt(design.get('T_return'), 3)} °C; "
            f"T_supply: {fmt(design.get('T_supply'), 3)} °C; "
            f"loop flow: {fmt(design.get('loop_flow'), 6)} "
            "(units as exported).",
            "",
        ]
        plants = [normalize_plant(design, name, raw)
                  for name, raw in plant_items(design) if isinstance(raw, dict)]
        interface_rows = []
        for plant in plants:
            interface = interface_for(design, plant["name"])
            interface_rows.append([
                plant["name"], plant["role"],
                fmt(num(interface, "flow", "loop_flow", "mass_flow"), 6),
                fmt(num(interface, "t_in", "T_in", "Tin"), 3),
                fmt(num(interface, "t_out", "T_out", "Tout"), 3),
                fmt(num(interface, "duty", "duty_kW", "Q", "q"), 3),
                plant["structure"],
            ])
        lines.append(table(["Plant", "Role", "Interface flow", "T_in (°C)",
                            "T_out (°C)", "Duty (kW)", "Structure"], interface_rows))
        lines += ["", "### Exchangers", ""]
        exchanger_rows = []
        for plant in plants:
            for e in plant["exchangers"]:
                exchanger_rows.append([
                    plant["name"], e["id"], e["kind"], e["hot"], e["cold"],
                    fmt(e["duty"], 3), fmt(e["required"], 3),
                    fmt(e["installed"], 3), fmt(e["augmentation"], 3),
                ])
        lines.append(table(["Plant", "ID", "Kind", "Hot", "Cold", "Duty (kW)",
                            "Required area (m²)", "Installed area (m²)",
                            "Augmentation area (m²)"], exchanger_rows))
        lines += ["", "### Utilities", ""]
        utility_rows = []
        for plant in plants:
            for utility in plant["utilities"]:
                utility_rows.append([
                    plant["name"], utility["stream"], utility["utility"],
                    fmt(utility["duty"], 3), fmt(utility["cost"]),
                ])
        lines.append(table(["Plant", "Stream", "Utility", "Duty (kW)",
                            "Annual cost ($/yr)"], utility_rows))
    lines += [
        "", "## Paper cost reference", "",
        "Table C1: hot utility 330,453; cold utility 81,021; augmentation "
        "investment 79,401; intermediate-exchanger investment 149,614; "
        "bypasses 1,000; intermediate-fluid investment 82,609; AF 0.1349. "
        f"The listed investments annualise to {fmt(PAPER_CAPITAL)}. "
        "Their rounded component sum gives TAC 453,646.98, whereas the "
        "reported total is 453,648; plots use the reported total TAC.",
        "", "## Figures", "",
    ]
    figures = [
        ("convergence_grand", 'Strictly feasible "-++" convergence'),
        ("cases", "Role-case savings"),
        ("sac_policy", "Deterministic SAC policy probes"),
        ("sac_samples_grand", 'All SAC evaluations of "-++"'),
        ("best_design_costs", "Best exported design costs"),
        ("shapley", "Shapley allocations"),
    ]
    for filename, caption in figures:
        lines += [f"### {caption}", "",
                  f"![{caption}](figures/{filename}.png)", "",
                  f"[SVG](figures/{filename}.svg)", ""]
    if notes:
        lines += ["## Missing data and read warnings", ""]
        lines += [f"- {note}" for note in dict.fromkeys(notes)]
        lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Results directory does not exist: {root}")
    notes = []
    runs = discover(root, notes)
    if not runs:
        raise SystemExit("No sac_seed*, de_seed*, or random_seed* run directories found.")
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    style()
    best = overall_design(runs)
    convergence(runs, output)
    cases_plot(runs, output)
    sac_policy(runs, output)
    sac_samples(runs, output)
    design_costs(best, output, notes)
    shapley_plot(runs, output)
    report(root, runs, best, notes)
    print(f"Wrote six PNG/SVG figure pairs to {output}")
    print(f"Wrote {root / 'report.md'}")
    for note in dict.fromkeys(notes):
        print(f"Warning: {note}")


if __name__ == "__main__":
    main()
