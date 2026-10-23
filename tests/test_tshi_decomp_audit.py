"""Integration and corruption checks for the independent exported-design audit."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tshi_decomp.audit import audit_plant, audit_site
from tshi_decomp.case import load_case
from tshi_decomp.site import SiteEvaluator


@pytest.fixture(scope="module")
def designs():
    # Each action is solved exactly once; all audits/tampering reuse these exports.
    with SiteEvaluator(workers=1) as ev:
        zero = np.zeros(ev.model.action_dim)
        published = ev.model.encode(
            100.7, {"P1": -395.9, "P2": 110.5, "P3": 285.4},
            {"P2": 140.0, "P3": 153.0})
        evaluations = ev.evaluate([zero, published])
    return dict(zip(("zero", "published"), evaluations))


@pytest.mark.parametrize("name", ["zero", "published"])
def test_exported_design_passes(designs, name):
    report = audit_site(designs[name])
    assert report["passed"], "\n".join(report["failures"])
    assert set(report["plants"]) == {"P1", "P2", "P3"}
    assert report["max_residual"] >= 0
    assert report["residuals"]


@pytest.mark.parametrize("name", ["zero", "published"])
def test_plant_entry_point(designs, name):
    plants, costs = load_case()
    for pid, exported in designs[name]["plants"].items():
        report = audit_plant(exported, plants[pid], costs)
        assert report["passed"], "\n".join(report["failures"])


@pytest.mark.parametrize("tamper, expected", [
    ("duty", "energy.branch"),
    ("area", "area.exchanger_required"),
    ("utility_cost", "cost.utility_annual"),
    ("loop_flow", "site.source_flow"),
    ("remove_existing", "equipment.existing_count"),
    ("loop_side", "equipment.loop_side"),
])
def test_tampering_fails(designs, tamper, expected):
    design = copy.deepcopy(designs["published"])
    plant = design["plants"]["P1"]
    exchanger = next(e for e in plant["exchangers"] if e["kind"] == "existing")
    if tamper == "duty":
        exchanger["duty"] += 10.0
    elif tamper == "area":
        exchanger["area_required"] += 5.0
    elif tamper == "utility_cost":
        plant["utilities"][0]["annual_cost"] += 1000.0
    elif tamper == "loop_flow":
        design["loop_flow"] += 1.0
    elif tamper == "remove_existing":
        plant["exchangers"].remove(exchanger)
    elif tamper == "loop_side":
        loop = next(e for e in plant["exchangers"] if e["kind"] == "loop")
        loop["hot"], loop["cold"] = loop["cold"], loop["hot"]
    report = audit_site(design)
    assert not report["passed"]
    assert any(expected in failure for failure in report["failures"]), report["failures"]


@pytest.mark.parametrize("tamper, expected", [
    ("fraction", "topology.fraction_sum"),
    ("end", "temperature.stream_end"),
    ("installed", "area.installed"),
    ("strict", "site.strictly_feasible"),
    ("nan", "export.schema"),
])
def test_additional_corruptions(designs, tamper, expected):
    design = copy.deepcopy(designs["zero"])
    plant = design["plants"]["P1"]
    exchanger = plant["exchangers"][0]
    if tamper == "fraction":
        exchanger["hot_fraction"] *= 0.5
    elif tamper == "end":
        sid = next(iter(plant["stream_end_temperatures"]))
        plant["stream_end_temperatures"][sid] += 1
    elif tamper == "installed":
        exchanger["area_existing"] += 1
    elif tamper == "strict":
        design["strictly_feasible"] = not design["strictly_feasible"]
    elif tamper == "nan":
        exchanger["duty"] = float("nan")
    report = audit_site(design)
    assert not report["passed"]
    assert any(expected in failure for failure in report["failures"]), report["failures"]


def test_coalition_baseline(designs):
    design = copy.deepcopy(designs["zero"])
    plants, costs = load_case()
    omitted = design["plants"].pop("P3")
    design["plant_TAC"].pop("P3")
    design["TAC"] -= omitted["TAC"]
    design["TAC_without_penalty"] -= omitted["TAC"]
    baseline = sum(plants[pid].baseline_utility_cost for pid in design["plants"])
    design["TACS"] = baseline - design["TAC"]
    report = audit_site(design)
    assert report["passed"], "\n".join(report["failures"])
    design["TACS"] += plants["P3"].baseline_utility_cost
    assert not audit_site(design)["passed"]


def test_audit_has_no_optimizer_dependency():
    # Covers direct and lazy imports; the audit's sole package import is case.
    import tshi_decomp.audit as module
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "plant" and node.module != "tshi_decomp.plant"
        elif isinstance(node, ast.Import):
            assert all(alias.name != "tshi_decomp.plant" for alias in node.names)
        elif isinstance(node, ast.Name):
            assert node.id not in {"Network", "PlantSolver"}


def test_cli_generated_designs(designs, tmp_path):
    # Use the interpreter running pytest (the prescribed Anaconda interpreter).
    files = []
    for name, design in designs.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(design), encoding="utf-8")
        files.append(str(path))
    command = [sys.executable, "-m", "tshi_decomp.audit"]
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(command + files, cwd=repository, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("PASS ") == 2
    corrupt = copy.deepcopy(designs["published"])
    corrupt["loop_flow"] += 1
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(corrupt), encoding="utf-8")
    completed = subprocess.run(command + [str(path)], cwd=repository, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "FAIL " in completed.stdout
