"""Independent audit of tshi_sim exports."""
import copy

import numpy as np
import pytest

from tshi_decomp.case import load_case
from tshi_decomp.plant import Interface, paper_rules
from tshi_decomp.site import RoleCase, SiteConfig, SiteEvaluator
from tshi_sim.audit import audit_plant, audit_site
from tshi_sim.encode import actions_from_spec, from_decomp_export
from tshi_sim.plant import INACTIVE, PlantLayout, PlantSimulator
from tshi_sim.site import SiteSim, stages_for_case

from test_tshi_sim import PAPER_FIG3, SITE, original_spec


@pytest.fixture(scope="module")
def case():
    return load_case()


@pytest.fixture(scope="module")
def fig3_exports(case):
    plants, costs = case
    c3 = RoleCase.from_code("-++")
    with SiteEvaluator(site_cfg=SITE, solver_cfg=paper_rules(), workers=1) as ev:
        r = ev.evaluate([ev.model.encode(case=c3, **PAPER_FIG3)], [c3])[0]
    K = stages_for_case(plants, c3)
    out = {}
    for pid, x in r["plants"].items():
        itf = Interface(x["interface"]["flow"], x["interface"]["T_in"], x["interface"]["T_out"])
        sim = PlantSimulator(PlantLayout(plants[pid], costs, x["interface"]["role"], K))
        out[pid] = sim.export(actions_from_spec(sim, itf, from_decomp_export(x, plants[pid], K)), itf)
    return out


@pytest.mark.parametrize("pid", ["P1", "P2", "P3"])
def test_original_networks_pass_the_audit(case, pid):
    plants, costs = case
    sim = PlantSimulator(PlantLayout(plants[pid], costs, INACTIVE, 2))
    d = sim.export(actions_from_spec(sim, Interface(), original_spec(plants[pid])), Interface())
    rep = audit_plant(d, plants[pid], costs)
    assert rep["passed"], rep["failures"]


@pytest.mark.parametrize("pid", ["P1", "P2", "P3"])
def test_fig3_designs_pass_the_audit(case, fig3_exports, pid):
    plants, costs = case
    rep = audit_plant(fig3_exports[pid], plants[pid], costs)
    assert rep["passed"], rep["failures"]


def test_audit_catches_tampering(case, fig3_exports):
    plants, costs = case
    x = copy.deepcopy(fig3_exports["P1"])
    x["exchangers"][0]["duty"] += 1.0
    rep = audit_plant(x, plants["P1"], costs)
    assert not rep["passed"]
    assert any(f.startswith(("energy.", "area.", "temperature.")) for f in rep["failures"])
    x = copy.deepcopy(fig3_exports["P3"])
    x["costs"]["TAC"] -= 100.0
    assert not audit_plant(x, plants["P3"], costs)["passed"]
    x = copy.deepcopy(fig3_exports["P2"])
    x["exchangers"][0]["augmentation_area"] = 0.5  # below the 1 m2 minimum (Eq. 22)
    assert any("rule.augmentation_min" in f for f in audit_plant(x, plants["P2"], costs)["failures"])


def test_random_site_exports_are_internally_consistent(case):
    """Whatever the design quality, the export is consistent with the physics it declares."""
    site = SiteSim(SITE)
    c3 = RoleCase.from_code("-++")
    dims = site.plant_dims(c3)
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(6):
        sa = rng.uniform(-1, 1, site.model.action_dim)
        acts = {p: rng.uniform(-1, 1, d) for p, d in dims.items()}
        x = site.export(sa, c3, acts)
        rep = audit_site(x)
        if x["strictly_feasible"]:
            assert rep["passed"], rep["failures"]
        # A random design is usually infeasible; the audit may then fail on rule checks
        # (minimum duties, approaches, Eq. 9/12 of a unit a bypass could not throttle, slack),
        # never on balances, mixing, utilities or cost sums.
        bad = [f for f in rep["failures"] if not any(t in f for t in (
            "rule.", "approach.", "area.at_least_installed", "area.installed_plus_augmentation",
            "area.utility_at_least_installed", "energy.loop", "utility_allowed", "no_overshoot", "strictly_feasible"))]
        assert not bad, bad
        checked += 1
    assert checked == 6
