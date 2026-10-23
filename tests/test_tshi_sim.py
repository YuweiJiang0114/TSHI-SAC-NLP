"""Simulation model (tshi_sim): rating reproduces known designs; site assembly."""
import numpy as np
import pytest

from tshi_decomp.case import load_case
from tshi_decomp.plant import Interface, paper_rules
from tshi_decomp.site import RoleCase, SiteConfig, SiteEvaluator
from tshi_sim.encode import actions_from_spec, from_decomp_export
from tshi_sim.plant import INACTIVE, PlantLayout, PlantSimulator
from tshi_sim.site import SiteSim, stages_for_case

PAPER_FIG3 = dict(t_return=100.7, duties={"P1": -395.9, "P2": 110.5, "P3": 285.4}, outlets={"P2": 140.0, "P3": 153.0})
SITE = SiteConfig(balance="sources", duty_snap=0.1, min_residual_utility=30.0)


@pytest.fixture(scope="module")
def case():
    return load_case()


def original_spec(plant):
    """Fig. 2 network: units at their original stages, original split fractions."""
    spec = {"stage": {e.id: e.stage for e in plant.exchangers.values()}, "split": {}}
    for sid, chain in plant.chains.items():
        for group in chain:
            if len(group) > 1:
                total = sum(plant.exchangers[u].duty for u in group)
                k = plant.exchangers[group[0]].stage
                spec["split"][(sid, k)] = {u: plant.exchangers[u].duty / total for u in group}
    return spec


@pytest.mark.parametrize("pid", ["P1", "P2", "P3"])
@pytest.mark.parametrize("K", [2, 3])
def test_original_network_is_reproduced_by_rating(case, pid, K):
    plants, costs = case
    sim = PlantSimulator(PlantLayout(plants[pid], costs, INACTIVE, K))
    a = actions_from_spec(sim, Interface(), original_spec(plants[pid]))
    d = sim.export(a, Interface())
    assert d["converged"] and d["strictly_feasible"]
    assert d["TAC"] == pytest.approx(plants[pid].baseline_utility_cost, abs=0.01)
    assert d["costs"]["capital"] == 0.0
    duties = {e["id"]: e["duty"] for e in d["exchangers"]}
    for e in plants[pid].exchangers.values():
        assert duties[e.id] == pytest.approx(e.duty, abs=0.01)  # fixed-point tolerance 1e-6 x load


def test_fig3_interface_designs_match_the_decomposition_solver(case):
    """The old solver's designs at the published interface, rated at their own areas,
    give the same cost (the site total is the paper's 453,648)."""
    plants, costs = case
    c3 = RoleCase.from_code("-++")
    with SiteEvaluator(site_cfg=SITE, solver_cfg=paper_rules(), workers=1) as ev:
        r = ev.evaluate([ev.model.encode(case=c3, **PAPER_FIG3)], [c3])[0]
    K = stages_for_case(plants, c3)
    assert K == 3
    total = costs.af * r["loop_capital"]
    for pid, x in r["plants"].items():
        itf = Interface(x["interface"]["flow"], x["interface"]["T_in"], x["interface"]["T_out"])
        sim = PlantSimulator(PlantLayout(plants[pid], costs, x["interface"]["role"], K))
        d = sim.export(actions_from_spec(sim, itf, from_decomp_export(x, plants[pid], K)), itf)
        assert d["strictly_feasible"], (pid, d["violation_kW"], d["slack_kW"])
        assert d["TAC"] == pytest.approx(x["TAC"], abs=1.0), pid
        total += d["TAC"]
    assert total == pytest.approx(453648.0, abs=5.0)


def test_random_designs_are_rated_and_costed(case):
    plants, costs = case
    sim = PlantSimulator(PlantLayout(plants["P3"], costs, "source", 3))
    rng = np.random.default_rng(0)
    ev = sim.evaluate(rng.uniform(-1, 1, (64, sim.L.dim)), Interface(2.84, 100.7, 153.0))
    assert np.isfinite(ev["tac"]).all() and (ev["tac"] > 0).all()
    assert ev["converged"].mean() >= 0.95
    assert (ev["violation"] >= 0).all() and (ev["slack"] >= 0).all()
    # An approach below dt_min on an active unit is always a priced violation.
    on = ev["q"] > sim.cfg.duty_tolerance
    below = (on & ((ev["hot_in"] - ev["cold_out"] < sim.cfg.dt_min - 1e-6)
                   | (ev["hot_out"] - ev["cold_in"] < sim.cfg.dt_min - 1e-6))).any(axis=1)
    assert (ev["violation"][below] > 0).all()
    assert not ev["strict"][below].any()


def test_stages_follow_the_paper_rule(case):
    plants, _ = case
    assert stages_for_case(plants, RoleCase.from_code("-++")) == 3
    assert stages_for_case(plants, RoleCase.from_code("-0+")) == 2
    assert stages_for_case(plants, RoleCase.from_code("0-+")) == 2
    assert stages_for_case(plants, RoleCase.from_code("+-0")) == 3


def test_site_evaluation_assembles_plant_costs(case):
    site = SiteSim(SITE)
    c3 = RoleCase.from_code("-++")
    dims = site.plant_dims(c3)
    assert dims == {"P1": 26, "P2": 25, "P3": 42}
    rng = np.random.default_rng(1)
    n = 5
    acts = {p: rng.uniform(-1, 1, (n, d)) for p, d in dims.items()}
    res = site.evaluate(rng.uniform(-1, 1, (n, site.model.action_dim)), c3, acts)
    assert len(res) == n
    for r in res:
        assert r["TAC"] == pytest.approx(sum(r["plant_TAC"].values()) + r["loop_annual_cost"])
        assert set(r["plant"]) == {"P1", "P2", "P3"}
    x = site.export(res[0]["action"], c3, {p: a[0] for p, a in acts.items()})
    assert set(x["plants"]) == {"P1", "P2", "P3"} and "plant_actions" in x
