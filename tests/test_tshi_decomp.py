"""Decomposed TSHI model: plant optimizer, site interfaces, runner helpers."""
import itertools
import json

import numpy as np
import pytest

from tshi_decomp.audit import audit_site
from tshi_decomp.case import load_case
from tshi_decomp.plant import Interface, PlantSolver, Placement, Structure
from tshi_decomp.site import RoleCase, SiteEvaluator, SiteModel

PAPER_FIG3 = dict(t_return=100.7, duties={"P1": -395.9, "P2": 110.5, "P3": 285.4}, outlets={"P2": 140.0, "P3": 153.0})


@pytest.fixture(scope="module")
def case():
    return load_case()


@pytest.fixture(scope="module")
def evaluator():
    with SiteEvaluator(workers=1) as ev:
        yield ev


def test_original_chains_come_from_fig2_not_the_retrofit(case):
    plants, _ = case
    assert plants["P3"].chains["C1"] == [["EX3-2", "EX3-3"], ["EX3-1"]]
    assert plants["P2"].chains["H1"] == [["EX2-1"], ["EX2-2"]]
    assert plants["P1"].chains["C2"] == []


@pytest.mark.parametrize("pid", ["P1", "P2", "P3"])
def test_inactive_plant_reproduces_original_utility_cost(case, pid):
    plants, costs = case
    r = PlantSolver(plants[pid], costs).solve(Interface())
    assert r["strictly_feasible"]
    assert r["TAC"] == pytest.approx(plants[pid].baseline_utility_cost, abs=1.0)
    assert r["costs"]["capital"] == 0.0


def test_published_p1_design_is_feasible_in_the_plant_model(case):
    plants, costs = case
    solver = PlantSolver(plants["P1"], costs)
    net = solver.network(Structure((Placement("C2", "series", 0), Placement("C1", "parallel", 0))),
                         Interface(8.2609, 148.6, 100.7))
    d = net.export(np.array([623.0, 298.9 / 395.9, 0.79, 0.0]))
    assert d["strictly_feasible"]
    areas = {e["id"]: e["area_required"] for e in d["exchangers"]}
    # Fig. 3 prints 566.7 / 284.9 / 13.2 m2 from rounded temperatures.
    assert areas["EX1-1"] == pytest.approx(566.7, rel=0.02)
    assert areas["LOOP-C2"] == pytest.approx(284.9, rel=0.03)
    assert areas["LOOP-C1"] == pytest.approx(13.2, rel=0.06)


def test_paper_interface_recovers_paper_structures_and_cost(evaluator):
    c3 = RoleCase.from_code("-++")
    a = evaluator.model.encode(case=c3, **PAPER_FIG3)
    r = evaluator.evaluate([a], [c3])[0]
    assert r["strictly_feasible"]
    assert r["TAC"] < 453648.0  # paper Strategy I; no bypass charge here
    assert r["TAC"] > 453648.0 - 1000.0
    assert {p: x["structure"]["label"] for p, x in r["plants"].items()} == {
        "P1": "C2:series0 -> C1:parallel0", "P2": "H1:series1", "P3": "H2:series1"}
    assert audit_site(r)["passed"]


def test_paper_rules_reproduce_published_strategy_one_cost():
    """At the Fig. 3 interface, independent plant solves under the paper's Sec. 6 bounds
    reproduce the coupled MINLP result of Table C1 (TAC 453,648; bypasses $1,000)."""
    from tshi_decomp.audit import PAPER_RULES
    from tshi_decomp.plant import paper_rules
    c3 = RoleCase.from_code("-++")
    with SiteEvaluator(solver_cfg=paper_rules(), workers=1) as ev:
        r = ev.evaluate([ev.model.encode(case=c3, **PAPER_FIG3)], [c3])[0]
    assert r["strictly_feasible"]
    assert r["TAC"] == pytest.approx(453648.0, abs=5.0)
    assert sum(x["costs"]["bypass_capital"] for x in r["plants"].values()) == pytest.approx(1000.0)
    assert audit_site(r, rules=PAPER_RULES)["passed"]
    for x in r["plants"].values():
        assert all(u["duty"] == 0 or u["duty"] >= 30 - 1e-6 for u in x["utilities"])


def test_relaxed_design_is_checked_against_paper_rules():
    """The audit flags sub-30 kW utilities when asked for the paper's bounds."""
    from tshi_decomp.audit import PAPER_RULES
    c3 = RoleCase.from_code("-++")
    m = SiteModel()
    with SiteEvaluator(workers=1) as ev:
        a = m.encode(94.2, {"P1": -392.6, "P2": 109.2, "P3": 283.5}, {"P2": 139.7, "P3": 187.2}, c3)
        r = ev.evaluate([a], [c3])[0]
    assert audit_site(r)["passed"]
    assert not audit_site(r, rules=PAPER_RULES)["passed"]


def test_twelve_role_cases_as_in_lo_tables():
    m = SiteModel()
    assert len(m.cases) == 12
    assert len({c.code for c in m.cases}) == 12
    assert all("-" in c.code and "+" in c.code for c in m.cases)


@pytest.mark.parametrize("code", [c.code for c in SiteModel().cases])
def test_decoded_loop_balances_mass_and_energy(code):
    m = SiteModel()
    c = RoleCase.from_code(code)
    rng = np.random.default_rng(0)
    for a in rng.uniform(-1, 1, (50, m.action_dim)):
        d = m.decode(a, c)
        assert abs(d["balance_kW"]) < 1e-8
        itf = d["_interfaces"]
        src = [p for p in m.ids if itf[p].role == "source"]
        snk = [p for p in m.ids if itf[p].role == "sink"]
        assert set(p for p in m.ids if itf[p].active) <= set(d["members"])
        if src:
            assert sum(itf[p].flow for p in src) == pytest.approx(sum(itf[p].flow for p in snk))
            assert sum(itf[p].flow * itf[p].t_out for p in src) / d["loop_flow"] == pytest.approx(d["T_supply"])
            assert all(itf[p].t_in == pytest.approx(d["T_return"]) for p in src)
            assert all(itf[p].t_out == pytest.approx(d["T_return"]) for p in snk)
            assert all(itf[p].duty >= 30.0 - 1e-9 for p in src + snk)
        for p, ch in zip(m.ids, code):
            assert itf[p].role in ({"-": "sink", "+": "source", "0": "inactive"}[ch], "inactive")


PAPER_SITE = dict(balance="sources", duty_snap=0.1, min_residual_utility=30.0)


@pytest.mark.parametrize("code", [c.code for c in SiteModel().cases])
def test_source_priority_map_balances_and_never_leaves_a_forbidden_residual(code):
    from tshi_decomp.site import SiteConfig
    m = SiteModel(SiteConfig(**PAPER_SITE))
    c = RoleCase.from_code(code)
    rng = np.random.default_rng(1)
    for a in rng.uniform(-1, 1, (100, m.action_dim)):
        d = m.decode(a, c)
        assert abs(d["balance_kW"]) < 1e-7
        for k, p in enumerate(m.ids):
            itf = d["_interfaces"][p]
            if itf.active:
                assert itf.duty <= m.duty_max[p][itf.role] + 1e-9
            dropped = any(d["requested"][q]["role"] != "inactive" and not d["_interfaces"][q].active for q in m.ids)
            if itf.role == "source" and not dropped:
                # sources export exactly their request unless the sinks cannot absorb it
                total_cap = sum(m.duty_max[q]["sink"] for q in m.ids if d["requested"][q]["role"] == "sink")
                if sum(d["requested"][q]["duty"] for q in m.ids if d["requested"][q]["role"] == "source") <= total_cap:
                    assert itf.duty == pytest.approx(d["requested"][p]["duty"])
            req = d["requested"][p]
            if req["role"] == "source":
                residual = m.duty_max[p]["source"] - req["duty"]
                assert residual == pytest.approx(0, abs=1e-9) or residual >= 30.0 - 1e-9


def test_source_priority_roundtrip_hits_full_export():
    from tshi_decomp.site import SiteConfig
    m = SiteModel(SiteConfig(**PAPER_SITE))
    c3 = RoleCase.from_code("-++")
    a = m.encode(case=c3, **PAPER_FIG3)
    d = m.decode(a, c3)
    assert d["_interfaces"]["P2"].duty == pytest.approx(110.5)  # snap zone: cooler eliminated
    assert d["_interfaces"]["P3"].duty == pytest.approx(285.4)
    assert d["_interfaces"]["P1"].duty == pytest.approx(395.9)
    a[2] = 0.97  # anywhere in the top 10% of the P2 duty action
    assert m.decode(a, c3)["_interfaces"]["P2"].duty == pytest.approx(110.5)


def test_encode_decode_roundtrip():
    m = SiteModel()
    c3 = RoleCase.from_code("-++")
    d = m.decode(m.encode(case=c3, **PAPER_FIG3), c3)
    itf = d["_interfaces"]
    assert d["T_return"] == pytest.approx(100.7)
    assert itf["P1"].duty == pytest.approx(395.9)
    assert itf["P2"].t_out == pytest.approx(140.0)
    assert itf["P3"].t_out == pytest.approx(153.0)
    assert d["T_supply"] == pytest.approx(148.6, abs=0.05)


def test_random_designs_pass_independent_audit(evaluator):
    rng = np.random.default_rng(11)
    cases = SiteModel().cases
    rs = evaluator.evaluate(rng.uniform(-1, 1, (len(cases), 7)), cases)
    for r in rs:
        assert audit_site(r)["passed"], r["case"]


def test_shapley_and_characteristic_function():
    from run_tshi_decomp import characteristic, shapley
    players = ["P1", "P2", "P3"]
    v = characteristic({"-+0": 53876, "-0+": 138019, "0-+": 15669, "-++": 176702, "+-0": 0}, players)
    assert v[frozenset({"P1", "P2", "P3"})] == 176702
    assert v[frozenset({"P1"})] == 0
    phi = shapley(v, players)
    assert sum(phi.values()) == pytest.approx(176702)
    # Lo et al. Sec. 7.1 report 85,660 / 24,486 / 66,557 for Strategy I.
    assert phi["P1"] == pytest.approx(85660, abs=1)
    assert phi["P2"] == pytest.approx(24486, abs=1)
    assert phi["P3"] == pytest.approx(66557, abs=1)


@pytest.mark.parametrize("method", ["random", "de", "sac"])
def test_runner_smoke(tmp_path, method):
    from run_tshi_decomp import experiment
    import yaml
    cfg = yaml.safe_load(open("configs/tshi_decomp.yaml", encoding="utf-8"))
    cfg.update(method=method, seed=3, budget=36, workers=1, warmup=12, batch=6, policy_eval_every=1,
               cases=["-++", "-+0"])
    cfg["sac"].update(batch_size=8, hidden_dims=[32, 32])
    cfg["de"] = {"population": 6, "F": 0.6, "CR": 0.9}
    s = experiment(cfg, tmp_path / method)
    assert s["evaluations"] <= 36
    assert (tmp_path / method / "log.csv").exists()
    summary = json.loads((tmp_path / method / "summary.json").read_text())
    assert set(summary["cases"]) == {"-++", "-+0"}
    for code, x in summary["cases"].items():
        if x["recheck"]:
            assert x["recheck"]["strict"]
            assert x["recheck"]["TACS"] == pytest.approx(x["best_TACS"], rel=1e-9, abs=1e-6)


# ---- Strategies II and III --------------------------------------------------------------------

@pytest.mark.parametrize("pid,expected", [("P1", 2), ("P2", 2), ("P3", 6)])
def test_strategy_two_assignments_are_same_type_permutations(case, pid, expected):
    from tshi_decomp.plant import paper_rules
    plants, costs = case
    s = PlantSolver(plants[pid], costs, paper_rules(strategy=2))
    assert len(s.assignments()) == expected
    assert s.assignments()[0] == ()


@pytest.mark.parametrize("strategy", [2, 3])
def test_strategies_keep_original_network_without_loop(case, strategy):
    from tshi_decomp.plant import paper_rules
    plants, costs = case
    for pid, plant in plants.items():
        r = PlantSolver(plant, costs, paper_rules(strategy=strategy)).solve(Interface())
        assert r["TAC"] == pytest.approx(plant.baseline_utility_cost, abs=1.0)


@pytest.fixture(scope="module")
def paper_interface_by_strategy():
    from tshi_decomp.plant import paper_rules
    from tshi_decomp.site import SiteConfig
    site = SiteConfig(balance="sources", duty_snap=0.1, min_residual_utility=30.0)
    c3 = RoleCase.from_code("-++")
    out = {}
    for strategy in (1, 2, 3):
        with SiteEvaluator(site, paper_rules(strategy=strategy), workers=1) as ev:
            out[strategy] = ev.evaluate([ev.model.encode(case=c3, **PAPER_FIG3)], [c3])[0]
    return out


def test_strategy_two_equals_one_at_published_interface(paper_interface_by_strategy):
    # Lo et al. Tables 3 and 4 report the same case-3 saving for Strategies I and II.
    r1, r2 = paper_interface_by_strategy[1], paper_interface_by_strategy[2]
    assert r2["TAC"] == pytest.approx(r1["TAC"], abs=1.0)


def test_strategy_three_designs_are_valid_and_not_worse(paper_interface_by_strategy):
    from tshi_decomp.audit import PAPER_RULES
    r1, r3 = paper_interface_by_strategy[1], paper_interface_by_strategy[3]
    assert r3["strictly_feasible"]
    assert r3["TAC"] <= r1["TAC"] + 1.0
    assert audit_site(r3)["passed"]
    assert audit_site(r3, rules=PAPER_RULES)["passed"]
    for x in r3["plants"].values():
        units = [e["housing_unit"] for e in x["exchangers"] if e["kind"] in ("existing", "new")]
        existing = [e["id"] for e in x["exchangers"] if e["kind"] == "existing"]
        assert sorted(u for u in units if u not in (None, "new")) == sorted(existing)  # Eq. 54


def test_audit_rejects_strategy_three_design_under_strategy_one_rules(paper_interface_by_strategy):
    r3 = paper_interface_by_strategy[3]
    if not any(e["kind"] == "new" and e["duty"] > 0 for x in r3["plants"].values() for e in x["exchangers"]):
        pytest.skip("no new process match at this interface")
    assert not audit_site(r3, rules={"strategy": 1})["passed"]
