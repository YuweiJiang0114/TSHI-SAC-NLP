"""Site NLP (tshi_nlp): the paper's design is reproduced; actions decode and evaluate."""
import numpy as np
import pytest

from tshi_decomp.site import RoleCase
from tshi_nlp.evaluate import SiteNLP
from tshi_nlp.model import LOOP, PlantSpec, UnitSpec


@pytest.fixture(scope="module")
def site():
    return SiteNLP()


def paper_specs(site):
    plants, c, inst = site.plants, site.costs, site.installed

    def U(p, a, b):
        h = {s.id: s.h for s in plants[p].streams.values()}
        h[LOOP] = c.loop_h
        return 1.0 / (1.0 / h[a] + 1.0 / h[b])

    return {
        "P1": PlantSpec(plants["P1"], "sink", 8.2687, [
            UnitSpec("EX1-1", "existing", "H1", "C1", 1, inst["P1"]["EX1-1"] + 416.0, U("P1", "H1", "C1"), "EX1-1"),
            UnitSpec("LOOP-C2@1", "loop", LOOP, "C2", 0, 284.02, U("P1", LOOP, "C2")),
            UnitSpec("LOOP-C1@2", "loop", LOOP, "C1", 1, 13.23, U("P1", LOOP, "C1"))]),
        "P2": PlantSpec(plants["P2"], "source", 2.8117, [
            UnitSpec("EX2-1", "existing", "H1", "C2", 0, inst["P2"]["EX2-1"], U("P2", "H1", "C2"), "EX2-1"),
            UnitSpec("EX2-2", "existing", "H1", "C1", 2, inst["P2"]["EX2-2"] + 9.73, U("P2", "H1", "C1"), "EX2-2"),
            UnitSpec("LOOP-H1@2", "loop", "H1", LOOP, 1, 24.62, U("P2", "H1", LOOP))]),
        "P3": PlantSpec(plants["P3"], "source", 5.4570, [
            UnitSpec("EX3-1", "existing", "H1", "C1", 0, inst["P3"]["EX3-1"], U("P3", "H1", "C1"), "EX3-1"),
            UnitSpec("EX3-2", "existing", "H1", "C1", 1, inst["P3"]["EX3-2"], U("P3", "H1", "C1"), "EX3-2"),
            UnitSpec("EX3-3", "existing", "H2", "C1", 1, inst["P3"]["EX3-3"], U("P3", "H2", "C1"), "EX3-3"),
            UnitSpec("LOOP-H2@3", "loop", "H2", LOOP, 2, 272.47, U("P3", "H2", LOOP))]),
    }


def test_paper_design_is_reproduced_by_the_site_nlp(site):
    """Definite Fig. 3 areas and loop flows -> the NLP finds the paper's operation:
    TAC within 300 USD of 453,648 (areas rounded to 0.01 m2), loop 100.7 -> 148.6 C."""
    r = site.solve(paper_specs(site), RoleCase.from_code("-++"), 8.2687)
    assert r["solver"]["ok"]
    assert r["TAC"] == pytest.approx(453648.0, abs=300.0)
    assert r["T_return"] == pytest.approx(100.7, abs=0.1)
    assert r["T_supply"] == pytest.approx(148.58, abs=0.1)
    assert r["plants"]["P1"]["TAC"] == pytest.approx(176436.0, abs=150.0)
    assert r["plants"]["P2"]["TAC"] == pytest.approx(24644.0, abs=50.0)
    assert r["plants"]["P3"]["TAC"] == pytest.approx(241415.0, abs=150.0)
    assert r["strictly_feasible"]


def test_action_layout_and_random_evaluations(site):
    c3 = RoleCase.from_code("-++")
    layout = site.layout(c3)
    kinds = [k for k, _, _ in layout]
    assert site.action_dim(c3) == 29
    assert kinds.count("stage") == 6 and kinds.count("area") == 6 and kinds.count("loop") == 15
    assert kinds.count("flow") == 1 and kinds.count("split") == 1
    rng = np.random.default_rng(0)
    for _ in range(3):
        r = site.evaluate(rng.uniform(-1, 1, 29), c3)
        assert np.isfinite(r["TAC"]) and r["TAC"] > 0
        assert r["violation_kW"] >= 0 and set(r["plants"]) == {"P1", "P2", "P3"}
        assert "seconds" in r["solver"]
