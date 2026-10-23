"""Lo, Hsu & Chang (2021) three-plant case, read from ``case/data.json``.

Only original-plant inputs are used: Table 1 streams, Table 2 utilities, the
Fig. 2 existing networks (two-stage placement, areas, duties) and the Sec. 6
cost parameters.  The published retrofit (Fig. 3/4) is not read here; the
``stage_nok3`` field, which was transcribed from the retrofit figure, is ignored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASE_FILE = ROOT / "case" / "data.json"


@dataclass(frozen=True)
class Stream:
    id: str
    kind: str  # "hot" | "cold"
    t_in: float
    t_out: float
    fcp: float
    h: float

    @property
    def hot(self) -> bool:
        return self.kind == "hot"

    @property
    def load(self) -> float:
        return self.fcp * abs(self.t_in - self.t_out)


@dataclass(frozen=True)
class Utility:
    id: str
    kind: str  # "hot" | "cold"
    t_in: float
    t_out: float
    h: float
    price: float  # USD/(kW yr)


@dataclass(frozen=True)
class Exchanger:
    """Existing process-process unit; ``stage`` is its Fig. 2 (original) stage."""
    id: str
    hot: str
    cold: str
    area: float
    duty: float
    stage: int


@dataclass(frozen=True)
class UtilityUnit:
    id: str
    kind: str  # "cooler" | "heater"
    stream: str
    utility: str
    area: float
    duty: float


@dataclass(frozen=True)
class Costs:
    af: float
    beta: float
    ca_exchanger: float
    ca_cooler: float
    ca_heater: float
    ca_loop: float
    cf_unit: float
    cf_loop: float
    cim: float
    cm: float  # USD per existing unit moved to another stream match (Eq. 76/78)
    min_augmentation: float
    loop_h: float


@dataclass(frozen=True)
class Plant:
    id: str
    streams: dict  # id -> Stream
    utilities: dict  # id -> Utility
    exchangers: dict  # id -> Exchanger
    utility_units: dict  # id -> UtilityUnit
    chains: dict  # stream id -> list of parallel groups (lists of exchanger ids), in flow order

    @property
    def baseline_utility_cost(self) -> float:
        return sum(u.duty * self.utilities[u.utility].price for u in self.utility_units.values())

    def utility_duty(self, kind: str) -> float:
        """Original hot (``heater``) or cold (``cooler``) utility duty."""
        return sum(u.duty for u in self.utility_units.values() if u.kind == kind)


def _chains(streams: dict, exchangers: dict) -> dict:
    """Original series/parallel arrangement along every process stream.

    Hot streams pass the stages in increasing order, cold streams in decreasing
    order (counter-current stage-wise superstructure).  Units of one stream in
    one stage are parallel branches of that stream.
    """
    chains = {}
    for sid, s in streams.items():
        on = [e for e in exchangers.values() if sid in (e.hot, e.cold)]
        stages = sorted({e.stage for e in on}, reverse=not s.hot)
        chains[sid] = [[e.id for e in on if e.stage == k] for k in stages]
    return chains


def load_case(path: Path | str = CASE_FILE) -> tuple[dict, Costs]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    c = data["cost_parameters"]["lo2021"]
    costs = Costs(
        af=c["annualization_factor"], beta=c["area_exponent_beta"],
        ca_exchanger=c["CA_exchanger_usd_per_m1.66"], ca_cooler=c["CA_cooler_usd_per_m1.66"],
        ca_heater=c["CA_heater_usd_per_m1.66"], ca_loop=c["CA_process_intermediate_usd_per_m1.66"],
        cf_unit=c["CF_inner_plant_process_exchanger_cooler_heater_usd"],
        cf_loop=c["CF_process_intermediate_exchanger_usd"],
        cim=c["CIM_intermediate_fluid_usd_degC_per_kW"],
        cm=c["CM_unit_reassignment_usd"],
        min_augmentation=c["lower_bounds"]["augmented_area_m2"],
        loop_h=data["intermediate_fluid"]["h_kW_per_m2C"])
    plants = {}
    for pid, pdata in data["plants"].items():
        streams = {s["id"]: Stream(s["id"], s["type"], s["Tin_C"], s["Tout_C"],
                                   s["FCp_kW_per_C"], s["h_kW_per_m2C"]) for s in pdata["streams"]}
        utilities = {}
        for uid, u in data["utilities"][pid].items():
            if not isinstance(u, dict):
                continue
            kind = "cold" if u["Tout_C"] > u["Tin_C"] else "hot"
            utilities[uid] = Utility(uid, kind, u["Tin_C"], u["Tout_C"], u["h_kW_per_m2C"],
                                     u["cost_usd_per_kW_yr"])
        exchangers, units = {}, {}
        for u in pdata["existing_hen"]["units"]:
            if u["kind"] == "exchanger":
                exchangers[u["id"]] = Exchanger(u["id"], u["hot"], u["cold"], u["area_m2"],
                                                u["duty_kW"], int(u["stage"]))
            else:
                cooler = u["kind"] == "cooler"
                units[u["id"]] = UtilityUnit(u["id"], u["kind"], u["hot"] if cooler else u["cold"],
                                             u["cold"] if cooler else u["hot"], u["area_m2"], u["duty_kW"])
        plants[pid] = Plant(pid, streams, utilities, exchangers, units, _chains(streams, exchangers))
    return plants, costs
