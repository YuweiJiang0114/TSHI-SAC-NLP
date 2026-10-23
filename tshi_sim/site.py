"""Site level of the simulation model: role cases, stage count, interfaces, assembly.

The site action is the 7-dim interface action of ``tshi_decomp.site.SiteModel`` (return
temperature, duty request and source outlet of every plant); it is decoded exactly as
before into one intermediate-fluid interface per plant.  Every member plant is then
*simulated* at its interface with its own plant action (``PlantSimulator``), and the
site cost is the sum of the plant costs and the intermediate-fluid cost (Eq. 74).
Plant costs separate given the interface, so a plant agent's reward is its own TAC.
"""
from __future__ import annotations

import numpy as np

from tshi_decomp.case import CASE_FILE
from tshi_decomp.site import RoleCase, SiteConfig, SiteModel, all_role_cases, reward_from_savings
from tshi_decomp.plant import Interface

from .plant import INACTIVE, PlantLayout, PlantSimulator, SimConfig

__all__ = ["RoleCase", "SiteConfig", "SiteSim", "all_role_cases", "stages_for_case"]


def stages_for_case(plants: dict, case: RoleCase) -> int:
    """Lo et al. Sec. 2: a plant needs max(#hot, #cold) stages counting its intermediate
    stream; the superstructure takes the largest number over the plants of the case."""
    K = 0
    for pid, role in zip(plants, case.roles):
        if role is None:
            continue
        p = plants[pid]
        nh = sum(s.hot for s in p.streams.values()) + (role == "sink")
        nc = sum(not s.hot for s in p.streams.values()) + (role == "source")
        K = max(K, nh, nc)
    return K


class SiteSim:
    """Decode site actions, simulate the member plants, assemble site costs."""

    def __init__(self, site_cfg: SiteConfig | None = None, sim_cfg: SimConfig | None = None, case_file=CASE_FILE):
        self.model = SiteModel(site_cfg, case_file)
        self.plants, self.costs = self.model.plants, self.model.costs
        self.sim_cfg = sim_cfg or SimConfig(dt_min=self.model.cfg.dt_min)
        self.baseline = self.model.baseline
        self.site_baseline = self.model.site_baseline
        self._sims = {}

    # -- plant simulators ---------------------------------------------------------------
    def simulator(self, pid: str, role: str, stages: int) -> PlantSimulator:
        key = (pid, role, stages)
        if key not in self._sims:
            self._sims[key] = PlantSimulator(PlantLayout(self.plants[pid], self.costs, role, stages, self.sim_cfg))
        return self._sims[key]

    def plant_roles(self, case: RoleCase) -> dict:
        return {pid: r for pid, r in zip(self.plants, case.roles) if r is not None}

    def plant_dims(self, case: RoleCase) -> dict:
        K = stages_for_case(self.plants, case)
        return {pid: self.simulator(pid, role, K).L.dim for pid, role in self.plant_roles(case).items()}

    @staticmethod
    def observation(itf: Interface) -> np.ndarray:
        """Plant-agent observation: its interface (flow, temperatures, duty), scaled."""
        return np.array([np.log1p(itf.flow) / 4.0, (itf.t_in - 100.0) / 100.0, (itf.t_out - 100.0) / 100.0,
                         itf.duty / 1000.0], np.float32)

    def site_observation(self, pid: str, decoded: dict) -> np.ndarray:
        """Plant-agent observation with the whole site's interfaces: its own interface,
        the loop headers and every plant's signed duty (+ export, - import)."""
        own = self.observation(decoded["_interfaces"][pid])
        site = [(decoded["T_return"] - 100.0) / 100.0, (decoded["T_supply"] - 100.0) / 100.0,
                np.log1p(decoded["loop_flow"]) / 4.0]
        duties = [(1.0 if decoded["roles"][p] == "source" else -1.0) * decoded["_interfaces"][p].duty / 1000.0
                  for p in self.plants]
        return np.concatenate([own, np.array(site + duties, np.float32)])

    @property
    def site_observation_dim(self) -> int:
        return 4 + 3 + len(self.plants)

    def feasible_interface(self, pid: str, itf: Interface) -> bool:
        """Necessary condition for a plant to serve its interface: a source must hold at
        least the duty above the loop inlet + dt_min in its hot streams, a sink must have
        that much cold-stream demand below the loop inlet - dt_min."""
        if not itf.active:
            return False
        p, dt = self.plants[pid], self.sim_cfg.dt_min
        if itf.role == "source":
            avail = sum(s.fcp * max(0.0, s.t_in - (itf.t_in + dt)) for s in p.streams.values() if s.hot)
            return avail >= itf.duty and itf.t_out + dt <= max(s.t_in for s in p.streams.values() if s.hot)
        avail = sum(s.fcp * max(0.0, (itf.t_in - dt) - s.t_in) for s in p.streams.values() if not s.hot)
        return avail >= itf.duty and itf.t_out - dt >= min(s.t_in for s in p.streams.values() if not s.hot)

    def plant_reward(self, pid: str, tac) -> np.ndarray:
        """A plant agent's reward: its own TAC saving over its own baseline, losses
        compressed into (-1, 0) as for the site reward."""
        s = (self.baseline[pid] - np.asarray(tac, float)) / self.baseline[pid]
        return np.where(s >= 0, s, s / (1.0 - s))

    # -- evaluation ---------------------------------------------------------------------
    def evaluate(self, site_actions, case: RoleCase, plant_actions: dict) -> list:
        """Evaluate n site actions with the plant actions ``{pid: (n, dim_pid)}``.

        A member plant whose interface is inactive (dropped out of the loop) keeps its
        original network at its baseline cost; its plant action is unused.
        """
        a = np.atleast_2d(np.asarray(site_actions, float))
        n = a.shape[0]
        K = stages_for_case(self.plants, case)
        decoded = [self.model.decode(x, case) for x in a]
        roles = self.plant_roles(case)
        per_plant = {}
        for pid, role in roles.items():
            sim = self.simulator(pid, role, K)
            pa = np.atleast_2d(np.asarray(plant_actions[pid], float))
            if pa.shape != (n, sim.L.dim):
                raise ValueError(f"{pid}: plant actions must be ({n}, {sim.L.dim}), got {pa.shape}")
            itfs = [d["_interfaces"][pid] for d in decoded]
            active = np.array([i.active and i.role == role for i in itfs])
            out = {"active": active, "TAC": np.full(n, self.baseline[pid]), "TAC_clean": np.full(n, self.baseline[pid]),
                   "slack": np.zeros(n), "violation": np.zeros(n), "strict": np.ones(n, bool),
                   "obs": np.stack([self.observation(i) for i in itfs])}
            idx = np.flatnonzero(active)
            if len(idx):
                ev = sim.evaluate(pa[idx], [itfs[i] for i in idx])
                out["TAC"][idx] = ev["tac"]
                out["TAC_clean"][idx] = ev["tac_clean"]
                out["slack"][idx] = ev["slack"]
                out["violation"][idx] = ev["violation"]
                out["strict"][idx] = ev["strict"]
            out["reward"] = self.plant_reward(pid, out["TAC"])
            out["feasible_interface"] = np.array([self.feasible_interface(pid, i) for i in itfs])
            per_plant[pid] = out
        results = []
        c = self.costs
        for i, d in enumerate(decoded):
            members = d["members"]
            baseline = sum(self.baseline[p] for p in members)
            loop_capital = c.cim * d["loop_flow"]
            plant_tac = {p: float(per_plant[p]["TAC"][i]) for p in members}
            tac = sum(plant_tac.values()) + c.af * loop_capital
            penalty = sum(float(per_plant[p]["TAC"][i] - per_plant[p]["TAC_clean"][i]) for p in members)
            strict = all(bool(per_plant[p]["strict"][i]) for p in members)
            out = {k: v for k, v in d.items() if not k.startswith("_")}
            out.update({"TAC": tac, "TACS": baseline - tac, "coalition_baseline_TAC": baseline,
                        "TAC_without_penalty": tac - penalty, "penalty": penalty,
                        "slack_kW": sum(float(per_plant[p]["slack"][i]) for p in members),
                        "violation_kW": sum(float(per_plant[p]["violation"][i]) for p in members),
                        "strictly_feasible": strict, "loop_capital": loop_capital,
                        "loop_annual_cost": c.af * loop_capital, "plant_TAC": plant_tac,
                        "plant": {p: {k: (bool(v[i]) if v.dtype == bool else float(v[i]))
                                      for k, v in per_plant[p].items() if k != "obs"} for p in members},
                        "plant_obs": {p: per_plant[p]["obs"][i] for p in members},
                        "stages": K, "reward": reward_from_savings((baseline - tac) / self.site_baseline)})
            results.append(out)
        return results

    def export(self, site_action, case: RoleCase, plant_actions: dict) -> dict:
        """Full description of one site design (plant exports included)."""
        r = self.evaluate([site_action], case, {p: np.asarray(a, float)[None, :] for p, a in plant_actions.items()})[0]
        K = r["stages"]
        d = self.model.decode(site_action, case)
        plants = {}
        for pid, role in self.plant_roles(case).items():
            itf = d["_interfaces"][pid]
            if itf.active and itf.role == role:
                plants[pid] = self.simulator(pid, role, K).export(np.asarray(plant_actions[pid], float), itf)
            else:
                plants[pid] = {"plant": pid, "role": INACTIVE, "TAC": self.baseline[pid], "strictly_feasible": True,
                               "note": "not in the loop: original network kept"}
        r["plants"] = plants
        r.pop("plant_obs", None)
        r["plant_actions"] = {p: list(map(float, a)) for p, a in plant_actions.items()}
        return r
