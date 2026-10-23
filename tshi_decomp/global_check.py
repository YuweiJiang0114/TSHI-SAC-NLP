"""Global verification of the plant subproblem (SCIP by default; Couenne optional).

Status (2026-10-16): Couenne 0.5.8 confirms the fast solver's P1 and P2 optima at
the published interface and at SAC seed 1 (lower bound = fast value within $0.1),
but returns invalid lower bounds for P3 structures (it reports 242,997 on the
structure where a feasible design of 241,414.7 exists in this same model) and
leaves many P3 structures open within 120 s.  Do not use it as a P3 certificate.

For a fixed interface, the plant problem of ``tshi_decomp.plant`` is

    min over structures s  [ min over continuous and on/off decisions of TAC_p(s) ].

The outer minimum is an exact enumeration (``PlantSolver.structures``).  Here each
inner problem is written as a Pyomo MINLP with the same equations as
``Network.evaluate`` and solved to global optimality with Couenne (spatial branch
and bound).  The certified plant optimum is the minimum over structures; Couenne's
lower bounds give a certificate even when a time limit is hit.

Binaries per structure: augmentation of every existing exchanger and utility unit,
use of every existing unit (0 or >= ``min_unit_duty``), throttling bypass of a used
unit whose required area is below the installed area, purchase of a new utility
unit.  Chen's mean is written as theta^3 = d1*d2*(d1+d2)/2 and every A^0.83 cost
through w with A = w^(1/0.83), which keeps all expressions differentiable.

Only strictly feasible designs are certified (no intermediate-fluid slack).

    python -m tshi_decomp.global_check results/tshi_decomp/paper_rules/case3/sac_seed3/best_design.json
    python -m tshi_decomp.global_check --backend couenne ...   # Couenne 0.5.8 (see status note)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .case import load_case
from .plant import LOOP, Interface, PlantSolver, PlantSolverConfig, paper_rules

COUENNE = os.environ.get("COUENNE_EXE", r"C:\coinor\couenne.exe")
# Couenne 0.5.8 crashes on a fractional power whose base may reach 0, so every area
# cost is written as CA*((A + DELTA)^beta - DELTA^beta), exact to ~1e-5 USD.
DELTA = 1e-6
# Couenne 0.5.8 sometimes aborts in CouenneRecordBestSol after its NLP heuristic
# ("modSol == NULL").  Disabling heuristics or FBBT avoids it; the branch-and-bound
# lower bound stays valid in every setting.
RETRY_OPTIONS = ("", "local_optimization_heuristic no\n", "feasibility_bt no\n",
                 "local_optimization_heuristic no\nfeasibility_bt no\n")


def build_model(net, housing=None):
    """Pyomo MINLP of one fixed structure (``net``: a ``plant.Network``).

    ``housing`` (Strategy III only) fixes the unit of every process match: an existing
    unit id, ``"new"`` for a purchased unit, or ``None`` for a dropped match.
    """
    import pyomo.environ as pe

    cfg, c = net.cfg, net.costs
    inv = 1.0 / c.beta
    dt, md = cfg.dt_min, cfg.min_unit_duty
    free = net.free_area
    temps = [d[1] for d in net.stream_data.values()] + [d[2] for d in net.stream_data.values()]
    t_lo, t_hi = min(temps) - 1.0, max(temps) + 1.0
    span = t_hi - t_lo
    m = pe.ConcreteModel()
    E = list(range(net.n_ex))
    loads = {sid: d[3] * abs(d[1] - d[2]) for sid, d in net.stream_data.items()}

    def qmax(i):
        return min(loads[net.ex_hot[i]], loads[net.ex_cold[i]])

    m.q = pe.Var(E, bounds=lambda m, i: (0.0, qmax(i)))
    s3 = bool(getattr(net, "s3", False))
    if s3 and housing is None:
        raise ValueError("Strategy III models need the unit of every match (housing)")
    dropped = {int(i) for i in getattr(net, "match_idx", []) if s3 and housing.get(net.ex_ids[int(i)]) is None}
    for i in dropped:
        m.q[i].fix(0.0)
    for i in net.loop_idx:
        m.q[int(i)].setlb(cfg.min_loop_duty)
    if len(net.loop_idx):
        m.loop_total = pe.Constraint(expr=sum(m.q[int(i)] for i in net.loop_idx) == net.interface.duty)

    # Stream walk: group inlet temperatures are linear in the duties; branch outlets of
    # split groups are variables tied by the bilinear branch energy balance.
    m.frac = pe.Var(pe.Any, dense=False, bounds=(cfg.min_fraction, 1.0 - cfg.min_fraction))
    m.tb = pe.Var(pe.Any, dense=False, bounds=(t_lo, t_hi))
    m.branch = pe.ConstraintList()
    ends, side_in, side_out = {}, {}, {}
    for sid, chain in net.chains.items():
        hot, t_in, _, fcp, _ = net.stream_data[sid]
        sign = -1.0 if hot else 1.0
        T = t_in
        for g, group in enumerate(chain):
            idx = [net.index[e] for e in group]
            if len(idx) == 1:
                i = idx[0]
                side_in[(sid, i)] = T
                side_out[(sid, i)] = T + sign * m.q[i] / fcp
            else:
                for j, i in enumerate(idx):
                    m.frac[(sid, g, j)].setlb(cfg.min_fraction)
                    m.frac[(sid, g, j)].setub(1.0 - cfg.min_fraction)
                    m.frac[(sid, g, j)].value = 1.0 / len(idx)
                    m.tb[(sid, g, j)].value = T if isinstance(T, float) else None
                    side_in[(sid, i)] = T
                    side_out[(sid, i)] = m.tb[(sid, g, j)]
                    m.branch.add(m.q[i] == m.frac[(sid, g, j)] * fcp * sign * (m.tb[(sid, g, j)] - T))
                m.branch.add(sum(m.frac[(sid, g, j)] for j in range(len(idx))) == 1.0)
            T = T + sign * sum(m.q[i] for i in idx) / fcp
        ends[sid] = T

    # Exchangers: approaches, Chen mean, area, costs.
    m.d1 = pe.Var(E, bounds=(dt, span))
    m.d2 = pe.Var(E, bounds=(dt, span))
    m.th = pe.Var(E, bounds=(dt, span))
    amax = {i: qmax(i) / (net.U[i] * dt) + 1.0 for i in E}
    m.A = pe.Var(E, bounds=lambda m, i: (0.0, amax[i]))
    m.ex = pe.ConstraintList()
    capital = []
    for i in E:
        if i in dropped:  # no unit, no approach
            m.A[i].fix(0.0)
            continue
        h, cname = net.ex_hot[i], net.ex_cold[i]
        m.ex.add(m.d1[i] == side_in[(h, i)] - side_out[(cname, i)])
        m.ex.add(m.d2[i] == side_out[(h, i)] - side_in[(cname, i)])
        m.ex.add(m.th[i] ** 3 == m.d1[i] * m.d2[i] * (m.d1[i] + m.d2[i]) / 2.0)
        m.ex.add(m.q[i] == net.U[i] * m.A[i] * m.th[i])
    m.w = pe.Var(pe.Any, dense=False, bounds=(0.0, None))
    m.X = pe.Var(pe.Any, dense=False, bounds=(0.0, None))
    m.z = pe.Var(pe.Any, dense=False, within=pe.Binary)
    m.units = pe.ConstraintList()
    w0 = DELTA ** c.beta

    def area_cost(key, area, upper, ca):
        """ca * area^beta through w = (area + DELTA)^beta (w >= DELTA^beta > 0)."""
        w = m.w[key]
        w.setlb(w0)
        w.setub((upper + DELTA) ** c.beta)
        w.value = w0
        m.units.add(area + DELTA == w ** inv)
        return ca * (w - w0)

    def augmentation(key, area, installed, upper, ca):
        """Binary augmentation: X >= max(A - A_ex, min area) when bought."""
        z = m.z[("aug", key)]
        X = m.X[key]
        X.setub(upper)
        m.units.add(area - installed <= free + upper * z)
        m.units.add(X >= area - installed - upper * (1 - z))
        m.units.add(X >= c.min_augmentation * z)
        m.units.add(X <= upper * z)
        return c.cf_unit * z + area_cost(key, X, upper, ca)

    def use_and_bypass(key, duty, dmax, area, installed, upper):
        """0 or >= min duty; a used unit with surplus area needs a bypass."""
        terms = []
        if md > 0:
            u = m.z[("use", key)]
            m.units.add(duty <= dmax * u)
            m.units.add(duty >= md * u)
            if cfg.bypass_cost > 0:
                b = m.z[("bypass", key)]
                m.units.add(area >= installed - free - (upper + installed) * (b + 1 - u))
                terms.append(cfg.bypass_cost * b)
        return terms

    if not s3:
        for i in net.existing_idx:
            i = int(i)
            key = net.ex_ids[i]
            capital.append(augmentation(key, m.A[i], net.area_existing[i], amax[i], net.ca[i]))
            capital += use_and_bypass(key, m.q[i], qmax(i), m.A[i], net.area_existing[i], amax[i])
    else:
        units = {u: k for k, u in enumerate(net.units)}
        for j, i in enumerate(net.match_idx):
            i = int(i)
            key = net.ex_ids[i]
            unit = housing.get(key)
            if unit is None:
                continue
            if unit == "new":  # purchased unit, active match
                m.units.add(m.q[i] >= max(md, cfg.duty_tolerance))
                capital.append(c.cf_unit + area_cost(key, m.A[i], amax[i], c.ca_exchanger))
                continue
            k = units[unit]
            m.units.add(m.q[i] >= max(md, cfg.duty_tolerance))  # Eq. 54: a housed match is active
            capital.append(augmentation(key, m.A[i], float(net.unit_area[k]), amax[i], c.ca_exchanger))
            capital.append(float(net.unit_cm[j, k]))
            if cfg.bypass_cost > 0:
                b = m.z[("bypass", key)]
                m.units.add(m.A[i] >= float(net.unit_area[k]) - free - (amax[i] + float(net.unit_area[k])) * b)
                capital.append(cfg.bypass_cost * b)
    for i in net.loop_idx:
        i = int(i)
        key = net.ex_ids[i]
        capital.append(c.cf_loop + area_cost(key, m.A[i], amax[i], net.ca[i]))

    # Utilities at stream ends.
    annual = []
    for row in net.utility_rows:
        sid = row["stream"]
        hot, _, t_out, fcp, _ = net.stream_data[sid]
        u = row["utility"]
        T = ends[sid]
        R = fcp * ((T - t_out) if hot else (t_out - T))
        rmax = loads[sid]
        key = row["unit"].id if row["unit"] else f"new:{sid}"
        if not isinstance(R, (int, float)):  # a stream without exchangers has a constant duty
            m.units.add(R >= 0.0)
        d1 = (T - u.t_out) if hot else (u.t_in - t_out)
        d2 = (t_out - u.t_in) if hot else (u.t_out - T)
        dvars = []
        for n, d in (("d1", d1), ("d2", d2)):
            v = m.X[(n, key)]
            v.setlb(dt)
            v.setub(span + 600.0)
            m.units.add(v == d)
            dvars.append(v)
        th = m.X[("th", key)]
        th.setlb(dt)
        th.setub(span + 600.0)
        m.units.add(th ** 3 == dvars[0] * dvars[1] * (dvars[0] + dvars[1]) / 2.0)
        upper = rmax / (row["U"] * dt) + 1.0
        A = m.X[("A", key)]
        A.setub(upper)
        m.units.add(R == row["U"] * A * th)
        annual.append(R * u.price)
        if row["unit"] is not None:
            capital.append(augmentation(key, A, row["area_existing"], upper, row["ca"]))
            capital += use_and_bypass(key, R, rmax, A, row["area_existing"], upper)
        else:
            z = m.z[("new", key)]
            m.units.add(R <= rmax * z)
            if md > 0:
                m.units.add(R >= md * z)
            if not cfg.allow_new_utility_units:
                m.units.add(z == 0)
            capital.append(c.cf_unit * z + area_cost(key, A, upper, row["ca"]))
    capital.append(net.reassign_cost)
    m.obj = pe.Objective(expr=sum(annual) + c.af * sum(capital))
    return m


def decision_vector(net, m):
    """Map a Pyomo solution back onto the plant model's decision vector."""
    x = np.zeros(net.dim)
    for i, col in net.col_q.items():
        x[col] = float(m.q[int(i)].value)
    if net.col_phi is not None:
        a, b = (float(m.q[int(i)].value) for i in net.loop_idx)
        x[net.col_phi] = a / (a + b)
    for (sid, g), cols in net.split_cols.items():
        fr = [float(m.frac[(sid, g, j)].value) for j in range(len(net.chains[sid][g]))]
        if len(cols) == 1:
            x[cols[0]] = fr[0]
        else:
            x[cols] = fr
    return np.clip(x, net.lo, net.hi)


def _parse_bounds(log: str):
    """Couenne's final 'Lower bound:' / 'Upper bound:' lines ('-' when none is known)."""
    def value(label):
        mt = re.search(label + r":\s+(\S+)", log)
        try:
            return float(mt.group(1)) if mt else math.nan
        except ValueError:
            return math.nan
    return value("Lower bound"), value("Upper bound")


def certify_structure(plant_id, interface, structure_index, cfg_dict, time_limit, workdir):
    """Global solve of one structure; returns bounds and the re-evaluated design."""
    import pyomo.environ as pe
    from pyomo.opt import TerminationCondition as TC

    plants, costs = load_case()
    cfg = PlantSolverConfig(**cfg_dict)
    solver = PlantSolver(plants[plant_id], costs, cfg)
    itf = Interface(*interface)
    st = solver.structures(itf)[structure_index]
    net = solver.network(st, itf)
    started = time.perf_counter()
    m = build_model(net)
    wd = Path(workdir) / f"{plant_id}_{structure_index}_{os.getpid()}"
    wd.mkdir(parents=True, exist_ok=True)
    log = wd / "couenne.log"
    res, used = None, None
    for extra in RETRY_OPTIONS:
        # A very large node_limit crashes Couenne 0.5.8; the time limit bounds the search.
        (wd / "couenne.opt").write_text(
            f"allowable_fraction_gap 1e-5\nallowable_gap 1.0\ntime_limit {time_limit}\n{extra}", encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(wd)  # Couenne reads couenne.opt from its working directory
        try:
            res = pe.SolverFactory("couenne", executable=COUENNE).solve(m, logfile=str(log), load_solutions=False)
            used = extra.strip() or "default"
            break
        except Exception:  # Couenne 0.5.8 aborts on some models; retry with other settings
            continue
        finally:
            os.chdir(cwd)
    if res is None:
        return {"structure": st.label(), "structure_index": structure_index, "lower_bound": math.nan,
                "upper_bound": math.nan, "status": "solver error", "couenne_options": None,
                "wall_s": time.perf_counter() - started}
    text = log.read_text(errors="replace") if log.exists() else ""
    lb, ub = _parse_bounds(text)
    out = {"structure": st.label(), "structure_index": structure_index, "lower_bound": lb,
           "upper_bound": ub, "termination": str(res.solver.termination_condition),
           "couenne_options": used, "wall_s": time.perf_counter() - started}
    if res.solver.termination_condition == TC.infeasible:
        out.update(lower_bound=math.inf, status="infeasible")
        return out
    if len(res.solution) == 0:
        out["status"] = "no solution"
        return out
    out["status"] = "solved"
    # Cross-check: Couenne's point re-evaluated by the fast model.  At active bounds a
    # 1e-6 rounding can flip a fixed charge (augmentation, bypass), so this is reported
    # separately from Couenne's own bounds, which define the certificate.
    try:
        m.solutions.load_from(res)
        out["pyomo_objective"] = float(pe.value(m.obj))
        exported = net.export(decision_vector(net, m))
        out.update(recheck_TAC=exported["TAC"], recheck_feasible=exported["strictly_feasible"],
                   recheck_violation=exported["max_violation"])
    except Exception as exc:  # a variable without a returned value, etc.
        out["recheck_error"] = str(exc)[:200]
    return out


def certify_structure_scip(plant_id, interface, structure_index, cfg_dict, time_limit, workdir):
    """Global solve of one structure with SCIP (spatial branch and bound, via .nl)."""
    import pyscipopt

    plants, costs = load_case()
    cfg = PlantSolverConfig(**cfg_dict)
    solver = PlantSolver(plants[plant_id], costs, cfg)
    itf = Interface(*interface)
    st = solver.structures(itf)[structure_index]
    net = solver.network(st, itf)
    started = time.perf_counter()
    m = build_model(net)
    wd = Path(workdir) / f"scip_{plant_id}_{structure_index}_{os.getpid()}"
    wd.mkdir(parents=True, exist_ok=True)
    nl = wd / "model.nl"
    _, smap_id = m.write(str(nl), io_options={"symbolic_solver_labels": True})
    # SCIP names variables from model.col; line k is Pyomo's symbol "v<k>".
    symbols = m.solutions.symbol_map[smap_id].bySymbol
    columns = nl.with_suffix(".col").read_text(encoding="utf-8").splitlines()
    by_label = {name.strip(): symbols[f"v{k}"] for k, name in enumerate(columns) if f"v{k}" in symbols}
    s = pyscipopt.Model()
    s.hideOutput()
    s.readProblem(str(nl))
    s.setParam("limits/time", float(time_limit))
    s.setParam("limits/gap", 1e-6)
    s.setParam("limits/absgap", 1.0)  # USD/yr
    s.setParam("parallel/maxnthreads", 1)
    s.optimize()
    status = s.getStatus()
    out = {"structure": st.label(), "structure_index": structure_index, "solver": "SCIP " + str(s.version()),
           "termination": status, "lower_bound": s.getDualbound(), "wall_s": None}
    if status == "infeasible":
        out.update(lower_bound=math.inf, upper_bound=math.inf, status="infeasible")
    elif s.getNSols() == 0:
        out.update(upper_bound=math.nan, status="no solution")
    else:
        out.update(upper_bound=s.getObjVal(), status="solved")
        sol = s.getBestSol()
        try:
            for v in s.getVars():
                name = v.name.strip()
                if name in by_label:
                    by_label[name].value = sol[v]
            out["pyomo_objective"] = float(__import__("pyomo.environ", fromlist=["value"]).value(m.obj))
            exported = net.export(decision_vector(net, m))
            out.update(recheck_TAC=exported["TAC"], recheck_feasible=exported["strictly_feasible"],
                       recheck_violation=exported["max_violation"])
        except Exception as exc:
            out["recheck_error"] = str(exc)[:200]
    out["wall_s"] = time.perf_counter() - started
    # The key name is shared with the Couenne path: the primal bound of the solver used.
    return out


def structure_from_export(x: dict):
    """The plant ``Structure`` of an exported design (placements, units, new matches)."""
    from .plant import NewMatch, Placement, Structure
    st = x["structure"]
    label = st.get("label", "")
    # Exports written before these fields existed carry them only in the label.
    new = st.get("new_matches")
    if new is None:
        new = [dict(hot=h, cold=c, hot_gap=int(a), cold_gap=int(b))
               for h, c, a, b in re.findall(r"\+ new (\w+)-(\w+) @(\d+)/(\d+)", label)]
    assignment = st.get("assignment")
    if assignment is None:
        part = label.split(" ; ", 1)[1].split(" + new", 1)[0] if " ; " in label else ""
        assignment = [p.split(" in ") for p in part.split(", ") if " in " in p]
    return Structure(tuple(Placement(**p) for p in st["placements"]), st["loop_arrangement"],
                     tuple(tuple(a) for a in assignment), tuple(NewMatch(**n) for n in new))


def certify_design_plant(plant_id, x: dict, cfg_dict: dict, time_limit: float, workdir: str):
    """SCIP on the design's own configuration (structure and units fixed)."""
    import pyscipopt
    import pyomo.environ as pe

    plants, costs = load_case()
    cfg = PlantSolverConfig(**cfg_dict)
    solver = PlantSolver(plants[plant_id], costs, cfg)
    itf = Interface(x["interface"]["flow"], x["interface"]["T_in"], x["interface"]["T_out"])
    net = solver.network(structure_from_export(x), itf)
    housing = {e["id"]: e.get("housing_unit") for e in x["exchangers"] if e["kind"] in ("existing", "new")}
    started = time.perf_counter()
    m = build_model(net, housing if net.s3 else None)
    wd = Path(workdir) / f"fixed_{plant_id}_{os.getpid()}"
    wd.mkdir(parents=True, exist_ok=True)
    nl = wd / "model.nl"
    m.write(str(nl), io_options={"symbolic_solver_labels": True})
    s = pyscipopt.Model()
    s.hideOutput()
    s.readProblem(str(nl))
    s.setParam("limits/time", float(time_limit))
    s.setParam("limits/gap", 1e-6)
    s.setParam("limits/absgap", 1.0)
    s.setParam("parallel/maxnthreads", 1)
    s.optimize()
    return {"plant": plant_id, "structure": x["structure"]["label"], "design_TAC": x["TAC"],
            "status": s.getStatus(), "upper_bound": s.getObjVal() if s.getNSols() else math.nan,
            "lower_bound": s.getDualbound(), "wall_s": time.perf_counter() - started}


def certify_plant(plant_id, interface, cfg: PlantSolverConfig, time_limit=300, workers=None, pool=None,
                  workdir=None, backend="scip"):
    """All structures of one plant at one interface; returns the certified optimum."""
    plants, costs = load_case()
    solver = PlantSolver(plants[plant_id], costs, cfg)
    itf = Interface(*interface)
    n = len(solver.structures(itf))
    workdir = workdir or tempfile.mkdtemp(prefix="couenne_")
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    args = [(plant_id, tuple(interface), k, cfg_dict, time_limit, workdir) for k in range(n)]
    own = pool is None
    pool = pool or ProcessPoolExecutor(workers)
    try:
        worker = certify_structure_scip if backend == "scip" else certify_structure
        rows = list(pool.map(worker, *zip(*args)))
    finally:
        if own:
            pool.shutdown()
    solved = [r for r in rows if r.get("status") == "solved" and math.isfinite(r["upper_bound"])]
    best = min(solved, key=lambda r: r["upper_bound"]) if solved else None

    def closed(r):
        # Branch and bound finished within its gap tolerance ($1/yr or 1e-5 relative).
        if r.get("status") == "infeasible":
            return True
        lb, ub = r["lower_bound"], r["upper_bound"]
        return math.isfinite(lb) and math.isfinite(ub) and ub - lb <= max(1.0, 1e-5 * abs(ub)) + 1e-9

    lbs = [r["lower_bound"] for r in rows]
    return {"plant": plant_id, "interface": list(interface), "structures": n, "rows": rows,
            "global_best_TAC": best["upper_bound"] if best else None,
            "global_best_recheck_TAC": best.get("recheck_TAC") if best else None,
            "global_best_structure": best["structure"] if best else None,
            # A valid lower bound on the plant optimum needs a bound for every structure.
            "certified_lower_bound": min(lbs) if all(not math.isnan(v) for v in lbs) else None,
            "all_structures_closed": all(closed(r) for r in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("designs", nargs="+", type=Path, help="site evaluation JSON (best_design.json)")
    ap.add_argument("--relaxed", action="store_true", help="use the relaxed rules instead of paper_rules()")
    ap.add_argument("--time-limit", type=float, default=300.0, help="seconds per structure")
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--backend", choices=["scip", "couenne"], default="scip")
    ap.add_argument("--fixed", action="store_true",
                    help="certify each design's own structure and unit assignment (Strategy II/III designs)")
    ap.add_argument("--out", type=Path, help="JSON report (default: <design>.global_check.json)")
    args = ap.parse_args(argv)
    cfg = PlantSolverConfig() if args.relaxed else paper_rules()
    if args.fixed:
        with ProcessPoolExecutor(args.workers) as pool:
            for path in args.designs:
                design = json.loads(path.read_text(encoding="utf-8"))
                design = design.get("paper_interface_evaluation", design)
                jobs = {}
                for pid, x in design["plants"].items():
                    strategy = x.get("rules", {}).get("strategy", 1)
                    c = PlantSolverConfig() if args.relaxed else paper_rules(strategy=strategy)
                    jobs[pid] = pool.submit(certify_design_plant, pid, x, {k: getattr(c, k) for k in c.__dataclass_fields__},
                                            args.time_limit, tempfile.mkdtemp(prefix="scip_fixed_"))
                report = {pid: j.result() for pid, j in jobs.items()}
                for pid, r in report.items():
                    print(f"{path.parent.name} {pid}: design {r['design_TAC']:.1f}  SCIP {r['upper_bound']:.1f}  "
                          f"lower bound {r['lower_bound']:.1f}  [{r['structure']}]", flush=True)
                out = args.out or path.with_suffix(".fixed_check.json")
                out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        return
    with ProcessPoolExecutor(args.workers) as pool:
        for path in args.designs:
            design = json.loads(path.read_text(encoding="utf-8"))
            design = design.get("paper_interface_evaluation", design)  # run_tshi_decomp --validate output
            report = {"design": str(path), "rules": "relaxed" if args.relaxed else "paper",
                      "backend": args.backend, "time_limit_s": args.time_limit, "plants": {}}
            for pid, x in design["plants"].items():
                itf = x["interface"]
                if itf["role"] == "inactive":
                    continue
                cert = certify_plant(pid, (itf["flow"], itf["T_in"], itf["T_out"]), cfg, args.time_limit,
                                     pool=pool, backend=args.backend)
                cert["fast_solver_TAC"] = x["TAC"]
                cert["fast_solver_structure"] = x["structure"]["label"]
                report["plants"][pid] = cert
                gb, lb = cert["global_best_TAC"], cert["certified_lower_bound"]
                print(f"{path.parent.name} {pid}: fast {x['TAC']:.1f} [{x['structure']['label']}]  "
                      f"{args.backend} best {gb if gb is None else round(gb, 1)} [{cert['global_best_structure']}]  "
                      f"lower bound {lb if lb is None else round(lb, 1)}  closed {cert['all_structures_closed']}",
                      flush=True)
            out = args.out or path.with_suffix(".global_check.json")
            out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
