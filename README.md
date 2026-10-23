# TSHI-SAC-NLP — reinforcement learning for the retrofit of multi-plant heat exchanger networks

Soft Actor-Critic (SAC) agents design the indirect total-site heat integration (TSHI)
retrofit of Lo, Hsu & Chang (2021, *Chem. Eng. Res. Des.* 174, 137–157): three plants
coupled by an intermediate-fluid loop, Strategy I of the paper, with the paper's own
stage-wise superstructure equations, cost parameters and Sec. 6 bounds.

Three approaches live in this repository, in the order they were developed:

| Package | Idea | Best TAC (case `-++`, paper 453,648) |
|---|---|---:|
| `tshi_decomp` | SAC chooses the 7-dim loop interface; each plant is solved by structure enumeration + SLSQP (specific to this case) | 453,303 (−0.08 %) |
| `tshi_sim` | Every design variable (areas, stages, splits, bypasses) is an action; plants are only *simulated* (rating fixed point); plant agents + site agent | 516,409 (+13.8 %) |
| `tshi_nlp` | **Current method.** One SAC agent sets the discrete and capital decisions (stages, areas, loop flows); one site NLP (Pyomo + IPOPT) solves the operation of the whole site | 477,253 (+5.2 %), 471,241 (+3.9 %) after trimming areas to use |

See [docs/RESULTS.md](docs/RESULTS.md) for the result summary and
[docs/TSHI_NLP.md](docs/TSHI_NLP.md), [docs/TSHI_SIM.md](docs/TSHI_SIM.md),
[docs/TSHI_DECOMP_SAC.md](docs/TSHI_DECOMP_SAC.md) for the methods.

## Setup

```bash
conda env create -f environment.yml      # or: pip install numpy scipy torch pyomo pyyaml matplotlib pytest
conda activate tshi
```

IPOPT is needed for `tshi_nlp` (`conda install -c conda-forge ipopt`, or any `ipopt` executable
on the PATH that Pyomo can find).  `tshi_decomp` and `tshi_sim` need no solver.  Optional:
`pyscipopt` (SCIP) for the global verification in `tshi_decomp/global_check.py`.

```bash
python -m pytest tests -q                 # ~1 min; reproduces the paper's design in every model
```

## Run

```bash
# current method: one agent + site NLP (18 worker processes, ~2.5 h for 30k evaluations)
python run_tshi_nlp.py configs/tshi_nlp_s1_upper_shaped_30k.yaml --seed 11 --out results/tshi_nlp/seed11
python tools/summarize_tshi_nlp_seeds.py results/tshi_nlp seed      # table over seeds

# simulation-only model with plant agents + site agent
python run_tshi_sim.py configs/tshi_sim_s1_twophase.yaml --out results/tshi_sim/run1

# decomposition study (enumeration inner solver)
python run_tshi_decomp.py configs/tshi_decomp_paper_rules.yaml --out results/tshi_decomp/run1
```

Configuration keys are documented in the YAML files.  Every run writes `evals.csv`
(one row per site evaluation), `best_design.json` (best strictly feasible design with all
plant details), `final_policy.json`, `summary.json` and the policy checkpoint.

## Layout

```
case/            Lo et al. (2021) data: streams, utilities, Fig. 2 networks, costs (case/data.json)
tshi_decomp/     decomposition model: plant enumeration solver, site interface model, audit, SCIP check
tshi_sim/        simulation model: superstructure layout, rating simulator, encoder, audit, site
tshi_nlp/        site NLP model (Pyomo) and the evaluator (actions -> NLP -> costs and rules)
total_site/      the SAC implementation (central_sac.py)
configs/         run configurations
tools/           summaries, diagnostics
tests/           pytest suites (paper design reproduced by every model)
docs/            method notes and the results summary
results_summary/ the best designs found (JSON) and the decomposition study's result tables
```

The paper's text files used to build the case are not included; `case/data.json` holds
every input value with its source (table / figure / section of the paper).
