# TSHI retrofit by simulation: areas as the agent's variables (`tshi_sim`)

Second model of the Lo, Hsu & Chang (2021) three-plant retrofit.  The first model
(`tshi_decomp`, see `TSHI_DECOMP_SAC.md`) solves every plant with an enumeration of
structures plus SLSQP.  That inner solver is specific to the three plants (hand-written
structure rules) and grows combinatorially; it cannot be used for larger sites.  This
model removes the inner optimizer: **every plant is only simulated**, and all design
decisions are actions of SAC agents.

## Model

Superstructure: the paper's stage-wise superstructure (SI Part A).  The number of stages
follows Sec. 2: each plant needs max(#hot, #cold) stages counting its intermediate
stream, and the site takes the largest number (case `-++`: P1 2, P2 3, P3 2, so K = 3).
Every process stream has a bypass at every stage and at its end (A25–A36); the
intermediate stream has none (A27–A28).  Coolers and heaters sit at the stream ends.

Strategy I (Sec. 4.1): the process matches are the existing ones; each existing unit is
placed in exactly one stage (Eq. 7–8) and keeps at least its installed area (Eq. 9); an
augmentation is at least 1 m² (Eq. 22).  New units are the matches between process
streams and the intermediate stream, candidates at every stage.

Physics: the paper's area equation A76 with Chen's mean, ΔT_min = 1 K at both ends,
counter-current units, isothermal stage mixing (A31–A34).

### Plant action (all in [-1, 1])

| Block | Per | Meaning |
|---|---|---|
| stage | existing unit | its stage (K bins; a collision between two units of one match moves the later unit on) |
| aug | existing unit | augmentation X: 0 for a ≤ 0, else 1 m² … X_max |
| area | candidate new unit (stream × stage) | 0 below a dead zone of 15 % of the range, else A_max·u² |
| split | stream × stage with ≥ 2 candidates | n − 1 stick-breaking fractions over the active candidates, each ≥ 0.1 |
| bypass | process stream × (stage, end) | 0 for a ≤ 0, else 0.1 … 0.9 |

Case `-++`, K = 3: P1 26, P2 25, P3 42 action entries.

### Simulation (no optimizer)

Given the areas, the duties follow from A76 as a *rating* problem, q = U A Chen(Δt₁, Δt₂),
solved per unit by the Illinois method (1e-12 kW).  The stage-wise network is
counter-current, so the duties are a fixed point; it is solved by successive substitution
with Anderson acceleration (fallback: damped iteration).

Two deterministic rules make every design feasible and let it meet equality targets
exactly — things the paper's MINLP does by choosing continuous variables:

* a unit never violates ΔT_min: its duty is capped where an approach reaches 1 K;
* a stream never overshoots its target (A46, A49, A52, A55): the duties of its last stages
  are clipped, new units first.

A capped new unit is bought at the area its duty needs (≤ the action).  A capped existing
unit keeps its installed area and is throttled by the paper's bypass on one of its streams
(Eq. 9 then holds with equality); the bypass is charged 500 USD and obeys the 0.1 lower
bound (Eq. 25–26) — applied from the continuous solution, because the bound is a
discontinuity.  A unit whose required area is within 0.05 m² of the installed area is an
exact fit.  An existing cooler/heater whose required area is below its installed area gets
the end bypass that raises the required area to the installed one (Eq. 10–11, A35–A36).

Priced at 2,000 USD/(kW·yr), which exceeds any saving a kW can earn: unmet intermediate
duty (slack), existing units or utilities below 30 kW, streams without a utility unit
missing their target, approach deficits a bypass cannot remove, unconverged residuals.
Only designs with none of these are reported as feasible.

### Site

The site action is the 7-dim interface action of `tshi_decomp.site.SiteModel` (return
temperature, duty request and source outlet per plant), decoded as before into one
intermediate-fluid interface per plant.  Site TAC = Σ plant TAC + annualized CIM·F.
Given the interface the plant costs separate, so a plant agent's reward is its own TAC
saving (same scale as the site reward); the site agent's reward is the total saving.

## Agents and training (`run_tshi_sim.py`)

All agents are SAC with one-step episodes and learn **together** from the first
evaluation: every iteration the site agent proposes a batch of interfaces (uniform random
during the warmup, and a 10 % random share afterwards), each plant agent designs its plant
at its interface (context: flow, temperatures, duty), the site is simulated, and every
agent stores and learns from the same evaluation with the same reward: the site's total
TAC saving (losses compressed into (-1, 0)).  `plant_reward: own` would give a plant agent
its own plant's saving instead (equal in expectation given the interface).
A plant agent only learns at interfaces it can possibly serve (enough stream heat above /
below the loop temperatures).  Deterministic probes of all agents every few iterations.
An optional separate plant pretraining (`phase1.iterations > 0`) exists but is off.

Outputs: `evals.csv` (every site evaluation), `best_design.json` (best strictly feasible
design, full plant exports), `summary.json`, policies.

```powershell
$py = 'C:\Users\yj206\AppData\Local\anaconda3\python.exe'
& $py run_tshi_sim.py configs/tshi_sim_s1.yaml --out results/tshi_sim/s1/case3/seed1
& $py -m pytest tests/test_tshi_sim.py -q
```

## Parameter provenance

From the paper (Sec. 6, Tables 1–2): 30 kW minimum duty, 0.1 split and bypass lower
bounds, 1 m² minimum augmentation, h = 1.5 for the intermediate fluid, af 0.1349, β 0.83,
CA 380/700, CF 10,000/15,000, bypass 500, CM 2,000, CIM 10,000, Chen's mean (A76).

Ours: ΔT_min = 1 K (inferred from Fig. 2, not printed in the paper); the slack price;
the 0.05 m² area tolerance (half a printed digit of Fig. 2); the action→area maps (dead
zone, u², A_max = full load at a 1.2 K mean approach); the duty tolerance 1e-3 kW; the
SAC hyperparameters.

## Verification

`tests/test_tshi_sim.py`: the original Fig. 2 networks are reproduced exactly by rating
(duties, utilities, baseline TAC, no capital); the `tshi_decomp` designs at the published
interface, rated at their own areas, give the same plant costs to the dollar and the site
TAC 453,648.6 (paper 453,648); random designs converge and every approach below ΔT_min is
a priced violation; the stage rule; site assembly.
