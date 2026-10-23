# Decomposed SAC for the Lo et al. (2021) total-site retrofit

Code: [`tshi_decomp/`](../tshi_decomp), runner [`run_tshi_decomp.py`](../run_tshi_decomp.py),
configuration [`configs/tshi_decomp.yaml`](../configs/tshi_decomp.yaml), tests
[`tests/test_tshi_decomp.py`](../tests/test_tshi_decomp.py) and
[`tests/test_tshi_decomp_audit.py`](../tests/test_tshi_decomp_audit.py).
Results: [`results/tshi_decomp/`](../results/tshi_decomp).

## Idea

Lo, Hsu and Chang solve one coupled MINLP per coalition and role case with BARON.
Their plants are coupled **only through the intermediate fluid**: once each plant's
intermediate stream (heat-capacity flowrate, inlet and outlet temperature) is fixed,
every remaining equation, unit and cost term (SI A7 onwards; Eqs. 67-74) belongs to
one plant. So

    TAC(interface) = sum_p  min_{plant-p retrofit} TAC_p(interface_p)  +  AF * CIM * F_loop

and the site problem splits into

* a small **site problem**: choose the interface (a few continuous numbers), solved by SAC;
* one **plant problem** per plant: retrofit that plant for its given interface. These are
  independent and run in parallel processes.

At the published Fig. 3 interface, the three independent plant solves under the
paper's own bounds (`paper_rules`, below) reproduce the paper's structure in all three
plants and give **site TAC 453,648.6 USD/yr against the paper's 453,648**. Bypass capital
is $1,000, as in Table C1. The decomposition therefore loses nothing at the optimum; the
remaining task is the interface search, which is what SAC is used for.

## Site problem (SAC)

Loop topology: sources heat parallel branches of the loop from a common return
temperature `T_r` to their own outlet temperatures, and the branches mix into the supply
header `T_s`. Sinks cool parallel branches from `T_s` back to `T_r`. Mass and energy
balances hold by construction. With one sink and two sources this is the paper's Fig. 3/4.

**Observation = role case.** Lo et al. study every coalition with every source/sink role
assignment (Tables 3-5, 12 cases for three plants). The case is the SAC observation
(`+1` source, `-1` sink, `0` not in the coalition), so **one contextual policy** learns the
interface of every case. Its per-case values give the coalition characteristic function
and hence the Shapley allocation of Sec. 7.

**Action (7 numbers in [-1, 1]).**

| Action | Physical meaning | Range |
|---|---|---|
| `T_return` | loop return temperature | thermodynamic window of the case: every sink can cool to it, every source can heat from it |
| `duty:Pp` | plant duty as a fraction of its own original utility | sink: 0 to original hot utility; source: 0 to original cold utility |
| `T_out:Pp` | source branch outlet temperature | `T_r` + 2 K to the plant's hottest hot stream − ΔT_min |

Balancing the loop (`SiteConfig.balance`):

* `sources` (paper-rule study): each source exports exactly its request. The sinks
  share the total in proportion to their requests, each capped at its maximum import.
  Sources are scaled down only if all sinks together cannot absorb the total.
* `min` (relaxed study): the larger of total export and total import is scaled down
  to the smaller.

In both, a plant whose duty falls below the paper's 30 kW minimum exchanger duty drops
out. Every action therefore yields a balanced, physically consistent loop.

**Utility-elimination snap (paper-rule study).** Under the paper's 30 kW minimum duty,
a plant either keeps at least 30 kW of a utility or eliminates it, so residuals between
0 and 30 kW are forbidden. With `duty_snap = 0.1` and `min_residual_utility = 30`, the
top 10% of a duty action means "replace all of my utility", and the rest spans 0 to
(maximum − 30 kW). No action then requests a forbidden residual.

**Learning.** The retrofit is a static design problem, so every episode is one absolute
proposal: `gamma = 0`, `done = 1`. The twin critics learn each case's savings surface, and
the maximum-entropy actor concentrates on its maximum. The reward is the coalition's
TACS divided by the whole site's original utility cost (630,350); losses are compressed
smoothly into (−1, 0). Proposals are evaluated in batches of 24 for parallel efficiency,
with 4 gradient updates per evaluated proposal. Every 5 batches the deterministic policy
is evaluated (these probes count toward the budget). Agent:
`total_site.central_sac.CentralSAC`.

**Why not let SAC pick the roles itself?** The first design used signed duties (sign =
role) on the grand coalition. About half of that action space has no loop at all
(every plant exporting, or every plant importing), and most of the rest is penalized.
SAC learned to make every plant a source and collapsed onto the zero-saving plateau. The
role-free mode is kept as `case=None` in `tshi_decomp.site` for this ablation.

**Why the snap and the source-priority balance?** Under the paper's rules, the published
optimum exports exactly P2's full 110.5 kW cooler duty, eliminating the cooler. With the
plain fractional map and the scaled balance, every P2 export between 80.5 and 110.5 kW
leaves a forbidden 0-30 kW cooler, which is a cost cliff. P2's export also depended on
P1's action through the scaling. Two SAC seeds stalled at TACS ~160.5k with P2 exporting
~80 kW; that ablation is kept in `results/tshi_decomp/ablation_paper_rules_scaled_balance/`.
Encoding the paper's own discrete rule ("keep at least 30 kW, or nothing") in the action
map removes the cliff.

## Plant problem (independent, per plant)

`tshi_decomp.plant` designs the plant's retrofit for the given interface. The revamp
strategies of Lo et al. Sec. 3 are selected with `PlantSolverConfig.strategy`; the rules
below describe Strategy I, and the Strategy II/III additions follow the list.

* Existing process matches stay in their units and in the **original Fig. 2**
  series/parallel order. The `stage_nok3` field of `case/data.json`, transcribed from the
  published retrofit figure, is not used. Duties are free and area may be augmented
  (CF 10,000 + 380·X^0.83, at least 1 m²).
* One or two purchased intermediate-fluid exchangers sit on distinct compatible process
  streams (cost 15,000 + 380·A^0.83). Each goes in series at any gap of the stream's
  original chain or as a new parallel branch of an existing unit. With two of them, the
  loop passes them in either order or in parallel.
* Existing coolers/heaters keep their utility, sit at the stream end and may be turned
  down, switched off or augmented (heater CA 700). A stream without one may buy one.
* Physics: countercurrent units, Chen's approximation, ΔT_min = 1 K at both ends, exact
  stream energy balances, and non-isothermal mixing of parallel branches.

**Strategy II** (`strategy=2`): every existing unit houses exactly one existing match of
the same type in the same plant: process exchanger, or heater/cooler with the same utility.
The unit-to-match assignments are enumerated (P1: 2, P2: 2, P3: 6). CM = $2,000 is charged
when a unit's stream pair changes; a move between stages of the same pair is free (Eq. 76).
Each alternative assignment is screened on the loop structures that rank best with units in
their own matches.

**Strategy III** (`strategy=3`): existing matches may be dropped (duty 0, no approach
constraint), and one new process-process match may be added per plant at any pair of series
gaps of its hot and cold streams. Every existing unit houses exactly one active match
(Eq. 54); any other active match buys a unit (CF 10,000 + 380·A^0.83), and new utility
units are allowed. The unit assignment is solved exactly inside the evaluator: for every
evaluated point, the cheapest injective map from existing units to matches (augmentation,
bypass and CM included). This also covers every Strategy II reassignment of process units.
New-match variants are generated for the three best loop structures.

Scope limits relative to the paper's superstructure: at most two intermediate exchangers
per plant on distinct streams, and at most one new process match per plant. The paper's
Strategy III solution uses three intermediate exchangers in P1 (two on C1).

Two rule sets are available.

| | `paper_rules()` (main study) | relaxed (default `PlantSolverConfig`) |
|---|---|---|
| Minimum duty of an active unit | 30 kW for every unit | 30 kW for new intermediate exchangers only |
| Minimum split fraction | 0.1 | 0.05 |
| Throttling an oversized existing unit | bypass, $500 | free |
| Excess area left uncharged | none (1e-6 m²) | 0.05 m² |

The relaxed set uses the simplifications the user allowed ("you are not required to use
bypass or other settings"). Its optimum can exploit sub-30 kW utility residuals, which the
paper forbids, so only the paper-rule study is compared with the published numbers.

In both sets, a printed Fig. 2 area is replaced by the area the original duties require
when the two agree within half a printed digit (≤ 0.05 m²). The original network then
operates exactly, with no augmentation and no bypass. Duty residuals ≤ 0.001 kW are
numerical zero.

Optimizer (deterministic for a given interface):

1. Enumerate structures (P1 sink: 13, P2 source: 5, P3 source: 53).
2. For each, draw 512 Sobol samples. Half keep existing duties near the original operating
   point (0.5 to 1.05 × original duty), since retrofits rarely move them far.
3. Refine every one-exchanger structure. Refine all two-exchanger structures when there
   are at most 12, otherwise the 5 best by sample.
4. Each refinement runs SLSQP from the best sample and from the original operating point,
   in three passes: (a) with the augmentation decisions of the start point; (b) with every
   augmentation allowed, which is a relaxation since fixed costs are then constants;
   (c) keeping only augmentations of at least 1 m².
5. A priced slack on the intermediate stream (2,000 USD/kW·yr) keeps infeasible interfaces
   evaluable with a graded penalty. That price exceeds the most any kW of loop heat can
   save (1,250), so slack is never part of an optimum. Only zero-slack designs are
   reported as feasible.

Accuracy near the optimum region, measured over 40 perturbed copies of the Fig. 3
interface against a far more exhaustive local search (2,048 samples, 10 structures,
3 starts, 200 iterations): mean excess P1 +30, P2 0, P3 −22 USD/yr.

## Global verification of the plant subproblem

`tshi_decomp/global_check.py` writes every structure of a plant problem as a Pyomo MINLP with
the same equations and costs. Binaries cover augmentation, unit use (0 or ≥ 30 kW), bypass
and new utility units; Chen's mean is written as θ³ = d₁d₂(d₁+d₂)/2. Each model is solved
globally, and the plant optimum is the minimum over structures; the lower bounds give a
certificate even when a time limit stops a structure.

* **SCIP 10** (`pip install pyscipopt`; default backend) certifies the Strategy I designs of
  the main study. P1 and P2 match the fast solver to the dollar. In P3 the fast solver keeps
  one unnecessary throttling bypass (USD 67/yr). At every reported design the fast solver
  is within 0.05% of the certified site optimum (see RESULTS.md).
* **Couenne 0.5.8** (`--backend couenne`) aborts on some models (a `modSol == NULL` bug,
  worked around by retrying with heuristics or FBBT switched off). It also returns
  **invalid lower bounds** on P3: it reports 242,997 for the structure where this same model
  has a feasible design of 241,414.7. Its results are not used as certificates.
* BARON is not available: the free version is limited to 10 variables and 10 constraints,
  and the University of Waikato has no licence.

## Verification

* `tshi_decomp/audit.py` (written by Codex, independent of the optimizer's evaluator)
  re-walks every stream from the exported chains. It checks branch energy balances,
  mixing, approach temperatures, Chen areas, Strategy I equipment rules, loop
  mass/energy balance, supply-header mixing and every cost term. All 144 random designs
  over the 12 cases pass, and so does every reported design.
* `tests/test_tshi_decomp.py`: original networks reproduced exactly with no loop; the
  paper's P1 design is feasible with Fig. 3 areas; the paper interface recovers the paper
  structures, and under `paper_rules` reproduces TAC 453,648 within $5 with $1,000 of
  bypasses and passes the paper-rule audit; loop balances hold for all cases; the Shapley code reproduces the paper's
  Strategy I allocation (85,660 / 24,486 / 66,557) from the paper's Table 3; runner smoke
  tests for all three methods.

## Commands

```powershell
$py = 'C:\Users\yj206\AppData\Local\anaconda3\python.exe'
$paper = 'configs/tshi_decomp_paper_rules.yaml'
& $py run_tshi_decomp.py --config $paper --validate          # zero loop + Fig. 3 interface (453,648.6)
# Main study (paper rules): SAC / DE / random x 3 seeds on case 3, then all 12 cases.
& $py tools/run_tshi_decomp_queue.py --config $paper --study paper_rules --lanes 2 --workers 10
# Single runs
& $py run_tshi_decomp.py --config $paper --method sac --seed 1 --budget 3000 --cases=-++ --out results/tshi_decomp/paper_rules/case3/sac_seed1
& $py run_tshi_decomp.py --config $paper --method sac --seed 1 --budget 6000 --out results/tshi_decomp/paper_rules/all_cases/sac_seed1
# Relaxed rules (user-permitted simplifications): default config
& $py run_tshi_decomp.py --method sac --seed 1 --budget 3000 --cases=-++ --out results/tshi_decomp/case3/sac_seed1
# Figures, network diagram, independent audit
& $py plot_tshi_decomp.py results/tshi_decomp/paper_rules/case3
& $py plot_tshi_network.py results/tshi_decomp/paper_rules/case3/sac_seed1/best_design.json --out results/tshi_decomp/paper_rules/case3/figures/sac_seed1_network
& $py -m tshi_decomp.audit --paper-rules results/tshi_decomp/paper_rules/case3/sac_seed1/best_design.json
# Global certificate of the plant subproblems at a design (SCIP; 300 s per structure)
& $py -m tshi_decomp.global_check --time-limit 300 results/tshi_decomp/paper_rules/case3/sac_seed1/best_design.json
# Strategies II and III (paper rules)
& $py tools/run_tshi_decomp_queue.py --config configs/tshi_decomp_paper_rules_s3.yaml --study paper_rules_s3 --lanes 1 --workers 18 --runs "case3:sac:1:3000:-++,case3:de:1:3000:-++,all_cases:sac:1:6000"
& $py tools/run_tshi_decomp_queue.py --config configs/tshi_decomp_paper_rules_s2.yaml --study paper_rules_s2 --lanes 1 --workers 18 --runs "case3:sac:1:3000:-++,all_cases:sac:1:6000"
& $py tools/summarize_tshi_decomp.py                            # RESULTS.md from all run files
```

## Results

See [`results/tshi_decomp/RESULTS.md`](../results/tshi_decomp/RESULTS.md).
