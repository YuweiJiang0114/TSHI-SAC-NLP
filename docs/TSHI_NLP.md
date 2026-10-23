# One agent, one site NLP (`tshi_nlp`)

Third model of the Lo, Hsu & Chang (2021) three-plant retrofit, following the idea of the
`RL_evolution` code (an SAC agent sets the heat-exchanger areas of a stage-wise
superstructure; an equation model solved by IPOPT gives the operation), extended to three
plants and the intermediate-fluid loop.  The model equations are the paper's (SI Part A:
non-isothermal mixing with split and bypass fractions, A76 with Chen's mean), so the
result compares directly with the published numbers.

## Division of labour

**The agent (one SAC, one-step episodes)** fixes every capital and discrete decision:

| Block | Per | Meaning |
|---|---|---|
| stage | existing unit | its stage (K bins; two units of one match never share a stage, Eq. 7) |
| area | existing unit | its definite area: installed, or installed + X with X ≥ 1 m² (Eq. 9, 12, 22) |
| loop | candidate intermediate exchanger (stream × stage) | its definite area; a dead zone of 15 % means "not built" |
| flow | site | the loop flow (log scale, 0.5–40 kW/K) |
| split | each source/sink but the last | the loop flow's share of that branch (stick-breaking, ≥ 5 %) |

Case `-++` with K = 3 (paper rule, Sec. 2): 6 + 6 + 15 + 1 + 1 = 29 entries.

**The site NLP (Pyomo + IPOPT)** solves the operation of the whole site at once:

* variables: duty of every unit, branch outlet temperatures, split fractions (≥ 0.1 when
  a unit is active), stage and end bypass fractions, utility duties and areas, the loop
  return and supply temperatures T_r and T_s;
* equations: branch balances and mixing (A17–A36), the area equation A76 with the given
  (definite) areas as equalities, the supply header as the flow-weighted mix of the source
  outlets, every sink returning the fluid at T_r (A52), every source receiving it at T_r
  (A53);
* elastic constraints with priced slacks: ΔT_min at every unit (A56–A75), stream targets
  (A46, A49), a stream without a utility unit reaching its target (Strategy I, Eq. 2–6),
  and Eq. 10–11 (an existing cooler/heater needs at least its installed area; a smooth
  on/off factor Q/(Q+1) switches it off when the unit is unused);
* objective: utility cost + slack penalties (continuation 2 → 50 per K or kW) + a tiny
  weight on bypass fractions.

Costs and the Sec. 6 rules are evaluated on the solution: augmentation
10,000 + 380·X^0.83, new units 15,000 + 380·A^0.83, utility augmentation, 500 per bypass
(stage or end) in use, CIM·F for the fluid, 0.1349 annualization.  Rule residuals
(active units or utilities below 30 kW, bypasses below 0.1, approach/target/Eq. 10
slacks, missed loop closure) are summed as kW-equivalents, priced at 2,000 USD/(kW·yr)
in the reward, and a design is reported as strictly feasible only when they total
≤ 1 kW.  The reward is the site's TAC saving over the coalition's original utility cost,
losses compressed into (−1, 0).

### Starting point of the NLP

IPOPT is a local solver and the model is nonconvex: from naive starts (equal splits, no
bypass) it settles in a basin where P1's main exchanger is throttled on both sides
(P1 TAC 205 k instead of 176 k), and random split starts do the same.  The working
procedure is

1. a fixed-fraction solve (equal splits, no bypass) that settles the loop temperatures;
2. the rating simulator of `tshi_sim` samples 32 split/bypass settings per plant at those
   loop temperatures with open-ended loop outlet targets, scored by utility + capital +
   a mild 50 USD/kW on rule residuals (the loop-duty slack is deliberately left out — it
   would reward dumping heat into the loop — and the 30 kW rule is not priced at its full
   2,000 USD/kW, which would reject exact designs over a 0.01 kW residual);
3. the best sample of every plant is loaded as the start and the free solve runs with the
   penalty continuation.

With that start the published Fig. 3 design (definite areas and loop flows) is reproduced:
TAC 453,657 vs the paper's 453,648 (areas rounded to 0.01 m²), T_r 100.7 / T_s 148.6 °C,
the paper's two bypasses, in 0.7 s.  Random designs take 2.5–4 s; with 18 worker
processes a run of 3,000 evaluations takes about 10 minutes.

## Files

* `tshi_nlp/model.py` – `build_site_model(specs, costs, K, cfg)`, `NLPConfig`, `UnitSpec`,
  `PlantSpec`.
* `tshi_nlp/evaluate.py` – `SiteNLP`: action layout, `decode`, `solve` (stages A–C),
  cost/rule evaluation, `evaluate(action, case)`; `Rules`.
* `run_tshi_nlp.py`, `configs/tshi_nlp_s1.yaml` – the SAC run (parallel evaluations,
  warmup, random share, deterministic probes, `best_design.json`, `evals.csv`).
* `tests/test_tshi_nlp.py` – paper design reproduced; layout and random evaluations.

```powershell
$py = 'C:\Users\yj206\AppData\Local\anaconda3\python.exe'
& $py run_tshi_nlp.py configs/tshi_nlp_s1.yaml --out results/tshi_nlp/s1/case3/seed1
& $py tools/summarize_tshi_sim.py results/tshi_nlp/s1/case3/seed1
```

## Parameter provenance

From the paper: everything listed in `TSHI_SIM.md` (bounds, costs, Chen's mean, stage
rule).  Ours: ΔT_min = 1 K (inferred), the elastic penalty weights and their
continuation, the 1 kW strictness tolerance, the 0.05 m² area tolerance in Eq. 10–11,
the action→area maps (dead zone, u², A_max at a 1.2 K mean approach), the loop-flow
range, the simulator sampling of the NLP start, the SAC hyperparameters.
