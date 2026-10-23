# Results summary

Case: Lo, Hsu & Chang (2021), three plants, Strategy I, role case `-++` (P1 heat sink, P2 and
P3 heat sources), paper Sec. 6 bounds (30 kW minimum duty, 0.1 minimum split and bypass
fractions, 1 m² minimum augmentation, ΔT_min = 1 K inferred from Fig. 2).  Baseline utility
cost 630,350 USD/yr.  Paper: TAC 453,648, TACS 176,702.

All TACs below are of strictly feasible designs (rule residuals ≤ 1 kW-equivalent),
verified by an independent re-implementation (`tshi_sim.audit`) for the reported ones.

## 1. Decomposition study (`tshi_decomp`, enumeration inner solver)

SAC over the 7-dim loop interface, plants solved by structure enumeration + SLSQP,
3,000 site evaluations, 3 seeds: best TAC 453,923 / 453,359 / 453,303 (+0.06 / −0.06 /
−0.08 % vs paper).  Differential evolution with the same decomposition: 452,387.  Plant
optima certified with SCIP within 0.05 %.  Full tables: `results_summary/RESULTS_tshi_decomp.md`.
This inner solver is specific to the three plants (hand-written structure rules; 53
structures for P3) and does not scale, which motivated the two models below.

## 2. Simulation-only model (`tshi_sim`)

Areas, stages, splits and bypasses as actions (P1 26, P2 25, P3 42 dims), plants rated by
a fixed-point simulator (~7 ms per site evaluation).  Plant agents trained first on random
interfaces, then the site agent: best 516,409 (+13.8 %).  Joint training with the total
cost as every agent's reward collapsed to the trivial design (nobody joins the loop).
The simulator reproduces the paper's design exactly (453,648.6) and is reused by
`tshi_nlp` to seed the NLP.

## 3. One agent + site NLP (`tshi_nlp`) — current method

Agent: 29 actions (stage and bought area of each existing unit, bought area of each
candidate loop exchanger per stage, loop flow and split).  NLP: the paper's equations for
the whole site with the loop closure; IPOPT, seeded from the simulator.  Rules evaluated
on the solution; unused-area capital charged 10× extra in the training reward; areas may
be trimmed to what the operation uses and the operation re-solved for reporting.

### 3.1 Seeds (30,000 evaluations each, `configs/tshi_nlp_s1_upper_shaped_30k.yaml`)

| Seed | Best strict TAC | vs paper | Trimmed | Final policy | P3 loop stage |
|---:|---:|---:|---:|---:|---|
| 1 | 486,910 | +7.3% | 484,161 | 568,359 | 2 |
| 2 | 478,927 | +5.6% | 477,028 | 484,559 | 2 |
| 3 | 489,603 | +7.9% | 488,019 | 536,242 (infeasible) | 3 |
| 4 | 493,891 | +8.9% | 493,300 | 498,219 | 2 |
| 5 | 514,075 | +13.3% | 512,693 | 536,019 | 3 |
| 6 | 522,672 | +15.2% | 517,480 | 537,292 | 2 |
| 7 | 514,757 | +13.5% | 506,163 | 568,587 (infeasible) | 3 |
| 8 | 479,915 | +5.8% | 478,047 | 511,755 | 3 |
| 9 | 529,341 | +16.7% | 527,238 | 551,208 | 1 |
| 10 | 488,511 | +7.7% | 487,496 | 499,706 | 2 |
| **11** | **477,253** | **+5.2%** | 474,119 | 521,874 | 3 |
| 12 | 530,345 | +16.9% | 526,574 | 536,482 | 3 |
| 13 | 496,689 | +9.5% | 495,757 | 535,267 | 2 |
| 14 | 556,818 | +22.7% | 555,107 | 576,013 | 1 |
| 15 | 538,380 | +18.7% | 535,473 | 567,796 (infeasible) | 1, 2 |
| 16 | 531,448 | +17.1% | 528,288 | 554,648 | 1, 3 |
| 17 | 480,702 | +6.0% | 476,605 | 513,726 | 3 |
| 18 | 494,209 | +8.9% | 493,756 | 547,755 | 3 |
| 19 | 496,985 | +9.6% | 496,946 | 498,959 | 3 |

Median 494k; four seeds within 477–481k; eight seeds in poorer basins above 514k.
Each seed takes about 2.5 h with 18 worker processes (2–7 s per site evaluation).

### 3.2 Stronger exploration + hindsight relabeling (`..._30k_explore.yaml`)

20 % random site actions per batch, 30 % re-drawn stage entries, and for every evaluation a
second stored transition whose action buys exactly the areas the operation used
(re-evaluated by a warm-started NLP solve).

| Seed | Best strict TAC | vs paper | Final policy | P3 loop stage |
|---:|---:|---:|---:|---|
| 21 | 478,105 | +5.4% | 520,995 | 3 |
| 22 | 502,780 | +10.8% | 586,093 (infeasible) | 3 |
| 23 | 480,770 | +6.0% | 546,533 | 2 |
| 24 | 523,476 | +15.4% | 585,443 | 2 |
| 25 | 502,263 (stopped at 26,098 evaluations) | +10.7% | – | – |

Same distribution as 3.1: the hindsight transitions make the agent buy exactly the area
it uses (trimmed TAC = TAC for every seed) but do not change the structural basins.
Each evaluation costs about twice as much (the extra NLP solve).

### 3.3 Best design (seed 11, areas trimmed to use, operation re-solved)

`results_summary/best_design_seed11_trimmed.json` — TAC **471,241** (+3.9 % vs paper),
TACS 159,109; loop T_r 107.5 °C, T_s 143.2 °C, 9.93 kW/K (paper 100.7 / 148.6 / 8.27).

| Plant | Paper | This work | Structure |
|---|---:|---:|---|
| P1 | 176,436 | 186,909 | EX1-1 at stage 2, 623 kW, augmented to 410 m² (paper 565 m²); loop exchangers C2 at stage 1 (258 kW) and C1 at stage 2 in parallel with EX1-1 (97 kW) — the paper's placement; C1 heater eliminated |
| P2 | 24,644 | 24,343 | identical to the paper: EX2-1 stage 1 (324.5 kW), EX2-2 stage 3 (280 kW), loop exchanger stage 2 (110.5 kW) |
| P3 | 241,415 | 246,592 | identical structure: EX3-1 stage 1 (510 kW), EX3-2/EX3-3 stage 2 in parallel (150 / 250.5 kW), loop exchanger on H2 at stage 3 (244 kW; paper 285 kW) |
| Loop | 11,154 | 13,397 | |

The remaining 17.6 k (3.9 %) comes from the loop operating point (less heat exported by
P3, a hotter return) and the smaller EX1-1 augmentation.  Other seeds found different,
nearly equally cheap designs (seed 2: P1 at 175,898 — cheaper than the paper's P1 — with
the P3 loop exchanger at stage 2), so several site designs lie within ~1 % of each other.

### 3.4 What was learned about the method

* IPOPT needs a start in the right basin: from naive starts P1's main exchanger ends up
  throttled on both sides (205k instead of 176k).  Seeding the NLP with the best of 32
  split/bypass samples from the rating simulator fixes it; the paper's design is then
  reproduced to within 9 USD.
* "Definite" areas as equalities let the agent exploit ΔT_min slack with oversized loop
  exchangers; treating the bought area as an upper bound (`area_mode: upper`) plus a 10×
  charge on unused-area capital in the reward removes that failure class (compliance
  70–85 % of evaluations).
* Rule residuals priced at 2,000 USD/kW were worth paying for the agent; 10,000 USD/kW
  was needed.
* Results depend strongly on the seed; running many seeds and keeping the best is
  necessary.  Twenty-four seeds did not break a floor of ~477k; the top designs share the
  paper's discrete structure for P2 and P3 and differ in the loop operating point and
  P1's augmentation.

### 3.5 Ideas not yet tried

Decomposing a plant further — by temperature interval (pinch), by stage (a K-step MDP),
or by stream groups with a few permitted cross-group matches — to shrink each agent's
action space and make the discrete structure easier to find.
