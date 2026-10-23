# SAC coordination of independently optimized plant HENs

## Source and intended problem

Lo, Fang-Yi; Hsu, Ching-Hsiang; Chang, Chuei-Tin (2021), **Practicable total-site heat integration plan for retrofitting multiple heat exchanger networks**, Chemical Engineering Research and Design 174, 137–157. DOI: [10.1016/j.cherd.2021.07.032](https://doi.org/10.1016/j.cherd.2021.07.032).

The local sources are [main paper](../case/sources/lo2021_main.txt) and [supplement](../case/sources/lo2021_supplement.txt). Line numbers below refer to those UTF-8 text files. The supplement retains an earlier manuscript title. The original GAMS implementation and unrounded author inputs are unavailable.

The intended problem uses the original three-plant stream data, original equipment inventory, and **Strategy III** retrofit freedom. A total-site SAC policy coordinates the intermediate-fluid interfaces. Three independent GEKKO subproblems optimize the local plant networks for those interfaces. This is **hybrid SAC plus local MINLP optimization**: SAC does not independently discover every exchanger assignment. The complexity reduction comes from partitioning the physical optimization and moving local equipment decisions into smaller subproblems.

The original unretrofitted network supplies the physical baseline and installed equipment. Neither the published retrofit topology nor its temperatures, flows, areas, or costs initializes, steers, or shapes the policy. Published retrofit results are reserved for validation and post hoc comparison. Historical reference-guided pilots are excluded from independent search claims.

## Why the plant problems separate

SI Part A introduces constraints for each plant (line 6). Equations A1–A6, SI lines 221–237, couple plants only through intermediate-stream splitter/mixer mass and enthalpy balances. A7–A10, lines 238–248, impose each plant's process and intermediate-stream heat balances. The remaining stage, branch, area, equipment, and utility constraints are local once intermediate-stream inlet temperature, outlet temperature, and heat-capacity flowrate are fixed.

Let the fixed interface for plant p be b_p = (F_p, T_in,p, T_out,p). Define C_p(b_p) as the cost of a feasible local design returned by the plant optimizer, including annual utilities and annualized local retrofit investment. The coordinator evaluates

    TAC = sum_p C_p(b_p) + AF * CIM * sum_{p in sinks} F_p
    TACS = 630350 - TAC

The inner problems are nonconvex MINLPs. Their returned cost is an incumbent feasible cost, not automatically the global value function minimum. Timeouts and failed solves must remain distinguishable from proved physical infeasibility. Report solver budgets, feasibility audits, local incumbents, and the cost of all inner solves when comparing search methods.

## Eight coordinator controls and all source/sink roles

For three plants, let five nonnegative cycle weights be c12, c13, c23, c123, c132. The first three describe pairwise return cycles. The last two describe directed cycles 1→2→3→1 and 1→3→2→1. Construct edge heat-capacity flowrates as

    f12 = c12 + c123       f21 = c12 + c132
    f13 = c13 + c132       f31 = c13 + c123
    f23 = c23 + c123       f32 = c23 + c132

The other three controls are plant outlet temperatures theta_1, theta_2, theta_3. For every active node,

    F_p = sum_{j != p} f_pj = sum_{j != p} f_jp
    T_out,p = theta_p
    T_in,p = sum_{j != p} f_jp * theta_j / F_p
    Q_p = F_p * (T_out,p - T_in,p)

Positive Q_p means the plant heats the intermediate fluid and is a **source**. Negative Q_p means the plant receives intermediate-fluid heat and is a **sink**. A node with F_p = 0 has no intermediate interface and Q_p = 0; no division by zero is performed. Zero-duty circulation needs explicit treatment because the paper prohibits active exchangers with duty below 30 kW.

All node mass balances and the site balance sum_p Q_p = 0 hold by construction. Every nonnegative balanced directed flow graph on three nodes decomposes into these five simple cycles, so the representation includes the paper's parallel-branch arrangement and source-to-source routing. The cycle representation is redundant; different cycle weights can produce the same edge flows. Report executed edge flows and physical interfaces alongside policy actions.

Roles emerge from the interface rather than being fixed to the published best case. The formulation can cover the six nontrivial three-plant source/sink combinations in Table 5, main lines 1229–1252, and two-plant coalitions when the third node is inactive. A sufficiently broad temperature/flow domain and an action map that admits zero flow are needed to make that scope realizable. Numerical caps and topology thresholds must be recorded as implementation choices, with sensitivity checks where they affect results.

Omitting c23 would restrict the graph even when plant 1 is a sink and plants 2 and 3 are sources. For example, f12=f21=f13=f31=1, f23=f32=10, and outlet temperatures (100,159,160) give F=(2,11,11) and Q=(-119,49,70). Both plants 2 and 3 heat the fluid. A representation using only cycles through plant 1 cannot reproduce these flows. Source-only circulation cannot operate as an isolated heat-producing subsystem, but it can coexist with sink-connected circulation.

The source does not impose a 0.1 minimum on interplant route fractions in A1–A6. Its 0.1 limits concern exchanger branch fractions and process bypasses. Adding a route-fraction cutoff would narrow the problem and must be disclosed separately.

## Strategy III local design freedom and restrictions

The rules in main lines 159–166 permit new process–process and process–intermediate matches and removal of original **matches**. They do not allow unrestricted disposal of original process equipment.

| Decision | Required treatment | Evidence |
|---|---|---|
| Original process HX | Assign every original process exchanger exactly once to a process–process match/stage within its original plant | Main lines 651–657, Eq. 54 |
| Original matches | May be removed when their equipment is reassigned to other process–process matches | Main lines 159–166 |
| Match equipment | Each active process match/stage has one existing unit or one newly purchased unit | Main lines 622–630, Eq. 51 |
| Installed area | Assigned existing area cannot shrink; augmentation can be added in series | Main lines 661–721, Eqs. 55–60 |
| Process–loop HX | Purchased units; existing process-HX assignment set Z_p contains process hot/cold pairs only | Main lines 572–574 and 857–870 |
| Utility matches | New local cooler/heater matches and purchased units are allowed | Main lines 575–597, 631–650, 1069–1085; SI lines 520–529 |
| Existing utility units | May be omitted or reassigned within the plant subject to equipment-type compatibility; no duplicate use | Main lines 159–166, 609–621 |
| Utility selection | At most one active utility per process stream | SI lines 327–330, A42–A43 |
| Stage relocation | Process-stage position is a decision; a stage-only relocation does not incur the stream-match reassignment fee | Main lines 1027–1061, Eq. 78 |
| Interplant process matches | Not allowed: plants exchange heat through intermediate fluid | Main lines 65–71 |

The paper restricts reuse to equipment of the same type but does not explicitly define a type as an individual utility label. Keeping existing LP equipment eligible only for LP service and existing hot-oil equipment eligible only for hot-oil service is a defensible implementation interpretation that must be stated. New utilities should use the plant's Table 2 utility options, subject to terminal approach feasibility and the one-utility-per-process-stream rule.

The stage superstructure must retain every thermodynamically eligible process pair at every configured stage, plus process–intermediate matches appropriate to the emergent plant role. Repeated stream pairs at different stages are allowed; the original plant 3 inventory itself contains two exchangers on the same process pair. Limiting stages is a structural modeling choice, not a consequence of SAC action dimension.

## Physical and economic checks

| Check | Value or equation | Evidence |
|---|---|---|
| Process and intermediate energy balances | Exact process targets and exact specified interface duty | SI A7–A16, lines 238–262 |
| Branch balances and mixing | Separate hot/cold branch temperatures and physical splitter/mixer balances | SI A17–A36, lines 263–313 |
| Heat transfer | Q = U A [dt_hot dt_cold (dt_hot+dt_cold)/2]^(1/3), harmonic film-coefficient U | SI A76–A80, lines 409–426 |
| Terminal approaches | Both ends meet the minimum approach | SI A56–A75, lines 362–408 |
| Active duty | At least 30 kW for every active process HX, loop HX, heater, and cooler | Main line 1090 |
| Active branch fraction | At least 0.1; inactive branch fraction is zero | Main lines 1091–1092; SI A86–A99 |
| Partial bypass | Zero or at least 0.1 | Main line 1093 |
| Augmentation area | Zero or at least 1 m² | Main line 1094 |
| Intermediate film coefficient | 1.5 kW/(m² K) | Main line 1095 |
| Annualization AF | 0.1349 | Main lines 1145–1146 |
| Area exponent | 0.83 | Main line 1147 |
| Process HX/cooler area coefficient | 380 | Main lines 1148–1149 |
| Heater area coefficient | 700 | Main line 1149 |
| New/augmented ordinary-unit fixed cost | $10,000 per purchased unit | Main lines 1151–1152; Eqs. 69/77/79 |
| Process–intermediate fixed cost | $15,000 per purchased unit | Main lines 1151–1152; Eq. 70 |
| Process bypass capital | $500 per installed partial bypass | Main line 1153 |
| Equipment reassignment capital | $2,000 for a changed process-stream match, not a stage-only move | Main lines 1154–1157; Eq. 78 |
| Intermediate-fluid capital | CIM=10,000 times total hot-intermediate/sink flow; charge once before annualization | Main lines 885–887 and 1158 |

Existing equipment is sunk capital. Do not add its original acquisition cost to retrofit TAC. Do not charge intermediate-fluid inventory independently at both source and sink plants. Use Table 2 plant-specific utility prices (main lines 1115–1144), including utilities newly selected by Strategy III.

Retain the documented source interpretations from [the GEKKO audit](PAPER_GEKKO_CONSTRAINT_AUDIT.md): SI A84/A85 print the wrong sign for a stated duty lower bound, A97 refers to the wrong match binary, and main Eq. 19 uses a hot-bypass symbol for a cold stream. The present 1 K minimum approach is inferred, not a reported case input. Original Fig. 2 areas are rounded; any reconstruction must use original-network duties and balances only. Empty stages and complete removal of a utility are straight-through routing, distinct from a newly installed partial bypass; the corresponding elimination of zero-duty stage equations must be explicit.

## Training and evaluation interpretation

The eight policy controls reduce the coordinator dimension without freezing local topology. The GEKKO subproblems retain discrete equipment choices and continuous thermal variables. Their optimization work is part of the algorithm's computational cost.

For absolute interface proposals, a one-step episode and gamma=0 give SAC a static constrained black-box optimization interpretation. For incremental interface proposals, observations and replay must describe the actual transition and remaining horizon, and policy actions must use the same coordinates in actor, critic, target, and replay paths. Choose and report one interpretation rather than manufacturing a trajectory by mixing both.

Use the original-network baseline, generic numerical starts, physical bounds, and previously discovered feasible solutions only. Reference designs, controller directions toward published results, optimum-centered action limits, and expert replay from the paper are excluded from the discovery run. Any warm start from the search's own history should be logged so caching and local-solver behavior remain reproducible.

Log each raw policy action, cycle weights, edge flows, plant interfaces, role assignment, local solver status, local topology, constraint residuals, objective components, and total solver time. An accepted point must pass independent physical and cost audits. Report the best independently feasible design and the deterministic final-policy result separately. Include equal-budget random or space-filling search and multiple seeds before attributing improvement to SAC.

The published Strategy III grand-coalition result is TACS=$178,702/year from a $630,350/year utility baseline, implying retrofit TAC=$451,648/year (SI lines 685–706). This is a post hoc reference. It is not an initialization, reward target, convergence proof, or proof that a local GEKKO solution matches BARON's original optimization result.
