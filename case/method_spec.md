# Method specification — Lo, Hsu & Chang (2021) CERD 174:137-157

Primary: Lo 2021 (L). Loop/cost model: Chang et al. 2017 (C). Context: Zhou et al. 2024 (Z).
Equation numbers are the papers' own. "SI-A" = Lo 2021 Supplementary Part A, which holds the base superstructure model. The full supplement is now available in `case/sources/lo2021_supplement.docx` and `.txt`. The earlier C-equation analogues below are historical notes; the corrected implementation uses Lo 2021 A1-A99 directly. See `docs/PAPER_GEKKO_CONSTRAINT_AUDIT.md` for current equation coverage, source corrections, and explicit rounding assumptions.

## 1. Solution procedure (L §1, §2, §6, §7)
1. **Stand-alone baseline.** Each plant's existing HEN is given as an optimal single-plant design: Fig. 2, with stand-alone TACs of 316,565 / 56,294 / 287,769 $/yr. TUC is the sum of their utility costs and is a constant in Eq. 67.
2. **Role enumeration.** Each plant in a coalition is labelled either heat source or heat sink. A superstructure is built per role assignment (Fig. 1a/1b). Tables 3-5 cover all 2- and 3-plant coalitions × role cases.
3. **Superstructure per plant.** A stage-wise (Yee-Grossmann type) superstructure is used. NOK = max over plants of the stage count. An extra bypass is placed on every process stream in every stage and at the stream end (L §2).
4. **MINLP per (coalition, role case, strategy).** The model is SI-A plus the strategy constraints (L §4.1 Eqs. 1-28; §4.2 Eqs. 29-47; §4.3 Eqs. 48-66). The objective maximises TACS (Eq. 67) with cost models Eqs. 68-79. It is solved with BARON / GAMS 27.3.
5. **Benefit allocation (stage 2).** The characteristic function is the TACS of every coalition. Shapley values and the core are then computed (L §7; formulas in SI-D, not attached).

## 2. Intermediate-fluid representation
- **L §2.** In a **source** plant the loop is a *cold* stream `CM_p` that receives heat from hot process streams. In a **sink** plant it is a *hot* stream `HM_p` that releases heat to cold process streams. These streams sit inside the plant's stage-wise superstructure, competing with process matches; they are not modelled as a utility. Matches between process streams and the loop are new units (gray circles).
- **Flow.** `F_hm` is a decision variable. It enters cost only through `TIMC = Σ_p Σ_hm CIM·F_hm` (Eq. 74). Fig. 3 reports F = 8.3 kW/°C; Fig. 4 reports F = 8.7 kW/°C.
- **Temperatures** are variables, not fixed. Figs. 3-4 report the optimised supply, return and branch temperatures; for example, in Fig. 3 the supply is 148.6 °C, the return 100.7 °C, and the P2/P3 branch outlets 140/153 °C.
- **Source-side topology.** The source-side loop is split in parallel across the source plants (split fractions 0.34/0.66 in Figs. 3-4) and mixed before the sink.
- **Film coefficient.** `hcm_p = hhm_p = 1.5 kW/m²°C` (L §6).
- **Loop mass/energy balances** are in SI-A, which is unavailable. The published analogue is C Eqs. 1-7: splitter/mixer heat-capacity-flow balances `Mh_p = Σ m_{p,1,p',s'}`, mixer energy balance `Mh_p·thmin_p = Σ m·thmout + Σ m·tcmout`, and logical `m ≤ y·U`. C Eqs. 20-21 give the stage balances for the intermediate streams: `Mh_p(thm_{p,k} − thm_{p,k+1}) = Σ_j qh_{p,j,k}` and `Mc_p(tcm_{p,k} − tcm_{p,k+1}) = Σ_i qc_{p,i,k}`. The loop is adiabatic, and the overall closure is Σ(source duties) = Σ(sink duties). Fig. 3 closes this as 110.5 + 285.4 = 298.9 + 97.0 = 395.9 kW.

## 3. Constraints
**Base (C Eqs. 8-48, the published equivalent of SI-A):**
- end temperatures: Eqs. 8-13
- overall stream balances: Eqs. 14-15
- monotone stage temperatures: Eqs. 16-19
- stage balances: Eqs. 20-23
- duty–binary big-M: Eqs. 24-28
- approach-temperature big-M: Eqs. 29-36
- `dt ≥ EMAT`: Eqs. 37-41
- big-M bounds: Eqs. 42-46
- utility duties: Eqs. 47-48

**L case lower bounds (§6):**
- duty: 30 kW
- match flow fraction: 0.1
- bypass flow fraction: 0.1
- augmented area: 1 m²

**EMAT** is not stated in L. It is inferred as 1 K from the Fig. 2 approaches, and the problem table confirms it (see data.json).

**Strategy I (L §4.1):**
- unit count limits: Eq. 1 for exchangers, Eqs. 2-3 for coolers/heaters
- binaries are zero unless the match exists: Eqs. 4-6
- each existing unit houses its own match in exactly one stage: Eqs. 7-8
- new area ≥ existing area when the unit is reused: Eqs. 9-11
- augmented area `X = (A − A_EX)·e`: Eqs. 12-14
- logic for whether an augmented unit exists: Eqs. 15-17
- logic for whether a bypass exists: Eqs. 18-21
- minimum augmented area: Eqs. 22-24
- minimum bypass fraction: Eqs. 25-28

**Strategy II (L §4.2):**
- plant-wide count limits: Eqs. 29-31; Eqs. 4-6 still hold
- reassignment within the same type and plant: Eqs. 32-35
- area bounds: Eqs. 36-38
- augmentation: Eqs. 39-41
- logic: Eqs. 42-44
- minimum area: Eqs. 45-47
- Eqs. 18-21 and 25-28 are retained

**Strategy III (L §4.3):**
- count limits: Eqs. 48-50
- each match is housed in an existing unit or a new unit `u`: Eqs. 51-53
- each existing unit houses exactly one match: Eq. 54
- area bounds, augmentation, logic and minimum area: Eqs. 55-66
- Eqs. 18-21 and 25-28 are retained

## 4. TAC / TACS model (L §5)
- **Objective (Eq. 67):** `TACS = TUC − TUC' − AF·(ATCC + NTCC1 + NTCC2 + NTCC3 + NTCC4 + TIMC)`. Post-retrofit TAC = TUC − TACS.
- **TUC' (Eq. 68):** `Σ CQ_{i,n} q_{i,n} + Σ CQ_{m,j} q_{m,j}`. The utility prices CQ are per-plant values from Table 2 (data.json), in $/kW·yr.
- **ATCC (augmented units):**
  - Strategy I: Eq. 69, `Σ [CF·ESC + CA·X^β]`
  - Strategy II: Eq. 75
  - Strategy III: Eq. 77
- **NTCC1 (new process–loop units, Eq. 70):** `Σ [CF·z + CA·A^β]`.
- **NTCC2 (bypasses, Eq. 71):** `Σ BY·rz`.
- **NTCC3 (reassignment):**
  - Strategy I: 0 (Eq. 72)
  - Strategy II: Eq. 76, `Σ CM·e` for i≠ĩ ∨ j≠j̃
  - Strategy III: Eq. 78
- **NTCC4 (new process–process units):**
  - Strategies I and II: 0 (Eq. 73)
  - Strategy III: Eq. 79
- **TIMC (Eq. 74):** `Σ CIM·F_hm`.
- **Coefficients (L §6):**

  | Parameter | Value |
  |---|---|
  | Annualisation factor AF | 0.1349 (10 yr, 5.85 %) |
  | Area exponent β | 0.83 |
  | CA, exchanger / cooler / process–loop | 380 $/m^1.66 |
  | CA, heater | 700 $/m^1.66 |
  | CF, inner-plant process unit | 10,000 $ |
  | CF, process–loop unit | 15,000 $ |
  | BY (bypass) | 500 $ |
  | CM (reassignment) | 2,000 $ |
  | CIM | 10,000 $·°C/kW |

- **Loop pipe and pump costs.** L has no separate terms; TIMC is the only loop-specific cost. C provides the distance-based model:
  - pipe diameter, weight and cost per metre: Eqs. 49-53 (coefficients A1-A4 not printed)
  - velocity and Reynolds number: Eqs. 54-55
  - friction factor `f = 0.046 Re^−0.2`: Eq. 56
  - pressure drop: Eq. 57
  - pump capital `a·y + b(m·ΔP/(cp·ρ))^c`, with a = 8600, b = 7310, c = 0.2: Eq. 58
  - pump power: Eq. 59
  - pump operating cost `Pe·Hy·Q`, with Pe = 0.1 $/kWh and Hy = 8000 h: Eq. 60
  - pumping total: Eq. 61
  - exchanger cost `1100 + 150·A`; Af = 0.264; total objective: Eq. 62
- **Z** approximates C's pipe/pump cost as piecewise-linear in fcp (Z Eqs. 3-8).

## 5. Revamp strategies (L §3)
| | Strategy I | Strategy II | Strategy III |
|---|---|---|---|
| New units allowed | process–loop matches only | process–loop matches only | process–loop **and** process–process |
| Existing process matches | kept, each in **its original unit** | kept; may be housed in **another existing unit of the same type, same plant** | may be dropped; any match may use an existing or a purchased unit |
| Existing coolers/heaters | inner-plant utility only; removable if duty not needed | same | same |
| Area increase | extra unit in series (augmentation) | same | same (existing units only) |
| Extra costs | ATCC, NTCC1, NTCC2, TIMC | + NTCC3 | + NTCC3, NTCC4 |

## 6. Decision-variable mapping for the RL formulation
| Paper variable (eq.) | Meaning | Mapping |
|---|---|---|
| Source/sink role per plant (Tables 3-5 case) | coalition topology | **Loop-agent action** (discrete), or fixed parameter (case 3: P1 sink, P2/P3 source) |
| `F_hm` (Eq. 74), branch flows `m` (C Eqs. 1-5) | loop heat-capacity flow and split | **Loop-agent action** |
| Loop supply/return temperatures `thmin`, `tcmin` (C Eqs. 6-9) | loop operating temperatures | **Loop-agent action** |
| Branch outlet temperatures `thmout`, `tcmout` | after each plant | **derived by simulation** |
| Binary `z` for process–loop match at stage k (Eq. 70) | new loop exchanger exists | **HEN-agent action** (topology) |
| `A_{i,cm,k}`, `A_{hm,j,k}` (Eq. 70) | loop exchanger area | **HEN-agent action** |
| `ẽ`, `e^{y}` reuse/reassignment binaries (Eqs. 7-8, 32-35, 51-54) | which existing unit houses which match | **HEN-agent action** (Strategy I: fixed parameter) |
| `A_{i,j,k}`, `X` (Eqs. 9-14, 36-41, 55-60) | total and augmented area | **HEN-agent action** (A); X **derived** |
| `ESC` (Eqs. 15-17) | augmented unit exists | **derived** (X ≥ 1 m²) |
| `u` new process–process unit (Eq. 79) | Strategy III only | **HEN-agent action** |
| Split fractions `rr`, bypass `rrh/rrc` and `rz` (Eqs. 18-28) | flow split / bypass | **HEN-agent action** (fraction), `rz` **derived** |
| Stage temperatures `th`, `tc`, duties `q`, `qcu`, `qhu` (C Eqs. 14-23, 47-48) | thermal state | **derived by simulation** |
| `TUC'`, ATCC, NTCC1-4, TIMC, TACS (Eqs. 67-79) | economics | **derived by simulation** |
| Stream data, CQ, CA, CF, BY, CM, CIM, AF, β, h, LQ/LR/LX bounds, `A_EX`, EMAT | case data | **fixed parameter** |
