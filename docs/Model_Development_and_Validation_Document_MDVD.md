# Model Development & Validation Document (MDVD)

**FinGuard Malaysian consumer-credit case study**  
**Document status:** retrospective development and validation record; not production approval  
**Models:** Champion WOE Logistic Scorecard (19 features); Challenger Monotonic LightGBM M4 (73 features)

> 本文件记录本项目可复核的开发、时间外验证及模拟决策结果。除明确标注的模拟假设外，文中数字来自项目 Stage 1–6 报告与产出。它不是生产模型批准、法律意见或对未来表现的保证。

## 1. Executive Summary & Model Inventory

FinGuard builds a pre-disbursement default-risk ranking model and a deterministic limit/pricing simulation on the Home Credit-style application dataset. The interpretable Champion is a WOE-transformed `statsmodels.Logit` scorecard with 19 retained predictors. Challenger M4 is a 73-feature LightGBM with domain monotonic constraints and a Stage 3 isotonic calibration artifact. Stage 5 separately evaluated intercept/Platt recalibration on `OOT_Valid`; those calibrators were then forward-applied to `OOT_Monitor`.

在 blind `OOT_Valid`，Champion `KS=0.4346`、`ROC-AUC=0.7768`；M4 raw `KS=0.4929`、`ROC-AUC=0.8173`。在独立的后续 `OOT_Monitor`，Champion `KS=0.4222`，M4 `KS=0.5006`。模型排序能力较强并不自动证明校准、因果增量或生产适用性；这三类判断须分别验证。

| Inventory item | Champion | Challenger |
|---|---|---|
| Algorithm | WOE Logistic Regression | Monotonic LightGBM M4 |
| Feature count | 19 | 73 |
| Training / development | `Train_Dev`, W0–64 | `Dev_Fit`, W0–54; `Dev_Calib`, W55–64 |
| Blind temporal evaluation | `OOT_Valid`, W65–77 | `OOT_Valid`, W65–77 |
| Forward monitoring | `OOT_Monitor`, W78–91 | `OOT_Monitor`, W78–91 |
| Primary intended use | Transparent ranking and points table | Higher-capacity ranking challenger |

## 2. Portfolio Scope, Target Definition & Strict Temporal Splits

The target is the project-defined observed loan-default label (`target`, 0/1) at application/case level. It is an observed-outcome dataset, not a randomized lending experiment. The temporal partitions are:

| Partition | Weeks | Applications | Bad rate | Permitted use |
|---|---:|---:|---:|---|
| `Train_Dev` | 0–64 | 1,292,350 | 3.3154% | Fit Champion; temporal Dev_Fit/Dev_Calib for Challenger |
| `OOT_Valid` | 65–77 | 82,602 | 2.3401% | Frozen-model evaluation; Stage 5 recalibration/cutoff fitting as specifically documented |
| `OOT_Monitor` | 78–91 | 151,707 | 2.1192% | Forward monitoring and final retrospective evaluation only |

`OOT_Valid` and `OOT_Monitor` are not interchangeable: Stage 5 used OOT_Valid outcomes to fit recalibration and decision cutoffs, then forward-applied them to OOT_Monitor. Thus Monitor remains blind to fitting, but OOT_Valid is no longer a pristine untouched validation set for Stage 5 post-processing. `Train_Dev` counts are rounded in prose only; exact counts are retained above.

## 3. Data Engineering & Pre-Disbursement Hygiene

The feature mart was built with DuckDB using a 6 GB memory cap, four threads, disk-backed `duckdb_temp/` spill, case-level aggregation and float32 model matrices. This supports out-of-core processing of the large source tables; it does not by itself establish a production SLA. Aggregations must maintain one row per `case_id` and avoid joining many-to-many source rows without explicit aggregation.

`Total_DSR_Proxy` uses active bureau monthly debt as the current-obligation proxy, divided by income where observed. This differs from `PTI`, which measures a payment/income relationship and is not the same as total debt service. Zero-inflated indicators and missing income are retained/flagged rather than silently treated as evidence of no risk; Stage 4 limit logic applies a tier-based fallback and marks income as unverified. Exact operational assumptions belong in the Stage 4 report.

The post-decision field `disbursedcredamount_1113A` was excluded from model predictors because it can encode a decision/outcome of the loan process. No target, future-week, or post-origination field should enter scoring-time features.

## 4. Champion WOE Scorecard Specification

The Stage 2 funnel began with 104 candidate features, removed three sensitive attributes before 101 entered screening, retained 89 after Gate 1, 41 after Gate 2, 39 after internal PSI Gate 3, and 23 after Gate 4 correlation screening; the VIF screen removed none. Therefore the accurate funnel is `104 → 101 → 89 → 41 → 39 → 23`; the compact `104 → 23 → 19` summarizes candidate-to-final Champion only. Stage 3 began from the selected feature list, excluded `disbursedcredamount_1113A`, then backward-eliminated to 19.

Each retained feature satisfies `coef < 0`, `p-value < 0.05`, and `VIF < 5.0`; the maximum reported VIF is `3.0331` (`static_days_overdue_tol`). PDO scaling uses `Base_Score=600` at `bad/good odds=1/20`, `PDO=50`; increasing score indicates lower estimated risk. The reported score-to-PD reconstruction error is `2.776e-16`. Exported feature/bin points are in `data/processed/scorecard_points_table.csv`.

For log-odds `ln(PD/(1−PD))`, with higher score defined as lower risk, score scaling follows `Factor = PDO / ln(2)` and `Offset = Base_Score − Factor × ln(Base_Odds)`, where `Base_Odds=bad/good=1/20`; feature points are the corresponding signed coefficient/WOE contributions under the implemented score convention. Operational implementations must reproduce the saved points table and test boundary/special/missing bins.

## 5. Challenger, Calibration Diagnostics & Four-Group Ablation

M4 uses 73 predictors and domain constraints: delinquency/rejection/query indicators generally constrained `+1`, early-payment ratio and employment years `−1`, and ambiguous/category predictors `0`. These encode risk-direction assumptions; constraints are not a causal claim. Group 4 proxies (`has_tax_registry_record`, `has_debitcard`, `tax_verified_income_ratio`, `app_cnt_last_30d`) were explicitly made available for the ablation. This is a deliberate sensitivity override: `has_tax_registry_record` had internal PSI `1.1490`, `tax_verified_income_ratio` `0.3119`, both above `0.10`; findings involving these unstable proxies require heightened validation.

| Model | Feature groups added | OOT_Valid KS | Dev_Calib KS |
|---|---|---:|---:|
| M1 | Group 1 | 0.3236 | 0.3113 |
| M2 | Groups 1 + 2 | 0.4290 | 0.4058 |
| M3 | Groups 1 + 2 + 3 | 0.4671 | 0.4248 |
| M4 | Groups 1 + 2 + 3 + 4 / affordability | 0.4929 | 0.4604 |

On `Dev_Calib` W55–64, observed bad rate was `4.64%` versus raw M4 mean PD `3.86%`; isotonic calibration aligned the sample mean to `4.64%`. OOT_Valid raw mean PD was `3.02%`, actual `2.34%`; the Stage 3 isotonic mean was `3.69%` and `A/E=0.6336`. Discrimination remained similar (`KS 0.4929` raw vs `0.4880` calibrated), while calibrated Brier was `0.0219` vs raw `0.0216`. This shows calibration did not transfer perfectly across periods.

The W55–64 prevalence increase is a temporal association and was discussed as a COVID-era/high-prevalence slice; the data alone do not identify COVID as its cause. A single isotonic mapping fitted on this ten-week calibration window is sensitive to prevalence and sampling composition. Native ranking/log-loss and calibrated absolute PD answer different questions; isotonic's in-window mean alignment is not proof of transportability. Monitor evidence in Stage 5 supports separate, time-frozen recalibration governance.

## 6. Decision Engine, Limit Drivers & KGB Swap-Set Caveats

The simulated recommended limit is `min(L_requested, L_risk_behavior, L_affordability)`. The affordability cap uses a 50% DSR ceiling after active bureau monthly debt, an annuity-based amount/term calculation, and a 3–60 month term range. Risk multiples vary by tier (A/B/C/D: 4x/3x/2x/1x); Tier E is declined. Behavioral history can adjust limits (clean history 1.00; early payment ratio ≥0.70: 1.15; minor positive DPD: 0.75; hard knockout: 0). When income is missing/low, a tier-based request-percentage fallback is used with an unverified-income flag.

| Assumption | Simulation value | Governance interpretation |
|---|---:|---|
| DSR ceiling | 50% | Project policy assumption; not a universal statutory cap |
| Maximum term | 60 months | Engine bound, subject to product/legal review |
| Tier A/B/C/D APR | 9% / 12% / 15% / 18% | Scenario pricing inputs |
| Cost of funds / operating cost | 4% / 1.5% | Assumed annual rates |
| LGD | 75% | Assumed loss severity |

The OOT_Valid 75%-approval matched swap set compares exactly 61,952 approvals per model: both approve 57,335 cases (bad rate `0.9035%`); challenger-only swap-in is 4,617 (bad rate `2.6857%`, simulated ENP +`5.118769M`); scorecard-only swap-out is 4,617 (bad rate `5.7180%`, simulated ENP `8.178317M`). This is a retrospective comparison among observed labeled applications. KGB outcomes are only observed for historically funded loans: rejected applicants have no counterfactual outcome. No reject inference, fabricated labels, or causal approval-effect claim is made. Under Iso-Approval, ticket-size/exposure composition can differ even when counts match; hence approval-count parity does not imply equal volume or profit.

## 7. Monitoring, Recalibration & Iso-Risk Result

Stage 5 fitted intercept and Platt recalibration only on OOT_Valid and applied frozen mappings forward to OOT_Monitor. Monitor outcomes were not used to fit either mapping or select cutoffs. On Monitor, score PSI was `0.0049` for Champion score/PD and raw Challenger PD PSI `0.0772`; both are below the project `0.08` score-drift watch reference, though raw PD is near it. Auxiliary CSI was much higher: tax registry `1.3064`, education `1.2974`, tax-income ratio `0.7491`, family status `0.5973`, employment years `0.2494`. Stable aggregate score PSI does not negate feature-level drift.

| Monitor Iso-Risk | Champion | Challenger M4 | Difference |
|---|---:|---:|---:|
| Approval rate | 70.7970% | 75.1587% | +4.3617 pp |
| Approved cases | 107,404 | 114,021 | +6,617 |
| Observed bad rate | 0.9990% | 0.8832% | −0.1158 pp |
| S1 Baseline Model-Implied Scenario Expected Net Profit Proxy (not realized lifecycle profit; Intercept-Recalibrated PD) | 133.659752M (133,659,752) | 142.305881M (142,305,881) | +8.646129M (+8,646,129; +6.47%) |

This is a blind forward application of Valid-frozen calibration/cutoff policies to observed Monitor data, with simulated ENP assumptions; it is not a randomized business impact estimate. Iso-Approval answers a different question and may favor the Champion because the challenger can select fewer but larger-ticket accounts under matched approval count.

Project operational watch references are PSI `<0.10` normal, `0.10–0.25` amber, `>0.25` red; CSI values use the same proposed triage thresholds, not statutory limits. Stage 5 report defines exact trigger actions and feature-level results.

## 8. Fair Lending & Sensitive Attribute Audit

Sensitive attributes were excluded from predictors. `Gender` was `99.26%` unknown in OOT_Monitor (150,591 records); Train_Dev missingness was `99.17%`, severely limiting gender fairness conclusions. Age-band diagnostic showed applicants `<25` had Challenger AIR `0.5988`, observed bad rate `6.0845%`, mean recalibrated PD `4.8381%`, and `A/E=1.2576`. AIR is a screening diagnostic, not an automatic legal test or evidence of lawful/unlawful discrimination. Higher observed risk partly explains—but does not settle—the disparity; assess policy, proxy effects, access, calibration, and local legal requirements with adequate labels and protected-group data before deployment.

## 9. Concentration, Prior-DPD Migration & Stress Tests

`train_applprev_2` contains 14,075,487 rows and fields `case_id, cacccardblochreas_147M, conts_type_509L, credacc_cards_status_52L, num_group1, num_group2`; it has no consecutive monthly DPD status snapshots. Therefore no synthetic MOB or monthly roll-rate curves were constructed. Instead, the project aggregates 6,525,979 prior-application records and `actualdpd_943P` into empirical prior-contract DPD severity migration against current observed default. “No prior application” and “prior history with missing DPD” remain distinct from zero DPD.

Under S4 Severe Tail Shock, expected profit was `51.00M` Champion and `42.48M` Challenger; realized stress proxies were `83.13M` and `109.14M`. The latter are scenario transformations of observed outcome rates using stated stress multipliers, not observed future downturn P&L. Model-PD and realized-default spreads, including Tier C, reflect different quantities and scenario assumptions; they should not be represented as calibrated stress forecasts. Concentration tables and scenario details are in the Stage 6 report and Power BI marts.

## 10. Limitations Register & Production Governance

| Risk / limitation | Current evidence | Proposed trigger / action |
|---|---|---|
| Population / feature drift | Score PSI low, several CSI values >0.25 | Green `<0.10`; Amber `0.10–0.25` investigate; Red `>0.25` restrict affected feature/use, investigate and escalate |
| Calibration/prevalence shift | Monitor A/E after frozen recalibration differs from 1 | Monthly A/E and calibration plots; consider quarterly intercept review only with recent matured, representative labels |
| Proxy instability | Group 4 tax-related PSI overrides; CSI spikes | Revalidate/remove unstable proxies; never silently retain override |
| Fairness evidence gaps | Gender mostly unknown; youth AIR <0.8 | Obtain valid lawful monitoring data, test proxy pathways and outcome sufficiency; independent fair-lending review |
| Outcome selection | Historical labels only on funded loans | No reject inference without approved methodology and validation; monitor policy/booking changes |
| Simulated economics | APR, COF, OPEX, LGD and stress are assumptions | Replace with approved product economics, realized loss data, funding/servicing costs and sensitivity analysis |
| Lifecycle / data limitations | Cross-sectional records, no monthly MOB snapshots | Do not claim vintage roll rates or causal impact; establish performance data collection |
| Model stability | Retrospective single-dataset work | Independent validation, implementation testing, change control, challenger monitoring, rollback plan before use |

Quarterly intercept recalibration is a proposed governance cadence, not an automatic rule: use only after label maturity and representativeness checks, compare calibration slope/intercept by cohort, document approval, version the mapping, and preserve holdout evaluation. Full rebuild or suspension should be considered when Red drift persists, rank ordering/calibration materially degrades, policy/data definitions change, fairness concerns remain unresolved, or the model ceases to represent the booking population.

### Malaysian regulatory scope

BNM's December 2024 Credit Risk policy document states its scope for financial institutions and is issued under relevant financial-services legislation; it is not a blanket set of rules for every credit provider. Malaysia's Moneylenders Act 1951 (Act 400) includes First Schedule exemptions, including specified authorized persons under the FSA/IFSA, and states annual interest ceilings of 12% for secured and 18% for unsecured loans for transactions to which the Act applies. Entity and product classification must be confirmed by counsel/compliance before applying either framework. Project APR tiers and DSR assumptions above are simulation inputs and should not be described as universal legal caps. BNM's 2025 Personal Financing policy has its own scope and phased effective dates; applicability must be assessed separately.

Sources: [BNM Credit Risk policy (Dec 2024)](https://www.bnm.gov.my/documents/20124/938039/pd_credit%2Brisk_dec2024.pdf); [Moneylenders Act 1951 (Act 400), AGC 2026 reprint](https://lom.agc.gov.my/ilims/upload/portal/akta/outputaktap/2879417_BI/Act%20400%20%28Reprint%202026%29.pdf); [BNM Personal Financing policy (2025)](https://alumni.bnm.gov.my/web/guest/-/pd-pf-925).

### Document control

This is a retrospective project MDVD. Before any real-world use, appoint model owner and independent validator, verify source data/definitions, reproduce metrics from versioned inputs, test scoring implementation and points rounding, approve policy and fairness controls, define monitoring owners and escalation timelines, and obtain applicable legal/compliance review.
