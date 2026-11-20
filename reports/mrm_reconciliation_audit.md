# FinGuard MRM v2 Empirical Reconciliation Audit

生成时间：2026-11-19 16:49:00。本步骤仅写入 reconciled JSON 与本报告；HTML unchanged = **True**。

## 1. Sample selection、outcome observability 与 split reconciliation

Canonical split truth 以已保存 `WEEK_NUM` 分配为准；Train_Dev case-ID artifact、Feature Mart、Stage 3 与 Stage 5 均作交叉核验。

Home Credit 2024 dataset：1.53M historical application-level records；historical decision window 2019-Q1 to 2020-Q4；OOT_Monitor covers 2020-06-30 to 2020-10-05. This is a retrospective historical window, not a 2024 time series.

| Split | Rows | Week range | Date range | Observed target % | Observed bad rate % |
|---|---|---|---|---|---|
| Train_Dev | 1292350 | W0–64 | 2019-01-01 00:00:00–2020-03-30 00:00:00 | 100.0000 | 3.3154 |
| OOT_Valid | 82602 | W65–77 | 2020-03-31 00:00:00–2020-06-29 00:00:00 | 100.0000 | 2.3401 |
| OOT_Monitor | 151707 | W78–91 | 2020-06-30 00:00:00–2020-10-05 00:00:00 | 100.0000 | 2.1192 |

**35-row discrepancy:** current artifacts reproduce no membership mismatch; the reported 1,292,315 / 82,637 pair is not supported by the saved case IDs. Exact discrepant `case_id`s are therefore not identifiable from these inputs. Canonical rows are 1,292,350 / 82,602. See JSON for source-level joins and the empty mismatch audit.

## 2. Reproducible resource telemetry

| Operation | Elapsed seconds | Peak RSS MB | Peak RSS delta MB |
|---|---|---|---|
| DuckDB 3GB sample/mart query | 0.2056 | 333.8895 | 108.9905 |
| OOT_Monitor Champion + M4 + limits | 16.7334 | 541.9663 | 306.8150 |
| Standalone bundle export + JSON round-trip | 8.8204 | 491.4586 | 97.5258 |

Bundle: 13,702,072 UTF-8 bytes, 1,218 trees, schema valid = True.

## 3. Weekly Gini, label maturity and bootstrap CIs

`A/E = Observed Defaults / Expected Defaults`; `E/A = Expected Defaults / Observed Defaults`. Paired applicant bootstrap uses the same sampled case indices for both models; weekly block bootstrap resamples whole `WEEK_NUM` clusters.

| Split | Model | Mean weekly Gini | Weekly Gini slope | Residual std | Stability metric |
|---|---|---|---|---|---|
| OOT_Valid | Challenger LightGBM M4 | — | — | — | — |
| OOT_Valid | Champion Scorecard | — | — | — | — |
| OOT_Monitor | Challenger LightGBM M4 | — | — | — | — |
| OOT_Monitor | Champion Scorecard | — | — | — | — |

OOT_Monitor point estimates and paired differences with 95% intervals are in `section_3_weekly_stability_and_bootstrap.oot_monitor_bootstrap`.

Late-window conclusion: Recorded target labels are 100% observed in all saved splits; there is no missing-label evidence for W86-W91 censoring. These artifacts lack a separate label-maturity/observation-window timestamp, so full outcome maturity cannot be certified. Reported late PD/bad-rate gaps are empirical divergence against recorded labels.

## 4. Economic contribution, frozen cutoffs and reject-bias sensitivity

Realized Contribution = (APR_benchmark - Funding Cost - Opex) / 12 × Principal × amortizing balance factor - Principal × target × LGD; APR_benchmark=15%, COF=4%, Opex=1.5%, LGD=75%.

| Model | Approved | Approval % | Observed bad % | Model-neutral realized contribution | Scenario expected contribution |
|---|---|---|---|---|---|
| Champion Scorecard | 107404 | 70.7970 | 0.9990 | 263,287,512.2541 | 132,180,253.7712 |
| Challenger LightGBM M4 | 114021 | 75.1587 | 0.8832 | 272,540,636.6857 | 140,125,997.0085 |

S4 uses Stage 6 severe shock assumptions: COF 5.5%, LGD 85%, PD/target multipliers 1.30x for tiers A-B and 2.00x for C-E. `k*` sensitivity inflates target-based loss on challenger-only approvals; it is a declared scenario, not observed hidden labels.

| Model | Approved | Expected S4 proxy | Realized-target S4 proxy |
|---|---|---|---|
| Champion Scorecard | 107404 | 50,999,674.1452 | 83,134,886.7645 |
| Challenger LightGBM M4 | 114021 | 42,484,976.8677 | 109,139,074.4980 |

The exact OOT_Valid cutoffs, OOT_Valid metrics, OOT_Monitor results, bootstrap deltas, and Stage 5 provenance are structured in the JSON output.

## 5. Missing income, CSI remediation and fairness proxies

Missing-income alternative policies are counterfactual routing rules; outcomes are not changed or imputed. Fairness rates are descriptive and conditioned on each model's own calibrated-PD tier.

**Validation independence:** Canonical Champion and M4 Full use strict blind `OOT_Monitor` evaluation (`monitor_used_for_fit_or_selection: false`). M4-NoDrift is a **post-hoc drift remediation sensitivity analysis**: its feature exclusions were informed by `Train_Dev vs OOT_Monitor` CSI (`monitor_used_for_fit_or_selection: true`), so it is **not an independent blind OOT validation**. Production promotion requires independent validation on a subsequent mature time window or Shadow Mode.

| Gender value | OOT_Monitor count | Share % |
|---|---|---|
| Unknown | 150591 | 99.2644 |
| F | 926 | 0.6104 |
| M | 190 | 0.1252 |

| Variant | Features | Monitor KS | Score PSI | Max retained-feature CSI | A/E |
|---|---|---|---|---|---|
| M4-NoDrift | 69 | 0.5001 | 0.0069 | 0.2494 | 0.8724 |
| M2 Zero Demographic/Employment Proxies | 45 | 0.4546 | 0.0098 | — | 0.8534 |

Full weekly trajectory, per-tier missing-income metrics, age-band conditional rates and all exact paired/bootstrap tables are in `data/processed/mrm_reconciled_truth.json`.

## Reproducibility notes

Bootstrap B=1000, seed=42; DuckDB memory limit 3GB, 4 threads. The report distinguishes empirical labels from scenario-based assumptions. HTML hashes match before/after: True.

## Detailed statistical tables

### Full W0–W91 trajectory

| Week | Split | Model | n | Observed target % | Bad rate % | Mean recalibrated PD % | Gini |
|---|---|---|---|---|---|---|---|
| 78 | OOT_Monitor | Champion Scorecard | 8755 | 100.0000 | 2.2273 | 2.3320 | 0.6191 |
| 78 | OOT_Monitor | Challenger LightGBM M4 | 8755 | 100.0000 | 2.2273 | 2.7456 | 0.7304 |
| 79 | OOT_Monitor | Champion Scorecard | 5158 | 100.0000 | 2.1714 | 2.4328 | 0.5584 |
| 79 | OOT_Monitor | Challenger LightGBM M4 | 5158 | 100.0000 | 2.1714 | 2.9004 | 0.6558 |
| 80 | OOT_Monitor | Champion Scorecard | 6246 | 100.0000 | 1.9212 | 2.4145 | 0.4749 |
| 80 | OOT_Monitor | Challenger LightGBM M4 | 6246 | 100.0000 | 1.9212 | 2.8622 | 0.5738 |
| 81 | OOT_Monitor | Champion Scorecard | 6743 | 100.0000 | 2.7881 | 2.4682 | 0.4992 |
| 81 | OOT_Monitor | Challenger LightGBM M4 | 6743 | 100.0000 | 2.7881 | 2.8672 | 0.5883 |
| 82 | OOT_Monitor | Champion Scorecard | 5423 | 100.0000 | 2.2497 | 2.5392 | 0.4775 |
| 82 | OOT_Monitor | Challenger LightGBM M4 | 5423 | 100.0000 | 2.2497 | 3.0166 | 0.5853 |
| 83 | OOT_Monitor | Champion Scorecard | 7295 | 100.0000 | 2.1385 | 2.5511 | 0.5203 |
| 83 | OOT_Monitor | Challenger LightGBM M4 | 7295 | 100.0000 | 2.1385 | 2.9581 | 0.6322 |
| 84 | OOT_Monitor | Champion Scorecard | 9721 | 100.0000 | 2.2220 | 2.4684 | 0.5275 |
| 84 | OOT_Monitor | Challenger LightGBM M4 | 9721 | 100.0000 | 2.2220 | 2.8596 | 0.6089 |
| 85 | OOT_Monitor | Champion Scorecard | 16239 | 100.0000 | 1.8967 | 2.4528 | 0.5733 |
| 85 | OOT_Monitor | Challenger LightGBM M4 | 16239 | 100.0000 | 1.8967 | 2.8317 | 0.6736 |
| 86 | OOT_Monitor | Champion Scorecard | 15630 | 100.0000 | 1.7722 | 2.7080 | 0.6260 |
| 86 | OOT_Monitor | Challenger LightGBM M4 | 15630 | 100.0000 | 1.7722 | 3.0373 | 0.7069 |
| 87 | OOT_Monitor | Champion Scorecard | 17886 | 100.0000 | 1.8394 | 2.8207 | 0.5704 |
| 87 | OOT_Monitor | Challenger LightGBM M4 | 17886 | 100.0000 | 1.8394 | 3.0880 | 0.6494 |
| 88 | OOT_Monitor | Champion Scorecard | 14234 | 100.0000 | 1.8617 | 2.9089 | 0.5730 |
| 88 | OOT_Monitor | Challenger LightGBM M4 | 14234 | 100.0000 | 1.8617 | 3.1570 | 0.6818 |
| 89 | OOT_Monitor | Champion Scorecard | 13600 | 100.0000 | 2.2721 | 3.0308 | 0.5450 |
| 89 | OOT_Monitor | Challenger LightGBM M4 | 13600 | 100.0000 | 2.2721 | 3.2386 | 0.6438 |
| 90 | OOT_Monitor | Champion Scorecard | 12103 | 100.0000 | 2.7266 | 3.0943 | 0.5137 |
| 90 | OOT_Monitor | Challenger LightGBM M4 | 12103 | 100.0000 | 2.7266 | 3.3669 | 0.6413 |
| 91 | OOT_Monitor | Champion Scorecard | 12674 | 100.0000 | 2.2724 | 3.2541 | 0.5436 |
| 91 | OOT_Monitor | Challenger LightGBM M4 | 12674 | 100.0000 | 2.2724 | 3.3630 | 0.6562 |
| 65 | OOT_Valid | Champion Scorecard | 4258 | 100.0000 | 2.2546 | 2.4270 | 0.5577 |
| 65 | OOT_Valid | Challenger LightGBM M4 | 4258 | 100.0000 | 2.2546 | 2.3739 | 0.6118 |
| 66 | OOT_Valid | Champion Scorecard | 2278 | 100.0000 | 2.2827 | 2.4108 | 0.4074 |
| 66 | OOT_Valid | Challenger LightGBM M4 | 2278 | 100.0000 | 2.2827 | 2.2690 | 0.5460 |
| 67 | OOT_Valid | Champion Scorecard | 1434 | 100.0000 | 2.0921 | 2.1631 | 0.5662 |
| 67 | OOT_Valid | Challenger LightGBM M4 | 1434 | 100.0000 | 2.0921 | 2.0869 | 0.6315 |
| 68 | OOT_Valid | Champion Scorecard | 825 | 100.0000 | 2.5455 | 2.1286 | 0.4513 |
| 68 | OOT_Valid | Challenger LightGBM M4 | 825 | 100.0000 | 2.5455 | 2.0740 | 0.5050 |
| 69 | OOT_Valid | Champion Scorecard | 1582 | 100.0000 | 2.2124 | 2.5170 | 0.5660 |
| 69 | OOT_Valid | Challenger LightGBM M4 | 1582 | 100.0000 | 2.2124 | 2.5267 | 0.5565 |
| 70 | OOT_Valid | Champion Scorecard | 3808 | 100.0000 | 2.1534 | 2.2514 | 0.6417 |
| 70 | OOT_Valid | Challenger LightGBM M4 | 3808 | 100.0000 | 2.1534 | 2.2915 | 0.7182 |
| 71 | OOT_Valid | Champion Scorecard | 7921 | 100.0000 | 2.2219 | 2.2384 | 0.6005 |
| 71 | OOT_Valid | Challenger LightGBM M4 | 7921 | 100.0000 | 2.2219 | 2.2763 | 0.6645 |
| 72 | OOT_Valid | Champion Scorecard | 8858 | 100.0000 | 2.3369 | 2.3089 | 0.5507 |
| 72 | OOT_Valid | Challenger LightGBM M4 | 8858 | 100.0000 | 2.3369 | 2.3703 | 0.6119 |
| 73 | OOT_Valid | Champion Scorecard | 8687 | 100.0000 | 2.1757 | 2.3262 | 0.4985 |
| 73 | OOT_Valid | Challenger LightGBM M4 | 8687 | 100.0000 | 2.1757 | 2.3211 | 0.5858 |
| 74 | OOT_Valid | Champion Scorecard | 11280 | 100.0000 | 2.6862 | 2.3911 | 0.5579 |
| 74 | OOT_Valid | Challenger LightGBM M4 | 11280 | 100.0000 | 2.6862 | 2.4306 | 0.6603 |
| 75 | OOT_Valid | Champion Scorecard | 13656 | 100.0000 | 2.1236 | 2.3164 | 0.5751 |
| 75 | OOT_Valid | Challenger LightGBM M4 | 13656 | 100.0000 | 2.1236 | 2.2594 | 0.6655 |
| 76 | OOT_Valid | Champion Scorecard | 10261 | 100.0000 | 2.4851 | 2.3445 | 0.5537 |
| 76 | OOT_Valid | Challenger LightGBM M4 | 10261 | 100.0000 | 2.4851 | 2.3595 | 0.6410 |
| 77 | OOT_Valid | Champion Scorecard | 7754 | 100.0000 | 2.5406 | 2.4514 | 0.5304 |
| 77 | OOT_Valid | Challenger LightGBM M4 | 7754 | 100.0000 | 2.5406 | 2.4407 | 0.6036 |
| 0 | Train_Dev | Champion Scorecard | 16735 | 100.0000 | 2.3902 | 1.9262 | 0.5210 |
| 0 | Train_Dev | Challenger LightGBM M4 | 16735 | 100.0000 | 2.3902 | 2.2146 | 0.6450 |
| 1 | Train_Dev | Champion Scorecard | 18841 | 100.0000 | 2.5052 | 2.3060 | 0.4786 |
| 1 | Train_Dev | Challenger LightGBM M4 | 18841 | 100.0000 | 2.5052 | 2.0593 | 0.6494 |
| 2 | Train_Dev | Champion Scorecard | 17476 | 100.0000 | 2.6608 | 2.3760 | 0.4695 |
| 2 | Train_Dev | Challenger LightGBM M4 | 17476 | 100.0000 | 2.6608 | 2.1661 | 0.6437 |
| 3 | Train_Dev | Champion Scorecard | 16108 | 100.0000 | 2.6012 | 2.4346 | 0.4766 |
| 3 | Train_Dev | Challenger LightGBM M4 | 16108 | 100.0000 | 2.6012 | 2.1701 | 0.6462 |
| 4 | Train_Dev | Champion Scorecard | 14309 | 100.0000 | 2.8234 | 2.4463 | 0.4659 |
| 4 | Train_Dev | Challenger LightGBM M4 | 14309 | 100.0000 | 2.8234 | 2.1443 | 0.6412 |
| 5 | Train_Dev | Champion Scorecard | 15335 | 100.0000 | 2.6541 | 2.4253 | 0.5040 |
| 5 | Train_Dev | Challenger LightGBM M4 | 15335 | 100.0000 | 2.6541 | 2.0623 | 0.6678 |
| 6 | Train_Dev | Champion Scorecard | 16385 | 100.0000 | 2.7159 | 2.4358 | 0.4840 |
| 6 | Train_Dev | Challenger LightGBM M4 | 16385 | 100.0000 | 2.7159 | 2.0878 | 0.6363 |
| 7 | Train_Dev | Champion Scorecard | 17238 | 100.0000 | 2.6801 | 2.4684 | 0.5010 |
| 7 | Train_Dev | Challenger LightGBM M4 | 17238 | 100.0000 | 2.6801 | 2.0614 | 0.6745 |
| 8 | Train_Dev | Champion Scorecard | 13076 | 100.0000 | 3.2426 | 2.6172 | 0.4938 |
| 8 | Train_Dev | Challenger LightGBM M4 | 13076 | 100.0000 | 3.2426 | 2.2936 | 0.6532 |
| 9 | Train_Dev | Champion Scorecard | 14716 | 100.0000 | 2.6502 | 2.5699 | 0.5024 |
| 9 | Train_Dev | Challenger LightGBM M4 | 14716 | 100.0000 | 2.6502 | 2.2051 | 0.6698 |
| 10 | Train_Dev | Champion Scorecard | 16650 | 100.0000 | 2.8288 | 2.4752 | 0.4949 |
| 10 | Train_Dev | Challenger LightGBM M4 | 16650 | 100.0000 | 2.8288 | 2.3514 | 0.6398 |
| 11 | Train_Dev | Champion Scorecard | 16101 | 100.0000 | 2.2794 | 2.3934 | 0.5206 |
| 11 | Train_Dev | Challenger LightGBM M4 | 16101 | 100.0000 | 2.2794 | 2.1679 | 0.6853 |
| 12 | Train_Dev | Champion Scorecard | 17105 | 100.0000 | 2.8004 | 2.5301 | 0.4660 |
| 12 | Train_Dev | Challenger LightGBM M4 | 17105 | 100.0000 | 2.8004 | 2.3638 | 0.6318 |
| 13 | Train_Dev | Champion Scorecard | 15791 | 100.0000 | 2.5964 | 2.4372 | 0.5419 |
| 13 | Train_Dev | Challenger LightGBM M4 | 15791 | 100.0000 | 2.5964 | 2.2503 | 0.7069 |
| 14 | Train_Dev | Champion Scorecard | 17534 | 100.0000 | 2.9942 | 2.4614 | 0.4872 |
| 14 | Train_Dev | Challenger LightGBM M4 | 17534 | 100.0000 | 2.9942 | 2.3211 | 0.6746 |
| 15 | Train_Dev | Champion Scorecard | 17713 | 100.0000 | 2.8341 | 2.5031 | 0.5463 |
| 15 | Train_Dev | Challenger LightGBM M4 | 17713 | 100.0000 | 2.8341 | 2.3677 | 0.6872 |
| 16 | Train_Dev | Champion Scorecard | 16598 | 100.0000 | 3.2233 | 2.5780 | 0.5194 |
| 16 | Train_Dev | Challenger LightGBM M4 | 16598 | 100.0000 | 3.2233 | 2.4557 | 0.6714 |
| 17 | Train_Dev | Champion Scorecard | 12146 | 100.0000 | 3.6226 | 2.7609 | 0.5311 |
| 17 | Train_Dev | Challenger LightGBM M4 | 12146 | 100.0000 | 3.6226 | 2.6990 | 0.6736 |
| 18 | Train_Dev | Champion Scorecard | 11729 | 100.0000 | 3.5809 | 2.7519 | 0.5241 |
| 18 | Train_Dev | Challenger LightGBM M4 | 11729 | 100.0000 | 3.5809 | 2.7248 | 0.6723 |
| 19 | Train_Dev | Champion Scorecard | 17194 | 100.0000 | 3.0767 | 2.5196 | 0.5180 |
| 19 | Train_Dev | Challenger LightGBM M4 | 17194 | 100.0000 | 3.0767 | 2.4411 | 0.6728 |
| 20 | Train_Dev | Champion Scorecard | 16366 | 100.0000 | 3.0612 | 2.5922 | 0.4914 |
| 20 | Train_Dev | Challenger LightGBM M4 | 16366 | 100.0000 | 3.0612 | 2.4591 | 0.6753 |
| 21 | Train_Dev | Champion Scorecard | 16962 | 100.0000 | 3.2013 | 2.6381 | 0.5519 |
| 21 | Train_Dev | Challenger LightGBM M4 | 16962 | 100.0000 | 3.2013 | 2.5151 | 0.6945 |
| 22 | Train_Dev | Champion Scorecard | 23261 | 100.0000 | 2.4032 | 2.4966 | 0.5376 |
| 22 | Train_Dev | Challenger LightGBM M4 | 23261 | 100.0000 | 2.4032 | 2.3065 | 0.6796 |
| 23 | Train_Dev | Champion Scorecard | 26734 | 100.0000 | 2.3678 | 2.4926 | 0.5461 |
| 23 | Train_Dev | Challenger LightGBM M4 | 26734 | 100.0000 | 2.3678 | 2.2503 | 0.6863 |
| 24 | Train_Dev | Champion Scorecard | 19503 | 100.0000 | 2.7432 | 2.5843 | 0.5153 |
| 24 | Train_Dev | Challenger LightGBM M4 | 19503 | 100.0000 | 2.7432 | 2.4023 | 0.6625 |
| 25 | Train_Dev | Champion Scorecard | 20744 | 100.0000 | 2.8346 | 2.6109 | 0.5040 |
| 25 | Train_Dev | Challenger LightGBM M4 | 20744 | 100.0000 | 2.8346 | 2.4373 | 0.6648 |
| 26 | Train_Dev | Champion Scorecard | 20621 | 100.0000 | 2.7739 | 2.5894 | 0.5484 |
| 26 | Train_Dev | Challenger LightGBM M4 | 20621 | 100.0000 | 2.7739 | 2.4160 | 0.6800 |
| 27 | Train_Dev | Champion Scorecard | 22450 | 100.0000 | 2.8062 | 2.6203 | 0.5186 |
| 27 | Train_Dev | Challenger LightGBM M4 | 22450 | 100.0000 | 2.8062 | 2.4824 | 0.6825 |
| 28 | Train_Dev | Champion Scorecard | 22013 | 100.0000 | 2.6848 | 2.5401 | 0.5530 |
| 28 | Train_Dev | Challenger LightGBM M4 | 22013 | 100.0000 | 2.6848 | 2.3716 | 0.6960 |
| 29 | Train_Dev | Champion Scorecard | 22841 | 100.0000 | 2.6838 | 2.5395 | 0.5575 |
| 29 | Train_Dev | Challenger LightGBM M4 | 22841 | 100.0000 | 2.6838 | 2.3373 | 0.7077 |
| 30 | Train_Dev | Champion Scorecard | 20688 | 100.0000 | 2.9099 | 2.5718 | 0.5102 |
| 30 | Train_Dev | Challenger LightGBM M4 | 20688 | 100.0000 | 2.9099 | 2.4091 | 0.6681 |
| 31 | Train_Dev | Champion Scorecard | 21275 | 100.0000 | 2.5570 | 2.5548 | 0.5550 |
| 31 | Train_Dev | Challenger LightGBM M4 | 21275 | 100.0000 | 2.5570 | 2.3892 | 0.7039 |
| 32 | Train_Dev | Champion Scorecard | 23889 | 100.0000 | 2.3944 | 2.4924 | 0.5536 |
| 32 | Train_Dev | Challenger LightGBM M4 | 23889 | 100.0000 | 2.3944 | 2.2794 | 0.7070 |
| 33 | Train_Dev | Champion Scorecard | 21525 | 100.0000 | 2.9361 | 2.5639 | 0.5425 |
| 33 | Train_Dev | Challenger LightGBM M4 | 21525 | 100.0000 | 2.9361 | 2.3999 | 0.6912 |
| 34 | Train_Dev | Champion Scorecard | 23477 | 100.0000 | 2.7176 | 2.4994 | 0.4989 |
| 34 | Train_Dev | Challenger LightGBM M4 | 23477 | 100.0000 | 2.7176 | 2.3265 | 0.6617 |
| 35 | Train_Dev | Champion Scorecard | 24137 | 100.0000 | 2.8048 | 2.5273 | 0.5510 |
| 35 | Train_Dev | Challenger LightGBM M4 | 24137 | 100.0000 | 2.8048 | 2.4422 | 0.7004 |
| 36 | Train_Dev | Champion Scorecard | 23812 | 100.0000 | 3.4016 | 2.6696 | 0.5343 |
| 36 | Train_Dev | Challenger LightGBM M4 | 23812 | 100.0000 | 3.4016 | 2.7415 | 0.6836 |
| 37 | Train_Dev | Champion Scorecard | 22255 | 100.0000 | 3.7475 | 2.5910 | 0.5233 |
| 37 | Train_Dev | Challenger LightGBM M4 | 22255 | 100.0000 | 3.7475 | 2.5877 | 0.7011 |
| 38 | Train_Dev | Champion Scorecard | 22424 | 100.0000 | 3.8396 | 2.6533 | 0.5129 |
| 38 | Train_Dev | Challenger LightGBM M4 | 22424 | 100.0000 | 3.8396 | 2.6376 | 0.6731 |
| 39 | Train_Dev | Champion Scorecard | 21039 | 100.0000 | 3.8975 | 2.6415 | 0.5254 |
| 39 | Train_Dev | Challenger LightGBM M4 | 21039 | 100.0000 | 3.8975 | 2.6666 | 0.7130 |
| 40 | Train_Dev | Champion Scorecard | 23907 | 100.0000 | 3.4843 | 2.5952 | 0.5429 |
| 40 | Train_Dev | Challenger LightGBM M4 | 23907 | 100.0000 | 3.4843 | 2.5622 | 0.7042 |
| 41 | Train_Dev | Champion Scorecard | 21125 | 100.0000 | 3.9905 | 2.6392 | 0.5141 |
| 41 | Train_Dev | Challenger LightGBM M4 | 21125 | 100.0000 | 3.9905 | 2.6986 | 0.6830 |
| 42 | Train_Dev | Champion Scorecard | 20643 | 100.0000 | 3.8802 | 2.6020 | 0.5226 |
| 42 | Train_Dev | Challenger LightGBM M4 | 20643 | 100.0000 | 3.8802 | 2.6006 | 0.6684 |
| 43 | Train_Dev | Champion Scorecard | 21029 | 100.0000 | 3.9992 | 2.5762 | 0.4943 |
| 43 | Train_Dev | Challenger LightGBM M4 | 21029 | 100.0000 | 3.9992 | 2.5401 | 0.6774 |
| 44 | Train_Dev | Champion Scorecard | 25887 | 100.0000 | 3.6234 | 2.5367 | 0.5379 |
| 44 | Train_Dev | Challenger LightGBM M4 | 25887 | 100.0000 | 3.6234 | 2.4414 | 0.6998 |
| 45 | Train_Dev | Champion Scorecard | 28947 | 100.0000 | 3.4580 | 2.5281 | 0.4944 |
| 45 | Train_Dev | Challenger LightGBM M4 | 28947 | 100.0000 | 3.4580 | 2.4512 | 0.6845 |
| 46 | Train_Dev | Champion Scorecard | 22299 | 100.0000 | 4.1033 | 2.6289 | 0.4918 |
| 46 | Train_Dev | Challenger LightGBM M4 | 22299 | 100.0000 | 4.1033 | 2.6106 | 0.6541 |
| 47 | Train_Dev | Champion Scorecard | 31888 | 100.0000 | 2.9353 | 2.3770 | 0.5605 |
| 47 | Train_Dev | Challenger LightGBM M4 | 31888 | 100.0000 | 2.9353 | 2.1653 | 0.7322 |
| 48 | Train_Dev | Champion Scorecard | 22298 | 100.0000 | 3.9914 | 2.6920 | 0.5072 |
| 48 | Train_Dev | Challenger LightGBM M4 | 22298 | 100.0000 | 3.9914 | 2.7362 | 0.6834 |
| 49 | Train_Dev | Champion Scorecard | 30938 | 100.0000 | 3.2775 | 2.5234 | 0.5280 |
| 49 | Train_Dev | Challenger LightGBM M4 | 30938 | 100.0000 | 3.2775 | 2.4373 | 0.6925 |
| 50 | Train_Dev | Champion Scorecard | 26244 | 100.0000 | 4.0390 | 2.6385 | 0.4891 |
| 50 | Train_Dev | Challenger LightGBM M4 | 26244 | 100.0000 | 4.0390 | 2.6481 | 0.6856 |
| 51 | Train_Dev | Champion Scorecard | 35920 | 100.0000 | 3.1626 | 2.5441 | 0.5592 |
| 51 | Train_Dev | Challenger LightGBM M4 | 35920 | 100.0000 | 3.1626 | 2.4528 | 0.7156 |
| 52 | Train_Dev | Champion Scorecard | 22915 | 100.0000 | 3.0068 | 2.5778 | 0.5509 |
| 52 | Train_Dev | Challenger LightGBM M4 | 22915 | 100.0000 | 3.0068 | 2.4090 | 0.7057 |
| 53 | Train_Dev | Champion Scorecard | 22194 | 100.0000 | 3.7082 | 2.6623 | 0.5566 |
| 53 | Train_Dev | Challenger LightGBM M4 | 22194 | 100.0000 | 3.7082 | 2.7033 | 0.7192 |
| 54 | Train_Dev | Champion Scorecard | 18740 | 100.0000 | 4.6371 | 2.7613 | 0.4859 |
| 54 | Train_Dev | Challenger LightGBM M4 | 18740 | 100.0000 | 4.6371 | 2.8894 | 0.6733 |
| 55 | Train_Dev | Champion Scorecard | 18538 | 100.0000 | 4.2561 | 2.7120 | 0.5065 |
| 55 | Train_Dev | Challenger LightGBM M4 | 18538 | 100.0000 | 4.2561 | 2.7717 | 0.6059 |
| 56 | Train_Dev | Champion Scorecard | 17072 | 100.0000 | 4.7505 | 2.8651 | 0.5110 |
| 56 | Train_Dev | Challenger LightGBM M4 | 17072 | 100.0000 | 4.7505 | 2.9177 | 0.6141 |
| 57 | Train_Dev | Champion Scorecard | 18221 | 100.0000 | 4.5991 | 2.8966 | 0.4935 |
| 57 | Train_Dev | Challenger LightGBM M4 | 18221 | 100.0000 | 4.5991 | 3.0072 | 0.6043 |
| 58 | Train_Dev | Champion Scorecard | 17819 | 100.0000 | 4.8656 | 2.9946 | 0.5337 |
| 58 | Train_Dev | Challenger LightGBM M4 | 17819 | 100.0000 | 4.8656 | 3.1707 | 0.6303 |
| 59 | Train_Dev | Champion Scorecard | 17729 | 100.0000 | 5.0144 | 2.9510 | 0.5193 |
| 59 | Train_Dev | Challenger LightGBM M4 | 17729 | 100.0000 | 5.0144 | 3.1123 | 0.6135 |
| 60 | Train_Dev | Champion Scorecard | 15864 | 100.0000 | 4.5953 | 2.9422 | 0.4939 |
| 60 | Train_Dev | Challenger LightGBM M4 | 15864 | 100.0000 | 4.5953 | 3.1913 | 0.5883 |
| 61 | Train_Dev | Champion Scorecard | 15568 | 100.0000 | 5.1837 | 3.0742 | 0.4818 |
| 61 | Train_Dev | Challenger LightGBM M4 | 15568 | 100.0000 | 5.1837 | 3.3161 | 0.5847 |
| 62 | Train_Dev | Champion Scorecard | 17164 | 100.0000 | 5.2144 | 2.9842 | 0.5111 |
| 62 | Train_Dev | Challenger LightGBM M4 | 17164 | 100.0000 | 5.2144 | 3.2829 | 0.5942 |
| 63 | Train_Dev | Champion Scorecard | 14337 | 100.0000 | 4.1083 | 2.7626 | 0.5520 |
| 63 | Train_Dev | Challenger LightGBM M4 | 14337 | 100.0000 | 4.1083 | 2.9431 | 0.6459 |
| 64 | Train_Dev | Champion Scorecard | 10268 | 100.0000 | 3.1360 | 2.8014 | 0.5380 |
| 64 | Train_Dev | Challenger LightGBM M4 | 10268 | 100.0000 | 3.1360 | 2.8650 | 0.5880 |

### OOT_Monitor bootstrap B=1,000, seed=42

A/E = Observed Defaults / Expected Defaults; E/A = Expected Defaults / Observed Defaults. Applicant resampling is paired across models; weekly-block resampling retains whole weeks.

| Model / delta | Metric | Estimate | Applicant 95% CI | Weekly block 95% CI |
|---|---|---|---|---|
| Champion Scorecard | roc_auc | 0.7751 | [0.7672619741753773, 0.7828641872624451] | [0.7644465375502919, 0.7860395908093195] |
| Champion Scorecard | gini | 0.5502 | [0.5345239483507546, 0.5657283745248902] | [0.5288930751005837, 0.5720791816186389] |
| Champion Scorecard | ks | 0.4222 | [0.40878445153699383, 0.43920265183346074] | [0.40529542363537735, 0.4461685108999088] |
| Champion Scorecard | brier | 0.0203 | [0.01963725530323543, 0.020888783579818927] | [0.01888826384578767, 0.02207252896067564] |
| Champion Scorecard | log_loss | 0.0926 | [0.090247435821232, 0.09479441630110323] | [0.08716825758195346, 0.0994585281954161] |
| Champion Scorecard | a_over_e | 0.7747 | [0.7479709093816597, 0.7997433488073885] | [0.7162321900495644, 0.8486041706998534] |
| Champion Scorecard | e_over_a | 1.2908 | [1.2504011467193483, 1.336950394504515] | [1.1784057097346032, 1.396195280840867] |
| Challenger LightGBM M4 | roc_auc | 0.8257 | [0.8184698960202236, 0.8321619049411415] | [0.8153157054182705, 0.8357209367475471] |
| Challenger LightGBM M4 | gini | 0.6514 | [0.6369397920404472, 0.6643238098822831] | [0.6306314108365411, 0.6714418734950942] |
| Challenger LightGBM M4 | ks | 0.5006 | [0.4869926430807004, 0.515828780650629] | [0.4815534685431676, 0.5251551340832736] |
| Challenger LightGBM M4 | brier | 0.0195 | [0.018953038041703667, 0.02015366769181994] | [0.01823639503749671, 0.02118217095027739] |
| Challenger LightGBM M4 | log_loss | 0.0873 | [0.08534691174314092, 0.08939951972337282] | [0.08246213046541896, 0.09303367022106702] |
| Challenger LightGBM M4 | a_over_e | 0.6944 | [0.6703736232035715, 0.7159858480800403] | [0.6495493139425808, 0.7501180972888156] |
| Challenger LightGBM M4 | e_over_a | 1.4401 | [1.3966756512414586, 1.4917054720974916] | [1.3331234157577239, 1.5395290143300426] |
| M4 - Champion | ks | 0.0785 | [0.06424260645302894, 0.09132391386740635] | [0.06716822731752267, 0.08865050905264026] |
| M4 - Champion | roc_auc | 0.0506 | [0.04518624847433188, 0.05594810567581218] | [0.046559030724152836, 0.054515157428587105] |
| M4 - Champion | gini | 0.1012 | [0.09037249694866376, 0.11189621135162436] | [0.09311806144830567, 0.10903031485717421] |
| M4 - Champion | brier | -0.0007 | [-0.0008335498718507799, -0.0006246492906997877] | [-0.0009302435504586039, -0.0005533110478755205] |
| M4 - Champion | log_loss | -0.0052 | [-0.005799978547403749, -0.0046160332051879156] | [-0.006307703128764585, -0.004398732586575401] |
| M4 - Champion | a_over_e | -0.0803 | [-0.08507317234769811, -0.07531151479887162] | [-0.10390534259452094, -0.06300148304744405] |
| M4 - Champion | e_over_a | 0.1493 | [0.1414549793976644, 0.15763776720168937] | [0.12023126951617906, 0.17856938656721952] |

### Valid-frozen policy results (OOT_Valid calibration, OOT_Monitor blind evaluation)

| Split | Model | Policy | Cutoff | Apps | Approved | Approval % | Bads | Bad rate % |
|---|---|---|---|---|---|---|---|---|
| OOT_Valid | Champion Scorecard | Valid-Frozen 75% Quantile Policy | 0.0308 | 82602 | 61952 | 75.0006 | 782 | 1.2623 |
| OOT_Valid | Champion Scorecard | Valid-Frozen Iso-Risk Policy | 0.0308 | 82602 | 61952 | 75.0006 | 782 | 1.2623 |
| OOT_Valid | Challenger LightGBM M4 | Valid-Frozen 75% Quantile Policy | 0.0336 | 82602 | 61952 | 75.0006 | 642 | 1.0363 |
| OOT_Valid | Challenger LightGBM M4 | Valid-Frozen Iso-Risk Policy | 0.0465 | 82602 | 66547 | 80.5634 | 840 | 1.2623 |
| OOT_Monitor | Champion Scorecard | Valid-Frozen 75% Quantile Policy | 0.0308 | 151707 | 107404 | 70.7970 | 1073 | 0.9990 |
| OOT_Monitor | Champion Scorecard | Valid-Frozen Iso-Risk Policy | 0.0308 | 151707 | 107404 | 70.7970 | 1073 | 0.9990 |
| OOT_Monitor | Challenger LightGBM M4 | Valid-Frozen 75% Quantile Policy | 0.0336 | 151707 | 101772 | 67.0846 | 715 | 0.7026 |
| OOT_Monitor | Challenger LightGBM M4 | Valid-Frozen Iso-Risk Policy | 0.0465 | 151707 | 114021 | 75.1587 | 1007 | 0.8832 |

Stage 5 policy provenance: `[{"Scenario": "S1_Baseline", "Model": "Champion Scorecard", "Approved_Count": 107404, "Monitor_Approval_Pct": 70.7969968425979, "Observed_Bad_Rate_Pct": 0.9990316934192395, "Valid_PD_Cutoff": 0.030810455894633697, "Source": "data/powerbi/pbi_04_stress_scenarios.csv"}, {"Scenario": "S1_Baseline", "Model": "Challenger LightGBM M4", "Approved_Count": 114021, "Monitor_Approval_Pct": 75.15869406157923, "Observed_Bad_Rate_Pct": 0.8831706440041746, "Valid_PD_Cutoff": 0.04648983019655445, "Source": "data/powerbi/pbi_04_stress_scenarios.csv"}]`

Paired policy bootstrap: `{"point_estimates": {"delta_approval_pp": 4.36169721898132, "delta_bad_rate_bps": -11.586104941506491, "delta_realized_contribution": 9253124.431561798}, "bootstrap_replicates": 1000, "seed": 42, "confidence_intervals_95": {"delta_approval_pp": [4.1692044533212105, 4.535716875292495], "delta_bad_rate_bps": [-15.675153931055599, -7.280464732638189], "delta_realized_contribution": [6923463.617650903, 11363226.992321525]}, "paired_same_applicants": true}`

### KGB sensitivity

| k | Net swap contribution |
|---|---|
| 1.0000 | 14,838,834.2210 |
| 1.2500 | 12,979,095.7678 |
| 1.5000 | 11,119,357.3146 |
| 2.0000 | 7,399,880.4082 |
| 2.5000 | 3,680,403.5017 |
| 3.0000 | -39,073.4047 |
| 4.0000 | -7,478,027.2175 |

### Missing-income tier outcomes and fallback multipliers

| Split | Income status | Risk tier | Count | Share % | Approval % | Bad rate % | Mean PD % | Fallback multiplier |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |

### Gender missingness on OOT_Monitor

| Group | Count | Share % |
|---|---|---|
| Unknown | 150591 | 99.2644 |
| F | 926 | 0.6104 |
| M | 190 | 0.1252 |

### Age fairness by model and risk tier

| Model | Age | Risk tier | n | Approval % | Bad rate % | False rejection good % | False approval bad % | AIR |
|---|---|---|---|---|---|---|---|---|
| Champion Scorecard | <25 | Tier A: Super Prime | 586 | 98.8055 | 1.7065 | 1.0417 | 90.0000 | 0.8458 |
| Champion Scorecard | <25 | Tier B: Prime | 2314 | 98.5307 | 2.7658 | 1.4222 | 96.8750 | 0.8458 |
| Champion Scorecard | <25 | Tier C: Standard | 2051 | 69.6733 | 6.9235 | 29.9109 | 64.0845 | 0.8458 |
| Champion Scorecard | <25 | Tier D: High Risk | 832 | 0.0000 | 8.7740 | 100.0000 | 0.0000 | 0.8458 |
| Champion Scorecard | <25 | Tier E: Decline | 561 | 0.0000 | 17.2906 | 100.0000 | 0.0000 | 0.8458 |
| Champion Scorecard | 25-34 | Tier A: Super Prime | 4450 | 94.3371 | 0.2697 | 5.6557 | 91.6667 | 0.7559 |
| Champion Scorecard | 25-34 | Tier B: Prime | 9044 | 92.9898 | 0.8846 | 7.0393 | 96.2500 | 0.7559 |
| Champion Scorecard | 25-34 | Tier C: Standard | 7890 | 60.4436 | 2.9785 | 39.4252 | 56.1702 | 0.7559 |
| Champion Scorecard | 25-34 | Tier D: High Risk | 3851 | 0.0000 | 5.0117 | 100.0000 | 0.0000 | 0.7559 |
| Champion Scorecard | 25-34 | Tier E: Decline | 3531 | 0.0000 | 9.6290 | 100.0000 | 0.0000 | 0.7559 |
| Champion Scorecard | 35-49 | Tier A: Super Prime | 10049 | 92.6560 | 0.4578 | 7.3378 | 91.3043 | 0.8033 |
| Champion Scorecard | 35-49 | Tier B: Prime | 15509 | 91.7274 | 0.8189 | 8.3084 | 96.0630 | 0.8033 |
| Champion Scorecard | 35-49 | Tier C: Standard | 11461 | 59.7068 | 1.6578 | 40.1029 | 48.4211 | 0.8033 |
| Champion Scorecard | 35-49 | Tier D: High Risk | 5346 | 0.0000 | 4.0778 | 100.0000 | 0.0000 | 0.8033 |
| Champion Scorecard | 35-49 | Tier E: Decline | 4959 | 0.0000 | 8.7719 | 100.0000 | 0.0000 | 0.8033 |
| Champion Scorecard | 50+ | Tier A: Super Prime | 31386 | 97.1675 | 0.4620 | 2.8328 | 97.2414 | 1.0000 |
| Champion Scorecard | 50+ | Tier B: Prime | 19091 | 94.3324 | 0.9219 | 5.6886 | 96.5909 | 1.0000 |
| Champion Scorecard | 50+ | Tier C: Standard | 10955 | 62.5559 | 1.9261 | 37.3697 | 58.7678 | 1.0000 |
| Champion Scorecard | 50+ | Tier D: High Risk | 4695 | 0.0000 | 3.8552 | 100.0000 | 0.0000 | 1.0000 |
| Champion Scorecard | 50+ | Tier E: Decline | 3146 | 0.0000 | 7.6287 | 100.0000 | 0.0000 | 1.0000 |
| M2 Zero Demographic/Employment Proxies | <25 | Tier A: Super Prime | 1998 | 95.2452 | 6.3063 | 4.4872 | 91.2698 | 1.0000 |
| M2 Zero Demographic/Employment Proxies | <25 | Tier B: Prime | 2072 | 96.7181 | 6.1293 | 2.9306 | 91.3386 | 1.0000 |
| M2 Zero Demographic/Employment Proxies | <25 | Tier C: Standard | 1238 | 88.6107 | 6.6236 | 10.9862 | 82.9268 | 1.0000 |
| M2 Zero Demographic/Employment Proxies | <25 | Tier D: High Risk | 647 | 0.0000 | 5.2550 | 100.0000 | 0.0000 | 1.0000 |
| M2 Zero Demographic/Employment Proxies | <25 | Tier E: Decline | 389 | 0.0000 | 4.3702 | 100.0000 | 0.0000 | 1.0000 |
| M2 Zero Demographic/Employment Proxies | 25-34 | Tier A: Super Prime | 10703 | 89.1059 | 2.7469 | 10.7983 | 85.7143 | 0.9356 |
| M2 Zero Demographic/Employment Proxies | 25-34 | Tier B: Prime | 7906 | 91.2472 | 2.8965 | 8.6101 | 86.4629 | 0.9356 |
| M2 Zero Demographic/Employment Proxies | 25-34 | Tier C: Standard | 5308 | 84.3632 | 3.4476 | 15.5122 | 80.8743 | 0.9356 |
| M2 Zero Demographic/Employment Proxies | 25-34 | Tier D: High Risk | 2839 | 0.0000 | 3.1349 | 100.0000 | 0.0000 | 0.9356 |
| M2 Zero Demographic/Employment Proxies | 25-34 | Tier E: Decline | 2010 | 0.0000 | 3.2338 | 100.0000 | 0.0000 | 0.9356 |
| M2 Zero Demographic/Employment Proxies | 35-49 | Tier A: Super Prime | 17926 | 87.8501 | 2.1087 | 12.1324 | 87.0370 | 0.9112 |
| M2 Zero Demographic/Employment Proxies | 35-49 | Tier B: Prime | 12198 | 90.1377 | 2.2217 | 9.6588 | 81.1808 | 0.9112 |
| M2 Zero Demographic/Employment Proxies | 35-49 | Tier C: Standard | 8788 | 82.7492 | 2.0824 | 17.1412 | 77.5956 | 0.9112 |
| M2 Zero Demographic/Employment Proxies | 35-49 | Tier D: High Risk | 4885 | 0.0000 | 2.1699 | 100.0000 | 0.0000 | 0.9112 |
| M2 Zero Demographic/Employment Proxies | 35-49 | Tier E: Decline | 3527 | 0.0000 | 2.2115 | 100.0000 | 0.0000 | 0.9112 |
| M2 Zero Demographic/Employment Proxies | 50+ | Tier A: Super Prime | 27246 | 93.1953 | 1.3507 | 6.7713 | 90.7609 | 0.9716 |
| M2 Zero Demographic/Employment Proxies | 50+ | Tier B: Prime | 17762 | 94.0378 | 1.4356 | 5.9348 | 92.1569 | 0.9716 |
| M2 Zero Demographic/Employment Proxies | 50+ | Tier C: Standard | 12620 | 87.1157 | 1.3550 | 12.7962 | 80.7018 | 0.9716 |
| M2 Zero Demographic/Employment Proxies | 50+ | Tier D: High Risk | 6836 | 0.0000 | 1.2873 | 100.0000 | 0.0000 | 0.9716 |
| M2 Zero Demographic/Employment Proxies | 50+ | Tier E: Decline | 4809 | 0.0000 | 1.4764 | 100.0000 | 0.0000 | 0.9716 |
| M4-NoDrift | <25 | Tier A: Super Prime | 2619 | 95.6090 | 6.6819 | 4.0917 | 91.4286 | 1.0000 |
| M4-NoDrift | <25 | Tier B: Prime | 1455 | 95.8076 | 5.9794 | 3.8743 | 90.8046 | 1.0000 |
| M4-NoDrift | <25 | Tier C: Standard | 1247 | 96.8725 | 5.6135 | 2.7188 | 90.0000 | 1.0000 |
| M4-NoDrift | <25 | Tier D: High Risk | 561 | 13.7255 | 6.5954 | 86.2595 | 13.5135 | 1.0000 |
| M4-NoDrift | <25 | Tier E: Decline | 462 | 0.0000 | 3.6797 | 100.0000 | 0.0000 | 1.0000 |
| M4-NoDrift | 25-34 | Tier A: Super Prime | 13088 | 89.1198 | 2.8117 | 10.7390 | 84.2391 | 0.9376 |
| M4-NoDrift | 25-34 | Tier B: Prime | 5965 | 91.0813 | 2.9673 | 8.8804 | 89.8305 | 0.9376 |
| M4-NoDrift | 25-34 | Tier C: Standard | 4997 | 91.2948 | 3.4421 | 8.5181 | 86.0465 | 0.9376 |
| M4-NoDrift | 25-34 | Tier D: High Risk | 2547 | 14.7232 | 2.8269 | 85.1717 | 11.1111 | 0.9376 |
| M4-NoDrift | 25-34 | Tier E: Decline | 2169 | 0.0000 | 3.2734 | 100.0000 | 0.0000 | 0.9376 |
| M4-NoDrift | 35-49 | Tier A: Super Prime | 21665 | 88.2068 | 2.1417 | 11.7164 | 84.6983 | 0.9157 |
| M4-NoDrift | 35-49 | Tier B: Prime | 9594 | 89.2329 | 2.1368 | 10.6934 | 85.8537 | 0.9157 |
| M4-NoDrift | 35-49 | Tier C: Standard | 8026 | 89.6337 | 2.3424 | 10.2322 | 84.0426 | 0.9157 |
| M4-NoDrift | 35-49 | Tier D: High Risk | 4331 | 12.4683 | 1.8471 | 87.4618 | 8.7500 | 0.9157 |
| M4-NoDrift | 35-49 | Tier E: Decline | 3708 | 0.0000 | 2.1305 | 100.0000 | 0.0000 | 0.9157 |
| M4-NoDrift | 50+ | Tier A: Super Prime | 32460 | 93.2348 | 1.4048 | 6.7210 | 90.1316 | 0.9758 |
| M4-NoDrift | 50+ | Tier B: Prime | 13935 | 93.7424 | 1.3061 | 6.2314 | 91.7582 | 0.9758 |
| M4-NoDrift | 50+ | Tier C: Standard | 11718 | 93.6252 | 1.3910 | 6.3003 | 88.3436 | 0.9758 |
| M4-NoDrift | 50+ | Tier D: High Risk | 5958 | 15.5589 | 1.4938 | 84.4266 | 14.6067 | 0.9758 |
| M4-NoDrift | 50+ | Tier E: Decline | 5202 | 0.0000 | 1.2111 | 100.0000 | 0.0000 | 0.9758 |
| M4 Full | <25 | Tier A: Super Prime | 659 | 96.3581 | 1.2140 | 3.6866 | 100.0000 | 0.7585 |
| M4 Full | <25 | Tier B: Prime | 1064 | 96.2406 | 1.9737 | 3.4516 | 80.9524 | 0.7585 |
| M4 Full | <25 | Tier C: Standard | 1897 | 97.3643 | 3.3737 | 2.6732 | 98.4375 | 0.7585 |
| M4 Full | <25 | Tier D: High Risk | 1478 | 32.4763 | 6.5629 | 66.9080 | 23.7113 | 0.7585 |
| M4 Full | <25 | Tier E: Decline | 1246 | 0.0000 | 15.7303 | 100.0000 | 0.0000 | 0.7585 |
| M4 Full | 25-34 | Tier A: Super Prime | 6803 | 90.6953 | 0.1764 | 9.3064 | 91.6667 | 0.8064 |
| M4 Full | 25-34 | Tier B: Prime | 6129 | 90.5531 | 0.8811 | 9.4486 | 90.7407 | 0.8064 |
| M4 Full | 25-34 | Tier C: Standard | 6763 | 91.3204 | 1.8187 | 8.6596 | 90.2439 | 0.8064 |
| M4 Full | 25-34 | Tier D: High Risk | 4658 | 28.3169 | 4.4225 | 71.3387 | 20.8738 | 0.8064 |
| M4 Full | 25-34 | Tier E: Decline | 4413 | 0.0000 | 10.5370 | 100.0000 | 0.0000 | 0.8064 |
| M4 Full | 35-49 | Tier A: Super Prime | 14080 | 90.0000 | 0.2557 | 9.9972 | 88.8889 | 0.8531 |
| M4 Full | 35-49 | Tier B: Prime | 10445 | 89.2963 | 0.5744 | 10.6981 | 88.3333 | 0.8531 |
| M4 Full | 35-49 | Tier C: Standard | 10670 | 89.3627 | 1.3308 | 10.6193 | 88.0282 | 0.8531 |
| M4 Full | 35-49 | Tier D: High Risk | 6616 | 28.8089 | 3.0834 | 71.0075 | 23.0392 | 0.8531 |
| M4 Full | 35-49 | Tier E: Decline | 5513 | 0.0000 | 10.4118 | 100.0000 | 0.0000 | 0.8531 |
| M4 Full | 50+ | Tier A: Super Prime | 29961 | 95.8513 | 0.2704 | 4.1600 | 100.0000 | 1.0000 |
| M4 Full | 50+ | Tier B: Prime | 17231 | 94.5157 | 0.8473 | 5.4843 | 94.5205 | 1.0000 |
| M4 Full | 50+ | Tier C: Standard | 11606 | 91.3752 | 1.4734 | 8.6926 | 95.9064 | 1.0000 |
| M4 Full | 50+ | Tier D: High Risk | 5953 | 29.7497 | 2.8389 | 70.1072 | 24.8521 | 1.0000 |
| M4 Full | 50+ | Tier E: Decline | 4522 | 0.0000 | 8.5360 | 100.0000 | 0.0000 | 1.0000 |

Fairness definitions: `{"false_rejection_good": "good declined / all good in age × model-tier cell", "false_approval_bad": "bad approved / all bad in age × model-tier cell", "AIR": "age-band approval rate / maximum age-band approval rate for the same model"}`


### Missing-income aggregate and execution metadata

Verified executable: `D:\data analysis\FinGuard\.venv\Scripts\python.exe`.

- Train_Dev: 436,089/1,292,350 missing/unverified income (33.7439%).
- OOT_Monitor: 48,782/151,707 missing/unverified income (32.1554%).