# Stage 5 OOT Monitoring、Recalibration 与 Fair Lending 审计

`OOT_Valid` (`WEEK_NUM 65–77`) exclusively fitted both recalibrators. `OOT_Monitor` (`WEEK_NUM 78–91`, 151,707 rows) was scored only after parameters were frozen and used only for forward blind evaluation. Stage 5 Challenger is the approved `pd_lgbm_raw`, not the Stage 3 isotonic output. `OOT_Monitor` labels were not used in recalibration, CSI bins, tier/cutoff fitting, or Iso-Risk threshold selection.

## Recalibration parameters

- Scorecard Method A: alpha=-0.268171, beta fixed at 1. `Base_Score_Offset` changes from 383.9036 to 364.5592, a shift of -19.3444 points (PDO factor 72.1348); the score ranking and PDO factor are preserved.
- Scorecard Method B: alpha=-0.164385, beta=1.034478.
- Raw LightGBM M4 Method A: alpha=-0.278122, beta fixed at 1.
- Raw LightGBM M4 Method B: alpha=-0.109729, beta=1.059846.

## Table 1: Score PSI 与 Top-15 Feature CSI

| Type        | Feature / Score               | Comparison               |   PSI / CSI | Reference Status   |
|:------------|:------------------------------|:-------------------------|------------:|:-------------------|
| Score PSI   | Champion Scorecard Score      | Train_Dev vs OOT_Valid   |      0.0271 | Below 0.10         |
| Score PSI   | Champion Scorecard Score      | Train_Dev vs OOT_Monitor |      0.0049 | Below 0.10         |
| Score PSI   | Champion Scorecard PD         | Train_Dev vs OOT_Valid   |      0.0272 | Below 0.10         |
| Score PSI   | Champion Scorecard PD         | Train_Dev vs OOT_Monitor |      0.0049 | Below 0.10         |
| Score PSI   | Raw LightGBM M4 PD            | Train_Dev vs OOT_Valid   |      0.0044 | Below 0.10         |
| Score PSI   | Raw LightGBM M4 PD            | Train_Dev vs OOT_Monitor |      0.0772 | Below 0.10         |
| Feature CSI | static_days_overdue_tol       | Train_Dev vs OOT_Valid   |      0.0388 | Below 0.10         |
| Feature CSI | static_days_overdue_tol       | Train_Dev vs OOT_Monitor |      0.0608 | Below 0.10         |
| Feature CSI | static_pct_inst_paid_late_10d | Train_Dev vs OOT_Valid   |      0.0300 | Below 0.10         |
| Feature CSI | static_pct_inst_paid_late_10d | Train_Dev vs OOT_Monitor |      0.0475 | Below 0.10         |
| Feature CSI | static_num_inst_paid_early    | Train_Dev vs OOT_Valid   |      0.0523 | Below 0.10         |
| Feature CSI | static_num_inst_paid_early    | Train_Dev vs OOT_Monitor |      0.0739 | Below 0.10         |
| Feature CSI | bureau_queries_120d           | Train_Dev vs OOT_Valid   |      0.1472 | Moderate shift     |
| Feature CSI | bureau_queries_120d           | Train_Dev vs OOT_Monitor |      0.1297 | Moderate shift     |
| Feature CSI | prev_appl_avg_tenor           | Train_Dev vs OOT_Valid   |      0.0357 | Below 0.10         |
| Feature CSI | prev_appl_avg_tenor           | Train_Dev vs OOT_Monitor |      0.0625 | Below 0.10         |
| Feature CSI | bureau_queries_30d            | Train_Dev vs OOT_Valid   |      0.0863 | Below 0.10         |
| Feature CSI | bureau_queries_30d            | Train_Dev vs OOT_Monitor |      0.0763 | Below 0.10         |
| Feature CSI | person_income_type            | Train_Dev vs OOT_Valid   |      0.1474 | Moderate shift     |
| Feature CSI | person_income_type            | Train_Dev vs OOT_Monitor |      0.2245 | Moderate shift     |
| Feature CSI | person_employment_years       | Train_Dev vs OOT_Valid   |      0.3288 | Major shift        |
| Feature CSI | person_employment_years       | Train_Dev vs OOT_Monitor |      0.2494 | Moderate shift     |
| Feature CSI | person_education_level        | Train_Dev vs OOT_Valid   |      0.2689 | Major shift        |
| Feature CSI | person_education_level        | Train_Dev vs OOT_Monitor |      1.2974 | Major shift        |
| Feature CSI | static_cnt_pmts_months_24m    | Train_Dev vs OOT_Valid   |      0.0475 | Below 0.10         |
| Feature CSI | static_cnt_pmts_months_24m    | Train_Dev vs OOT_Monitor |      0.0818 | Below 0.10         |
| Feature CSI | person_family_status          | Train_Dev vs OOT_Valid   |      0.0864 | Below 0.10         |
| Feature CSI | person_family_status          | Train_Dev vs OOT_Monitor |      0.5973 | Major shift        |
| Feature CSI | bureau_recent_query_velocity  | Train_Dev vs OOT_Valid   |      0.0303 | Below 0.10         |
| Feature CSI | bureau_recent_query_velocity  | Train_Dev vs OOT_Monitor |      0.0492 | Below 0.10         |
| Feature CSI | prev_appl_total_pmt_count     | Train_Dev vs OOT_Valid   |      0.0654 | Below 0.10         |
| Feature CSI | prev_appl_total_pmt_count     | Train_Dev vs OOT_Monitor |      0.1021 | Moderate shift     |
| Feature CSI | has_tax_registry_record       | Train_Dev vs OOT_Valid   |      0.6710 | Major shift        |
| Feature CSI | has_tax_registry_record       | Train_Dev vs OOT_Monitor |      1.3064 | Major shift        |
| Feature CSI | tax_verified_income_ratio     | Train_Dev vs OOT_Valid   |      0.3756 | Major shift        |
| Feature CSI | tax_verified_income_ratio     | Train_Dev vs OOT_Monitor |      0.7491 | Major shift        |

0.10 (moderate) and 0.25 (major) are configurable operational monitoring reference thresholds only, not universal statutory standards.

### Top 15 Feature CSI ranking

| Feature / Score               |   Train_Dev vs OOT_Monitor |   Train_Dev vs OOT_Valid |   Max CSI |
|:------------------------------|---------------------------:|-------------------------:|----------:|
| has_tax_registry_record       |                     1.3064 |                   0.6710 |    1.3064 |
| person_education_level        |                     1.2974 |                   0.2689 |    1.2974 |
| tax_verified_income_ratio     |                     0.7491 |                   0.3756 |    0.7491 |
| person_family_status          |                     0.5973 |                   0.0864 |    0.5973 |
| person_employment_years       |                     0.2494 |                   0.3288 |    0.3288 |
| person_income_type            |                     0.2245 |                   0.1474 |    0.2245 |
| bureau_queries_120d           |                     0.1297 |                   0.1472 |    0.1472 |
| prev_appl_total_pmt_count     |                     0.1021 |                   0.0654 |    0.1021 |
| bureau_queries_30d            |                     0.0763 |                   0.0863 |    0.0863 |
| static_cnt_pmts_months_24m    |                     0.0818 |                   0.0475 |    0.0818 |
| static_num_inst_paid_early    |                     0.0739 |                   0.0523 |    0.0739 |
| prev_appl_avg_tenor           |                     0.0625 |                   0.0357 |    0.0625 |
| static_days_overdue_tol       |                     0.0608 |                   0.0388 |    0.0608 |
| bureau_recent_query_velocity  |                     0.0492 |                   0.0303 |    0.0492 |
| static_pct_inst_paid_late_10d |                     0.0475 |                   0.0300 |    0.0475 |

## Table 2: Blind OOT_Monitor Performance Before / After Recalibration

| Model              | Method               |   ROC-AUC |     KS |   Gini |   Brier Score |   Mean Predicted PD % |   Actual Bad Rate % |   A/E Ratio |
|:-------------------|:---------------------|----------:|-------:|-------:|--------------:|----------------------:|--------------------:|------------:|
| Champion Scorecard | Before Recalibration |    0.7751 | 0.4222 | 0.5502 |        0.0206 |                3.4839 |              2.1192 |      0.6083 |
| Champion Scorecard | Intercept Recal      |    0.7751 | 0.4222 | 0.5502 |        0.0202 |                2.7205 |              2.1192 |      0.7790 |
| Champion Scorecard | Platt Recal          |    0.7751 | 0.4222 | 0.5502 |        0.0203 |                2.7355 |              2.1192 |      0.7747 |
| Raw LightGBM M4    | Before Recalibration |    0.8257 | 0.5006 | 0.6514 |        0.0199 |                3.8975 |              2.1192 |      0.5437 |
| Raw LightGBM M4    | Intercept Recal      |    0.8257 | 0.5006 | 0.6514 |        0.0195 |                3.0259 |              2.1192 |      0.7004 |
| Raw LightGBM M4    | Platt Recal          |    0.8257 | 0.5006 | 0.6514 |        0.0195 |                3.0519 |              2.1192 |      0.6944 |

### Tier-level A/E by method (tiered on each method's PD bands)

| Model              | Method               | Risk Tier           |   Count |   Actual Bad Rate % |   Mean Predicted PD % |   A/E Ratio |
|:-------------------|:---------------------|:--------------------|--------:|--------------------:|----------------------:|------------:|
| Champion Scorecard | Before Recalibration | Tier A: Super Prime |   27941 |              0.3686 |                0.7280 |      0.5064 |
| Champion Scorecard | Before Recalibration | Tier B: Prime       |   45216 |              0.7055 |                1.4670 |      0.4809 |
| Champion Scorecard | Before Recalibration | Tier C: Standard    |   41144 |              1.7062 |                2.7975 |      0.6099 |
| Champion Scorecard | Before Recalibration | Tier D: High Risk   |   19971 |              3.5001 |                5.2203 |      0.6705 |
| Champion Scorecard | Before Recalibration | Tier E: Decline     |   17435 |              7.9839 |               12.7615 |      0.6256 |
| Champion Scorecard | Intercept Recal      | Tier A: Super Prime |   43898 |              0.4351 |                0.6759 |      0.6437 |
| Champion Scorecard | Intercept Recal      | Tier B: Prime       |   47362 |              0.9565 |                1.4477 |      0.6607 |
| Champion Scorecard | Intercept Recal      | Tier C: Standard    |   33643 |              2.3749 |                2.7961 |      0.8494 |
| Champion Scorecard | Intercept Recal      | Tier D: High Risk   |   14979 |              4.4796 |                5.2120 |      0.8595 |
| Champion Scorecard | Intercept Recal      | Tier E: Decline     |   11825 |              9.3108 |               12.0374 |      0.7735 |
| Champion Scorecard | Platt Recal          | Tier A: Super Prime |   46471 |              0.4584 |                0.6573 |      0.6973 |
| Champion Scorecard | Platt Recal          | Tier B: Prime       |   45958 |              0.9726 |                1.4414 |      0.6748 |
| Champion Scorecard | Platt Recal          | Tier C: Standard    |   32357 |              2.4044 |                2.7971 |      0.8596 |
| Champion Scorecard | Platt Recal          | Tier D: High Risk   |   14724 |              4.5164 |                5.2233 |      0.8647 |
| Champion Scorecard | Platt Recal          | Tier E: Decline     |   12197 |              9.1170 |               12.3627 |      0.7375 |
| Raw LightGBM M4    | Before Recalibration | Tier A: Super Prime |   34625 |              0.2224 |                0.5952 |      0.3736 |
| Raw LightGBM M4    | Before Recalibration | Tier B: Prime       |   34661 |              0.5193 |                1.4675 |      0.3539 |
| Raw LightGBM M4    | Before Recalibration | Tier C: Standard    |   35789 |              1.1763 |                2.8569 |      0.4118 |
| Raw LightGBM M4    | Before Recalibration | Tier D: High Risk   |   23662 |              2.5568 |                5.2886 |      0.4835 |
| Raw LightGBM M4    | Before Recalibration | Tier E: Decline     |   22970 |              8.4110 |               12.7306 |      0.6607 |
| Raw LightGBM M4    | Intercept Recal      | Tier A: Super Prime |   47182 |              0.2565 |                0.5657 |      0.4534 |
| Raw LightGBM M4    | Intercept Recal      | Tier B: Prime       |   36842 |              0.7247 |                1.4485 |      0.5003 |
| Raw LightGBM M4    | Intercept Recal      | Tier C: Standard    |   33097 |              1.5802 |                2.8564 |      0.5532 |
| Raw LightGBM M4    | Intercept Recal      | Tier D: High Risk   |   19564 |              3.6496 |                5.2356 |      0.6971 |
| Raw LightGBM M4    | Intercept Recal      | Tier E: Decline     |   15022 |             10.5845 |               12.1176 |      0.8735 |
| Raw LightGBM M4    | Platt Recal          | Tier A: Super Prime |   51503 |              0.2660 |                0.5412 |      0.4915 |
| Raw LightGBM M4    | Platt Recal          | Tier B: Prime       |   34869 |              0.8059 |                1.4467 |      0.5571 |
| Raw LightGBM M4    | Platt Recal          | Tier C: Standard    |   30936 |              1.6162 |                2.8592 |      0.5653 |
| Raw LightGBM M4    | Platt Recal          | Tier D: High Risk   |   18705 |              3.6140 |                5.2466 |      0.6888 |
| Raw LightGBM M4    | Platt Recal          | Tier E: Decline     |   15694 |             10.3288 |               12.6214 |      0.8184 |

Intercept-only recalibration preserves exact rank when beta=1. Platt recalibration also preserves ordering if beta remains positive; both conditions are verified from fitted parameters. Weekly and monthly tracking for weeks 0–91 is saved in `data/processed/stage5_temporal_tracking.csv`.

## Table 3: OOT_Monitor Iso-Approval vs Iso-Risk Commercial Comparison

Iso-Approval thresholds match 75% acceptance on OOT_Valid by unlabeled risk rank and then apply forward to OOT_Monitor. Iso-Risk sets the Challenger cutoff on OOT_Valid to match Champion Scorecard's observed bad rate at its 75% OOT_Valid operating point, then applies the cutoff forward to OOT_Monitor. OOT_Monitor outcomes are not used to choose either cutoff.

| Operating Mode   | Model                  |   PD Cutoff (OOT_Valid) |   Approval Rate % |   Approved Count |   Observed Bad Rate % |   Total Recommended Loan Volume |   Simulated ENP |   Valid Reference Bad Rate % |
|:-----------------|:-----------------------|------------------------:|------------------:|-----------------:|----------------------:|--------------------------------:|----------------:|-----------------------------:|
| Iso-Approval     | Champion Scorecard     |                  0.0308 |           70.7970 |           107404 |                0.9990 |                 5052707765.7302 |  132180253.7712 |                     nan      |
| Iso-Approval     | Challenger LightGBM M4 |                  0.0336 |           67.0846 |           101772 |                0.7026 |                 4672695317.6058 |  123501046.5098 |                     nan      |
| Iso-Risk         | Champion Scorecard     |                  0.0308 |           70.7970 |           107404 |                0.9990 |                 5052707765.7302 |  132180253.7712 |                       1.2623 |
| Iso-Risk         | Challenger LightGBM M4 |                  0.0465 |           75.1587 |           114021 |                0.8832 |                 5095059964.0588 |  140125997.0085 |                       1.2623 |

Parametric ENP assumptions follow Stage 4: COF=4.0%, OPEX=1.5%, assumed LGD=75%, APR 9%/12%/15%/18% by tiers A–D capped at 18%; these are commercial simulation assumptions, not statutory constants.

## Table 4: OOT_Monitor Fair Lending Subgroup Audit

| Sensitive Attribute       | Group   | Model                  |   Sample Count |   Sample Share % |   Approval Rate % |   Reference Highest-Group Approval % |    AIR | AIR < 0.80 Flag   |   Observed Bad Rate % |   Mean Recalibrated PD % |   Subgroup A/E Ratio |
|:--------------------------|:--------|:-----------------------|---------------:|-----------------:|------------------:|-------------------------------------:|-------:|:------------------|----------------------:|-------------------------:|---------------------:|
| Gender                    | F       | Champion Scorecard     |            926 |           0.6104 |           57.6674 |                              59.4737 | 0.9696 | False             |                2.8078 |                   3.6332 |               0.7728 |
| Gender                    | M       | Champion Scorecard     |            190 |           0.1252 |           59.4737 |                              59.4737 | 1.0000 | False             |                0.5263 |                   3.2102 |               0.1640 |
| Gender                    | Unknown | Champion Scorecard     |         150591 |          99.2644 |           70.8920 |                              59.4737 | 1.1920 | False             |                2.1170 |                   2.7294 |               0.7756 |
| Gender                    | F       | Challenger LightGBM M4 |            926 |           0.6104 |           52.1598 |                              52.1598 | 1.0000 | False             |                2.8078 |                   4.3086 |               0.6517 |
| Gender                    | M       | Challenger LightGBM M4 |            190 |           0.1252 |           45.7895 |                              52.1598 | 0.8779 | False             |                0.5263 |                   4.4396 |               0.1185 |
| Gender                    | Unknown | Challenger LightGBM M4 |         150591 |          99.2644 |           67.2032 |                              52.1598 | 1.2884 | False             |                2.1170 |                   3.0424 |               0.6958 |
| Age Band                  | 25-34   | Champion Scorecard     |          28766 |          18.9616 |           60.4081 |                              79.9143 | 0.7559 | True              |                2.9896 |                   3.6231 |               0.8252 |
| Age Band                  | 35-49   | Champion Scorecard     |          47324 |          31.1943 |           64.1958 |                              79.9143 | 0.8033 | False             |                2.1469 |                   3.2143 |               0.6679 |
| Age Band                  | 50+     | Champion Scorecard     |          69273 |          45.6624 |           79.9143 |                              79.9143 | 1.0000 | False             |                1.3757 |                   1.9867 |               0.6925 |
| Age Band                  | <25     | Champion Scorecard     |           6344 |           4.1817 |           67.5914 |                              79.9143 | 0.8458 | False             |                6.0845 |                   3.3146 |               1.8356 |
| Age Band                  | 25-34   | Challenger LightGBM M4 |          28766 |          18.9616 |           56.7684 |                              76.9983 | 0.7373 | True              |                2.9896 |                   3.9851 |               0.7502 |
| Age Band                  | 35-49   | Challenger LightGBM M4 |          47324 |          31.1943 |           61.6558 |                              76.9983 | 0.8007 | False             |                2.1469 |                   3.3647 |               0.6381 |
| Age Band                  | 50+     | Challenger LightGBM M4 |          69273 |          45.6624 |           76.9983 |                              76.9983 | 1.0000 | False             |                1.3757 |                   2.2871 |               0.6015 |
| Age Band                  | <25     | Challenger LightGBM M4 |           6344 |           4.1817 |           46.1066 |                              76.9983 | 0.5988 | True              |                6.0845 |                   4.8381 |               1.2576 |
| Age (Train_Dev quintiles) | Age Q1  | Champion Scorecard     |          20435 |          13.4700 |           62.2461 |                              84.5108 | 0.7365 | True              |                4.2085 |                   3.6286 |               1.1598 |
| Age (Train_Dev quintiles) | Age Q2  | Champion Scorecard     |          24766 |          16.3249 |           61.1524 |                              84.5108 | 0.7236 | True              |                2.4873 |                   3.4747 |               0.7158 |
| Age (Train_Dev quintiles) | Age Q3  | Champion Scorecard     |          27748 |          18.2905 |           64.4371 |                              84.5108 | 0.7625 | True              |                2.1659 |                   3.1877 |               0.6795 |
| Age (Train_Dev quintiles) | Age Q4  | Champion Scorecard     |          29033 |          19.1375 |           67.6334 |                              84.5108 | 0.8003 | False             |                1.6877 |                   2.9102 |               0.5799 |
| Age (Train_Dev quintiles) | Age Q5  | Champion Scorecard     |          49725 |          32.7770 |           84.5108 |                              84.5108 | 1.0000 | False             |                1.3032 |                   1.6459 |               0.7918 |
| Age (Train_Dev quintiles) | Age Q1  | Challenger LightGBM M4 |          20435 |          13.4700 |           52.2290 |                              81.9869 | 0.6370 | True              |                4.2085 |                   4.4313 |               0.9497 |
| Age (Train_Dev quintiles) | Age Q2  | Challenger LightGBM M4 |          24766 |          16.3249 |           58.9801 |                              81.9869 | 0.7194 | True              |                2.4873 |                   3.6925 |               0.6736 |
| Age (Train_Dev quintiles) | Age Q3  | Challenger LightGBM M4 |          27748 |          18.2905 |           61.7486 |                              81.9869 | 0.7532 | True              |                2.1659 |                   3.3320 |               0.6500 |
| Age (Train_Dev quintiles) | Age Q4  | Challenger LightGBM M4 |          29033 |          19.1375 |           64.0306 |                              81.9869 | 0.7810 | True              |                1.6877 |                   3.1126 |               0.5422 |
| Age (Train_Dev quintiles) | Age Q5  | Challenger LightGBM M4 |          49725 |          32.7770 |           81.9869 |                              81.9869 | 1.0000 | False             |                1.3032 |                   1.9741 |               0.6601 |

AIR uses the highest observed subgroup approval rate within each protected attribute and model as reference. The 0.80 four-fifths value is a diagnostic benchmark only, not a legal conclusion. `person_gender_sensitive` is 99.17% missing on Train_Dev; the `Unknown` subgroup is retained, and sparse observed gender groups should be interpreted with their sample counts. Subgroup A/E uses the full observed subgroup, not only approved cases. Sensitive attributes were not included in either model.

## Rank, drift, and portfolio interpretation

Weekly/monthly `Loan Volume`, `Actual Bad Rate`, `Mean Predicted PD`, `A/E Ratio`, `KS`, and `Gini` cover Weeks 0–91 in the saved temporal CSV. Score/feature drift is diagnosed alongside discrimination metrics so calibration shift can be distinguished from rank-order deterioration. Feature CSI includes all 19 Champion scorecard features, top LightGBM gain features, and flagged `has_tax_registry_record` / `tax_verified_income_ratio`.

## Audit Summary and Artifacts

Full predictions contain 1,526,659 rows, weeks 0–91; `OOT_Monitor` rows=151,707; monitoring split count and dates are guarded by code. `recalibrators_stage5.json` records fit weeks 65–77 and `oot_monitor_used_for_fit=false`. `has_tax_registry_record` and `tax_verified_income_ratio` remain Stage 5 governance features and their PSI/CSI are included above.

Saved `data/processed/stage5_full_predictions.parquet`, `data/processed/models/recalibrators_stage5.json`, `data/processed/stage5_temporal_tracking.csv`, `data/processed/stage5_psi_csi_summary.csv`, and figures in `reports/figures/`.

**HARD STOP:** Stage 5 complete. Wait for human review before Stage 6 or Stage 7.
