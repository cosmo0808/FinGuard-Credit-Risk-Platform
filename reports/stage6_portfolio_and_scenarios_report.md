# Stage 6 — Portfolio Risk Intelligence、Stress Scenarios 与 Power BI Mart

## 数据完整性与历史逾期字段核验

`train_applprev_2.parquet` 含 14,075,487 条记录、字段 `case_id, cacccardblochreas_147M, conts_type_509L, credacc_cards_status_52L, num_group1, num_group2`。其中只有 card-block reason、contact type、credit-account card status 和层级索引；没有 consecutive monthly DPD status，也没有可用于构造 MOB 的月度 performance snapshot。相关定义如下：

| Variable                 | Description                                                                              |
|:-------------------------|:-----------------------------------------------------------------------------------------|
| actualdpd_943P           | Days Past Due (DPD) of previous contract (actual).                                       |
| cacccardblochreas_147M   | Card blocking reason.                                                                    |
| credacc_cards_status_52L | Card status of the previous credit account.                                              |
| dpdmax_139P              | Maximal days past due for active contract.                                               |
| dpdmax_757P              | Maximum days past due for a closed contract.                                             |
| dpdmaxdatemonth_442T     | Max DPD occurrence month for terminated contracts from credit bureau data.               |
| dpdmaxdatemonth_89T      | Month when maximum days past due occurred on the active contract with the credit bureau. |
| dpdmaxdateyear_596T      | Year when maximum Days Past Due (DPD) occurred for the active contract.                  |
| dpdmaxdateyear_896T      | Year of maximum Days Past Due of closed contract obtained from Credit Bureau.            |
| maxdpdtolerance_577P     | Maximum DPD with tolerance (on previous application/s).                                  |

`train_applprev_1_*.parquet` 含 6,525,979 条 previous-application records；`actualdpd_943P` 的定义为 “Days Past Due (DPD) of previous contract (actual).”，可用来构造 case-level prior-contract maximum DPD severity。`credit_bureau_a_1_*` 字段包括 `dpdmax_139P, dpdmax_757P, dpdmaxdatemonth_442T, dpdmaxdatemonth_89T, dpdmaxdateyear_596T, dpdmaxdateyear_896T`，是 DPD maxima / date tags 而不是连续月度状态快照。因此本报告采用 **Prior-Delinquency Severity -> Current Loan Default Migration Table**，没有伪造 roll-rate matrix 或 MOB curves。每位申请人先在 DuckDB 内聚合 previous contracts 的 maximum observed actual DPD，再按 prior-application history 分层；无历史记录及有 prior history 但 DPD missing 分开标示，避免误列为 0 DPD。

## 决策定义与数据边界

Risk tiers 用 Stage 5 `pd_*_recal_intercept`，阈值沿用 A `<1%`, B `<2%`, C `<4%`, D `<7%`, E `>=7%`。Operational approval cutoffs、growth cutoff 与 severe-stress emergency cutoff 均仅由 `OOT_Valid` 确定，随后 forward-apply 至 `OOT_Monitor`。Monitor `target` 只用于事后度量，不参与 cutoffs、stress emergency selection 或分箱拟合。Exposure 为 Stage 5/4 deterministic `min(requested, risk, affordability)` cap，经 hard policy decline 后的 approved recommended limit。

Profit 是参数化生命周期净现金流近似：利息收入扣 `COF`/`OPEX` 后减 `principal * loss rate * LGD`。S1/S2 的 realized estimate 使用 Monitor observed `target`；S3/S4 的 `Realized_Net_Profit_Stress_Proxy` 是 **observed target rate multiplied by stated scenario stress factor** 的情景损失代理，不是已观察到的真实 downturn P&L。Stress assumptions are not empirical forecasts.

## Origination Cohort & Portfolio Concentration

Temporal data contains weekly and monthly aggregates, model, split and intercept-calibrated `Risk_Tier`; eligible applications are approved under the corresponding Valid-frozen Iso-Risk cutoff for exposure reporting. The OOT Monitor months are cohort-level point-in-time outcomes, not MOB curves.

| Model                  | Mode         | Risk_Tier           | Income_Type   | Ticket_Band   | Tenor_Band   | Binding_Driver   |   Binding_Driver_Share_Pct |   Approved_Count |   Exposure_Share_Pct |   Total_Recommended_Volume |   Observed_Bad_Rate_Pct |   Expected_Loss_Proxy |   Risk_Tier_HHI |   Income_Type_HHI |   Ticket_Band_HHI |   Tenor_Band_HHI |
|:-----------------------|:-------------|:--------------------|:--------------|:--------------|:-------------|:-----------------|---------------------------:|-----------------:|---------------------:|---------------------------:|------------------------:|----------------------:|----------------:|------------------:|------------------:|-----------------:|
| Challenger LightGBM M4 | Iso-Risk     | ALL                 | ALL           | ALL           | ALL          | Requested Amount |                        nan |            61750 |              57.4292 |            2915846229.9688 |                  0.6947 |         26280432.5614 |             nan |               nan |               nan |              nan |
| Challenger LightGBM M4 | Iso-Approval | ALL                 | ALL           | ALL           | ALL          | Requested Amount |                        nan |            58504 |              60.1627 |            2801457252.9556 |                  0.6119 |         22987872.6190 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Approval | ALL                 | ALL           | ALL           | ALL          | Requested Amount |                        nan |            55949 |              55.5233 |            2800543578.6812 |                  0.6506 |         24672434.6908 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Risk     | ALL                 | ALL           | ALL           | ALL          | Requested Amount |                        nan |            55949 |              55.5233 |            2800543578.6812 |                  0.6506 |         24672434.6908 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Risk     | ALL                 | ALL           | ALL           | 13-24m       | ALL              |                        nan |            33080 |              47.4124 |            2391438906.9417 |                  1.6354 |         22530444.4892 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Approval | ALL                 | ALL           | ALL           | 13-24m       | ALL              |                        nan |            33080 |              47.4124 |            2391438906.9417 |                  1.6354 |         22530444.4892 |             nan |               nan |               nan |              nan |
| Challenger LightGBM M4 | Iso-Risk     | ALL                 | ALL           | ALL           | 13-24m       | ALL              |                        nan |            34324 |              45.6857 |            2319594654.1878 |                  1.3518 |         27829747.7496 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Risk     | Tier A: Super Prime | ALL           | ALL           | ALL          | ALL              |                        nan |            42162 |              43.9796 |            2218290647.9677 |                  0.4293 |         11082402.6254 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Approval | Tier A: Super Prime | ALL           | ALL           | ALL          | ALL              |                        nan |            42162 |              43.9796 |            2218290647.9677 |                  0.4293 |         11082402.6254 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Risk     | ALL                 | ALL           | 100k+         | ALL          | ALL              |                        nan |            20678 |              42.8952 |            2163596112.0618 |                  1.6201 |         19549438.2282 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Approval | ALL                 | ALL           | 100k+         | ALL          | ALL              |                        nan |            20678 |              42.8952 |            2163596112.0618 |                  1.6201 |         19549438.2282 |             nan |               nan |               nan |              nan |
| Challenger LightGBM M4 | Iso-Risk     | ALL                 | ALL           | ALL           | 7-12m        | ALL              |                        nan |            57966 |              42.0020 |            2132562966.5816 |                  0.6935 |         22938019.2249 |             nan |               nan |               nan |              nan |

## Audit Summary

### Prediction integrity by split

| split_name   |    Rows |   Target_Null_Pct |   Raw_PD_Null_Pct |
|:-------------|--------:|------------------:|------------------:|
| OOT_Monitor  |  151707 |            0.0000 |            0.0000 |
| OOT_Valid    |   82602 |            0.0000 |            0.0000 |
| Train_Dev    | 1292350 |            0.0000 |            0.0000 |

### Source and model diagnostics

| Dataset / scope | Rows | Null-rate / integrity check | Feature-quality check | Performance / stability evidence |
|---|---:|---|---|---|
| `train_applprev_1_*` historical rows | 6,525,979 | `actualdpd_943P` null = 0.04% | Historical observed field; no Stage 6 IV/re-binning | Severity migration in `pbi_06` |
| `train_applprev_2` depth-2 rows | 14,075,487 | `credacc_cards_status_52L` null = 97.57% | Card status/block flags only; no monthly DPD | No MOB or roll-rate claim |
| Champion Scorecard, OOT_Monitor | 151,707 | Stage 5 recalibrated PDs non-null; labels observed | No Stage 6 feature refit; Stage 2/3 IV and monotonicity gates retained | ROC-AUC 0.7751; KS 0.4222; Gini 0.5502; Score PSI 0.0049 |
| Raw LightGBM M4, OOT_Monitor | 151,707 | Stage 5 recalibrated PDs non-null; labels observed | No Stage 6 feature refit; Stage 2/3 gates retained | ROC-AUC 0.8257; KS 0.5006; Gini 0.6514; raw PD PSI 0.0772 |
| Drift diagnostics | 26 feature CSIs + score PSIs | All records retained in Power BI table 05 | No new binning learned on OOT | Highest OOT_Monitor CSI: `has_tax_registry_record` 1.3064; `person_education_level` 1.2974 |

`IV` and monotonicity are not re-estimated in Stage 6 because no model/features are fitted here; the saved Stage 2/3 screening results remain the governing evidence. `pbi_05_psi_csi_and_fairness.csv` contains all 26 feature CSIs, score PSIs, decile calibration rows and fairness rows.

## Empirical Prior-Delinquency Migration

| Prior_Delinquency_Severity      |   Applications |   Current_Defaults |   Current_Default_Rate_Pct |
|:--------------------------------|---------------:|-------------------:|---------------------------:|
| 0 DPD                           |        2435260 |         81624.0000 |                     3.3518 |
| 1-10 DPD                        |           6666 |           714.0000 |                    10.7111 |
| 11-30 DPD                       |            630 |           138.0000 |                    21.9048 |
| 31-89 DPD                       |            182 |            36.0000 |                    19.7802 |
| 90+ DPD                         |            272 |            38.0000 |                    13.9706 |
| No prior application            |         610274 |         13434.0000 |                     2.2013 |
| Prior contract: DPD unavailable |             34 |             4.0000 |                    11.7647 |

Tier-level detailed results are in `pbi_06_delinquency_migration.csv`. It describes a cross-sectional migration from prior-contract severity into the current application target, not transitions between observed repayment months.

## Parametric Commercial Stress Simulations

| Scenario             | Model                  |   Valid_Target_Approval_Pct |   Monitor_Approval_Pct |   Approved_Count |   Observed_Bad_Rate_Pct |   COF_Pct |   LGD_Pct |   Recommended_Volume |   Realized_Net_Profit_Stress_Proxy |   Model_Implied_Expected_Net_Profit |
|:---------------------|:-----------------------|----------------------------:|-----------------------:|-----------------:|------------------------:|----------:|----------:|---------------------:|-----------------------------------:|------------------------------------:|
| S1_Baseline          | Champion Scorecard     |                     75.0006 |                70.7970 |           107404 |                  0.9990 |    4.0000 |   75.0000 |      5043906826.6780 |                     139546517.0483 |                      133659751.9881 |
| S2_Controlled_Growth | Champion Scorecard     |                     80.0006 |                76.7262 |           116399 |                  1.1598 |    4.0000 |   75.0000 |      5386681321.8350 |                     151713807.3031 |                      145869783.8316 |
| S3_Mild_Downturn     | Champion Scorecard     |                     75.0006 |                70.7970 |           107404 |                  0.9990 |    5.0000 |   80.0000 |      5043906826.6780 |                     104911595.9977 |                       85871116.0105 |
| S4_Severe_Tail_Shock | Champion Scorecard     |                     75.0006 |                70.7970 |           107404 |                  0.9990 |    5.5000 |   85.0000 |      5043906826.6780 |                      86195202.9840 |                       51244908.9358 |
| S1_Baseline          | Challenger LightGBM M4 |                     80.5634 |                75.1587 |           114021 |                  0.8832 |    4.0000 |   75.0000 |      5077289608.1822 |                     166435935.5090 |                      142305881.3731 |
| S2_Controlled_Growth | Challenger LightGBM M4 |                     85.5634 |                81.9850 |           124377 |                  1.1465 |    4.0000 |   75.0000 |      5378250910.6933 |                     180766900.9180 |                      154179975.6757 |
| S3_Mild_Downturn     | Challenger LightGBM M4 |                     80.5634 |                75.1587 |           114021 |                  0.8832 |    5.0000 |   80.0000 |      5077289608.1822 |                     132288027.5850 |                       91152183.5963 |
| S4_Severe_Tail_Shock | Challenger LightGBM M4 |                     80.5634 |                75.1587 |           114021 |                  0.8832 |    5.5000 |   85.0000 |      5077289608.1822 |                     114093768.8137 |                       42167086.1971 |

`S4_Emergency_Cutoff` is a separate Valid-selected counterfactual. It selects the cutoff maximizing positive expected severe-stress profit on `OOT_Valid`, then forward-tests it on Monitor. If that forward empirical proxy remains non-positive, no claim of restored Monitor margin is made; the cutoff remains an unvalidated sensitivity result.

| Model                  |   Emergency_Valid_Cutoff |   Monitor_Approval_Pct |   Realized_Net_Profit_Stress_Proxy |   Model_Implied_Expected_Net_Profit |
|:-----------------------|-------------------------:|-----------------------:|-----------------------------------:|------------------------------------:|
| Champion Scorecard     |                   0.0309 |                70.7535 |                      86084655.2365 |                       51235231.0322 |
| Challenger LightGBM M4 |                   0.0297 |                63.3781 |                      90331061.6185 |                       45755367.0799 |

## Power BI Data Mart

| File | Rows | Size (MB) |
|---|---:|---:|
| File                                  |   Rows |   Size_MB |
|:--------------------------------------|-------:|----------:|
| pbi_01_temporal_cohort_monitoring.csv |   1392 |  0.284243 |
| pbi_02_cutoff_tradeoff_curve.csv      |    400 |  0.055096 |
| pbi_03_portfolio_concentration.csv    |   4412 |  1.81987  |
| pbi_04_stress_scenarios.csv           |     50 |  0.010198 |
| pbi_05_psi_csi_and_fairness.csv       |    122 |  0.01806  |
| pbi_06_delinquency_migration.csv      |     60 |  0.009806 |

Total CSV size: **2.20 MB** (<15 MB). Star-schema wiring and DAX are documented in `reports/powerbi_dashboard_blueprint.md`.

## Stage 6 Outputs

Six aggregate CSV tables under `data/powerbi/`; figures are `stage6_cohort_vintage_trend.png`, `stage6_stress_scenarios.png`, and `stage6_exposure_concentration.png`. Detailed cutoff metrics, PSI/CSI, decile calibration and fairness rows are in the Power BI tables.

**HARD STOP:** Stage 6 complete. Wait for human review before Stage 7.
