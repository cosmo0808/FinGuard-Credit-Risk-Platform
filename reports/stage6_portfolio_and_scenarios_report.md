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
| Challenger LightGBM M4 | Iso-Risk     | ALL                 | ALL           | ALL           | ALL          | Requested Amount |                        nan |            62017 |              57.7474 |            2942266930.6055 |                  0.6950 |         25042305.9022 |             nan |               nan |               nan |              nan |
| Challenger LightGBM M4 | Iso-Approval | ALL                 | ALL           | ALL           | ALL          | Requested Amount |                        nan |            58745 |              60.4853 |            2826295201.5942 |                  0.6111 |         21726101.5716 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Risk     | ALL                 | ALL           | ALL           | ALL          | Requested Amount |                        nan |            56072 |              55.6911 |            2813906043.4976 |                  0.6509 |         23991406.8945 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Approval | ALL                 | ALL           | ALL           | ALL          | Requested Amount |                        nan |            56072 |              55.6911 |            2813906043.4976 |                  0.6509 |         23991406.8945 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Risk     | ALL                 | ALL           | ALL           | 13-24m       | ALL              |                        nan |            33080 |              47.4408 |            2397043617.5298 |                  1.6354 |         21886130.5376 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Approval | ALL                 | ALL           | ALL           | 13-24m       | ALL              |                        nan |            33080 |              47.4408 |            2397043617.5298 |                  1.6354 |         21886130.5376 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Approval | Tier A: Super Prime | ALL           | ALL           | ALL          | ALL              |                        nan |            44585 |              46.4117 |            2345047967.0639 |                  0.4553 |         11393209.6412 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Risk     | Tier A: Super Prime | ALL           | ALL           | ALL          | ALL              |                        nan |            44585 |              46.4117 |            2345047967.0639 |                  0.4553 |         11393209.6412 |             nan |               nan |               nan |              nan |
| Challenger LightGBM M4 | Iso-Risk     | ALL                 | ALL           | ALL           | 13-24m       | ALL              |                        nan |            34324 |              45.7427 |            2330617945.3895 |                  1.3518 |         26711079.5836 |             nan |               nan |               nan |              nan |
| Challenger LightGBM M4 | Iso-Approval | Tier A: Super Prime | ALL           | ALL           | ALL          | ALL              |                        nan |            48195 |              48.9381 |            2286727380.4863 |                  0.2739 |          9424451.2159 |             nan |               nan |               nan |              nan |
| Challenger LightGBM M4 | Iso-Risk     | Tier A: Super Prime | ALL           | ALL           | ALL          | ALL              |                        nan |            48195 |              44.8813 |            2286727380.4863 |                  0.2739 |          9424451.2159 |             nan |               nan |               nan |              nan |
| Champion Scorecard     | Iso-Risk     | ALL                 | ALL           | 100k+         | ALL          | ALL              |                        nan |            20678 |              42.9388 |            2169570487.8531 |                  1.6201 |         18975463.6544 |             nan |               nan |               nan |              nan |

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
| S1_Baseline          | Champion Scorecard     |                     75.0006 |                70.7970 |           107404 |                  0.9990 |    4.0000 |   75.0000 |      5052707765.7302 |                     136584548.2081 |                      132180253.7712 |
| S2_Controlled_Growth | Champion Scorecard     |                     80.0006 |                76.7262 |           116399 |                  1.1598 |    4.0000 |   75.0000 |      5395482260.8872 |                     148751838.4629 |                      144401421.7066 |
| S3_Mild_Downturn     | Champion Scorecard     |                     75.0006 |                70.7970 |           107404 |                  0.9990 |    5.0000 |   80.0000 |      5052707765.7302 |                     101885977.4853 |                       84799142.0269 |
| S4_Severe_Tail_Shock | Champion Scorecard     |                     75.0006 |                70.7970 |           107404 |                  0.9990 |    5.5000 |   85.0000 |      5052707765.7302 |                      83134886.7645 |                       50999674.1452 |
| S1_Baseline          | Challenger LightGBM M4 |                     80.5634 |                75.1587 |           114021 |                  0.8832 |    4.0000 |   75.0000 |      5095059964.0588 |                     161667461.5621 |                      140125997.0085 |
| S2_Controlled_Growth | Challenger LightGBM M4 |                     85.5634 |                81.9850 |           124377 |                  1.1465 |    4.0000 |   75.0000 |      5396021266.5699 |                     175998426.9711 |                      151823511.2885 |
| S3_Mild_Downturn     | Challenger LightGBM M4 |                     80.5634 |                75.1587 |           114021 |                  0.8832 |    5.0000 |   80.0000 |      5095059964.0588 |                     127397582.5585 |                       89687081.3507 |
| S4_Severe_Tail_Shock | Challenger LightGBM M4 |                     80.5634 |                75.1587 |           114021 |                  0.8832 |    5.5000 |   85.0000 |      5095059964.0588 |                     109139074.4980 |                       42484976.8677 |

`S4_Emergency_Cutoff` is a separate Valid-selected counterfactual. It selects the cutoff maximizing positive expected severe-stress profit on `OOT_Valid`, then forward-tests it on Monitor. If that forward empirical proxy remains non-positive, no claim of restored Monitor margin is made; the cutoff remains an unvalidated sensitivity result.

| Model                  |   Emergency_Valid_Cutoff |   Monitor_Approval_Pct |   Realized_Net_Profit_Stress_Proxy |   Model_Implied_Expected_Net_Profit |
|:-----------------------|-------------------------:|-----------------------:|-----------------------------------:|------------------------------------:|
| Champion Scorecard     |                   0.0308 |                70.7535 |                      83024339.0170 |                       50989281.6689 |
| Challenger LightGBM M4 |                   0.0307 |                64.8658 |                      88402285.3278 |                       45953259.6385 |

## Power BI Data Mart

| File | Rows | Size (MB) |
|---|---:|---:|
| File                                  |   Rows |   Size_MB |
|:--------------------------------------|-------:|----------:|
| pbi_01_temporal_cohort_monitoring.csv |   1392 |  0.284218 |
| pbi_02_cutoff_tradeoff_curve.csv      |    400 |  0.055147 |
| pbi_03_portfolio_concentration.csv    |   4413 |  1.81524  |
| pbi_04_stress_scenarios.csv           |     50 |  0.010238 |
| pbi_05_psi_csi_and_fairness.csv       |    122 |  0.018049 |
| pbi_06_delinquency_migration.csv      |     60 |  0.00981  |

Total CSV size: **2.19 MB** (<15 MB). Star-schema wiring and DAX are documented in `reports/powerbi_dashboard_blueprint.md`.

## Stage 6 Outputs

Six aggregate CSV tables under `data/powerbi/`; figures are `stage6_cohort_vintage_trend.png`, `stage6_stress_scenarios.png`, and `stage6_exposure_concentration.png`. Detailed cutoff metrics, PSI/CSI, decile calibration and fairness rows are in the Power BI tables.

**HARD STOP:** Stage 6 complete. Wait for human review before Stage 7.
