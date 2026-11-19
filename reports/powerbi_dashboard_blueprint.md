# Power BI 3-Page Risk Command Center — 数据模型与 DAX

## 导入与关系

导入 `pbi_01_temporal_cohort_monitoring.csv` 至 `pbi_06_delinquency_migration.csv`。将 `pbi_01[Period]` 作为日期/期别轴字段（按 `Period_Order` 排序）；不要把不同 Grain 的 Weekly / Monthly 同时加总。事实表均为 stage-level aggregate，不能再次求平均代替加权比率。

建议建立 `DimModel = DISTINCT(pbi_01[Model])`、`DimSplit = DISTINCT(pbi_01[Split])`、`DimTier = DISTINCT(pbi_01[Risk_Tier])`，用单向一对多关系连到 `pbi_01`。`pbi_02` 通过 `Model` 与 `DimModel` 相连，使用 `Target_Eligible_Approval_Pct` 做 what-if/数值切片器；情景、集中度、PSI/CSI 与逾期迁移表可通过各自维度表连接。避免多事实表直接互连造成双向过滤。

## Page 1 — Portfolio Pulse（组合脉搏）

展示 monthly actual bad rate 与 recalibrated PD、weekly applications/approvals、KS/Gini、Split、Risk_Tier 筛选器。KPI 用加权度量：

```DAX
Applications = SUM('pbi_01_temporal_cohort_monitoring'[Applications])
Approvals = SUM('pbi_01_temporal_cohort_monitoring'[Approvals])
Approval Rate % = DIVIDE([Approvals], [Applications])
Bads = SUM('pbi_01_temporal_cohort_monitoring'[Bads])
Actual Bad Rate % = DIVIDE([Bads], [Applications])
Recommended Volume = SUM('pbi_01_temporal_cohort_monitoring'[Recommended_Volume])
Weighted Recalibrated PD % = DIVIDE(
    SUMX('pbi_01_temporal_cohort_monitoring', 'pbi_01_temporal_cohort_monitoring'[Applications] * 'pbi_01_temporal_cohort_monitoring'[Recalibrated_PD_Pct]),
    [Applications])
```

KS/Gini in `pbi_01` are already computed per grain/period/split/model/tier; use them at that exact grain and do not average across risk tiers without weights.

## Page 2 — Strategy & Stress Lab（策略与压力测试）

Use `pbi_02_cutoff_tradeoff_curve` with slicers for `Model`, `Split`, and `Target_Eligible_Approval_Pct`. Plot `Approval_Rate_Pct`, `Observed_Bad_Rate_Pct`, `Recommended_Volume` and `Mean_PD_Pct` against the valid-frozen PD cutoff. Show `pbi_04_stress_scenarios` as clustered bars of `Realized_Net_Profit_Stress_Proxy` and `Model_Implied_Expected_Net_Profit`; keep scenario, model, and tier filters visible.

```DAX
Expected Stress Profit = SUM('pbi_04_stress_scenarios'[Model_Implied_Expected_Net_Profit])
Observed-Target Stress Proxy = SUM('pbi_04_stress_scenarios'[Realized_Net_Profit_Stress_Proxy])
Stress Margin Positive = IF([Expected Stress Profit] > 0, "Positive", "Non-positive")
```

## Page 3 — Exposure, Stability & Fairness（敞口、稳定性与公平性）

Use concentration heatmap by `Risk_Tier`, `Income_Type`, `Ticket_Band`, `Tenor_Band`, `Binding_Driver`; show `Exposure_Share_Pct`, `Expected_Loss_Proxy`, and HHI. Add PSI/CSI ranking, decile calibration, `AIR` and `A_E_Ratio` from `pbi_05`. Use `pbi_06` for prior DPD severity versus current default and label it cross-sectional. Do not title it vintage roll-rate or MOB.

```DAX
Exposure = SUM('pbi_03_portfolio_concentration'[Total_Recommended_Volume])
Expected Loss Proxy = SUM('pbi_03_portfolio_concentration'[Expected_Loss_Proxy])
Exposure Share % = DIVIDE([Exposure], CALCULATE([Exposure], ALL('pbi_03_portfolio_concentration'[Risk_Tier], 'pbi_03_portfolio_concentration'[Income_Type], 'pbi_03_portfolio_concentration'[Ticket_Band], 'pbi_03_portfolio_concentration'[Tenor_Band], 'pbi_03_portfolio_concentration'[Binding_Driver])))
Fairness AIR = MIN('pbi_05_psi_csi_and_fairness'[AIR])
Current Default Rate % = DIVIDE(SUM('pbi_06_delinquency_migration'[Current_Defaults]), SUM('pbi_06_delinquency_migration'[Application_Count]))
```

`0.10 / 0.25` drift thresholds and `AIR < 0.80` are monitoring diagnostics used by this analysis, not statutory or legal conclusions. `Realized_Net_Profit_Stress_Proxy` under S3/S4 is transformed observed-target sensitivity, not actual future downturn performance.
