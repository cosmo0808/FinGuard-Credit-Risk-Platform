# Stage 3 模型评估报告

本阶段仅使用 `Train_Dev` 训练；LightGBM 使用 W0–54 拟合、W55–64 early stopping 与 isotonic 校准。`OOT_Valid` 仅在模型和校准器冻结后用于推理与盲评估；`OOT_Monitor` 未读取。训练特征矩阵以 float32 保存，DuckDB 内存限制 6GB、4 threads。

## A-Card 事前纯度与 Stage 2 代理特征复核

`disbursedcredamount_1113A` 已从 scorecard 和 challenger 排除，因为它属于放款后结果字段。Stage 2 中 `has_tax_registry_record` 与 `has_debitcard` 的经验原因：

- `has_tax_registry_record`: Gate 2 IV=0.017311 (<0.02)
- `has_debitcard`: Gate 2 IV=0.005740 (<0.02)

Train_Dev Gate 1 与内部 PSI 检查（W0–44 对 W45–64；仅训练期数据）：

| Feature                   |   Null % |   Dominant % |   Internal PSI | Gate 1   | PSI < 0.10   | M4 Candidate   | PSI Override                     |
|:--------------------------|---------:|-------------:|---------------:|:---------|:-------------|:---------------|:---------------------------------|
| has_tax_registry_record   |   0.0000 |      65.0361 |         1.1490 | Pass     | Fail         | Yes            | Explicit Group 4 proxy inclusion |
| has_debitcard             |   0.0000 |      92.8923 |         0.0002 | Pass     | Pass         | Yes            | —                                |
| tax_verified_income_ratio |  33.7394 |      65.6975 |         0.3119 | Pass     | Fail         | Yes            | Explicit Group 4 proxy inclusion |
| app_cnt_last_30d          |   0.0000 |      89.7857 |         0.0000 | Pass     | Pass         | Yes            | —                                |

按明确建模要求纳入四个 Group 4 proxy 作额外 challenger lift 测试。若 `PSI Override` 非空，该变量虽通过 Gate 1 但未通过内部 PSI < 0.10；其 Stage 3 结果仅作敏感性/探索性比较。

## Table 1: Champion Scorecard 回归与计分卡点数

| Feature                       |    coef |   std_err |        z |   p_value |    VIF |
|:------------------------------|--------:|----------:|---------:|----------:|-------:|
| static_avg_dpd_tol_closure24m | -0.2410 |    0.0155 | -15.5762 |    0.0000 | 2.9568 |
| bureau_active_max_dpd         | -0.5950 |    0.0097 | -61.1666 |    0.0000 | 1.1210 |
| static_avg_dbd_dpd_last24m    | -0.0675 |    0.0139 |  -4.8613 |    0.0000 | 2.0793 |
| static_days_overdue_tol       | -0.2224 |    0.0202 | -11.0337 |    0.0000 | 3.0331 |
| prev_appl_max_dpd_tolerance   | -0.3043 |    0.0147 | -20.7548 |    0.0000 | 1.9342 |
| static_pct_inst_paid_late_10d | -0.1404 |    0.0175 |  -8.0024 |    0.0000 | 2.3655 |
| static_avg_max_dpd_last9m     | -0.2632 |    0.0154 | -17.1072 |    0.0000 | 1.5797 |
| static_last_rejected_amount   | -0.6235 |    0.0362 | -17.2409 |    0.0000 | 1.4084 |
| static_num_inst_paid_early    | -0.6196 |    0.0188 | -33.0409 |    0.0000 | 1.6647 |
| static_avg_dbd_dpd_last3m     | -0.2277 |    0.0171 | -13.2932 |    0.0000 | 1.3901 |
| bureau_queries_120d           | -0.4594 |    0.0185 | -24.8476 |    0.0000 | 1.7030 |
| prev_appl_avg_tenor           | -0.7155 |    0.0193 | -37.0180 |    0.0000 | 1.5044 |
| bureau_queries_30d            | -0.2604 |    0.0217 | -11.9759 |    0.0000 | 1.6935 |
| person_income_type            | -0.9476 |    0.0223 | -42.4131 |    0.0000 | 1.0964 |
| person_employment_years       | -1.0174 |    0.0247 | -41.1662 |    0.0000 | 1.0679 |
| app_cnt_last_30d              | -0.3557 |    0.0253 | -14.0409 |    0.0000 | 1.1080 |
| person_education_level        | -1.0769 |    0.0283 | -38.0883 |    0.0000 | 1.0098 |
| static_cnt_pmts_months_24m    | -0.6364 |    0.0399 | -15.9650 |    0.0000 | 1.3302 |
| person_family_status          | -0.8419 |    0.0345 | -24.3986 |    0.0000 | 1.0550 |

模型仅保留满足 `coef < 0`、`p_value < 0.05`、`VIF < 5` 的变量；保留数 19.

PDO scaling: Base Score=600 at bad/good odds=1/20; PDO=50. 最大 score→PD 重建误差=2.776e-16.

Sample points table:

| Feature                       | Group                       | Bin Range      |   Train Count % |   Train Bad Rate % |     WOE |    Coef |   Bin Points |
|:------------------------------|:----------------------------|:---------------|----------------:|-------------------:|--------:|--------:|-------------:|
| static_avg_dpd_tol_closure24m | Group_2_Repayment_Behaviour | (-inf, 0.50)   |         49.1634 |             2.0920 |  0.4730 | -0.2410 |       8.2217 |
| static_avg_dpd_tol_closure24m | Group_2_Repayment_Behaviour | [0.50, 1.50)   |          8.1696 |             4.4241 | -0.3000 | -0.2410 |      -5.2153 |
| static_avg_dpd_tol_closure24m | Group_2_Repayment_Behaviour | [1.50, 5.50)   |          5.3946 |             7.2020 | -0.8168 | -0.2410 |     -14.1979 |
| static_avg_dpd_tol_closure24m | Group_2_Repayment_Behaviour | [5.50, inf)    |          5.4199 |            10.5676 | -1.2372 | -0.2410 |     -21.5049 |
| static_avg_dpd_tol_closure24m | Group_2_Repayment_Behaviour | Special        |         31.8525 |             3.0269 |  0.0000 | -0.2410 |       0.0000 |
| static_avg_dpd_tol_closure24m | Group_2_Repayment_Behaviour | Missing        |          0.0000 |           nan      |  0.0940 | -0.2410 |       1.6341 |
| bureau_active_max_dpd         | Group_1_Traditional_Credit  | (-inf, 0.50)   |         73.3357 |             2.4589 |  0.3077 | -0.5950 |      13.2065 |
| bureau_active_max_dpd         | Group_1_Traditional_Credit  | [0.50, 3.50)   |          8.3530 |             3.4627 | -0.0450 | -0.5950 |      -1.9321 |
| bureau_active_max_dpd         | Group_1_Traditional_Credit  | [3.50, 10.50)  |          5.8251 |             4.6081 | -0.3427 | -0.5950 |     -14.7096 |
| bureau_active_max_dpd         | Group_1_Traditional_Credit  | [10.50, 27.50) |          6.8946 |             5.5789 | -0.5441 | -0.5950 |     -23.3550 |
| bureau_active_max_dpd         | Group_1_Traditional_Credit  | [27.50, inf)   |          5.5915 |            10.1907 | -1.1967 | -0.5950 |     -51.3639 |
| bureau_active_max_dpd         | Group_1_Traditional_Credit  | Special        |          0.0000 |           nan      |  0.0000 | -0.5950 |       0.0000 |

## Table 2: Champion 与 Challenger 指标

| Split                      | Model                  |   ROC-AUC |   PR-AUC |     KS |   Gini |   Brier |   Mean Predicted PD |   Actual Bad Rate |   A/E Ratio |
|:---------------------------|:-----------------------|----------:|---------:|-------:|-------:|--------:|--------------------:|------------------:|------------:|
| Train_Dev (fit diagnostic) | Champion Scorecard     |    0.7612 |   0.1042 | 0.3981 | 0.5223 |  0.0309 |              0.0332 |            0.0332 |      1.0000 |
| Train_Dev (fit diagnostic) | Raw LightGBM M4        |    0.8370 |   0.2136 | 0.5131 | 0.6740 |  0.0289 |              0.0322 |            0.0332 |      1.0305 |
| Train_Dev (fit diagnostic) | Calibrated LightGBM M4 |    0.8369 |   0.2057 | 0.5128 | 0.6738 |  0.0290 |              0.0393 |            0.0332 |      0.8438 |
| Dev_Calib                  | Raw LightGBM M4        |    0.8043 |   0.1855 | 0.4604 | 0.6085 |  0.0409 |              0.0386 |            0.0464 |      1.1997 |
| Dev_Calib                  | Calibrated LightGBM M4 |    0.8051 |   0.1825 | 0.4604 | 0.6102 |  0.0407 |              0.0464 |            0.0464 |      1.0000 |
| OOT_Valid (blind)          | Champion Scorecard     |    0.7768 |   0.0865 | 0.4346 | 0.5536 |  0.0222 |              0.0301 |            0.0234 |      0.7782 |
| OOT_Valid (blind)          | Raw LightGBM M4        |    0.8173 |   0.1290 | 0.4929 | 0.6345 |  0.0216 |              0.0302 |            0.0234 |      0.7749 |
| OOT_Valid (blind)          | Calibrated LightGBM M4 |    0.8171 |   0.1238 | 0.4880 | 0.6342 |  0.0219 |              0.0369 |            0.0234 |      0.6336 |

说明：`Train_Dev (fit diagnostic)` 是同一训练样本的拟合诊断，不代表泛化指标。LightGBM calibration 指标另列 Dev_Calib；OOT_Valid 在整个训练、early stopping 和 calibration 过程中保持盲态。

## Table 3: OOT_Valid 十分位 Lift 与 Capture

| Model               | Decile                  |   Count |   Bad Count |   Marginal Bad Rate % |   Cum Bad Capture % |      KS |   Lift |
|:--------------------|:------------------------|--------:|------------:|----------------------:|--------------------:|--------:|-------:|
| Champion Scorecard  | Decile 1 [Highest Risk] |    8261 |         743 |                8.9941 |             38.4377 | 28.4367 | 3.8434 |
| Champion Scorecard  | Decile 2 [Risk]         |    8260 |         351 |                4.2494 |             56.5960 | 36.5952 | 1.8159 |
| Champion Scorecard  | Decile 3 [Risk]         |    8260 |         288 |                3.4867 |             71.4951 | 41.4946 | 1.4899 |
| Champion Scorecard  | Decile 4 [Risk]         |    8260 |         174 |                2.1065 |             80.4966 | 40.4964 | 0.9002 |
| Champion Scorecard  | Decile 5 [Risk]         |    8260 |         127 |                1.5375 |             87.0667 | 37.0667 | 0.6570 |
| Champion Scorecard  | Decile 6 [Risk]         |    8260 |          81 |                0.9806 |             91.2571 | 31.2574 | 0.4190 |
| Champion Scorecard  | Decile 7 [Risk]         |    8260 |          55 |                0.6659 |             94.1024 | 24.1029 | 0.2845 |
| Champion Scorecard  | Decile 8 [Risk]         |    8260 |          54 |                0.6538 |             96.8960 | 16.8967 | 0.2794 |
| Champion Scorecard  | Decile 9 [Risk]         |    8260 |          40 |                0.4843 |             98.9653 |  8.9663 | 0.2069 |
| Champion Scorecard  | Decile 10 [Lowest Risk] |    8261 |          20 |                0.2421 |            100.0000 |  0.0000 | 0.1035 |
| Calibrated LightGBM | Decile 1 [Highest Risk] |    8261 |         896 |               10.8461 |             46.3528 | 36.3519 | 4.6348 |
| Calibrated LightGBM | Decile 2 [Risk]         |    8260 |         377 |                4.5642 |             65.8562 | 45.8555 | 1.9504 |
| Calibrated LightGBM | Decile 3 [Risk]         |    8260 |         226 |                2.7361 |             77.5479 | 47.5474 | 1.1692 |
| Calibrated LightGBM | Decile 4 [Risk]         |    8260 |         149 |                1.8039 |             85.2561 | 45.2558 | 0.7708 |
| Calibrated LightGBM | Decile 5 [Risk]         |    8260 |          93 |                1.1259 |             90.0673 | 40.0673 | 0.4811 |
| Calibrated LightGBM | Decile 6 [Risk]         |    8260 |          71 |                0.8596 |             93.7403 | 33.7405 | 0.3673 |
| Calibrated LightGBM | Decile 7 [Risk]         |    8260 |          40 |                0.4843 |             95.8096 | 25.8101 | 0.2069 |
| Calibrated LightGBM | Decile 8 [Risk]         |    8260 |          35 |                0.4237 |             97.6203 | 17.6210 | 0.1811 |
| Calibrated LightGBM | Decile 9 [Risk]         |    8260 |          28 |                0.3390 |             99.0688 |  9.0698 | 0.1449 |
| Calibrated LightGBM | Decile 10 [Lowest Risk] |    8261 |          18 |                0.2179 |            100.0000 |  0.0000 | 0.0931 |

严格边际 bad-rate 递减检查（Decile 1 至 10）: `{"Calibrated LightGBM": true, "Champion Scorecard": true}`。

## Table 4: 4-Group Ablation Study

| Model   |   Features |   Dev_Calib ROC-AUC |   Dev_Calib KS |   Dev_Calib Gini |   OOT_Valid ROC-AUC |   OOT_Valid KS |   OOT_Valid Gini |   Dev_Calib ROC-AUC Lift |   OOT_Valid ROC-AUC Lift |   Dev_Calib KS Lift |   OOT_Valid KS Lift |   Dev_Calib Gini Lift |   OOT_Valid Gini Lift |
|:--------|-----------:|--------------------:|---------------:|-----------------:|--------------------:|---------------:|-----------------:|-------------------------:|-------------------------:|--------------------:|--------------------:|----------------------:|----------------------:|
| M1      |         13 |              0.7084 |         0.3113 |           0.4167 |              0.7176 |         0.3236 |           0.4352 |                 —      |                 —      |            —      |            —      |              —      |              —      |
| M2      |         45 |              0.7694 |         0.4058 |           0.5388 |              0.7864 |         0.4290 |           0.5728 |                   0.0610 |                   0.0688 |              0.0945 |              0.1054 |                0.1221 |                0.1376 |
| M3      |         50 |              0.7839 |         0.4248 |           0.5678 |              0.8056 |         0.4671 |           0.6111 |                   0.0145 |                   0.0191 |              0.0190 |              0.0381 |                0.0290 |                0.0383 |
| M4      |         73 |              0.8043 |         0.4604 |           0.6085 |              0.8173 |         0.4929 |           0.6345 |                   0.0204 |                   0.0117 |              0.0356 |              0.0258 |                0.0408 |                0.0234 |

## Monotonic constraints

约束向量按各模型 feature 列顺序应用：DPD/delinquency/rejection rate/recent inquiry counts = +1；early payment ratios/employment years = -1；categorical/ambiguous ratios = 0。Categorical fields are passed to LightGBM as categorical features and use constraint 0；各组模型约束映射见下表。

| Feature                                | Group                           |   Constraint |   Internal PSI |
|:---------------------------------------|:--------------------------------|-------------:|---------------:|
| credamount_770A                        | Core_Affordability              |            0 |         0.0111 |
| annuity_780A                           | Core_Affordability              |            0 |         0.0026 |
| maininc_215A                           | Core_Affordability              |            0 |         0.0089 |
| annuitynextmonth_57A                   | Core_Affordability              |            0 |         0.0015 |
| pti                                    | Core_Affordability              |            0 |         0.0027 |
| loan_to_income                         | Core_Affordability              |            0 |         0.0005 |
| static_currdebt                        | Group_1_Traditional_Credit      |            0 |         0.0026 |
| static_currdebt_by_credtype            | Group_1_Traditional_Credit      |            0 |         0.0000 |
| static_totaldebt                       | Group_1_Traditional_Credit      |            0 |         0.0026 |
| static_last_approved_amount            | Group_1_Traditional_Credit      |            0 |         0.0144 |
| static_last_rejected_amount            | Group_1_Traditional_Credit      |            0 |         0.0099 |
| bureau_recent_query_velocity           | Group_1_Traditional_Credit      |            0 |         0.0455 |
| bureau_active_contracts_cnt            | Group_1_Traditional_Credit      |            0 |         0.0404 |
| active_bureau_monthly_debt             | Core_Affordability              |            0 |         0.0392 |
| has_active_bureau_debt                 | Core_Affordability              |            0 |         0.0185 |
| total_dsr_proxy                        | Core_Affordability              |            0 |         0.0045 |
| bureau_active_total_debt               | Group_1_Traditional_Credit      |            0 |         0.0323 |
| bureau_active_max_debt                 | Group_1_Traditional_Credit      |            0 |         0.0331 |
| total_debt_to_income                   | Core_Affordability              |            0 |         0.0077 |
| bureau_active_max_dpd                  | Group_1_Traditional_Credit      |            1 |         0.0020 |
| bureau_active_overdue_amt              | Group_1_Traditional_Credit      |            0 |         0.0000 |
| bureau_active_max_overdue_amt          | Group_1_Traditional_Credit      |            0 |         0.0052 |
| bureau_active_overdue_instls_max       | Group_1_Traditional_Credit      |            0 |         0.0000 |
| static_avg_dbd_dpd_last24m             | Group_2_Repayment_Behaviour     |            1 |         0.0050 |
| static_avg_dbd_dpd_last3m              | Group_2_Repayment_Behaviour     |            1 |         0.0325 |
| static_dpd_short_vs_long_trend         | Group_2_Repayment_Behaviour     |            1 |         0.0330 |
| static_avg_dpd_tol_closure24m          | Group_2_Repayment_Behaviour     |            1 |         0.0034 |
| static_avg_max_dpd_last9m              | Group_2_Repayment_Behaviour     |            1 |         0.0023 |
| static_days_overdue_tol                | Group_2_Repayment_Behaviour     |            0 |         0.0075 |
| static_avg_instal_last24m              | Group_2_Repayment_Behaviour     |            0 |         0.0108 |
| static_max_instal_last24m              | Group_2_Repayment_Behaviour     |            0 |         0.0232 |
| static_num_inst_paid_early             | Group_2_Repayment_Behaviour     |           -1 |         0.0046 |
| static_num_inst_paid_late_1d           | Group_2_Repayment_Behaviour     |            0 |         0.0050 |
| static_pct_inst_paid_late_10d          | Group_2_Repayment_Behaviour     |            0 |         0.0040 |
| static_pct_inst_paid_early_3d          | Group_2_Repayment_Behaviour     |           -1 |         0.0073 |
| static_cnt_incoming_pmts_9m            | Group_2_Repayment_Behaviour     |            0 |         0.0076 |
| static_cnt_pmts_months_24m             | Group_2_Repayment_Behaviour     |            0 |         0.0100 |
| static_avg_outstand_balance_6m         | Group_2_Repayment_Behaviour     |            0 |         0.0592 |
| static_max_outstand_balance_12m        | Group_2_Repayment_Behaviour     |            0 |         0.0676 |
| prev_appl_total_cnt                    | Group_2_Repayment_Behaviour     |            0 |         0.0173 |
| prev_appl_approved_cnt                 | Group_2_Repayment_Behaviour     |            0 |         0.0123 |
| prev_appl_rejected_cnt                 | Group_2_Repayment_Behaviour     |            0 |         0.0142 |
| prev_appl_approval_rate                | Group_2_Repayment_Behaviour     |            0 |         0.0084 |
| prev_appl_rejection_rate               | Group_2_Repayment_Behaviour     |            1 |         0.0077 |
| prev_appl_max_dpd_tolerance            | Group_2_Repayment_Behaviour     |            1 |         0.0047 |
| prev_appl_total_approved_amount        | Group_2_Repayment_Behaviour     |            0 |         0.0023 |
| prev_appl_max_credamount               | Group_2_Repayment_Behaviour     |            0 |         0.0234 |
| prev_appl_total_annuity                | Group_2_Repayment_Behaviour     |            0 |         0.0153 |
| prev_appl_max_annuity                  | Group_2_Repayment_Behaviour     |            0 |         0.0137 |
| prev_appl_max_downpayment              | Group_2_Repayment_Behaviour     |            0 |         0.0010 |
| prev_appl_total_pmt_count              | Group_2_Repayment_Behaviour     |            0 |         0.0163 |
| prev_appl_avg_tenor                    | Group_2_Repayment_Behaviour     |            0 |         0.0098 |
| prev_appl_outstanding_debt             | Group_2_Repayment_Behaviour     |            0 |         0.0026 |
| prev_card_records_cnt                  | Group_2_Repayment_Behaviour     |            0 |         0.0308 |
| prev_card_blocked_cnt                  | Group_2_Repayment_Behaviour     |            0 |         0.0304 |
| person_employment_years                | Group_3_Application_Demographic |           -1 |         0.0180 |
| person_income_type                     | Group_3_Application_Demographic |            0 |         0.0037 |
| person_education_level                 | Group_3_Application_Demographic |            0 |         0.0009 |
| person_family_status                   | Group_3_Application_Demographic |            0 |         0.0004 |
| person_role_type                       | Group_3_Application_Demographic |            0 |         0.0001 |
| app_cnt_last_30d                       | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| app_cnt_same_phone                     | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| app_cnt_same_mobile                    | Group_4_Alternative_Proxies     |            0 |         0.0192 |
| app_cnt_same_employer_7d               | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| app_cnt_same_employer_phone_30d        | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| clients_sharing_mobile_cnt             | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| clients_sharing_phone_cnt              | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| clients_matching_empl_and_client_phone | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| clients_same_mobile_12m                | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| has_debitcard                          | Group_4_Alternative_Proxies     |            0 |         0.0002 |
| debit_records_cnt                      | Group_4_Alternative_Proxies     |            0 |         0.0000 |
| has_tax_registry_record                | Group_4_Alternative_Proxies     |            0 |         1.1490 |
| tax_verified_income_ratio              | Group_4_Alternative_Proxies     |            0 |         0.3119 |

## 审计摘要与工件

Train_Dev rows=1,292,350; OOT_Valid rows=82,602; selected scorecard features=22; retained=19; M4 features=73. Score reconstruction verified. Decile strict rank check: {'Calibrated LightGBM': True, 'Champion Scorecard': True}.

`data/processed/models/scorecard_logit.pkl`, `lgbm_m4.pkl`, `isotonic_calibrator.pkl`; `data/processed/scorecard_points_table.csv`; `data/processed/stage3_predictions.parquet`; `data/processed/woe_bin_rules.json`; `reports/figures/shap_summary_bar.png`; `reports/figures/shap_beeswarm.png`.

**HARD STOP:** Stage 3 已完成；等待人工审阅后再开始 Stage 4.
