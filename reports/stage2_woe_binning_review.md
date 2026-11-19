# FinGuard Stage 2 审计报告 — Zero-Leakage Feature Selection & Monotonic WOE Binning

> **生成时间 (Generated)**: 2026-11-16 16:15:09
> **执行耗时 (Elapsed)**: 64.3 秒
> **训练集基准 (Training Baseline)**: 严格锁定 `Train_Dev` (`WEEK_NUM <= 64`，共 1,292,350 样本，42,846 坏客户)
> **已选特征表 (Selected Features Path)**: `D:\data analysis\FinGuard\data\processed\stage2_selected_features.csv`
> **分箱模型序列化 (Binning Pickle Path)**: `D:\data analysis\FinGuard\data\processed\woe_binning_dict.pkl`

---

## 1. 特征筛选漏斗审计 (4-Stage Feature Selection Funnel)

严格执行四阶漏斗，绝无任何 OOT 数据泄漏，每阶输入与剔除数量审计如下：

| Funnel Stage | Count | Drop Count | Criteria / Action |
| --- | --- | --- | --- |
| Initial Candidate Features | 104 | 0 | Starting engineered feature pool from Stage 1B |
| Fair Lending Sensitive Exclusion | 101 | 3 | Excluded gender_sensitive, age_band_sensitive, person_age |
| Gate 1: Missing & Dominance | 89 | 12 | Null Rate > 95% OR Dominant Single Value > 98% |
| Gate 2: Univariate Monotonic IV | 41 | 48 | OptimalBinning 0.02 <= IV <= 0.60 (monotonic auto_asc_desc) |
| Gate 3: Internal Temporal PSI | 39 | 2 | Early_Dev (W 0-44) vs Late_Dev (W 45-64) PSI < 0.10 |
| Gate 4: Pairwise Correlation | 23 | 16 | Spearman |r| <= 0.70 (keep higher-IV feature) |
| Gate 4: VIF Multicollinearity | 23 | 0 | Variance Inflation Factor (VIF) < 5.0 |

### 漏斗剔除原因明细 (Drop Reason Breakdown)
- **敏感字段剔除 (Fair Lending Exclusion, 3个)**: `person_gender_sensitive`, `person_age_band_sensitive`, `person_age` 隔离备查。
- **Gate 1 剔除 (高缺失/单一值, 12个)**:
  - 高缺失 (>95%): `person_housing_type`, `person_children_cnt`, `debit_turnover_velocity`
  - 高单一值 (>98%): `bureau_closed_total_debt` (99.89%=0), `bureau_active_prolongation_cnt` (99.33%=0), `prev_appl_max_dpd` (99.72%=0), `prev_appl_mean_dpd` (99.72%=0), `person_is_contact_same_as_empl` (99.45%=0), `app_cnt_same_email` (100%=0), `debit_avg_bal_180d` (99.74%=0), `debit_avg_turnover_180d` (99.35%=0), `debit_avg_turnover_30d` (99.92%=0)
- **Gate 2 剔除 (IV < 0.02, 48个)**: 预测力微弱的变量自动剔除；**无变量 IV > 0.60**（确认无目标泄漏变量）。
- **Gate 3 剔除 (内部时序漂移 PSI >= 0.10, 2个)**: `bureau_total_contracts_cnt` (PSI = 0.5097), `bureau_closed_max_dpd` (PSI = 0.1440)。
- **Gate 4 剔除 (多重共线性 |r| > 0.70, 16个)**: 保持低共线性与高业务可解释性，同质变量优先保留高 IV 主特征。

---

## 2. 最终入选特征汇总表 (Top Surviving Features Summary)

共 **23 个高质量风控特征** 成功通过全部四阶漏斗，且多重共线性指标满足 **VIF < 5.0**：

| feature_name | feature_group | iv | internal_psi | vif | monotonic_direction | dtype |
| --- | --- | --- | --- | --- | --- | --- |
| static_avg_dpd_tol_closure24m | Group_2_Repayment_Behaviour | 0.3051 | 0.0006 | 3.1500 | auto_asc_desc | numerical |
| bureau_active_max_dpd | Group_1_Traditional_Credit | 0.2385 | 0.0039 | 1.1800 | auto_asc_desc | numerical |
| static_avg_dbd_dpd_last24m | Group_2_Repayment_Behaviour | 0.2269 | 0.0026 | 2.2600 | auto_asc_desc | numerical |
| static_days_overdue_tol | Group_2_Repayment_Behaviour | 0.2181 | 0.0044 | 3.3000 | auto_asc_desc | numerical |
| static_pct_inst_paid_early_3d | Group_2_Repayment_Behaviour | 0.2128 | 0.0045 | 2.7600 | auto_asc_desc | numerical |
| prev_appl_max_dpd_tolerance | Group_2_Repayment_Behaviour | 0.2117 | 0.0048 | 2.3800 | auto_asc_desc | numerical |
| static_pct_inst_paid_late_10d | Group_2_Repayment_Behaviour | 0.2033 | 0.0007 | 2.4400 | auto_asc_desc | numerical |
| static_avg_max_dpd_last9m | Group_2_Repayment_Behaviour | 0.1472 | 0.0011 | 1.6400 | auto_asc_desc | numerical |
| static_last_rejected_amount | Group_1_Traditional_Credit | 0.1285 | 0.0097 | 2.1600 | auto_asc_desc | numerical |
| static_num_inst_paid_early | Group_2_Repayment_Behaviour | 0.1107 | 0.0026 | 2.8100 | auto_asc_desc | numerical |
| static_avg_dbd_dpd_last3m | Group_2_Repayment_Behaviour | 0.1094 | 0.0197 | 1.4100 | auto_asc_desc | numerical |
| bureau_queries_120d | Group_1_Traditional_Credit | 0.1088 | 0.0177 | 1.7500 | auto_asc_desc | numerical |
| prev_appl_avg_tenor | Group_2_Repayment_Behaviour | 0.1007 | 0.0095 | 1.7600 | auto_asc_desc | numerical |
| bureau_queries_30d | Group_1_Traditional_Credit | 0.0802 | 0.0029 | 1.7300 | auto_asc_desc | numerical |
| person_income_type | Group_3_Application_Demographic | 0.0660 | 0.0017 | 1.1000 | auto | categorical |
| person_employment_years | Group_3_Application_Demographic | 0.0567 | 0.0023 | 1.1000 | auto_asc_desc | numerical |
| prev_appl_total_cnt | Group_2_Repayment_Behaviour | 0.0487 | 0.0165 | 2.5000 | auto_asc_desc | numerical |
| app_cnt_last_30d | Group_4_Alternative_Proxies | 0.0376 | 0.0018 | 1.1900 | auto_asc_desc | numerical |
| prev_appl_rejection_rate | Group_2_Repayment_Behaviour | 0.0367 | 0.0074 | 2.1100 | auto_asc_desc | numerical |
| person_education_level | Group_3_Application_Demographic | 0.0346 | 0.0001 | 1.0100 | auto | categorical |
| disbursedcredamount_1113A | Core_Affordability | 0.0311 | 0.0069 | 1.0400 | auto_asc_desc | numerical |
| static_cnt_pmts_months_24m | Group_2_Repayment_Behaviour | 0.0258 | 0.0054 | 1.6000 | auto_asc_desc | numerical |
| person_family_status | Group_3_Application_Demographic | 0.0219 | 0.0001 | 1.0600 | auto | categorical |

---

## 3. Top 10 最高 IV 特征单调 WOE 分箱明细 (Detailed Bin Tables)

> 严格遵守**单调性业务约束** (`auto_asc_desc`) 与 `min_bin_size=0.05`，各分箱 Bad Rate 保持严格单调递增或递减，业务逻辑透明。


### 1. `static_avg_dpd_tol_closure24m`
- **所属分组 (Group)**: Group_2_Repayment_Behaviour
- **信息量 (IV)**: **0.3051** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0006 | **VIF**: 3.15

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 0.50) | 635363 | 49.16% | 622071 | 13292 | 2.09% | 0.4730 | 0.0887 |
| [0.50, 1.50) | 105580 | 8.17% | 100909 | 4671 | 4.42% | -0.3000 | 0.0085 |
| [1.50, 5.50) | 69717 | 5.39% | 64696 | 5021 | 7.20% | -0.8168 | 0.0534 |
| [5.50, inf) | 70044 | 5.42% | 62642 | 7402 | 10.57% | -1.2372 | 0.1517 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 411646 | 31.85% | 399186 | 12460 | 3.03% | 0.0940 | 0.0027 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.3051 |

### 2. `bureau_active_max_dpd`
- **所属分组 (Group)**: Group_1_Traditional_Credit
- **信息量 (IV)**: **0.2385** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0039 | **VIF**: 1.18

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 0.50) | 947754 | 73.34% | 924450 | 23304 | 2.46% | 0.3077 | 0.0603 |
| [0.50, 3.50) | 107950 | 8.35% | 104212 | 3738 | 3.46% | -0.0450 | 0.0002 |
| [3.50, 10.50) | 75281 | 5.83% | 71812 | 3469 | 4.61% | -0.3427 | 0.0081 |
| [10.50, 27.50) | 89103 | 6.89% | 84132 | 4971 | 5.58% | -0.5441 | 0.0265 |
| [27.50, inf) | 72262 | 5.59% | 64898 | 7364 | 10.19% | -1.1967 | 0.1435 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.2385 |

### 3. `static_avg_dbd_dpd_last24m`
- **所属分组 (Group)**: Group_2_Repayment_Behaviour
- **信息量 (IV)**: **0.2269** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0026 | **VIF**: 2.26

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, -8.50) | 248711 | 19.24% | 244015 | 4696 | 1.89% | 0.5776 | 0.0495 |
| [-8.50, -5.50) | 96667 | 7.48% | 94640 | 2027 | 2.10% | 0.4706 | 0.0134 |
| [-5.50, -3.50) | 86924 | 6.73% | 84704 | 2220 | 2.55% | 0.2688 | 0.0043 |
| [-3.50, -1.50) | 112549 | 8.71% | 109269 | 3280 | 2.91% | 0.1331 | 0.0015 |
| [-1.50, 1.50) | 133607 | 10.34% | 127705 | 5902 | 4.42% | -0.2985 | 0.0106 |
| [1.50, inf) | 78971 | 6.11% | 71143 | 7828 | 9.91% | -1.1659 | 0.1466 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 534921 | 41.39% | 518028 | 16893 | 3.16% | 0.0502 | 0.0010 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.2269 |

### 4. `static_days_overdue_tol`
- **所属分组 (Group)**: Group_2_Repayment_Behaviour
- **信息量 (IV)**: **0.2181** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0044 | **VIF**: 3.30

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 2.50) | 475117 | 36.76% | 465430 | 9687 | 2.04% | 0.4993 | 0.0731 |
| [2.50, 4.50) | 69843 | 5.40% | 67734 | 2109 | 3.02% | 0.0965 | 0.0005 |
| [4.50, 12.50) | 119690 | 9.26% | 115261 | 4429 | 3.70% | -0.1139 | 0.0013 |
| [12.50, 28.50) | 83963 | 6.50% | 79918 | 4045 | 4.82% | -0.3894 | 0.0119 |
| [28.50, 69.50) | 68586 | 5.31% | 64536 | 4050 | 5.90% | -0.6044 | 0.0259 |
| [69.50, inf) | 75868 | 5.87% | 69241 | 6627 | 8.73% | -1.0264 | 0.1019 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 399283 | 30.90% | 387384 | 11899 | 2.98% | 0.1101 | 0.0036 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.2181 |

### 5. `static_pct_inst_paid_early_3d`
- **所属分组 (Group)**: Group_2_Repayment_Behaviour
- **信息量 (IV)**: **0.2128** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0045 | **VIF**: 2.76

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 0.16) | 100111 | 7.75% | 92986 | 7125 | 7.12% | -0.8041 | 0.0739 |
| [0.16, 0.30) | 85839 | 6.64% | 81044 | 4795 | 5.59% | -0.5455 | 0.0257 |
| [0.30, 0.52) | 175607 | 13.59% | 168091 | 7516 | 4.28% | -0.2654 | 0.0109 |
| [0.52, 0.72) | 179168 | 13.86% | 173817 | 5351 | 2.99% | 0.1078 | 0.0015 |
| [0.72, 0.84) | 114152 | 8.83% | 111729 | 2423 | 2.12% | 0.4582 | 0.0151 |
| [0.84, inf) | 232882 | 18.02% | 229385 | 3497 | 1.50% | 0.8106 | 0.0827 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 404591 | 31.31% | 392452 | 12139 | 3.00% | 0.1031 | 0.0032 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.2128 |

### 6. `prev_appl_max_dpd_tolerance`
- **所属分组 (Group)**: Group_2_Repayment_Behaviour
- **信息量 (IV)**: **0.2117** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0048 | **VIF**: 2.38

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 0.50) | 863458 | 66.81% | 841606 | 21852 | 2.53% | 0.2781 | 0.0455 |
| [0.50, 3.50) | 198984 | 15.40% | 193278 | 5706 | 2.87% | 0.1497 | 0.0032 |
| [3.50, 12.50) | 104916 | 8.12% | 100188 | 4728 | 4.51% | -0.3193 | 0.0096 |
| [12.50, inf) | 124992 | 9.67% | 114432 | 10560 | 8.45% | -0.9900 | 0.1533 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.2117 |

### 7. `static_pct_inst_paid_late_10d`
- **所属分组 (Group)**: Group_2_Repayment_Behaviour
- **信息量 (IV)**: **0.2033** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0007 | **VIF**: 2.44

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 0.03) | 562869 | 43.55% | 549703 | 13166 | 2.34% | 0.3589 | 0.0476 |
| [0.03, 0.06) | 64922 | 5.02% | 63158 | 1764 | 2.72% | 0.2052 | 0.0019 |
| [0.06, 0.17) | 112398 | 8.70% | 107913 | 4485 | 3.99% | -0.1923 | 0.0035 |
| [0.17, 0.32) | 76024 | 5.88% | 71468 | 4556 | 5.99% | -0.6201 | 0.0305 |
| [0.32, inf) | 69111 | 5.35% | 62512 | 6599 | 9.55% | -1.1244 | 0.1169 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 407026 | 31.50% | 394750 | 12276 | 3.02% | 0.0977 | 0.0029 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.2033 |

### 8. `static_avg_max_dpd_last9m`
- **所属分组 (Group)**: Group_2_Repayment_Behaviour
- **信息量 (IV)**: **0.1472** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0011 | **VIF**: 1.64

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 0.50) | 509448 | 39.42% | 497662 | 11786 | 2.31% | 0.3701 | 0.0456 |
| [0.50, inf) | 131321 | 10.16% | 121834 | 9487 | 7.22% | -0.8202 | 0.1016 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 651581 | 50.42% | 630008 | 21573 | 3.31% | 0.0014 | 0.0000 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.1472 |

### 9. `static_last_rejected_amount`
- **所属分组 (Group)**: Group_1_Traditional_Credit
- **信息量 (IV)**: **0.1285** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0097 | **VIF**: 2.16

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 9999.90) | 89136 | 6.90% | 85879 | 3257 | 3.65% | -0.1008 | 0.0007 |
| [9999.90, 39955.00) | 248351 | 19.22% | 237116 | 11235 | 4.52% | -0.3234 | 0.0234 |
| [39955.00, 69191.00) | 140480 | 10.87% | 134093 | 6387 | 4.55% | -0.3286 | 0.0137 |
| [69191.00, inf) | 147414 | 11.41% | 140353 | 7061 | 4.79% | -0.3833 | 0.0201 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 666969 | 51.61% | 652063 | 14906 | 2.23% | 0.4055 | 0.0705 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.1285 |

### 10. `static_num_inst_paid_early`
- **所属分组 (Group)**: Group_2_Repayment_Behaviour
- **信息量 (IV)**: **0.1107** | **单调性约束 (Direction)**: `auto_asc_desc` | **内部稳定性 (PSI)**: 0.0026 | **VIF**: 2.81

| Bin | Count | Count (%) | Non-event | Event | Event rate | WoE | IV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-inf, 0.50) | 110938 | 8.58% | 104356 | 6582 | 5.93% | -0.6094 | 0.0427 |
| [0.50, 3.50) | 164539 | 12.73% | 156913 | 7626 | 4.63% | -0.3488 | 0.0183 |
| [3.50, 9.50) | 210102 | 16.26% | 202779 | 7323 | 3.49% | -0.0518 | 0.0004 |
| [9.50, 22.50) | 215065 | 16.64% | 209409 | 5656 | 2.63% | 0.2387 | 0.0085 |
| [22.50, 41.50) | 109866 | 8.50% | 107439 | 2427 | 2.21% | 0.4174 | 0.0122 |
| [41.50, inf) | 82558 | 6.39% | 81225 | 1333 | 1.61% | 0.7369 | 0.0250 |
| Special | 0 | 0.00% | 0 | 0 | 0.00% | 0.0000 | 0.0000 |
| Missing | 399282 | 30.90% | 387383 | 11899 | 2.98% | 0.1101 | 0.0036 |
|  | 1292350 | 100.00% | 1249504 | 42846 | 3.32% | — | 0.1107 |


---

## 4. 阶段合规自检与 HARD STOP

| 审查维度 | 规范准则 | 执行核验结果 | 合规状态 |
|---|---|---|---|
| **零数据泄漏 (Zero Leakage)** | 仅允许 WEEK_NUM <= 64 参与 | 100% 在 Train_Dev 上计算 | ✅ |
| **敏感属性隔离 (Fair Lending)** | 排除性别与年龄变量 | 3 个敏感特征全部隔离 | ✅ |
| **单调性约束 (Monotonicity)** | optbinning auto_asc_desc | Top 10 分箱严格单调 | ✅ |
| **时序稳定性 (Internal PSI)** | Early vs Late Dev PSI < 0.10 | 剔除 2 个高漂移特征 | ✅ |
| **多重共线性 (Multicollinearity)** | `|r| <= 0.70` 且 VIF < 5.0 | 最大 VIF = 3.30 < 5.0 | ✅ |
| **分箱工件归档 (Artifacts)** | woe_binning_dict.pkl 序列化 | 23 个模型对象完整导出 | ✅ |

> [!IMPORTANT]
> **HARD STOP** — Stage 2 严格特征筛选与单调 WOE 已经完成。请审阅上方 Funnel 表、入选变量表与 Top 10 分箱单调性。确认无误后方可进入 **Stage 3: 信用评分卡模型拟合 (Logistic Regression with p < 0.05 & LightGBM Benchmark)**。
