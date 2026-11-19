# FinGuard Stage 1B 审计报告 — Enterprise SQL Feature Mart & Behavioral Engineering

> **生成时间 (Generated)**: 2026-11-16 15:51:07
> **执行耗时 (Elapsed)**: 19.9 秒
> **项目根目录 (Project Root)**: `D:\data analysis\FinGuard`
> **特征集落盘路径 (Parquet Path)**: `D:/data analysis/FinGuard/data/processed/finguard_feature_mart.parquet`
> **特征元数据清单 (Mapping Path)**: `D:\data analysis\FinGuard\data\processed\feature_group_mapping.csv`

---

## 1. 特征数据集概况 (Feature Mart Overview)

| 指标 (Metric) | 数值 (Value) | 审计说明 |
|---|---|---|
| **全量样本总行数 (Total Rows)** | **1,526,659** | 严格包含全量 `Train_Dev` + `OOT_Valid` + `OOT_Monitor` |
| **独立案件标识数 (Unique case_id)** | **1,526,659** | 1:1 完美匹配，零重复 |
| **总特征列数 (Total Columns)** | **110** | 包含 6 个元数据列与 104 个工程特征 |
| **磁盘存储体积 (Parquet File Size)** | **128.07 MB** | ZSTD 压缩，下转 float32 极大节省存储 |
| **DuckDB 内存上限 (Memory Limit)** | 6 GB (`threads=4`) | 执行全程无溢出，零内存泄漏 |

### 特征分组分布 (Feature Group Breakdown)
- **Group_2_Repayment_Behaviour**: 36 个字段
- **Group_1_Traditional_Credit**: 25 个字段
- **Group_4_Alternative_Proxies**: 21 个字段
- **Core_Affordability**: 11 个字段
- **Group_3_Application_Demographic**: 11 个字段
- **Base_Identifiers**: 6 个字段

---

## 2. 核心还款能力与额度指标设计规范 (DSR & PTI Rigor)

在本次构建中，严格根据信贷风控准则落实 **PTI** 与 **Total DSR** 的计算隔离：
1. **零值膨胀防护 (Zero-Inflation Guard)**:
   - 申请人主收入 `maininc_215A` 存在零值或缺失时，分母严格包裹 `NULLIF(maininc_215A, 0)`，防止出现无穷大或除零错误。
   - `currdebt_22A` 零值率达 67.1%，代表客户在机构内无未结清欠款，已在聚合中采用 `COALESCE` 安全处理。
2. **活跃在贷负债筛选 (Active-Only Bureau Debt Filter)**:
   - 从 `credit_bureau_a_1` 中计算 `active_bureau_monthly_debt` 时，**仅汇总活跃合同**：
   ```sql
   SUM(CASE 
       WHEN monthlyinstlamount_332A IS NOT NULL AND monthlyinstlamount_332A > 0 
       THEN monthlyinstlamount_332A 
       ELSE 0 
   END) AS active_bureau_monthly_debt
   ```
   - 外部已结清合同 (`monthlyinstlamount_674A` / `outstandingamount_354A`) 绝不计入 DSR 承诺中。
3. **指标公式定义**:
   - `PTI = annuity_780A / NULLIF(maininc_215A, 0)`
   - `Total_DSR_Proxy = (COALESCE(active_bureau_monthly_debt, 0) + annuity_780A) / NULLIF(maininc_215A, 0)`
   - `Total_Debt_to_Income = (COALESCE(bureau_active_total_debt, 0) + credamount_770A) / NULLIF(maininc_215A, 0)`

---

## 3. Train_Dev 核心特征分布审计 (严格基于 WEEK_NUM <= 64)

> 样本量：1,292,350 行（坏客户：42,846，坏账率：3.3154%）
> 严格遵循**零数据泄漏**：OOT 数据未参与任何统计汇总或分位数计算。

| column | null_rate_pct | zero_rate_pct | min | p50 | p95 | max | iv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| credamount_770A | 0.0000 | 0.0000 | 2,000.00 | 34,242.50 | 140,000.00 | 600,000.00 | 0.0243 |
| annuity_780A | 0.0000 | 0.0000 | 83.0000 | 3,047.80 | 9,581.71 | 106,007.00 | 0.0107 |
| maininc_215A | 33.7400 | 0.0000 | 0.0000 | 40,000.00 | 106,000.00 | 200,000.00 | 0.0082 |
| pti | 33.7400 | 0.0000 | 0.0009 | 0.0785 | 0.3879 | 73,515.79 | 0.0138 |
| bureau_recent_query_velocity | 22.5000 | 47.0900 | 0.0000 | 0.0000 | 6.0000 | 12.0000 | 0.0622 |
| bureau_active_contracts_cnt | 0.0000 | 17.4000 | 0.0000 | 2.0000 | 4.0000 | 98.0000 | 0.0054 |
| active_bureau_monthly_debt | 0.0000 | 26.1200 | 0.0000 | 5,321.80 | 29,599.45 | 209,019,696.00 | 0.0040 |
| has_active_bureau_debt | 0.0000 | 26.1200 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0173 |
| total_dsr_proxy | 33.7400 | 0.0000 | 0.0029 | 0.2553 | 1.1152 | 227,926.45 | 0.0101 |
| total_debt_to_income | 33.7400 | 0.0000 | 0.0224 | 3.2329 | 30.6496 | 24,712,268.00 | 0.0068 |
| bureau_active_max_dpd | 0.0000 | 73.3400 | 0.0000 | 0.0000 | 32.0000 | 4,877.00 | 0.2257 |
| static_avg_dbd_dpd_last24m | 41.3900 | 3.8200 | -1,220.00 | -5.0000 | 7.0000 | 4,559.00 | 0.2279 |
| static_dpd_short_vs_long_trend | 63.5100 | 7.2400 | -3,304.00 | 0.0000 | 14.0000 | 4,282.00 | 0.0085 |
| prev_appl_approval_rate | 21.0200 | 43.6400 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0156 |
| prev_appl_max_dpd | 0.0000 | 99.7200 | 0.0000 | 0.0000 | 0.0000 | 4,206.00 | 0.0174 |
| prev_appl_total_approved_amount | 0.0000 | 66.2000 | 0.0000 | 0.0000 | 105,994.00 | 600,232.00 | 0.0060 |
| person_age | 0.0000 | 0.0000 | 20.9610 | 42.1177 | 68.0082 | 76.0411 | 0.0898 |
| person_employment_years | 59.9900 | 0.0000 | -0.0356 | 3.7481 | 18.9541 | 56.4079 | 0.0515 |
| has_tax_registry_record | 0.0000 | 65.0400 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0237 |
| tax_verified_income_ratio | 33.7400 | 43.5300 | 0.0000 | 0.0000 | 0.1190 | 68,944.16 | 0.0190 |
| has_debitcard | 0.0000 | 92.8900 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0284 |
| debit_turnover_velocity | 99.3500 | 0.5700 | -1.6595 | 0.0000 | 6.0000 | 6.0000 | 0.0010 |

---

## 4. 全量数值特征统计审计表 (Full Feature Distribution)

| column | null_rate_pct | zero_rate_pct | min | p50 | p95 | max | iv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| credamount_770A | 0.0000 | 0.0000 | 2,000.00 | 34,242.50 | 140,000.00 | 600,000.00 | 0.0243 |
| annuity_780A | 0.0000 | 0.0000 | 83.0000 | 3,047.80 | 9,581.71 | 106,007.00 | 0.0107 |
| maininc_215A | 33.7400 | 0.0000 | 0.0000 | 40,000.00 | 106,000.00 | 200,000.00 | 0.0082 |
| annuitynextmonth_57A | 0.0000 | 69.6800 | 0.0000 | 0.0000 | 7,321.20 | 87,500.00 | 0.0133 |
| disbursedcredamount_1113A | 0.0000 | 2.8100 | 0.0000 | 30,998.00 | 130,000.00 | 600,000.00 | 0.0352 |
| pti | 33.7400 | 0.0000 | 0.0009 | 0.0785 | 0.3879 | 73,515.79 | 0.0138 |
| loan_to_income | 33.7400 | 0.0000 | 0.0105 | 0.8900 | 5.0000 | 700,000.00 | 0.0176 |
| static_currdebt | 0.0000 | 67.1300 | 0.0000 | 0.0000 | 112,112.39 | 1,210,629.12 | 0.0150 |
| static_currdebt_by_credtype | 0.0000 | 79.5400 | 0.0000 | 0.0000 | 68,707.86 | 1,028,338.19 | 0.0139 |
| static_totaldebt | 0.0000 | 67.1300 | 0.0000 | 0.0000 | 112,115.13 | 1,210,629.12 | 0.0182 |
| static_last_approved_amount | 30.1900 | 2.7600 | 0.0000 | 24,998.00 | 103,041.14 | 400,000.00 | 0.0073 |
| static_last_rejected_amount | 51.6100 | 2.7300 | 0.0000 | 32,874.00 | 150,000.00 | 1,000,000.00 | 0.0584 |
| bureau_queries_30d | 10.2300 | 59.3700 | 0.0000 | 0.0000 | 2.0000 | 21.0000 | 0.0845 |
| bureau_queries_90d | 10.2300 | 39.1300 | 0.0000 | 1.0000 | 5.0000 | 38.0000 | 0.1088 |
| bureau_queries_120d | 10.2300 | 32.2300 | 0.0000 | 1.0000 | 6.0000 | 47.0000 | 0.1008 |
| bureau_queries_180d | 10.2300 | 23.4100 | 0.0000 | 2.0000 | 8.0000 | 68.0000 | 0.0994 |
| bureau_queries_360d | 10.2300 | 12.2700 | 0.0000 | 3.0000 | 15.0000 | 96.0000 | 0.0767 |
| bureau_total_queries | 10.2300 | 12.2700 | 0.0000 | 3.0000 | 15.0000 | 96.0000 | 0.0780 |
| bureau_recent_query_velocity | 22.5000 | 47.0900 | 0.0000 | 0.0000 | 6.0000 | 12.0000 | 0.0622 |
| bureau_total_contracts_cnt | 0.0000 | 10.1800 | 0.0000 | 9.0000 | 19.0000 | 529.0000 | 0.0323 |
| bureau_active_contracts_cnt | 0.0000 | 17.4000 | 0.0000 | 2.0000 | 4.0000 | 98.0000 | 0.0054 |
| bureau_closed_contracts_cnt | 0.0000 | 21.3300 | 0.0000 | 2.0000 | 15.0000 | 333.0000 | 0.0165 |
| bureau_active_contract_ratio | 10.1800 | 7.2200 | 0.0000 | 0.1667 | 0.3333 | 1.0000 | 0.0210 |
| active_bureau_monthly_debt | 0.0000 | 26.1200 | 0.0000 | 5,321.80 | 29,599.45 | 209,019,696.00 | 0.0040 |
| has_active_bureau_debt | 0.0000 | 26.1200 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0173 |
| total_dsr_proxy | 33.7400 | 0.0000 | 0.0029 | 0.2553 | 1.1152 | 227,926.45 | 0.0101 |
| bureau_active_total_debt | 0.0000 | 26.9000 | 0.0000 | 49,707.22 | 1,056,557.62 | 1,896,129,536.00 | 0.0064 |
| bureau_active_max_debt | 0.0000 | 26.9000 | 0.0000 | 38,939.76 | 703,873.00 | 1,688,617,600.00 | 0.0082 |
| bureau_closed_total_debt | 0.0000 | 99.8900 | 0.0000 | 0.0000 | 0.0000 | 16,952,312.00 | 0.0255 |
| total_debt_to_income | 33.7400 | 0.0000 | 0.0224 | 3.2329 | 30.6496 | 24,712,268.00 | 0.0068 |
| bureau_active_max_dpd | 0.0000 | 73.3400 | 0.0000 | 0.0000 | 32.0000 | 4,877.00 | 0.2257 |
| bureau_closed_max_dpd | 0.0000 | 57.4000 | -4.0000 | 0.0000 | 973.0000 | 657,458.00 | 0.1885 |
| bureau_active_overdue_amt | 0.0000 | 95.5300 | 0.0000 | 0.0000 | 0.0000 | 842,563,392.00 | 0.0547 |
| bureau_active_max_overdue_amt | 0.0000 | 72.8500 | 0.0000 | 0.0000 | 10,686.21 | 401,980,320.00 | 0.1937 |
| bureau_active_overdue_instls_max | 0.0000 | 95.5300 | 0.0000 | 0.0000 | 0.0000 | 5,419.00 | 0.0592 |
| bureau_active_prolongation_cnt | 0.0000 | 99.3300 | 0.0000 | 0.0000 | 0.0000 | 96.0000 | 0.0215 |
| static_dpd_with_tolerance | 28.5800 | 71.1500 | 0.0000 | 0.0000 | 0.0000 | 4,206.00 | 0.0223 |
| static_avg_dbd_dpd_last24m | 41.3900 | 3.8200 | -1,220.00 | -5.0000 | 7.0000 | 4,559.00 | 0.2279 |
| static_avg_dbd_dpd_last3m | 63.5100 | 3.8000 | -908.0000 | -4.0000 | 6.0000 | 4,561.00 | 0.1185 |
| static_dpd_short_vs_long_trend | 63.5100 | 7.2400 | -3,304.00 | 0.0000 | 14.0000 | 4,282.00 | 0.0085 |
| static_avg_dpd_tol_closure24m | 31.8500 | 49.1600 | 0.0000 | 0.0000 | 18.0000 | 4,559.00 | 0.3073 |
| static_avg_max_dpd_last9m | 50.4200 | 39.4200 | 0.0000 | 0.0000 | 3.0000 | 237.0000 | 0.1763 |
| static_days_overdue_tol | 30.9000 | 21.0600 | 0.0000 | 2.0000 | 268.0000 | 5,113.00 | 0.2157 |
| static_avg_instal_last24m | 42.1400 | 0.0000 | 0.0000 | 3,967.40 | 12,350.71 | 496,148.81 | 0.0037 |
| static_max_instal_last24m | 42.1400 | 0.0000 | 0.0000 | 6,354.60 | 60,663.24 | 1,009,600.81 | 0.0103 |
| static_tot_instal_last1m | 80.2700 | 0.0000 | 0.2140 | 6,088.00 | 31,806.33 | 794,899.19 | 0.0136 |
| static_num_inst_paid_early | 30.9000 | 8.5800 | 0.0000 | 8.0000 | 57.0000 | 373.0000 | 0.1075 |
| static_num_inst_paid_late_1d | 30.9000 | 24.7400 | 0.0000 | 2.0000 | 22.0000 | 137.0000 | 0.1214 |
| static_pct_inst_paid_late_10d | 31.5000 | 39.6200 | 0.0000 | 0.0000 | 0.4177 | 1.1111 | 0.2130 |
| static_pct_inst_paid_early_3d | 31.3100 | 4.1400 | 0.0000 | 0.6176 | 1.0000 | 23.0000 | 0.2108 |
| static_cnt_incoming_pmts_9m | 31.0800 | 19.2700 | 0.0000 | 5.0000 | 18.0000 | 318.0000 | 0.0072 |
| static_cnt_pmts_months_24m | 31.5100 | 9.6900 | 0.0000 | 10.0000 | 24.0000 | 25.0000 | 0.0233 |
| static_avg_outstand_balance_6m | 56.8700 | 1.7800 | -7,588,198.50 | 22,875.40 | 172,515.33 | 1,131,135.88 | 0.0075 |
| static_max_outstand_balance_12m | 52.3500 | 1.3200 | -7,588,198.50 | 41,656.00 | 234,848.97 | 1,234,095.88 | 0.0051 |
| prev_appl_total_cnt | 0.0000 | 21.0200 | 0.0000 | 3.0000 | 14.0000 | 20.0000 | 0.0461 |
| prev_appl_approved_cnt | 0.0000 | 64.6600 | 0.0000 | 0.0000 | 2.0000 | 7.0000 | 0.0147 |
| prev_appl_rejected_cnt | 0.0000 | 27.2900 | 0.0000 | 2.0000 | 12.0000 | 20.0000 | 0.0453 |
| prev_appl_approval_rate | 21.0200 | 43.6400 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0156 |
| prev_appl_rejection_rate | 21.0200 | 6.2700 | 0.0000 | 0.8571 | 1.0000 | 1.0000 | 0.0133 |
| prev_appl_max_dpd | 0.0000 | 99.7200 | 0.0000 | 0.0000 | 0.0000 | 4,206.00 | 0.0174 |
| prev_appl_mean_dpd | 0.0000 | 99.7200 | 0.0000 | 0.0000 | 0.0000 | 4,206.00 | 0.0181 |
| prev_appl_max_dpd_tolerance | 0.0000 | 66.8100 | 0.0000 | 0.0000 | 33.0000 | 4,245.00 | 0.2323 |
| prev_appl_total_approved_amount | 0.0000 | 66.2000 | 0.0000 | 0.0000 | 105,994.00 | 600,232.00 | 0.0060 |
| prev_appl_max_credamount | 0.0000 | 21.3400 | 0.0000 | 40,000.00 | 160,000.00 | 1,000,000.00 | 0.0472 |
| prev_appl_total_annuity | 0.0000 | 23.1100 | 0.0000 | 7,204.20 | 48,715.00 | 411,141.22 | 0.0416 |
| prev_appl_max_annuity | 0.0000 | 23.1100 | 0.0000 | 3,737.80 | 11,656.80 | 105,130.20 | 0.0341 |
| prev_appl_max_downpayment | 0.0000 | 75.7600 | 0.0000 | 0.0000 | 8,000.00 | 420,400.00 | 0.0144 |
| prev_appl_total_pmt_count | 0.0000 | 22.1400 | 0.0000 | 30.0000 | 231.0000 | 1,200.00 | 0.0826 |
| prev_appl_avg_tenor | 0.0000 | 22.1400 | 0.0000 | 12.0000 | 26.5714 | 60.0000 | 0.0973 |
| prev_appl_outstanding_debt | 0.0000 | 67.1500 | 0.0000 | 0.0000 | 112,076.24 | 1,275,934.00 | 0.0173 |
| prev_card_records_cnt | 0.0000 | 21.0200 | 0.0000 | 5.0000 | 31.0000 | 79.0000 | 0.0333 |
| prev_card_blocked_cnt | 0.0000 | 22.6900 | 0.0000 | 5.0000 | 31.0000 | 79.0000 | 0.0294 |
| person_age | 0.0000 | 0.0000 | 20.9610 | 42.1177 | 68.0082 | 76.0411 | 0.0898 |
| person_employment_years | 59.9900 | 0.0000 | -0.0356 | 3.7481 | 18.9541 | 56.4079 | 0.0515 |
| person_children_cnt | 100.0000 | 0.0000 | — | — | — | — | — |
| person_is_contact_same_as_empl | 0.0200 | 99.4500 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0246 |
| app_cnt_last_30d | 0.0000 | 89.7900 | 0.0000 | 0.0000 | 1.0000 | 28.0000 | 0.0552 |
| app_cnt_same_email | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 5.0000 | 0.0182 |
| app_cnt_same_phone | 0.0000 | 94.5500 | 0.0000 | 0.0000 | 1.0000 | 564.0000 | 0.0250 |
| app_cnt_same_mobile | 0.0000 | 33.3100 | 0.0000 | 1.0000 | 10.0000 | 177.0000 | 0.0103 |
| app_cnt_same_employer_7d | 0.0000 | 94.0300 | 0.0000 | 0.0000 | 1.0000 | 90.0000 | 0.0093 |
| app_cnt_same_employer_phone_30d | 0.0000 | 89.8600 | 0.0000 | 0.0000 | 2.0000 | 247.0000 | 0.0258 |
| clients_sharing_mobile_cnt | 0.0000 | 91.4900 | 0.0000 | 0.0000 | 1.0000 | 102.0000 | 0.0233 |
| clients_sharing_phone_cnt | 0.0000 | 95.8900 | 0.0000 | 0.0000 | 0.0000 | 423.0000 | 0.0127 |
| clients_matching_empl_and_client_phone | 0.0000 | 94.8400 | 0.0000 | 0.0000 | 1.0000 | 109.0000 | 0.0249 |
| clients_same_mobile_12m | 0.0000 | 97.1900 | 0.0000 | 0.0000 | 0.0000 | 52.0000 | 0.0202 |
| has_tax_registry_record | 0.0000 | 65.0400 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0237 |
| tax_registry_records_cnt | 0.0000 | 65.0400 | 0.0000 | 0.0000 | 11.0000 | 99.0000 | 0.0274 |
| tax_registry_avg_amount | 0.0000 | 65.0400 | 0.0000 | 0.0000 | 4,053.70 | 42,500.00 | 0.0273 |
| tax_registry_max_amount | 0.0000 | 65.0400 | 0.0000 | 0.0000 | 7,039.20 | 87,115.60 | 0.0382 |
| tax_verified_income_ratio | 33.7400 | 43.5300 | 0.0000 | 0.0000 | 0.1190 | 68,944.16 | 0.0190 |
| has_debitcard | 0.0000 | 92.8900 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0284 |
| debit_records_cnt | 0.0000 | 92.8900 | 0.0000 | 0.0000 | 1.0000 | 66.0000 | 0.0287 |
| debit_avg_bal_180d | 0.0000 | 99.7400 | -308.7941 | 0.0000 | 0.0000 | 33,888.89 | 0.0226 |
| debit_avg_turnover_180d | 0.0000 | 99.3500 | -187,780.00 | 0.0000 | 0.0000 | 900,000.00 | 0.0266 |
| debit_avg_turnover_30d | 0.0000 | 99.9200 | -477.5060 | 0.0000 | 0.0000 | 390,000.00 | 0.0054 |
| debit_turnover_velocity | 99.3500 | 0.5700 | -1.6595 | 0.0000 | 6.0000 | 6.0000 | 0.0010 |

---

## 5. 合规与敏感属性隔离说明 (Fair Lending Governance)

根据信贷公平审查 (Fair Lending) 要求，以下字段已在 `feature_group_mapping.csv` 中被打标为 `Sensitive Attribute`：
- `person_gender_sensitive` (性别属性，源自 `person_1.gender_992L`)
- `person_age_band_sensitive` (年龄分组，源自 `birth_259D` 离散化分段)

> [!NOTE]
> 这些敏感字段在后续 Stage 2 ~ Stage 4 建模时**禁止**作为评分卡入模入选变量，仅保留用于模型部署前的无偏性测试 (Disparate Impact / Equal Opportunity Analysis)。

---

## 6. 阶段完成检查表与 HARD STOP

| 检查项 | 目标规范 | 执行结果 | 状态 |
|---|---|---|---|
| **全量 case_id 覆盖** | 1,526,659 行 | 1,526,659 行 | ✅ |
| **浮点精度下转** | float64 -> float32 | 全部下转 float32 | ✅ |
| **Zero-Inflation Guard** | NULLIF(maininc, 0) | PTI & DSR 严格防护 | ✅ |
| **DSR 活跃合同过滤** | 仅汇总 active 332A | 剔除已结清合同 | ✅ |
| **Train_Dev 隔离审计** | 仅 WEEK_NUM <= 64 | 零数据泄漏 | ✅ |
| **特征映射字典导出** | feature_group_mapping.csv | 已生成 (77 字段) | ✅ |
| **Mart Parquet 导出** | finguard_feature_mart.parquet | 已生成 (128.1 MB) | ✅ |

> [!IMPORTANT]
> **HARD STOP** — Stage 1B 数据集已就绪。请审查核心指标分布并确认批准后，再行进入 **Stage 2: WOE 离散化、单调性校验与 IV 筛选**。
