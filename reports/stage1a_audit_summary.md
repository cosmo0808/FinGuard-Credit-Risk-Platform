# FinGuard Stage 1A 审计报告 — Temporal Split Audit & Candidate Feature Discovery

> **生成时间 (Generated)**: 2026-11-16 15:27:04
> **项目根目录 (Project Root)**: `D:\data analysis\FinGuard`
> **执行脚本 (Script)**: `src/audit_stage1a.py`
> **数据规则 (Rules)**: 参见 `GEMINI.md` — Zero Leakage, 16GB RAM, Human-in-the-Loop

---

## 1. 总体数据集概况 (Overall Dataset Summary)

> `train_base.parquet` 全量统计，**未过滤**任何 split，仅用于数据完整性确认。

| 指标 (Metric) | 数值 (Value) |
|---|---|
| Total Rows | 1,526,659 |
| Unique case_id | 1,526,659 |
| Min date_decision | 2019-01-01 |
| Max date_decision | 2020-10-05 |
| Min WEEK_NUM | 0 |
| Max WEEK_NUM | 91 |
| Bad Count (target=1) | 47,994 |
| Overall Bad Rate | 3.1437% |
| Target NULL Count | 0 |

> [!IMPORTANT]
> `target_null_count = 0` — 若不为零，需在 Stage 1B 中排除 target 为空的行。

---

## 2. 时序分割审计 (Temporal Split Audit)

> 严格按 `WEEK_NUM` 切分，**零数据泄漏**：Train_Dev 边界 ≤ 64。

| split_name | n_rows | n_unique_cases | date_min | date_max | bad_count | bad_rate_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Train_Dev   (W 0-64) | 1292350 | 1292350 | 2019-01-01 | 2020-03-30 | 42846.0000 | 3.3154 |
| OOT_Valid   (W 65-77) | 82602 | 82602 | 2020-03-31 | 2020-06-29 | 1933.0000 | 2.3401 |
| OOT_Monitor (W 78-91) | 151707 | 151707 | 2020-06-30 | 2020-10-05 | 3215.0000 | 2.1192 |

> [!NOTE]
> - **Train_Dev**：所有 WOE binning、IV 计算、VIF 筛查、模型训练均在此集上进行。
> - **OOT_Valid**：仅用于校准与策略模拟，严禁回流至训练环节。
> - **OOT_Monitor**：PSI/CSI 监控与再校准，时间线最晚，模拟真实部署后监控。
> - 若有 `Out_of_Range` 行出现，需人工核查其 `WEEK_NUM` 值后决定处理方式。

---

## 3. 月度量/坏账率趋势 (Monthly Volume & Bad Rate Trend)

> 用于检测宏观/季节性飘移，以及 OOT 切分点是否与市场事件吻合。

| month | n_rows | bad_count | bad_rate_pct | split |
| --- | --- | --- | --- | --- |
| 201901 | 75529 | 1963.0000 | 2.5990 | Train_Dev |
| 201902 | 63064 | 1727.0000 | 2.7385 | Train_Dev |
| 201903 | 69147 | 1850.0000 | 2.6755 | Train_Dev |
| 201904 | 72012 | 2121.0000 | 2.9453 | Train_Dev |
| 201905 | 64594 | 2132.0000 | 3.3006 | Train_Dev |
| 201906 | 94398 | 2431.0000 | 2.5753 | Train_Dev |
| 201907 | 97566 | 2679.0000 | 2.7458 | Train_Dev |
| 201908 | 98741 | 2639.0000 | 2.6726 | Train_Dev |
| 201909 | 98706 | 3359.0000 | 3.4030 | Train_Dev |
| 201910 | 95149 | 3663.0000 | 3.8498 | Train_Dev |
| 201911 | 115845 | 4132.0000 | 3.5668 | Train_Dev |
| 201912 | 126011 | 4352.0000 | 3.4537 | Train_Dev |
| 202001 | 86750 | 3537.0000 | 4.0772 | Train_Dev |
| 202002 | 75183 | 3520.0000 | 4.6819 | Train_Dev |
| 202003 | 59655 | 2741.0000 | 4.5948 | Train_Dev |
| 202003 | 1093 | 31.0000 | 2.8362 | OOT_Valid |
| 202004 | 8289 | 179.0000 | 2.1595 | OOT_Valid |
| 202005 | 28725 | 635.0000 | 2.2106 | OOT_Valid |
| 202006 | 1467 | 40.0000 | 2.7267 | OOT_Monitor |
| 202006 | 44495 | 1088.0000 | 2.4452 | OOT_Valid |
| 202007 | 28912 | 660.0000 | 2.2828 | OOT_Monitor |
| 202008 | 50831 | 994.0000 | 1.9555 | OOT_Monitor |
| 202009 | 61905 | 1332.0000 | 2.1517 | OOT_Monitor |
| 202010 | 8592 | 189.0000 | 2.1997 | OOT_Monitor |

> [!TIP]
> 若 OOT_Monitor 期间坏账率出现断崖式变化，Stage 6 的 PSI 阈值需相应收紧（建议 PSI > 0.25 触发再建模警报）。

---

## 4. 文件清单 & Schema 概览 (File Inventory & Schema Summary)

| 文件家族 (Family) | 文件数 | 列数 | 备注 |
|---|---|---|---|
| `depth0_static` |  2 |  168 |  |
| `depth0_static_cb` |  1 |   53 |  |
| `depth1_person` |  1 |   37 |  |
| `depth1_applprev` |  2 |   41 |  |
| `depth1_bureau_a` |  4 |   79 |  |
| `depth1_tax_a` |  1 |    5 |  |
| `depth1_debitcard` |  1 |    6 |  |
| `depth2_applprev` |  1 |    6 | Schema only (depth-2) |

> [!NOTE]
> `depth2_applprev` 仅做 schema 检视，Stage 1B 才会在 DuckDB 内 GROUP BY case_id 聚合，**绝不直接 load 进 Pandas**。

---

## 5. Train_Dev 零泄漏 Null Rate（Top-15 最完整列）

> 以下 null rate **仅基于 `WEEK_NUM <= 64` 的 Train_Dev 样本**，在 DuckDB 内计算，depth-1 表已先做 `GROUP BY case_id` 聚合。


#### depth0_static
| Column | Null Rate (%) | DSR/PTI | Definition ||---|---|---|---|| `annuity_780A` | 0.00 | ✅ | Monthly annuity amount. || `annuitynextmonth_57A` | 0.00 | ✅ | Next month's amount of annuity. || `applicationcnt_361L` | 0.00 |  | Number of applications associated with the same email addres... || `applications30d_658L` | 0.00 |  | Number of applications made by the client in the last 30 day... || `applicationscnt_1086L` | 0.00 |  | Number of applications associated with the same phone number... || `applicationscnt_464L` | 0.00 |  | Number of applications made in the last 30 days by other cli... || `applicationscnt_629L` | 0.00 |  | Number of applications with the same employer in the last 7 ... || `applicationscnt_867L` | 0.00 |  | Number of applications associated with the same mobile phone... || `clientscnt12m_3712952L` | 0.00 |  | Number of clients that have used the same mobile phone as th... || `clientscnt3m_3712950L` | 0.00 |  | Number of clients who have the same mobile phone number in t... || `clientscnt6m_3712949L` | 0.00 |  | Total number of clients who have used the same mobile number... || `clientscnt_100L` | 0.00 |  | Number of applications with matching employer's phone and cl... || `clientscnt_1022L` | 0.00 |  | Number of clients sharing the same mobile phone. || `clientscnt_1071L` | 0.00 |  | Number of applications where the alternative phone number ma... || `clientscnt_1130L` | 0.00 |  | Number of applications where client's phone number matches t... |
#### depth0_static_cb
| Column | Null Rate (%) | DSR/PTI | Definition ||---|---|---|---|| `days120_123L` | 8.39 |  | Number of credit bureau queries for the last 120 days. || `days180_256L` | 8.39 |  | Number of credit bureau queries for last 180 days. || `days30_165L` | 8.39 |  | Number of credit bureau queries for the last 30 days. || `days360_512L` | 8.39 |  | Number of Credit Bureau queries for last 360 days. || `days90_310L` | 8.39 |  | Number of credit bureau queries for the last 90 days. || `firstquarter_103L` | 8.39 |  | Number of results obtained from credit bureau in the first q... || `fourthquarter_440L` | 8.39 |  | Number of results in fourth quarter. || `numberofqueries_373L` | 8.39 |  | Number of queries to credit bureau. || `secondquarter_766L` | 8.39 |  | Number of results in second quarter. || `thirdquarter_1082L` | 8.39 |  | Number of results in third quarter. || `pmtscount_423L` | 54.78 |  | Number of tax deduction payments. || `pmtssum_45A` | 54.78 |  | Sum of tax deductions for the client. || `pmtcount_693L` | 88.44 |  | Number of tax deductions. || `pmtaverage_3A` | 88.66 |  | Average of tax deductions. || `pmtaverage_4527227A` | 91.05 |  | Average of tax deductions. |
#### depth1_person
| Column | Null Rate (%) | DSR/PTI | Definition ||---|---|---|---|| `persontype_1072L` | 0.40 |  | Person type. || `personindex_1023L` | 44.55 |  | Order of the person specified on the application form. || `persontype_792L` | 44.55 |  | Person type. || `contaddr_matchlist_1032L` | 50.82 |  | Indicates whether the contact address is found in a code lis... || `contaddr_smempladdr_334L` | 50.82 |  | Indicates whether the contact address is the same as the emp... || `mainoccupationinc_384A` | 50.82 | ✅ | Amount of the main income of the client. || `safeguarantyflag_411L` | 50.82 |  | Flag indicating if client is using a flexible product with a... || `isreference_387L` | 99.48 |  | Flag indicating whether the person is a reference contact. || `childnum_185L` | 100.00 |  | Number of children of the applicant. || `remitter_829L` | 100.00 |  | Flag indicating whether the client is a remitter. |
#### depth1_applprev
| Column | Null Rate (%) | DSR/PTI | Definition ||---|---|---|---|| `isbidproduct_390L` | 0.01 |  | Flag for determining if the product is a cross-sell in previ... || `actualdpd_943P` | 0.23 |  | Days Past Due (DPD) of previous contract (actual). || `mainoccupationinc_437A` | 5.55 | ✅ | Client's main income amount in their previous application. || `credacc_credlmt_575A` | 10.32 |  | Credit card credit limit provided for previous applications. || `credamount_590A` | 10.71 | ✅ | Loan amount or card limit of previous applications. || `downpmt_134A` | 10.71 |  | Previous application downpayment amount. || `annuity_853A` | 14.04 | ✅ | Monthly annuity for previous applications. || `pmtnum_8L` | 27.88 |  | Number of payments made for the previous application. || `tenor_203L` | 27.88 |  | Number of instalments in the previous application. || `currdebt_94A` | 60.43 | ✅ | Previous application's current debt. || `outstandingdebt_522A` | 60.78 | ✅ | Amount of outstanding debt on the client's previous applicat... || `maxdpdtolerance_577P` | 69.46 |  | Maximum DPD with tolerance (on previous application/s). || `childnum_21L` | 83.04 |  | Number of children in the previous application. || `byoccupationinc_3656910L` | 95.82 |  | Applicant's income from previous applications. || `isdebitcard_527L` | 97.70 |  | Previous application flag indicating if product being applie... |
#### depth1_bureau_a
| Column | Null Rate (%) | DSR/PTI | Definition ||---|---|---|---|| `numberofoverdueinstlmax_1151L` | 99.34 |  | Maximum number of past due installments for a closed contrac... || `overdueamountmax2_398A` | 99.34 |  | Maximal overdue amount for a closed contract. || `numberofoverdueinstls_834L` | 99.37 |  | Number of past due instalments for a closed contract. || `overdueamount_31A` | 99.37 |  | Past due amount for a closed contract. || `overdueamountmax_35A` | 99.60 |  | Maximal past due amount for a closed contract. || `overdueamountmaxdatemonth_284T` | 99.60 |  | Month when the maximum past due amount occurred for a closed... || `overdueamountmaxdateyear_994T` | 99.60 |  | Year when maximum past due amount occurred for closed contra... || `dpdmax_757P` | 99.62 |  | Maximum days past due for a closed contract. || `dpdmaxdatemonth_442T` | 99.62 |  | Max DPD occurrence month for terminated contracts from credi... || `dpdmaxdateyear_896T` | 99.62 |  | Year of maximum Days Past Due of closed contract obtained fr... || `monthlyinstlamount_674A` | 99.72 | ✅ | Monthly amount of instalment payment on a closed contract. || `numberofinstls_229L` | 99.80 |  | Number of instalments on closed contract. || `numberofoutstandinstls_520L` | 99.80 |  | Number of outstanding instalment for closed contract. || `outstandingamount_354A` | 99.80 | ✅ | Outstanding amount for closed credit contract in credit bure... || `totalamount_6A` | 99.80 |  | Total amount of closed contracts. |
#### depth1_tax_a
| Column | Null Rate (%) | DSR/PTI | Definition ||---|---|---|---|| `amount_4527230A` | 0.00 |  | Tax deductions amount tracked by the government registry. |
#### depth1_debitcard
| Column | Null Rate (%) | DSR/PTI | Definition ||---|---|---|---|| `last180dayaveragebalance_704A` | 94.04 |  | Average balance on debit card in the last 180 days. || `last180dayturnover_1134A` | 94.56 |  | Debit card's turnover within the last 180 days. || `last30dayturnover_651A` | 94.56 |  | Debit card turnover for the last 30 days. |

---

## 6. 候选特征分组（Candidate Feature Groups）

> 基于列名关键词进行域标注，每组展示最多 15 列（Stage 2 IV 筛选后精简至 10–15 个）。
> 标注方法：纯字符串匹配，不代表最终特征选择结论，需人工审核。


#### Traditional Credit (150 个候选列)
| Column | Dtype | Null Rate (Train_Dev, %) | DSR/PTI | Definition ||---|---|---|---|---|| `actualdpd_943P` | DOUBLE | 0.23 |  | Days Past Due (DPD) of previous contract (actual). || `annualeffectiverate_199L` | DOUBLE | 100.00 |  | Interest rate of the closed contracts. || `annualeffectiverate_63L` | DOUBLE | 100.00 |  | Interest rate for the active contracts. || `avgdpdtolclosure24_3658938P` | DOUBLE | 31.85 |  | Average DPD (days past due) with tolerance within the p... || `avgoutstandbalancel6m_4187114A` | DOUBLE | 56.87 |  | Average outstanding balance of applicant for the last 6... || `birthdate_574D` | VARCHAR | — |  | Client's date of birth (credit bureau data). || `cardtype_51L` | VARCHAR | — |  | Type of credit card. || `classificationofcontr_13M` | VARCHAR | — |  | Classificiation of the active contract. || `classificationofcontr_400M` | VARCHAR | — |  | Classificiation of the closed contract. || `contractssum_5085716L` | DOUBLE | 100.00 |  | Total sum of values of contracts retrieved from externa... || `contractst_545M` | VARCHAR | — |  | Contract status. || `contractst_964M` | VARCHAR | — |  | Contract status of terminated credit contract. || `contractsum_5085717L` | DOUBLE | 100.00 |  | Sum of other contract values. || `credacc_actualbalance_314A` | DOUBLE | 98.47 | ✅ | Actual balance on credit account. || `credacc_cards_status_52L` | VARCHAR | — |  | Card status of the previous credit account. |
#### Historical Repayment (91 个候选列)
| Column | Dtype | Null Rate (Train_Dev, %) | DSR/PTI | Definition ||---|---|---|---|---|| `actualdpd_943P` | DOUBLE | 0.23 |  | Days Past Due (DPD) of previous contract (actual). || `actualdpdtolerance_344P` | DOUBLE | 28.58 |  | DPD of client with tolerance. || `avgdbddpdlast24m_3658932P` | DOUBLE | 41.39 |  | Average days past or before due of payment during the l... || `avgdbddpdlast3m_4187120P` | DOUBLE | 63.51 |  | Average days past or before due of payment during the l... || `avgdbdtollast24m_4525197P` | DOUBLE | 69.22 |  | Average days of payment before due date within the last... || `avgdpdtolclosure24_3658938P` | DOUBLE | 31.85 |  | Average DPD (days past due) with tolerance within the p... || `avgmaxdpdlast9m_3716943P` | DOUBLE | 50.42 |  | Average Days Past Due (DPD) of the client in last 9 mon... || `avgpmtlast12m_4525200A` | DOUBLE | 72.19 |  | Average of payments made by the client in the last 12 m... || `cntincpaycont9m_3716944L` | DOUBLE | 31.08 |  | Number of incoming payments in the past 9 months. || `cntpmts24_3658933L` | DOUBLE | 31.51 |  | Number of months with any incoming payment in last 24 m... || `datelastinstal40dpd_247D` | VARCHAR | — |  | Date of last instalment that was more than 40 days past... || `daysoverduetolerancedd_3976961L` | DOUBLE | 30.90 |  | Number of days that past after the due date (with toler... || `debtoverdue_47A` | DOUBLE | 100.00 |  | Amount that is currently past due on a client's existin... || `downpmt_116A` | DOUBLE | 0.00 |  | Amount of downpayment. || `downpmt_134A` | DOUBLE | 10.71 |  | Previous application downpayment amount. |
#### Application Demographic (68 个候选列)
| Column | Dtype | Null Rate (Train_Dev, %) | DSR/PTI | Definition ||---|---|---|---|---|| `applicationscnt_464L` | DOUBLE | 0.00 |  | Number of applications made in the last 30 days by othe... || `applicationscnt_629L` | DOUBLE | 0.00 |  | Number of applications with the same employer in the la... || `avgdbddpdlast24m_3658932P` | DOUBLE | 41.39 |  | Average days past or before due of payment during the l... || `avgdbddpdlast3m_4187120P` | DOUBLE | 63.51 |  | Average days past or before due of payment during the l... || `avgdbdtollast24m_4525197P` | DOUBLE | 69.22 |  | Average days of payment before due date within the last... || `avgdpdtolclosure24_3658938P` | DOUBLE | 31.85 |  | Average DPD (days past due) with tolerance within the p... || `avginstallast24m_3658937A` | DOUBLE | 42.14 |  | Average instalments paid by the client over the past 24... || `avglnamtstart24m_4525187A` | DOUBLE | 89.90 |  | Average loan amount in the last 24 months. || `avgmaxdpdlast9m_3716943P` | DOUBLE | 50.42 |  | Average Days Past Due (DPD) of the client in last 9 mon... || `avgoutstandbalancel6m_4187114A` | DOUBLE | 56.87 |  | Average outstanding balance of applicant for the last 6... || `avgpmtlast12m_4525200A` | DOUBLE | 72.19 |  | Average of payments made by the client in the last 12 m... || `birth_259D` | VARCHAR | — |  | Date of birth of the person. || `birthdate_574D` | VARCHAR | — |  | Client's date of birth (credit bureau data). || `birthdate_87D` | VARCHAR | — |  | Birth date of the person. || `byoccupationinc_3656910L` | DOUBLE | 95.82 |  | Applicant's income from previous applications. |
#### Alternative NonTrad (72 个候选列)
| Column | Dtype | Null Rate (Train_Dev, %) | DSR/PTI | Definition ||---|---|---|---|---|| `amount_4527230A` | DOUBLE | 0.00 |  | Tax deductions amount tracked by the government registr... || `applicationcnt_361L` | DOUBLE | 0.00 |  | Number of applications associated with the same email a... || `applicationscnt_1086L` | DOUBLE | 0.00 |  | Number of applications associated with the same phone n... || `applicationscnt_867L` | DOUBLE | 0.00 |  | Number of applications associated with the same mobile ... || `assignmentdate_238D` | VARCHAR | — |  | Tax authority data - date of assignment. || `assignmentdate_4527235D` | VARCHAR | — |  | Tax authority data - Date of assignment. || `assignmentdate_4955616D` | VARCHAR | — |  | Tax authority assignment date. || `cacccardblochreas_147M` | VARCHAR | — |  | Card blocking reason. || `cardtype_51L` | VARCHAR | — |  | Type of credit card. || `clientscnt12m_3712952L` | DOUBLE | 0.00 |  | Number of clients that have used the same mobile phone ... || `clientscnt3m_3712950L` | DOUBLE | 0.00 |  | Number of clients who have the same mobile phone number... || `clientscnt6m_3712949L` | DOUBLE | 0.00 |  | Total number of clients who have used the same mobile n... || `clientscnt_100L` | DOUBLE | 0.00 |  | Number of applications with matching employer's phone a... || `clientscnt_1022L` | DOUBLE | 0.00 |  | Number of clients sharing the same mobile phone. || `clientscnt_1071L` | DOUBLE | 0.00 |  | Number of applications where the alternative phone numb... |

---

## 7. Total DSR / PTI 候选列核查 (DSR/PTI Column Check)

> **PTI** = 新贷款月供 / 收入；**Total DSR** = (现有在贷月供 + 新月供) / 收入。
> 以下列是区分 PTI 与 Total DSR 所需的原始字段候选，需在 Stage 1B 人工确认准确的列名。

| Column | Source Family | Dtype | Null Rate (Train_Dev, %) | Definition |
|---|---|---|---|---|
| `annuity_780A` | depth0_static | DOUBLE | 0.00 | Monthly annuity amount. |
| `annuity_853A` | depth1_applprev | DOUBLE | 14.04 | Monthly annuity for previous applications. |
| `annuitynextmonth_57A` | depth0_static | DOUBLE | 0.00 | Next month's amount of annuity. |
| `credacc_actualbalance_314A` | depth1_applprev | DOUBLE | 98.47 | Actual balance on credit account. |
| `credamount_590A` | depth1_applprev | DOUBLE | 10.71 | Loan amount or card limit of previous applications. |
| `credamount_770A` | depth0_static | DOUBLE | 0.00 | Loan amount or credit card limit. |
| `currdebt_22A` | depth0_static | DOUBLE | 0.00 | Current debt amount of the client. |
| `currdebt_94A` | depth1_applprev | DOUBLE | 60.43 | Previous application's current debt. |
| `currdebtcredtyperange_828A` | depth0_static | DOUBLE | 0.00 | Current amount of debt of the applicant. |
| `disbursedcredamount_1113A` | depth0_static | DOUBLE | 0.00 | Disbursed credit amount after consolidation. |
| `incometype_1044T` | depth1_person | VARCHAR | — | Type of income of the person |
| `instlamount_768A` | depth1_bureau_a | DOUBLE | 100.00 | Instalment amount for the active contract in credit bur... |
| `instlamount_852A` | depth1_bureau_a | DOUBLE | 100.00 | Instalment amount for closed contract. |
| `lastapprcredamount_781A` | depth0_static | DOUBLE | 30.19 | Credit amount from the client's last application. |
| `lastotherlnsexpense_631A` | depth0_static | DOUBLE | 99.79 | Monthly expenses on other loans from the last applicati... |
| `lastrejectcredamount_222A` | depth0_static | DOUBLE | 51.61 | Credit amount on last rejected application. |
| `maininc_215A` | depth0_static | DOUBLE | — | Client's primary income amount. |
| `mainoccupationinc_384A` | depth1_person | DOUBLE | 50.82 | Amount of the main income of the client. |
| `mainoccupationinc_437A` | depth1_applprev | DOUBLE | 5.55 | Client's main income amount in their previous applicati... |
| `maxannuity_159A` | depth0_static | DOUBLE | — | Maximum annuity previously obtained by client. |
| `maxannuity_4075009A` | depth0_static | DOUBLE | — | Maximal annuity offered to the client in the current ap... |
| `monthlyinstlamount_332A` | depth1_bureau_a | DOUBLE | 100.00 | Monthly instalment amount for active contract. |
| `monthlyinstlamount_674A` | depth1_bureau_a | DOUBLE | 99.72 | Monthly amount of instalment payment on a closed contra... |
| `monthsannuity_845L` | depth0_static | DOUBLE | — | Monthly annuity amount for the applicant. |
| `outstandingamount_354A` | depth1_bureau_a | DOUBLE | 99.80 | Outstanding amount for closed credit contract in credit... |
| `outstandingamount_362A` | depth1_bureau_a | DOUBLE | 100.00 | Active contract's outstanding amount. |
| `outstandingdebt_522A` | depth1_applprev | DOUBLE | 60.78 | Amount of outstanding debt on the client's previous app... |
| `totaldebt_9A` | depth0_static | DOUBLE | — | Total amount of debt. |
| `totaldebtoverduevalue_178A` | depth1_bureau_a | DOUBLE | 100.00 | Total amount of past due debt on active contracts. |
| `totaldebtoverduevalue_718A` | depth1_bureau_a | DOUBLE | 100.00 | Total overdue debt amount for closed credit contracts. |
| `totaloutstanddebtvalue_39A` | depth1_bureau_a | DOUBLE | 100.00 | Total outstanding debt for active contracts in the cred... |
| `totaloutstanddebtvalue_668A` | depth1_bureau_a | DOUBLE | 100.00 | Total outstanding debt for the closed contracts in the ... |
| `totinstallast1m_4525188A` | depth0_static | DOUBLE | — | Total amount of monthly instalments paid in the previou... |

> [!CAUTION]
> 在 Stage 1B 中，`AMT_ANNUITY`（新月供）与 `credit_sum_debt`（在贷余额）这两个字段的来源表必须经人工确认后方可纳入特征工程 mart，防止混淆信贷局数据的 outstanding balance 与新申请月供。

---

## 8. 阶段审计摘要表 (Stage 1A Audit Summary Table)

| 检查项 (Check) | 结果 (Result) | 状态 (Status) |
|---|---|---|
| DuckDB memory_limit=6GB | 已设置 | ✅ |
| DuckDB threads=4 | 已设置 | ✅ |
| train_base.parquet 可读 | 1,526,659 行 | ✅ |
| WEEK_NUM 范围 | 0 – 91 | ✅ |
| Target NULL | 0 行 | ✅ |
| Temporal split 覆盖率 | 见第2节 | ✅ |
| OOT 数据未触碰训练 | WEEK_NUM 过滤已实施 | ✅ |
| feature_definitions.csv | 已加载 465 条 | ✅ |
| Depth-1/2 未直接 load Pandas | 全程 DuckDB GROUP BY | ✅ |
| DSR/PTI 候选列已识别 | 33 列 | ✅ |

---

## 9. 下一步指引 (Next Steps — Pending Human Approval)

> [!IMPORTANT]
> **HARD STOP** — 请审查以上所有内容，确认无误后方可进入 Stage 1B。

**人工审核要点 (Human Review Checklist)**：
1. ✅/❌ Temporal split 三段样本量是否与预期比例吻合（建议 Train ~70%，OOT 合计 ~30%）？
2. ✅/❌ 月度坏账率趋势是否有异常跳变（若有，记录周数与可能原因）？
3. ✅/❌ `target_null_count` 是否为 0？若不为 0，确认排除策略。
4. ✅/❌ DSR/PTI 候选列是否正确识别了 `AMT_ANNUITY`、`AMT_CREDIT` 以及信贷局在贷月供字段？
5. ✅/❌ 是否有需要额外纳入或剔除的 feature family？
6. ✅/❌ feature_definitions.csv 中的 feature 名称是否与实际 Parquet schema 列名一致？

**待确认后进入 Stage 1B**：DuckDB SQL 特征聚合 mart（仅 Train_Dev）。
