# Stage 4 信用决策、额度建议与 Swap Set 策略评估

本阶段仅用 `Train_Dev` (`WEEK_NUM<=64`) 确定 tier 概率边界和 approval cutoff；随后对 `OOT_Valid` (`65–77`, 82,602 笔已观察贷款)进行盲策略评价。`OOT_Monitor` (`78–91`) 未查询。Challenger 使用 Stage 3 批准的 `pd_lgbm_raw`，不使用 Dev_Calib isotonic 输出。

## Tier 边界与 Train_Dev 校准

风险分层采用固定 PD bands，并以 `Train_Dev` 的分层 observed bad rate / mean PD 审核；`Tier E` (`PD>=7%`) 为 decline。Scorecard 转换边界：

- `A/B` at PD 1%: Champion Score boundary 715.4
- `B/C` at PD 2%: Champion Score boundary 664.6
- `C/D` at PD 4%: Champion Score boundary 613.2
- `D/E` at PD 7%: Champion Score boundary 570.5

Scorecard band score boundaries are inverse PDO score cutoffs; higher scores indicate lower PD. `LightGBM` uses the same PD bands on raw M4 PD. Train_Dev calibration table:

| Model              | Risk Tier           |   Train Count |   Train Share % |   Observed Bad Rate % |   Mean Predicted PD % |
|:-------------------|:--------------------|--------------:|----------------:|----------------------:|----------------------:|
| Champion Scorecard | Tier A: Super Prime |        211503 |         16.3658 |                0.5636 |                0.7369 |
| Champion Scorecard | Tier B: Prime       |        411629 |         31.8512 |                1.2161 |                1.4524 |
| Champion Scorecard | Tier C: Standard    |        361113 |         27.9424 |                2.9622 |                2.8074 |
| Champion Scorecard | Tier D: High Risk   |        172667 |         13.3607 |                6.0423 |                5.2129 |
| Champion Scorecard | Tier E: Decline     |        135438 |         10.4800 |               11.4576 |               11.9394 |
| Raw LightGBM M4    | Tier A: Super Prime |        439888 |         34.0378 |                0.3387 |                0.5600 |
| Raw LightGBM M4    | Tier B: Prime       |        285838 |         22.1177 |                1.1667 |                1.4440 |
| Raw LightGBM M4    | Tier C: Standard    |        261597 |         20.2420 |                2.5864 |                2.8575 |
| Raw LightGBM M4    | Tier D: High Risk   |        155529 |         12.0346 |                5.2762 |                5.2640 |
| Raw LightGBM M4    | Tier E: Decline     |        149498 |         11.5679 |               15.4176 |               12.9269 |

## Tenor 与数据完整性检查

Train_Dev `credamount_770A / annuity_780A` 有效率 100.00%；median rounded tenor=12 months，P01=5，P99=29；clip-to-3 share=0.011%，clip-to-60 share=0.006%。Stage 4 does not drop missing-income rows; `maininc_215A` missing/<=100 uses tier-specific requested-amount fallback caps and `income_unverified_flag=1`.

## Hard Policy / Driver Rules

- `bureau_active_max_dpd >= 90` → policy decline.
- For positive declared income, `active_bureau_monthly_debt / maininc_215A > 0.60` → policy decline.
- `Tier E` → model risk decline. Tiers A–D income multiples are 4x/3x/2x/1x monthly income; Tier E is 0x.
- `M_behavior=1.15x` for zero observed DPD with `static_pct_inst_paid_early_3d>=0.70`; `1.00x` for other clean history; `0.75x` for minor positive DPD; `0.0x` for policy knockout.
- Missing/low income fallback caps are 80%/75%/70%/60% of requested amount in tiers A–D, multiplied by behavior; DSR is marked untestable for those rows. Tier E remains zero.
- `L_affordability` is the PV of the maximum payment under the selected Total DSR cap and tier APR. Recommended principal is `min(requested, L_risk_behavior, L_affordability)`; declined rows receive zero.

## Parametric Commercial Simulation Assumptions

Parametric Commercial Simulation Assumptions (not statutory constants): COF=4.0% p.a.; OPEX=1.5% p.a.; assumed LGD=75%; APR by tiers A/B/C/D/E=9%/12%/15%/18%/0%, capped at 18%; fixed tier risk spread = APR - COF - OPEX. Expected credit loss is `principal × PD × LGD`; interest margin is accrued on the amortizing monthly balance; no recovery timing, tax, capital, or acquisition-cost estimate is included.

## Table 1: OOT_Valid Approval / Bad-Rate / Profit Trade-off

PD cutoffs are quantiles of Train_Dev model predictions only. Actual OOT approval rate can differ due to drift, hard policy, Tier E, and zero-limit declines. `100%` is the all-risk-score acceptance baseline after these overlays. Bad reduction is relative to that model's baseline.

| Model              |   Target Approval % |   PD Cutoff (Train_Dev) |   Approval Rate % |   Approved Count |   Observed Bad Rate % |   Bad Reduction vs. 100% Baseline % |   Simulated Expected Net Profit |
|:-------------------|--------------------:|------------------------:|------------------:|-----------------:|----------------------:|------------------------------------:|--------------------------------:|
| Champion Scorecard |             50.0000 |                  0.0207 |           53.8353 |            44469 |                0.7039 |                             57.3280 |                   46927922.8209 |
| Champion Scorecard |             60.0000 |                  0.0258 |           62.0881 |            51286 |                0.8696 |                             47.2779 |                   57536244.8363 |
| Champion Scorecard |             70.0000 |                  0.0331 |           70.1048 |            57908 |                1.1104 |                             32.6824 |                   66917110.9422 |
| Champion Scorecard |             75.0000 |                  0.0385 |           74.0660 |            61180 |                1.2292 |                             25.4814 |                   70940765.1246 |
| Champion Scorecard |             80.0000 |                  0.0457 |           77.8020 |            64266 |                1.3646 |                             17.2678 |                   74809592.7565 |
| Champion Scorecard |             90.0000 |                  0.0720 |           84.6977 |            69962 |                1.6495 |                              0.0000 |                   80182609.2979 |
| Champion Scorecard |            100.0000 |                  0.7057 |           84.6977 |            69962 |                1.6495 |                              0.0000 |                   80182609.2979 |
| Raw LightGBM M4    |             50.0000 |                  0.0165 |           49.1308 |            40583 |                0.4953 |                             64.0774 |                   35279316.5176 |
| Raw LightGBM M4    |             60.0000 |                  0.0226 |           58.1935 |            48069 |                0.6116 |                             55.6394 |                   45903156.1035 |
| Raw LightGBM M4    |             70.0000 |                  0.0317 |           66.9802 |            55327 |                0.7898 |                             42.7125 |                   57550008.3069 |
| Raw LightGBM M4    |             75.0000 |                  0.0380 |           71.3130 |            58906 |                0.9337 |                             32.2798 |                   62143927.5531 |
| Raw LightGBM M4    |             80.0000 |                  0.0462 |           75.7403 |            62563 |                1.0709 |                             22.3266 |                   66834794.6324 |
| Raw LightGBM M4    |             90.0000 |                  0.0766 |           83.0646 |            68613 |                1.3787 |                              0.0000 |                   73418887.1988 |
| Raw LightGBM M4    |            100.0000 |                  0.9219 |           83.0646 |            68613 |                1.3787 |                              0.0000 |                   73418887.1988 |

## Table 2: Matched-Approval 2×2 Swap Set (75% target)

Both models approve exactly 61,952 loans (75.00% of OOT_Valid). The matched operating point is selected by each model's unlabeled OOT risk ranks among policy- and Tier-E-eligible loans; outcomes are not used to set selections. Policy knockouts are declines for both models.

| Swap Set                                    |   Loan Count |   Observed Bad Count |   Observed Bad Rate % |   Scorecard Recommended Volume |   Raw LightGBM Recommended Volume |   Recommended Volume Delta (LGBM - Scorecard) |   Scorecard Simulated ENP |   Raw LightGBM Simulated ENP |   ENP Impact (LGBM - Scorecard) |   Policy Declines in Set |
|:--------------------------------------------|-------------:|---------------------:|----------------------:|-------------------------------:|----------------------------------:|----------------------------------------------:|--------------------------:|-----------------------------:|--------------------------------:|-------------------------:|
| Both Approve                                |        57335 |                  518 |                0.9035 |                2431526674.5638 |                   2409423678.8865 |                                -22102995.6773 |             63721210.2184 |                60857669.6661 |                   -2863540.5523 |                        0 |
| Both Decline                                |        16033 |                 1027 |                6.4055 |                         0.0000 |                            0.0000 |                                        0.0000 |                    0.0000 |                       0.0000 |                          0.0000 |                     4864 |
| Swap-In (Scorecard Decline / LGBM Approve)  |         4617 |                  124 |                2.6857 |                         0.0000 |                    144214455.9206 |                                144214455.9206 |                    0.0000 |                 5118769.3930 |                    5118769.3930 |                        0 |
| Swap-Out (Scorecard Approve / LGBM Decline) |         4617 |                  264 |                5.7180 |                 206075609.1731 |                            0.0000 |                               -206075609.1731 |              8178317.0371 |                       0.0000 |                   -8178317.0371 |                        0 |

Observed results are for Known Good/Bad (KGB) funded/observed loans only. KGB Sample Selection Bias means results do not identify outcomes for historically rejected applicants; no reject inference, fabricated labels, or causal approval effect is claimed.

## Table 3: DSR Limit Sensitivity at Train_Dev-Derived 75% Cutoff

|   DSR Cap % | Model              |   Approval Rate % |   Approved Count |   Recommended Volume |   Simulated Expected Net Profit |   Verified-Income Approved Count |   DSR Compliant % of Verified-Income Approvals |   Income-Unverified Approved Count |
|------------:|:-------------------|------------------:|-----------------:|---------------------:|--------------------------------:|---------------------------------:|-----------------------------------------------:|-----------------------------------:|
|     40.0000 | Champion Scorecard |           71.6944 |            59221 |      2466740737.5092 |                   66982723.7505 |                            37008 |                                       100.0000 |                              22213 |
|     40.0000 | Raw LightGBM M4    |           68.8276 |            56853 |      2317148815.0873 |                   58638825.6633 |                            37307 |                                       100.0000 |                              19546 |
|     50.0000 | Champion Scorecard |           74.0660 |            61180 |      2609898770.4646 |                   70940765.1246 |                            38967 |                                       100.0000 |                              22213 |
|     50.0000 | Raw LightGBM M4    |           71.3130 |            58906 |      2458916279.4561 |                   62143927.5531 |                            39360 |                                       100.0000 |                              19546 |
|     60.0000 | Champion Scorecard |           74.0660 |            61180 |      2670446701.9402 |                   72483369.2090 |                            38967 |                                       100.0000 |                              22213 |
|     60.0000 | Raw LightGBM M4    |           71.3130 |            58906 |      2517294470.7340 |                   63505479.4201 |                            39360 |                                       100.0000 |                              19546 |

DSR compliance is measured only among approved rows with verified positive income. Income-unverified approved loans are separately counted and do not claim DSR compliance.

## Stage 5 Governance

Stage 3 governance flags remain: `has_tax_registry_record` internal PSI 1.149 and `tax_verified_income_ratio` PSI 0.312. They were included in the explicitly requested Stage 3 M4 proxy test, but remain flagged for Stage 5 CSI monitoring.

## Audit Summary and Artifacts

OOT_Valid rows=82,602; `case_id` unique=True; Train_Dev predictions used for all tier/approval cutoffs; OOT_Valid used only for strategy evaluation; OOT_Monitor rows read=0. `pd_lgbm_raw` is the selected Challenger PD. `APR`, COF, OPEX and LGD are parametric commercial assumptions.

Saved `data/processed/stage4_oot_valid_decisions.parquet` and figures `reports/figures/stage4_approval_bad_rate.png`, `stage4_expected_net_profit.png`, and `stage4_limit_sensitivity.png`.

**HARD STOP:** Stage 4 complete. Wait for human review before Stage 5.
