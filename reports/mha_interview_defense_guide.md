# MHA Senior Credit Risk Modeller — Interview Defense Guide

本指南根据 FinGuard 实际项目结果整理。回答时区分**观察结果、模拟假设与因果结论**；不要把 retrospective case study 描述成已上线系统。

## 10 个高难度问题 / Ten challenging questions

### 1. Why does the challenger beat the Champion, and is the gain credible?

**答题要点：** OOT_Valid raw M4 的 `ROC-AUC=0.8173`, `KS=0.4929`; Champion 是 `0.7768`, `0.4346`。后续 OOT_Monitor KS 分别为 `0.5006` 与 `0.4222`，方向一致。时间外结果增强可信度，但仍是单一历史数据源，不等于跨机构/未来保证；还应独立复现、评估成本与稳定性。

### 2. Why retain an interpretable scorecard if LightGBM ranks better?

Champion 有 19 个 WOE 特征，满足 `coef<0`, `p<0.05`, `VIF<5`; maximum `VIF=3.0331`。它提供可沟通的点数表、单调可解释贡献与基准控制；M4 作为增量 challenger，需另行治理其复杂性和代理变量风险。更高 KS 不会自动取代可解释性要求。

### 3. How did you prevent temporal leakage?

Fit data ended at W64. Challenger `Dev_Fit` W0–54, `Dev_Calib` W55–64; OOT_Valid W65–77 for frozen model evaluation and explicitly documented Stage 5 recalibration/cutoff fitting; OOT_Monitor W78–91 was forward-only and never used for fitting those mappings or selecting cutoffs. I disclose that OOT_Valid is no longer untouched after Stage 5 post-processing; Monitor is the later policy forward-test.

### 4. Is isotonic calibration successful?

It matched the Dev_Calib mean: actual and calibrated mean PD both `4.64%`, versus raw `3.86%`. On OOT_Valid, actual was `2.34%`, raw mean `3.02%`, isotonic mean `3.69%` (`A/E=0.6336`); raw Brier `0.0216` compared with calibrated `0.0219`. Therefore in-window fit is not transportability. Recalibration needs mature recent labels, temporal validation, and monitoring; I would not call isotonic universally superior.

### 5. What did the ablation establish?

Sequential OOT_Valid KS was M1 `0.3236`, M2 `0.4290`, M3 `0.4671`, M4 `0.4929`. This measures incremental ranking change under the chosen feature-group order. It is not causal feature value, and the Group 4 override includes unstable proxies (`has_tax_registry_record` internal PSI `1.1490`, `tax_verified_income_ratio` `0.3119`). I would test robustness with those features removed and across later cohorts.

### 6. Why is the Iso-Risk finding commercially interesting but not a causal uplift claim?

On forward OOT_Monitor, challenger approval was `75.1587%` vs Champion `70.7970%` (+6,617), bad rate `0.8832%` vs `0.9990%`, simulated ENP `142.31M` vs `133.66M`. The lending sample has labels only for historically funded cases, and ENP uses assumed economics. There is no randomized assignment or observed outcome for rejected applicants; thus this is a policy replay, not incremental causal profit.

### 7. How do you explain Iso-Approval ticket-size asymmetry?

Equal approval counts do not enforce equal exposure, requested amounts, limits, or risk mix. Under Iso-Approval the model can approve fewer large-ticket or more small-ticket accounts depending on the chosen cutoff and deterministic limit engine. I report both count and exposure/profit metrics, and would compare matched exposure as an additional sensitivity—not present approval parity as economic parity.

### 8. Why did you reject synthetic MOB curves?

`train_applprev_2` has 14.08M rows but only card/contact/status fields and hierarchy indices, no monthly DPD snapshots. `train_applprev_1` has 6.53M prior contracts and `actualdpd_943P`, supporting case-level prior-DPD severity. We therefore report prior-contract DPD migration and keep “no prior history” distinct from missing DPD/zero DPD. Cross-sectional maxima cannot reconstruct monthly transitions.

### 9. What does the stress test tell us?

S4 expected profit was `51.00M` Champion and `42.48M` Challenger; realized stress proxies `83.13M` and `109.14M`. Those proxies scale observed target rates under scenario multipliers; they are not realized recession losses or forecasts. I would replace assumptions with approved funding, servicing and recovery data, validate LGD and run sensitivity bands before capital or pricing use.

### 10. What are the key fairness and Malaysian regulatory caveats?

Sensitive features were excluded, but `Gender` was 99.26% unknown on Monitor. Under-25 AIR was `0.5988`, with observed bad rate `6.0845%` and A/E `1.2576`; AIR is a diagnostic, not a legal conclusion. BNM policy applicability and Act 400 depend on institution/product scope and statutory exemptions; project APR/DSR are assumptions. I would obtain lawful adequate data, investigate proxy pathways, and get compliance/legal review before deployment.

## 4 verified resume bullets

- Built a time-aware credit-risk pipeline over **1.53M applications** using DuckDB case-level aggregation and a strict W0–64 / W65–77 / W78–91 temporal design; kept the forward monitoring window out of model and cutoff fitting.
- Developed a **19-feature WOE Logistic scorecard** with all retained coefficients negative and significant (`p<0.05`), `VIF<5` (maximum `3.0331`), achieving OOT_Valid `ROC-AUC 0.7768` and `KS 0.4346`.
- Benchmarked a **73-feature monotonic LightGBM challenger** with four-group ablation; OOT_Valid KS increased from `0.3236` (M1) to `0.4929` (M4), and OOT_Monitor KS was `0.5006`.
- Forward-tested a Valid-frozen Iso-Risk policy on OOT_Monitor: challenger approval increased from `70.80%` to `75.16%` (+6,617 observed cases), bad rate decreased from `0.9990%` to `0.8832%`, and assumption-based simulated ENP increased by `7.95M`; explicitly documented sample-selection and non-causal limitations.

简历使用建议：保留“simulated / forward-tested / retrospective”限定词，不要将回放结果写成真实上线收益或因果 uplift。
