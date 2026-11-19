import os
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".matplotlib"))
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
OUT = DATA / "stage4_oot_valid_decisions.parquet"
MART = DATA / "finguard_feature_mart.parquet"
PRED = DATA / "stage3_predictions.parquet"

TARGET_RATES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]
PD_EDGES = [0.01, 0.02, 0.04, 0.07]
TIERS = ["Tier A: Super Prime", "Tier B: Prime", "Tier C: Standard", "Tier D: High Risk", "Tier E: Decline"]
COF, OPEX, LGD, APR_CAP = 0.04, 0.015, 0.75, 0.18
APR = {TIERS[0]: 0.09, TIERS[1]: 0.12, TIERS[2]: 0.15, TIERS[3]: 0.18, TIERS[4]: 0.0}
INCOME_MULTIPLE = {TIERS[0]: 4.0, TIERS[1]: 3.0, TIERS[2]: 2.0, TIERS[3]: 1.0, TIERS[4]: 0.0}
UNVERIFIED_FRACTION = {TIERS[0]: 0.80, TIERS[1]: 0.75, TIERS[2]: 0.70, TIERS[3]: 0.60, TIERS[4]: 0.0}


def say(s):
    print(s, flush=True)


def tier(pd_values):
    return np.select(
        [pd_values < 0.01, pd_values < 0.02, pd_values < 0.04, pd_values < 0.07],
        TIERS[:4], default=TIERS[4])


def score_at_pd(p):
    factor = 50.0 / np.log(2.0)
    return 600.0 + factor * np.log(0.05) - factor * np.log(p / (1.0 - p))


def pv_annuity(payment, monthly_rate, term):
    payment = np.maximum(np.asarray(payment, float), 0.0)
    monthly_rate = np.maximum(np.asarray(monthly_rate, float), 0.0)
    term = np.maximum(np.asarray(term, int), 1)
    result = np.empty_like(payment)
    zero = monthly_rate < 1e-12
    result[zero] = payment[zero] * term[zero]
    result[~zero] = payment[~zero] * (1.0 - (1.0 + monthly_rate[~zero]) ** (-term[~zero])) / monthly_rate[~zero]
    return result


def loan_limits(df, pd_values, tier_values, dsr_cap):
    requested = np.nan_to_num(df.credamount_770A.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0).clip(0)
    annuity = np.nan_to_num(df.annuity_780A.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0).clip(0)
    income = df.maininc_215A.to_numpy(float)
    debt = np.nan_to_num(df.active_bureau_monthly_debt.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0).clip(0)
    verified = np.isfinite(income) & (income > 100.0)
    income_safe = np.where(verified, income, 0.0)
    term = np.clip(np.rint(np.divide(requested, annuity, out=np.full_like(requested, 12.0), where=annuity > 0)), 3, 60).astype(int)
    rates = np.array([APR[t] for t in tier_values], float)
    multiples = np.array([INCOME_MULTIPLE[t] for t in tier_values], float)
    fallback = np.array([UNVERIFIED_FRACTION[t] for t in tier_values], float)
    behavior_dpd = df.bureau_active_max_dpd.to_numpy(float)
    static_dpd = df.static_avg_dpd_tol_closure24m.to_numpy(float)
    early = df.static_pct_inst_paid_early_3d.to_numpy(float)
    any_dpd = (np.nan_to_num(behavior_dpd, nan=0.0) > 0) | (np.nan_to_num(static_dpd, nan=0.0) > 0)
    clean = ~any_dpd
    mult = np.where(any_dpd, 0.75, 1.00)
    mult = np.where(clean & np.isfinite(early) & (early >= 0.70), 1.15, mult)
    debt_ratio = np.divide(debt, income_safe, out=np.zeros_like(debt), where=income_safe > 0)
    policy = (np.nan_to_num(behavior_dpd, nan=-1.0) >= 90) | (verified & (debt_ratio > 0.60))
    mult = np.where(policy | (tier_values == TIERS[4]), 0.0, mult)

    risk_income = income_safe * multiples * mult
    risk_fallback = requested * fallback * mult
    risk_limit = np.where(verified, risk_income, risk_fallback)
    max_new_payment = np.maximum(0.0, dsr_cap * income_safe - debt)
    affordable_income = pv_annuity(max_new_payment, rates / 12.0, term)
    affordable_fallback = requested * fallback
    affordability_limit = np.where(verified, affordable_income, affordable_fallback)
    raw_limit = np.minimum(requested, np.minimum(risk_limit, affordability_limit))
    raw_limit = np.nan_to_num(raw_limit, nan=0.0, posinf=0.0, neginf=0.0).clip(0)
    raw_limit = np.where(policy | (tier_values == TIERS[4]), 0.0, raw_limit)

    pmt = np.zeros_like(raw_limit)
    positive_rate = rates > 0
    pmt[positive_rate] = raw_limit[positive_rate] * (rates[positive_rate] / 12) / (1 - (1 + rates[positive_rate] / 12) ** (-term[positive_rate]))
    pmt[~positive_rate] = raw_limit[~positive_rate] / term[~positive_rate]
    realized_dsr = np.divide(debt + pmt, income_safe, out=np.full_like(debt, np.nan), where=income_safe > 0)
    return {
        "requested": requested, "term": term, "income_unverified": (~verified).astype(np.int8),
        "behavior_multiplier": mult, "apr": rates, "policy_decline": policy,
        "policy_reason": np.select(
            [np.nan_to_num(behavior_dpd, nan=-1.0) >= 90, verified & (debt_ratio > .60)],
            ["Severe active delinquency: bureau_active_max_dpd >= 90", "Existing debt / income > 0.60"], default=""),
        "max_new_payment": max_new_payment, "risk_limit": risk_limit,
        "affordability_limit": affordability_limit, "limit": raw_limit,
        "monthly_payment": pmt, "realized_dsr": realized_dsr, "income_verified": verified,
        "tenor_unverified": ((annuity <= 0) | ~np.isfinite(annuity)).astype(np.int8),
    }


def expected_profit(principal, pd_values, apr_values, terms):
    principal = np.maximum(np.nan_to_num(np.asarray(principal, float), nan=0.0), 0.0)
    pds = np.clip(np.nan_to_num(np.asarray(pd_values, float), nan=0.0), 0.0, 1.0)
    aprs = np.asarray(apr_values, float)
    terms = np.asarray(terms, int)
    r = aprs / 12.0
    payment = np.zeros_like(principal)
    nonzero = r > 1e-12
    payment[nonzero] = principal[nonzero] * r[nonzero] / (1 - (1 + r[nonzero]) ** (-terms[nonzero]))
    payment[~nonzero] = np.divide(principal[~nonzero], terms[~nonzero], out=np.zeros_like(principal[~nonzero]), where=terms[~nonzero] > 0)
    balance = principal.copy()
    balance_sum = np.zeros_like(principal)
    for month in range(60):
        active = (month < terms) & (principal > 0)
        balance_sum[active] += balance[active]
        balance[active] = np.maximum(0.0, balance[active] * (1.0 + r[active]) - payment[active])
    net_interest_margin = np.maximum(aprs - COF - OPEX, 0.0) / 12.0 * balance_sum
    expected_loss = principal * pds * LGD
    return net_interest_margin - expected_loss


def strategy_summary(y, approve, enp, name, target_rate, cutoff, baseline_bad_rate):
    n = int(approve.sum())
    bad = int(y[approve].sum())
    rate = bad / n if n else np.nan
    return {"Model": name, "Target Approval %": target_rate * 100,
            "PD Cutoff (Train_Dev)": cutoff, "Approval Rate %": 100.0 * n / len(y),
            "Approved Count": n, "Observed Bad Rate %": 100.0 * rate,
            "Bad Reduction vs. 100% Baseline %": 100.0 * (baseline_bad_rate - rate) / baseline_bad_rate if baseline_bad_rate else np.nan,
            "Simulated Expected Net Profit": float(enp[approve].sum())}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(ROOT / 'duckdb_temp').as_posix()}'")

    # Train_Dev tenor and tier checks are performed before reading any OOT feature rows.
    fields = ["case_id", "WEEK_NUM", "target", "split_name", "scorecard_score", "pd_scorecard", "pd_lgbm_raw"]
    pred_cols = ", ".join('p."' + c + '"' for c in fields)
    train_pred = con.execute(f"SELECT {pred_cols} FROM read_parquet('{PRED.as_posix()}') p WHERE p.split_name='Train_Dev' AND p.WEEK_NUM<=64").fetchdf()
    train_raw_cols = ["credamount_770A", "annuity_780A"]
    rawq = ", ".join('m."' + c + '"' for c in train_raw_cols)
    tenor_train = con.execute(f"SELECT {rawq} FROM read_parquet('{MART.as_posix()}') m WHERE m.WEEK_NUM<=64").fetchdf()
    valid_tenor = tenor_train.annuity_780A.gt(0) & tenor_train.credamount_770A.notna()
    raw_tenor = np.rint(tenor_train.loc[valid_tenor, "credamount_770A"] / tenor_train.loc[valid_tenor, "annuity_780A"])
    tenor_audit = {"valid_count": int(valid_tenor.sum()), "valid_pct": float(100 * valid_tenor.mean()),
                   "median_raw": float(raw_tenor.median()), "p01_raw": float(raw_tenor.quantile(.01)),
                   "p99_raw": float(raw_tenor.quantile(.99)), "clipped_below_3_pct": float(100 * (raw_tenor < 3).mean()),
                   "clipped_above_60_pct": float(100 * (raw_tenor > 60).mean())}
    say(f"Train_Dev implied tenor check: {tenor_audit}")

    train_tiers = {"Champion Scorecard": tier(train_pred.pd_scorecard.to_numpy(float)),
                   "Raw LightGBM M4": tier(train_pred.pd_lgbm_raw.to_numpy(float))}
    tier_rows = []
    for model, tv in train_tiers.items():
        for t in TIERS:
            mask = tv == t
            if not mask.any(): continue
            yt = train_pred.target.to_numpy(np.int8)[mask]
            pdv = train_pred.pd_scorecard.to_numpy(float)[mask] if model == "Champion Scorecard" else train_pred.pd_lgbm_raw.to_numpy(float)[mask]
            tier_rows.append({"Model": model, "Risk Tier": t, "Train Count": int(mask.sum()),
                              "Train Share %": 100 * float(mask.mean()), "Observed Bad Rate %": 100 * float(yt.mean()),
                              "Mean Predicted PD %": 100 * float(pdv.mean())})
    tier_table = pd.DataFrame(tier_rows)
    threshold_rows = []
    score_cuts = {"<1%": score_at_pd(.01), "<2%": score_at_pd(.02), "<4%": score_at_pd(.04), "<7%": score_at_pd(.07)}
    for label, cut in score_cuts.items(): threshold_rows.append({"PD Boundary": label, "Champion Score Boundary": cut})
    score_thresholds = pd.DataFrame(threshold_rows)

    # OOT_Valid only: exact 65-77 guard. OOT_Monitor is not selected or queried.
    raw_cols = ["credamount_770A", "annuity_780A", "maininc_215A", "active_bureau_monthly_debt", "pti",
                "total_dsr_proxy", "bureau_active_max_dpd", "static_avg_dpd_tol_closure24m", "static_pct_inst_paid_early_3d"]
    rawq = ", ".join('m."' + c + '"' for c in raw_cols)
    oot = con.execute(f"SELECT p.*, {rawq} FROM read_parquet('{PRED.as_posix()}') p JOIN read_parquet('{MART.as_posix()}') m USING(case_id, WEEK_NUM) WHERE p.split_name='OOT_Valid' AND p.WEEK_NUM BETWEEN 65 AND 77 AND m.WEEK_NUM BETWEEN 65 AND 77").fetchdf()
    if len(oot) != 82602 or (oot.WEEK_NUM.min() < 65 or oot.WEEK_NUM.max() > 77):
        raise ValueError(f"OOT_Valid guard failed: rows={len(oot)}, weeks={oot.WEEK_NUM.min()}-{oot.WEEK_NUM.max()}")
    n = len(oot); y = oot.target.to_numpy(np.int8)
    pd_values = {"Champion Scorecard": oot.pd_scorecard.to_numpy(float), "Raw LightGBM M4": oot.pd_lgbm_raw.to_numpy(float)}
    tier_values = {k: tier(v) for k, v in pd_values.items()}
    policy_dpd = oot.bureau_active_max_dpd.to_numpy(float)
    income = oot.maininc_215A.to_numpy(float)
    debt = np.nan_to_num(oot.active_bureau_monthly_debt.to_numpy(float), nan=0.0)
    debt_ratio = np.divide(debt, income, out=np.full(n, np.nan), where=np.isfinite(income) & (income > 0))
    policy_decline = (np.nan_to_num(policy_dpd, nan=-1.0) >= 90) | ((income > 0) & np.isfinite(income) & (debt_ratio > .60))
    policy_reason = np.select([np.nan_to_num(policy_dpd, nan=-1.0) >= 90, (income > 0) & np.isfinite(income) & (debt_ratio > .60)],
                              ["Severe active delinquency: bureau_active_max_dpd >= 90", "Existing debt / income > 0.60"], default="")
    results = {}
    for model in pd_values:
        results[model] = loan_limits(oot, pd_values[model], tier_values[model], .50)
        results[model]["pd"] = pd_values[model]
        results[model]["tier"] = tier_values[model]
        results[model]["enp"] = expected_profit(results[model]["limit"], pd_values[model], results[model]["apr"], results[model]["term"])

    # Approval-rate cutoffs are fitted on Train_Dev prediction distributions only.
    train_pd = {"Champion Scorecard": train_pred.pd_scorecard.to_numpy(float),
                "Raw LightGBM M4": train_pred.pd_lgbm_raw.to_numpy(float)}
    cutoffs = {model: {rate: float(np.quantile(values, rate, method="higher")) for rate in TARGET_RATES}
               for model, values in train_pd.items()}
    trade_rows = []
    for model in pd_values:
        eligible = (~policy_decline) & (tier_values[model] != TIERS[4]) & (results[model]["limit"] > 0)
        approvals = {}
        for rate in TARGET_RATES:
            approve = eligible & (pd_values[model] <= cutoffs[model][rate])
            approvals[rate] = approve
        baseline_n = approvals[1.0].sum()
        baseline_bad = y[approvals[1.0]].mean() if baseline_n else np.nan
        for rate in TARGET_RATES:
            trade_rows.append(strategy_summary(y, approvals[rate], results[model]["enp"], model,
                                               rate, cutoffs[model][rate], baseline_bad))
        results[model]["approval_75_train"] = approvals[.75]
        results[model]["limit_75_train"] = np.where(approvals[.75], results[model]["limit"], 0.0)
    trade = pd.DataFrame(trade_rows)

    # Matched 75% acceptance is set using risk ranks and policy eligibility, without using OOT outcomes.
    eligible_sc = (~policy_decline) & (tier_values["Champion Scorecard"] != TIERS[4]) & (results["Champion Scorecard"]["limit"] > 0)
    eligible_lgb = (~policy_decline) & (tier_values["Raw LightGBM M4"] != TIERS[4]) & (results["Raw LightGBM M4"]["limit"] > 0)
    requested_match = int(round(.75 * n))
    matched_count = min(requested_match, int(eligible_sc.sum()), int(eligible_lgb.sum()))
    def top_k(mask, pdv, k):
        idx = np.flatnonzero(mask)
        order = np.lexsort((oot.case_id.to_numpy()[idx], pdv[idx]))
        selected = np.zeros(n, dtype=bool)
        selected[idx[order[:k]]] = True
        return selected
    approve_sc_matched = top_k(eligible_sc, pd_values["Champion Scorecard"], matched_count)
    approve_lgb_matched = top_k(eligible_lgb, pd_values["Raw LightGBM M4"], matched_count)
    decision_code = np.select([approve_sc_matched & approve_lgb_matched,
                               ~approve_sc_matched & ~approve_lgb_matched,
                               ~approve_sc_matched & approve_lgb_matched],
                              ["Both Approve", "Both Decline", "Swap-In (Scorecard Decline / LGBM Approve)"],
                              default="Swap-Out (Scorecard Approve / LGBM Decline)")
    sc_match_limit = np.where(approve_sc_matched, results["Champion Scorecard"]["limit"], 0.0)
    lgb_match_limit = np.where(approve_lgb_matched, results["Raw LightGBM M4"]["limit"], 0.0)
    sc_match_enp = expected_profit(sc_match_limit, pd_values["Champion Scorecard"], results["Champion Scorecard"]["apr"], results["Champion Scorecard"]["term"])
    lgb_match_enp = expected_profit(lgb_match_limit, pd_values["Raw LightGBM M4"], results["Raw LightGBM M4"]["apr"], results["Raw LightGBM M4"]["term"])
    swap_rows = []
    for label in ["Both Approve", "Both Decline", "Swap-In (Scorecard Decline / LGBM Approve)", "Swap-Out (Scorecard Approve / LGBM Decline)"]:
        mask = decision_code == label; count = int(mask.sum()); bads = int(y[mask].sum())
        swap_rows.append({"Swap Set": label, "Loan Count": count, "Observed Bad Count": bads,
                          "Observed Bad Rate %": 100 * bads / count if count else np.nan,
                          "Scorecard Recommended Volume": float(sc_match_limit[mask].sum()),
                          "Raw LightGBM Recommended Volume": float(lgb_match_limit[mask].sum()),
                          "Recommended Volume Delta (LGBM - Scorecard)": float(lgb_match_limit[mask].sum() - sc_match_limit[mask].sum()),
                          "Scorecard Simulated ENP": float(sc_match_enp[mask].sum()),
                          "Raw LightGBM Simulated ENP": float(lgb_match_enp[mask].sum()),
                          "ENP Impact (LGBM - Scorecard)": float(lgb_match_enp[mask].sum() - sc_match_enp[mask].sum()),
                          "Policy Declines in Set": int((mask & policy_decline).sum())})
    swap = pd.DataFrame(swap_rows)

    # DSR limit sensitivity, holding the Train_Dev-derived 75% approval thresholds fixed.
    sensitivity_rows = []
    for cap in [.40, .50, .60]:
        for model in pd_values:
            lim = loan_limits(oot, pd_values[model], tier_values[model], cap)
            approval = results[model]["approval_75_train"] & (lim["limit"] > 0)
            principal = np.where(approval, lim["limit"], 0.0)
            enp = expected_profit(principal, pd_values[model], lim["apr"], lim["term"])
            tested = approval & lim["income_verified"]
            compliant = tested & np.isfinite(lim["realized_dsr"]) & (lim["realized_dsr"] <= cap + 1e-8)
            sensitivity_rows.append({"DSR Cap %": cap * 100, "Model": model,
                "Approval Rate %": 100 * approval.mean(), "Approved Count": int(approval.sum()),
                "Recommended Volume": float(principal.sum()), "Simulated Expected Net Profit": float(enp.sum()),
                "Verified-Income Approved Count": int(tested.sum()),
                "DSR Compliant % of Verified-Income Approvals": 100 * compliant.sum() / tested.sum() if tested.any() else np.nan,
                "Income-Unverified Approved Count": int((approval & (lim["income_unverified"] == 1)).sum())})
    sensitivity = pd.DataFrame(sensitivity_rows)

    # Export decision-level OOT_Valid only, preserving both Train_Dev cutoff and matched-approval decisions.
    out = pd.DataFrame({"case_id": oot.case_id, "WEEK_NUM": oot.WEEK_NUM, "split_name": oot.split_name,
        "target": y, "scorecard_score": oot.scorecard_score, "pd_scorecard": pd_values["Champion Scorecard"],
        "pd_lgbm_raw": pd_values["Raw LightGBM M4"], "scorecard_tier": tier_values["Champion Scorecard"],
        "lgbm_tier": tier_values["Raw LightGBM M4"], "policy_decline": policy_decline.astype(np.int8),
        "policy_reason": policy_reason, "credamount_770A": oot.credamount_770A,
        "requested_amount": results["Champion Scorecard"]["requested"],
        "annuity_780A": oot.annuity_780A, "maininc_215A": oot.maininc_215A,
        "active_bureau_monthly_debt": oot.active_bureau_monthly_debt, "pti": oot.pti,
        "total_dsr_proxy": oot.total_dsr_proxy, "bureau_active_max_dpd": oot.bureau_active_max_dpd,
        "static_avg_dpd_tol_closure24m": oot.static_avg_dpd_tol_closure24m,
        "static_pct_inst_paid_early_3d": oot.static_pct_inst_paid_early_3d,
        "implied_tenor_months": results["Champion Scorecard"]["term"],
        "income_unverified_flag": results["Champion Scorecard"]["income_unverified"],
        "behavior_multiplier": results["Champion Scorecard"]["behavior_multiplier"],
        "max_allowable_new_pmt_dsr50": results["Champion Scorecard"]["max_new_payment"],
        "dsr_after_scorecard_limit": results["Champion Scorecard"]["realized_dsr"],
        "dsr_after_lgbm_limit": results["Raw LightGBM M4"]["realized_dsr"],
        "apr_scorecard": results["Champion Scorecard"]["apr"], "apr_lgbm": results["Raw LightGBM M4"]["apr"],
        "l_risk_scorecard": results["Champion Scorecard"]["risk_limit"], "l_affordability_scorecard": results["Champion Scorecard"]["affordability_limit"],
        "recommended_amount_scorecard_75_train_cutoff": results["Champion Scorecard"]["limit_75_train"],
        "l_risk_lgbm": results["Raw LightGBM M4"]["risk_limit"], "l_affordability_lgbm": results["Raw LightGBM M4"]["affordability_limit"],
        "recommended_amount_lgbm_75_train_cutoff": results["Raw LightGBM M4"]["limit_75_train"],
        "scorecard_approve_75_matched": approve_sc_matched.astype(np.int8),
        "lgbm_approve_75_matched": approve_lgb_matched.astype(np.int8),
        "recommended_amount_scorecard_75_matched": sc_match_limit,
        "recommended_amount_lgbm_75_matched": lgb_match_limit,
    })
    con.register("stage4_out", out)
    con.execute(f"COPY stage4_out TO '{OUT.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("stage4_out")

    # Strategy plots.
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in trade.groupby("Model"):
        ax.plot(group["Approval Rate %"], group["Observed Bad Rate %"], marker="o", label=model)
    ax.set(xlabel="Observed OOT approval rate (%)", ylabel="Observed bad rate among approvals (%)", title="OOT_Valid Approval / Bad-Rate Trade-off")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "stage4_approval_bad_rate.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in trade.groupby("Model"):
        ax.plot(group["Approval Rate %"], group["Simulated Expected Net Profit"] / 1e6, marker="o", label=model)
    ax.set(xlabel="Observed OOT approval rate (%)", ylabel="Parametric simulated ENP (millions)", title="OOT_Valid Approval / Expected Net Profit")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "stage4_expected_net_profit.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in sensitivity.groupby("Model"):
        ax.plot(group["DSR Cap %"], group["Simulated Expected Net Profit"] / 1e6, marker="o", label=model)
    ax.set(xlabel="Total DSR cap (%)", ylabel="Parametric simulated ENP (millions)", title="Limit Sensitivity to Total DSR Cap")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "stage4_limit_sensitivity.png", dpi=160); plt.close(fig)

    # Diagnostic tier calibration table and Stage 4 report.
    tier_notes = "\n".join(f"- `{label}` at PD {edge:.0%}: Champion Score boundary {score_at_pd(edge):.1f}" for label, edge in [("A/B", .01), ("B/C", .02), ("C/D", .04), ("D/E", .07)])
    def md(df): return df.to_markdown(index=False, floatfmt=".4f")
    proxy_note = "Stage 3 governance flags remain: `has_tax_registry_record` internal PSI 1.149 and `tax_verified_income_ratio` PSI 0.312. They were included in the explicitly requested Stage 3 M4 proxy test, but remain flagged for Stage 5 CSI monitoring."
    assumptions = ("Parametric Commercial Simulation Assumptions (not statutory constants): COF=4.0% p.a.; OPEX=1.5% p.a.; assumed LGD=75%; APR by tiers A/B/C/D/E=9%/12%/15%/18%/0%, capped at 18%; fixed tier risk spread = APR - COF - OPEX. Expected credit loss is `principal × PD × LGD`; interest margin is accrued on the amortizing monthly balance; no recovery timing, tax, capital, or acquisition-cost estimate is included.")
    report = f"""# Stage 4 信用决策、额度建议与 Swap Set 策略评估

本阶段仅用 `Train_Dev` (`WEEK_NUM<=64`) 确定 tier 概率边界和 approval cutoff；随后对 `OOT_Valid` (`65–77`, 82,602 笔已观察贷款)进行盲策略评价。`OOT_Monitor` (`78–91`) 未查询。Challenger 使用 Stage 3 批准的 `pd_lgbm_raw`，不使用 Dev_Calib isotonic 输出。

## Tier 边界与 Train_Dev 校准

风险分层采用固定 PD bands，并以 `Train_Dev` 的分层 observed bad rate / mean PD 审核；`Tier E` (`PD>=7%`) 为 decline。Scorecard 转换边界：

{tier_notes}

Scorecard band score boundaries are inverse PDO score cutoffs; higher scores indicate lower PD. `LightGBM` uses the same PD bands on raw M4 PD. Train_Dev calibration table:

{md(tier_table)}

## Tenor 与数据完整性检查

Train_Dev `credamount_770A / annuity_780A` 有效率 {tenor_audit['valid_pct']:.2f}%；median rounded tenor={tenor_audit['median_raw']:.0f} months，P01={tenor_audit['p01_raw']:.0f}，P99={tenor_audit['p99_raw']:.0f}；clip-to-3 share={tenor_audit['clipped_below_3_pct']:.3f}%，clip-to-60 share={tenor_audit['clipped_above_60_pct']:.3f}%。Stage 4 does not drop missing-income rows; `maininc_215A` missing/<=100 uses tier-specific requested-amount fallback caps and `income_unverified_flag=1`.

## Hard Policy / Driver Rules

- `bureau_active_max_dpd >= 90` → policy decline.
- For positive declared income, `active_bureau_monthly_debt / maininc_215A > 0.60` → policy decline.
- `Tier E` → model risk decline. Tiers A–D income multiples are 4x/3x/2x/1x monthly income; Tier E is 0x.
- `M_behavior=1.15x` for zero observed DPD with `static_pct_inst_paid_early_3d>=0.70`; `1.00x` for other clean history; `0.75x` for minor positive DPD; `0.0x` for policy knockout.
- Missing/low income fallback caps are 80%/75%/70%/60% of requested amount in tiers A–D, multiplied by behavior; DSR is marked untestable for those rows. Tier E remains zero.
- `L_affordability` is the PV of the maximum payment under the selected Total DSR cap and tier APR. Recommended principal is `min(requested, L_risk_behavior, L_affordability)`; declined rows receive zero.

## Parametric Commercial Simulation Assumptions

{assumptions}

## Table 1: OOT_Valid Approval / Bad-Rate / Profit Trade-off

PD cutoffs are quantiles of Train_Dev model predictions only. Actual OOT approval rate can differ due to drift, hard policy, Tier E, and zero-limit declines. `100%` is the all-risk-score acceptance baseline after these overlays. Bad reduction is relative to that model's baseline.

{md(trade)}

## Table 2: Matched-Approval 2×2 Swap Set (75% target)

Both models approve exactly {matched_count:,} loans ({100*matched_count/n:.2f}% of OOT_Valid). The matched operating point is selected by each model's unlabeled OOT risk ranks among policy- and Tier-E-eligible loans; outcomes are not used to set selections. Policy knockouts are declines for both models.

{md(swap)}

Observed results are for Known Good/Bad (KGB) funded/observed loans only. KGB Sample Selection Bias means results do not identify outcomes for historically rejected applicants; no reject inference, fabricated labels, or causal approval effect is claimed.

## Table 3: DSR Limit Sensitivity at Train_Dev-Derived 75% Cutoff

{md(sensitivity)}

DSR compliance is measured only among approved rows with verified positive income. Income-unverified approved loans are separately counted and do not claim DSR compliance.

## Stage 5 Governance

{proxy_note}

## Audit Summary and Artifacts

OOT_Valid rows={len(oot):,}; `case_id` unique={oot.case_id.is_unique}; Train_Dev predictions used for all tier/approval cutoffs; OOT_Valid used only for strategy evaluation; OOT_Monitor rows read=0. `pd_lgbm_raw` is the selected Challenger PD. `APR`, COF, OPEX and LGD are parametric commercial assumptions.

Saved `data/processed/stage4_oot_valid_decisions.parquet` and figures `reports/figures/stage4_approval_bad_rate.png`, `stage4_expected_net_profit.png`, and `stage4_limit_sensitivity.png`.

**HARD STOP:** Stage 4 complete. Wait for human review before Stage 5.
"""
    (REPORTS / "stage4_credit_decision_report.md").write_text(report, encoding="utf-8")
    say("TABLE 1 — Approval strategy trade-off\n" + md(trade))
    say("TABLE 2 — Matched-approval swap set\n" + md(swap))
    say("TABLE 3 — DSR sensitivity\n" + md(sensitivity))
    say(f"Saved {OUT}; {len(oot):,} OOT_Valid decisions; HARD STOP for human review.")
    con.close()


if __name__ == "__main__":
    main()
