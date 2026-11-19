"""Stage 6: portfolio intelligence, parametric scenarios, and Power BI mart.

All application-level aggregates are computed in DuckDB. Depth-1 historical
records are aggregated by case_id in DuckDB before joining or materializing.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\data analysis\FinGuard")
DATA = ROOT / "data"
PROC = DATA / "processed"
PBI = DATA / "powerbi"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
PRED = PROC / "stage5_full_predictions.parquet"
MART = PROC / "finguard_feature_mart.parquet"
AP1 = DATA / "parquet_files" / "train" / "train_applprev_1_*.parquet"
AP2 = DATA / "parquet_files" / "train" / "train_applprev_2.parquet"
CB1 = DATA / "parquet_files" / "train" / "train_credit_bureau_a_1_*.parquet"
DEFS = DATA / "feature_definitions.csv"
S5CSV = PROC / "stage5_psi_csi_summary.csv"
RECAL = PROC / "models" / "recalibrators_stage5.json"
OPEX = 0.015
LGD_BASE = 0.75
COF_BASE = 0.04
TIERS = ["Tier A: Super Prime", "Tier B: Prime", "Tier C: Standard", "Tier D: High Risk", "Tier E: Decline"]
APR = {TIERS[0]: .09, TIERS[1]: .12, TIERS[2]: .15, TIERS[3]: .18, TIERS[4]: 0.}
MULT = {TIERS[0]: 4., TIERS[1]: 3., TIERS[2]: 2., TIERS[3]: 1., TIERS[4]: 0.}
FALLBACK = {TIERS[0]: .80, TIERS[1]: .75, TIERS[2]: .70, TIERS[3]: .60, TIERS[4]: 0.}
MODEL_COLS = {
    "Champion Scorecard": ("pd_scorecard", "pd_scorecard_recal_intercept", "pd_scorecard_recal_platt"),
    "Challenger LightGBM M4": ("pd_lgbm_raw", "pd_lgbm_recal_intercept", "pd_lgbm_recal_platt"),
}


def tier(pd_values):
    p = np.asarray(pd_values, dtype=float)
    return np.select([p < .01, p < .02, p < .04, p < .07], TIERS[:4], default=TIERS[4])


def loan_engine(df: pd.DataFrame, p: np.ndarray) -> dict:
    """Mirror the approved Stage 5/4 deterministic cap and hard-rule logic."""
    p = np.asarray(p, dtype=float)
    ts = tier(p)
    req = np.nan_to_num(df.credamount_770A.to_numpy(float), nan=0.).clip(0)
    ann = np.nan_to_num(df.annuity_780A.to_numpy(float), nan=0.).clip(0)
    income = df.maininc_215A.to_numpy(float)
    debt = np.nan_to_num(df.active_bureau_monthly_debt.to_numpy(float), nan=0.).clip(0)
    verified = np.isfinite(income) & (income > 100)
    inc = np.where(verified, income, 0.)
    term = np.clip(np.rint(np.divide(req, ann, out=np.full_like(req, 12.), where=ann > 0)), 3, 60).astype(int)
    apr = np.array([APR[t] for t in ts]); multiple = np.array([MULT[t] for t in ts]); fallback = np.array([FALLBACK[t] for t in ts])
    dpd = df.bureau_active_max_dpd.to_numpy(float)
    hist = df.static_avg_dpd_tol_closure24m.to_numpy(float)
    early = df.static_pct_inst_paid_early_3d.to_numpy(float)
    bad_hist = (np.nan_to_num(dpd, nan=0.) > 0) | (np.nan_to_num(hist, nan=0.) > 0)
    behavior = np.where(bad_hist, .75, 1.)
    behavior = np.where(~bad_hist & np.isfinite(early) & (early >= .70), 1.15, behavior)
    policy = (np.nan_to_num(dpd, nan=-1.) >= 90) | (np.isfinite(income) & (income > 0) & (debt / np.where(income != 0, income, np.nan) > .60))
    behavior = np.where(policy | (ts == TIERS[4]), 0., behavior)
    risk = np.where(verified, inc * multiple * behavior, req * fallback * behavior)
    maxpay = np.maximum(0., .50 * inc - debt)
    mr = apr / 12.
    aff = np.where(verified, np.where(mr > 0, maxpay * (1 - (1 + mr) ** (-term)) / np.where(mr > 0, mr, 1.), maxpay * term), req * fallback)
    limit = np.maximum(0., np.minimum(req, np.minimum(risk, aff)))
    limit = np.where(policy | (ts == TIERS[4]), 0., limit)
    binding = np.where(policy | (ts == TIERS[4]), "Policy Decline", np.where(~verified, "Unverified_Income_Cap", np.where((req <= risk) & (req <= aff), "Requested Amount", np.where(risk <= aff, "Risk_Behavior", "Affordability"))))
    return {"tier": ts, "apr": apr, "term": term, "limit": limit, "behavior": behavior,
            "policy": policy, "verified": verified, "binding": binding, "request": req}


def balance_factor(term, apr):
    """Sum of monthly opening balances per unit of amortizing principal."""
    term = np.asarray(term, int); r = np.asarray(apr, float) / 12.
    out = (term + 1.) / 2.
    nz = r > 0
    x = np.power(1 + r[nz], term[nz])
    out[nz] = (term[nz] * x - (x - 1.) / r[nz]) / (x - 1.)
    return out


def profit(principal, pd_loss, apr, term, cof, lgd):
    principal = np.maximum(np.asarray(principal, float), 0.)
    net_rev = np.maximum(np.asarray(apr, float) - cof - OPEX, 0.) / 12. * principal * balance_factor(term, apr)
    return net_rev - principal * np.clip(np.asarray(pd_loss, float), 0., 1.) * lgd


def markdown(df: pd.DataFrame, digits=4) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def main():
    for d in (PBI, FIGS, PROC / "models", ROOT / "duckdb_temp"):
        d.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit='6GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(ROOT / 'duckdb_temp').as_posix()}'")
    pred_path, mart_path = PRED.as_posix(), MART.as_posix()

    # The mart is already case-level. Raw depth-1 prior applications are reduced
    # to one empirical maximum DPD per case before any join.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE prior_dpd AS
        SELECT case_id, COUNT(*) prior_contract_count, COUNT(actualdpd_943P) prior_dpd_observation_count,
               MAX(GREATEST(COALESCE(actualdpd_943P,0),0)) prior_max_dpd,
               MAX(GREATEST(COALESCE(maxdpdtolerance_577P,0),0)) prior_max_dpd_tolerance
        FROM read_parquet('{AP1.as_posix()}') GROUP BY case_id
    """)
    ap1_schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{AP1.as_posix()}')").fetchdf()
    ap2_schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{AP2.as_posix()}')").fetchdf()
    cb_schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{CB1.as_posix()}')").fetchdf()
    defs = con.execute(f"SELECT * FROM read_csv_auto('{DEFS.as_posix()}') WHERE Variable IN ('actualdpd_943P','maxdpdtolerance_577P','cacccardblochreas_147M','credacc_cards_status_52L','dpdmax_139P','dpdmax_757P','dpdmaxdatemonth_442T','dpdmaxdatemonth_89T','dpdmaxdateyear_596T','dpdmaxdateyear_896T')").fetchdf()
    ap2_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{AP2.as_posix()}')").fetchone()[0]
    ap1_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{AP1.as_posix()}')").fetchone()[0]
    split_audit=con.execute(f"SELECT split_name,COUNT(*) Rows,AVG(CASE WHEN target IS NULL THEN 1.0 ELSE 0.0 END)*100 Target_Null_Pct,AVG(CASE WHEN pd_lgbm_raw IS NULL THEN 1.0 ELSE 0.0 END)*100 Raw_PD_Null_Pct FROM read_parquet('{pred_path}') GROUP BY split_name ORDER BY split_name").fetchdf()
    raw_dpd_null=con.execute(f"SELECT AVG(CASE WHEN actualdpd_943P IS NULL THEN 1.0 ELSE 0.0 END)*100 FROM read_parquet('{AP1.as_posix()}')").fetchone()[0]
    applprev2_status_null=con.execute(f"SELECT AVG(CASE WHEN credacc_cards_status_52L IS NULL THEN 1.0 ELSE 0.0 END)*100 FROM read_parquet('{AP2.as_posix()}')").fetchone()[0]

    # Calculate case-level deterministic caps inside DuckDB; duplicate two model
    # views and retain only required columns. No full mart is materialized in pandas.
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE cases AS
    WITH joined AS (
      SELECT p.case_id, p.WEEK_NUM, p.MONTH, p.split_name, p.target,
             p.pd_scorecard, p.pd_lgbm_raw,
             p.pd_scorecard_recal_intercept, p.pd_lgbm_recal_intercept,
             p.pd_scorecard_recal_platt, p.pd_lgbm_recal_platt,
             m.credamount_770A, m.annuity_780A, m.maininc_215A,
             m.active_bureau_monthly_debt, m.bureau_active_max_dpd,
             m.static_avg_dpd_tol_closure24m, m.static_pct_inst_paid_early_3d,
             m.person_income_type, m.person_gender_sensitive,
             m.person_age_band_sensitive, m.person_age,
             COALESCE(d.prior_contract_count,0) prior_contract_count,
             COALESCE(d.prior_dpd_observation_count,0) prior_dpd_observation_count,
             COALESCE(d.prior_max_dpd,0) prior_max_dpd,
             COALESCE(d.prior_max_dpd_tolerance,0) prior_max_dpd_tolerance
      FROM read_parquet('{pred_path}') p
      JOIN read_parquet('{mart_path}') m USING(case_id)
      LEFT JOIN prior_dpd d USING(case_id)
    ), model_pd AS (
      SELECT *, 'Champion Scorecard' Model, pd_scorecard PD_Raw,
             pd_scorecard_recal_intercept PD_Intercept, pd_scorecard_recal_platt PD_Operational
      FROM joined
      UNION ALL
      SELECT *, 'Challenger LightGBM M4', pd_lgbm_raw,
             pd_lgbm_recal_intercept, pd_lgbm_recal_platt
      FROM joined
    ), tagged AS (
      SELECT *, CASE WHEN PD_Intercept < .01 THEN '{TIERS[0]}' WHEN PD_Intercept < .02 THEN '{TIERS[1]}'
                     WHEN PD_Intercept < .04 THEN '{TIERS[2]}' WHEN PD_Intercept < .07 THEN '{TIERS[3]}' ELSE '{TIERS[4]}' END Risk_Tier,
             CASE WHEN PD_Operational < .01 THEN '{TIERS[0]}' WHEN PD_Operational < .02 THEN '{TIERS[1]}'
                     WHEN PD_Operational < .04 THEN '{TIERS[2]}' WHEN PD_Operational < .07 THEN '{TIERS[3]}' ELSE '{TIERS[4]}' END Operational_Tier,
             GREATEST(COALESCE(credamount_770A,0),0) Requested_Amount,
             GREATEST(COALESCE(annuity_780A,0),0) Annuity,
             CASE WHEN maininc_215A > 100 THEN maininc_215A ELSE 0 END Income_Safe,
             CASE WHEN maininc_215A > 100 THEN TRUE ELSE FALSE END Income_Verified,
             GREATEST(COALESCE(active_bureau_monthly_debt,0),0) Debt,
              CASE WHEN Operational_Tier = '{TIERS[0]}' THEN .09 WHEN Operational_Tier = '{TIERS[1]}' THEN .12 WHEN Operational_Tier = '{TIERS[2]}' THEN .15 WHEN Operational_Tier = '{TIERS[3]}' THEN .18 ELSE 0 END APR,
              CASE WHEN Operational_Tier = '{TIERS[0]}' THEN 4. WHEN Operational_Tier = '{TIERS[1]}' THEN 3. WHEN Operational_Tier = '{TIERS[2]}' THEN 2. WHEN Operational_Tier = '{TIERS[3]}' THEN 1. ELSE 0 END Risk_Multiple,
              CASE WHEN Operational_Tier = '{TIERS[0]}' THEN .80 WHEN Operational_Tier = '{TIERS[1]}' THEN .75 WHEN Operational_Tier = '{TIERS[2]}' THEN .70 WHEN Operational_Tier = '{TIERS[3]}' THEN .60 ELSE 0 END Fallback
      FROM model_pd
    ), eng1 AS (
      SELECT *, CAST(LEAST(60,GREATEST(3,ROUND(CASE WHEN Annuity>0 THEN Requested_Amount/Annuity ELSE 12 END))) AS INTEGER) Term,
             CASE WHEN COALESCE(bureau_active_max_dpd,0)>0 OR COALESCE(static_avg_dpd_tol_closure24m,0)>0 THEN .75
                  WHEN COALESCE(static_pct_inst_paid_early_3d,0)>=.70 THEN 1.15 ELSE 1. END Behavior,
             COALESCE(bureau_active_max_dpd,-1)>=90 OR COALESCE((maininc_215A>0 AND Debt/NULLIF(maininc_215A,0)>.60),FALSE) Policy
      FROM tagged
    ), eng2 AS (
      SELECT *, CASE WHEN Policy OR Operational_Tier='{TIERS[4]}' THEN 0
                    WHEN Income_Verified THEN Income_Safe*Risk_Multiple*Behavior
                    ELSE Requested_Amount*Fallback*Behavior END Risk_Cap,
             GREATEST(0,.50*Income_Safe-Debt) Max_Payment,
             APR/12.0 Monthly_Rate
      FROM eng1
    ), eng3 AS (
      SELECT *, CASE WHEN Income_Verified THEN
                    CASE WHEN Monthly_Rate>0 THEN Max_Payment*(1-POWER(1+Monthly_Rate,-Term))/Monthly_Rate ELSE Max_Payment*Term END
                    ELSE Requested_Amount*Fallback END Affordability_Cap
      FROM eng2
    )
    SELECT *, CASE WHEN Policy OR Operational_Tier='{TIERS[4]}' THEN 0 ELSE GREATEST(0,LEAST(Requested_Amount,Risk_Cap,Affordability_Cap)) END Recommended_Limit,
       CASE WHEN Policy OR Operational_Tier='{TIERS[4]}' THEN 'Policy Decline'
            WHEN Requested_Amount<=Risk_Cap AND Requested_Amount<=Affordability_Cap THEN 'Requested Amount'
            WHEN Risk_Cap<=Affordability_Cap THEN 'Risk Appetite' ELSE 'Affordability' END Binding_Driver,
       CASE WHEN Requested_Amount<20000 THEN '<20k' WHEN Requested_Amount<50000 THEN '20k-50k' WHEN Requested_Amount<100000 THEN '50k-100k' ELSE '100k+' END Loan_Ticket_Band,
       CASE WHEN Term<=6 THEN '<=6m' WHEN Term<=12 THEN '7-12m' WHEN Term<=24 THEN '13-24m' ELSE '25m+' END Implied_Tenor_Band,
       CASE WHEN prior_contract_count=0 THEN 'No prior application'
            WHEN prior_dpd_observation_count=0 THEN 'Prior contract: DPD unavailable'
            WHEN prior_max_dpd<=0 THEN '0 DPD' WHEN prior_max_dpd<=10 THEN '1-10 DPD'
            WHEN prior_max_dpd<=30 THEN '11-30 DPD' WHEN prior_max_dpd<90 THEN '31-89 DPD' ELSE '90+ DPD' END Prior_Delinquency_Severity,
       CASE WHEN NOT Policy AND Operational_Tier<>'{TIERS[4]}' AND GREATEST(0,LEAST(Requested_Amount,Risk_Cap,Affordability_Cap))>0 THEN TRUE ELSE FALSE END Eligible
    FROM eng3
    """)

    # Calibrate valid-only operational cutoffs. Retain the exact Stage 5 75%
    # target semantics; Iso-Risk matches scorecard observed Valid bad rate.
    def load_split(split):
        return con.execute(f"SELECT * FROM cases WHERE split_name='{split}'").fetchdf()
    valid = load_split("OOT_Valid")
    monitor = load_split("OOT_Monitor")
    cutoffs = {}
    for model in MODEL_COLS:
        v = valid[valid.Model.eq(model)].copy()
        eligible = v.Eligible.to_numpy(bool)
        k75 = min(round(.75 * len(v)), int(eligible.sum()))
        idx = np.flatnonzero(eligible)
        order = np.lexsort((v.case_id.to_numpy()[idx], v.PD_Operational.to_numpy()[idx]))
        chosen = idx[order[:k75]]
        cut75 = float(v.PD_Operational.to_numpy()[chosen].max()) if len(chosen) else 0.
        cutoffs[model] = {"Iso-Approval": cut75, "Iso-Risk": cut75}
    score_valid = valid[valid.Model.eq("Champion Scorecard")]
    scmask = score_valid.Eligible.to_numpy(bool) & (score_valid.PD_Operational.to_numpy(float) <= cutoffs["Champion Scorecard"]["Iso-Approval"])
    appetite = float(score_valid.target.to_numpy(int)[scmask].mean()) if scmask.any() else 0.
    chv = valid[(valid.Model == "Challenger LightGBM M4") & valid.Eligible].copy()
    chv=chv.sort_values(["PD_Operational", "case_id"])
    rates = np.cumsum(chv.target.to_numpy(int)) / np.arange(1, len(chv)+1)
    krisk = int(np.argmin(np.abs(rates-appetite))) if len(rates) else 0
    cutoffs["Challenger LightGBM M4"]["Iso-Risk"] = float(chv.PD_Operational.iloc[krisk]) if len(chv) else 0.

    def choose_target_cut(df, target_fraction):
        elig = df.Eligible.to_numpy(bool); ids = np.flatnonzero(elig)
        k = min(round(target_fraction * len(df)), len(ids))
        if not k: return 0.
        ids = ids[np.lexsort((df.case_id.to_numpy()[ids],df.PD_Operational.to_numpy()[ids]))]
        return float(df.PD_Operational.to_numpy()[ids[k-1]])

    baseline_valid_approval = {}
    for m in MODEL_COLS:
        v=valid[valid.Model.eq(m)]
        baseline_valid_approval[m]=float((v.Eligible & (v.PD_Operational<=cutoffs[m]["Iso-Risk"])).mean())
    growth_target = {m:min(1.,baseline_valid_approval[m]+.05) for m in MODEL_COLS}
    growth_cut = {m: choose_target_cut(valid[valid.Model.eq(m)], growth_target[m]) for m in MODEL_COLS}
    # Stress-specific emergency cutoff is selected on Valid by maximum expected
    # severe-stress profit, then carried forward untouched to Monitor.
    emergency = {}
    for model in MODEL_COLS:
        v = valid[valid.Model.eq(model)].copy()
        p = v.PD_Operational.to_numpy(float); eng = loan_engine(v, p)
        base_tier = eng["tier"]
        mult = np.where(np.isin(base_tier, TIERS[:2]), 1.30, 2.00)
        exp_profit = profit(eng["limit"], np.minimum(p*mult,1), eng["apr"],eng["term"],.055,.85)
        eligible = eng["limit"]>0
        ids=np.flatnonzero(eligible); order=np.lexsort((v.case_id.to_numpy()[ids],p[ids])); ids=ids[order]
        # Include every prefix and choose the positive-margin one with greatest expected total profit.
        cum=np.cumsum(exp_profit[ids]); positive=np.flatnonzero(cum>0)
        k=int(positive[np.argmax(cum[positive])]+1) if len(positive) else 0
        emergency[model]=float(p[ids[k-1]]) if k else 0.

    mode_cut={m:{"Iso-Approval":cutoffs[m]["Iso-Approval"],"Iso-Risk":cutoffs[m]["Iso-Risk"]} for m in MODEL_COLS}
    print("VALID-FROZEN CUTOFFS", json.dumps(cutoffs, ensure_ascii=False))
    # Stage 6.1: weekly and monthly cohort metrics using intercept recalibration tiers.
    # Selected Valid cutoffs are used to describe indicative approvals/exposure.
    con.register("cutoff_map", pd.DataFrame([{"Model":m,"cut":cutoffs[m]["Iso-Risk"]} for m in MODEL_COLS]))
    temporal = con.execute(f"""
      WITH x AS (
       SELECT *, CASE WHEN Eligible AND PD_Operational<=cut THEN TRUE ELSE FALSE END approved
       FROM cases JOIN cutoff_map USING(Model)
      ), temporal AS (
       SELECT 'Weekly' Grain, CAST(WEEK_NUM AS VARCHAR) Period, WEEK_NUM Period_Order, split_name Split, Model, Risk_Tier,
              COUNT(*) Applications, SUM(target) Bads, SUM(approved::INT) Approvals,
              AVG(target)*100 Actual_Bad_Rate_Pct, AVG(PD_Raw)*100 Raw_PD_Pct,
              AVG(PD_Intercept)*100 Recalibrated_PD_Pct,
              AVG(CASE WHEN approved THEN Recommended_Limit END) Avg_Recommended_Exposure,
              SUM(CASE WHEN approved THEN Recommended_Limit ELSE 0 END) Recommended_Volume
       FROM x GROUP BY ALL
       UNION ALL
       SELECT 'Monthly', CAST(MONTH AS VARCHAR), MONTH, split_name, Model, Risk_Tier,
              COUNT(*), SUM(target), SUM(approved::INT), AVG(target)*100, AVG(PD_Raw)*100, AVG(PD_Intercept)*100,
              AVG(CASE WHEN approved THEN Recommended_Limit END), SUM(CASE WHEN approved THEN Recommended_Limit ELSE 0 END)
       FROM x GROUP BY ALL
      ), temporal_all AS (
       SELECT 'Weekly' Grain, CAST(WEEK_NUM AS VARCHAR) Period, WEEK_NUM Period_Order, split_name Split, Model, 'ALL_PORTFOLIO' Risk_Tier,
              COUNT(*) Applications, SUM(target) Bads, SUM(approved::INT) Approvals,
              AVG(target)*100 Actual_Bad_Rate_Pct, AVG(PD_Raw)*100 Raw_PD_Pct,
              AVG(PD_Intercept)*100 Recalibrated_PD_Pct,
              AVG(CASE WHEN approved THEN Recommended_Limit END) Avg_Recommended_Exposure,
              SUM(CASE WHEN approved THEN Recommended_Limit ELSE 0 END) Recommended_Volume
       FROM x GROUP BY ALL
       UNION ALL
       SELECT 'Monthly', CAST(MONTH AS VARCHAR), MONTH, split_name, Model, 'ALL_PORTFOLIO',
              COUNT(*), SUM(target), SUM(approved::INT), AVG(target)*100, AVG(PD_Raw)*100, AVG(PD_Intercept)*100,
              AVG(CASE WHEN approved THEN Recommended_Limit END), SUM(CASE WHEN approved THEN Recommended_Limit ELSE 0 END)
       FROM x GROUP BY ALL
      ), pg AS (
       SELECT 'Weekly' Grain,CAST(WEEK_NUM AS VARCHAR) Period,WEEK_NUM Period_Order,split_name Split,Model,Risk_Tier,PD_Intercept,
              SUM(target) bad_n,COUNT(*)-SUM(target) good_n
       FROM x GROUP BY ALL
       UNION ALL
       SELECT 'Monthly',CAST(MONTH AS VARCHAR),MONTH,split_name,Model,Risk_Tier,PD_Intercept,
              SUM(target),COUNT(*)-SUM(target)
       FROM x GROUP BY ALL
       UNION ALL
       SELECT 'Weekly',CAST(WEEK_NUM AS VARCHAR),WEEK_NUM,split_name,Model,'ALL_PORTFOLIO',PD_Intercept,SUM(target),COUNT(*)-SUM(target)
       FROM x GROUP BY ALL
       UNION ALL
       SELECT 'Monthly',CAST(MONTH AS VARCHAR),MONTH,split_name,Model,'ALL_PORTFOLIO',PD_Intercept,SUM(target),COUNT(*)-SUM(target)
       FROM x GROUP BY ALL
      ), ranked AS (
       SELECT *, SUM(good_n) OVER(PARTITION BY Grain,Period,Split,Model,Risk_Tier ORDER BY PD_Intercept ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) good_before,
                 SUM(bad_n) OVER(PARTITION BY Grain,Period,Split,Model,Risk_Tier ORDER BY PD_Intercept ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) bad_cum,
                 SUM(good_n) OVER(PARTITION BY Grain,Period,Split,Model,Risk_Tier ORDER BY PD_Intercept ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) good_cum,
                 SUM(bad_n) OVER(PARTITION BY Grain,Period,Split,Model,Risk_Tier) total_bad,
                 SUM(good_n) OVER(PARTITION BY Grain,Period,Split,Model,Risk_Tier) total_good
       FROM pg
      ), rank_metrics AS (
       SELECT Grain,Period,Split,Model,Risk_Tier,
         CASE WHEN MAX(total_bad)>0 AND MAX(total_good)>0 THEN SUM(bad_n*(COALESCE(good_before,0)+good_n/2.0))/(MAX(total_bad)*MAX(total_good)) END AUC,
         CASE WHEN MAX(total_bad)>0 AND MAX(total_good)>0 THEN MAX(ABS(bad_cum/total_bad-good_cum/total_good)) END KS
       FROM ranked GROUP BY ALL
      )
      SELECT t.*, r.KS, 2*r.AUC-1 Gini FROM temporal t LEFT JOIN rank_metrics r USING(Grain,Period,Split,Model,Risk_Tier)
      UNION ALL
      SELECT t.*, r.KS, 2*r.AUC-1 Gini FROM temporal_all t LEFT JOIN rank_metrics r USING(Grain,Period,Split,Model,Risk_Tier)
      ORDER BY CASE Grain WHEN 'Weekly' THEN 1 ELSE 2 END,Period_Order,Model,Split,Risk_Tier
    """).fetchdf()
    temporal.to_csv(PBI/"pbi_01_temporal_cohort_monitoring.csv",index=False)

    # Stage 6.2: valid-calibrated 1-100 percentile cutoff curves, forward-applied.
    cutoff_rows=[]
    for model in MODEL_COLS:
        v=valid[(valid.Model==model)&valid.Eligible].sort_values(["PD_Operational","case_id"])
        if v.empty: continue
        n=len(v)
        for pct in range(1,101):
            cutoff_rows.append({"Model":model,"Target_Eligible_Approval_Pct":pct,"PD_Cutoff":float(v.PD_Operational.iloc[max(0,math.ceil(pct*n/100)-1)])})
    cuts=pd.DataFrame(cutoff_rows)
    con.register("trade_cuts",cuts)
    trade = con.execute("""
      SELECT c.Model,c.Target_Eligible_Approval_Pct,c.PD_Cutoff,x.split_name Split,
       COUNT(*) FILTER(WHERE x.Eligible AND x.PD_Operational<=c.PD_Cutoff) Approved_Count,
       100.0*COUNT(*) FILTER(WHERE x.Eligible AND x.PD_Operational<=c.PD_Cutoff)/COUNT(*) Approval_Rate_Pct,
       AVG(x.target) FILTER(WHERE x.Eligible AND x.PD_Operational<=c.PD_Cutoff)*100 Observed_Bad_Rate_Pct,
       SUM(x.Recommended_Limit) FILTER(WHERE x.Eligible AND x.PD_Operational<=c.PD_Cutoff) Recommended_Volume,
       AVG(x.PD_Operational) FILTER(WHERE x.Eligible AND x.PD_Operational<=c.PD_Cutoff)*100 Mean_PD_Pct
      FROM trade_cuts c JOIN cases x ON c.Model=x.Model AND x.split_name IN ('OOT_Valid','OOT_Monitor')
      GROUP BY c.Model,c.Target_Eligible_Approval_Pct,c.PD_Cutoff,x.split_name
      ORDER BY c.Model,c.Target_Eligible_Approval_Pct,x.split_name
    """).fetchdf()
    trade.to_csv(PBI/"pbi_02_cutoff_tradeoff_curve.csv",index=False)

    # Stage 6.3: OOT_Monitor portfolio concentration, by requested dimensions
    # and valid-frozen Iso-Approval / Iso-Risk strategies.
    op_rows=[]
    for model in MODEL_COLS:
        for mode in ("Iso-Approval","Iso-Risk"):
            cut=mode_cut[model][mode]
            m=monitor[(monitor.Model==model)&monitor.Eligible&(monitor.PD_Operational<=cut)].copy()
            if m.empty: continue
            p=m.PD_Operational.to_numpy(float); eng=loan_engine(m,p)
            m["Risk_Tier"]=eng["tier"]; m["Recommended_Limit"]=eng["limit"];m["APR"]=eng["apr"];m["Term"]=eng["term"];m["Binding_Driver"]=eng["binding"]
            m["Mode"]=mode; m["Ticket_Band"]=np.select([eng["request"]<20000,eng["request"]<50000,eng["request"]<100000],["<20k","20k-50k","50k-100k"],default="100k+")
            m["Tenor_Band"]=np.select([eng["term"]<=6,eng["term"]<=12,eng["term"]<=24],["<=6m","7-12m","13-24m"],default="25m+")
            m["Income_Type"]=m.person_income_type.fillna("Unknown")
            m["Expected_Loss"]=m.Recommended_Limit*p*LGD_BASE
            op_rows.append(m[["Model","Mode","case_id","target","PD_Operational","Risk_Tier","Income_Type","Ticket_Band","Tenor_Band","Binding_Driver","Recommended_Limit","Expected_Loss"]])
    ops=pd.concat(op_rows,ignore_index=True)
    dims=["Risk_Tier","Income_Type","Ticket_Band","Tenor_Band","Binding_Driver"]
    group_cols=["Model","Mode"]+dims
    concentration=ops.groupby(group_cols,dropna=False).agg(
        Approved_Count=("case_id","count"), Total_Recommended_Volume=("Recommended_Limit","sum"),
        Observed_Bad_Rate_Pct=("target",lambda x:100*x.mean()),
        Mean_Recalibrated_PD_Pct=("PD_Operational",lambda x:100*x.mean()),
        Expected_Loss_Proxy=("Expected_Loss","sum"),Average_Recommended_Limit=("Recommended_Limit","mean")).reset_index()
    totals=ops.groupby(["Model","Mode"]).Recommended_Limit.sum().rename("Portfolio_Volume")
    concentration=concentration.join(totals,on=["Model","Mode"])
    concentration["Exposure_Share_Pct"]=100*concentration.Total_Recommended_Volume/concentration.Portfolio_Volume.replace(0,np.nan)
    approved_counts=ops.groupby(["Model","Mode"]).case_id.count().rename("Portfolio_Approved_Count")
    driver_counts=ops.groupby(["Model","Mode","Binding_Driver"]).case_id.count().rename("Driver_Approved_Count")
    driver_volume=ops.groupby(["Model","Mode","Binding_Driver"]).Recommended_Limit.sum().rename("Driver_Exposure")
    concentration=concentration.join(approved_counts,on=["Model","Mode"]).join(driver_counts,on=["Model","Mode","Binding_Driver"]).join(driver_volume,on=["Model","Mode","Binding_Driver"])
    tier_driver_counts=ops.groupby(["Model","Mode","Risk_Tier","Binding_Driver"]).case_id.count().rename("Tier_Driver_Approved_Count")
    tier_totals=ops.groupby(["Model","Mode","Risk_Tier"]).case_id.count().rename("Tier_Approved_Count")
    concentration=concentration.join(tier_driver_counts,on=["Model","Mode","Risk_Tier","Binding_Driver"]).join(tier_totals,on=["Model","Mode","Risk_Tier"])
    concentration["Binding_Driver_Share_Pct"]=100*concentration.Tier_Driver_Approved_Count/concentration.Tier_Approved_Count
    concentration.insert(0,"Dimension","Portfolio_Segment")
    concentration["Slice"]="All_Segment_Dimensions"
    concentration["Binding_Driver_Exposure_Share_Pct"]=100*concentration.Driver_Exposure/concentration.Portfolio_Volume.replace(0,np.nan)
    for d in dims[:4]:
        # HHI based on exposure share within that attribute, avoiding false HHI
        # from the full cartesian multidimensional cell.
        dimension=concentration.groupby(["Model","Mode",d],as_index=False).Total_Recommended_Volume.sum()
        dimension=dimension.merge(totals.reset_index(),on=["Model","Mode"],how="left")
        dimension["_share"]=dimension.Total_Recommended_Volume/dimension.Portfolio_Volume.replace(0,np.nan)
        hhi=dimension.groupby(["Model","Mode"])._share.apply(lambda x: float(np.square(x).sum())).rename(f"{d}_HHI").reset_index()
        concentration=concentration.merge(hhi,on=["Model","Mode"],how="left")
    cross=ops.groupby(["Model","Mode","Risk_Tier","Binding_Driver"],dropna=False).agg(
        Approved_Count=("case_id","count"), Total_Recommended_Volume=("Recommended_Limit","sum"),
        Observed_Bad_Rate_Pct=("target",lambda x:100*x.mean()),
        Mean_Recalibrated_PD_Pct=("PD_Operational",lambda x:100*x.mean()),
        Expected_Loss_Proxy=("Expected_Loss","sum"), Average_Recommended_Limit=("Recommended_Limit","mean")).reset_index()
    cross=cross.merge(totals.rename("Portfolio_Volume").reset_index(),on=["Model","Mode"],how="left")
    cross=cross.merge(approved_counts.rename("Portfolio_Approved_Count").reset_index(),on=["Model","Mode"],how="left")
    cross["Exposure_Share_Pct"]=100*cross.Total_Recommended_Volume/cross.Portfolio_Volume.replace(0,np.nan)
    cross["Driver_Approved_Count"]=cross.Approved_Count
    cross["Driver_Exposure"]=cross.Total_Recommended_Volume
    cross["Tier_Driver_Approved_Count"]=cross.Approved_Count
    cross=cross.merge(tier_totals.rename("Tier_Approved_Count").reset_index(),on=["Model","Mode","Risk_Tier"],how="left")
    cross["Binding_Driver_Share_Pct"]=100*cross.Tier_Driver_Approved_Count/cross.Tier_Approved_Count
    cross["Binding_Driver_Exposure_Share_Pct"]=100*cross.Driver_Exposure/cross.groupby(["Model","Mode","Risk_Tier"]).Driver_Exposure.transform("sum").replace(0,np.nan)
    for d in dims[:4]: cross[f"{d}_HHI"]=np.nan
    cross["Dimension"]="Tier_x_Binding_Driver"
    cross["Slice"]=cross["Risk_Tier"].astype(str)+" | "+cross["Binding_Driver"].astype(str)
    for d in ["Income_Type","Ticket_Band","Tenor_Band"]: cross[d]="ALL"
    single_slices=[]
    for d in dims:
        one=ops.groupby(["Model","Mode",d],dropna=False).agg(
            Approved_Count=("case_id","count"), Total_Recommended_Volume=("Recommended_Limit","sum"),
            Observed_Bad_Rate_Pct=("target",lambda x:100*x.mean()),
            Mean_Recalibrated_PD_Pct=("PD_Operational",lambda x:100*x.mean()),
            Expected_Loss_Proxy=("Expected_Loss","sum"),Average_Recommended_Limit=("Recommended_Limit","mean")).reset_index()
        one=one.merge(totals.rename("Portfolio_Volume").reset_index(),on=["Model","Mode"],how="left")
        one=one.merge(approved_counts.rename("Portfolio_Approved_Count").reset_index(),on=["Model","Mode"],how="left")
        one["Exposure_Share_Pct"]=100*one.Total_Recommended_Volume/one.Portfolio_Volume.replace(0,np.nan)
        for col in dims: one[col]=one[d].astype("string").fillna("Unknown") if col==d else "ALL"
        one["Binding_Driver_Share_Pct"]=np.nan; one["Binding_Driver_Exposure_Share_Pct"]=np.nan
        one["Dimension"]=d; one["Slice"]=one[d]
        for col in ["Driver_Approved_Count","Driver_Exposure","Tier_Driver_Approved_Count","Tier_Approved_Count"]: one[col]=np.nan
        for col in ["Risk_Tier_HHI","Income_Type_HHI","Ticket_Band_HHI","Tenor_Band_HHI"]: one[col]=np.nan
        single_slices.append(one.reindex(columns=concentration.columns))
    concentration=pd.concat([concentration,cross.reindex(columns=concentration.columns),*single_slices],ignore_index=True)
    concentration.to_csv(PBI/"pbi_03_portfolio_concentration.csv",index=False)

    # Stage 6.4: four commercial stress cases. S4 emergency action threshold was
    # chosen on Valid expected profit, then tested on Monitor; no Monitor labels
    # are used to select the emergency cutoff.
    scenario_specs={
      "S1_Baseline":(.04,.75,"base"),
      "S2_Controlled_Growth":(.04,.75,"growth"),
      "S3_Mild_Downturn":(.05,.80,"mild"),
      "S4_Severe_Tail_Shock":(.055,.85,"severe"),
    }
    stress_rows=[]
    for model in MODEL_COLS:
        m=monitor[monitor.Model.eq(model)].copy(); p=m.PD_Operational.to_numpy(float); y=m.target.to_numpy(float)
        base=loan_engine(m,p); base_tier=base["tier"]
        for scenario,(cof,lgd,kind) in scenario_specs.items():
            if kind=="growth": cut=growth_cut[model]; mult=np.ones(len(m))
            elif kind=="severe": cut=mode_cut[model]["Iso-Risk"]; mult=np.where(np.isin(base_tier,TIERS[:2]),1.30,2.00)
            elif kind=="mild": cut=mode_cut[model]["Iso-Risk"]; mult=np.full(len(m),1.25)
            else: cut=mode_cut[model]["Iso-Risk"]; mult=np.ones(len(m))
            approved=base["limit"]>0
            # Model-specific baseline thresholds on Valid; S2 uses +5pp Valid target.
            if kind!="growth": approved &= p<=cut
            else: approved &= p<=cut
            principal=np.where(approved,base["limit"],0.)
            stressed_pd=np.minimum(p*mult,1.)
            # Translate observed defaults by the same scenario multiplier as a
            # parametric loss proxy; raw target itself remains in the output rates.
            realized=profit(principal,np.minimum(y*mult,1.),base["apr"],base["term"],cof,lgd)
            expected=profit(principal,stressed_pd,base["apr"],base["term"],cof,lgd)
            stress_rows.append({"Scenario":scenario,"Model":model,"Risk_Tier":"Portfolio Total","Valid_PD_Cutoff":cut,
                "Valid_Target_Approval_Pct":100*growth_target[model] if kind=="growth" else 100*baseline_valid_approval[model],
                "Monitor_Approval_Pct":100*approved.mean(),"Approved_Count":int(approved.sum()),
                "Observed_Bad_Rate_Pct":100*y[approved].mean() if approved.any() else np.nan,
                "Stress_PD_Multiplier":"1.30x A-B / 2.00x C-E" if kind=="severe" else f"{mult[0]:.2f}x",
                "COF_Pct":100*cof,"LGD_Pct":100*lgd,"Recommended_Volume":float(principal.sum()),
                "Realized_Net_Profit_Stress_Proxy":float(realized.sum()),"Model_Implied_Expected_Net_Profit":float(expected.sum()),
                "Emergency_Valid_Cutoff":emergency[model] if kind=="severe" else np.nan})
            for t in TIERS:
                mask=approved&(base_tier==t)
                stress_rows.append({"Scenario":scenario,"Model":model,"Risk_Tier":t,"Valid_PD_Cutoff":cut,
                 "Valid_Target_Approval_Pct":100*growth_target[model] if kind=="growth" else 100*baseline_valid_approval[model],
                 "Monitor_Approval_Pct":100*mask.sum()/max(len(m),1),"Approved_Count":int(mask.sum()),
                 "Observed_Bad_Rate_Pct":100*y[mask].mean() if mask.any() else np.nan,
                 "Stress_PD_Multiplier":"1.30x A-B / 2.00x C-E" if kind=="severe" else f"{mult[0]:.2f}x",
                 "COF_Pct":100*cof,"LGD_Pct":100*lgd,"Recommended_Volume":float(principal[mask].sum()),
                 "Realized_Net_Profit_Stress_Proxy":float(realized[mask].sum()),"Model_Implied_Expected_Net_Profit":float(expected[mask].sum()),
                 "Emergency_Valid_Cutoff":emergency[model] if kind=="severe" else np.nan})
        # Severe-tail counterfactual with Valid-optimized emergency cutoff.
        cut=emergency[model]; mult=np.where(np.isin(base_tier,TIERS[:2]),1.30,2.)
        approved=(base["limit"]>0)&(p<=cut); principal=np.where(approved,base["limit"],0.)
        rr=profit(principal,np.minimum(y*mult,1.),base["apr"],base["term"],.055,.85)
        ee=profit(principal,np.minimum(p*mult,1.),base["apr"],base["term"],.055,.85)
        stress_rows.append({"Scenario":"S4_Emergency_Cutoff","Model":model,"Risk_Tier":"Portfolio Total","Valid_PD_Cutoff":cut,"Valid_Target_Approval_Pct":np.nan,
             "Monitor_Approval_Pct":100*approved.mean(),"Approved_Count":int(approved.sum()),"Observed_Bad_Rate_Pct":100*y[approved].mean() if approved.any() else np.nan,
             "Stress_PD_Multiplier":"1.30x A-B / 2.00x C-E","COF_Pct":5.5,"LGD_Pct":85.,"Recommended_Volume":principal.sum(),
             "Realized_Net_Profit_Stress_Proxy":rr.sum(),"Model_Implied_Expected_Net_Profit":ee.sum(),"Emergency_Valid_Cutoff":cut})
    stress=pd.DataFrame(stress_rows)
    stress.to_csv(PBI/"pbi_04_stress_scenarios.csv",index=False)

    # Stage 6.5: combine PSI/CSI, decile calibration, and subgroup fairness.
    csi=pd.read_csv(S5CSV)
    csi_rows=pd.DataFrame({"Record_Type":csi.Type,"Model":"All","Split":csi.Comparison,"Feature":csi["Feature / Score"],"Metric":csi["PSI / CSI"],"Reference_Status":csi["Reference Status"]})
    decile_rows=[]
    for model in MODEL_COLS:
        for split in ("OOT_Valid","OOT_Monitor"):
            z=con.execute(f"SELECT * FROM cases WHERE Model='{model}' AND split_name='{split}'").fetchdf()
            z["Decile"]=pd.qcut(z.PD_Operational.rank(method="first"),10,labels=False)+1
            for d,g in z.groupby("Decile"):
                decile_rows.append({"Record_Type":"Decile Calibration","Model":model,"Split":split,"Feature":f"Decile {int(d)}",
                    "Metric":float(g.PD_Operational.mean()),"Observed_Bad_Rate":float(g.target.mean()),"Sample_Count":len(g),
                    "Expected_PD_Pct":100*g.PD_Operational.mean(),"Actual_Bad_Rate_Pct":100*g.target.mean(),"A_E_Ratio":g.target.mean()/g.PD_Operational.mean() if g.PD_Operational.mean() else np.nan})
    age_q=con.execute(f"SELECT quantile_cont(person_age, [0.2,0.4,0.6,0.8]) FROM read_parquet('{mart_path}') WHERE WEEK_NUM<=64 AND person_age IS NOT NULL").fetchone()[0]
    fair_rows=[]
    for model in MODEL_COLS:
        z=monitor[monitor.Model.eq(model)].copy()
        eng=loan_engine(z,z.PD_Operational.to_numpy(float)); cut=mode_cut[model]["Iso-Approval"]
        approved=(eng["limit"]>0)&(z.PD_Operational.to_numpy(float)<=cut)
        age_bin=pd.cut(z.person_age,[-np.inf,*age_q,np.inf],labels=["Age Q1","Age Q2","Age Q3","Age Q4","Age Q5"])
        for attr,groups in [("Gender",z.person_gender_sensitive.fillna("Unknown")),("Age Band",z.person_age_band_sensitive.fillna("Unknown")),("Age Train_Dev Quintile",age_bin.astype("string").fillna("Missing"))]:
            for grp in sorted(groups.astype(str).unique()):
                mask=groups.astype(str).to_numpy()==grp
                if not mask.any(): continue
                ar=float(approved[mask].mean()); bad=float(z.target.to_numpy(float)[mask].mean()); meanp=float(z.PD_Operational.to_numpy(float)[mask].mean())
                fair_rows.append({"Record_Type":"Fairness","Model":model,"Split":"OOT_Monitor","Feature":attr,"Group":grp,"Sample_Count":int(mask.sum()),
                    "Approval_Rate_Pct":100*ar,"Observed_Bad_Rate_Pct":100*bad,"Mean_Recalibrated_PD_Pct":100*meanp,"A_E_Ratio":bad/meanp if meanp else np.nan})
    fairness=pd.DataFrame(fair_rows)
    if len(fairness):
        fairness["Highest_Group_Approval_Pct"]=fairness.groupby(["Model","Feature"]).Approval_Rate_Pct.transform("max")
        fairness["AIR"]=fairness.Approval_Rate_Pct/fairness.Highest_Group_Approval_Pct
        fairness["AIR_LT_080"]=fairness.AIR<.8
    deciles=pd.DataFrame(decile_rows)
    full5=pd.concat([csi_rows,pd.DataFrame(decile_rows),fairness],ignore_index=True,sort=False)
    full5.to_csv(PBI/"pbi_05_psi_csi_and_fairness.csv",index=False)

    # Stage 6.6: observed historical prior-contract DPD severity to current
    # application target. applprev_2 itself supplies no monthly DPD snapshots.
    migration=con.execute("""
      SELECT Model, Risk_Tier, Prior_Delinquency_Severity,
       COUNT(*) Application_Count, SUM(target) Current_Defaults,
       AVG(target)*100 Current_Default_Rate_Pct,
       AVG(PD_Intercept)*100 Mean_Intercept_Recalibrated_PD_Pct
      FROM cases GROUP BY ALL ORDER BY Model,Risk_Tier,Prior_Delinquency_Severity
    """).fetchdf()
    migration.insert(0,"Table_Label","Prior-Delinquency Severity -> Current Loan Default Migration Table")
    migration.to_csv(PBI/"pbi_06_delinquency_migration.csv",index=False)

    # Figures from aggregates only.
    monthly=temporal[temporal.Grain=="Monthly"].copy()
    monthly["Period_Label"]=monthly.Period.astype(str)
    month_labels=sorted(monthly.Period_Label.unique())
    month_index={label:i for i,label in enumerate(month_labels)}
    monthly["Month_Index"]=monthly.Period_Label.map(month_index)
    monthly=monthly.assign(_bad_num=monthly.Actual_Bad_Rate_Pct*monthly.Applications,
                           _pd_num=monthly.Recalibrated_PD_Pct*monthly.Applications)
    cohort=monthly.groupby(["Month_Index","Period_Label","Split","Model"],as_index=False).agg(
        Applications=("Applications","sum"),_bad_num=("_bad_num","sum"),_pd_num=("_pd_num","sum"))
    cohort["Actual_Bad_Rate_Pct"]=cohort._bad_num/cohort.Applications
    cohort["Recalibrated_PD_Pct"]=cohort._pd_num/cohort.Applications
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    for ax,split in zip(axes,["Train_Dev","OOT_Valid","OOT_Monitor"]):
        sub=cohort[cohort.Split==split]
        actual=sub[sub.Model.eq("Champion Scorecard")].sort_values("Month_Index")
        ax.plot(actual.Month_Index,actual.Actual_Bad_Rate_Pct,marker="o",color="black",label="Observed actual bad rate")
        for color,(model,g) in zip(["tab:blue","tab:orange"],sub.groupby("Model")):
            g=g.sort_values("Month_Index")
            ax.plot(g.Month_Index,g.Recalibrated_PD_Pct,linestyle="--",color=color,label=f"{model} expected PD")
        ax.set_title(split);ax.set_ylabel("Rate (%)");ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
    ticks=np.arange(0,len(month_labels),2)
    axes[-1].set_xticks(ticks,[month_labels[i] for i in ticks],rotation=30,ha="right")
    axes[-1].set_xlabel("Decision month (YYYYMM)");fig.suptitle("Origination Cohort Risk Curves — all 22 months",y=.995);fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(FIGS/"stage6_cohort_vintage_trend.png",dpi=160);plt.close(fig)

    summ=stress[(stress.Risk_Tier=="Portfolio Total")&stress.Scenario.isin(scenario_specs)].copy()
    fig,ax=plt.subplots(figsize=(10,5.5)); x=np.arange(len(summ)); w=.38
    ax.bar(x-w/2,summ.Realized_Net_Profit_Stress_Proxy/1e6,w,label="Observed-target stress proxy")
    ax.bar(x+w/2,summ.Model_Implied_Expected_Net_Profit/1e6,w,label="Recalibrated-PD expected")
    ax.axhline(0,color="black",lw=.8);ax.set_xticks(x,[f"{s}\n{m.split()[0]}" for s,m in zip(summ.Scenario,summ.Model)],rotation=20,ha="right")
    ax.set_ylabel("Net profit (million currency units)");ax.set_title("Parametric Commercial Stress Simulations");ax.legend();ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(FIGS/"stage6_stress_scenarios.png",dpi=160);plt.close(fig)

    top=concentration.sort_values("Total_Recommended_Volume",ascending=False).head(15).copy()
    fig,ax=plt.subplots(figsize=(12,6)); labels=top.Model.str.replace(" Scorecard", "").str.replace(" LightGBM M4", " LGBM")+" | "+top.Mode+" | "+top.Risk_Tier+" | "+top.Income_Type+" | "+top.Ticket_Band
    ax.barh(labels,top.Exposure_Share_Pct)
    ax.invert_yaxis();ax.set_xlabel("Exposure share (%)");ax.set_title("Largest OOT_Monitor Exposure Segments");ax.grid(axis="x",alpha=.25);fig.tight_layout();fig.savefig(FIGS/"stage6_exposure_concentration.png",dpi=160);plt.close(fig)

    pbi_files=sorted(PBI.glob("*.csv"));total_bytes=sum(f.stat().st_size for f in pbi_files)
    pbi_audit=pd.DataFrame([{"File":f.name,"Rows":sum(1 for _ in f.open(encoding="utf-8"))-1,"Size_MB":f.stat().st_size/1e6} for f in pbi_files])
    if len(pbi_files)!=6 or total_bytes>=15_000_000:
        raise RuntimeError(f"Power BI export guard failed: {len(pbi_files)} files, {total_bytes/1e6:.2f} MB")

    # Selected auditable report tables.
    summary=stress[(stress.Risk_Tier=="Portfolio Total")&stress.Scenario.isin(scenario_specs)][[
       "Scenario","Model","Valid_Target_Approval_Pct","Monitor_Approval_Pct","Approved_Count","Observed_Bad_Rate_Pct","COF_Pct","LGD_Pct","Recommended_Volume","Realized_Net_Profit_Stress_Proxy","Model_Implied_Expected_Net_Profit"]]
    delinq_roll=migration.groupby(["Prior_Delinquency_Severity"],as_index=False).agg(Applications=("Application_Count","sum"),Current_Defaults=("Current_Defaults","sum"))
    delinq_roll["Current_Default_Rate_Pct"]=100*delinq_roll.Current_Defaults/delinq_roll.Applications
    concentration_top=concentration.sort_values("Total_Recommended_Volume",ascending=False).head(12)[["Model","Mode","Risk_Tier","Income_Type","Ticket_Band","Tenor_Band","Binding_Driver","Binding_Driver_Share_Pct","Approved_Count","Exposure_Share_Pct","Total_Recommended_Volume","Observed_Bad_Rate_Pct","Expected_Loss_Proxy","Risk_Tier_HHI","Income_Type_HHI","Ticket_Band_HHI","Tenor_Band_HHI"]]
    emergency_summary=stress[stress.Scenario=="S4_Emergency_Cutoff"][["Model","Emergency_Valid_Cutoff","Monitor_Approval_Pct","Realized_Net_Profit_Stress_Proxy","Model_Implied_Expected_Net_Profit"]]
    report=f"""# Stage 6 — Portfolio Risk Intelligence、Stress Scenarios 与 Power BI Mart

## 数据完整性与历史逾期字段核验

`train_applprev_2.parquet` 含 {ap2_rows:,} 条记录、字段 `{', '.join(ap2_schema.column_name.tolist())}`。其中只有 card-block reason、contact type、credit-account card status 和层级索引；没有 consecutive monthly DPD status，也没有可用于构造 MOB 的月度 performance snapshot。相关定义如下：

{markdown(defs)}

`train_applprev_1_*.parquet` 含 {ap1_rows:,} 条 previous-application records；`actualdpd_943P` 的定义为 “Days Past Due (DPD) of previous contract (actual).”，可用来构造 case-level prior-contract maximum DPD severity。`credit_bureau_a_1_*` 字段包括 `{', '.join(cb_schema.column_name[cb_schema.column_name.str.contains('dpd',case=False)].tolist())}`，是 DPD maxima / date tags 而不是连续月度状态快照。因此本报告采用 **Prior-Delinquency Severity -> Current Loan Default Migration Table**，没有伪造 roll-rate matrix 或 MOB curves。每位申请人先在 DuckDB 内聚合 previous contracts 的 maximum observed actual DPD，再按 prior-application history 分层；无历史记录及有 prior history 但 DPD missing 分开标示，避免误列为 0 DPD。

## 决策定义与数据边界

Risk tiers 用 Stage 5 `pd_*_recal_intercept`，阈值沿用 A `<1%`, B `<2%`, C `<4%`, D `<7%`, E `>=7%`。Operational approval cutoffs、growth cutoff 与 severe-stress emergency cutoff 均仅由 `OOT_Valid` 确定，随后 forward-apply 至 `OOT_Monitor`。Monitor `target` 只用于事后度量，不参与 cutoffs、stress emergency selection 或分箱拟合。Exposure 为 Stage 5/4 deterministic `min(requested, risk, affordability)` cap，经 hard policy decline 后的 approved recommended limit。

Profit 是参数化生命周期净现金流近似：利息收入扣 `COF`/`OPEX` 后减 `principal * loss rate * LGD`。S1/S2 的 realized estimate 使用 Monitor observed `target`；S3/S4 的 `Realized_Net_Profit_Stress_Proxy` 是 **observed target rate multiplied by stated scenario stress factor** 的情景损失代理，不是已观察到的真实 downturn P&L。Stress assumptions are not empirical forecasts.

## Origination Cohort & Portfolio Concentration

Temporal data contains weekly and monthly aggregates, model, split and intercept-calibrated `Risk_Tier`; eligible applications are approved under the corresponding Valid-frozen Iso-Risk cutoff for exposure reporting. The OOT Monitor months are cohort-level point-in-time outcomes, not MOB curves.

{markdown(concentration_top)}

## Audit Summary

### Prediction integrity by split

{markdown(split_audit)}

### Source and model diagnostics

| Dataset / scope | Rows | Null-rate / integrity check | Feature-quality check | Performance / stability evidence |
|---|---:|---|---|---|
| `train_applprev_1_*` historical rows | {ap1_rows:,} | `actualdpd_943P` null = {raw_dpd_null:.2f}% | Historical observed field; no Stage 6 IV/re-binning | Severity migration in `pbi_06` |
| `train_applprev_2` depth-2 rows | {ap2_rows:,} | `credacc_cards_status_52L` null = {applprev2_status_null:.2f}% | Card status/block flags only; no monthly DPD | No MOB or roll-rate claim |
| Champion Scorecard, OOT_Monitor | 151,707 | Stage 5 recalibrated PDs non-null; labels observed | No Stage 6 feature refit; Stage 2/3 IV and monotonicity gates retained | ROC-AUC 0.7751; KS 0.4222; Gini 0.5502; Score PSI 0.0049 |
| Raw LightGBM M4, OOT_Monitor | 151,707 | Stage 5 recalibrated PDs non-null; labels observed | No Stage 6 feature refit; Stage 2/3 gates retained | ROC-AUC 0.8257; KS 0.5006; Gini 0.6514; raw PD PSI 0.0772 |
| Drift diagnostics | 26 feature CSIs + score PSIs | All records retained in Power BI table 05 | No new binning learned on OOT | Highest OOT_Monitor CSI: `has_tax_registry_record` 1.3064; `person_education_level` 1.2974 |

`IV` and monotonicity are not re-estimated in Stage 6 because no model/features are fitted here; the saved Stage 2/3 screening results remain the governing evidence. `pbi_05_psi_csi_and_fairness.csv` contains all 26 feature CSIs, score PSIs, decile calibration rows and fairness rows.

## Empirical Prior-Delinquency Migration

{markdown(delinq_roll)}

Tier-level detailed results are in `pbi_06_delinquency_migration.csv`. It describes a cross-sectional migration from prior-contract severity into the current application target, not transitions between observed repayment months.

## Parametric Commercial Stress Simulations

{markdown(summary)}

`S4_Emergency_Cutoff` is a separate Valid-selected counterfactual. It selects the cutoff maximizing positive expected severe-stress profit on `OOT_Valid`, then forward-tests it on Monitor. If that forward empirical proxy remains non-positive, no claim of restored Monitor margin is made; the cutoff remains an unvalidated sensitivity result.

{markdown(emergency_summary)}

## Power BI Data Mart

| File | Rows | Size (MB) |
|---|---:|---:|
{pbi_audit.to_markdown(index=False)}

Total CSV size: **{total_bytes/1e6:.2f} MB** (<15 MB). Star-schema wiring and DAX are documented in `reports/powerbi_dashboard_blueprint.md`.

## Stage 6 Outputs

Six aggregate CSV tables under `data/powerbi/`; figures are `stage6_cohort_vintage_trend.png`, `stage6_stress_scenarios.png`, and `stage6_exposure_concentration.png`. Detailed cutoff metrics, PSI/CSI, decile calibration and fairness rows are in the Power BI tables.

**HARD STOP:** Stage 6 complete. Wait for human review before Stage 7.
"""
    (REPORTS/"stage6_portfolio_and_scenarios_report.md").write_text(report,encoding="utf-8")

    blueprint="""# Power BI 3-Page Risk Command Center — 数据模型与 DAX

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
"""
    (REPORTS/"powerbi_dashboard_blueprint.md").write_text(blueprint,encoding="utf-8")

    print("HISTORICAL APPLPREV_2 FINDING")
    print(f"rows={ap2_rows:,}; columns={', '.join(ap2_schema.column_name.tolist())}; no consecutive monthly DPD fields.")
    print("STRESS TEST SUMMARY")
    print(markdown(summary))
    print("POWER BI CSV EXPORTS")
    print(markdown(pbi_audit))
    print(f"TOTAL_CSV_MB={total_bytes/1e6:.3f}; HARD STOP before Stage 7")
    con.close()


if __name__ == "__main__":
    main()

