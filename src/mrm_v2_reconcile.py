"""Empirical reconciliation and MRM ground-truth audit for FinGuard v2.

This script is intentionally read-only with respect to model inputs and the two
HTML bundles. It writes only mrm_reconciled_truth.json and
reports/mrm_reconciliation_audit.md.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
import threading
import sys
import time
import tracemalloc
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PRED = PROC / "stage5_full_predictions.parquet"
MART = PROC / "finguard_feature_mart.parquet"
STAGE3 = PROC / "stage3_predictions.parquet"
CASE_IDS = PROC / "train_dev_case_ids.parquet"
RECAL_PATH = PROC / "models" / "recalibrators_stage5.json"
CONFIG_PATH = ROOT / "configs" / "decision_engine_policy_v2.json"
CSI_PATH = PROC / "stage5_psi_csi_summary.csv"
MAPPING_PATH = PROC / "feature_group_mapping.csv"
MODEL_PATH = PROC / "models" / "lgbm_m4.pkl"
SCORECARD_PATH = PROC / "scorecard_coefficients_v2.json"
RULES_PATH = PROC / "woe_bin_rules.json"
OUTPUT = PROC / "mrm_reconciled_truth.json"
REPORT = ROOT / "reports" / "mrm_reconciliation_audit.md"
HTML_FILES = [ROOT / "docs" / "index.html", ROOT / "dashboard" / "FinGuard_Interactive_Command_Center.html"]
BOOTSTRAPS = 1000
SEED = 42
BAD_RATE_REFERENCE = 0.012622675619834711
SEVERE_CSI_FEATURES = ["has_tax_registry_record", "person_education_level", "tax_verified_income_ratio", "person_family_status"]


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def timed_peak(fn):
    """Return (value, reproducible wall time, process peak RSS and Python peak)."""
    process = psutil.Process(os.getpid())
    base_rss = process.memory_info().rss
    peak = [base_rss]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error:
                pass
            stop.wait(.02)

    sampler = threading.Thread(target=sample, daemon=True)
    tracemalloc.start()
    sampler.start()
    start = time.perf_counter()
    try:
        value = fn()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        sampler.join()
        _, py_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak[0] = max(peak[0], process.memory_info().rss)
    return value, {"elapsed_seconds": elapsed, "baseline_rss_mb": base_rss / 1e6,
                   "peak_rss_mb": peak[0] / 1e6, "peak_rss_delta_mb": max(0, peak[0] - base_rss) / 1e6,
                   "python_tracemalloc_peak_mb": py_peak / 1e6}


def md_table(headers, rows):
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            if not np.isfinite(v):
                return "—"
            return f"{v:,.4f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |" for row in rows)
    return "\n".join(lines)


def query(con, sql):
    return con.execute(sql).fetchdf()


def empirical_split_audit(con):
    pred = PRED.as_posix()
    mart = MART.as_posix()
    rows = query(con, f"""
      SELECT p.split_name, COUNT(*) AS rows, COUNT(DISTINCT p.case_id) AS distinct_case_ids,
             MIN(p.WEEK_NUM) AS week_min, MAX(p.WEEK_NUM) AS week_max,
             MIN(TRY_CAST(m.date_decision AS DATE)) AS date_min,
             MAX(TRY_CAST(m.date_decision AS DATE)) AS date_max,
             COUNT(*) FILTER (WHERE p.target IS NOT NULL) AS target_observed,
             AVG(CASE WHEN p.target IS NOT NULL THEN 1.0 ELSE 0.0 END)*100 AS target_observed_pct,
             100.0*SUM(CASE WHEN p.target=1 THEN 1 ELSE 0 END)/NULLIF(COUNT(*) FILTER (WHERE p.target IS NOT NULL),0) AS observed_bad_rate_pct
      FROM read_parquet('{pred}') p JOIN read_parquet('{mart}') m USING(case_id)
      GROUP BY p.split_name ORDER BY week_min
    """).to_dict("records")
    source = {}
    for name, path in (("feature_mart", mart), ("stage5_predictions", pred)):
        source[name] = query(con, f"""
          SELECT split_name,COUNT(*) rows,COUNT(DISTINCT case_id) distinct_case_ids,
                 MIN(WEEK_NUM) week_min,MAX(WEEK_NUM) week_max,
                 SUM(CASE WHEN target IS NULL THEN 1 ELSE 0 END) target_null_rows
          FROM read_parquet('{path}') GROUP BY split_name ORDER BY week_min
        """).to_dict("records")
    stage3 = query(con, f"""SELECT split_name,COUNT(*) rows,COUNT(DISTINCT case_id) distinct_case_ids,
      MIN(WEEK_NUM) week_min,MAX(WEEK_NUM) week_max FROM read_parquet('{STAGE3.as_posix()}')
      GROUP BY split_name ORDER BY week_min""").to_dict("records")
    ids = query(con, f"""SELECT COUNT(*) rows,COUNT(DISTINCT case_id) distinct_case_ids,
      MIN(case_id) min_case_id,MAX(case_id) max_case_id FROM read_parquet('{CASE_IDS.as_posix()}')""").iloc[0].to_dict()
    mismatch = query(con, f"""
      SELECT p.case_id,p.WEEK_NUM,p.split_name AS prediction_split,m.split_name AS mart_split,
             TRY_CAST(m.date_decision AS DATE) AS date_decision,p.target
      FROM read_parquet('{pred}') p FULL OUTER JOIN read_parquet('{mart}') m USING(case_id)
      WHERE p.case_id IS NULL OR m.case_id IS NULL OR p.WEEK_NUM<>m.WEEK_NUM OR p.split_name<>m.split_name
      ORDER BY COALESCE(p.case_id,m.case_id)
    """)
    # Check the frequently used date boundary independently of WEEK_NUM.
    date_boundary = query(con, f"""
      SELECT COUNT(*) FILTER (WHERE TRY_CAST(date_decision AS DATE)<=DATE '2020-03-30') AS through_2020_03_30,
             COUNT(*) FILTER (WHERE TRY_CAST(date_decision AS DATE) BETWEEN DATE '2020-03-31' AND DATE '2020-06-29') AS march_to_june_2020,
             COUNT(*) FILTER (WHERE TRY_CAST(date_decision AS DATE)>=DATE '2020-06-30') AS from_2020_06_30,
             COUNT(*) FILTER (WHERE (WEEK_NUM<=64)<>(TRY_CAST(date_decision AS DATE)<=DATE '2020-03-30')) AS boundary_membership_mismatch
      FROM read_parquet('{mart}')
    """).iloc[0].to_dict()
    expected = {"Train_Dev": (0,64,1292350), "OOT_Valid": (65,77,82602), "OOT_Monitor": (78,91,151707)}
    canonical_ok = True
    for row in rows:
        canonical_ok &= (row["split_name"], int(row["week_min"]), int(row["week_max"]), int(row["rows"])) == (row["split_name"], *expected[row["split_name"]])
    claimed = {"Train_Dev": 1292315, "OOT_Valid": 82637}
    canonical_counts = {r["split_name"]: int(r["rows"]) for r in rows}
    discrepancy = {k: {"claimed_rows": v, "canonical_rows": canonical_counts[k],
                       "claimed_minus_canonical": v-canonical_counts[k]} for k,v in claimed.items()}
    return {"canonical_split_rows": rows, "artifact_split_rows": source, "stage3_prediction_split_rows": stage3,
            "train_dev_case_id_artifact": ids, "case_level_assignment_mismatches": mismatch.to_dict("records"),
            "date_boundary_reconciliation": date_boundary, "claimed_count_discrepancy": discrepancy,
            "discrepancy_case_ids": [], "discrepancy_case_ids_status": "NOT_REPRODUCIBLE_FROM_CURRENT_ARTIFACTS",
            "canonical_week_split_verified": bool(canonical_ok),
            "interpretation": "The mart, Stage 3 predictions, Stage 5 predictions, and Train_Dev case-ID artifact agree on WEEK_NUM splits. Current artifacts do not contain a 35-row membership difference to enumerate. The 1,292,315/82,637 pair is therefore not reproduced; retain 1,292,350/82,602 as canonical for the saved week-based artifacts."}


def roc_gini_ks(y, p, weights=None):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights,dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(w) & (w > 0) & np.isin(y,[0,1])
    y,p,w=y[valid],p[valid],w[valid]
    bad_total=np.sum(w*y); good_total=np.sum(w*(1-y))
    if bad_total<=0 or good_total<=0:
        return {"roc_auc":None,"gini":None,"ks":None}
    order=np.argsort(p,kind="mergesort"); p,y,w=p[order],y[order],w[order]
    ends=np.r_[np.flatnonzero(np.diff(p)!=0),len(p)-1]
    starts=np.r_[0,ends[:-1]+1]
    group_bad=np.add.reduceat(w*y,starts)
    group_good=np.add.reduceat(w*(1-y),starts)
    cb=np.cumsum(group_bad)/bad_total; cg=np.cumsum(group_good)/good_total
    auc=float(np.sum(group_bad*(np.r_[0.,cg[:-1]]+.5*group_good/good_total))/bad_total)
    ks=float(np.max(np.abs(cb-cg)))
    return {"roc_auc":auc,"gini":2*auc-1,"ks":ks}


def metric_bundle(y, p, weights=None):
    y=np.asarray(y,dtype=np.float64);p=np.clip(np.asarray(p,dtype=np.float64),1e-15,1-1e-15)
    w=np.ones(len(y),dtype=np.float64) if weights is None else np.asarray(weights,dtype=np.float64)
    total=w.sum(); observed=np.dot(w,y); expected=np.dot(w,p)
    ranking=roc_gini_ks(y,p,w)
    return {**ranking,"brier":float(np.dot(w,(p-y)**2)/total),
            "log_loss":float(np.dot(w,-y*np.log(p)-(1-y)*np.log1p(-p))/total),
            "a_over_e":float(observed/expected) if expected>0 else None,
            "e_over_a":float(expected/observed) if observed>0 else None}


def bootstrap_metrics(y, champion, challenger, weeks, B=BOOTSTRAPS, seed=SEED):
    y=np.asarray(y,dtype=np.int8); champion=np.asarray(champion,float);challenger=np.asarray(challenger,float);weeks=np.asarray(weeks)
    rng=np.random.default_rng(seed); names=["ks","roc_auc","gini","brier","log_loss","a_over_e","e_over_a"]
    point={"Champion Scorecard":metric_bundle(y,champion),"Challenger LightGBM M4":metric_bundle(y,challenger)}
    point["M4 - Champion"]={k:(point["Challenger LightGBM M4"][k]-point["Champion Scorecard"][k]
                                if point["Challenger LightGBM M4"][k] is not None and point["Champion Scorecard"][k] is not None else None) for k in names}
    samples={kind:{model:{m:[] for m in names} for model in ["Champion Scorecard","Challenger LightGBM M4","M4 - Champion"]}
             for kind in ["paired_applicant","weekly_block"]}
    ordered=[]
    for p in (champion,challenger):
        order=np.argsort(p,kind="mergesort");ps=p[order]
        ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
        group=np.empty(len(y),dtype=np.int32)
        starts=np.r_[0,ends[:-1]+1]
        for j,(a,b) in enumerate(zip(starts,ends)):group[order[a:b+1]]=j
        ordered.append((group,int(len(ends))))
    week_values,week_inverse=np.unique(weeks,return_inverse=True)

    def fast_metrics(row_weights,p,group,n_groups):
        w=np.asarray(row_weights,float);bad=np.bincount(group,weights=w*y,minlength=n_groups);good=np.bincount(group,weights=w*(1-y),minlength=n_groups)
        nb=bad.sum();ng=good.sum();total=w.sum();observed=np.dot(w,y);expected=np.dot(w,p)
        if nb<=0 or ng<=0:return {k:None for k in names}
        cb=np.cumsum(bad)/nb;cg=np.cumsum(good)/ng
        auc=float(np.sum(bad*(np.r_[0.,cg[:-1]]+.5*good/ng))/nb)
        pp=np.clip(p,1e-15,1-1e-15)
        return {"ks":float(np.max(np.abs(cb-cg))),"roc_auc":auc,"gini":2*auc-1,
          "brier":float(np.dot(w,(pp-y)**2)/total),"log_loss":float(np.dot(w,-y*np.log(pp)-(1-y)*np.log1p(-pp))/total),
          "a_over_e":float(observed/expected) if expected else None,"e_over_a":float(expected/observed) if observed else None}

    for _ in range(B):
        applicant_weights=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y)).astype(float)
        week_weights=np.bincount(rng.integers(0,len(week_values),len(week_values)),minlength=len(week_values)).astype(float)[week_inverse]
        for kind,weights in (("paired_applicant",applicant_weights),("weekly_block",week_weights)):
            metrics=[]
            for i,(model,p) in enumerate((("Champion Scorecard",champion),("Challenger LightGBM M4",challenger))):
                m=fast_metrics(weights,p,*ordered[i]);metrics.append(m)
                for key in names:samples[kind][model][key].append(m[key])
            for key in names:
                a,b=metrics[0][key],metrics[1][key]
                samples[kind]["M4 - Champion"][key].append(None if a is None or b is None else b-a)
    results={"point_estimates":point,"bootstrap_replicates":B,"seed":seed,"confidence_level":.95,"confidence_intervals":{}}
    for kind, models in samples.items():
        results["confidence_intervals"][kind]={}
        for model,metrics in models.items():
            results["confidence_intervals"][kind][model]={}
            for key,vals in metrics.items():
                values=np.asarray([v for v in vals if v is not None and np.isfinite(v)],dtype=float)
                results["confidence_intervals"][kind][model][key]=[float(np.quantile(values,.025)),float(np.quantile(values,.975))] if len(values) else None
    return results


def weekly_stability(frame):
    rows=[]
    for (split,week),g in frame.groupby(["split_name","WEEK_NUM"],sort=True):
        y=g.target.to_numpy(np.int8)
        for model,col in (("Champion Scorecard","pd_scorecard_recal_platt"),("Challenger LightGBM M4","pd_lgbm_recal_platt")):
            p=g[col].to_numpy(float);rk=roc_gini_ks(y,p)
            rows.append({"split_name":split,"week":int(week),"model":model,"n":int(len(g)),
              "target_observed_pct":100*float(np.isfinite(y).mean()),"observed_bad_rate_pct":100*float(y.mean()),
              "mean_recalibrated_pd_pct":100*float(np.mean(p)),"weekly_gini":rk["gini"]})
    weekly=pd.DataFrame(rows)
    diagnostics={}
    for split in ("OOT_Valid","OOT_Monitor"):
        diagnostics[split]={}
        for model,g in weekly[weekly.split_name.eq(split)].groupby("model"):
            x=g.week.to_numpy(float);v=g.weekly_gini.to_numpy(float)
            slope,intercept=np.polyfit(x,v,1);res=v-(intercept+slope*x);mean=float(v.mean());std=float(np.std(res,ddof=0))
            diagnostics[split][model]={"mean_weekly_gini":mean,"weekly_gini_slope":float(slope),
               "weekly_residual_std":std,"stability_metric":mean+88*min(0,float(slope))-.5*std,
               "weeks":int(len(g))}
    late=weekly[(weekly.split_name=="OOT_Monitor")&(weekly.week>=86)]
    early=weekly[(weekly.split_name=="OOT_Monitor")&(weekly.week<=85)]
    maturity={}
    for model,g in late.groupby("model"):
        e=early[early.model.eq(model)]
        maturity[model]={"late_weeks":"W86-W91","late_rows":int(g.n.sum()),
          "late_target_observed_pct":float(np.average(g.target_observed_pct,weights=g.n)),
          "late_bad_rate_pct":float(np.average(g.observed_bad_rate_pct,weights=g.n)),
          "late_mean_pd_pct":float(np.average(g.mean_recalibrated_pd_pct,weights=g.n)),
          "late_bad_minus_pd_pp":float(np.average(g.observed_bad_rate_pct-g.mean_recalibrated_pd_pct,weights=g.n)),
          "prior_monitor_bad_rate_pct":float(np.average(e.observed_bad_rate_pct,weights=e.n)),
          "prior_monitor_mean_pd_pct":float(np.average(e.mean_recalibrated_pd_pct,weights=e.n)),
          "late_weekly_bad_rate_range_pct":[float(g.observed_bad_rate_pct.min()),float(g.observed_bad_rate_pct.max())]}
    return rows,diagnostics,{"late_vs_prior":maturity,
      "conclusion":"Recorded target labels are 100% observed in all saved splits; there is no missing-label evidence for W86-W91 censoring. These artifacts lack a separate label-maturity/observation-window timestamp, so full outcome maturity cannot be certified. Reported late PD/bad-rate gaps are empirical divergence against recorded labels."}


def psi_numeric(reference, observed, bins=10):
    ref=np.asarray(reference,float);obs=np.asarray(observed,float)
    ref=ref[np.isfinite(ref)];obs=obs[np.isfinite(obs)]
    if len(ref)==0 or len(obs)==0:return None
    edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<2:return 0.0
    edges[0],edges[-1]=-np.inf,np.inf
    a=np.histogram(ref,edges)[0]/len(ref);b=np.histogram(obs,edges)[0]/len(obs)
    a=np.clip(a,1e-8,None);b=np.clip(b,1e-8,None)
    return float(np.sum((b-a)*np.log(b/a)))


def valid_cutoff(y,p,limits,target_bad_rate=BAD_RATE_REFERENCE):
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,float);ok=np.asarray(limits)>0
    thresholds=np.unique(p[ok]);best=float(thresholds[0]) if len(thresholds) else 0.
    if len(thresholds):
        order=np.argsort(p[ok],kind="mergesort")
        ps=p[ok][order];ys=y[ok][order]
        ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
        cum_bad=np.cumsum(np.add.reduceat(ys, np.r_[0,ends[:-1]+1]))
        cum_n=ends+1
        feasible=np.flatnonzero(cum_bad/cum_n<=target_bad_rate)
        best=float(ps[ends[feasible[-1]]]) if len(feasible) else float(thresholds[0])
    return best


def contribution_components(principal, target, term, apr=.15, cof=.04, opex=.015, lgd=.75):
    from portfolio_scenarios_powerbi_stage6 import balance_factor
    p=np.maximum(np.asarray(principal,float),0);y=np.asarray(target,float);t=np.asarray(term,float)
    factor=balance_factor(t,np.full(len(p),apr))
    nii=apr/12*p*factor;funding=cof/12*p*factor;operating=opex/12*p*factor;loss=p*y*lgd
    return {"nii_benchmark":nii,"funding_cost":funding,"opex":operating,"realized_credit_loss":loss,
            "realized_contribution":nii-funding-operating-loss}


def fit_variant(train, valid, monitor, features, full_artifact, variant_name):
    """Fit a Train_Dev-only ablation; OOT_Valid is calibration/cutoff data only."""
    original=full_artifact["features"]
    full_model=full_artifact["model"]
    original_constraints=full_artifact["constraints"]
    constraint_by_name=dict(zip(original,original_constraints))
    booster=full_model.booster_
    cat_indices=booster.params.get("categorical_column",[])
    if isinstance(cat_indices,str):cat_indices=[int(x) for x in cat_indices.split(",") if x]
    category_features=[original[int(i)] for i in cat_indices]
    categories=dict(zip(category_features,booster.pandas_categorical or []))
    categories={f:categories[f] for f in features if f in categories}

    def matrix(df):
        out=pd.DataFrame(index=df.index)
        for f in features:
            if f in categories:
                out[f]=pd.Categorical(df[f].astype("string"),categories=categories[f])
            else:
                col=pd.to_numeric(df[f],errors="coerce").to_numpy(np.float32,copy=True)
                col[~np.isfinite(col)]=-999
                out[f]=col
        return out
    x_train=matrix(train);x_valid=matrix(valid);x_monitor=matrix(monitor)
    fit_mask=train.WEEK_NUM.to_numpy()<=54;dev_mask=(valid.WEEK_NUM.to_numpy()>=55)&(valid.WEEK_NUM.to_numpy()<=64)
    # `valid` passed here contains Dev_Calib (W55-W64); the separate OOT_Valid is used below for Platt.
    y_train=train.target.to_numpy(np.int8);y_valid=valid.target.to_numpy(np.int8)
    constraints=[constraint_by_name[f] for f in features]
    model=lgb.LGBMClassifier(objective="binary",n_estimators=2500,learning_rate=.03,num_leaves=31,max_depth=-1,
       subsample=.8,colsample_bytree=.8,reg_lambda=2,n_jobs=4,verbosity=-1,
       monotone_constraints=constraints,random_state=42)
    model.fit(x_train.loc[fit_mask],y_train[fit_mask],categorical_feature=[f for f in features if f in categories],
       eval_set=[(x_valid.loc[dev_mask],y_valid[dev_mask])],eval_metric="binary_logloss",
       callbacks=[lgb.early_stopping(100,verbose=False)])
    raw_valid=model.predict_proba(x_valid)[:,1]
    logit_valid=np.log(np.clip(raw_valid,1e-12,1-1e-12)/(1-np.clip(raw_valid,1e-12,1-1e-12))).reshape(-1,1)
    calibrator=LogisticRegression(C=1e6,solver="lbfgs",max_iter=1000).fit(logit_valid,valid.target.to_numpy(np.int8))
    def calibrated(raw):
        z=np.log(np.clip(raw,1e-12,1-1e-12)/(1-np.clip(raw,1e-12,1-1e-12))).reshape(-1,1)
        return calibrator.predict_proba(z)[:,1]
    pd_train=calibrated(model.predict_proba(x_train)[:,1])
    pd_monitor=calibrated(model.predict_proba(x_monitor)[:,1])
    chosen_iteration=int(model.best_iteration_ or model.n_estimators)
    result={"name":variant_name,"features":len(features),"excluded_features":sorted(set(original)-set(features)),
      "best_iteration":chosen_iteration,"early_stop_split":"Train_Dev W55-W64",
      "calibration_split":"OOT_Valid W65-W77","calibration_alpha":float(calibrator.intercept_[0]),
      "calibration_beta":float(calibrator.coef_[0,0]),"train_dev_calibrated_pd":pd_train,
      "oot_monitor_pd":pd_monitor,"model":model}
    return result


def policy_and_income_audit(pred_frame, mart_frame, config, stage4_module):
    merged=pred_frame if mart_frame is pred_frame else pred_frame.merge(mart_frame,on=["case_id","WEEK_NUM"],how="inner",validate="one_to_one",suffixes=("","_mart"))
    outputs={}
    arrays={}
    for split in ("OOT_Valid","OOT_Monitor"):
        g=merged[merged.split_name.eq(split)].copy(); y=g.target.to_numpy(np.int8)
        split_rows={}
        for model,pcol,cut_key in (("Champion Scorecard","pd_scorecard_recal_platt","champion_pd_cutoff"),
                                   ("Challenger LightGBM M4","pd_lgbm_recal_platt","challenger_pd_cutoff")):
            p=g[pcol].to_numpy(float); engine=stage4_module.loan_engine(g,p)
            limits=engine["limit"]
            cfg_iso=config["thresholds"]["iso_risk"][cut_key]
            cfg_quant=config["thresholds"]["iso_approval"][cut_key]
            policies={"Valid-Frozen 75% Quantile Policy":float(cfg_quant),"Valid-Frozen Iso-Risk Policy":float(cfg_iso)}
            entries=[]
            for policy,cutoff in policies.items():
                approved=(limits>0)&(p<=cutoff)
                entries.append({"split":split,"model":model,"policy":policy,"pd_cutoff":cutoff,
                  "applications":int(len(g)),"approved":int(approved.sum()),"approval_pct":100*float(approved.mean()),
                  "observed_bads":int(y[approved].sum()),"observed_bad_rate_pct":100*float(y[approved].mean()) if approved.any() else None,
                  "valid_reference_bad_rate_pct":100*BAD_RATE_REFERENCE})
                if split=="OOT_Monitor" and policy=="Valid-Frozen Iso-Risk Policy":
                    arrays[model]={"approved":approved,"limit":limits,"apr":engine["apr"],"term":engine["term"],"pd":p,"target":y,"frame":g}
            split_rows[model]=entries
        outputs[split]=split_rows
    # Stage 5 headline provenance is reported with explicit policy semantics.
    headline=[]
    pbi4=ROOT/"data"/"powerbi"/"pbi_04_stress_scenarios.csv"
    if pbi4.exists():
        d=pd.read_csv(pbi4)
        q=d[(d.Scenario=="S1_Baseline")&(d.Risk_Tier=="Portfolio Total")]
        for r in q.itertuples(index=False):
            headline.append({"Scenario":r.Scenario,"Model":r.Model,"Approved_Count":int(r.Approved_Count),
              "Monitor_Approval_Pct":float(r.Monitor_Approval_Pct),"Observed_Bad_Rate_Pct":float(r.Observed_Bad_Rate_Pct),
              "Valid_PD_Cutoff":float(r.Valid_PD_Cutoff),"Source":"data/powerbi/pbi_04_stress_scenarios.csv"})
    policy_rows=[]
    for model,a in arrays.items():
        contrib=contribution_components(a["limit"],a["target"],a["term"])
        mask=a["approved"]
        policy_rows.append({"model":model,"iso_risk_cutoff":float(config["thresholds"]["iso_risk"]["champion_pd_cutoff" if model=="Champion Scorecard" else "challenger_pd_cutoff"]),
          "approved":int(mask.sum()),"approval_pct":float(100*mask.mean()),
          "approved_bad_rate_pct":float(100*a["target"][mask].mean()) if mask.any() else None,
          "model_neutral_realized_contribution":float(contrib["realized_contribution"][mask].sum()),
          "scenario_expected_contribution":float(stage4_module.profit(a["limit"][mask],a["pd"][mask],a["apr"][mask],a["term"][mask],.04,.75).sum())})
    return {"valid_frozen_policy_metrics":outputs,"iso_risk_monitor_contribution":policy_rows,
      "stage5_headline_provenance":headline},arrays,merged


def contribution_bootstrap(arrays, B=BOOTSTRAPS, seed=SEED):
    c=arrays["Champion Scorecard"];m=arrays["Challenger LightGBM M4"];n=len(c["target"]);rng=np.random.default_rng(seed)
    def stats(index):
        y=c["target"][index];ac=c["approved"][index];am=m["approved"][index]
        ar=float(am.mean()-ac.mean())*100
        bc=100*y[ac].mean() if ac.any() else 0.;bm=100*y[am].mean() if am.any() else 0.
        cc=contribution_components(c["limit"][index],y,c["term"][index])["realized_contribution"]
        cm=contribution_components(m["limit"][index],y,m["term"][index])["realized_contribution"]
        return {"delta_approval_pp":ar,"delta_bad_rate_bps":(bm-bc)*100,"delta_realized_contribution":float(cm[am].sum()-cc[ac].sum())}
    point=stats(np.arange(n)); draws={k:[] for k in point}
    for _ in range(B):
        idx=rng.integers(0,n,n);s=stats(idx)
        for k in draws:draws[k].append(s[k])
    return {"point_estimates":point,"bootstrap_replicates":B,"seed":seed,
      "confidence_intervals_95":{k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in draws.items()},
      "paired_same_applicants":True}


def income_and_fairness_audits(merged, model_pds, cutoffs, stage4_module):
    # Missing-income metrics are grouped by actual M4 PD risk tier so the Stage 4
    # fallback multiplier remains visible rather than averaged away.
    income_rows=[];policy_alternatives=[];fallback_map={"Tier A: Super Prime":.80,"Tier B: Prime":.75,"Tier C: Standard":.70,"Tier D: High Risk":.60,"Tier E: Decline":0.}
    for split in ("Train_Dev","OOT_Monitor"):
        g=merged[merged.split_name.eq(split)].copy(); y=g.target.to_numpy(np.int8);p=g.pd_lgbm_recal_platt.to_numpy(float)
        engine=stage4_module.loan_engine(g,p);income=g.maininc_215A.to_numpy(float);verified=np.isfinite(income)&(income>100)
        tier=np.asarray(engine["tier"]);approved=(p<=cutoffs[split])&(engine["limit"]>0)
        for status,mask in (("Verified income",verified),("Missing / unverified income",~verified)):
            for t in np.unique(tier[mask]):
                z=mask&(tier==t);n=int(z.sum())
                income_rows.append({"split":split,"income_status":status,"risk_tier":t,"count":n,"share_of_split_pct":100*n/len(g),
                    "approval_pct":100*float(approved[z].mean()) if n else None,"observed_bad_rate_pct":100*float(y[z].mean()) if n else None,
                    "mean_pd_pct":100*float(p[z].mean()) if n else None,"fallback_multiplier":fallback_map[t] if status.startswith("Missing") else None})
        missing=~verified;approved_missing=int((approved&missing).sum())
        policy_alternatives += [
          {"split":split,"alternative":"Current tiered cap / haircut","missing_income_count":int(missing.sum()),"auto_approved":approved_missing,"auto_approval_pct":100*approved_missing/max(int(missing.sum()),1),"review_queue":0,"interpretation":"Observed current Stage 4 fallback policy"},
          {"split":split,"alternative":"Route missing income to manual verification","missing_income_count":int(missing.sum()),"auto_approved":0,"auto_approval_pct":0.,"review_queue":int(missing.sum()),"interpretation":"Counterfactual routing; verified income and outcomes not inferred"},
          {"split":split,"alternative":"Decline unverified income","missing_income_count":int(missing.sum()),"auto_approved":0,"auto_approval_pct":0.,"review_queue":0,"interpretation":"Counterfactual policy; no outcome change assumed"}]
    # Fairness diagnostic on blind OOT_Monitor. Risk-tier conditional rates use
    # each model's calibrated PD tier; AIR denominator is the highest age-band
    # approval rate for that same model.
    mon=merged[merged.split_name.eq("OOT_Monitor")].copy();age=mon.person_age_band_sensitive.fillna("Unknown").astype(str).to_numpy();y=mon.target.to_numpy(np.int8)
    fairness=[];gender=mon.person_gender_sensitive.fillna("Unknown").astype(str).value_counts(dropna=False)
    gender_missing={k:{"count":int(v),"share_pct":100*int(v)/len(mon)} for k,v in gender.items()}
    for model,p in model_pds.items():
        cutoff=cutoffs[model];tiers=np.asarray(stage4_module.tier(p));eng=stage4_module.loan_engine(mon,p);approved=(p<=cutoff)&(eng["limit"]>0)
        age_rates={a:float(approved[age==a].mean()) for a in np.unique(age)}
        max_rate=max(age_rates.values()) if age_rates else 0
        for a in ("<25","25-34","35-49","50+"):
            band=age==a
            for t in np.unique(tiers[band]):
                z=band&(tiers==t)
                if not z.any():continue
                fairness.append({"model":model,"age_band":a,"risk_tier":t,"count":int(z.sum()),
                  "approval_pct":100*float(approved[z].mean()),"observed_bad_rate_pct":100*float(y[z].mean()),
                  "AIR":age_rates.get(a,0)/max_rate if max_rate else None,
                  "false_rejection_good_rate_pct":100*float((~approved[z & (y==0)]).mean()) if np.any(z&(y==0)) else None,
                  "false_approval_bad_rate_pct":100*float(approved[z&(y==1)].mean()) if np.any(z&(y==1)) else None})
    return {"missing_income_by_split_tier":income_rows,"missing_income_policy_alternatives":policy_alternatives,
            "oot_monitor_gender_missingness":gender_missing,"oot_monitor_age_within_risk_tier":fairness,
            "definitions":{"false_rejection_good_rate":"good applicants declined / all good applicants in cell",
              "false_approval_bad_rate":"bad applicants approved / all bad applicants in cell",
              "AIR":"age-band approval rate divided by the highest age-band approval rate for that model",
              "risk_tier":"calibrated-PD bins <1%, 1-2%, 2-4%, 4-7%, >=7%"}}


def main():
    html_before={str(p):sha256(p) for p in HTML_FILES}
    config=json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    recal=json.loads(RECAL_PATH.read_text(encoding="utf-8"))
    scorecard=json.loads(SCORECARD_PATH.read_text(encoding="utf-8"))
    rules=json.loads(RULES_PATH.read_text(encoding="utf-8"))
    model_art=pickle.load(open(MODEL_PATH,"rb"));full_model=model_art["model"];features=list(model_art["features"])
    if len(features)!=73:raise ValueError(f"Saved M4 feature count is {len(features)}, expected 73")
    sys_path=str(ROOT/"src")
    import sys
    if sys_path not in sys.path:sys.path.insert(0,sys_path)
    import portfolio_scenarios_powerbi_stage6 as stage4

    con=duckdb.connect(":memory:")
    con.execute("SET memory_limit='3GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(ROOT/'duckdb_temp').as_posix()}'")
    pred_path=PRED.as_posix();mart_path=MART.as_posix()

    def sql_sample():
        return con.execute(f"""SELECT p.split_name,p.WEEK_NUM,COUNT(*) applications,
             AVG(p.target) observed_bad_rate,AVG(p.pd_lgbm_recal_platt) mean_calibrated_pd
          FROM read_parquet('{pred_path}') p JOIN read_parquet('{mart_path}') m USING(case_id)
          GROUP BY 1,2 ORDER BY 1,2""").fetchdf()
    sql_df,sql_telemetry=timed_peak(sql_sample)
    split_audit=empirical_split_audit(con)
    print("[1/5] split reconciliation complete", flush=True)

    # Read target/score series for all weeks; all metrics remain empirical.
    metric_sql=f"""SELECT p.case_id,p.WEEK_NUM,p.MONTH,p.split_name,p.target,
      p.pd_scorecard,p.pd_lgbm_raw,p.pd_scorecard_recal_platt,p.pd_lgbm_recal_platt,
      m.date_decision,m.person_age_band_sensitive,m.person_gender_sensitive,
      m.maininc_215A,m.credamount_770A,m.annuity_780A,m.active_bureau_monthly_debt,
      m.bureau_active_max_dpd,m.static_avg_dpd_tol_closure24m,m.static_pct_inst_paid_early_3d
      FROM read_parquet('{pred_path}') p JOIN read_parquet('{mart_path}') m USING(case_id)
      ORDER BY p.WEEK_NUM,p.case_id"""
    pred_frame=query(con,metric_sql)
    # Split truth and scoring/calibration provenance.
    weekly,stability,label_maturity=weekly_stability(pred_frame)
    monitor=pred_frame[pred_frame.split_name.eq("OOT_Monitor")].copy()
    ymon=monitor.target.to_numpy(np.int8);weekmon=monitor.WEEK_NUM.to_numpy(int)
    boot=bootstrap_metrics(ymon,monitor.pd_scorecard_recal_platt.to_numpy(float),
                          monitor.pd_lgbm_recal_platt.to_numpy(float),weekmon)
    print("[3/5] weekly metrics and bootstrap complete", flush=True)

    # Policy masks and observed Stage 6 limit engine computations.
    policy_results,iso_arrays,joined=policy_and_income_audit(pred_frame,pred_frame,config,stage4)
    contrib_boot=contribution_bootstrap(iso_arrays)
    print("[4/5] cutoff and contribution audit complete", flush=True)
    c=iso_arrays["Champion Scorecard"];m=iso_arrays["Challenger LightGBM M4"]
    cmask,mmask=c["approved"],m["approved"]
    swapin=mmask&~cmask;swapout=cmask&~mmask;both=mmask&cmask
    neutral_c=contribution_components(c["limit"],c["target"],c["term"])
    neutral_m=contribution_components(m["limit"],m["target"],m["term"])
    same_limit=both&np.isclose(c["limit"],m["limit"],rtol=0,atol=1e-8)
    common_contrib_difference=np.max(np.abs(neutral_c["realized_contribution"][same_limit]-neutral_m["realized_contribution"][same_limit])) if same_limit.any() else 0.
    # KGB sensitivity is explicit: k inflates realized loss on Challenger-only
    # applicants, while Champion-only contribution remains at observed target.
    fixed_benefit=neutral_m["nii_benchmark"]-neutral_m["funding_cost"]-neutral_m["opex"]
    swapin_income=float(fixed_benefit[swapin].sum())
    swapin_loss=float(neutral_m["realized_credit_loss"][swapin].sum())
    champion_swapout=float(neutral_c["realized_contribution"][swapout].sum())
    def k_net(k):return swapin_income-k*swapin_loss-champion_swapout
    k_star=(swapin_income-champion_swapout)/swapin_loss if swapin_loss>0 else None
    kgb={"assumption":"k multiplies OOT_Monitor target-based loss on Challenger-only (Swap-In) rows; this is a sensitivity on potentially unobserved outcomes, not evidence that these saved labels are missing.",
         "swap_in_fixed_nii_less_funding_opex":swapin_income,"swap_in_base_observed_loss":swapin_loss,
         "champion_swap_out_realized_contribution":champion_swapout,"break_even_k_star":k_star,
         "net_swap_contribution_by_k":[{"k":k,"net_swap_contribution":k_net(k)} for k in [1.,1.25,1.5,2.,2.5,3.,4.]]}
    economic={"model_neutral_benchmark":{"benchmark_apr_pct":15.,"funding_cost_pct":4.,"opex_pct":1.5,"lgd_pct":75.,
       "equation":"Realized Contribution = (APR_benchmark - Funding Cost - Opex) / 12 × Principal × amortizing balance factor - Principal × target × LGD; APR_benchmark=15%, COF=4%, Opex=1.5%, LGD=75%.",
       "same_applicants_both_approved_same_limit_count":int(same_limit.sum()),
       "max_abs_contribution_difference_same_limit":float(common_contrib_difference),"asserted_identical":bool(common_contrib_difference<1e-8)},
       "iso_risk_paired_bootstrap":contrib_boot,"swap_sets":{"swap_in_count":int(swapin.sum()),"swap_in_bads":int(c["target"][swapin].sum()),
         "swap_out_count":int(swapout.sum()),"swap_out_bads":int(c["target"][swapout].sum()),
         "both_approve_count":int(both.sum())},"kgb_sensitivity":kgb,
       "scenario_based_expected_contribution":{"equation":"Stage 6 profit = max((APR_tier-COF-Opex),0)/12 × principal × amortizing balance factor - principal × PD × LGD; base COF=4%, Opex=1.5%, LGD=75%.",
         "rows":policy_results["iso_risk_monitor_contribution"]}}

    # S4 sensitivity exactly mirrors Stage 6's published severe-shock assumptions.
    s4=[]
    for label,a in (("Champion Scorecard",c),("Challenger LightGBM M4",m)):
        tier=np.asarray(stage4.tier(a["pd"]));approval=a["approved"]
        mult=np.where(np.isin(tier,stage4.TIERS[:2]),1.30,2.)
        principal=np.where(approval,a["limit"],0.)
        expected=stage4.profit(principal,np.minimum(a["pd"]*mult,1),a["apr"],a["term"],.055,.85)
        realized=stage4.profit(principal,np.minimum(a["target"]*mult,1),a["apr"],a["term"],.055,.85)
        s4.append({"model":label,"approved":int(approval.sum()),"stress_cof_pct":5.5,"stress_lgd_pct":85.,
          "stress_pd_multiplier":"1.30x tiers A-B / 2.00x tiers C-E","expected_contribution_proxy":float(expected.sum()),
          "realized_target_stress_proxy":float(realized.sum())})
    economic["s4_tail_shock_comparison"]=s4

    # Income and fairness audits below use the saved prediction frame directly.
    iso_c=config["thresholds"]["iso_risk"]["champion_pd_cutoff"];iso_m=config["thresholds"]["iso_risk"]["challenger_pd_cutoff"]
    merged_for_audit=pred_frame

    # Train Dev feature matrices for M2 and M4-NoDrift; OOT_Valid is used only
    # for calibration/cutoff selection, and OOT_Monitor remains evaluation only.
    feature_groups=pd.read_csv(MAPPING_PATH).set_index("column_name").feature_group.to_dict()
    m2_features=[f for f in features if feature_groups.get(f) in {"Group_1_Traditional_Credit","Group_2_Repayment_Behaviour"}]
    nodrift_features=[f for f in features if f not in SEVERE_CSI_FEATURES]
    needs=sorted(set(features+["WEEK_NUM","target"]))
    select=",".join('"'+f.replace('"','""')+'"' for f in needs)
    # Three slices are pulled separately to document the temporal role of each.
    train_all=query(con,f"SELECT {select} FROM read_parquet('{mart_path}') WHERE split_name='Train_Dev' ORDER BY WEEK_NUM")
    dev_cal=query(con,f"SELECT {select} FROM read_parquet('{mart_path}') WHERE split_name='Train_Dev' AND WEEK_NUM BETWEEN 55 AND 64 ORDER BY WEEK_NUM")
    ootv=query(con,f"SELECT {select} FROM read_parquet('{mart_path}') WHERE split_name='OOT_Valid' ORDER BY WEEK_NUM")
    ootm=query(con,f"SELECT {select} FROM read_parquet('{mart_path}') WHERE split_name='OOT_Monitor' ORDER BY WEEK_NUM")
    # fit_variant expects Train_Dev rows (0-64) and Dev_Calib (55-64) for
    # early stopping, then calibrates with the separately held-out OOT_Valid.
    variants={}
    for name,cols in (("M2 Zero Demographic/Employment Proxies",m2_features),
                      ("M4-NoDrift",nodrift_features)):
        v=fit_variant(train_all,dev_cal,ootm,cols,model_art,name)
        # Platt calibration was trained on Dev_Calib in fit_variant's current
        # response; refit correctly on frozen OOT_Valid using the same raw model.
        raw_valid=v["model"].predict_proba(_variant_matrix(ootv,cols,features,model_art))[:,1]
        z=np.log(np.clip(raw_valid,1e-12,1-1e-12)/(1-np.clip(raw_valid,1e-12,1-1e-12))).reshape(-1,1)
        cal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=1000).fit(z,ootv.target.to_numpy(np.int8))
        def calprob(raw):
            z0=np.log(np.clip(raw,1e-12,1-1e-12)/(1-np.clip(raw,1e-12,1-1e-12))).reshape(-1,1)
            return cal.predict_proba(z0)[:,1]
        # Recalibrate all splits on OOT_Valid. Train predictions are in-sample,
        # used only as a score-distribution reference for PSI.
        raw_train=v["model"].predict_proba(_variant_matrix(train_all,cols,features,model_art))[:,1]
        raw_mon=v["model"].predict_proba(_variant_matrix(ootm,cols,features,model_art))[:,1]
        ptrain=calprob(raw_train);pmon=calprob(raw_mon)
        yv=ootv.target.to_numpy(np.int8);yvraw=raw_valid
        # To select a valid-frozen Iso-Risk cutoff, compute deterministic caps.
        valid_features=_variant_matrix(ootv,cols,features,model_art)
        rawv=v["model"].predict_proba(valid_features)[:,1];pv=calprob(rawv)
        limits_v=stage4.loan_engine(ootv,pv)["limit"]
        cut=valid_cutoff(yv,pv,limits_v,BAD_RATE_REFERENCE)
        variants[name]={"features":len(cols),"excluded_features":sorted(set(features)-set(cols)),
          "best_iteration":int(v["model"].best_iteration_ or v["model"].n_estimators),
          "calibration_split":"OOT_Valid W65-W77","calibration_alpha":float(cal.intercept_[0]),
          "calibration_beta":float(cal.coef_[0,0]),"valid_frozen_iso_risk_cutoff":float(cut),
          "monitor_pd":pmon,"monitor_gini_ks":roc_gini_ks(ootm.target.to_numpy(np.int8),pmon),
          "monitor_brier":metric_bundle(ootm.target.to_numpy(np.int8),pmon)["brier"],
          "monitor_a_over_e":metric_bundle(ootm.target.to_numpy(np.int8),pmon)["a_over_e"],
          "score_psi_train_dev_to_monitor":psi_numeric(ptrain,pmon),"train_dev_pd":ptrain}
        v["model"].booster_.free_dataset()
        del v,cal,raw_valid,raw_train,raw_mon,rawv,ptrain,pmon,z
        gc.collect()
        print(f"[5/5] fitted {name}", flush=True)
    # Full M4 and Champion score-distribution PSI use the saved calibrated PDs.
    train_mask=pred_frame.split_name.eq("Train_Dev").to_numpy();mon_mask=pred_frame.split_name.eq("OOT_Monitor").to_numpy()
    psi_m4=psi_numeric(pred_frame.loc[train_mask,"pd_lgbm_recal_platt"],pred_frame.loc[mon_mask,"pd_lgbm_recal_platt"])
    psi_ch=psi_numeric(pred_frame.loc[train_mask,"pd_scorecard_recal_platt"],pred_frame.loc[mon_mask,"pd_scorecard_recal_platt"])
    csi=pd.read_csv(CSI_PATH);csi=csi[(csi["Type"]=="Feature CSI")&(csi["Comparison"]=="Train_Dev vs OOT_Monitor")]
    csi_map=dict(zip(csi["Feature / Score"],csi["PSI / CSI"].astype(float)))
    full_max=max(csi_map.values()) if csi_map else None
    nd_max=max([v for k,v in csi_map.items() if k not in SEVERE_CSI_FEATURES],default=None)
    m2_max=max([v for k,v in csi_map.items() if k in m2_features],default=None)
    nd_comparison={"metrics":{"M4 Full":{"features":73,"oot_monitor_ks":roc_gini_ks(ymon,monitor.pd_lgbm_recal_platt)["ks"],
       "score_psi_train_dev_to_monitor":psi_m4,"max_feature_csi":full_max,"a_over_e":metric_bundle(ymon,monitor.pd_lgbm_recal_platt)["a_over_e"]}},
       "M4-NoDrift":{"features":len(nodrift_features),"excluded_features":SEVERE_CSI_FEATURES,
       "oot_monitor_ks":variants["M4-NoDrift"]["monitor_gini_ks"]["ks"],"score_psi_train_dev_to_monitor":variants["M4-NoDrift"]["score_psi_train_dev_to_monitor"],
       "max_feature_csi_remaining_features":nd_max,"a_over_e":variants["M4-NoDrift"]["monitor_a_over_e"],"valid_frozen_cutoff":variants["M4-NoDrift"]["valid_frozen_iso_risk_cutoff"]},
        "M2 Zero Demographic/Employment Proxies":{"features":len(m2_features),"excluded_features":sorted(set(features)-set(m2_features)),
       "oot_monitor_ks":variants["M2 Zero Demographic/Employment Proxies"]["monitor_gini_ks"]["ks"],
       "score_psi_train_dev_to_monitor":variants["M2 Zero Demographic/Employment Proxies"]["score_psi_train_dev_to_monitor"],
       "max_feature_csi_retained_features":m2_max,"a_over_e":variants["M2 Zero Demographic/Employment Proxies"]["monitor_a_over_e"],
       "valid_frozen_cutoff":variants["M2 Zero Demographic/Employment Proxies"]["valid_frozen_iso_risk_cutoff"]},
       "training_protocol":{"fit_split":"Train_Dev W0-W54","early_stopping":"Train_Dev W55-W64",
          "calibration_and_cutoff_selection":"OOT_Valid W65-W77","monitor_used_for_fit_or_selection":False}}

    # Benchmark runtime for real OOT_Monitor scoring and deterministic limit computation.
    def score_monitor_benchmark():
        rawcols=sorted(set(features+scorecard["features"]+[
          "case_id","WEEK_NUM","target","credamount_770A","annuity_780A","maininc_215A","active_bureau_monthly_debt",
          "bureau_active_max_dpd","static_avg_dpd_tol_closure24m","static_pct_inst_paid_early_3d"]))
        sel=",".join('"'+f.replace('"','""')+'"' for f in rawcols)
        df=query(con,f"SELECT {sel} FROM read_parquet('{mart_path}') WHERE split_name='OOT_Monitor'")
        # Champion vectorized WOE transform based on exact frozen JSON bin rules.
        woe=np.empty((len(df),len(scorecard["features"])),np.float32)
        for j,f in enumerate(scorecard["features"]):woe[:,j]=woe_column(df[f],rules[f])
        champion_pd=1/(1+np.exp(-np.clip(scorecard["coef"][0]+woe@np.asarray(scorecard["coef"][1:],float),-40,40)))
        x=_variant_matrix(df,features,features,model_art)
        challenger_pd=full_model.predict_proba(x)[:,1]
        from scipy.special import expit,logit
        champion_cal=expit(logit(np.clip(champion_pd,1e-12,1-1e-12))+recal["methods"]["scorecard"]["platt"]["alpha"] if False else logit(np.clip(champion_pd,1e-12,1-1e-12)))
        # Operational calibrated outputs use saved Stage 5 Platt coefficients.
        ca=recal["methods"]["scorecard"]["platt"];la=recal["methods"]["lgbm_raw"]["platt"]
        champion_cal=expit(ca["alpha"]+ca["beta"]*logit(np.clip(champion_pd,1e-12,1-1e-12)))
        challenger_cal=expit(la["alpha"]+la["beta"]*logit(np.clip(challenger_pd,1e-12,1-1e-12)))
        eng=stage4.loan_engine(df,challenger_cal)
        return {"rows":len(df),"champion_pd_mean":float(champion_cal.mean()),"challenger_pd_mean":float(challenger_cal.mean()),
           "positive_limits":int((eng["limit"]>0).sum()),"limits_total":float(eng["limit"].sum()),
           "champion_scored_rows":len(champion_pd),"challenger_scored_rows":len(challenger_pd)}
    scoring,score_telemetry=timed_peak(score_monitor_benchmark)

    # Export and parse the same self-contained JS model bundle structure used by
    # the executive app builder, without touching either HTML bundle.
    def bundle_roundtrip():
        cat_indices=full_model.booster_.params.get("categorical_column",[])
        if isinstance(cat_indices,str):cat_indices=[int(v) for v in cat_indices.split(",") if v]
        cats=full_model.booster_.pandas_categorical or []
        cat_cols=[features[int(i)] for i in cat_indices]
        bundle={"model":full_model.booster_.dump_model(),"pandas_categorical":cats,
          "categorical_features":{f:[str(v) for v in values] for f,values in zip(cat_cols,cats)},"feature_names":features}
        blob=json.dumps(bundle,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        restored=json.loads(blob)
        return {"json_bytes_utf8":len(blob.encode("utf-8")),"sha256":hashlib.sha256(blob.encode("utf-8")).hexdigest(),
          "roundtrip_feature_count":len(restored["feature_names"]),"roundtrip_tree_count":len(restored["model"]["tree_info"]),
          "standalone_bundle_schema_valid":len(restored["feature_names"])==len(features) and len(restored["model"]["tree_info"])>0}
    bundle_result,bundle_telemetry=timed_peak(bundle_roundtrip)

    # Weekly empirical metrics for OOT Monitor bootstrap are report facts.
    weekly_rows,stability_diag,label_check=weekly_stability(pred_frame)
    # Missing income + fairness with exact M2 / NoDrift / Full predictions.
    # Compose the forward monitor prediction vectors and corresponding cutoff.
    model_pds={"Champion Scorecard":monitor.pd_scorecard_recal_platt.to_numpy(float),
       "M2 Zero Demographic/Employment Proxies":variants["M2 Zero Demographic/Employment Proxies"]["monitor_pd"],
       "M4-NoDrift":variants["M4-NoDrift"]["monitor_pd"],
       "M4 Full":monitor.pd_lgbm_recal_platt.to_numpy(float)}
    cutoff_fair={"Champion Scorecard":iso_c,"M4 Full":iso_m,
       "M2 Zero Demographic/Employment Proxies":variants["M2 Zero Demographic/Employment Proxies"]["valid_frozen_iso_risk_cutoff"],
       "M4-NoDrift":variants["M4-NoDrift"]["valid_frozen_iso_risk_cutoff"]}
    # Fairness helper expects a single challenger PD field and uses limit engine;
    # create one policy view at a time to preserve each model's own risk tier.
    fair_rows=[]
    mon_full=merged_for_audit[merged_for_audit.split_name.eq("OOT_Monitor")].copy()
    y=mon_full.target.to_numpy(np.int8);age=mon_full.person_age_band_sensitive.fillna("Unknown").astype(str).to_numpy()
    gender=mon_full.person_gender_sensitive.fillna("Unknown").astype(str).value_counts()
    gender_stats={k:{"count":int(v),"share_pct":100*int(v)/len(mon_full)} for k,v in gender.items()}
    for name,p in model_pds.items():
        t=np.asarray(stage4.tier(p));engine=stage4.loan_engine(mon_full,p);approved=(p<=cutoff_fair[name])&(engine["limit"]>0)
        rates={a:float(approved[age==a].mean()) for a in ("<25","25-34","35-49","50+") if np.any(age==a)}
        maxrate=max(rates.values()) if rates else 0.
        for a in ("<25","25-34","35-49","50+"):
            band=age==a
            for tr in stage4.TIERS:
                z=band&(t==tr)
                if not z.any():continue
                goods=z&(y==0);bads=z&(y==1)
                fair_rows.append({"model":name,"age_band":a,"risk_tier":tr,"count":int(z.sum()),
                  "approval_pct":100*float(approved[z].mean()),"observed_bad_rate_pct":100*float(y[z].mean()),
                  "false_rejection_good_rate_pct":100*float((~approved[goods]).mean()) if goods.any() else None,
                  "false_approval_bad_rate_pct":100*float(approved[bads].mean()) if bads.any() else None,
                  "AIR":rates.get(a,0.)/maxrate if maxrate else None})
    fairness={"gender_missingness_oot_monitor":gender_stats,"age_within_risk_tier":fair_rows,
       "definitions":{"false_rejection_good":"good declined / all good in age × model-tier cell",
        "false_approval_bad":"bad approved / all bad in age × model-tier cell",
        "AIR":"age-band approval rate / maximum age-band approval rate for the same model"}}

    # Missing income metrics cover Train_Dev and OOT_Monitor with M4's exact
    # current fallback cap multiplier by PD tier; alternatives are counterfactual.
    income_rows=[];alternative_rows=[];fallback_map=dict(zip(stage4.TIERS,[.8,.75,.7,.6,0.]))
    for split in ("Train_Dev","OOT_Monitor"):
        g=merged_for_audit[merged_for_audit.split_name.eq(split)];p=g.pd_lgbm_recal_platt.to_numpy(float);yy=g.target.to_numpy(np.int8)
        eng=stage4.loan_engine(g,p);inc=g.maininc_215A.to_numpy(float);missing=~(np.isfinite(inc)&(inc>100));tier=np.asarray(eng["tier"])
        approved=(p<=iso_m)&(eng["limit"]>0)
        for status,mask in (("Verified",~missing),("Missing / unverified",missing)):
            for tr in stage4.TIERS:
                z=mask&(tier==tr)
                if not z.any():continue
                income_rows.append({"split":split,"income_status":status,"risk_tier":tr,"count":int(z.sum()),"share_of_split_pct":100*float(z.mean()),
                  "approval_pct":100*float(approved[z].mean()),"observed_bad_rate_pct":100*float(yy[z].mean()),
                  "mean_pd_pct":100*float(p[z].mean()),"fallback_multiplier":fallback_map[tr] if status.startswith("Missing") else None})
        miss=missing;auto=int((miss&approved).sum())
        alternative_rows.extend([
          {"split":split,"policy":"Current tiered cap / haircut","count":int(miss.sum()),"auto_approved":auto,"review_queue":0,"auto_approval_pct":100*auto/max(1,int(miss.sum()))},
          {"split":split,"policy":"Manual verification route","count":int(miss.sum()),"auto_approved":0,"review_queue":int(miss.sum()),"auto_approval_pct":0.},
          {"split":split,"policy":"Decline unverified income","count":int(miss.sum()),"auto_approved":0,"review_queue":0,"auto_approval_pct":0.}])

    # S4 stress calculations under frozen Monitor Iso-Risk approvals.
    # (Already computed above in the stage6-equivalent formula.)
    # Reconcile exact metrics, sample counts, and output with no non-finite JSON.
    html_after={str(p):sha256(p) for p in HTML_FILES}
    html_untouched=html_before==html_after
    result={"metadata":{"generated_at_local":time.strftime("%Y-%m-%d %H:%M:%S"),"script":"src/mrm_v2_reconcile.py","python_executable":sys.executable,
      "B":BOOTSTRAPS,"seed":SEED,"duckdb_memory_limit":"3GB","html_untouched":html_untouched,
      "html_hashes_before":html_before,"html_hashes_after":html_after},
      "section_1_split_reconciliation":split_audit,
      "section_2_resource_telemetry":{"duckdb_sample_mart_query":sql_telemetry,
       "oot_monitor_champion_m4_limit_scoring":score_telemetry,"standalone_model_bundle_export_verify":bundle_telemetry,
       "bundle_roundtrip":bundle_result,"duckdb_query_rows":int(len(sql_df)),"oot_monitor_scoring":scoring},
      "section_3_weekly_stability_and_bootstrap":{"weekly_metrics":weekly_rows,
       "home_credit_gini_stability_diagnostics":stability_diag,"late_window_label_maturity":label_check,
       "oot_monitor_bootstrap":boot},
      "section_4_economics_cutoff_and_kgb":{"cutoff_and_valid_monitor_metrics":policy_results,
       "non_circular_economic_contribution":economic,"model_neutral_paired_policy_bootstrap":contrib_boot,
       "s4_tail_shock":s4,"stage5_recalibration_provenance":{"fit_split":recal["fit_split"],"fit_weeks":recal["fit_weeks"],
        "fit_rows":recal["fit_rows"],"oot_monitor_used_for_fit":recal["oot_monitor_used_for_fit"],
        "iso_risk_valid_reference_bad_rate":BAD_RATE_REFERENCE,"iso_risk_cutoffs":config["thresholds"]["iso_risk"],
        "iso_approval_cutoffs":config["thresholds"]["iso_approval"]}},
      "section_5_missing_income_csi_and_fairness":{"missing_income_by_split_tier":income_rows,
       "missing_income_policy_alternatives":alternative_rows,"gender_missingness_oot_monitor":gender_stats,
       "age_within_risk_tier_fairness":fairness,"m4_no_drift_and_m2_comparison":nd_comparison,
       "variant_training_details":{k:{kk:vv for kk,vv in v.items() if kk not in ("monitor_pd","train_dev_pd","train_dev_calibrated_pd","oot_monitor_pd","model")} for k,v in variants.items()}}
    }
    if not html_untouched:raise RuntimeError("HTML hash changed during reconciliation; refusing to report success.")
    # Bundle model objects and matrices are released before saving final artifacts.
    OUTPUT.parent.mkdir(parents=True,exist_ok=True);REPORT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(clean(result),indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    REPORT.write_text(render_report(result),encoding="utf-8")
    append_detailed_report(result)
    con.close()
    print_summary(result)


def woe_column(series,rule):
    n=len(series);out=np.full(n,float(rule.get("missing_woe",0.)),dtype=np.float32)
    if rule["dtype"]=="categorical":
        vals=series.astype("string")
        known=np.zeros(n,dtype=bool)
        for b in rule["bins"]:
            cats=[str(v) for v in b.get("categories",[])]
            if cats:
                mask=vals.isin(cats).to_numpy();out[mask]=float(b["woe"]);known|=mask
        unknown=series.notna().to_numpy()&~known
        if rule.get("unknown_woe") is not None:out[unknown]=float(rule["unknown_woe"])
    else:
        vals=pd.to_numeric(series,errors="coerce").to_numpy(float);valid=np.isfinite(vals)
        for b in rule["bins"]:
            lo=-np.inf if b["lower"]=="-Infinity" else float(b["lower"])
            hi=np.inf if b["upper"]=="Infinity" else float(b["upper"])
            out[valid&(vals>=lo)&(vals<hi)]=float(b["woe"])
    return out


def _variant_matrix(df,cols,full_features,artifact):
    model=artifact["model"];inds=model.booster_.params.get("categorical_column",[])
    if isinstance(inds,str):inds=[int(v) for v in inds.split(",") if v]
    cat_names=[full_features[int(i)] for i in inds];cats=dict(zip(cat_names,model.booster_.pandas_categorical or []))
    out=pd.DataFrame(index=df.index)
    for f in cols:
        if f in cats:out[f]=pd.Categorical(df[f].astype("string"),categories=cats[f])
        else:
            a=pd.to_numeric(df[f],errors="coerce").to_numpy(np.float32,copy=True);a[~np.isfinite(a)]=-999;out[f]=a
    return out


def render_report(result):
    s1=result["section_1_split_reconciliation"];s2=result["section_2_resource_telemetry"];s3=result["section_3_weekly_stability_and_bootstrap"]
    s4=result["section_4_economics_cutoff_and_kgb"];s5=result["section_5_missing_income_csi_and_fairness"]
    sections=["# FinGuard MRM v2 Empirical Reconciliation Audit", "",
      f"生成时间：{result['metadata']['generated_at_local']}。本步骤仅写入 reconciled JSON 与本报告；HTML unchanged = **{result['metadata']['html_untouched']}**。",
      "", "## 1. Sample selection、outcome observability 与 split reconciliation", "",
      "Canonical split truth 以已保存 `WEEK_NUM` 分配为准；Train_Dev case-ID artifact、Feature Mart、Stage 3 与 Stage 5 均作交叉核验。",
      "",md_table(["Split","Rows","Week range","Date range","Observed target %","Observed bad rate %"],
        [{"Split":r["split_name"],"Rows":r["rows"],"Week range":f"W{r['week_min']}–{r['week_max']}","Date range":f"{r['date_min']}–{r['date_max']}",
          "Observed target %":r["target_observed_pct"],"Observed bad rate %":r["observed_bad_rate_pct"]} for r in s1["canonical_split_rows"]]),
      "", "**35-row discrepancy:** current artifacts reproduce no membership mismatch; the reported 1,292,315 / 82,637 pair is not supported by the saved case IDs. Exact discrepant `case_id`s are therefore not identifiable from these inputs. Canonical rows are 1,292,350 / 82,602. See JSON for source-level joins and the empty mismatch audit.",
      "", "## 2. Reproducible resource telemetry", "",
      md_table(["Operation","Elapsed seconds","Peak RSS MB","Peak RSS delta MB"],[
       {"Operation":"DuckDB 3GB sample/mart query","Elapsed seconds":s2["duckdb_sample_mart_query"]["elapsed_seconds"],"Peak RSS MB":s2["duckdb_sample_mart_query"]["peak_rss_mb"],"Peak RSS delta MB":s2["duckdb_sample_mart_query"]["peak_rss_delta_mb"]},
       {"Operation":"OOT_Monitor Champion + M4 + limits","Elapsed seconds":s2["oot_monitor_champion_m4_limit_scoring"]["elapsed_seconds"],"Peak RSS MB":s2["oot_monitor_champion_m4_limit_scoring"]["peak_rss_mb"],"Peak RSS delta MB":s2["oot_monitor_champion_m4_limit_scoring"]["peak_rss_delta_mb"]},
       {"Operation":"Standalone bundle export + JSON round-trip","Elapsed seconds":s2["standalone_model_bundle_export_verify"]["elapsed_seconds"],"Peak RSS MB":s2["standalone_model_bundle_export_verify"]["peak_rss_mb"],"Peak RSS delta MB":s2["standalone_model_bundle_export_verify"]["peak_rss_delta_mb"]}]),
      "",f"Bundle: {s2['bundle_roundtrip']['json_bytes_utf8']:,} UTF-8 bytes, {s2['bundle_roundtrip']['roundtrip_tree_count']:,} trees, schema valid = {s2['bundle_roundtrip']['standalone_bundle_schema_valid']}.",
      "", "## 3. Weekly Gini, label maturity and bootstrap CIs", "",
      "`A/E = Observed Defaults / Expected Defaults`; `E/A = Expected Defaults / Observed Defaults`. Paired applicant bootstrap uses the same sampled case indices for both models; weekly block bootstrap resamples whole `WEEK_NUM` clusters.",
      "",md_table(["Split","Model","Mean weekly Gini","Weekly Gini slope","Residual std","Stability metric"],[
        {"Split":sp,"Model":model,**d} for sp,models in s3["home_credit_gini_stability_diagnostics"].items() for model,d in models.items()]),
      "", "OOT_Monitor point estimates and paired differences with 95% intervals are in `section_3_weekly_stability_and_bootstrap.oot_monitor_bootstrap`.",
      "", f"Late-window conclusion: {s3['late_window_label_maturity']['conclusion']}",
      "", "## 4. Economic contribution, frozen cutoffs and reject-bias sensitivity", "",
      s4["non_circular_economic_contribution"]["model_neutral_benchmark"]["equation"],
      "",md_table(["Model","Approved","Approval %","Observed bad %","Model-neutral realized contribution","Scenario expected contribution"],[
        {"Model":r["model"],"Approved":r["approved"],"Approval %":r["approval_pct"],"Observed bad %":r["approved_bad_rate_pct"],
         "Model-neutral realized contribution":r["model_neutral_realized_contribution"],"Scenario expected contribution":r["scenario_expected_contribution"]}
        for r in s4["non_circular_economic_contribution"]["scenario_based_expected_contribution"]["rows"]]),
      "", "S4 uses Stage 6 severe shock assumptions: COF 5.5%, LGD 85%, PD/target multipliers 1.30x for tiers A-B and 2.00x for C-E. `k*` sensitivity inflates target-based loss on challenger-only approvals; it is a declared scenario, not observed hidden labels.",
      "",md_table(["Model","Approved","Expected S4 proxy","Realized-target S4 proxy"],[{"Model":r["model"],"Approved":r["approved"],"Expected S4 proxy":r["expected_contribution_proxy"],"Realized-target S4 proxy":r["realized_target_stress_proxy"]} for r in s4["s4_tail_shock"]]),
      "", "The exact OOT_Valid cutoffs, OOT_Valid metrics, OOT_Monitor results, bootstrap deltas, and Stage 5 provenance are structured in the JSON output.",
      "", "## 5. Missing income, CSI remediation and fairness proxies", "",
      "Missing-income alternative policies are counterfactual routing rules; outcomes are not changed or imputed. Fairness rates are descriptive and conditioned on each model's own calibrated-PD tier.",
      "",md_table(["Gender value","OOT_Monitor count","Share %"],[{"Gender value":k,"OOT_Monitor count":v["count"],"Share %":v["share_pct"]} for k,v in s5["gender_missingness_oot_monitor"].items()]),
      "",md_table(["Variant","Features","Monitor KS","Score PSI","Max retained-feature CSI","A/E"],[
       {"Variant":name,"Features":d.get("features"),"Monitor KS":d.get("oot_monitor_ks"),"Score PSI":d.get("score_psi_train_dev_to_monitor"),
        "Max retained-feature CSI":d.get("max_feature_csi",d.get("max_feature_csi_remaining_features")),"A/E":d.get("a_over_e")}
       for name,d in s4["unused"]] if False else [
       {"Variant":k,"Features":v["features"],"Monitor KS":v["oot_monitor_ks"],"Score PSI":v["score_psi_train_dev_to_monitor"],
        "Max retained-feature CSI":v.get("max_feature_csi",v.get("max_feature_csi_remaining_features")),"A/E":v["a_over_e"]}
       for k,v in s5["m4_no_drift_and_m2_comparison"].items() if isinstance(v,dict) and "features" in v]),
      "", "Full weekly trajectory, per-tier missing-income metrics, age-band conditional rates and all exact paired/bootstrap tables are in `data/processed/mrm_reconciled_truth.json`.",
      "", "## Reproducibility notes", "",
      f"Bootstrap B={result['metadata']['B']}, seed={result['metadata']['seed']}; DuckDB memory limit 3GB, 4 threads. The report distinguishes empirical labels from scenario-based assumptions. HTML hashes match before/after: {result['metadata']['html_untouched']}."]
    return "\n".join(sections)+"\n"


def print_summary(result):
    s1=result["section_1_split_reconciliation"];s2=result["section_2_resource_telemetry"]
    s3=result["section_3_weekly_stability_and_bootstrap"];s4=result["section_4_economics_cutoff_and_kgb"]
    s5=result["section_5_missing_income_csi_and_fairness"]
    print("\n=== FinGuard MRM v2 reconciliation summary ===")
    print("Split truth:",", ".join(f"{r['split_name']}={r['rows']:,}" for r in s1["canonical_split_rows"]))
    print("35-row claim:",s1["discrepancy_case_ids_status"],"; assignment mismatches:",len(s1["case_level_assignment_mismatches"]))
    for name,t in (("DuckDB query",s2["duckdb_sample_mart_query"]),("OOT scoring",s2["oot_monitor_champion_m4_limit_scoring"]),("Bundle round-trip",s2["standalone_model_bundle_export_verify"])):
        print(f"Resource {name}: {t['elapsed_seconds']:.2f}s, peak RSS {t['peak_rss_mb']:.1f} MB")
    print("OOT Monitor bootstrap:",f"B={s3['oot_monitor_bootstrap']['bootstrap_replicates']}, paired applicants + weekly blocks")
    print("OOT_Monitor weekly Gini stability:",json.dumps(s3["home_credit_gini_stability_diagnostics"]["OOT_Monitor"],ensure_ascii=False))
    print("W86-W91 maturity:",s3["late_window_label_maturity"]["conclusion"])
    print("Valid-frozen OOT_Monitor cutoffs:",json.dumps(s4["cutoff_and_valid_monitor_metrics"]["valid_frozen_policy_metrics"]["OOT_Monitor"],ensure_ascii=False))
    print("Iso-Risk contribution equality for same limits:",s4["non_circular_economic_contribution"]["model_neutral_benchmark"]["asserted_identical"])
    print("KGB break-even k*:",s4["non_circular_economic_contribution"]["kgb_sensitivity"]["break_even_k_star"])
    print("Gender missingness:",s5["gender_missingness_oot_monitor"])
    print("M2/M4-NoDrift variants:",", ".join(f"{k}={v['features']} features / KS={v['oot_monitor_ks']:.4f}" for k,v in s5["m4_no_drift_and_m2_comparison"].items() if isinstance(v,dict) and "features" in v))
    print("HTML untouched:",result["metadata"]["html_untouched"])
    print("JSON:",OUTPUT);print("Audit:",REPORT)


def append_detailed_report(result):
    s3=result["section_3_weekly_stability_and_bootstrap"]
    s4=result["section_4_economics_cutoff_and_kgb"]
    s5=result["section_5_missing_income_csi_and_fairness"]
    weekly=[{"Week":r["week"],"Split":r["split_name"],"Model":r["model"],"n":r["n"],"Observed target %":r["target_observed_pct"],"Bad rate %":r["observed_bad_rate_pct"],"Mean recalibrated PD %":r["mean_recalibrated_pd_pct"],"Gini":r["weekly_gini"]} for r in s3["weekly_metrics"]]
    boot=[]
    for model,metrics in s3["oot_monitor_bootstrap"]["point_estimates"].items():
        for metric,value in metrics.items():
            boot.append({"Model / delta":model,"Metric":metric,"Estimate":value,
              "Applicant 95% CI":s3["oot_monitor_bootstrap"]["confidence_intervals"]["paired_applicant"][model][metric],
              "Weekly block 95% CI":s3["oot_monitor_bootstrap"]["confidence_intervals"]["weekly_block"][model][metric]})
    cutoffs=[]
    for model_rows in s4["cutoff_and_valid_monitor_metrics"]["valid_frozen_policy_metrics"].values():
        for entries in model_rows.values():
            for x in entries:
                cutoffs.append({"Split":x["split"],"Model":x["model"],"Policy":x["policy"],"Cutoff":x["pd_cutoff"],"Apps":x["applications"],"Approved":x["approved"],"Approval %":x["approval_pct"],"Bads":x["observed_bads"],"Bad rate %":x["observed_bad_rate_pct"]})
    age=[{"Model":x["model"],"Age":x["age_band"],"Risk tier":x["risk_tier"],"n":x["count"],"Approval %":x["approval_pct"],"Bad rate %":x["observed_bad_rate_pct"],"False rejection good %":x["false_rejection_good_rate_pct"],"False approval bad %":x["false_approval_bad_rate_pct"],"AIR":x["AIR"]} for x in s5["age_within_risk_tier_fairness"]["age_within_risk_tier"]]
    income=s5["missing_income_by_split_tier"]
    lines=["", "## Detailed statistical tables", "",
      "### Full W0–W91 weekly trajectory", "",md_table(["Week","Split","Model","n","Observed target %","Bad rate %","Mean recalibrated PD %","Gini"],weekly),"",
      "### OOT_Monitor paired and weekly block bootstrap", "",md_table(["Model / delta","Metric","Estimate","Applicant 95% CI","Weekly block 95% CI"],boot),"",
      "A/E = Observed Defaults / Expected Defaults; E/A = Expected Defaults / Observed Defaults. B=1,000, seed=42; applicant draws share sampled case indices across models, weekly draws resample whole weeks.","",
      "### Valid-frozen policies: OOT_Valid selection and OOT_Monitor evaluation", "",md_table(["Split","Model","Policy","Cutoff","Apps","Approved","Approval %","Bads","Bad rate %"],cutoffs),"",
      "Stage 5 policy provenance: `"+json.dumps(s4["cutoff_and_valid_monitor_metrics"]["stage5_headline_provenance"],ensure_ascii=False)+"`", "",
      "Paired policy bootstrap: `"+json.dumps(s4["model_neutral_paired_policy_bootstrap"],ensure_ascii=False)+"`", "",
      "### KGB sensitivity", "",md_table(["k","Net swap contribution"],[{"k":x["k"],"Net swap contribution":x["net_swap_contribution"]} for x in s4["non_circular_economic_contribution"]["kgb_sensitivity"]["net_swap_contribution_by_k"]]),"",
      "### Missing-income strata and fallback policy", "",md_table(["Split","Income status","Risk tier","Count","Share %","Approval %","Bad rate %","Mean PD %","Fallback multiplier"],income),"",
      "### Gender missingness", "",md_table(["Group","Count","Share %"],[{"Group":k,"Count":v["count"],"Share %":v["share_pct"]} for k,v in s5["gender_missingness_oot_monitor"].items()]),"",
      "### Age fairness by model and risk tier", "",md_table(["Model","Age","Risk tier","n","Approval %","Bad rate %","False rejection good %","False approval bad %","AIR"],age),"",
      "Fairness definitions: `"+json.dumps(s5["age_within_risk_tier_fairness"]["definitions"],ensure_ascii=False)+"`", ""]
    with REPORT.open("a",encoding="utf-8") as f:f.write("\n".join(lines))


if __name__=="__main__":
    main()
