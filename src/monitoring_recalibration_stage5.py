import gc
import json
import os
import pickle
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".matplotlib"))
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MART = DATA / "finguard_feature_mart.parquet"
STAGE3 = DATA / "stage3_predictions.parquet"
STAGE4 = DATA / "stage4_oot_valid_decisions.parquet"
FULL_OUT = DATA / "stage5_full_predictions.parquet"
PARAM_OUT = DATA / "models" / "recalibrators_stage5.json"
REPORT_OUT = REPORTS / "stage5_monitoring_recalibration_report.md"

PSI_MODERATE = 0.10
PSI_MAJOR = 0.25
DSR_CAP = 0.50
COF, OPEX, LGD = 0.04, 0.015, 0.75
APR = {"Tier A: Super Prime": .09, "Tier B: Prime": .12, "Tier C: Standard": .15,
       "Tier D: High Risk": .18, "Tier E: Decline": 0.0}
MULTIPLE = {"Tier A: Super Prime": 4., "Tier B: Prime": 3., "Tier C: Standard": 2.,
            "Tier D: High Risk": 1., "Tier E: Decline": 0.}
FALLBACK = {"Tier A: Super Prime": .80, "Tier B: Prime": .75, "Tier C: Standard": .70,
            "Tier D: High Risk": .60, "Tier E: Decline": 0.}
TIERS = list(APR)
SENSITIVE = ["person_gender_sensitive", "person_age_band_sensitive", "person_age"]
RAW_LIMIT_FEATURES = ["credamount_770A", "annuity_780A", "maininc_215A", "active_bureau_monthly_debt", "pti",
                      "total_dsr_proxy", "bureau_active_max_dpd", "static_avg_dpd_tol_closure24m", "static_pct_inst_paid_early_3d"]


def say(text):
    print(text, flush=True)


def tier(pd_values):
    p = np.asarray(pd_values, float)
    return np.select([p < .01, p < .02, p < .04, p < .07], TIERS[:4], default=TIERS[4])


def safe_logit(p):
    return logit(np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8))


def metrics(y, p):
    y = np.asarray(y, np.int8); p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    auc = float(roc_auc_score(y, p)); fpr, tpr, _ = roc_curve(y, p)
    mean_p = float(p.mean()); actual = float(y.mean())
    return {"ROC-AUC": auc, "KS": float(np.max(tpr - fpr)), "Gini": 2 * auc - 1,
            "Brier Score": float(brier_score_loss(y, p)), "Mean Predicted PD %": 100 * mean_p,
            "Actual Bad Rate %": 100 * actual, "A/E Ratio": actual / mean_p if mean_p else np.nan}


def quantile_edges(values, bins=10):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x): return np.array([-np.inf, np.inf])
    e = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(e) < 2: return np.array([-np.inf, np.inf])
    e[0], e[-1] = -np.inf, np.inf
    return e


def numeric_bins(values, edges):
    x = np.asarray(values, float)
    ids = np.searchsorted(edges[1:-1], x, side="right").astype(int)
    ids[~np.isfinite(x)] = len(edges) - 1
    return ids


def psi_from_bins(base_ids, current_ids, n_bins):
    a = np.bincount(np.asarray(base_ids, int), minlength=n_bins).astype(float)
    b = np.bincount(np.asarray(current_ids, int), minlength=n_bins).astype(float)
    pa = np.maximum(a / max(1, a.sum()), 1e-4); pb = np.maximum(b / max(1, b.sum()), 1e-4)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def make_feature_rule(train_series, max_bins=10):
    s = train_series
    if pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy(np.float32, copy=False)
        unique = np.unique(v[np.isfinite(v)])
        if len(unique) <= max_bins:
            cats = [float(x) for x in unique]
            return {"kind": "numeric_categorical", "categories": cats, "other": False}
        return {"kind": "numeric", "edges": quantile_edges(v, max_bins)}
    x = s.astype("string")
    top = x.dropna().value_counts().head(max_bins - 1).index.astype(str).tolist()
    return {"kind": "categorical", "categories": top, "other": True}


def apply_feature_rule(series, rule):
    if rule["kind"] == "numeric":
        values = series.to_numpy(np.float32, copy=False)
        return numeric_bins(values, rule["edges"]), len(rule["edges"])  # last slot is Missing
    if rule["kind"] == "numeric_categorical":
        vals = series.to_numpy(np.float32, copy=False)
        cats = rule["categories"]
        mp = {v: i for i, v in enumerate(cats)}
        ids = np.array([mp.get(float(v), len(cats)) if np.isfinite(v) else len(cats) + 1 for v in vals], int)
        return ids, len(cats) + 2  # Other and Missing
    vals = series.astype("string").to_numpy(dtype=object)
    cats = rule["categories"]; mp = {v: i for i, v in enumerate(cats)}
    other_id = len(cats); missing_id = len(cats) + int(rule["other"])
    ids = np.array([missing_id if pd.isna(v) else mp.get(str(v), other_id) for v in vals], int)
    return ids, missing_id + 1


def feature_csi(train_series, valid_series, monitor_series):
    rule = make_feature_rule(train_series)
    a, nb = apply_feature_rule(train_series, rule)
    b, _ = apply_feature_rule(valid_series, rule)
    c, _ = apply_feature_rule(monitor_series, rule)
    return psi_from_bins(a, b, nb), psi_from_bins(a, c, nb), rule


def tier_ae(y, p):
    ts = tier(p); rows = []
    for name in TIERS:
        m = ts == name
        if not m.any(): continue
        act = float(np.asarray(y)[m].mean()); pred = float(np.asarray(p)[m].mean())
        rows.append({"Risk Tier": name, "Count": int(m.sum()), "Actual Bad Rate %": 100 * act,
                     "Mean Predicted PD %": 100 * pred, "A/E Ratio": act / pred if pred else np.nan})
    return rows


def report_table(df):
    return df.to_markdown(index=False, floatfmt=".4f")


def json_woe(rules, frame, features):
    x = np.empty((len(frame), len(features)), np.float32)
    for j, name in enumerate(features):
        rule = rules[name]; s = frame[name]
        if rule["dtype"] == "numerical":
            values = s.to_numpy(np.float32, copy=False)
            out = np.full(len(s), float(rule["missing_woe"]), np.float32)
            special = rule.get("special_codes")
            for i, value in enumerate(values):
                if not np.isfinite(value): continue
                is_special = special is not None and value in special
                if is_special:
                    out[i] = float(rule.get("special_woe") or 0.0); continue
                for b in rule["bins"]:
                    lo = -np.inf if b["lower"] == "-Infinity" else float(b["lower"])
                    hi = np.inf if b["upper"] == "Infinity" else float(b["upper"])
                    if lo <= value < hi:
                        out[i] = float(b["woe"]); break
            x[:, j] = out
        else:
            groups = [(set(b["categories"]), float(b["woe"])) for b in rule["bins"]]
            out = np.full(len(s), float(rule["missing_woe"]), np.float32)
            for i, value in enumerate(s.astype("string").to_numpy(dtype=object)):
                if pd.isna(value): continue
                val = str(value)
                out[i] = next((w for cats, w in groups if val in cats), float(rule.get("unknown_woe") or 0.0))
            x[:, j] = out
    return x


def calibrators(y, p):
    z = safe_logit(p); target = float(np.mean(y))
    alpha = brentq(lambda a: float(expit(a + z).mean() - target), -40.0, 40.0)
    design = sm.add_constant(z, has_constant="add")
    fitted = sm.Logit(np.asarray(y, np.int8), design).fit(disp=False, maxiter=100)
    alpha_p, beta = map(float, fitted.params)
    if beta <= 0:
        raise ValueError(f"Platt recalibration slope must stay positive to preserve risk ranking; got beta={beta}")
    return {"intercept": {"alpha": float(alpha), "beta": 1.0},
            "platt": {"alpha": alpha_p, "beta": beta}, "platt_summary": fitted.summary2().as_text()}


def rec_p(p, spec):
    return expit(spec["alpha"] + spec["beta"] * safe_logit(p))


def loan_engine(df, pd_values):
    p = np.asarray(pd_values, float); ts = tier(p)
    req = np.nan_to_num(df.credamount_770A.to_numpy(float), nan=0.0).clip(0)
    ann = np.nan_to_num(df.annuity_780A.to_numpy(float), nan=0.0).clip(0)
    income = df.maininc_215A.to_numpy(float); debt = np.nan_to_num(df.active_bureau_monthly_debt.to_numpy(float), nan=0.0).clip(0)
    verified = np.isfinite(income) & (income > 100)
    income_safe = np.where(verified, income, 0.0)
    term = np.clip(np.rint(np.divide(req, ann, out=np.full_like(req, 12.0), where=ann > 0)), 3, 60).astype(int)
    apr = np.array([APR[t] for t in ts]); mult = np.array([MULTIPLE[t] for t in ts]); fallback = np.array([FALLBACK[t] for t in ts])
    dpd = df.bureau_active_max_dpd.to_numpy(float); hist = df.static_avg_dpd_tol_closure24m.to_numpy(float); early = df.static_pct_inst_paid_early_3d.to_numpy(float)
    bad_history = (np.nan_to_num(dpd, nan=0) > 0) | (np.nan_to_num(hist, nan=0) > 0)
    beh = np.where(bad_history, .75, 1.0)
    beh = np.where(~bad_history & np.isfinite(early) & (early >= .70), 1.15, beh)
    ratio = np.divide(debt, income_safe, out=np.zeros_like(debt), where=income_safe > 0)
    policy = (np.nan_to_num(dpd, nan=-1) >= 90) | ((income > 0) & np.isfinite(income) & (np.divide(debt, income, out=np.full_like(debt, np.nan), where=income != 0) > .60))
    beh = np.where(policy | (ts == TIERS[4]), 0.0, beh)
    risk = np.where(verified, income_safe * mult * beh, req * fallback * beh)
    maxpay = np.maximum(0, DSR_CAP * income_safe - debt)
    mr = apr / 12
    aff = np.where(verified, np.where(mr > 0, maxpay * (1 - (1 + mr) ** (-term)) / np.where(mr > 0, mr, 1), maxpay * term), req * fallback)
    limit = np.maximum(0, np.minimum(req, np.minimum(risk, aff)))
    limit = np.where(policy | (ts == TIERS[4]), 0.0, limit)
    den = np.where(mr > 0, 1 - (1 + mr) ** (-term), 1.0)
    payment = np.where(mr > 0, limit * mr / den, limit / term)
    dsr = np.divide(debt + payment, income_safe, out=np.full_like(debt, np.nan), where=income_safe > 0)
    return {"tier": ts, "apr": apr, "term": term, "limit": limit, "behavior": beh,
            "income_unverified": (~verified), "policy": policy, "verified": verified,
            "dsr": dsr}


def expected_profit(principal, p, apr, term):
    principal = np.maximum(np.nan_to_num(np.asarray(principal, float), nan=0), 0)
    p = np.clip(np.nan_to_num(np.asarray(p, float), nan=0), 0, 1); r = np.asarray(apr, float) / 12
    term = np.asarray(term, int); den = np.where(r > 0, 1 - (1 + r) ** (-term), 1.0)
    payment = np.where(r > 0, principal * r / den, principal / term)
    bal = principal.copy(); total_bal = np.zeros_like(bal)
    for m in range(60):
        active = (m < term) & (principal > 0); total_bal[active] += bal[active]
        bal[active] = np.maximum(0, bal[active] * (1 + r[active]) - payment[active])
    return np.maximum(np.asarray(apr) - COF - OPEX, 0) / 12 * total_bal - principal * p * LGD


def eligible_mask(df, engine):
    return (~engine["policy"]) & (engine["tier"] != TIERS[4]) & (engine["limit"] > 0)


def exact_k_approval(mask, pdv, case_ids, k):
    ids = np.flatnonzero(mask); order = np.lexsort((np.asarray(case_ids)[ids], np.asarray(pdv)[ids]))
    chosen = np.zeros(len(mask), bool); chosen[ids[order[:k]]] = True
    return chosen


def main():
    FIGS.mkdir(parents=True, exist_ok=True); (DATA / "models").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(); con.execute("SET memory_limit='6GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(ROOT / 'duckdb_temp').as_posix()}'")
    rules = json.load(open(DATA / "woe_bin_rules.json", encoding="utf-8"))
    score_art = pickle.load(open(DATA / "models" / "scorecard_logit.pkl", "rb"))
    lgb_art = pickle.load(open(DATA / "models" / "lgbm_m4.pkl", "rb"))
    score_features = score_art["features"]; factor = float(score_art["factor"]); offset0 = float(score_art["base_score"])
    lgb_model = lgb_art["model"]; lgb_features = lgb_art["features"]
    mapping = pd.read_csv(DATA / "feature_group_mapping.csv").set_index("column_name")
    category_features = [c for c in lgb_features if str(mapping.loc[c, "dtype"]).upper() in {"VARCHAR", "STRING", "OBJECT"}]
    cat_levels = getattr(lgb_model.booster_, "pandas_categorical", None) or []
    if len(category_features) != len(cat_levels):
        raise ValueError(f"LightGBM category metadata mismatch: {len(category_features)} columns vs {len(cat_levels)} category lists")
    cat_map = {c: list(v) for c, v in zip(category_features, cat_levels)}

    # Calibration fit inputs are only OOT_Valid; the monitor partition has not yet been read.
    val = con.execute(f"SELECT case_id,WEEK_NUM,target,scorecard_score,pd_scorecard,pd_lgbm_raw FROM read_parquet('{STAGE3.as_posix()}') WHERE split_name='OOT_Valid' AND WEEK_NUM BETWEEN 65 AND 77").fetchdf()
    if len(val) != 82602: raise ValueError(f"OOT_Valid calibration sample count mismatch: {len(val)}")
    yv = val.target.to_numpy(np.int8)
    specs = {"scorecard": calibrators(yv, val.pd_scorecard.to_numpy(float)),
             "lgbm_raw": calibrators(yv, val.pd_lgbm_raw.to_numpy(float))}
    score_a = specs["scorecard"]["intercept"]["alpha"]
    shift = factor * score_a
    params = {"fit_split": "OOT_Valid", "fit_weeks": [65, 77], "fit_rows": int(len(val)),
              "oot_monitor_used_for_fit": False, "scorecard_base_score_offset_original": offset0,
              "scorecard_intercept_only_alpha": score_a,
              "scorecard_intercept_only_base_score_offset_new": offset0 + shift,
              "scorecard_intercept_only_base_score_offset_shift_points": shift,
              "pdo_factor": factor, "methods": {}}
    for model_name, s in specs.items():
        params["methods"][model_name] = {k: {"alpha": float(v["alpha"]), "beta": float(v["beta"])} for k, v in s.items() if k in ("intercept", "platt")}
    with open(PARAM_OUT, "w", encoding="utf-8") as f: json.dump(params, f, ensure_ascii=False, indent=2)

    # Load and score OOT_Monitor only after both recalibration specifications are frozen.
    all_features = list(dict.fromkeys(score_features + lgb_features + RAW_LIMIT_FEATURES + SENSITIVE))
    rawq = ",".join('"' + c + '"' for c in all_features)
    monitor = con.execute(f"SELECT case_id,WEEK_NUM,MONTH,split_name,target,{rawq} FROM read_parquet('{MART.as_posix()}') WHERE WEEK_NUM BETWEEN 78 AND 91").fetchdf()
    if len(monitor) != 151707 or monitor.WEEK_NUM.min() < 78 or monitor.WEEK_NUM.max() > 91:
        raise ValueError(f"OOT_Monitor guard failed: {len(monitor)} rows, weeks {monitor.WEEK_NUM.min()}-{monitor.WEEK_NUM.max()}")
    for c in all_features:
        if pd.api.types.is_numeric_dtype(monitor[c]): monitor[c] = monitor[c].astype(np.float32)
    xwoe = json_woe(rules, monitor, score_features)
    pd_sc_mon = np.asarray(score_art["result"].predict(sm.add_constant(xwoe, has_constant="add")), float)
    score_mon = offset0 - factor * safe_logit(pd_sc_mon)
    x_lgb = pd.DataFrame(index=monitor.index)
    for c in lgb_features:
        if c in cat_map: x_lgb[c] = pd.Categorical(monitor[c].astype("string"), categories=cat_map[c])
        else: x_lgb[c] = monitor[c].to_numpy(np.float32, copy=False)
    pd_lgb_mon = lgb_model.predict_proba(x_lgb)[:, 1]
    mon_pred = pd.DataFrame({"case_id": monitor.case_id, "WEEK_NUM": monitor.WEEK_NUM, "MONTH": monitor.MONTH,
        "split_name": monitor.split_name, "target": monitor.target, "scorecard_score": score_mon,
        "pd_scorecard": pd_sc_mon, "pd_lgbm_raw": pd_lgb_mon})
    del xwoe, x_lgb; gc.collect()

    # Full split chronology and recalibrated values; OOT_Monitor scores are newly generated.
    prior = con.execute(f"SELECT p.case_id,p.WEEK_NUM,m.MONTH,p.split_name,p.target,p.scorecard_score,p.pd_scorecard,p.pd_lgbm_raw FROM read_parquet('{STAGE3.as_posix()}') p JOIN read_parquet('{MART.as_posix()}') m USING(case_id,WEEK_NUM) WHERE p.WEEK_NUM<=77").fetchdf()
    full = pd.concat([prior, mon_pred], ignore_index=True)
    for key, col in [("scorecard", "pd_scorecard"), ("lgbm_raw", "pd_lgbm_raw")]:
        for method in ["intercept", "platt"]:
            pnew = rec_p(full[col].to_numpy(float), specs[key][method])
            outcol = "pd_scorecard" if key == "scorecard" else "pd_lgbm"
            full[f"{outcol}_recal_{method}"] = pnew
    full["scorecard_score_recal_intercept"] = (offset0 + shift) - factor * safe_logit(full.pd_scorecard_recal_intercept.to_numpy(float))
    full["scorecard_score_recal_platt"] = offset0 - factor * safe_logit(full.pd_scorecard_recal_platt.to_numpy(float))
    con.register("stage5_full", full)
    con.execute(f"COPY stage5_full TO '{FULL_OUT.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("stage5_full")

    # OOT_Monitor blind performance by model/calibration method.
    ym = monitor.target.to_numpy(np.int8); monitor_values = {"scorecard": pd_sc_mon, "lgbm_raw": pd_lgb_mon}
    monitor_names = {"scorecard": "Champion Scorecard", "lgbm_raw": "Raw LightGBM M4"}
    perf_rows = []; tier_rows = []
    for key, pname in monitor_names.items():
        raw = monitor_values[key]
        for method, spec in [("Before Recalibration", None), ("Intercept Recal", specs[key]["intercept"]), ("Platt Recal", specs[key]["platt"])]:
            p = raw if spec is None else rec_p(raw, spec)
            perf_rows.append({"Model": pname, "Method": method, **metrics(ym, p)})
            for tr in tier_ae(ym, p): tier_rows.append({"Model": pname, "Method": method, **tr})
    perf = pd.DataFrame(perf_rows); tier_perf = pd.DataFrame(tier_rows)

    # Score PSI with quantile bins fixed on Train_Dev.
    train_stage3 = con.execute(f"SELECT scorecard_score,pd_scorecard,pd_lgbm_raw FROM read_parquet('{STAGE3.as_posix()}') WHERE split_name='Train_Dev' AND WEEK_NUM<=64").fetchdf()
    valid_stage3 = val[["scorecard_score", "pd_scorecard", "pd_lgbm_raw"]].copy()
    mon_stage3 = mon_pred[["scorecard_score", "pd_scorecard", "pd_lgbm_raw"]].copy()
    psi_rows = []
    for col, label in [("scorecard_score", "Champion Scorecard Score"), ("pd_scorecard", "Champion Scorecard PD"), ("pd_lgbm_raw", "Raw LightGBM M4 PD")]:
        edges = quantile_edges(train_stage3[col].to_numpy(float), 10)
        base_ids = numeric_bins(train_stage3[col].to_numpy(float), edges)
        for split_name, frame in [("OOT_Valid", valid_stage3), ("OOT_Monitor", mon_stage3)]:
            cur_ids = numeric_bins(frame[col].to_numpy(float), edges)
            psi = psi_from_bins(base_ids, cur_ids, len(edges))
            psi_rows.append({"Type": "Score PSI", "Feature / Score": label, "Comparison": f"Train_Dev vs {split_name}",
                             "PSI / CSI": psi, "Reference Status": "Major shift" if psi >= PSI_MAJOR else "Moderate shift" if psi >= PSI_MODERATE else "Below 0.10"})
    del train_stage3, valid_stage3, mon_stage3; gc.collect()

    # Characteristic stability: Train_Dev quantile/category bins define all CSI cuts.
    gain = lgb_model.booster_.feature_importance(importance_type="gain")
    top_lgb = [lgb_features[i] for i in np.argsort(gain)[::-1][:15]]
    csi_features = list(dict.fromkeys(score_features + top_lgb + ["has_tax_registry_record", "tax_verified_income_ratio"]))
    fq = ",".join('"' + c + '"' for c in csi_features)
    train_features = con.execute(f"SELECT {fq} FROM read_parquet('{MART.as_posix()}') WHERE WEEK_NUM<=64").fetchdf()
    valid_features = con.execute(f"SELECT {fq} FROM read_parquet('{MART.as_posix()}') WHERE WEEK_NUM BETWEEN 65 AND 77").fetchdf()
    mon_features = monitor[csi_features].copy()
    csi_rows = []
    for c in csi_features:
        vpsi, mpsi, _ = feature_csi(train_features[c], valid_features[c], mon_features[c])
        for split_name, value in [("OOT_Valid", vpsi), ("OOT_Monitor", mpsi)]:
            csi_rows.append({"Type": "Feature CSI", "Feature / Score": c, "Comparison": f"Train_Dev vs {split_name}",
                             "PSI / CSI": value, "Reference Status": "Major shift" if value >= PSI_MAJOR else "Moderate shift" if value >= PSI_MODERATE else "Below 0.10"})
    csi = pd.DataFrame(csi_rows)
    csi_pivot = csi.pivot(index="Feature / Score", columns="Comparison", values="PSI / CSI").reset_index()
    csi_pivot["Max CSI"] = csi_pivot[["Train_Dev vs OOT_Valid", "Train_Dev vs OOT_Monitor"]].max(axis=1)
    top_csi = csi_pivot.sort_values("Max CSI", ascending=False).head(15)
    stability = pd.concat([pd.DataFrame(psi_rows), csi[csi["Feature / Score"].isin(top_csi["Feature / Score"]) ]], ignore_index=True)
    del train_features, valid_features, mon_features; gc.collect()

    # Weekly/monthly tracking for all 0-91 weeks and recalibration methods.
    pred_map = {"Champion Scorecard": {"Before": "pd_scorecard", "Intercept": "pd_scorecard_recal_intercept", "Platt": "pd_scorecard_recal_platt"},
                "Raw LightGBM M4": {"Before": "pd_lgbm_raw", "Intercept": "pd_lgbm_recal_intercept", "Platt": "pd_lgbm_recal_platt"}}
    temporal_rows = []
    for grain, keycol in [("Week", "WEEK_NUM"), ("Month", "MONTH")]:
        for period, grp in full.groupby(keycol, dropna=False, sort=True):
            yy = grp.target.to_numpy(np.int8); count = len(grp)
            for model_name, methods in pred_map.items():
                for method, col in methods.items():
                    pp = grp[col].to_numpy(float); m = metrics(yy, pp) if np.unique(yy).size == 2 else {"KS": np.nan, "Gini": np.nan}
                    temporal_rows.append({"Grain": grain, "Period": period, "Model": model_name, "Method": method,
                        "Loan Volume": count, "Actual Bad Rate %": 100 * yy.mean(), "Mean Predicted PD %": 100 * pp.mean(),
                        "A/E Ratio": yy.mean() / pp.mean() if pp.mean() else np.nan, "KS": m["KS"], "Gini": m["Gini"]})
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(DATA / "stage5_temporal_tracking.csv", index=False)

    # OOT_Valid development cutoffs and risk appetite, then forward-apply to Monitor.
    valid_decisions = pd.read_parquet(STAGE4)
    valid_frame = valid_decisions.copy()
    valid_frame["pd_scorecard_recal_platt"] = rec_p(valid_frame.pd_scorecard.to_numpy(float), specs["scorecard"]["platt"])
    valid_frame["pd_lgbm_recal_platt"] = rec_p(valid_frame.pd_lgbm_raw.to_numpy(float), specs["lgbm_raw"]["platt"])
    valid_frame["target"] = valid_decisions.target.to_numpy(np.int8)
    vsc = loan_engine(valid_frame, valid_frame.pd_scorecard_recal_platt.to_numpy(float))
    vlgb = loan_engine(valid_frame, valid_frame.pd_lgbm_recal_platt.to_numpy(float))
    esc, elgb = eligible_mask(valid_frame, vsc), eligible_mask(valid_frame, vlgb)
    k75 = int(round(.75 * len(valid_frame))); k75 = min(k75, int(esc.sum()), int(elgb.sum()))
    va_sc = exact_k_approval(esc, valid_frame.pd_scorecard_recal_platt, valid_frame.case_id, k75)
    va_lgb = exact_k_approval(elgb, valid_frame.pd_lgbm_recal_platt, valid_frame.case_id, k75)
    appetite = float(valid_frame.target.to_numpy(np.int8)[va_sc].mean())
    # Choose the challenger Valid prefix whose observed bad rate is nearest the Scorecard 75% appetite.
    ids = np.flatnonzero(elgb); ord_l = np.lexsort((valid_frame.case_id.to_numpy()[ids], valid_frame.pd_lgbm_recal_platt.to_numpy()[ids]))
    ids = ids[ord_l]; y_order = valid_frame.target.to_numpy(np.int8)[ids]
    cum_rate = np.cumsum(y_order) / np.arange(1, len(ids) + 1)
    k_risk = int(np.argmin(np.abs(cum_rate - appetite)) + 1)
    risk_cut_lgb = float(valid_frame.pd_lgbm_recal_platt.to_numpy()[ids[k_risk - 1]])
    # Scorecard Iso-Risk baseline is the 75% scorecard cutoff learned on Valid.
    iso_cut_sc = float(np.max(valid_frame.pd_scorecard_recal_platt.to_numpy()[va_sc]))
    iso_cut_lgb_approval = float(np.max(valid_frame.pd_lgbm_recal_platt.to_numpy()[va_lgb]))
    iso_cut_lgb_risk = risk_cut_lgb

    # Stage 5 recalibrated values and Stage 4 driver engine on OOT_Monitor.
    mon = mon_pred.copy()
    mon["pd_scorecard_recal_platt"] = rec_p(mon.pd_scorecard, specs["scorecard"]["platt"])
    mon["pd_lgbm_recal_platt"] = rec_p(mon.pd_lgbm_raw, specs["lgbm_raw"]["platt"])
    mon_sc = loan_engine(monitor, mon.pd_scorecard_recal_platt.to_numpy(float))
    mon_lgb = loan_engine(monitor, mon.pd_lgbm_recal_platt.to_numpy(float))
    commercial_rows = []
    for mode, cut_sc, cut_lgb in [("Iso-Approval", iso_cut_sc, iso_cut_lgb_approval), ("Iso-Risk", iso_cut_sc, iso_cut_lgb_risk)]:
        for name, p, eng, cut in [("Champion Scorecard", mon.pd_scorecard_recal_platt.to_numpy(float), mon_sc, cut_sc),
                                  ("Challenger LightGBM M4", mon.pd_lgbm_recal_platt.to_numpy(float), mon_lgb, cut_lgb)]:
            approved = eligible_mask(monitor, eng) & (p <= cut)
            princ = np.where(approved, eng["limit"], 0.0)
            enp = expected_profit(princ, p, eng["apr"], eng["term"])
            commercial_rows.append({"Operating Mode": mode, "Model": name, "PD Cutoff (OOT_Valid)": cut,
                "Approval Rate %": 100 * approved.mean(), "Approved Count": int(approved.sum()),
                "Observed Bad Rate %": 100 * mon.target.to_numpy(np.int8)[approved].mean() if approved.any() else np.nan,
                "Total Recommended Loan Volume": float(princ.sum()), "Simulated ENP": float(enp.sum()),
                "Valid Reference Bad Rate %": 100 * appetite if mode == "Iso-Risk" else np.nan})
    commercial = pd.DataFrame(commercial_rows)

    # Fair Lending audit on monitor only, using Iso-Approval operational cutoffs and Platt PDs.
    approval_sc = eligible_mask(monitor, mon_sc) & (mon.pd_scorecard_recal_platt.to_numpy(float) <= iso_cut_sc)
    approval_lgb = eligible_mask(monitor, mon_lgb) & (mon.pd_lgbm_recal_platt.to_numpy(float) <= iso_cut_lgb_approval)
    # Age bins are learned only on Train_Dev, then applied to Monitor.
    train_age = con.execute(f"SELECT person_age FROM read_parquet('{MART.as_posix()}') WHERE WEEK_NUM<=64").fetchdf().person_age.to_numpy(np.float32)
    age_edges = np.unique(np.quantile(train_age, [0, .2, .4, .6, .8, 1]))
    age_edges[0], age_edges[-1] = -np.inf, np.inf
    age_bins = ["Age Q1", "Age Q2", "Age Q3", "Age Q4", "Age Q5"][:len(age_edges)-1]
    age_group = pd.cut(monitor.person_age, bins=age_edges, labels=age_bins, include_lowest=True).astype("string").fillna("Missing").to_numpy(dtype=object)
    group_specs = {
        "Gender": monitor.person_gender_sensitive.astype("string").fillna("Unknown").to_numpy(dtype=object),
        "Age Band": monitor.person_age_band_sensitive.astype("string").fillna("Unknown").to_numpy(dtype=object),
        "Age (Train_Dev quintiles)": age_group,
    }
    fairness_rows = []
    for feature, groups in group_specs.items():
        vals = np.unique(groups)
        for model_name, approved, recalpd in [("Champion Scorecard", approval_sc, mon.pd_scorecard_recal_platt.to_numpy(float)),
                                             ("Challenger LightGBM M4", approval_lgb, mon.pd_lgbm_recal_platt.to_numpy(float))]:
            observed_groups = [g for g in vals if g != "Unknown"]
            ref = max((float(approved[groups == g].mean()) for g in observed_groups if np.any(groups == g)), default=np.nan)
            for g in vals:
                mask = groups == g
                if not mask.any(): continue
                ar = float(approved[mask].mean()); bad = float(mon.target.to_numpy(np.int8)[mask].mean()); meanp = float(recalpd[mask].mean())
                fairness_rows.append({"Sensitive Attribute": feature, "Group": str(g), "Model": model_name,
                    "Sample Count": int(mask.sum()), "Sample Share %": 100 * float(mask.mean()), "Approval Rate %": 100 * ar,
                    "Reference Highest-Group Approval %": 100 * ref, "AIR": ar / ref if ref and np.isfinite(ref) else np.nan,
                    "AIR < 0.80 Flag": bool((ar / ref) < .80) if ref and np.isfinite(ref) else False,
                    "Observed Bad Rate %": 100 * bad, "Mean Recalibrated PD %": 100 * meanp,
                    "Subgroup A/E Ratio": bad / meanp if meanp else np.nan})
    fairness = pd.DataFrame(fairness_rows)
    del train_age; gc.collect()

    # Reliability curves on OOT_Monitor (diagnostic only; no model fitting here).
    fig, ax = plt.subplots(figsize=(7, 6)); y_arr = ym
    for key, label in [("scorecard", "Scorecard"), ("lgbm_raw", "Raw LightGBM M4")]:
        raw = monitor_values[key]
        ss = specs[key]
        for method, p, ls in [("Before", raw, "--"), ("Intercept", rec_p(raw, ss["intercept"]), ":"), ("Platt", rec_p(raw, ss["platt"]), "-")]:
            frac, mean = calibration_curve(y_arr, p, n_bins=10, strategy="quantile")
            ax.plot(mean, frac, marker="o", linestyle=ls, label=f"{label} {method}")
    ax.plot([0, .12], [0, .12], color="black", linewidth=1, alpha=.6, label="Perfect calibration")
    ax.set(xlabel="Mean predicted PD", ylabel="Observed bad rate", title="OOT_Monitor Calibration Before / After")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGS / "stage5_calibration_before_after.png", dpi=160); plt.close(fig)

    # Temporal trend plot from W0-91; monthly actual/predicted PD and KS.
    monthly = temporal[(temporal.Grain == "Month") & (temporal.Method.isin(["Before", "Platt"]))]
    months = sorted(monthly.Period.dropna().unique(), key=lambda x: str(x))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    actual = monthly[monthly.Model == "Champion Scorecard"].drop_duplicates("Period").set_index("Period")["Actual Bad Rate %"]
    axes[0].plot(months, [actual.get(m, np.nan) for m in months], color="black", marker="o", label="Actual Bad Rate")
    for model_name, method in [("Champion Scorecard", "Platt"), ("Raw LightGBM M4", "Platt"), ("Champion Scorecard", "Before"), ("Raw LightGBM M4", "Before")]:
        z = monthly[(monthly.Model == model_name) & (monthly.Method == method)].set_index("Period")
        axes[0].plot(months, [z["Mean Predicted PD %"].get(m, np.nan) for m in months], marker=".", label=f"{model_name} {method}")
        axes[1].plot(months, [z["KS"].get(m, np.nan) for m in months], marker=".", label=f"{model_name} {method}")
    axes[0].set_ylabel("Rate (%)"); axes[0].set_title("Monthly Bad Rate / Mean PD")
    axes[1].set_ylabel("KS"); axes[1].set_title("Monthly KS Stability"); axes[1].set_xlabel("MONTH")
    for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "stage5_temporal_ks_badrate_trend.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7)); plot = top_csi.sort_values("Max CSI").reset_index(drop=True)
    y = np.arange(len(plot))
    ax.barh(y - .18, plot["Train_Dev vs OOT_Valid"], height=.35, label="OOT_Valid")
    ax.barh(y + .18, plot["Train_Dev vs OOT_Monitor"], height=.35, label="OOT_Monitor")
    ax.set_yticks(y, plot["Feature / Score"])
    ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label="0.10 reference")
    ax.axvline(PSI_MAJOR, color="red", linestyle=":", label="0.25 reference")
    ax.set_xlabel("CSI (operational reference thresholds)"); ax.set_title("Top 15 Feature CSI Drift")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "stage5_top_csi_drift.png", dpi=160); plt.close(fig)

    # Save monitoring tables as CSV so weekly/monthly and CSI values remain auditable.
    pd.concat([pd.DataFrame(psi_rows), csi], ignore_index=True).to_csv(DATA / "stage5_psi_csi_summary.csv", index=False)
    tier_perf_df = pd.DataFrame(tier_rows)
    perf_df = perf
    drift_info = ("0.10 (moderate) and 0.25 (major) are configurable operational monitoring reference thresholds only, not universal statutory standards.")
    report = f"""# Stage 5 OOT Monitoring、Recalibration 与 Fair Lending 审计

`OOT_Valid` (`WEEK_NUM 65–77`) exclusively fitted both recalibrators. `OOT_Monitor` (`WEEK_NUM 78–91`, 151,707 rows) was scored only after parameters were frozen and used only for forward blind evaluation. Stage 5 Challenger is the approved `pd_lgbm_raw`, not the Stage 3 isotonic output. `OOT_Monitor` labels were not used in recalibration, CSI bins, tier/cutoff fitting, or Iso-Risk threshold selection.

## Recalibration parameters

- Scorecard Method A: alpha={specs['scorecard']['intercept']['alpha']:.6f}, beta fixed at 1. `Base_Score_Offset` changes from {offset0:.4f} to {offset0 + shift:.4f}, a shift of {shift:.4f} points (PDO factor {factor:.4f}); the score ranking and PDO factor are preserved.
- Scorecard Method B: alpha={specs['scorecard']['platt']['alpha']:.6f}, beta={specs['scorecard']['platt']['beta']:.6f}.
- Raw LightGBM M4 Method A: alpha={specs['lgbm_raw']['intercept']['alpha']:.6f}, beta fixed at 1.
- Raw LightGBM M4 Method B: alpha={specs['lgbm_raw']['platt']['alpha']:.6f}, beta={specs['lgbm_raw']['platt']['beta']:.6f}.

## Table 1: Score PSI 与 Top-15 Feature CSI

{report_table(stability)}

{drift_info}

### Top 15 Feature CSI ranking

{report_table(top_csi)}

## Table 2: Blind OOT_Monitor Performance Before / After Recalibration

{report_table(perf_df)}

### Tier-level A/E by method (tiered on each method's PD bands)

{report_table(tier_perf_df)}

Intercept-only recalibration preserves exact rank when beta=1. Platt recalibration also preserves ordering if beta remains positive; both conditions are verified from fitted parameters. Weekly and monthly tracking for weeks 0–91 is saved in `data/processed/stage5_temporal_tracking.csv`.

## Table 3: OOT_Monitor Iso-Approval vs Iso-Risk Commercial Comparison

Iso-Approval thresholds match 75% acceptance on OOT_Valid by unlabeled risk rank and then apply forward to OOT_Monitor. Iso-Risk sets the Challenger cutoff on OOT_Valid to match Champion Scorecard's observed bad rate at its 75% OOT_Valid operating point, then applies the cutoff forward to OOT_Monitor. OOT_Monitor outcomes are not used to choose either cutoff.

{report_table(commercial)}

Parametric ENP assumptions follow Stage 4: COF=4.0%, OPEX=1.5%, assumed LGD=75%, APR 9%/12%/15%/18% by tiers A–D capped at 18%; these are commercial simulation assumptions, not statutory constants.

## Table 4: OOT_Monitor Fair Lending Subgroup Audit

{report_table(fairness)}

AIR uses the highest observed subgroup approval rate within each protected attribute and model as reference. The 0.80 four-fifths value is a diagnostic benchmark only, not a legal conclusion. `person_gender_sensitive` is 99.17% missing on Train_Dev; the `Unknown` subgroup is retained, and sparse observed gender groups should be interpreted with their sample counts. Subgroup A/E uses the full observed subgroup, not only approved cases. Sensitive attributes were not included in either model.

## Rank, drift, and portfolio interpretation

Weekly/monthly `Loan Volume`, `Actual Bad Rate`, `Mean Predicted PD`, `A/E Ratio`, `KS`, and `Gini` cover Weeks 0–91 in the saved temporal CSV. Score/feature drift is diagnosed alongside discrimination metrics so calibration shift can be distinguished from rank-order deterioration. Feature CSI includes all {len(score_features)} Champion scorecard features, top LightGBM gain features, and flagged `has_tax_registry_record` / `tax_verified_income_ratio`.

## Audit Summary and Artifacts

Full predictions contain {len(full):,} rows, weeks 0–91; `OOT_Monitor` rows={len(mon):,}; monitoring split count and dates are guarded by code. `recalibrators_stage5.json` records fit weeks 65–77 and `oot_monitor_used_for_fit=false`. `has_tax_registry_record` and `tax_verified_income_ratio` remain Stage 5 governance features and their PSI/CSI are included above.

Saved `data/processed/stage5_full_predictions.parquet`, `data/processed/models/recalibrators_stage5.json`, `data/processed/stage5_temporal_tracking.csv`, `data/processed/stage5_psi_csi_summary.csv`, and figures in `reports/figures/`.

**HARD STOP:** Stage 5 complete. Wait for human review before Stage 6 or Stage 7.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    say("TABLE 1 — Score PSI and Top CSI\n" + report_table(stability))
    say("TABLE 2 — OOT_Monitor recalibration metrics\n" + report_table(perf_df))
    say("TABLE 3 — Iso-Approval / Iso-Risk\n" + report_table(commercial))
    say("TABLE 4 — Fair Lending\n" + report_table(fairness))
    say(f"Saved {FULL_OUT}; {len(full):,} timeline predictions; OOT_Monitor rows {len(mon):,}; HARD STOP.")
    con.close()


if __name__ == "__main__":
    main()
