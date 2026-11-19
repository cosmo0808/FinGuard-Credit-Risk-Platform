import gc
import json
import os
import pickle
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".matplotlib"))

import duckdb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=RuntimeWarning)
DATA = ROOT / "data" / "processed"
REPORT = ROOT / "reports"
FIG = REPORT / "figures"
MODELS = DATA / "models"
for p in (FIG, MODELS, ROOT / "duckdb_temp"):
    p.mkdir(parents=True, exist_ok=True)
MART = DATA / "finguard_feature_mart.parquet"

def say(x):
    print(x, flush=True)

def psi_numeric(a, b, bins=10):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    n_a, n_b = len(a), len(b)
    miss_a = ~np.isfinite(a); miss_b = ~np.isfinite(b)
    af = a[~miss_a]; bf = b[~miss_b]
    categories=np.union1d(np.unique(af),np.unique(bf))
    if len(categories) <= bins:
        pa=np.r_[np.array([np.sum(af==v)/max(1,n_a) for v in categories]),np.mean(miss_a)]
        pb=np.r_[np.array([np.sum(bf==v)/max(1,n_b) for v in categories]),np.mean(miss_b)]
        pa=np.maximum(pa,1e-4); pb=np.maximum(pb,1e-4)
        return float(np.sum((pa-pb)*np.log(pa/pb)))
    a = af; b = bf
    if len(a) == 0 or len(b) == 0: return 0.0
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 2: return 0.0
    q[0], q[-1] = -np.inf, np.inf
    ha = np.histogram(a, q)[0].astype(float); hb = np.histogram(b, q)[0].astype(float)
    pa = np.r_[ha / max(1, n_a), np.mean(miss_a)]
    pb = np.r_[hb / max(1, n_b), np.mean(miss_b)]
    pa = np.maximum(pa, 1e-4); pb = np.maximum(pb, 1e-4)
    return float(np.sum((pa - pb) * np.log(pa / pb)))

def psi_categories(a, b):
    a=np.asarray(["__MISSING__" if pd.isna(v) else str(v) for v in a],dtype=object)
    b=np.asarray(["__MISSING__" if pd.isna(v) else str(v) for v in b],dtype=object)
    cats=np.union1d(np.unique(a),np.unique(b))
    pa=np.array([(a==v).mean() for v in cats],float)
    pb=np.array([(b==v).mean() for v in cats],float)
    pa=np.maximum(pa,1e-4); pb=np.maximum(pb,1e-4)
    return float(np.sum((pa-pb)*np.log(pa/pb)))

def metrics(y, p):
    y = np.asarray(y, np.int8); p = np.clip(np.asarray(p, np.float64), 1e-8, 1-1e-8)
    auc = roc_auc_score(y, p); fpr, tpr, _ = roc_curve(y, p)
    return {"ROC-AUC": auc, "PR-AUC": average_precision_score(y, p), "KS": float(np.max(tpr-fpr)),
            "Gini": 2*auc-1, "Brier": brier_score_loss(y, p), "Mean Predicted PD": float(p.mean()),
            "Actual Bad Rate": float(y.mean()), "A/E Ratio": float(y.mean()/p.mean())}

def vif_values(x):
    if x.shape[1] == 0: return np.array([])
    return np.array([variance_inflation_factor(x, i) for i in range(x.shape[1])])

def scorecard_score(logit, x):
    # score = base + factor * ln(good/bad odds); logit predicts bad/good odds.
    factor = 50.0 / np.log(2)
    base = 600.0 + factor * np.log(0.05)
    return base - factor * np.asarray(logit, np.float64)

def deciles(y, p, name):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    # rank(method=first) makes ties deterministic and guarantees ten populated groups.
    d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=False) + 1
    d = d.groupby("decile", observed=False).agg(Count=("y","size"), **{"Bad Count":("y","sum"), "Mean PD":("p","mean")}).reset_index()
    d = d.sort_values("decile"); d["Marginal Bad Rate %"] = d["Bad Count"] / d.Count * 100
    total = d["Bad Count"].sum(); d["Cum Bad Capture %"] = d["Bad Count"].cumsum()/total*100
    overall = total/d.Count.sum(); d["Lift"] = (d["Bad Count"]/d.Count)/overall
    cum_bad = d["Bad Count"].cumsum()/total; cum_pop=d.Count.cumsum()/d.Count.sum()
    d["KS"] = (cum_bad-cum_pop)*100
    d["Decile"] = [f"Decile {i} [{ 'Highest' if i==1 else 'Lowest' if i==10 else 'Risk' } Risk]" if i in (1,10) else f"Decile {i} [Risk]" for i in d.decile]
    d.insert(0,"Model",name)
    return d[["Model","Decile","Count","Bad Count","Marginal Bad Rate %","Cum Bad Capture %","KS","Lift"]]

def md_table(df, digits=4):
    return df.to_markdown(index=False, floatfmt=f".{digits}f")

def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'"); con.execute("SET threads=4")
    con.execute(f"SET temp_directory='{(ROOT/'duckdb_temp').as_posix()}'")
    mapping = pd.read_csv(DATA/"feature_group_mapping.csv")
    names = mapping.column_name.tolist(); groups=dict(zip(mapping.column_name,mapping.feature_group))
    base={"case_id","date_decision","MONTH","WEEK_NUM","target","split_name"}
    sensitive={"person_gender_sensitive","person_age_band_sensitive","person_age"}
    # Train_Dev only is read to calculate eligibility and fit/calibrate models.
    train_cols=[c for c in names if c not in base]
    qcols=", ".join('"'+c+'"' for c in ["case_id","WEEK_NUM","target","split_name"]+train_cols)
    say("Loading Train_Dev feature mart through DuckDB...")
    train=con.execute(f"SELECT {qcols} FROM read_parquet('{MART.as_posix()}') WHERE WEEK_NUM<=64").fetchdf()
    if train.shape[0] != 1292350: say(f"WARNING: Train_Dev rows={len(train):,} (expected 1,292,350)")
    # Ensure floats are float32 before feature matrices are materialized.
    for c in train_cols:
        if pd.api.types.is_numeric_dtype(train[c]): train[c]=train[c].astype(np.float32)
    y=train.target.to_numpy(dtype=np.int8); weeks=train.WEEK_NUM.to_numpy()
    selected=pd.read_csv(DATA/"stage2_selected_features.csv").feature_name.tolist()
    selected=[c for c in selected if c!="disbursedcredamount_1113A"]
    with open(DATA/"woe_binning_dict.pkl","rb") as f: binners=pickle.load(f)
    say(f"WOE pickle inspected: {type(binners).__name__}, {len(binners)} OptimalBinning objects; transform(..., metric='woe').")
    # Portable frozen lookup so downstream stages do not need optbinning/ortools.
    rules={}
    for name,ob in binners.items():
        bt=ob.binning_table.build()
        special=bt[bt["Bin"].astype(str)=="Special"]
        missing=bt[bt["Bin"].astype(str)=="Missing"]
        regular=bt[~bt["Bin"].astype(str).str.strip().isin(["","Special","Missing","Totals"])].reset_index(drop=True)
        rows=[]
        if ob.dtype=="numerical":
            bounds=["-Infinity"]+np.asarray(ob.splits,dtype=float).tolist()+["Infinity"]
            for i,(_,r) in enumerate(regular.iterrows()): rows.append({"lower":bounds[i],"upper":bounds[i+1],"woe":float(r["WoE"])})
        else:
            for i,(_,r) in enumerate(regular.iterrows()): rows.append({"categories":[str(v) for v in np.asarray(ob.splits[i]).tolist()],"woe":float(r["WoE"])})
        special_codes=getattr(ob,"special_codes",None)
        if isinstance(special_codes,np.ndarray): special_codes=special_codes.tolist()
        probe=np.array([np.nan],dtype=np.float64 if ob.dtype=="numerical" else object)
        missing_woe=float(ob.transform(probe,metric="woe")[0])
        unknown_woe=None
        if ob.dtype=="categorical":
            unknown_woe=float(ob.transform(np.array(["__FIN_GUARD_UNKNOWN_CATEGORY__"],dtype=object),metric="woe")[0])
        rules[name]={"dtype":ob.dtype,"bins":rows,"special_codes":special_codes,
                     "special_woe":float(special["WoE"].iloc[0]) if len(special) else None,
                     "missing_woe":missing_woe,"unknown_woe":unknown_woe}
    (DATA/"woe_bin_rules.json").write_text(json.dumps(rules,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")
    # Feature audit for named Group 4 proxies; Stage 2 eligibility from Train_Dev only.
    audit=[]
    for c in ["has_tax_registry_record","has_debitcard","tax_verified_income_ratio","app_cnt_last_30d"]:
        if c not in train: continue
        a=train.loc[weeks<=44,c].to_numpy(np.float32); b=train.loc[weeks>=45,c].to_numpy(np.float32)
        null=float(np.isnan(train[c].to_numpy(np.float32)).mean()); vals=train[c].dropna().to_numpy(np.float32)
        dom=float(pd.Series(vals).value_counts(normalize=True).iloc[0]) if len(vals) else 1.
        psi=psi_numeric(a,b)
        audit.append({"Feature":c,"Null %":100*null,"Dominant %":100*dom,"Internal PSI":psi,
                      "Gate 1": "Pass" if null<=.95 and dom<=.98 else "Fail",
                      "PSI < 0.10": "Pass" if psi<.1 else "Fail"})
    auditdf=pd.DataFrame(audit)
    # Explain actual Stage 2 exclusion reason for tax/debit by re-fitting exact Gate 2 binners.
    from optbinning import OptimalBinning
    gate_reason={}
    for c in ["has_tax_registry_record","has_debitcard"]:
        if c not in train: gate_reason[c]="Column absent from feature mart"; continue
        x=train[c].to_numpy(np.float32)
        try:
            ob=OptimalBinning(name=c,dtype="numerical",monotonic_trend="auto_asc_desc",min_bin_size=.05,max_n_bins=6)
            ob.fit(x,y); ob.binning_table.build(); iv=float(ob.binning_table.iv)
            if iv<.02: gate_reason[c]=f"Gate 2 IV={iv:.6f} (<0.02)"
            elif iv>.60: gate_reason[c]=f"Gate 2 IV={iv:.6f} (>0.60)"
            else:
                psi=psi_categories(ob.transform(train.loc[weeks<=44,c].to_numpy(np.float32),metric="indices"),ob.transform(train.loc[weeks>=45,c].to_numpy(np.float32),metric="indices"))
                gate_reason[c]=f"Gate 2 passed (IV={iv:.6f}); internal WOE-bin PSI={psi:.6f}; Stage 2 report does not preserve per-feature Gate 4 exclusion records."
        except Exception as e: gate_reason[c]=f"Binning error: {e}"
    say("Verified Stage 2 proxy reasons: " + json.dumps(gate_reason,ensure_ascii=False))
    # Part A: frozen bin transformations; bounded one feature at a time, float32 design.
    say("Transforming scorecard WOE features...")
    X=np.empty((len(train),len(selected)),dtype=np.float32)
    for j,c in enumerate(selected): X[:,j]=binners[c].transform(train[c].to_numpy(),metric="woe").astype(np.float32)
    fit_idx=np.flatnonzero(weeks<=64); ytd=y[fit_idx]; x=X[fit_idx]
    keep=list(range(len(selected)))
    while keep:
        xx=sm.add_constant(x[:,keep],has_constant="add")
        try: res=sm.Logit(ytd,xx,missing="raise").fit(disp=False,maxiter=100)
        except Exception:
            # singularity: remove the least informative retained column
            keep.pop(); continue
        co=np.asarray(res.params[1:]); pv=np.asarray(res.pvalues[1:]); vf=vif_values(x[:,keep])
        bad=[i for i in range(len(keep)) if co[i]>=0 or pv[i]>=.05 or vf[i]>=5 or not np.isfinite(vf[i])]
        if not bad: break
        # remove worst failure, prioritizing sign, then p-value, then VIF
        ranks=[]
        for i in bad:
            ranks.append(((2 if co[i]>=0 else 0)+(1 if pv[i]>=.05 else 0)+(1 if vf[i]>=5 or not np.isfinite(vf[i]) else 0), max(pv[i],0), max(vf[i],0), i))
        keep.pop(max(ranks)[-1])
    if not keep:
        res=sm.Logit(ytd,np.ones((len(ytd),1))).fit(disp=False)
    kept=[selected[i] for i in keep]
    coef=pd.Series(np.asarray(res.params[1:]),index=kept,dtype=float)
    pvals=pd.Series(np.asarray(res.pvalues[1:]),index=kept,dtype=float)
    stderr=pd.Series(np.asarray(res.bse[1:]),index=kept,dtype=float)
    zvals=pd.Series(np.asarray(res.tvalues[1:]),index=kept,dtype=float)
    finalvif=pd.Series(vif_values(x[:,keep]),index=kept,dtype=float) if keep else pd.Series(dtype=float)
    pd_sc=np.asarray(res.predict(sm.add_constant(x[:,keep],has_constant="add")),np.float64)
    score=scorecard_score(np.log(pd_sc/(1-pd_sc)),x[:,keep])
    recon=1/(1+np.exp(-((600-score)/(50/np.log(2))+np.log(.05))))
    recon_err=float(np.max(np.abs(recon-pd_sc)))
    assert recon_err<1e-10, f"Score reconstruction error {recon_err}"
    with open(MODELS/"scorecard_logit.pkl","wb") as f: pickle.dump({"result":res,"features":kept,"all_features":selected,"factor":50/np.log(2),"base_score":600+50/np.log(2)*np.log(.05)},f)
    # Points per WOE bin, with Stage 2 ranges and Train_Dev proportions/rates.
    points=[]
    for c in kept:
        ob=binners[c]; bt=ob.binning_table.build().reset_index()
        for _,r in bt.iterrows():
            woe=pd.to_numeric(pd.Series([r.get("WoE",np.nan)]),errors="coerce").iloc[0]
            if not np.isfinite(woe): continue
            mask=np.isclose(X[:,selected.index(c)],float(woe),rtol=0,atol=1e-7)
            cnt=int(mask.sum()); badn=int(y[mask].sum())
            points.append({"Feature":c,"Group":groups.get(c,"Other"),"Bin Range":str(r.get("Bin","")),"Train Count %":cnt/len(y)*100,
                          "Train Bad Rate %":(100*badn/cnt if cnt else np.nan),"WOE":woe,"Coef":coef[c],"Bin Points":-(50/np.log(2))*coef[c]*woe})
    pd.DataFrame(points).to_csv(DATA/"scorecard_points_table.csv",index=False)
    # Part B candidate pool: Gate 1 and Train_Dev early/late raw PSI, groups 1-4 and affordability only.
    excluded=base|sensitive|{"disbursedcredamount_1113A"}
    candidates=[]; psi_map={}; category_levels={}
    for c in train_cols:
        if c in excluded or groups.get(c) not in {"Group_1_Traditional_Credit","Group_2_Repayment_Behaviour","Group_3_Application_Demographic","Group_4_Alternative_Proxies","Core_Affordability"}: continue
        s=train[c]
        if pd.api.types.is_numeric_dtype(s):
            a=s.to_numpy(np.float32); null=np.isnan(a).mean(); vals=a[np.isfinite(a)]
        else:
            null=float(s.isna().mean()); vals=s.dropna().astype(str).to_numpy()
        dom=float(pd.Series(vals).value_counts(normalize=True).iloc[0]) if len(vals) else 1.
        if null>.95 or dom>.98: continue
        if pd.api.types.is_numeric_dtype(s): ps=psi_numeric(a[weeks<=44],a[weeks>=45])
        else:
            vals=s.astype("string").fillna("__MISSING__").to_numpy(dtype=object)
            ps=psi_categories(vals[weeks<=44],vals[weeks>=45])
            category_levels[c]=sorted(s.dropna().astype(str).unique().tolist())
        psi_map[c]=ps
        if ps<.10: candidates.append(c)
    # Include the four explicitly requested Group 4 proxies for challenger testing.
    desired=["has_tax_registry_record","has_debitcard","tax_verified_income_ratio","app_cnt_last_30d"]
    for c in desired:
        if c in train_cols and c not in candidates:
            s=train[c]
            if pd.api.types.is_numeric_dtype(s):
                a=s.to_numpy(np.float32); vals=a[np.isfinite(a)]; null=np.isnan(a).mean()
            else:
                vals=s.dropna().astype(str).to_numpy(); null=float(s.isna().mean())
            dom=float(pd.Series(vals).value_counts(normalize=True).iloc[0]) if len(vals) else 1.
            if null<=.95 and dom<=.98:
                candidates.append(c)
                if pd.api.types.is_numeric_dtype(s): psi_map.setdefault(c,psi_numeric(a[weeks<=44],a[weeks>=45]))
                else:
                    vals=s.astype("string").fillna("__MISSING__").to_numpy(dtype=object)
                    psi_map.setdefault(c,psi_categories(vals[weeks<=44],vals[weeks>=45]))
                    category_levels[c]=sorted(s.dropna().astype(str).unique().tolist())
    auditdf["M4 Candidate"] = auditdf.Feature.map(lambda c: "Yes" if c in candidates else "No")
    auditdf["PSI Override"] = auditdf.apply(lambda r: "Explicit Group 4 proxy inclusion" if r.Feature in desired and r.Feature in candidates and r["Internal PSI"]>=.10 else "—",axis=1)
    say(f"LightGBM candidate pool: {len(candidates)} features (Gate 1 and PSI eligible, plus explicit Gate 1-qualified Group 4 proxy inclusions).")
    # Convert missing to -999, retain float32. Explicit constraints by domain semantics.
    constraints=[]
    for c in candidates:
        cl=c.lower()
        if any(t in cl for t in ["dpd","delinq","rejection_rate","queries_","inquiry"]): constraints.append(1)
        elif any(t in cl for t in ["paid_early","early_payment","employment_years"]): constraints.append(-1)
        else: constraints.append(0)
    lgb_x=pd.DataFrame(index=train.index)
    for c in candidates:
        if c in category_levels:
            lgb_x[c]=pd.Categorical(train[c].astype("string"),categories=category_levels[c])
        else:
            a=train[c].to_numpy(dtype=np.float32,copy=True); a[~np.isfinite(a)]=-999; lgb_x[c]=a
    tr=weeks<=54; cal=(weeks>=55)&(weeks<=64)
    yall=y
    group_order=["Group_1_Traditional_Credit","Group_2_Repayment_Behaviour","Group_3_Application_Demographic","Group_4_Alternative_Proxies","Core_Affordability"]
    ablations=[]; models={}; probs_cal={}; probs_oot={}
    # Load OOT only after all fitting/calibration steps; it is used for blind inference/evaluation.
    for stage,gs in enumerate([group_order[:1],group_order[:2],group_order[:3],group_order],1):
        cols=[c for c in candidates if groups.get(c) in gs]
        if not cols: continue
        idx=[candidates.index(c) for c in cols]; cons=[constraints[i] for i in idx]
        cat_cols=[c for c in cols if c in category_levels]
        model=lgb.LGBMClassifier(objective="binary",n_estimators=3000,learning_rate=.03,num_leaves=31,max_depth=-1,
            subsample=.8,colsample_bytree=.8,reg_lambda=2,n_jobs=4,verbosity=-1,monotone_constraints=cons)
        model.fit(lgb_x.loc[tr,cols],yall[tr],categorical_feature=cat_cols,
                  eval_set=[(lgb_x.loc[cal,cols],yall[cal])],callbacks=[lgb.early_stopping(100,verbose=False)])
        models[f"M{stage}"]=model
        pcal=model.predict_proba(lgb_x.loc[cal,cols])[:,1]
        # M4 Dev_Calib only: calibration must not see OOT.
        if stage==4:
            calibrator=IsotonicRegression(out_of_bounds="clip").fit(pcal,yall[cal])
            with open(MODELS/"isotonic_calibrator.pkl","wb") as f: pickle.dump(calibrator,f)
            m4cols=cols; m4idx=idx; m4cons=cons; m4model=model
        probs_cal[f"M{stage}"]=pcal
        ablations.append((f"M{stage}",model,idx,cols))
    if "M4" not in models: raise RuntimeError("M4 has no eligible features")
    with open(MODELS/"lgbm_m4.pkl","wb") as f: pickle.dump({"model":m4model,"features":m4cols,"constraints":m4cons},f)
    # Release Train_Dev matrix before pulling OOT feature data.
    del lgb_x, x, X
    gc.collect()
    oot=con.execute(f"SELECT {qcols} FROM read_parquet('{MART.as_posix()}') WHERE WEEK_NUM BETWEEN 65 AND 77").fetchdf()
    for c in candidates:
        if pd.api.types.is_numeric_dtype(oot[c]): oot[c]=oot[c].astype(np.float32)
    yo=oot.target.to_numpy(np.int8); weeks_o=oot.WEEK_NUM.to_numpy()
    def matrix(df, cols):
        out=pd.DataFrame(index=df.index)
        for c in cols:
            if c in category_levels:
                out[c]=pd.Categorical(df[c].astype("string"),categories=category_levels[c])
            else:
                a=df[c].to_numpy(np.float32,copy=True); a[~np.isfinite(a)]=-999; out[c]=a
        return out
    p_oot={}
    for name,model,idx,cols in ablations: p_oot[name]=model.predict_proba(matrix(oot,cols))[:,1]
    pcal_raw=probs_cal["M4"]; pcal_cal=calibrator.predict(pcal_raw)
    poot_raw=p_oot["M4"]; poot_cal=calibrator.predict(poot_raw)
    train_raw=m4model.predict_proba(matrix(train,m4cols))[:,1]
    train_cal=calibrator.predict(train_raw)
    # Scorecard OOT transformations with saved binning objects.
    xso=np.empty((len(oot),len(kept)),np.float32)
    for j,c in enumerate(kept): xso[:,j]=binners[c].transform(oot[c].to_numpy(),metric="woe").astype(np.float32)
    pso=np.asarray(res.predict(sm.add_constant(xso,has_constant="add")),np.float64)
    scoreo=scorecard_score(np.log(pso/(1-pso)),xso)
    # Metrics rows, Train_Dev scorecard and LightGBM uses all training predictions from fitted model;
    # report in-sample diagnostic explicitly. Train_Dev scorecard is model-fit data.
    trainmetrics=[{"Split":"Train_Dev (fit diagnostic)","Model":"Champion Scorecard",**metrics(y,pd_sc)},
                  {"Split":"Train_Dev (fit diagnostic)","Model":"Raw LightGBM M4",**metrics(y,train_raw)},
                  {"Split":"Train_Dev (fit diagnostic)","Model":"Calibrated LightGBM M4",**metrics(y,train_cal)},
                  {"Split":"Dev_Calib","Model":"Raw LightGBM M4",**metrics(yall[cal],pcal_raw)},
                  {"Split":"Dev_Calib","Model":"Calibrated LightGBM M4",**metrics(yall[cal],pcal_cal)},
                  {"Split":"OOT_Valid (blind)","Model":"Champion Scorecard",**metrics(yo,pso)},
                  {"Split":"OOT_Valid (blind)","Model":"Raw LightGBM M4",**metrics(yo,poot_raw)},
                  {"Split":"OOT_Valid (blind)","Model":"Calibrated LightGBM M4",**metrics(yo,poot_cal)}]
    mdf=pd.DataFrame(trainmetrics)
    # Ablation ROC-AUC / KS / Gini on calibration and blind OOT.
    abl=[]
    for name,model,idx,cols in ablations:
        cmet=metrics(yall[cal],probs_cal[name]); omet=metrics(yo,p_oot[name])
        abl.append({"Model":name,"Features":len(cols),"Dev_Calib ROC-AUC":cmet["ROC-AUC"],"Dev_Calib KS":cmet["KS"],"Dev_Calib Gini":cmet["Gini"],
                    "OOT_Valid ROC-AUC":omet["ROC-AUC"],"OOT_Valid KS":omet["KS"],"OOT_Valid Gini":omet["Gini"]})
    abldf=pd.DataFrame(abl)
    for metric in ["ROC-AUC","KS","Gini"]:
        for split in ["Dev_Calib","OOT_Valid"]:
            col=f"{split} {metric}"
            abldf[f"{split} {metric} Lift"] = abldf[col].diff()
            abldf.loc[0,f"{split} {metric} Lift"] = "—"
    # Decile risk-order check: Marginal bad rate descending expected. Flag if violated.
    dec=pd.concat([deciles(yo,pso,"Champion Scorecard"),deciles(yo,poot_cal,"Calibrated LightGBM")],ignore_index=True)
    rankcheck=dec.groupby("Model")["Marginal Bad Rate %"].apply(lambda s: bool(np.all(np.diff(s.to_numpy())<=1e-12)))
    say(f"OOT decile strict marginal bad-rate descending check: {rankcheck.to_dict()}")
    # Save Train_Dev plus OOT predictions; OOT_Monitor never queried.
    # Fit prediction vectors from feature arrays already held: Train_Dev model outputs above.
    predtr=pd.DataFrame({"case_id":train.case_id,"WEEK_NUM":weeks,"split_name":train.split_name,"target":y,
        "scorecard_score":score,"pd_scorecard":pd_sc,"pd_lgbm_raw":train_raw,"pd_lgbm_cal":train_cal})
    pred_o=pd.DataFrame({"case_id":oot.case_id,"WEEK_NUM":weeks_o,"split_name":oot.split_name,"target":yo,
        "scorecard_score":scoreo,"pd_scorecard":pso,"pd_lgbm_raw":poot_raw,"pd_lgbm_cal":poot_cal})
    pred=pd.concat([predtr,pred_o],ignore_index=True)
    con.register("stage3_pred",pred)
    con.execute(f"COPY stage3_pred TO '{(DATA/'stage3_predictions.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("stage3_pred")
    # TreeSHAP sample from blind OOT, 10k rows.
    rng=np.random.default_rng(42); si=rng.choice(len(oot),size=min(10000,len(oot)),replace=False)
    shap_x=matrix(oot.iloc[si],m4cols)
    expl=shap.TreeExplainer(m4model); sv=expl.shap_values(shap_x)
    if isinstance(sv,list): sv=sv[1]
    shap_plot_x=shap_x.copy()
    for c in category_levels:
        if c in shap_plot_x: shap_plot_x[c]=shap_plot_x[c].cat.codes.astype(np.float32)
    plt.figure(); shap.summary_plot(sv,shap_plot_x,feature_names=m4cols,plot_type="bar",max_display=15,show=False); plt.tight_layout(); plt.savefig(FIG/"shap_summary_bar.png",dpi=160,bbox_inches="tight"); plt.close()
    plt.figure(); shap.summary_plot(sv,shap_plot_x,feature_names=m4cols,max_display=15,show=False); plt.tight_layout(); plt.savefig(FIG/"shap_beeswarm.png",dpi=160,bbox_inches="tight"); plt.close()
    reg=pd.DataFrame({"Feature":kept,"coef":coef.values,"std_err":stderr.values,"z":zvals.values,"p_value":pvals.values,"VIF":finalvif.values})
    # Construct bilingual audit report.
    report=("# Stage 3 模型评估报告\n\n"
      "本阶段仅使用 `Train_Dev` 训练；LightGBM 使用 W0–54 拟合、W55–64 early stopping 与 isotonic 校准。`OOT_Valid` 仅在模型和校准器冻结后用于推理与盲评估；`OOT_Monitor` 未读取。训练特征矩阵以 float32 保存，DuckDB 内存限制 6GB、4 threads。\n\n"
      "## A-Card 事前纯度与 Stage 2 代理特征复核\n\n"
      "`disbursedcredamount_1113A` 已从 scorecard 和 challenger 排除，因为它属于放款后结果字段。Stage 2 中 `has_tax_registry_record` 与 `has_debitcard` 的经验原因：\n\n"
      + "\n".join(f"- `{k}`: {v}" for k,v in gate_reason.items()) + "\n\n"
      + "Train_Dev Gate 1 与内部 PSI 检查（W0–44 对 W45–64；仅训练期数据）：\n\n" + (md_table(auditdf) if len(auditdf) else "无可用字段") + "\n\n"
      + "按明确建模要求纳入四个 Group 4 proxy 作额外 challenger lift 测试。若 `PSI Override` 非空，该变量虽通过 Gate 1 但未通过内部 PSI < 0.10；其 Stage 3 结果仅作敏感性/探索性比较。\n\n"
      "## Table 1: Champion Scorecard 回归与计分卡点数\n\n"
      + md_table(reg) + "\n\n模型仅保留满足 `coef < 0`、`p_value < 0.05`、`VIF < 5` 的变量；保留数 " + str(len(kept)) + ".\n\n"
      "PDO scaling: Base Score=600 at bad/good odds=1/20; PDO=50. 最大 score→PD 重建误差=" + f"{recon_err:.3e}" + ".\n\n"
      "Sample points table:\n\n" + md_table(pd.DataFrame(points).head(12)) + "\n\n"
      "## Table 2: Champion 与 Challenger 指标\n\n" + md_table(mdf) + "\n\n"
      "说明：`Train_Dev (fit diagnostic)` 是同一训练样本的拟合诊断，不代表泛化指标。LightGBM calibration 指标另列 Dev_Calib；OOT_Valid 在整个训练、early stopping 和 calibration 过程中保持盲态。\n\n"
      "## Table 3: OOT_Valid 十分位 Lift 与 Capture\n\n" + md_table(dec) + "\n\n"
      "严格边际 bad-rate 递减检查（Decile 1 至 10）: `" + json.dumps(rankcheck.to_dict()) + "`。\n\n"
      "## Table 4: 4-Group Ablation Study\n\n" + md_table(abldf) + "\n\n"
      "## Monotonic constraints\n\n"
      "约束向量按各模型 feature 列顺序应用：DPD/delinquency/rejection rate/recent inquiry counts = +1；early payment ratios/employment years = -1；categorical/ambiguous ratios = 0。Categorical fields are passed to LightGBM as categorical features and use constraint 0；各组模型约束映射见下表。\n\n"
      + md_table(pd.DataFrame({"Feature":m4cols,"Group":[groups.get(c) for c in m4cols],"Constraint":m4cons,"Internal PSI":[psi_map.get(c,np.nan) for c in m4cols]})) + "\n\n"
      "## 审计摘要与工件\n\n"
      f"Train_Dev rows={len(train):,}; OOT_Valid rows={len(oot):,}; selected scorecard features={len(selected)}; retained={len(kept)}; M4 features={len(m4cols)}. Score reconstruction verified. Decile strict rank check: {rankcheck.to_dict()}.\n\n"
      "`data/processed/models/scorecard_logit.pkl`, `lgbm_m4.pkl`, `isotonic_calibrator.pkl`; `data/processed/scorecard_points_table.csv`; `data/processed/stage3_predictions.parquet`; `data/processed/woe_bin_rules.json`; `reports/figures/shap_summary_bar.png`; `reports/figures/shap_beeswarm.png`.\n\n"
      "**HARD STOP:** Stage 3 已完成；等待人工审阅后再开始 Stage 4.\n")
    (REPORT/"stage3_model_evaluation.md").write_text(report,encoding="utf-8")
    say("TABLE 2\n"+md_table(mdf)); say("TABLE 3\n"+md_table(dec)); say("TABLE 4\n"+md_table(abldf))
    say("Saved Stage 3 artifacts and report. HARD STOP for human review.")
    con.close()

if __name__=="__main__": main()
