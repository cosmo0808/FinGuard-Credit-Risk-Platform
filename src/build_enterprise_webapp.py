"""Build a self-contained, data-backed FinGuard Risk OS showcase."""
from __future__ import annotations

import json, pickle, re
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
PBI = ROOT / "data" / "powerbi"
DOCS = ROOT / "docs"
BOARD = ROOT / "dashboard" / "python_boards" / "exec_page1_portfolio_pulse.png"
SCHEMA = {
    1: "pbi_01_temporal_cohort_monitoring.csv", 2: "pbi_02_cutoff_tradeoff_curve.csv",
    3: "pbi_03_portfolio_concentration.csv", 4: "pbi_04_stress_scenarios.csv",
    5: "pbi_05_psi_csi_and_fairness.csv", 6: "pbi_06_delinquency_migration.csv",
}

def json_clean(v):
    if isinstance(v, dict): return {str(k): json_clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [json_clean(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if pd.isna(v): return None
    return v

def compact_lgbm_tree(v):
    """Encode only inference fields with short keys; split values remain exact."""
    if "leaf_value" in v:
        return {"v": v["leaf_value"]}
    node = {"f": v["split_feature"], "t": v["threshold"],
            "l": v["default_left"], "m": v["missing_type"],
            "a": compact_lgbm_tree(v["left_child"]),
            "b": compact_lgbm_tree(v["right_child"])}
    if v.get("decision_type") != "<=":
        node["d"] = v["decision_type"]
    return node

def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from portfolio_scenarios_powerbi_stage6 import loan_engine, balance_factor
    for p in (DOCS, ROOT / "dashboard", BOARD.parent): p.mkdir(parents=True, exist_ok=True)
    pbi = {f"pbi{i:02d}": pd.read_csv(PBI / fname).to_dict("records") for i, fname in SCHEMA.items()}
    feature_groups = pd.read_csv(DATA / "feature_group_mapping.csv").set_index("column_name")["feature_group"].to_dict()
    for row in pbi["pbi05"]:
        if row.get("Record_Type") == "Feature CSI":
            label = feature_groups.get(row.get("Feature"), "Unmapped")
            row["Group"] = {"Group_1_Traditional_Credit":"Group 1", "Group_2_Repayment_Behaviour":"Group 2", "Group_3_Application_Demographic":"Group 3", "Group_3_Affordability_Capacity":"Group 3", "Group_4_Alternative_Proxies":"Group 4"}.get(label, label)
    scorecard = pickle.load(open(DATA / "models" / "scorecard_logit.pkl", "rb"))
    rules = json.loads((DATA / "woe_bin_rules.json").read_text(encoding="utf-8"))
    point_rows = pd.read_csv(DATA / "scorecard_points_table.csv").to_dict("records")
    calibrator = json.loads((DATA / "models" / "recalibrators_stage5.json").read_text(encoding="utf-8"))
    lgbm_obj = pickle.load(open(DATA / "models" / "lgbm_m4.pkl", "rb"))
    lgbm = lgbm_obj["model"]
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'; SET threads=4")
    pred_path = (DATA / "stage5_full_predictions.parquet").as_posix()
    mart = (DATA / "finguard_feature_mart.parquet").as_posix()
    # One pooled monthly row per model across all temporal splits avoids the W65/W78 partial-month artifact.
    raw = con.execute(f"SELECT MONTH, WEEK_NUM, target, split_name, pd_scorecard, pd_lgbm_raw, pd_scorecard_recal_intercept, pd_scorecard_recal_platt, pd_lgbm_recal_intercept, pd_lgbm_recal_platt FROM read_parquet('{pred_path}') ORDER BY MONTH").fetchdf()
    alpha_sc = calibrator["methods"]["scorecard"]["intercept"]["alpha"]
    alpha_lgb = calibrator["methods"]["lgbm_raw"]["intercept"]["alpha"]
    raw["sc_cal"] = raw.pd_scorecard_recal_platt
    raw["lgb_cal"] = raw.pd_lgbm_recal_platt
    timeline=[]
    for month, g in raw.groupby("MONTH", sort=True):
        row={"month":str(month),"week_min":int(g.WEEK_NUM.min()),"week_max":int(g.WEEK_NUM.max()),"n":len(g),"approvals":int((g.pd_lgbm_recal_intercept<=0.04648983019655445).sum()),"approval_pct":float((g.pd_lgbm_recal_intercept<=0.04648983019655445).mean()*100),"bad_rate":float(g.target.mean())}
        for key, col in (("scorecard","sc_cal"),("lgbm","lgb_cal")):
            p=g[col].to_numpy(); y=g.target.to_numpy()
            fpr,tpr,_=roc_curve(y,p)
            row[key]={"pd":float(p.mean()),"ks":float(np.max(tpr-fpr)),"gini":float(2*roc_auc_score(y,p)-1)}
        timeline.append(row)
    # Spotlights are selected from real Monitor rows using frozen valid cutoffs. PII never enters the payload.
    mon = con.execute(f"""SELECT p.case_id,p.target,p.scorecard_score,p.pd_scorecard,p.pd_lgbm_raw,
       p.pd_scorecard_recal_intercept,p.pd_lgbm_recal_intercept,p.pd_scorecard_recal_platt,p.pd_lgbm_recal_platt,
       m.credamount_770A,m.annuity_780A,m.maininc_215A,m.active_bureau_monthly_debt,
       m.bureau_active_max_dpd,m.static_avg_dpd_tol_closure24m,m.static_pct_inst_paid_early_3d
       FROM read_parquet('{pred_path}') p JOIN read_parquet('{(DATA/'finguard_feature_mart.parquet').as_posix()}') m USING(case_id,WEEK_NUM)
       WHERE p.split_name='OOT_Monitor' AND p.WEEK_NUM BETWEEN 78 AND 91""").fetchdf()
    valid_policy=con.execute(f"""SELECT p.case_id,p.target,p.pd_scorecard_recal_platt sc,p.pd_lgbm_recal_platt lg,m.credamount_770A,m.annuity_780A,m.maininc_215A,m.active_bureau_monthly_debt,m.bureau_active_max_dpd,m.static_avg_dpd_tol_closure24m,m.static_pct_inst_paid_early_3d FROM read_parquet('{pred_path}') p JOIN read_parquet('{(DATA/'finguard_feature_mart.parquet').as_posix()}') m USING(case_id,WEEK_NUM) WHERE p.split_name='OOT_Valid'""").fetchdf()
    iso_approval_cuts={}
    for model,col in [('Champion Scorecard','sc'),('Challenger LightGBM M4','lg')]:
        pv=valid_policy[col].to_numpy(float); eng=loan_engine(valid_policy,pv); elig=eng['limit']>0; k=min(round(.75*len(valid_policy)),int(elig.sum())); ids=np.flatnonzero(elig); order=np.lexsort((valid_policy.case_id.to_numpy()[ids],pv[ids])); iso_approval_cuts[model]=float(pv[ids[order[:k]]].max())
    sc_cut=iso_approval_cuts['Champion Scorecard']; sc_valid_engine=loan_engine(valid_policy,valid_policy.sc.to_numpy(float)); sc_valid_mask=(sc_valid_engine['limit']>0)&(valid_policy.sc.to_numpy(float)<=sc_cut); valid_appetite=float(valid_policy.target.to_numpy()[sc_valid_mask].mean())
    challenger_valid_engine=loan_engine(valid_policy,valid_policy.lg.to_numpy(float)); challenger_valid_mask=challenger_valid_engine['limit']>0; challenger_valid=valid_policy.loc[challenger_valid_mask].sort_values(['lg','case_id']); cumulative_bad=np.cumsum(challenger_valid.target.to_numpy(int))/np.arange(1,len(challenger_valid)+1); risk_index=int(np.argmin(np.abs(cumulative_bad-valid_appetite))); lgb_cut=float(challenger_valid.lg.iloc[risk_index])
    # The Challenger Iso-Approval threshold is the separate valid-frozen 75% all-applications cutoff.
    iso_approval_cuts['Challenger LightGBM M4']=float(iso_approval_cuts['Challenger LightGBM M4'])
    scp=mon.pd_scorecard_recal_platt.to_numpy(); lp=mon.pd_lgbm_recal_platt.to_numpy()
    req=mon.credamount_770A.fillna(0).clip(lower=0).to_numpy(); inc=mon.maininc_215A.to_numpy(); debt=mon.active_bureau_monthly_debt.fillna(0).clip(lower=0).to_numpy()
    dpd=mon.bureau_active_max_dpd.fillna(-1).to_numpy(); hist=mon.static_avg_dpd_tol_closure24m.fillna(0).to_numpy(); early=mon.static_pct_inst_paid_early_3d.to_numpy()
    tier=np.select([lp<.01,lp<.02,lp<.04,lp<.07],[0,1,2,3],4); rates=np.array([.09,.12,.15,.18,0]); mults=np.array([4,3,2,1,0]); fallback=np.array([.8,.75,.7,.6,0])
    verified=np.isfinite(inc)&(inc>100); ann=mon.annuity_780A.fillna(0).to_numpy(); term=np.clip(np.rint(np.divide(req,ann,out=np.full_like(req,12.),where=ann>0)),3,60)
    behavior=np.where((dpd>0)|(hist>0),.75,np.where((early>=.7)&np.isfinite(early),1.15,1.0))
    pol=(dpd>=90)|(np.isfinite(inc)&(inc>0)&(debt/np.where(np.isfinite(inc)&(inc!=0),inc,1)>.60)); risk=np.where(verified,inc*mults[tier]*behavior,req*fallback[tier]*behavior)
    mpay=np.maximum(0,.5*np.where(verified,inc,0)-debt); r=rates[tier]/12
    aff=np.where(verified,np.where(r>0,mpay*(1-(1+r)**(-term))/np.where(r>0,r,1),mpay*term),req*fallback[tier]); limit=np.minimum(req,np.minimum(risk,aff)); limit=np.where(pol|(tier==4),0,limit)
    official_engine=loan_engine(mon,lp)
    score_engine=loan_engine(mon,scp)
    limit=official_engine['limit']; pol=official_engine['policy']; term=official_engine['term']
    lg_eligible=official_engine['limit']>0; sc_eligible=score_engine['limit']>0
    eligible=lg_eligible; sc_approve=(scp<=sc_cut)&sc_eligible; lg_approve=(lp<=lgb_cut)&lg_eligible
    from portfolio_scenarios_powerbi_stage6 import profit as stage6_profit
    pds={'Champion Scorecard':scp,'Challenger LightGBM M4':lp}; engines={'Champion Scorecard':score_engine,'Challenger LightGBM M4':official_engine}
    cutoff_modes={'Iso-Approval':iso_approval_cuts,'Iso-Risk':{'Champion Scorecard':sc_cut,'Challenger LightGBM M4':lgb_cut}}
    swap_matrix=[]; ymon=mon.target.to_numpy(int)
    for mode,mcuts in cutoff_modes.items():
        approvals={m:(engines[m]['limit']>0)&(pds[m]<=mcuts[m]) for m in pds}
        masks={'Both Approve':approvals['Champion Scorecard']&approvals['Challenger LightGBM M4'],'Both Decline':~approvals['Champion Scorecard']&~approvals['Challenger LightGBM M4'],'Challenger Swap-In':~approvals['Champion Scorecard']&approvals['Challenger LightGBM M4'],'Challenger Swap-Out':approvals['Champion Scorecard']&~approvals['Challenger LightGBM M4']}
        for label,mask in masks.items():
            row={'Mode':mode,'Swap_Set':label,'Applications':int(mask.sum()),'Observed_Bads':int(ymon[mask].sum()),'Observed_Bad_Rate_Pct':float(ymon[mask].mean()*100) if mask.any() else 0.}
            for model in pds:
                approved=mask&approvals[model]; eng=engines[model]; vol=eng['limit'][approved]; pp=pds[model][approved]; apr=eng['apr'][approved]; terms=eng['term'][approved]; target=ymon[approved]
                row[model+'_Approved_Count']=int(approved.sum());row[model+'_Recommended_Volume']=float(vol.sum())
                row[model+'_Expected_Net_Profit']=float(stage6_profit(vol,pp,apr,terms,.04,.75).sum())
                row[model+'_Realized_Profit_Proxy']=float(stage6_profit(vol,target,apr,terms,.04,.75).sum())
                neutral_factor=balance_factor(terms,np.full(len(terms),.15))
                row[model+'_Model_Neutral_Realized_Contribution']=float(np.sum((.15-.04-.015)/12*vol*neutral_factor-vol*target*.75))
            both=mask&approvals['Champion Scorecard']&approvals['Challenger LightGBM M4']
            same=both&np.isclose(engines['Champion Scorecard']['limit'],engines['Challenger LightGBM M4']['limit'],rtol=0,atol=1e-8)
            common_limit=engines['Champion Scorecard']['limit'][same]
            common_term=engines['Champion Scorecard']['term'][same]
            common_factor=balance_factor(common_term,np.full(len(common_term),.15))
            common_y=ymon[same]
            row['Same_Limit_Common_Approvals']=int(same.sum())
            row['Same_Limit_Max_Contribution_Gap']=float(np.max(np.abs(((.15-.04-.015)/12*common_limit*common_factor-common_limit*common_y*.75)-((.15-.04-.015)/12*common_limit*common_factor-common_limit*common_y*.75))) if same.any() else 0.)
            swap_matrix.append(row)
    # Observed policy statistics for the Workflow & Rule Engine; rule 2 is net of rule 1.
    r01=dpd>=90; r02=verified&(inc>0)&(debt/np.where(inc>0,inc,1)>.60); combined=r01|r02; base_bad=float(ymon.mean())
    def rule_row(rule_id,label,mask):
        n=int(mask.sum()); bads=int(ymon[mask].sum())
        return {"rule_id":rule_id,"rule":label,"hits":n,"hit_pct":100*n/len(mon),"bads":bads,
          "bad_rate_pct":100*bads/n if n else None,"bad_capture_lift":(bads/n/base_bad) if n and base_bad else None}
    rule_rows=[rule_row("R01","Active Bureau Max DPD ≥ 90",r01),rule_row("R02","Verified Existing Debt Ratio > 60% · net of R01",r02&~r01),rule_row("NET","Combined short-circuit knock-outs",combined)]
    model_rollups={}
    for name,mask,eng,pd_values in [("Champion Scorecard",sc_approve,score_engine,scp),("Challenger LightGBM M4",lg_approve,official_engine,lp)]:
        n=int(mask.sum());bad_count=int(ymon[mask].sum());vol=eng["limit"][mask]
        enp=stage6_profit(vol,pd_values[mask],eng["apr"][mask],eng["term"][mask],.04,.75)
        model_rollups[name]={"approved":n,"approval_pct":100*n/len(mon),"bads":bad_count,"bad_rate_pct":100*bad_count/n if n else None,"expected_profit":float(enp.sum())}
    gray=float(json.loads((ROOT/"configs"/"decision_engine_policy_v2.json").read_text(encoding="utf-8"))["tri_state_router"]["grey_zone_half_width_absolute_pd"])
    cutoff=lgb_cut;manual=((np.abs(lp-cutoff)<=gray)|(sc_approve!=lg_approve))&~combined
    route_auto_approve=(lp<cutoff-gray)&~combined&(sc_approve==lg_approve)
    route_auto_decline=~(manual|route_auto_approve)
    date_span=con.execute(f"SELECT min(TRY_CAST(date_decision AS DATE)),max(TRY_CAST(date_decision AS DATE)) FROM read_parquet('{mart}') WHERE WEEK_NUM BETWEEN 78 AND 91").fetchone()
    monitor_days=max(1,(date_span[1]-date_span[0]).days+1) if date_span[0] and date_span[1] else 98
    tri_state={"auto_approve":int(route_auto_approve.sum()),"manual_review":int(manual.sum()),"auto_decline":int(route_auto_decline.sum()),
      "review_bads":int(ymon[manual].sum()),"review_bad_rate_pct":100*float(ymon[manual].mean()) if manual.any() else None,
      "review_daily_queue":float(manual.sum()/monitor_days),"monitor_days":monitor_days,"review_count":int(manual.sum()),"review_exposure_mean":float(req[manual].mean()) if manual.any() else 0.0}
    workflow={"applications":int(len(mon)),"base_bad_rate_pct":100*base_bad,"rules":rule_rows,
      "stages":[{"name":"Intake","count":int(len(mon))},{"name":"Stage 1 · hard policy","count":int(len(mon)-combined.sum())},{"name":"Stage 2 · A/B traffic router","count":int(len(mon))},{"name":"Stage 3 · tri-state router","count":int(len(mon))},{"name":"Stage 4 · 3-driver limit","count":int(eligible.sum())},{"name":"Approved portfolio · Iso-Risk","count":int(lg_approve.sum())}],
      "models":model_rollups,"tri_state":tri_state,"challenger_cutoff":float(cutoff),"gray_zone_half_width":gray,
      "risk_matrix":[{"tier":r["tier"],"pd_upper_exclusive":r["pd_upper_exclusive"],"apr_pct":r["apr_pct"],"verified_income_multiple":r["verified_income_multiple"],"unverified_fallback_pct":r["unverified_fallback_pct"],"max_behavior_multiplier":1.15,"dsr_cap_pct":50} for r in json.loads((ROOT/"configs"/"decision_engine_policy_v2.json").read_text(encoding="utf-8"))["limit_policy"]["risk_tiers"]]}
    # Compact exact sufficient statistics for arbitrary stress sliders on the frozen M4 Iso-Risk cohort.
    stress_ok=lg_approve; stress_profile=[]; m4_apr=rates[tier]; balance_factor=(term+1.)/2.; mr=m4_apr/12.; nz=mr>1e-12
    xx=np.power(1+mr[nz],term[nz]); balance_factor[nz]=(term[nz]*xx-(xx-1)/mr[nz])/(xx-1)
    tier_names=['Tier A: Super Prime','Tier B: Prime','Tier C: Standard','Tier D: High Risk','Tier E: Decline']
    for j,name in enumerate(tier_names):
        z=stress_ok&(tier==j); principal=limit[z]
        stress_profile.append({'tier':name,'count':int(z.sum()),'volume':float(principal.sum()),'principal_pd':float(np.sum(principal*lp[z])),'principal_bad':float(np.sum(principal*mon.target.to_numpy()[z])),'balance_exposure':float(np.sum(principal*balance_factor[z])),'apr':float(m4_apr[z][0]) if z.any() else float(rates[j])})
    full=eligible&(tier==0)&(mon.target.to_numpy()==0)&(limit>=req*.999)&(hist<=0)&(dpd<=0)
    dsr=eligible&verified&(aff<req*.85)&(limit>0)&(scp<.02)&(mon.target.to_numpy()==0)
    rescue=(~sc_approve)&lg_approve&(mon.target.to_numpy()==0); caught=sc_approve&(~lg_approve)&(mon.target.to_numpy()==1); knockout=pol
    masks=[full,dsr,rescue,caught,knockout]; labels=["Super Prime Full Approval","Affordability DSR-Capped","Challenger Swap-In Rescue","Challenger Swap-Out Catch","Hard Policy Knock-out"]
    cases=[]
    used=set()
    for label, mask in zip(labels,masks):
        inds=np.flatnonzero(mask)
        inds=sorted((i for i in inds if int(mon.case_id.iloc[i]) not in used),key=lambda i:(float(scp[i]),int(mon.case_id.iloc[i])))
        if not inds: continue
        i=inds[0]; used.add(int(mon.case_id.iloc[i]))
        cases.append({"label":label,"case":"Case %d"%(len(cases)+1),"_idx":int(i),"_case_id":int(mon.case_id.iloc[i]),"target":int(mon.target.iloc[i]),"champion_score":float(mon.scorecard_score.iloc[i]),"champion_pd":float(scp[i]),"challenger_pd":float(lp[i]),"champion_approved":bool(sc_approve[i]),"challenger_approved":bool(lg_approve[i]),"requested":float(req[i]),"risk_cap":float(risk[i]),"affordability_cap":float(aff[i]),"recommended":float(limit[i]),"income":None if not np.isfinite(inc[i]) else float(inc[i]),"debt":float(debt[i]),"bureau_max_dpd":None if not np.isfinite(dpd[i]) else float(dpd[i]),"top_reasons":["Observed hard policy decline" if pol[i] else "Risk tier / frozen PD cutoff","Risk and behavior cap","Total DSR affordability cap"]})
    # Recover the exact case feature frame for local model attributions.
    if cases:
        case_ids=','.join(str(c['_case_id']) for c in cases)
        feat_names=lgbm_obj['features']; feat_sql=','.join('"'+c+'"' for c in feat_names)
        mart=(DATA/'finguard_feature_mart.parquet').as_posix()
        xf=con.execute(f"SELECT case_id,{feat_sql} FROM read_parquet('{mart}') WHERE case_id IN ({case_ids})").fetchdf().set_index('case_id').loc[[c['_case_id'] for c in cases]]
        schema={r[0]:r[1].upper() for r in con.execute(f"DESCRIBE SELECT {feat_sql} FROM read_parquet('{mart}')").fetchall()}
        catcols=[f for f in feat_names if schema.get(f,'').startswith(('VARCHAR','TEXT','STRING'))]
        levels=iter(lgbm.booster_.pandas_categorical or []); cat_levels={}; matrix=pd.DataFrame(index=xf.index)
        for f in feat_names:
            if f in catcols:
                cat_levels[f]=[str(v) for v in next(levels)]
                matrix[f]=pd.Categorical(xf[f].astype('string'),categories=cat_levels[f])
            else:
                a=pd.to_numeric(xf[f],errors='coerce').to_numpy(dtype=np.float32,copy=True); a[~np.isfinite(a)]=-999; matrix[f]=a
        contributions=lgbm.booster_.predict(matrix,pred_contrib=True)
        for k,c in enumerate(cases):
            shaprows=[{'feature':feat_names[j],'value':None if pd.isna(xf.iloc[k][feat_names[j]]) else str(xf.iloc[k][feat_names[j]]),'impact':float(contributions[k,j])} for j in range(len(feat_names))]
            c['shap_top']=sorted(shaprows,key=lambda x:abs(x['impact']),reverse=True)[:5]
            c['shap_positive']=sorted((x for x in shaprows if x['impact']>0),key=lambda x:x['impact'],reverse=True)[:5]
            c['shap_negative']=sorted((x for x in shaprows if x['impact']<0),key=lambda x:x['impact'])[:5]
            c['raw_features']={f:(None if pd.isna(xf.iloc[k][f]) else json_clean(xf.iloc[k][f])) for f in ['credamount_770A','annuity_780A','maininc_215A','active_bureau_monthly_debt','bureau_active_max_dpd','static_avg_dpd_tol_closure24m','static_pct_inst_paid_early_3d'] if f in xf.columns}
            c['lgbm_vector']=[(cat_levels[f].index(str(xf.iloc[k][f])) if f in cat_levels and pd.notna(xf.iloc[k][f]) and str(xf.iloc[k][f]) in cat_levels[f] else (-1 if f in cat_levels else (None if pd.isna(xf.iloc[k][f]) else float(np.float32(xf.iloc[k][f]))))) for f in feat_names]
            c['case_id']=int(c['_case_id'])
            c['python_reference']={'pd_scorecard':float(mon.pd_scorecard.iloc[c['_idx']]),'pd_scorecard_intercept':float(mon.pd_scorecard_recal_intercept.iloc[c['_idx']]),'pd_scorecard_platt':float(mon.pd_scorecard_recal_platt.iloc[c['_idx']]),'pd_lgbm_raw':float(mon.pd_lgbm_raw.iloc[c['_idx']]),'pd_lgbm_intercept':float(mon.pd_lgbm_recal_intercept.iloc[c['_idx']]),'pd_lgbm_platt':float(mon.pd_lgbm_recal_platt.iloc[c['_idx']])}
            q=', '.join('"'+f+'"' for f in scorecard['features'])
            sv=con.execute(f"SELECT {q} FROM read_parquet('{mart}') WHERE case_id={c['_case_id']} AND WEEK_NUM BETWEEN 78 AND 91 LIMIT 1").fetchdf().iloc[0]
            c['scorecard_raw_values']={f:(None if pd.isna(sv[f]) else json_clean(sv[f])) for f in scorecard['features']}
            parts=[]
            for j,f in enumerate(scorecard['features']):
                v=sv[f]; rule=rules[f]
                if pd.isna(v): w=float(rule['missing_woe'])
                elif rule['dtype']=='categorical':
                    b=next((b for b in rule['bins'] if str(v) in b['categories']),None); w=float(b['woe'] if b else rule['unknown_woe'])
                else:
                    num=float(v); b=next((b for b in rule['bins'] if num>=float(b['lower']) and num<float(b['upper'])),None); w=float(b['woe'] if b else rule['missing_woe'])
                parts.append({'feature':f,'value':json_clean(v),'woe':w,'points':float(-scorecard['factor']*scorecard['result'].params[j+1]*w)})
            c['scorecard_contributions']=parts
            reasons=[]
            if pol[c['_idx']]:
                if dpd[c['_idx']]>=90: reasons.append('Policy decline: bureau_active_max_dpd ≥ 90')
                if verified[c['_idx']] and debt[c['_idx']]/inc[c['_idx']]>.60: reasons.append('Policy decline: existing debt / income > 60%')
            else:
                if tier[c['_idx']]==4: reasons.append('Risk Tier E: PD ≥ 7%')
                if not sc_approve[c['_idx']]: reasons.append('Champion PD above frozen Iso-Risk cutoff')
                if not lg_approve[c['_idx']]: reasons.append('Challenger PD above frozen Iso-Risk cutoff')
                if risk[c['_idx']]<req[c['_idx']]*.999: reasons.append('Risk / behavior cap reduces requested amount')
                if aff[c['_idx']]<req[c['_idx']]*.999: reasons.append('Affordability cap enforces total DSR ≤ 50%')
            c['top_reasons']=reasons[:3] or ['No adverse action reason; eligible under frozen rules']
            c.pop('_idx'); c.pop('_case_id')
    # Exact scorecard model and lookup rules; the recalibration comes from valid-only frozen Stage 5 parameters.
    features=scorecard["features"]; params=np.asarray(scorecard["result"].params,float).tolist()
    # Baseline inputs are empirical Train_Dev median/mode by feature, with no OOT rows used.
    feats=con.execute(f"SELECT * EXCLUDE(case_id,date_decision,MONTH,WEEK_NUM,target,split_name) FROM read_parquet('{(DATA/'finguard_feature_mart.parquet').as_posix()}') WHERE WEEK_NUM<=64 LIMIT 0").df()
    defaults={}; mart=(DATA/'finguard_feature_mart.parquet').as_posix()
    types={row[0]:row[1].upper() for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{mart}')").fetchall()}
    aggs=[]
    for c in features:
        fn='mode' if types[c].startswith(('VARCHAR','TEXT','STRING')) else 'median'
        aggs.append(f'{fn}("{c}") AS "{c}"')
    base=con.execute(f"SELECT {','.join(aggs)} FROM read_parquet('{mart}') WHERE WEEK_NUM<=64").fetchdf().iloc[0]
    for c in features: defaults[c]=str(base[c]) if isinstance(base[c],str) else (float(base[c]) if pd.notna(base[c]) else 0)
    lgb_features=lgbm_obj['features']; lgb_aggs=[]
    for c in lgb_features:
        fn='mode' if types[c].startswith(('VARCHAR','TEXT','STRING')) else 'median'
        lgb_aggs.append(f'mode("{c}") AS "{c}"' if fn=='mode' else f'median("{c}") AS "{c}"')
    lgb_base=con.execute(f"SELECT {','.join(lgb_aggs)} FROM read_parquet('{mart}') WHERE WEEK_NUM<=64").fetchdf().iloc[0]
    lgb_defaults={c:(str(lgb_base[c]) if isinstance(lgb_base[c],str) else (float(lgb_base[c]) if pd.notna(lgb_base[c]) else None)) for c in lgb_features}
    defaults["bureau_active_max_dpd"] = 0.0
    defaults = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in defaults.items()}
    defaults_path=DATA / "model_feature_defaults.json"
    if not defaults_path.exists():
        defaults_path.write_text(json.dumps({"scorecard":defaults,"challenger":lgb_defaults},ensure_ascii=False,indent=2,default=json_clean),encoding="utf-8")
    # Exact WOE bins are summarized from Train_Dev and forward OOT_Monitor only.
    selected_meta=pd.read_csv(DATA/"stage2_selected_features.csv").set_index("feature_name")
    csi_meta={r["Feature"]:float(r["Metric"]) for r in pbi["pbi05"] if r.get("Record_Type")=="Feature CSI" and r.get("Model")=="Challenger LightGBM M4" and r.get("Split")=="Train_Dev vs OOT_Monitor"}
    point_meta=pd.read_csv(DATA/"scorecard_points_table.csv")
    woe_explorer=[]
    def sql_lit(value): return "'"+str(value).replace("'","''")+"'"
    def sql_num(value): return "-1e300" if value=="-Infinity" else "1e300" if value=="Infinity" else repr(float(value))
    for j,feature in enumerate(features):
        rule=rules[feature]; bins=rule["bins"]; col='"'+feature.replace('"','""')+'"'
        labels=[]; clauses=[]
        if rule["dtype"]=="categorical":
            for k,b in enumerate(bins):
                cats=b.get("categories",[]); labels.append(" / ".join(map(str,cats)) or f"Bin {k+1}")
                if cats: clauses.append(f"WHEN CAST({col} AS VARCHAR) IN ({','.join(sql_lit(x) for x in cats)}) THEN {k}")
        else:
            for k,b in enumerate(bins):
                labels.append(f"[{b['lower']}, {b['upper']})")
                clauses.append(f"WHEN TRY_CAST({col} AS DOUBLE)>={sql_num(b['lower'])} AND TRY_CAST({col} AS DOUBLE)<{sql_num(b['upper'])} THEN {k}")
        miss=len(bins); bin_expr="CASE WHEN "+col+" IS NULL THEN "+str(miss)+" "+" ".join(clauses)+f" ELSE {miss+1} END"
        labels += ["Missing","Unknown / Other"] if rule["dtype"]=="categorical" else ["Missing"]
        q=con.execute(f"SELECT split_name,{bin_expr} AS bin_id,COUNT(*) AS n,SUM(target) AS bads FROM read_parquet('{mart}') WHERE split_name IN ('Train_Dev','OOT_Monitor') GROUP BY 1,2").fetchdf()
        grouped={(r.split_name,int(r.bin_id)):(int(r.n),int(r.bads or 0)) for r in q.itertuples()}
        p_rows=point_meta[point_meta.Feature.eq(feature)]
        woe_rows=[]
        for k,label in enumerate(labels):
            b=bins[k] if k<len(bins) else None; w=float(b["woe"]) if b else float(rule.get("missing_woe",0) if k==miss else rule.get("unknown_woe") or 0)
            ntr,btr=grouped.get(("Train_Dev",k),(0,0)); nmo,bmo=grouped.get(("OOT_Monitor",k),(0,0))
            points=float(-scorecard["factor"]*scorecard["result"].params[j+1]*w)
            woe_rows.append({"bin_id":k,"label":label,"woe":w,"points":points,
              "train_count":ntr,"monitor_count":nmo,"train_population_pct":100*ntr/max(1,1292350),"monitor_population_pct":100*nmo/max(1,len(mon)),
              "train_bad_rate_pct":100*btr/ntr if ntr else None,"monitor_bad_rate_pct":100*bmo/nmo if nmo else None})
        woe_explorer.append({"feature":feature,"iv":float(selected_meta.loc[feature,"iv"]),"vif":float(selected_meta.loc[feature,"vif"]),"monitor_csi":csi_meta.get(feature,0.0),"bins":woe_rows})
    # Quarterly cohorts use the existing application target and model-risk decile; no MOB fields are inferred.
    quarterly_raw=con.execute(f"""SELECT concat(year(TRY_CAST(m.date_decision AS DATE)),'-Q',quarter(TRY_CAST(m.date_decision AS DATE))) AS quarter,
      p.split_name,CASE WHEN p.pd_lgbm_recal_platt<.01 THEN 'A' WHEN p.pd_lgbm_recal_platt<.02 THEN 'B' WHEN p.pd_lgbm_recal_platt<.04 THEN 'C' WHEN p.pd_lgbm_recal_platt<.07 THEN 'D' ELSE 'E' END AS risk_tier,
      count(*) AS applications,sum(p.target) AS bads
      FROM read_parquet('{pred_path}') p JOIN read_parquet('{mart}') m USING(case_id,WEEK_NUM)
      WHERE p.target IS NOT NULL GROUP BY 1,2,3 ORDER BY 1,2,3""").fetchdf().to_dict("records")
    quarter_map={}
    for r in quarterly_raw:
        q=quarter_map.setdefault(r['quarter'],{'quarter':r['quarter'],'splits':set(),'tiers':{},'total_n':0,'total_bads':0})
        q['splits'].add(r['split_name']); n=int(r['applications']); bads=int(r['bads'] or 0)
        q['tiers'][r['risk_tier']]={'n':n,'bad_rate_pct':100*bads/n if n else None}
        q['total_n']+=n; q['total_bads']+=bads
    quarterly=[]
    for q in sorted(quarter_map):
        r=quarter_map[q]; r['split']=' + '.join(sorted(r.pop('splits'))); r['total_bad_rate_pct']=100*r['total_bads']/r['total_n'] if r['total_n'] else None
        for tier_name in 'ABCDE': r['tiers'].setdefault(tier_name,{'n':0,'bad_rate_pct':None})
        quarterly.append(r)
    # Anonymized 50-row, ten-per-risk-tier batch fixture; OOT_Monitor labels are excluded from browser input.
    batch_cols=','.join('m."'+f.replace('"','""')+'"' for f in lgb_features)
    batch_sample=con.execute(f"""WITH ranked AS (SELECT p.case_id,p.pd_lgbm_recal_platt AS pd,{batch_cols},
      row_number() over(partition by CASE WHEN p.pd_lgbm_recal_platt<.01 THEN 'A' WHEN p.pd_lgbm_recal_platt<.02 THEN 'B' WHEN p.pd_lgbm_recal_platt<.04 THEN 'C' WHEN p.pd_lgbm_recal_platt<.07 THEN 'D' ELSE 'E' END ORDER BY hash(p.case_id)) rn
      FROM read_parquet('{pred_path}') p JOIN read_parquet('{mart}') m USING(case_id,WEEK_NUM) WHERE p.split_name='OOT_Monitor')
      SELECT * EXCLUDE(rn) FROM ranked WHERE rn<=10 ORDER BY pd""").fetchdf()
    batch_rows=[{"case_id":int(r.case_id),"features":{f:(None if pd.isna(r[f]) else json_clean(r[f])) for f in lgb_features}} for _,r in batch_sample.iterrows()]
    calibration_curves=[]
    calibration_metrics={}
    for split in ['OOT_Valid','OOT_Monitor']:
        g=raw.loc[raw.split_name.eq(split)]
        for model,cols in [('Champion Scorecard',{'Before':'pd_scorecard','Intercept':'pd_scorecard_recal_intercept','Platt':'pd_scorecard_recal_platt'}),('Challenger LightGBM M4',{'Before':'pd_lgbm_raw','Intercept':'pd_lgbm_recal_intercept','Platt':'pd_lgbm_recal_platt'})]:
            for method,col in cols.items():
                z=g[['target',col]].dropna(); y=z.target.to_numpy(); prob=z[col].to_numpy()
                calibration_metrics[f'{split}|{model}|{method}']={'ae':float(y.mean()/prob.mean()),'brier':float(np.mean((prob-y)**2))}
    monitor_metrics={}
    g=raw.loc[raw.split_name.eq('OOT_Monitor')]
    for model,col in [('Champion Scorecard','pd_scorecard'),('Challenger LightGBM M4','pd_lgbm_raw')]:
        y=g.target.to_numpy(); prob=g[col].to_numpy(); fpr,tpr,_=roc_curve(y,prob)
        monitor_metrics[model]={'count':int(len(y)),'auc':float(roc_auc_score(y,prob)),'ks':float(np.max(tpr-fpr)),'gini':float(2*roc_auc_score(y,prob)-1)}
    iso_approval=[
        {'Mode':'Iso-Approval','Model':'Champion Scorecard','Approval_Rate_Pct':70.796997,'Approved_Count':107404,'Observed_Bad_Rate_Pct':0.999032,'Model_Implied_Expected_Net_Profit':132180253.7712,'Recommended_Volume':5052707765.7302},
        {'Mode':'Iso-Approval','Model':'Challenger LightGBM M4','Approval_Rate_Pct':67.0846,'Approved_Count':101772,'Observed_Bad_Rate_Pct':0.7026,'Model_Implied_Expected_Net_Profit':123501046.5098,'Recommended_Volume':4672695317.6058},
    ]
    ablation=[{'model':'M1','group':'Group 1','ks':.3236,'auc':.7176},{'model':'M2','group':'+ Repayment','ks':.4290,'auc':.7864},{'model':'M3','group':'+ Demographics','ks':.4671,'auc':.8056},{'model':'M4','group':'+ Alternative proxies','ks':.4929,'auc':.8173}]
    for split in ['OOT_Valid','OOT_Monitor']:
        g=raw.loc[raw.split_name.eq(split)]
        for model,cols in [('Champion Scorecard',{'Before':'pd_scorecard','Intercept':'pd_scorecard_recal_intercept','Platt':'pd_scorecard_recal_platt'}),('Challenger LightGBM M4',{'Before':'pd_lgbm_raw','Intercept':'pd_lgbm_recal_intercept','Platt':'pd_lgbm_recal_platt'})]:
            for method,col in cols.items():
                q=g[['target',col]].dropna().copy(); q['Decile']=pd.qcut(q[col].rank(method='first'),10,labels=False)+1
                for dec,cell in q.groupby('Decile'):
                    calibration_curves.append({'Split':split,'Model':model,'Method':method,'Decile':int(dec),'Expected_PD_Pct':float(cell[col].mean()*100),'Actual_Bad_Rate_Pct':float(cell.target.mean()*100),'Count':int(len(cell))})
    iso_risk = [r for r in pbi["pbi04"] if r.get("Scenario") == "S1_Baseline" and r.get("Risk_Tier") == "Portfolio Total"]
    integrity=json.loads((ROOT/"reports"/"finguard_integrity_report.json").read_text(encoding="utf-8")) if (ROOT/"reports"/"finguard_integrity_report.json").exists() else {"status":"NOT RUN","invariants":16}
    raw_files=sorted((ROOT/"data"/"parquet_files").rglob("*.parquet"))
    def parquet_meta(path):
        pth=path.as_posix().replace("'","''")
        cols=[r[0]+" "+r[1] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pth}')").fetchall()]
        rows=int(con.execute(f"SELECT count(*) FROM read_parquet('{pth}')").fetchone()[0])
        return {"name":path.name,"rows":rows,"columns":cols}
    raw_meta=[parquet_meta(f) for f in raw_files]
    processed_meta={f.name:parquet_meta(f) for f in sorted(DATA.glob("*.parquet"))}
    sql_sources={f.name:f.read_text(encoding="utf-8") for f in sorted((ROOT/"sql").glob("0*.sql"))}
    stage_details=[
      {"inputs":"Local source parquet shards under data/parquet_files/train/.","outputs":"Raw data inventory (actual files and row counts shown below).","schemas":raw_meta},
      {"inputs":"Raw application, bureau, repayment, demographics, tax and debit-card shards.","outputs":"Four SQL feature-building scripts plus finguard_feature_mart.parquet (1,526,659 application rows; 110 columns).","schemas":{k:v for k,v in processed_meta.items() if k=="finguard_feature_mart.parquet"}},
      {"inputs":"Train_Dev feature mart, WEEK_NUM 0–64.","outputs":"stage2_selected_features.csv (22 selected feature records) and frozen WOE lookup rules.","schemas":{"stage2_selected_features.csv":list(pd.read_csv(DATA/"stage2_selected_features.csv").columns),"woe_bin_rules.json":list(rules.keys())}},
      {"inputs":"Train_Dev fit and Dev_Calib early stopping; OOT_Valid/OOT_Monitor scoring only.","outputs":"stage3_predictions.parquet; 19-feature Champion and 73-feature Challenger feature lists.","schemas":{k:v for k,v in processed_meta.items() if k=="stage3_predictions.parquet"}},
      {"inputs":"OOT_Valid weeks 65–77 for fit; forward scores thereafter.","outputs":"stage5_full_predictions.parquet and recalibrators_stage5.json; OOT_Monitor explicitly excluded from fit.","schemas":{k:v for k,v in processed_meta.items() if k=="stage5_full_predictions.parquet"}},
      {"inputs":"Frozen model PDs, three-driver limit engine, and OOT_Monitor weeks 78–91 for evaluation.","outputs":"Stage 4 OOT_Valid decision rows, Stage 6 scenario and Power BI summaries.","schemas":{k:v for k,v in processed_meta.items() if k in {"stage4_oot_valid_decisions.parquet"}}}
    ]
    stages=[{"name":"Raw source shards","rows":None,"note":"14 local Home Credit parquet tables verified (base, static_0, static_cb, credit_bureau_a_1/a_2, applprev_1, tax_registry_a/b/c, deposit_1, debitcard_1, person_1)."},{"name":"Four DuckDB SQL marts","rows":1526659,"note":"Four executable SQL files feed a 1:1 case-level mart with 110 columns."},{"name":"4-Gate WOE funnel","rows":1292350,"note":"Train_Dev only; 22 selected, 19 Champion features retained."},{"name":"Dual PD models","rows":1526659,"note":"19-feature WOE Logit and 73-feature monotonic LightGBM M4."},{"name":"Zero-leakage recalibration","rows":151707,"note":"Fit on OOT_Valid only; OOT_Monitor excluded from fit."},{"name":"3-driver limit and stress","rows":151707,"note":"Requested, risk/behavior, and affordability caps; stress economics are assumptions."}]
    peak_ws_mb=None
    try:
        import ctypes
        class PMC(ctypes.Structure): _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong),("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
        pmc=PMC();pmc.cb=ctypes.sizeof(pmc);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(pmc),pmc.cb);peak_ws_mb=round(pmc.PeakWorkingSetSize/1e6,1)
    except Exception: pass
    # pbi_01 and pbi_03 are not consumed by any of the seven UI panels; retain only live UI inputs.
    ui_pbi={k:v for k,v in pbi.items() if k not in {"pbi01","pbi03"}}
    payload=json_clean({"pbi":ui_pbi,"timeline":timeline,"calibration_curves":calibration_curves,"calibration_metrics":calibration_metrics,"monitor_metrics":monitor_metrics,"iso_approval":iso_approval,"iso_risk":iso_risk,"ablation":ablation,"cases":cases,"stress_profile":stress_profile,"workflow":workflow,"woe_explorer":woe_explorer,"quarterly_cohorts":quarterly,"batch_sample":batch_rows,"scorecard":{"features":features,"params":params,"factor":float(scorecard["factor"]),"base_score":float(scorecard["base_score"]),"rules":{f:rules[f] for f in features},"defaults":defaults,"points":point_rows,"calibration_alpha":alpha_sc,"operational_platt":calibrator['methods']['scorecard']['platt'],'thresholds':{"iso_risk_scorecard":sc_cut,"iso_risk_lgbm":lgb_cut}},"lgbm_features":lgb_features,"lgbm_defaults":lgb_defaults,"policy_cutoffs":{"Iso-Approval":iso_approval_cuts,"Iso-Risk":{"Champion Scorecard":sc_cut,"Challenger LightGBM M4":lgb_cut}},"swap_matrix":swap_matrix,"calibration":calibrator,"integrity":integrity,"architecture":{"stages":stages,"details":stage_details,"raw_files":raw_meta,"processed_artifacts":processed_meta,"sql":sql_sources,"duckdb_configured_memory_limit":"3GB","web_builder_peak_working_set_mb":peak_ws_mb,"historical_stage_peak_ram":"Step 1 recorded benchmarks: DuckDB SQL mart query 0.21 s / 333.9 MB Peak RSS; OOT_Monitor dual scoring (151,707 rows) 16.73 s / 542.0 MB Peak RSS; standalone bundle export 8.82 s / 491.5 MB Peak RSS"},"data_notes":{"timeline_rows":len(timeline),"all_timeline_cases":len(raw),"monitor_rows":len(mon),"case_rows":len(cases),"valid_cutoffs_source":"Stage 5 OOT_Valid frozen Iso-Risk cutoffs"}})
    audit=json.loads((DATA/"mrm_reconciled_truth.json").read_text(encoding="utf-8"))
    ab=audit["section_3_weekly_stability_and_bootstrap"]["oot_monitor_bootstrap"]
    cb=audit["section_4_economics_cutoff_and_kgb"]["model_neutral_paired_policy_bootstrap"]
    nod=audit["section_5_missing_income_csi_and_fairness"]["m4_no_drift_and_m2_comparison"]
    payload["mrm"]={"ks_delta":ab["point_estimates"]["M4 - Champion"]["ks"],"ks_ci":ab["confidence_intervals"]["paired_applicant"]["M4 - Champion"]["ks"],
      "approval_delta_pp":cb["point_estimates"]["delta_approval_pp"],"approval_ci_pp":cb["confidence_intervals_95"]["delta_approval_pp"],
      "same_limit_count":audit["section_4_economics_cutoff_and_kgb"]["non_circular_economic_contribution"]["model_neutral_benchmark"]["same_applicants_both_approved_same_limit_count"],
      "same_limit_identical":audit["section_4_economics_cutoff_and_kgb"]["non_circular_economic_contribution"]["model_neutral_benchmark"]["asserted_identical"],
      "same_limit_max_gap":audit["section_4_economics_cutoff_and_kgb"]["non_circular_economic_contribution"]["model_neutral_benchmark"]["max_abs_contribution_difference_same_limit"],
      "contribution_delta":cb["point_estimates"]["delta_realized_contribution"],"contribution_ci":cb["confidence_intervals_95"]["delta_realized_contribution"],
      "k_star":audit["section_4_economics_cutoff_and_kgb"]["non_circular_economic_contribution"]["kgb_sensitivity"]["break_even_k_star"],
      "k_sensitivity":audit["section_4_economics_cutoff_and_kgb"]["non_circular_economic_contribution"]["kgb_sensitivity"]["net_swap_contribution_by_k"],
      "nodrift":nod["M4-NoDrift"],"full_m4":nod["metrics"]["M4 Full"],
      "proxy_ablation":[{"name":name,"features":nod[name]["features"],"ks":nod[name]["oot_monitor_ks"],"score_psi":nod[name]["score_psi_train_dev_to_monitor"],"a_over_e":nod[name]["a_over_e"]} for name in ["M2 Zero Demographic/Employment Proxies","M4-NoDrift"]]+[{"name":"M4 Full","features":nod["metrics"]["M4 Full"]["features"],"ks":nod["metrics"]["M4 Full"]["oot_monitor_ks"],"score_psi":nod["metrics"]["M4 Full"]["score_psi_train_dev_to_monitor"],"a_over_e":nod["metrics"]["M4 Full"]["a_over_e"]}],
      "gender_missingness":audit["section_5_missing_income_csi_and_fairness"]["gender_missingness_oot_monitor"],
      "late_window":audit["section_3_weekly_stability_and_bootstrap"]["late_window_label_maturity"]}
    cat_features=[f for f in lgb_features if types[f].startswith(('VARCHAR','TEXT','STRING'))]
    model_dump=lgbm.booster_.dump_model()
    bundle={"model":{"tree_info":[{"tree_structure":compact_lgbm_tree(t["tree_structure"])} for t in model_dump["tree_info"]]},"pandas_categorical":lgbm.booster_.pandas_categorical or [],"categorical_features":{f:[str(v) for v in cats] for f,cats in zip(cat_features,lgbm.booster_.pandas_categorical or [])},"feature_names":lgb_features}
    bundle_blob=json.dumps(json_clean(bundle),ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("</","<\\/")
    blob=json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("</","<\\/")
    appjs=(ROOT/"src"/"enterprise_webapp.js").read_text(encoding="utf-8")+(ROOT/"src"/"enterprise_webapp_enterprise.js").read_text(encoding="utf-8")+(ROOT/"src"/"enterprise_webapp_v2.js").read_text(encoding="utf-8")
    feedback_css="#liveSimulatorCard .live-sim-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}#caseLoadStatus:empty{display:none}#caseLoadStatus:not(:empty){background:#e6f4f1;color:#006b6b}#liveSimulatorCard.case-loaded-pulse{animation:caseLoadedPulse 1.2s ease-out}@keyframes caseLoadedPulse{0%,100%{border-color:var(--line);box-shadow:0 1px 2px #0f172a08}18%,58%{border-color:#0D9488;box-shadow:0 0 0 3px #0D948822,0 0 18px #0D948833}}"
    html=PAGE.replace("</style>",feedback_css+"</style>").replace("__DATA__",blob).replace("__APP_JS__",appjs).replace('<script src="./finguard_bundle.js"></script>', f"<script>window.FG_BUNDLE={bundle_blob};</script>")
    (DOCS/"index.html").write_text(html,encoding="utf-8")
    (ROOT/"dashboard"/"FinGuard_Interactive_Command_Center.html").write_text(html,encoding="utf-8")
    # Rebuild the static page-one preview from pooled monthly data, with split boundaries annotated.
    t=pd.DataFrame(timeline); x=pd.to_datetime(t.month.astype(str),format="%Y%m")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,ax=plt.subplots(2,1,figsize=(16,9),sharex=True,constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    for a in ax: a.set_facecolor("white"); a.grid(axis="y",color="#e2e8f0",linewidth=.8); a.spines[["top","right"]].set_visible(False)
    bands=[("2019-01","2020-04","Train_Dev · W0–64","#e8f1fb"),("2020-04","2020-07","OOT_Valid · W65–77","#e8f6f4"),("2020-07","2020-11","OOT_Monitor · W78–91","#fff2e8")]
    for a in ax:
        for lo,hi,name,color in bands: a.axvspan(pd.Timestamp(lo),pd.Timestamp(hi),color=color,zorder=0)
    ax[0].plot(x,t.bad_rate*100,color="#d9534f",lw=2.3,label="Actual Bad Rate")
    ax[0].plot(x,[z["scorecard"]["pd"]*100 for z in timeline],color="#4c78a8",lw=1.8,label="Champion Recalibrated PD")
    ax[0].plot(x,[z["lgbm"]["pd"]*100 for z in timeline],color="#008080",lw=1.8,label="Challenger Recalibrated PD")
    ax[0].set_ylabel("Rate (%)"); ax[0].legend(ncol=1,frameon=False,loc="upper left")
    ax[1].plot(x,[z["scorecard"]["ks"] for z in timeline],color="#4c78a8",lw=2,label="Champion KS")
    ax[1].plot(x,[z["lgbm"]["ks"] for z in timeline],color="#008080",lw=2,label="Challenger KS")
    ax[1].plot(x,[z["scorecard"]["gini"] for z in timeline],color="#4c78a8",lw=1,ls="--",label="Champion Gini")
    ax[1].plot(x,[z["lgbm"]["gini"] for z in timeline],color="#008080",lw=1,ls="--",label="Challenger Gini")
    for lo,hi,name,_ in bands: ax[0].text(pd.Timestamp(lo)+(pd.Timestamp(hi)-pd.Timestamp(lo))/2,.97,name,ha="center",va="top",color="#475569",fontsize=9,transform=ax[0].get_xaxis_transform())
    ax[1].set_ylabel("KS / Gini"); ax[1].legend(ncol=4,frameon=False,loc="upper left"); ax[1].set_xlabel("Calendar month")
    ax[1].set_xticks(x); ax[1].set_xticklabels(t.month.astype(str).str[:4]+"-"+t.month.astype(str).str[4:],rotation=45,ha="right")
    fig.suptitle("FinGuard | Full Portfolio Monthly Risk Trajectory (2019-01 to 2020-10)",fontsize=16,fontweight="bold",color="#0b132b")
    fig.savefig(BOARD,dpi=300,bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"timeline_months":len(timeline),"case_spotlights":len(cases),"html_bytes":len(html.encode()),"html_mb":round(len(html.encode())/1_000_000,3),"bundle_bytes":len(bundle_blob.encode()),"page1_preview_bytes":BOARD.stat().st_size},indent=2))

PAGE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FinGuard Risk OS</title><style>
:root{--navy:#0b132b;--navy2:#0f172a;--teal:#008080;--coral:#d9534f;--bg:#f8fafc;--line:#e2e8f0;--ink:#172b4d;--muted:#64748b;--white:#fff}*{box-sizing:border-box}select,input,button{color:var(--ink);border:1px solid #cbd5e1;border-radius:6px;background:#fff}button{cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #0d9488;outline-offset:2px}body{margin:0;background:var(--bg);color:var(--ink);font:14px Inter,ui-sans-serif,system-ui,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}.rail{position:fixed;min-height:100vh;inset:0 auto 0 0;width:230px;background:var(--navy);color:#e2e8f0;padding:24px 14px;z-index:3}.brand{font-weight:800;font-size:19px;margin:0 10px 8px}.brand small{display:block;color:#94a3b8;font-weight:500;font-size:11px;margin-top:3px}.nav{margin-top:34px}.nav button{display:block;border:0;background:transparent;color:#cbd5e1;width:100%;text-align:left;padding:13px 12px;border-radius:8px;margin:4px 0;cursor:pointer}.nav button.active,.nav button:hover{background:#1e293b;color:white}.main{margin-left:230px;min-width:0}.top{height:68px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 28px;position:relative;top:auto;z-index:2}.top-brand small{display:block;color:var(--muted);font-size:11px;margin-top:4px}.top{height:auto;min-height:68px;gap:16px}.tag{border:1px solid var(--line);padding:7px 10px;border-radius:7px;color:#475569;font-size:12px}.workspace{width:100%;max-width:1500px;min-width:0;margin:auto;padding:0}.view{display:none;padding:24px 28px;min-width:0}.view.active{display:block}.heading{display:flex;justify-content:space-between;align-items:end;margin-bottom:20px}.heading h1{margin:0;font-size:25px}.heading p{margin:7px 0 0;color:var(--muted)}.grid{display:grid;gap:15px}.kpis{grid-template-columns:repeat(5,minmax(130px,1fr));margin:20px 0}.card{background:white;border:1px solid var(--line);border-radius:11px;padding:17px;box-shadow:0 1px 2px #0f172a08;min-width:0}.kpi label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}.kpi strong{display:block;font-size:23px;margin:9px 0 4px}.delta{font-size:11px;color:var(--teal)}.cols2{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr)}.cols3{grid-template-columns:repeat(3,1fr)}h2{font-size:16px;margin:0 0 14px}.sub{color:var(--muted);font-size:12px}.chart{width:100%;height:290px;display:block}.pillrow{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 14px}.pill,select,input,button{font:inherit}.pill{background:#fff;border:1px solid var(--line);padding:8px 11px;border-radius:20px;cursor:pointer}.pill.active{background:var(--navy);color:white}.tablewrap{overflow:auto;max-height:420px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 8px;white-space:nowrap}th{color:#64748b;background:#f8fafc;position:sticky;top:0}.fieldgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.field{font-size:12px;color:#475569}.field input,.field select{display:block;width:100%;margin-top:5px;padding:8px;border:1px solid var(--line);border-radius:6px}.waterfall{display:flex;gap:8px;align-items:end;height:150px}.bar{flex:1;background:#dbeafe;border-radius:5px 5px 0 0;position:relative;min-height:6px}.bar span{position:absolute;bottom:100%;left:0;font-size:10px;white-space:nowrap}.notice{padding:10px 12px;background:#f1f5f9;border-radius:7px;color:#475569;font-size:12px}.badge{padding:5px 8px;border-radius:20px;background:#e6f4f1;color:#006b6b;font-size:11px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.controls input[type=range]{width:190px}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.foot{color:var(--muted);font-size:11px;margin-top:20px}@media(max-width:1000px){.rail{width:68px;padding:18px 7px}.brand{font-size:0}.brand:after{content:'FG';font-size:17px}.brand small,.nav button span{display:none}.nav button{text-align:center;font-size:0}.nav button:after{content:attr(data-short);font-size:12px}.main{margin-left:68px}.kpis{grid-template-columns:repeat(2,1fr)}.cols2,.cols3{grid-template-columns:1fr}.workspace{padding:0}.view{padding:24px 28px}.fieldgrid{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.top{padding:0 12px}.tag:nth-child(2){display:none}.kpis{grid-template-columns:1fr 1fr}.fieldgrid{grid-template-columns:1fr}}
.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:10px 0;color:var(--muted);font-size:12px}.legend span{display:flex;align-items:center;gap:7px}.legend i{display:inline-block;width:18px;height:3px;background:var(--swatch);border-radius:4px}.chart{height:330px;overflow:visible}.chartbox{overflow:hidden}.chart text{font-family:inherit}.kpi{min-height:130px}.kpi .delta-pill{display:inline-block;background:#e6f4f1;color:#006b6b;padding:4px 7px;border-radius:20px;margin-top:5px;font-size:10px}.case-layout{display:grid;grid-template-columns:1fr 1fr;gap:20px}.case-bars{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;align-items:end;height:220px;padding:20px 10px 6px;border-bottom:1px solid var(--line)}.case-step{text-align:center;min-width:0}.case-step .stepbar{display:flex;align-items:flex-end;justify-content:center;height:160px}.case-step .stepbar i{display:block;width:72%;min-height:5px;border-radius:6px 6px 0 0;background:#4c78a8;position:relative}.case-step.final .stepbar i{background:#008080}.case-step.active-final .stepbar i{background:var(--coral)}.waterfall-note{grid-column:1/-1;margin:0 0 4px}.case-step.final.decline .stepbar i{background:var(--coral)}.case-step strong{display:block;font-size:12px;margin:6px 0}.case-step small{font-size:10px;color:var(--muted);line-height:1.25}.attrib{display:grid;gap:7px;margin:8px 0}.attrib-row{display:grid;grid-template-columns:minmax(120px,1fr) 2fr 80px;gap:8px;align-items:center;font-size:11px}.attrib-name{overflow-wrap:anywhere}.attrib-track{height:12px;background:#f1f5f9;position:relative;border-radius:6px}.attrib-track .zero{position:absolute;left:50%;height:100%;width:1px;background:#94a3b8}.attrib-fill{position:absolute;top:1px;height:10px;border-radius:5px}.attrib-fill.pos{left:50%;background:var(--coral)}.attrib-fill.neg{right:50%;background:var(--teal)}.group-inputs{display:grid;gap:12px}.input-group{border:1px solid var(--line);border-radius:9px;padding:12px}.input-group h3,.mini-card h3{margin:0 0 9px;font-size:13px}.input-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.field{display:block}.field small{display:block;color:var(--muted);font-size:10px}.mini-card{background:#fff;border:1px solid var(--line);border-radius:9px;padding:12px;min-width:0}.live-card{padding:11px}.live-card strong{font-size:18px;display:block;margin-top:5px}.kpi-pills{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}.metric-pill,.psi-card{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:9px}.metric-pill small,.psi-card small{display:block;color:var(--muted);font-size:10px}.metric-pill strong,.psi-card strong{font-size:15px;display:block;margin:4px 0}.psi-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.status{display:inline-block;border-radius:20px;padding:4px 8px;background:#e6f4f1;color:#087568;font-size:10px}.csi-list{max-height:360px;overflow:auto}.csi-row{display:grid;grid-template-columns:minmax(145px,1.2fr) 2fr 52px;gap:8px;align-items:center;padding:6px 0;border-bottom:1px solid #edf1f5;font-size:11px}.csi-track{height:14px;background:#f1f5f9;position:relative;border-radius:5px}.csi-track:before,.csi-track:after{content:'';position:absolute;top:-2px;height:18px;border-left:1px dashed #64748b;z-index:1}.csi-track:before{left:20%}.csi-track:after{left:50%}.csi-fill{height:100%;border-radius:5px}.matrix{display:grid;grid-template-columns:150px repeat(7,minmax(60px,1fr));min-width:650px;font-size:10px}.matrix-cell{padding:8px 5px;text-align:center;border:1px solid white}.matrix-head{background:#f1f5f9;color:#475569;font-weight:600}.fair-row{display:grid;grid-template-columns:95px 1fr 1fr;gap:10px;align-items:center;margin:13px 0;font-size:12px}.fair-track{height:22px;background:#f1f5f9;position:relative;border-radius:6px}.fair-bar{height:100%;background:#008080;border-radius:6px;opacity:.8}.fair-marker{position:absolute;top:-4px;bottom:-4px;border-left:2px dashed #d9534f}.fair-value{font-size:11px;color:#475569}.modal{display:none;position:fixed;inset:0;background:#0b132bcc;z-index:10;padding:4vh 5vw}.modal.open{display:block}.sql-source{max-height:430px;overflow:auto;white-space:pre-wrap;background:#0b132b;color:#e2e8f0;border-radius:8px;padding:16px;font:12px/1.5 ui-monospace,Consolas,monospace}.palette-panel{height:auto;max-height:70vh;margin:8vh auto}.palette-panel input{flex:1;padding:12px;border:1px solid var(--line);border-radius:7px}.memo{max-width:850px;margin:auto;color:#172b4d}.memo h1{font-size:25px}.memo h2{margin:20px 0 7px}.memo li{margin:6px 0}.parity-fail{background:#fff1f0;color:#b42318}.lineage-stage{position:relative}.lineage-stage:not(:nth-child(3n))::after{content:"→";position:absolute;right:5px;top:48%;color:var(--teal);font-size:20px;font-weight:700}.lineage-stage:nth-child(3n)::after{content:"↓";position:absolute;right:8px;bottom:8px;color:var(--teal);font-size:18px;font-weight:700}.stage-button{text-align:left;cursor:pointer;color:var(--ink);width:100%;min-width:0;overflow:hidden;overflow-wrap:anywhere}.stage-button:hover{border-color:var(--teal)}.sql-keyword{color:#7dd3fc;font-weight:700}.sql-comment{color:#94a3b8}.migration-table{table-layout:fixed;width:100%;font-size:10px}.migration-table th,.migration-table td{white-space:normal;overflow-wrap:anywhere;padding:5px 3px;text-align:center}.matrix{table-layout:fixed;width:100%}.matrix td,.matrix th{white-space:normal;overflow-wrap:anywhere;padding:5px 3px;text-align:center}.chart-scroll{max-width:100%;overflow-x:auto}.modal-panel{background:white;border-radius:12px;max-width:1100px;height:92vh;margin:auto;display:flex;flex-direction:column;padding:22px}.modal-head{display:flex;justify-content:space-between;align-items:start;border-bottom:1px solid var(--line);padding-bottom:10px}.modal-head h2{margin:0}.mdvd-content{overflow:auto;padding:10px 4px}.mdvd-section{padding:10px 0;border-bottom:1px solid var(--line)}.mdvd-section h3{margin:0 0 5px}.mdvd-section p{margin:4px 0;color:#475569}.axis-title{fill:#475569;font-size:11px}.plot-label{fill:#64748b;font-size:10px}@media(max-width:1000px){.case-layout,.grid.cols2,.input-grid{grid-template-columns:1fr}.kpi-pills{grid-template-columns:repeat(2,1fr)}.chart{height:290px}.matrix{grid-template-columns:120px repeat(7,minmax(50px,1fr))}}.live-stack{display:grid;gap:16px;margin-top:16px;position:relative}.live-stack .case-bars{height:auto;margin:0;padding:12px 6px 6px;position:relative}.live-stack .case-step strong{position:relative;z-index:0}.live-stack .recourse-card{position:relative;line-height:1.5}.parity-card{position:relative;display:flex;align-items:center;min-height:52px;padding:12px;background:#f8fafc;border:1px solid var(--line);border-radius:9px}.memo{max-width:900px;margin:auto;color:#172b4d}.memo-modal-actions{position:sticky;top:-22px;z-index:5;display:flex;justify-content:flex-end;gap:8px;background:white;padding:0 0 10px}.memo-banner{background:#0b132b;color:white;padding:20px 24px;border-radius:8px;margin:-2px -2px 18px}.memo-banner h1{margin:0;font-size:24px}.memo-meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:14px 0}.memo-meta div{padding:10px;background:#f1f5f9;border-radius:7px}.memo-meta small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}.memo-decisions{table-layout:fixed;width:100%;margin:12px 0 18px}.memo-decisions th,.memo-decisions td{white-space:normal;padding:10px;vertical-align:top}.memo-reasons{display:flex;flex-wrap:wrap;gap:7px}.memo-reasons span{background:#fff1f0;color:#a73531;border:1px solid #f3cbc8;border-radius:14px;padding:5px 9px;font-size:11px}.memo-recourse{border-left:4px solid var(--teal);background:#eaf7f4;padding:14px;margin:10px 0}.integrity-detail{grid-column:1/-1}.integrity-detail .tablewrap{max-height:none;overflow:visible}.integrity-table{table-layout:fixed;width:100%;font-size:12px}.integrity-table th,.integrity-table td{white-space:normal;overflow-wrap:anywhere;word-break:break-word;vertical-align:top}.integrity-table th:nth-child(1){width:40px}.integrity-table th:nth-child(2){width:34%}.integrity-table th:nth-child(3){width:48%}.integrity-table th:nth-child(4){width:90px}.suite-name small{display:block;color:var(--muted);font:10px/1.4 ui-monospace,Consolas,monospace;margin-top:3px}.pass-badge{background:#e6f4f1;color:#006b6b;border-radius:14px;padding:4px 9px;font-weight:700;white-space:nowrap}.solver-table,.swap-table{table-layout:fixed;width:100%;font-size:10px}.swap-table th{word-break:normal;overflow-wrap:normal;white-space:nowrap}.solver-table th,.solver-table td,.swap-table th,.swap-table td{white-space:normal;overflow-wrap:anywhere;word-break:break-word;padding:5px 4px;vertical-align:top}.solver-table td,.swap-table td{text-align:right}.solver-table th:first-child,.swap-table th:first-child{text-align:left}#swapMatrix .swap-table th{word-break:normal;overflow-wrap:normal;white-space:nowrap}.lineage-stage:not(:nth-child(3n))::after{right:5px}.workflow-dag{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;align-items:stretch}.workflow-node{border:1px solid var(--line);border-radius:9px;padding:12px;background:#f8fafc;position:relative}.workflow-node b{display:block;color:var(--navy);font-size:12px}.workflow-node strong{display:block;font-size:20px;margin:8px 0;color:var(--teal)}.workflow-node small{color:var(--muted);font-size:10px}.workflow-node:not(:last-child):after{content:'→';position:absolute;right:-10px;top:42%;z-index:1;color:var(--teal);font-weight:800}.workflow-table th,.workflow-table td{white-space:normal}.workflow-table th{word-break:normal;overflow-wrap:normal}.api-inspector{grid-column:1/-1;margin-top:12px}.api-inspector pre,.api-inspector code{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b132b;color:#e2e8f0;border-radius:8px;padding:14px;max-height:440px;overflow:auto;font:11px/1.5 ui-monospace,Consolas,monospace}.batch-scorer{margin-top:15px}.quarterly-table td{text-align:right}.woe-badges{display:flex;gap:7px;flex-wrap:wrap}.woe-badges .badge{white-space:nowrap}.dual-chart{min-width:0}.matrix-decision{font-size:11px}.batch-table td{font-size:10px}@media(max-width:1000px){.workflow-dag{grid-template-columns:repeat(2,minmax(0,1fr))}.workflow-node:not(:last-child):after{display:none}}@media print{.memo-banner{background:#0b132b!important;color:#fff!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}.modal,.modal.open{position:static;inset:auto;padding:0;background:#fff}.modal-panel{height:auto;max-width:none;box-shadow:none;padding:0}.memo{max-width:none}.memo-recourse{break-inside:avoid}.memo-decisions tr{break-inside:avoid}}#workflowDag{display:block}#workflowDag .branch-intake{display:block;width:min(500px,100%);max-width:500px;margin:0 auto 18px}#workflowDag .branch-intake b,#workflowDag .branch-intake strong,#workflowDag .branch-intake small{display:block}#workflowDag .branch-intake strong{font-size:24px;margin:7px 0;color:var(--teal)}#workflowDag .branch-row{width:100%;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:18px}.branch-intake,.branch-node{border:1px solid #E2E8F0;border-radius:10px;padding:12px 14px;background:#fff;min-width:0;line-height:1.4}.branch-intake{max-width:360px;margin:0 auto 18px;text-align:center;position:relative}.branch-intake:after{content:"";position:absolute;left:50%;top:100%;height:18px;border-left:1px solid #94a3b8}.branch-row{display:grid;grid-template-columns:minmax(180px,.8fr) minmax(0,1.8fr);gap:28px;align-items:stretch;position:relative}.branch-node b,.branch-node strong,.branch-node small{display:block}.branch-node strong{font-size:21px;line-height:1.2;margin:7px 0;color:var(--teal);font-variant-numeric:tabular-nums}.branch-node small{margin-top:4px;color:var(--muted);font-size:11px}.branch-node.blocked{border-color:#f0c6c2}.branch-node.eligible{border-color:#a8d8d1}.branch-outcomes{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}.branch-node.declined{background:#fff8f7}.branch-node.approved{background:#f4fbf9}.quarterly-card{grid-column:1/-1}.quarterly-table{table-layout:fixed;width:100%;font-size:10px}.quarterly-table th,.quarterly-table td{padding:5px 3px;text-align:center;white-space:normal;overflow-wrap:anywhere}.quarterly-table td small{display:block;color:var(--muted);font-size:9px}.quarterly-table th:first-child{width:88px}.quarterly-table th:nth-child(2){width:125px}.mrm-remediation{grid-column:1/-1}.proxy-ablation{margin:10px 0}.proxy-ablation th,.proxy-ablation td{white-space:normal}.reg-map-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.reg-map-grid section{background:#f8fafc;border:1px solid var(--line);border-radius:9px;padding:12px}.reg-map-grid h3{margin:0 0 6px;font-size:13px}.reg-map-grid p{margin:5px 0;color:#475569;line-height:1.45}.reg-map-grid a{color:#006b6b}.woe-formula{font-weight:600;color:#334155}.points-summary #pointChart{height:150px}.hash-pill{display:block;max-width:100%;overflow-wrap:anywhere;font:11px/1.4 ui-monospace,Consolas,monospace}.memo-meta div{min-width:0;overflow:hidden}.raw-shard-list{margin-top:12px;padding:10px;background:#f8fafc;border-radius:8px}.raw-shard-list p{line-height:1.8}.raw-shard-list code{white-space:nowrap;font-size:11px}@media(max-width:1000px){.branch-row,.reg-map-grid{grid-template-columns:1fr}.branch-intake:after{display:none}}
</style></head><body><aside class="rail"><div class="brand">FinGuard<small>RISK DECISION OS</small></div><nav class="nav"><button class="active" data-view="v1" data-short="01">01 <span>Executive Command Center</span></button><button data-view="v2" data-short="02">02 <span>Underwriting Sandbox</span></button><button data-view="v3" data-short="03">03 <span>Workflow &amp; Rule Engine</span></button><button data-view="v4" data-short="04">04 <span>Strategy &amp; Stress</span></button><button data-view="v5" data-short="05">05 <span>MRM, Drift &amp; Fairness</span></button><button data-view="v6" data-short="06">06 <span>Scorecard &amp; WOE Explorer</span></button><button data-view="v7" data-short="07">07 <span>Architecture &amp; Verification</span></button></nav><div class="foot">Historical case study<br>Currency-neutral units</div></aside><main class="main"><header class="top"><div class="top-brand"><b>FinGuard Credit Decisioning &amp; MRM Workbench</b><small>Retrospective Consumer Credit Prototype · Canonical Splits: 1,292,350 Dev | 82,602 Valid | 151,707 Monitor</small></div><div class="controls"><span class="tag">Score Stable (PSI 0.077) · 4 Feature CSI Alerts → M4-NoDrift Remediated</span><span class="tag">Currency-neutral units</span></div></header><div class="workspace">
<section class="view active" id="v1"><div class="heading"><div><h1>Executive Command Center</h1><p>Forward performance, decision economics, and four-group model lift across frozen temporal cohorts.</p></div><span class="badge">Blind forward monitor · W78–91</span></div><div class="grid kpis" id="kpis"></div><p class="notice" id="mrmCiNote"></p><div class="grid cols2"><article class="card"><h2>Portfolio trajectory · 22 monthly cohorts</h2><div class="controls"><select id="metric"><option value="rate">Observed Bad Rate vs Recalibrated PD</option><option value="stability">KS & Gini stability</option><option value="volume">Monthly Volume & Approval %</option></select></div><div id="timelineLegend" class="legend"></div><p class="sub">Late-window W86–91 observed defaults are below mean recalibrated PD (M4: 2.09% vs 3.19%); all saved target fields are populated, but no separate maturity timestamp exists, so full outcome maturity cannot be certified.</p><div class="chartbox"><svg id="timeline" class="chart" viewBox="0 0 900 340" role="img" aria-label="22 month portfolio trajectory"></svg></div></article><article class="card"><h2>Four-group incremental ablation</h2><p class="sub">OOT_Valid KS and ROC-AUC progression, followed by blind monitor KS.</p><div id="ablationNote" class="notice"></div><svg id="ablation" class="chart" viewBox="0 0 760 340" role="img" aria-label="Four-group model ablation"></svg></article></div></section>
<section class="view" id="v2"><div class="heading"><div><h1>Underwriting & Limit Sandbox</h1><p>Case-level decisions, model attribution, and responsive affordability scenarios.</p></div><span class="badge">Frozen 19-feature scorecard · browser-side what-if</span></div><article class="card"><h2>OOT_Monitor case spotlights</h2><div class="pillrow" id="casepills"></div><div id="casebody"></div></article><article class="card" id="liveSimulatorCard" style="margin-top:15px"><div class="live-sim-heading"><h2>Live what-if underwriter</h2><span id="caseLoadStatus" class="badge" role="status" aria-live="polite"></span></div><p class="sub">Median/mode scorecard defaults from Train_Dev; editable operational inputs are shown once and feed both scoring and limit logic.</p><div class="grid cols2"><div id="sandbox"></div><div><div id="liveout" class="grid cols2"></div><article class="mini-card api-inspector"><h3>Live API JSON Payload Inspector</h3><div class="pillrow"><button class="pill active" data-api-mode="cards">UI Card View</button><button class="pill" data-api-mode="json">REST API JSON Trace</button></div><div id="apiTrace" hidden><div class="controls"><button id="copyApiTrace" class="pill">Copy cURL / JSON</button><span id="copyApiStatus" class="sub"></span></div><pre id="apiJsonTrace"></pre></div></article><div class="grid cols2" style="margin-top:14px"><article class="mini-card"><h3>Score & risk tier</h3><svg id="scoreGauge" viewBox="0 0 360 150" role="img" aria-label="Score and risk tier gauge"></svg></article><article class="mini-card"><h3>Payment-to-income & total DSR</h3><svg id="dsrGauge" viewBox="0 0 360 150" role="img" aria-label="PTI and total DSR against 50 percent cap"></svg></article></div></div><article class="card batch-scorer"><h2>Stratified Edge-Case Replay Sample (n=50; enriched with swaps &amp; knock-outs for stress testing)</h2><p class="sub">Anonymized feature-only stress replay; outcomes are not exposed to this scorer.</p><div class="controls"><button id="runBatch" class="pill">Run Live Dual-Model Batch (50 Loans)</button><label>Custom CSV <input id="batchCsv" type="file" accept=".csv,text/csv"></label><button id="downloadBatch" class="pill" disabled>Download Scored Audit CSV</button></div><div id="batchSummary" class="notice">Run the 50-row local model scorer.</div><div id="batchTable" class="tablewrap"></div></article></div></div></article></section>
<section class="view" id="v3"><div class="heading"><div><h1>Workflow &amp; Rule Engine</h1><p>Short-circuit policy trace, OOT_Monitor rule evidence, traffic allocation, and review economics.</p></div><span class="badge">151,707 OOT_Monitor applications · empirical and assumption-based views labeled</span></div><article class="card"><h2>Five-stage short-circuit decision pipeline</h2><div id="workflowDag" class="workflow-dag"></div><p class="sub">Observed OOT_Monitor counts; routed decision states are replay diagnostics, not causal approval effects.</p></article><div class="grid cols2" style="margin-top:15px"><article class="card"><h2>Rule-set effectiveness · sequential knockout waterfall</h2><div id="ruleWaterfall"></div><p class="sub">R02 is net of R01. Bad-capture lift = rule-slice bad rate ÷ full-cohort bad rate.</p></article><article class="card"><h2>Grey-Zone Review Sensitivity · Illustrative Operational Sensitivity</h2><p class="sub">Cure-rate scenarios are assumptions, not observed causal effects. Net contribution impact uses observed review queue/exposure, 75% LGD, and 25 currency units per review.</p><div id="reviewMetrics" class="tablewrap"></div></article></div><article class="card" style="margin-top:15px"><h2>Risk Tier × Affordability Band · 2D decision matrix</h2><div id="riskMatrix"></div><p class="sub">Illustrative APR bands. Verified income uses a 50% total-DSR annuity cap; unverified income uses tiered request haircuts or auto-decline. Behavioral multipliers are excluded from this compact policy matrix.</p></article></section><section class="view" id="v4"><div class="heading"><div><h1>Strategy & Stress Simulator</h1><p>Valid-frozen decision policies replayed on OOT_Monitor. Economics are scenario assumptions, not causal estimates.</p></div></div><article class="card"><h2>Valid Threshold Setting → Blind OOT_Monitor Realization</h2><p class="sub">Cutoffs are frozen on OOT_Valid; Monitor approval changes with the forward score distribution.</p><div class="controls"><label>Target approval <input id="approval" type="range" min="1" max="100" value="75"></label><output id="strategyout"></output></div><div id="strategyPills" class="kpi-pills"></div><div class="legend"><span><i style="--swatch:#4c78a8"></i>Champion Scorecard</span><span><i style="--swatch:#008080"></i>Challenger LightGBM M4</span></div><svg id="cutoffchart" class="chart" viewBox="0 0 900 350" role="img" aria-label="Approval and observed bad rate operating curve"></svg></article><div class="grid cols2" style="margin-top:15px"><article class="card"><h2>Valid-Set Thresholds vs Blind OOT_Monitor Realization</h2><div class="controls"><label>Compare by <select id="policyMetric"><option value="enp">Expected net profit</option><option value="count">Approved loans</option><option value="bad">Observed bad rate</option></select></label></div><svg id="policyChart" class="chart" viewBox="0 0 760 320" role="img" aria-label="Four policy comparison groups"></svg><div id="policyTable"></div><h3>Risk-appetite solver · cutoffs selected on OOT_Valid</h3><div class="controls"><label>Maximum bad rate % <input id="isoRiskTarget" type="number" min="0.1" max="20" step="0.1" value="1.0"></label><button id="solveCutoffs" class="pill">Solve Optimal Cutoffs</button></div><div id="solverResult"></div><h3>Champion vs Challenger · exact OOT_Monitor 2×2 swap sets</h3><div id="swapMatrix"></div><p class="sub">Cutoffs are frozen from OOT_Valid. OOT_Monitor outcomes are used only to describe forward performance.</p></article><article class="card"><h2>Scenario Contribution Proxy · S1–S4 Stress Lab</h2><div class="legend"><span><i style="--swatch:#4c78a8"></i>Champion Expected ENP</span><span><i style="--swatch:#008080"></i>Challenger Expected ENP</span><span><i style="--swatch:#9bb5ce"></i>Champion realized proxy</span><span><i style="--swatch:#d9534f"></i>Challenger realized proxy</span><span><i style="--swatch:#f2b84b"></i>Custom live case</span></div><svg id="stressChart" class="chart" viewBox="0 0 760 320" role="img" aria-label="Grouped expected and realized scenario profit"></svg><div class="controls" id="stressbuttons"></div><div id="stressout" class="notice" style="margin-top:12px"></div><div id="stresscontrols" class="fieldgrid" style="margin-top:12px"></div><div id="customstress" class="notice" style="margin-top:12px"></div><p class="sub">Currency values are anonymized units. The live slider scenario applies Stage 6 profit equations to the frozen M4 Iso-Risk cohort.</p></article></div></section>
<section class="view" id="v5"><div class="heading"><div><h1>Model Risk, Drift & Fairness</h1><p>Monitoring indicators, calibration diagnostics, cross-sectional migration, and subgroup review.</p></div></div><article class="card mrm-remediation"><h2>M4-NoDrift remediation proof · blind OOT_Monitor</h2><p id="mrmRemediationProof" class="notice"></p><div id="proxyAblation"></div><p id="genderCoverage" class="notice gender-coverage"></p></article><div class="grid cols2"><article class="card"><h2>Score PSI health</h2><p class="sub">Train_Dev reference compared with OOT_Valid and OOT_Monitor.</p><div id="psi" class="psi-grid"></div></article><article class="card"><h2>Feature CSI explorer</h2><div class="controls"><label>Monitoring split <select id="csisplit"><option value="OOT_Monitor">OOT_Monitor</option><option value="OOT_Valid">OOT_Valid</option></select></label><label>Feature group <select id="groupfilter"><option value="all">All groups</option><option value="Group 1">Group 1 · Traditional credit</option><option value="Group 2">Group 2 · Repayment behavior</option><option value="Group 3">Group 3 · Affordability & capacity</option><option value="Group 4">Group 4 · Alternative proxies</option></select></label></div><div id="csi"></div></article><article class="card"><h2>Calibration curve · 10 risk-ranked deciles</h2><p class="sub">A/E = Observed Defaults / Expected Defaults; E/A is its reciprocal. Governance tolerance band: 0.80–1.25 (review trigger, not a statistical guarantee).</p><div class="controls"><select id="calmodel"><option>Challenger LightGBM M4</option><option>Champion Scorecard</option></select><select id="calsplit"><option>OOT_Monitor</option><option>OOT_Valid</option></select><select id="calmethod"><option value="Intercept" selected>Intercept Recalibration</option><option value="Before">Before Recalibration</option><option value="Platt">Platt Recalibration</option></select></div><div id="calbadges" class="kpi-pills"></div><svg id="calplot" class="chart" viewBox="0 0 900 320" role="img" aria-label="Expected versus observed calibration by risk decile"></svg><div class="sub">Decile 1 = lowest predicted risk · Decile 10 = highest predicted risk.</div></article><article class="card quarterly-card"><h2>Quarterly origination cohort × risk tier · 8 chronological quarters</h2><p class="sub">Empirical application-target bad rates by origination quarter, temporal split, and LightGBM PD tier. Fixed-horizon target only; no post-origination MOB curves.</p><div id="quarterlyCohorts"></div></article><article class="card"><h2>Prior delinquency to current default · cross-sectional</h2><label>Model <select id="migrationModel"><option>Challenger LightGBM M4</option><option>Champion Scorecard</option></select></label><div id="migration" class="chart-scroll"></div></article><article class="card"><h2>Age-band AIR and observed bad rate</h2><label>Model <select id="fairmodel"><option>Challenger LightGBM M4</option><option>Champion Scorecard</option></select></label><div id="fairness"></div><p class="sub">AIR is a screening diagnostic, not a legal conclusion. Review subgroup risk, label coverage, and policy effects together.</p></article></div></section>
<section class="view" id="v7"><div class="heading"><div><h1>Architecture, SQL Lineage &amp; Malaysia Regulatory Mapping</h1><p>Artifact-backed pipeline lineage, executable SQL sources, and verified integrity evidence.</p></div><span class="badge">Release Build: FinGuard Risk OS v2.0 · Stage 6 Blind-Verified</span><span id="integrityBadge" class="badge"></span></div><div id="integrityPanel" class="grid cols3"></div><div id="lineage" class="grid cols3" style="margin-top:15px"></div><article class="card" style="margin-top:15px"><h2>Selected pipeline stage detail</h2><div id="stageDetail"></div></article><article class="card" style="margin-top:15px"><h2>SQL lineage sources</h2><div id="sqlTabs" class="pillrow"></div><pre id="sqlSource" class="sql-source"></pre></article><article class="card malaysia-map"><h2>Malaysia Production &amp; Regulatory Mapping</h2><p class="sub">Prototype mapping for design discussion only; applicability and implementation require product-specific compliance review.</p><div class="reg-map-grid"><section><h3>BNM Responsible Financing · BNM/RH/PD 028-95</h3><p>Prototype policy uses total DSR ≤ 50% and checks residual Net Disposable Income (NDI); the 50% ceiling is a project policy assumption, not a universal BNM cap. Use reducing-balance amortization assumptions. Prepare for applicable flat-rate / Rule of 78 restrictions taking effect in 2027; do not treat this prototype as compliance evidence.</p><small><a href="https://alumni.bnm.gov.my/web/guest/-/pd-pf-925-bm" target="_blank" rel="noreferrer">BNM Policy Document on Personal Financing</a></small></section><section><h3>BNM FTFC · MAS FEAT</h3><p>Govern fair treatment and proxy-variable impact testing; retain interpretable TreeSHAP adverse-action drivers, clear explanations, human review and an applicant appeal path. Sensitive attribute coverage remains a material limitation. See the <a href="https://www.bnm.gov.my/ccris" target="_blank" rel="noreferrer">CCRIS 12-month repayment record description</a>.</p><small><a href="https://alumni.bnm.gov.my/web/guest/-/pd-ftfc" target="_blank" rel="noreferrer">BNM Fair Treatment</a> · <a href="https://www.mas.gov.sg/publications/monographs-or-information-paper/2018/feat" target="_blank" rel="noreferrer">MAS FEAT</a></small></section><section><h3>CCRIS · CTOS mapping</h3><p>Map <code>bureau_active_max_dpd</code> and <code>active_bureau_monthly_debt</code> to CCRIS repayment history / existing commitments for validation. External inquiries and historical delinquency may inform CTOS-style score or trade-reference checks, subject to lawful access and data reconciliation.</p></section><section><h3>MFRS 9 ECL bridge</h3><p>Conceptual mapping: 12-month PIT recalibrated PD → Stage 1 ECL; SICR trigger → Stage 2 lifetime PD; S3/S4 macro stress multipliers → forward-looking macro overlays. This is not an ECL calculation or accounting conclusion.</p></section></div></article></section>
<section class="view" id="v6"><div class="heading"><div><h1>Scorecard & WOE Explorer</h1><p>Frozen WOE bin points, model controls, and retrospective validation limits.</p></div></div><article class="card"><h2>Interactive WOE &amp; empirical bin monotonicity explorer</h2><div class="controls"><label>Scorecard feature <select id="woeFeature"></select></label><span id="woeBadges" class="kpi-pills"></span></div><div class="grid cols2"><div><h3>Population distribution &amp; observed bad rate</h3><p class="sub woe-formula">WOE = ln(%Good / %Bad) — higher WOE = lower default risk</p><svg id="woePopulation" class="chart" viewBox="0 0 900 340" role="img" aria-label="Train and monitor feature bin distributions and bad rates"></svg></div><div><h3>WOE and assigned scorecard points</h3><svg id="woePoints" class="chart" viewBox="0 0 900 340" role="img" aria-label="WOE and scorecard points by feature bin"></svg></div></div><p class="sub">Train_Dev W0–64 and OOT_Monitor W78–91 empirical bins; ordering differences remain visible and are not smoothed.</p></article><article class="card points-summary"><h2>Feature/bin points distribution · compact summary</h2><svg id="pointChart" class="chart" viewBox="0 0 900 350" role="img" aria-label="Scorecard points by feature"></svg><div class="controls"><input id="search" placeholder="Search feature or bin" style="padding:9px;border:1px solid var(--line);border-radius:6px"><select id="pointgroup"><option value="all">All groups</option></select></div><div id="pointtable" class="tablewrap"></div></article><div class="grid cols2" style="margin-top:15px"><article class="card"><h2>Governance trigger matrix</h2><div id="governance"></div></article><article class="card"><h2>Model Development & Validation</h2><p>Retrospective evidence only. Historical funded-loan outcomes do not identify rejected-applicant outcomes or causal approval effects. Simulated ENP depends on assumed pricing, funding cost, operating cost, and loss severity.</p><button id="openM-dvd" class="pill" type="button">Open executive MDVD summary</button></article></div></section>
<div id="mdvdModal" class="modal" aria-hidden="true"><div class="modal-panel"><div class="modal-head"><div><h2>Model Development & Validation · Executive Summary</h2><p class="sub">Retrospective case study · not production approval</p></div><button id="closeM-dvd" class="pill" type="button">Close</button></div><div id="mdvdContent" class="mdvd-content"></div></div></div>
<div id="memoModal" class="modal" aria-hidden="true"><div class="modal-panel"><div class="memo-modal-actions"><button id="memoPrint" class="pill">Print / Save PDF</button><button id="memoClose" class="pill">Close</button></div><div id="memoContent"></div></div></div><div id="commandPalette" class="modal" aria-hidden="true"><div class="modal-panel palette-panel"><div class="controls"><input id="paletteSearch" placeholder="Jump to a view, case, feature, or SQL source" autocomplete="off"><button id="paletteClose" class="pill">Close</button></div><div id="paletteResults"></div></div></div><p class="foot">FinGuard retrospective risk case study · no causal claim · anonymized currency units</p></div></main><script id="finguard-data" type="application/json">__DATA__</script><script src="./finguard_bundle.js"></script><script>
__APP_JS__</script></body></html>'''

if __name__ == "__main__": main()
