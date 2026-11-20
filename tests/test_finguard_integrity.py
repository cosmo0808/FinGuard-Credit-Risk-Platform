"""Artifact-level checks for FinGuard's temporal, model, and policy invariants."""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PRED = (PROC / "stage5_full_predictions.parquet").as_posix()
MART = (PROC / "finguard_feature_mart.parquet").as_posix()


@pytest.fixture(scope="session")
def con():
    db = duckdb.connect()
    db.execute("SET memory_limit='4GB'; SET threads=4")
    yield db
    db.close()


@pytest.fixture(scope="session")
def scorecard():
    return pickle.load(open(PROC / "models" / "scorecard_logit.pkl", "rb"))


@pytest.fixture(scope="session")
def booster():
    return pickle.load(open(PROC / "models" / "lgbm_m4.pkl", "rb"))


def test_01_temporal_splits_are_disjoint_and_complete(con):
    total, distinct = con.execute(f"SELECT count(*),count(distinct case_id) FROM read_parquet('{PRED}')").fetchone()
    assert total == distinct == 1_526_659
    rows = con.execute(f"""SELECT split_name,min(WEEK_NUM),max(WEEK_NUM),count(*),count(distinct case_id)
      FROM read_parquet('{PRED}') GROUP BY 1""").fetchall()
    got = {r[0]: (r[1], r[2], r[3], r[4]) for r in rows}
    assert got == {"Train_Dev": (0, 64, 1292350, 1292350),
                   "OOT_Valid": (65, 77, 82602, 82602),
                   "OOT_Monitor": (78, 91, 151707, 151707)}
    assert sum(x[3] for x in got.values()) == 1_526_659


def test_02_fitted_models_exclude_leakage_and_sensitive_fields(scorecard, booster):
    forbidden = {"disbursedcredamount_1113A", "person_gender_sensitive",
                 "person_age_band_sensitive", "person_age"}
    assert forbidden.isdisjoint(scorecard["features"])
    assert forbidden.isdisjoint(booster["features"])


def test_03_champion_coefficients_pvalues_and_vif_pass(con, scorecard):
    result = scorecard["result"]
    coef, pval = np.asarray(result.params[1:], float), np.asarray(result.pvalues[1:], float)
    rules = json.loads((PROC / "woe_bin_rules.json").read_text(encoding="utf-8"))
    raw = con.execute(f"SELECT {','.join(chr(34)+f+chr(34) for f in scorecard['features'])} FROM read_parquet('{MART}') WHERE WEEK_NUM<=64").fetchdf()
    matrix = np.empty(raw.shape, dtype=np.float32)
    for j, feature in enumerate(scorecard["features"]):
        rule = rules[feature]; series = raw[feature]
        col = np.full(len(raw), float(rule["missing_woe"]), dtype=np.float32)
        if rule["dtype"] == "categorical":
            values = series.astype("string")
            for b in rule["bins"]:
                col[values.isin([str(x) for x in b["categories"]]).to_numpy()] = float(b["woe"])
            if rule.get("unknown_woe") is not None:
                col[series.notna().to_numpy() & ~values.isin([str(x) for b in rule["bins"] for x in b["categories"]]).to_numpy()] = float(rule["unknown_woe"])
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(float)
            valid = np.isfinite(values)
            for b in rule["bins"]:
                lo = -np.inf if b["lower"] == "-Infinity" else float(b["lower"])
                hi = np.inf if b["upper"] == "Infinity" else float(b["upper"])
                col[valid & (values >= lo) & (values < hi)] = float(b["woe"])
        matrix[:, j] = col
    gram = matrix.astype(np.float64).T @ matrix.astype(np.float64)
    vif = []
    for i in range(gram.shape[0]):
        other = np.arange(gram.shape[0]) != i
        beta = np.linalg.solve(gram[np.ix_(other, other)], gram[other, i])
        residual = gram[i, i] - gram[i, other] @ beta
        vif.append(gram[i, i] / residual)
    assert len(coef) == len(pval) == len(vif) == 19
    assert np.all(coef < 0) and np.all(pval < .05) and np.max(vif) < 5.0, f"max VIF={np.max(vif)}"


def test_04_pdo_score_reconstructs_saved_logit_pd(con, scorecard):
    q = con.execute(f"SELECT scorecard_score,pd_scorecard FROM read_parquet('{PRED}')").fetchdf()
    factor = float(scorecard["factor"])
    p = 1 / (1 + np.exp(-((600 - q.scorecard_score.to_numpy()) / factor + np.log(.05))))
    assert np.max(np.abs(p - q.pd_scorecard.to_numpy())) < 1e-12


def test_05_oot_valid_bad_rates_strictly_decrease_across_ten_risk_deciles(con):
    for model, column in [("Champion", "pd_scorecard"), ("Challenger", "pd_lgbm_raw")]:
        q = con.execute(f"SELECT target,{column} p FROM read_parquet('{PRED}') WHERE split_name='OOT_Valid'").fetchdf()
        q["decile"] = pd.qcut(q.p.rank(method="first", ascending=False), 10, labels=False)
        rates = q.groupby("decile").target.mean().to_numpy()
        assert len(rates) == 10 and np.all(np.diff(rates) < 0), f"{model} descending-risk deciles: {rates}"


def test_06_recalibrator_is_valid_only_and_intercept_preserves_rank(con):
    meta = json.loads((PROC / "models" / "recalibrators_stage5.json").read_text(encoding="utf-8"))
    assert meta["oot_monitor_used_for_fit"] is False
    assert meta["fit_split"] == "OOT_Valid" and meta["fit_weeks"] == [65, 77]
    q = con.execute(f"""SELECT target,pd_scorecard,pd_lgbm_raw,pd_scorecard_recal_intercept,
      pd_lgbm_recal_intercept FROM read_parquet('{PRED}') WHERE split_name='OOT_Monitor'""").fetchdf()
    for raw, cal in [("pd_scorecard", "pd_scorecard_recal_intercept"),
                     ("pd_lgbm_raw", "pd_lgbm_recal_intercept")]:
        y = q.target.to_numpy()
        ks = []
        for c in (raw, cal):
            fpr, tpr, _ = roc_curve(y, q[c].to_numpy())
            ks.append(float(np.max(tpr - fpr)))
        assert abs(ks[0] - ks[1]) < 1e-12


def _monitor_frame(con):
    return con.execute(f"""SELECT p.case_id,p.target,p.pd_lgbm_recal_intercept AS pd,
      m.credamount_770A,m.annuity_780A,m.maininc_215A,m.active_bureau_monthly_debt,
      m.bureau_active_max_dpd,m.static_avg_dpd_tol_closure24m,m.static_pct_inst_paid_early_3d
      FROM read_parquet('{PRED}') p JOIN read_parquet('{MART}') m USING(case_id,WEEK_NUM)
      WHERE p.split_name='OOT_Monitor'""").fetchdf()


@pytest.fixture(scope="session")
def policy(con):
    sys.path.insert(0, str(ROOT / "src"))
    from portfolio_scenarios_powerbi_stage6 import loan_engine
    frame = _monitor_frame(con)
    return frame, loan_engine(frame, frame.pd.to_numpy())


def test_07_recommended_limits_are_bounded_finite(policy):
    frame, out = policy
    limit = out["limit"]
    req = np.nan_to_num(frame.credamount_770A.to_numpy(float), nan=0.).clip(0)
    assert np.isfinite(limit).all() and np.isfinite(req).all()
    assert np.all(limit >= 0) and np.all(limit <= req + 1e-7)
    assert all(np.isfinite(v).all() for v in out.values() if isinstance(v, np.ndarray) and v.dtype.kind in "fiu")


def test_08_policy_knockouts_always_have_zero_limit(policy):
    frame, out = policy
    dpd = frame.bureau_active_max_dpd.to_numpy(float)
    income = frame.maininc_215A.to_numpy(float)
    debt = np.nan_to_num(frame.active_bureau_monthly_debt.to_numpy(float), nan=0.)
    knockout = (np.nan_to_num(dpd, nan=-1.) >= 90) | (
        np.isfinite(income) & (income > 0) & (debt / np.where(income != 0, income, np.nan) > .60))
    assert knockout.any() and np.all(out["limit"][knockout] == 0)


def test_09_verified_income_approvals_satisfy_total_dsr_cap(policy):
    frame, out = policy
    income = frame.maininc_215A.to_numpy(float)
    debt = np.nan_to_num(frame.active_bureau_monthly_debt.to_numpy(float), nan=0.).clip(0)
    rate, term, limit = out["apr"] / 12, out["term"], out["limit"]
    factor = np.divide(rate, 1 - (1 + rate) ** -term, out=1 / term.astype(float), where=rate > 0)
    payment = limit * factor
    valid = out["verified"] & (limit > 0) & ~out["policy"]
    dsr = (debt + payment) / np.where(income > 0, income, np.inf)
    assert valid.any() and np.max(dsr[valid]) <= .50 + 1e-6


def test_10_oot_monitor_isorisk_headline_matches_stage6_artifact():
    df = pd.read_csv(ROOT / "data" / "powerbi" / "pbi_04_stress_scenarios.csv")
    rows = df[(df.Scenario == "S1_Baseline") & (df.Risk_Tier == "Portfolio Total") & (df.Model == "Challenger LightGBM M4")]
    assert len(rows) == 1
    r = rows.iloc[0]
    assert int(r.Approved_Count) == 114021
    assert abs(float(r.Monitor_Approval_Pct) - 75.16) < .005
    assert abs(float(r.Observed_Bad_Rate_Pct) - .8832) < .0001
    assert abs(float(r.Model_Implied_Expected_Net_Profit) - 142_305_881) < 50_000


def test_11_exported_ui_has_no_fabricated_currency_labels():
    for path in [ROOT / "docs/index.html", ROOT / "dashboard/FinGuard_Interactive_Command_Center.html"]:
        html = path.read_text(encoding="utf-8")
        assert not re.search(r"\bMYR\b|(?:^|[\s=:])\$\s?\d", html, re.I)


def test_12_python_model_parity_fixtures_cover_five_monitor_spotlights(con, scorecard, booster):
    cases = con.execute(f"""SELECT case_id,pd_scorecard,pd_lgbm_raw,pd_scorecard_recal_intercept,
      pd_lgbm_recal_intercept,pd_scorecard_recal_platt,pd_lgbm_recal_platt
      FROM read_parquet('{PRED}') WHERE split_name='OOT_Monitor' ORDER BY case_id LIMIT 5""").fetchdf()
    assert len(cases) == 5 and len(scorecard["features"]) == 19 and len(booster["features"]) == 73
    assert np.isfinite(cases.iloc[:, 1:].to_numpy()).all()
    qa_path = ROOT / "reports" / "finguard_browser_qa_report.json"
    assert qa_path.exists(), "Run the headless browser QA before publishing the integrity report"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["status"] == "PASS" and not qa["js_errors"] and not qa["console_errors"]
    parity = qa["parity"]
    assert len(parity) == 5, f"Expected browser parity for all five cases, got {len(parity)}"
    assert len({r["case"] for r in parity}) == 5
    for row in parity:
        errors = [v for k, v in row.items() if k.endswith("Abs")]
        assert errors and max(errors) < 1e-6, f"{row.get('spotlight', row['case'])}: {errors}"


def test_13_policy_config_json_schema_and_thresholds():
    from jsonschema import validate
    cfg = json.loads((ROOT / "configs/decision_engine_policy_v2.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "configs/decision_engine_policy_v2.schema.json").read_text(encoding="utf-8"))
    validate(cfg, schema)
    matrix = cfg["limit_policy"]["decision_matrix"]
    assert len(matrix) == 10
    assert {(r["risk_tier"], r["affordability_band"]) for r in matrix} == {
        (tier, band) for tier in "ABCDE" for band in ("VERIFIED_INCOME", "UNVERIFIED_INCOME")}
    assert cfg["model_versions"]["oot_monitor_used_for_fit"] is False
    assert cfg["thresholds"]["decision_pd_scale"] == "stage5_intercept_recalibrated_pd"
    assert cfg["thresholds"]["iso_risk"]["challenger_pd_cutoff"] == pytest.approx(.0463125046661152)


def test_14_fastapi_health_and_single_underwrite_parity():
    from fastapi.testclient import TestClient
    from api.app import app
    from api.decision import runtime
    from api.schemas import DecisionResponse
    sys.path.insert(0, str(ROOT / "src"))
    from portfolio_scenarios_powerbi_stage6 import loan_engine
    con = duckdb.connect()
    con.execute("SET memory_limit='256MB'; SET threads=1")
    defaults = json.loads((PROC / "model_feature_defaults.json").read_text(encoding="utf-8"))
    features = list(dict.fromkeys([*defaults["challenger"], *defaults["scorecard"]]))
    cases = con.execute(f"""SELECT p.case_id,p.pd_scorecard,p.pd_lgbm_raw,p.pd_scorecard_recal_intercept,
      p.pd_lgbm_recal_intercept,p.pd_scorecard_recal_platt,p.pd_lgbm_recal_platt,
      m.credamount_770A op_request,m.annuity_780A op_annuity,m.maininc_215A op_income,
      m.active_bureau_monthly_debt op_debt,m.bureau_active_max_dpd op_dpd,
      m.static_avg_dpd_tol_closure24m op_history,m.static_pct_inst_paid_early_3d op_early,
      m.{','.join(chr(34)+f+chr(34) for f in features)}
      FROM read_parquet('{PRED}') p JOIN read_parquet('{MART}') m USING(case_id,WEEK_NUM)
      WHERE p.split_name='OOT_Monitor' LIMIT 2""").fetchdf()
    con.close()
    runtime()
    assert {"/v1/health", "/v1/underwrite", "/v1/batch-underwrite"}.issubset(app.openapi()["paths"])
    client = TestClient(app)
    health_result = client.get("/v1/health").json()
    assert health_result["feature_counts"] == {"champion": 19, "challenger": 73}
    assert health_result["calibration_status"]["oot_monitor_used_for_fit"] is False
    assert len(cases) == 2
    applicants = []
    responses = []
    browser_cases = []
    cfg = json.loads((ROOT / "configs/decision_engine_policy_v2.json").read_text(encoding="utf-8"))
    thresholds = cfg["thresholds"]["iso_risk"]
    for _, row in cases.iterrows():
        applicant_features = {f: (None if pd.isna(row[f]) else row[f].item() if isinstance(row[f], np.generic) else row[f])
                              for f in features}
        applicant_features.update({"bureau_active_max_dpd": None if pd.isna(row.op_dpd) else float(row.op_dpd),
                                   "static_avg_dpd_tol_closure24m": None if pd.isna(row.op_history) else float(row.op_history),
                                   "static_pct_inst_paid_early_3d": None if pd.isna(row.op_early) else float(row.op_early)})
        request = {"case_id": int(row.case_id), "requested_amount": float(row.op_request),
                   "annuity": None if pd.isna(row.op_annuity) else float(row.op_annuity),
                   "monthly_income": None if pd.isna(row.op_income) else float(row.op_income),
                   "existing_monthly_debt": None if pd.isna(row.op_debt) else float(row.op_debt),
                   "features": applicant_features}
        applicants.append(request)
        browser_cases.append({**applicant_features, "credamount_770A": request["requested_amount"],
                              "annuity_780A": request["annuity"], "maininc_215A": request["monthly_income"],
                              "active_bureau_monthly_debt": request["existing_monthly_debt"]})
        response = client.post("/v1/underwrite", json=request)
        assert response.status_code == 200, response.text
        result = response.json()
        responses.append(result)
        DecisionResponse.model_validate(result)
        assert result["champion"]["pd_raw"] == pytest.approx(float(row.pd_scorecard), abs=1e-9)
        assert result["challenger"]["pd_raw"] == pytest.approx(float(row.pd_lgbm_raw), abs=1e-9)
        assert result["champion"]["pd_intercept"] == pytest.approx(float(row.pd_scorecard_recal_intercept), abs=1e-9)
        assert result["challenger"]["pd_intercept"] == pytest.approx(float(row.pd_lgbm_recal_intercept), abs=1e-9)
        assert result["champion"]["pd_platt"] == pytest.approx(float(row.pd_scorecard_recal_platt), abs=1e-9)
        assert result["challenger"]["pd_platt"] == pytest.approx(float(row.pd_lgbm_recal_platt), abs=1e-9)
        assert len(result["trace_id"]) == 24 and len(result["decision_signature"]) == 64
        expected_sc_verdict = "APPROVE" if row.pd_scorecard_recal_intercept <= thresholds["champion_pd_cutoff"] else "DECLINE"
        expected_lg_verdict = "APPROVE" if row.pd_lgbm_recal_intercept <= thresholds["challenger_pd_cutoff"] else "DECLINE"
        assert result["champion"]["verdict"] == expected_sc_verdict
        assert result["challenger"]["verdict"] == expected_lg_verdict
        stage_features = {**applicant_features, "credamount_770A": float(row.op_request),
                          "annuity_780A": request["annuity"], "maininc_215A": request["monthly_income"],
                          "active_bureau_monthly_debt": request["existing_monthly_debt"]}
        stage4 = loan_engine(pd.DataFrame([stage_features]), np.asarray([float(row.pd_lgbm_recal_intercept)]))
        assert result["limit_math"]["risk_tier"] == stage4["tier"][0].split(":")[0][-1]
        expected_final_limit = float(stage4["limit"][0]) if row.pd_lgbm_recal_intercept <= thresholds["challenger_pd_cutoff"] else 0.0
        assert result["limit_math"]["final_limit"] == pytest.approx(expected_final_limit, abs=1e-6)
        assert result["limit_math"]["term_months"] == pytest.approx(float(stage4["term"][0]), abs=1e-6)
        assert result["limit_math"]["apr_pct"] == pytest.approx(float(stage4["apr"][0] * 100), abs=1e-6)
        income_value = request["monthly_income"]
        debt_ratio = request["existing_monthly_debt"] / income_value if income_value and income_value > 100 else None
        hard = (request["features"].get("bureau_active_max_dpd") is not None and request["features"]["bureau_active_max_dpd"] >= 90) or (debt_ratio is not None and debt_ratio > .60)
        dsr_exhausted = debt_ratio is not None and debt_ratio >= cfg["limit_policy"]["dsr_cap"]
        disagreement = expected_sc_verdict != expected_lg_verdict
        if hard or dsr_exhausted or stage4["tier"][0] == "Tier E: Decline":
            expected_route = "AUTO_DECLINE"
        elif abs(float(row.pd_lgbm_recal_intercept) - thresholds["challenger_pd_cutoff"]) <= cfg["tri_state_router"]["grey_zone_half_width_absolute_pd"] or disagreement:
            expected_route = "MANUAL_REVIEW"
        elif row.pd_lgbm_recal_intercept < thresholds["challenger_pd_cutoff"] and expected_final_limit > 0:
            expected_route = "AUTO_APPROVE"
        else:
            expected_route = "AUTO_DECLINE"
        assert result["tri_state_route"] == expected_route
        assert result["tri_state_route"] != "AUTO_APPROVE" or result["limit_math"]["final_limit"] > 0
    batch_response = client.post("/v1/batch-underwrite", json={"applicants": applicants})
    assert batch_response.status_code == 200
    batch = batch_response.json()
    assert batch["count"] == 2 and len(batch["decisions"]) == 2
    for single, batched in zip(responses, batch["decisions"]):
        assert batched["tri_state_route"] == single["tri_state_route"]
        assert batched["challenger"]["verdict"] == single["challenger"]["verdict"]
        assert batched["limit_math"]["risk_tier"] == single["limit_math"]["risk_tier"]
        assert batched["limit_math"]["final_limit"] == pytest.approx(single["limit_math"]["final_limit"], abs=1e-9)

    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright-browsers"))
    from playwright.sync_api import sync_playwright
    browser_errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
        page = browser.new_page()
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.on("console", lambda message: browser_errors.append(message.text) if message.type == "error" else None)
        page.goto((ROOT / "docs" / "index.html").as_uri(), wait_until="load")
        for (_, row), api_result, browser_features in zip(cases.iterrows(), responses, browser_cases):
            decision = page.evaluate("""data=>{
              const raw={};D.scorecard.features.forEach(f=>{raw[f]=Object.hasOwn(data,f)?data[f]:D.scorecard.defaults[f];const e=document.querySelector(`#f_${CSS.escape(f)}`);if(e)e.value=raw[f]===null?'':String(raw[f])});
              for(const [id,key] of [['request','credamount_770A'],['income','maininc_215A'],['debt','active_bureau_monthly_debt']]){const e=document.querySelector('#'+id);if(e)e.value=data[key]===null?'':String(data[key])}
              const req=Number(data.credamount_770A)||0,ann=Number(data.annuity_780A)||0,term=document.querySelector('#tenor');if(term)term.value=String(ann>0?Math.max(3,Math.min(60,Math.round(req/ann))):12);
              window.__liveVector=D.lgbm_features.map(f=>fgEncode(f,Object.hasOwn(data,f)?data[f]:D.lgbm_defaults[f]));window.FG_ENGINE.render(window.__liveVector,raw);return window.__lastDecision
            }""", browser_features)
            assert decision["champion"]["verdict"] == api_result["champion"]["verdict"]
            assert decision["challenger"]["verdict"] == api_result["challenger"]["verdict"]
            assert decision["challenger"]["risk_tier"].split(":")[0].replace("Tier ", "") == api_result["limit_math"]["risk_tier"]
            assert decision["tri_state_route"] == api_result["tri_state_route"]
            assert decision["final_limit"] == pytest.approx(api_result["limit_math"]["final_limit"], abs=1e-6)
        assert browser_errors == []
        browser.close()


def test_15_rule_waterfall_and_tristate_accounting_identity(con, policy):
    sys.path.insert(0, str(ROOT / "src"))
    from portfolio_scenarios_powerbi_stage6 import loan_engine
    cfg = json.loads((ROOT / "configs/decision_engine_policy_v2.json").read_text(encoding="utf-8"))
    cutoff = cfg["thresholds"]["iso_risk"]["challenger_pd_cutoff"]
    frame, _ = policy
    dpd = frame.bureau_active_max_dpd.to_numpy(float)
    income = frame.maininc_215A.to_numpy(float)
    debt = np.nan_to_num(frame.active_bureau_monthly_debt.to_numpy(float), nan=0.)
    r01 = np.nan_to_num(dpd, nan=-1.) >= 90
    r02 = np.isfinite(income) & (income > 0) & (debt / np.where(income > 0, income, 1) > .60)
    net_r02 = r02 & ~r01
    knocked = r01 | net_r02
    assert len(frame) == 151707
    assert int(r01.sum()) + int(net_r02.sum()) == int(knocked.sum())
    assert int(knocked.sum()) + int((~knocked).sum()) == 151707
    assert np.all(np.isfinite(np.asarray([r01.mean(), net_r02.mean(), knocked.mean()])))
    decisions = con.execute(f"""SELECT p.pd_scorecard_recal_intercept,p.pd_lgbm_recal_intercept,
      m.bureau_active_max_dpd,m.maininc_215A,m.active_bureau_monthly_debt,
      m.credamount_770A,m.annuity_780A,m.static_avg_dpd_tol_closure24m,m.static_pct_inst_paid_early_3d
      FROM read_parquet('{PRED}') p JOIN read_parquet('{MART}') m USING(case_id,WEEK_NUM)
      WHERE p.split_name='OOT_Monitor'""").fetchdf()
    sc = decisions.pd_scorecard_recal_intercept.to_numpy(float)
    lg = decisions.pd_lgbm_recal_intercept.to_numpy(float)
    d = decisions.bureau_active_max_dpd.to_numpy(float)
    inc = decisions.maininc_215A.to_numpy(float)
    existing_debt = np.nan_to_num(decisions.active_bureau_monthly_debt.to_numpy(float), nan=0.)
    hard = (np.nan_to_num(d, nan=-1.) >= 90) | (np.isfinite(inc) & (inc > 100) & (existing_debt / np.where(inc > 0, inc, 1) > .60))
    dsr_exhausted = np.isfinite(inc) & (inc > 100) & (existing_debt >= .50 * inc)
    scorecard_approve = sc <= cfg["thresholds"]["iso_risk"]["champion_pd_cutoff"]
    challenger_approve = lg <= cutoff
    tier_e = lg >= .07
    challenger_limits = loan_engine(decisions, lg)
    review = (np.abs(lg - cutoff) <= cfg["tri_state_router"]["grey_zone_half_width_absolute_pd"]) | (scorecard_approve != challenger_approve)
    routes = np.where(hard | dsr_exhausted | tier_e, "AUTO_DECLINE",
                      np.where(review, "MANUAL_REVIEW",
                               np.where(challenger_approve & (challenger_limits["limit"] > 0), "AUTO_APPROVE", "AUTO_DECLINE")))
    assert len(routes) == 151707
    assert sum(int((routes == route).sum()) for route in ("AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_DECLINE")) == 151707
    assert np.all(challenger_limits["limit"][routes == "AUTO_APPROVE"] > 0)


def test_16_woe_bin_explorer_completeness():
    rules = json.loads((PROC / "woe_bin_rules.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(PROC / "scorecard_points_table.csv").Feature.drop_duplicates().tolist()
    assert len(selected) == 19 and set(selected).issubset(rules)
    for feature in selected:
        rule = rules[feature]
        assert rule["bins"]
        values = [float(b["woe"]) for b in rule["bins"]]
        assert np.isfinite(values).all()
        if rule["dtype"] == "numerical":
            assert all(b["lower"] is not None and b["upper"] is not None for b in rule["bins"])
        else:
            assert all("categories" in b for b in rule["bins"])
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    payload_match = re.search(r'<script id="finguard-data" type="application/json">(.*?)</script>', html, re.S)
    assert payload_match, "Embedded WOE explorer payload is missing"
    explorer = json.loads(payload_match.group(1))["woe_explorer"]
    assert len(explorer) == 19 and {r["feature"] for r in explorer} == set(selected)
    for feature in explorer:
        assert feature["bins"]
        for split in ("train", "monitor"):
            pct = [b[f"{split}_population_pct"] for b in feature["bins"]]
            assert np.isfinite(pct).all() and sum(pct) == pytest.approx(100.0, abs=.01)
        for bin_row in feature["bins"]:
            assert np.isfinite([bin_row["woe"], bin_row["points"], bin_row["train_count"], bin_row["monitor_count"]]).all()
            for rate in (bin_row["train_bad_rate_pct"], bin_row["monitor_bad_rate_pct"]):
                assert rate is None or np.isfinite(rate)


def test_17_dsr_capacity_boundaries_and_auto_approve_limit_invariant(monkeypatch):
    from types import SimpleNamespace
    import api.decision as decision

    cfg = json.loads((ROOT / "configs/decision_engine_policy_v2.json").read_text(encoding="utf-8"))
    for model in cfg["calibration"].values():
        model["intercept_alpha"] = 0.0
        model["platt_alpha"] = 0.0
        model["platt_beta"] = 1.0

    class ConstantChallenger:
        booster_ = SimpleNamespace(pandas_categorical=[], params={})

        @staticmethod
        def predict_proba(_frame):
            return np.asarray([[0.995, 0.005]])

    fake_runtime = {
        "config": cfg,
        "scorecard": {"features": [], "coef": np.asarray([np.log(.005 / .995)])},
        "challenger": {"features": ["bureau_active_max_dpd"], "model": ConstantChallenger()},
        "rules": {},
        "defaults": {"scorecard": {}, "challenger": {}},
    }
    monkeypatch.setattr(decision, "runtime", lambda: fake_runtime)

    for ratio in (0.49, 0.50, 0.55, 0.60, 0.61):
        result = decision.evaluate({
            "case_id": int(ratio * 100), "requested_amount": 1000.0, "annuity": 100.0,
            "monthly_income": 1000.0, "existing_monthly_debt": ratio * 1000.0,
            "features": {"bureau_active_max_dpd": 0, "static_avg_dpd_tol_closure24m": 0,
                         "static_pct_inst_paid_early_3d": 0},
        })
        if ratio == 0.49:
            assert result["tri_state_route"] == "AUTO_APPROVE"
            assert result["limit_math"]["final_limit"] > 0.0
            assert "DSR_CAPACITY_EXHAUSTED" not in result["reason_codes"]
        else:
            assert result["tri_state_route"] == "AUTO_DECLINE", f"debt ratio {ratio:.0%}"
            assert result["limit_math"]["final_limit"] == 0.0
            assert "DSR_CAPACITY_EXHAUSTED" in result["reason_codes"]
            if ratio == 0.61:
                assert "EXISTING_DEBT_TO_INCOME_OVER_60_PERCENT" in result["reason_codes"]
        assert result["tri_state_route"] != "AUTO_APPROVE" or result["limit_math"]["final_limit"] > 0.0

    unverified_income = decision.evaluate({
        "case_id": 100, "requested_amount": 1000.0, "annuity": 100.0,
        "monthly_income": 100.0, "existing_monthly_debt": 80.0,
        "features": {"bureau_active_max_dpd": 0, "static_avg_dpd_tol_closure24m": 0,
                     "static_pct_inst_paid_early_3d": 0},
    })
    assert unverified_income["limit_math"]["income_verified"] is False
    assert unverified_income["limit_math"]["hard_policy_decline"] is False
    assert unverified_income["hard_rule_trace"][1]["triggered"] is False
    assert "EXISTING_DEBT_TO_INCOME_OVER_60_PERCENT" not in unverified_income["reason_codes"]
