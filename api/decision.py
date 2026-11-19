from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import pickle
import gc
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "decision_engine_policy_v2.json"


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))))


def _logit(probability: float) -> float:
    p = min(1.0 - 1e-15, max(1e-15, probability))
    return math.log(p / (1.0 - p))


def _woe_value(value: Any, rule: dict[str, Any]) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float(rule.get("missing_woe", 0.0))
    if rule["dtype"] == "categorical":
        for bin_spec in rule["bins"]:
            if str(value) in {str(category) for category in bin_spec.get("categories", [])}:
                return float(bin_spec["woe"])
        return float(rule.get("unknown_woe") or 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(rule.get("missing_woe", 0.0))
    if not math.isfinite(number):
        return float(rule.get("missing_woe", 0.0))
    for bin_spec in rule["bins"]:
        lower = -math.inf if bin_spec["lower"] == "-Infinity" else float(bin_spec["lower"])
        upper = math.inf if bin_spec["upper"] == "Infinity" else float(bin_spec["upper"])
        if lower <= number < upper:
            return float(bin_spec["woe"])
    return float(rule.get("missing_woe", 0.0))


@lru_cache(maxsize=1)
def runtime() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    coefficient_path = PROC / "scorecard_coefficients_v2.json"
    if coefficient_path.exists():
        scorecard = json.loads(coefficient_path.read_text(encoding="utf-8"))
        scorecard["coef"] = np.asarray(scorecard["coef"], dtype=np.float64)
    else:
        scorecard = pickle.load(open(PROC / "models" / "scorecard_logit.pkl", "rb"))
        scorecard["coef"] = np.asarray(scorecard["result"].params, dtype=np.float64)
        del scorecard["result"]  # Release the full Train_Dev design matrix after coefficient extraction.
        gc.collect()
    challenger = pickle.load(open(PROC / "models" / "lgbm_m4.pkl", "rb"))
    # Keep single-row service inference bounded and avoid machine-wide OpenMP fan-out.
    challenger["model"].set_params(n_jobs=1)
    rules = json.loads((PROC / "woe_bin_rules.json").read_text(encoding="utf-8"))
    defaults_path = PROC / "model_feature_defaults.json"
    if defaults_path.exists():
        defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    else:
        defaults = {"scorecard": {}, "challenger": {}}
    return {"config": config, "scorecard": scorecard, "challenger": challenger,
            "rules": rules, "defaults": defaults}


def _limit(pd_value: float, features: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    tiers = policy["risk_tiers"]
    tier = next((t for t in tiers if t["pd_upper_exclusive"] is None or pd_value < t["pd_upper_exclusive"]), tiers[-1])
    def num(name: str, default: float = 0.0) -> float:
        try:
            value = float(features.get(name, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default
    request = max(0.0, num("credamount_770A"))
    annuity = max(0.0, num("annuity_780A"))
    income = num("maininc_215A", float("nan"))
    debt = max(0.0, num("active_bureau_monthly_debt"))
    income_verified = math.isfinite(income) and income > policy["verified_income_threshold"]
    term = int(np.clip(round(request / annuity) if annuity > 0 else 12, 3, 60))
    dpd = num("bureau_active_max_dpd", -1.0)
    history = num("static_avg_dpd_tol_closure24m")
    early = num("static_pct_inst_paid_early_3d", float("nan"))
    hard_dpd = dpd >= 90
    debt_ratio = debt / income if math.isfinite(income) and income > 0 else None
    hard_debt = debt_ratio is not None and debt_ratio > 0.60
    behavior = policy["minor_dpd_behavior_multiplier"] if dpd > 0 or history > 0 else policy["clean_history_behavior_multiplier"]
    if behavior == policy["clean_history_behavior_multiplier"] and math.isfinite(early) and early >= policy["early_payment_ratio_threshold"]:
        behavior = policy["early_payment_behavior_multiplier"]
    if hard_dpd or hard_debt or tier["tier"] == "E":
        behavior = policy["hard_policy_behavior_multiplier"]
    if income_verified:
        risk = income * tier["verified_income_multiple"] * behavior
        maximum_payment = max(0.0, policy["dsr_cap"] * income - debt)
        monthly_rate = tier["apr_pct"] / 100.0 / 12.0
        affordability = maximum_payment * (1 - (1 + monthly_rate) ** -term) / monthly_rate if monthly_rate else maximum_payment * term
        affordability_band = "VERIFIED_INCOME"
    else:
        fallback = tier["unverified_fallback_pct"] / 100.0
        risk = request * fallback * behavior
        affordability = request * fallback
        affordability_band = "UNVERIFIED_INCOME"
    shadow = max(0.0, min(request, risk, affordability))
    final = 0.0 if hard_dpd or hard_debt or tier["tier"] == "E" else shadow
    return {"risk_tier": tier["tier"], "affordability_band": affordability_band,
            "requested": request, "risk_behavior_cap": risk, "affordability_cap": affordability,
            "shadow_limit": shadow, "final_limit": final, "apr_pct": tier["apr_pct"],
            "behavior_multiplier": behavior, "term_months": term, "dsr_cap": policy["dsr_cap"],
            "income_verified": bool(income_verified), "hard_policy_decline": bool(hard_dpd or hard_debt)}


def evaluate(applicant: dict[str, Any]) -> dict[str, Any]:
    rt = runtime()
    cfg, sc_art, lgb_art, defaults = rt["config"], rt["scorecard"], rt["challenger"], rt["defaults"]
    features = {**defaults.get("challenger", {}), **defaults.get("scorecard", {}), **applicant.get("features", {})}
    for key, value in (("credamount_770A", applicant.get("requested_amount")),
                       ("annuity_780A", applicant.get("annuity")),
                       ("maininc_215A", applicant.get("monthly_income")),
                       ("active_bureau_monthly_debt", applicant.get("existing_monthly_debt"))):
        if value is not None:
            features[key] = value
    score_features = sc_art["features"]
    woe = np.asarray([_woe_value(features.get(f), rt["rules"][f]) for f in score_features], dtype=np.float64)
    score_raw = _sigmoid(float(sc_art["coef"][0] + np.dot(sc_art["coef"][1:], woe)))
    challenger_features = lgb_art["features"]
    frame = pd.DataFrame([{f: features.get(f) for f in challenger_features}], columns=challenger_features)
    booster = lgb_art["model"].booster_
    category_levels = getattr(booster, "pandas_categorical", None) or []
    category_indices = booster.params.get("categorical_column", [])
    if isinstance(category_indices, str):
        category_indices = [int(x) for x in category_indices.split(",") if x]
    categorical_set = {int(i) for i in category_indices}
    for index, column in enumerate(challenger_features):
        if index not in categorical_set:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    for index, categories in zip(category_indices, category_levels):
        column = challenger_features[int(index)]
        frame[column] = pd.Categorical(frame[column], categories=categories)
    challenger_raw = float(lgb_art["model"].predict_proba(frame)[0, 1])
    champion_cal = cfg["calibration"]["champion"]
    challenger_cal = cfg["calibration"]["challenger"]
    champion = {"pd_raw": score_raw,
                "pd_intercept": _sigmoid(_logit(score_raw) + champion_cal["intercept_alpha"]),
                "pd_platt": _sigmoid(champion_cal["platt_alpha"] + champion_cal["platt_beta"] * _logit(score_raw))}
    challenger = {"pd_raw": challenger_raw,
                  "pd_intercept": _sigmoid(_logit(challenger_raw) + challenger_cal["intercept_alpha"]),
                  "pd_platt": _sigmoid(challenger_cal["platt_alpha"] + challenger_cal["platt_beta"] * _logit(challenger_raw))}
    thresholds = cfg["thresholds"]["iso_risk"]
    champion["verdict"] = "APPROVE" if champion["pd_platt"] <= thresholds["champion_pd_cutoff"] else "DECLINE"
    challenger["verdict"] = "APPROVE" if challenger["pd_platt"] <= thresholds["challenger_pd_cutoff"] else "DECLINE"
    lp = cfg["limit_policy"]
    limit = _limit(challenger["pd_platt"], features, lp)
    rule_dpd = float(features.get("bureau_active_max_dpd") or 0) >= 90
    income = features.get("maininc_215A")
    debt = float(features.get("active_bureau_monthly_debt") or 0)
    verified = income is not None and math.isfinite(float(income)) and float(income) > lp["verified_income_threshold"]
    rule_debt = bool(verified and debt / float(income) > .60)
    hard_trace = [
        {"rule_id": "R01", "triggered": rule_dpd, "reason_code": "SEVERE_ACTIVE_DELINQUENCY" if rule_dpd else None},
        {"rule_id": "R02", "triggered": rule_debt, "reason_code": "EXISTING_DEBT_TO_INCOME_OVER_60_PERCENT" if rule_debt else None},
    ]
    router = cfg["tri_state_router"]
    cutoff = thresholds["challenger_pd_cutoff"]
    grey = router["grey_zone_half_width_absolute_pd"]
    disagreement = champion["verdict"] != challenger["verdict"]
    if rule_dpd or rule_debt or limit["risk_tier"] == "E":
        route = "AUTO_DECLINE"
    elif abs(challenger["pd_platt"] - cutoff) <= grey or (router["model_disagreement_routes_to_review"] and disagreement):
        route = "MANUAL_REVIEW"
    elif challenger["pd_platt"] < cutoff:
        route = "AUTO_APPROVE"
    else:
        route = "AUTO_DECLINE"
    reasons = [r["reason_code"] for r in hard_trace if r["triggered"]]
    if challenger["verdict"] == "DECLINE":
        reasons.append("CHALLENGER_PD_ABOVE_ISO_RISK_CUTOFF")
    if disagreement:
        reasons.append("CHAMPION_CHALLENGER_DISAGREEMENT")
    if limit["shadow_limit"] < limit["requested"]:
        reasons.append("LIMIT_REDUCED_BY_RISK_OR_AFFORDABILITY_CAP")
    recourse = []
    if not verified:
        recourse.append("Provide verifiable income documentation so affordability can be assessed under the verified-income DSR policy.")
    if rule_debt:
        recourse.append("Reduce existing monthly debt or provide corrected debt evidence; the hard debt-to-income rule must clear.")
    if rule_dpd:
        recourse.append("Resolve severe active delinquency and provide updated bureau evidence for reassessment.")
    if not recourse and route != "AUTO_APPROVE":
        recourse.append("Request human review or reassessment after material, verifiable changes; no approval is guaranteed.")
    body = {"tri_state_route": route, "champion": champion, "challenger": challenger,
            "hard_rule_trace": hard_trace, "limit_math": limit, "reason_codes": reasons,
            "actionable_recourse": recourse}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    trace = hashlib.sha256((str(applicant.get("case_id") or "") + ":").encode() + canonical).hexdigest()[:24]
    signing_key = os.environ.get("FINGUARD_DECISION_SIGNING_KEY", "local-development-key-not-for-production").encode()
    key_id = "environment" if os.environ.get("FINGUARD_DECISION_SIGNING_KEY") else "local-development"
    signature = hmac.new(signing_key, canonical, hashlib.sha256).hexdigest()
    return {"trace_id": trace, "decision_signature": signature, "signature_algorithm": "HMAC-SHA256",
            "signature_key_id": key_id, **body}
