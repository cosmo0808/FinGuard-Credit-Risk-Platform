"""Fast, data-independent contracts intended for hosted CI."""
import json
from pathlib import Path

from jsonschema import validate
from pydantic import ValidationError

from api.schemas import Applicant, BatchRequest

ROOT = Path(__file__).resolve().parents[1]


def test_policy_schema_and_two_dimensional_matrix():
    config = json.loads((ROOT / "configs/decision_engine_policy_v2.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "configs/decision_engine_policy_v2.schema.json").read_text(encoding="utf-8"))
    validate(config, schema)
    assert config["thresholds"]["decision_pd_scale"] == "stage5_intercept_recalibrated_pd"
    assert config["thresholds"]["iso_approval"]["champion_pd_cutoff"] == config["thresholds"]["iso_risk"]["champion_pd_cutoff"]
    cells = {(r["risk_tier"], r["affordability_band"]) for r in config["limit_policy"]["decision_matrix"]}
    assert cells == {(tier, band) for tier in "ABCDE" for band in ("VERIFIED_INCOME", "UNVERIFIED_INCOME")}


def test_scorecard_pdo_constants_reconstruct_factor():
    pdo = json.loads((ROOT / "configs/decision_engine_policy_v2.json").read_text(encoding="utf-8"))["pdo_scorecard"]
    import math
    assert math.isclose(pdo["factor"], pdo["points_to_double_odds"] / math.log(2), rel_tol=1e-12)
    assert pdo["base_score"] == 600 and pdo["base_bad_good_odds"] == .05


def test_api_request_contract_and_bounds():
    applicant = Applicant(requested_amount=1000, features={"age": 40})
    assert applicant.requested_amount == 1000
    try:
        Applicant(requested_amount=-1)
    except ValidationError:
        pass
    else:
        raise AssertionError("negative requested amount must fail validation")
    assert len(BatchRequest(applicants=[applicant]).applicants) == 1
