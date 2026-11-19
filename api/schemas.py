from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: int | None = None
    requested_amount: float | None = Field(default=None, ge=0)
    annuity: float | None = Field(default=None, ge=0)
    monthly_income: float | None = None
    existing_monthly_debt: float | None = Field(default=None, ge=0)
    features: dict[str, Any] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=1000)


class DecisionResponse(BaseModel):
    trace_id: str
    decision_signature: str
    signature_algorithm: Literal["HMAC-SHA256"]
    signature_key_id: str
    tri_state_route: Literal["AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_DECLINE"]
    champion: dict[str, Any]
    challenger: dict[str, Any]
    hard_rule_trace: list[dict[str, Any]]
    limit_math: dict[str, Any]
    reason_codes: list[str]
    actionable_recourse: list[str]
