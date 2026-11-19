from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI

from .decision import evaluate, runtime
from .schemas import Applicant, BatchRequest, DecisionResponse

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="FinGuard Risk OS Decision API", version="2.0.0")


@app.get("/v1/health")
def health() -> dict:
    rt = runtime()
    digest = hashlib.sha256((ROOT / "configs" / "decision_engine_policy_v2.json").read_bytes()).hexdigest()
    return {"status": "healthy", "service_version": "2.0.0", "model_versions": rt["config"]["model_versions"],
            "feature_counts": {"champion": len(rt["scorecard"]["features"]), "challenger": len(rt["challenger"]["features"])},
            "calibration_status": {"fit_split": "OOT_Valid", "oot_monitor_used_for_fit": False},
            "policy_version": rt["config"]["policy_version"], "policy_sha256": digest}


@app.post("/v1/underwrite", response_model=DecisionResponse)
def underwrite(applicant: Applicant) -> dict:
    return evaluate(applicant.model_dump())


@app.post("/v1/batch-underwrite")
def batch_underwrite(batch: BatchRequest) -> dict:
    return {"count": len(batch.applicants), "decisions": [evaluate(a.model_dump()) for a in batch.applicants]}
