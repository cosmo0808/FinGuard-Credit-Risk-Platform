from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

_RESULTS = []

def pytest_runtest_makereport(item, call):
    if call.when == "call":
        report = getattr(item, "rep_call", None)
        if report is not None:
            _RESULTS.append({"test": item.name, "status": report.outcome, "duration_seconds": round(report.duration, 6)})

def pytest_runtest_logreport(report):
    if report.when == "call":
        _RESULTS.append({"test": report.nodeid, "status": report.outcome, "duration_seconds": round(report.duration, 6)})

def pytest_sessionfinish(session, exitstatus):
    # logreport is the authoritative hook; de-duplicate defensively by test id.
    rows = {r["test"]: r for r in _RESULTS}
    results = list(rows.values())
    integrity = [r for r in results if "test_finguard_integrity.py::test_" in r["test"]]
    passed = sum(r["status"] == "passed" for r in integrity)
    failed = sum(r["status"] == "failed" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    root = Path(__file__).resolve().parents[1]
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "suite": "FinGuard institutional integrity",
              "invariants": 16, "test_cases": len(results), "passed": passed, "failed": failed, "skipped": skipped,
              "exit_status": int(exitstatus), "status": "PASS" if exitstatus == 0 and failed == 0 else "FAIL",
              "tests": results}
    (root / "reports" / "finguard_integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
