"""Headless QA for the FinGuard standalone web app and review screenshots."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright-browsers"))
HTML = ROOT / "dashboard" / "FinGuard_Interactive_Command_Center.html"
OUT = ROOT / "dashboard" / "webapp_screenshots"
VIEWS = {
    "v1": "tab01_executive_command_center.png",
    "v2": "tab02_underwriting_sandbox.png",
    "v3": "tab03_workflow_rule_engine.png",
    "v4": "tab04_strategy_and_stress_simulator.png",
    "v5": "tab05_drift_calibration_fairness.png",
    "v6": "tab06_scorecard_woe_explorer.png",
    "v7": "tab07_architecture_sql_verification.png",
}
FORBIDDEN_TEXT = ("250 px", "15-shard", "2026-11-", ": .")


def overlaps(a, b):
    return bool(
        a and b
        and a["x"] < b["x"] + b["width"]
        and a["x"] + a["width"] > b["x"]
        and a["y"] < b["y"] + b["height"]
        and a["y"] + a["height"] > b["y"]
    )


def main():
    print("QA: starting", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not HTML.is_file():
        raise FileNotFoundError(HTML)

    errors: list[str] = []
    isolated = tempfile.TemporaryDirectory(prefix="finguard_standalone_")
    isolated_html = Path(isolated.name) / HTML.name
    shutil.copy2(HTML, isolated_html)
    if sorted(x.name for x in Path(isolated.name).iterdir()) != [HTML.name] or "finguard_bundle.js" in isolated_html.read_text(encoding="utf-8"):
        raise AssertionError("Standalone HTML has a sidecar dependency")
    console: list[str] = []
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "browser": "Playwright bundled Chromium headless",
        "viewport": "1680x1080",
        "device_scale_factor": 2,
        "tabs": {},
        "parity": [],
        "screenshots": [],
    }

    with sync_playwright() as pw:
        print("QA: launching bundled Chromium", flush=True)
        browser = pw.chromium.launch(headless=True, args=["--disable-gpu"])
        context = browser.new_context(
            viewport={"width": 1680, "height": 1080},
            device_scale_factor=2,
            accept_downloads=True,
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: console.append(message.text) if message.type == "error" else None)
        page.goto(isolated_html.as_uri(), wait_until="load", timeout=120000)
        print("QA: page loaded", flush=True)
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(900)

        if page.locator(".nav button").count() != 7:
            errors.append("Navigation does not contain seven views")
        pads = page.locator(".view").evaluate_all("es=>es.map(e=>({id:e.id,left:parseFloat(getComputedStyle(e).paddingLeft),right:parseFloat(getComputedStyle(e).paddingRight)}))")
        if len(pads) != 7 or any(x["left"] < 24 or x["right"] < 24 for x in pads):
            errors.append(f"Tab panels do not all have 24px horizontal padding: {pads}")
        if not page.evaluate("!!window.FG_BUNDLE && !!window.FG_ENGINE"):
            errors.append("Local model bundle or browser inference engine did not initialize")
        if os.environ.get("FINGUARD_QA_BOOTSTRAP") != "1" and page.locator("#integrityBadge").inner_text() != "16/16 integrity invariants | PASS":
            errors.append("Embedded integrity scorecard is not 16/16 PASS")
        if page.locator("#integrityPanel tbody tr").count() != 16:
            errors.append("Tab 07 does not show all 16 invariant rows")
        if page.locator("#integrityPanel tbody tr td:first-child").all_inner_texts() != [f"{i:02d}" for i in range(1, 17)]:
            errors.append("Tab 07 invariant row sequence is not 01 through 16")
        if page.locator("#integrityPanel th").all_inner_texts() != [
            "#", "Test invariant name", "Verified metric / tolerance", "Result"
        ]:
            errors.append("Tab 06 invariant table headers do not match the review contract")
        if page.locator("#integrityPanel").evaluate("e=>getComputedStyle(e.querySelector('.integrity-detail')).gridColumn") != "1 / -1":
            errors.append("Tab 06 verification table does not span the full width")
        integrity_layout = page.locator("#integrityPanel .integrity-detail .tablewrap").evaluate("e=>({maxHeight:getComputedStyle(e).maxHeight,scroll:e.scrollHeight,client:e.clientHeight,rows:[...e.querySelectorAll('tbody tr')].map(r=>({top:r.getBoundingClientRect().top,bottom:r.getBoundingClientRect().bottom}))})")
        if integrity_layout["maxHeight"] != "none":
            errors.append("Tab 06 invariant table has a scroll-height cutoff")
        if len(integrity_layout["rows"]) != 16:
            errors.append("Tab 07 does not expose all 16 rows in the expanded table")
        top_position = page.locator(".top").evaluate("e=>getComputedStyle(e).position")
        if top_position in ("sticky", "fixed"):
            errors.append(f"Top header is {top_position}; full-page captures could cut across content")

        for view in VIEWS:
            print(f"QA: checking {view}", flush=True)
            page.locator(f'[data-view="{view}"]').click()
            page.wait_for_timeout(250)
            page.evaluate("document.fonts.ready")
            state = page.evaluate("""(id)=>{
              const section=document.querySelector('#'+id);
              const nodes=[...section.querySelectorAll('.card,table,svg')].filter(e=>e.getBoundingClientRect().width>0);
              const overflow=nodes.filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>({tag:e.tagName,id:e.id||'',scroll:e.scrollWidth,client:e.clientWidth}));
              const svgs=[...section.querySelectorAll('svg')].filter(e=>e.getBoundingClientRect().width>0), overlaps=[];
              for(const svg of svgs){const t=[...svg.querySelectorAll('text')].map(e=>({text:e.textContent,r:e.getBoundingClientRect()})).filter(x=>x.r.width&&x.r.height);
                for(let i=0;i<t.length;i++)for(let j=i+1;j<t.length;j++){const a=t[i].r,b=t[j].r;if(a.left<b.right-1&&a.right>b.left+1&&a.top<b.bottom-1&&a.bottom>b.top+1)overlaps.push([t[i].text,t[j].text])}}
              const headers=[...section.querySelectorAll('th')].map(e=>({text:e.innerText,scroll:e.scrollWidth,client:e.clientWidth})),midword=[];
              for(const th of section.querySelectorAll('th')){const walker=document.createTreeWalker(th,NodeFilter.SHOW_TEXT);let node;while(node=walker.nextNode()){const re=/\\S+/g;let m;while(m=re.exec(node.textContent)){const range=document.createRange();range.setStart(node,m.index);range.setEnd(node,m.index+m[0].length);if(range.getClientRects().length>1)midword.push(m[0]);}}}
              return {overflow,overlaps,headers,midword,scrollWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth,text:section.innerText};
            }""", view)

            if state["overflow"]:
                errors.append(f"{view} card/table/SVG overflow: {state['overflow'][:8]}")
            if state["scrollWidth"] > state["clientWidth"] + 2:
                errors.append(f"{view} document horizontal overflow {state['scrollWidth']}>{state['clientWidth']}")
            if view in ("v5", "v6") and state["overlaps"]:
                errors.append(f"{view} overlapping SVG text: {state['overlaps'][:8]}")
            if view in ("v4", "v6"):
                clipped = [h for h in state["headers"] if h["scroll"] > h["client"] + 2]
                if clipped:
                    errors.append(f"{view} clipped table header cells: {clipped[:8]}")
            if state["midword"]:
                errors.append(f"{view} table headers split words across lines: {state['midword'][:8]}")
            for forbidden in FORBIDDEN_TEXT:
                if forbidden in state["text"]:
                    errors.append(f"{view} contains forbidden text {forbidden!r}")
            if re.search(r"\bMYR\b|\$\s?\d", state["text"], re.I):
                errors.append(f"{view} visible currency symbol/label found")
            report["tabs"][view] = {
                "overflow_count": len(state["overflow"]),
                "svg_text_overlaps": len(state["overlaps"]),
                "midword_header_breaks": len(state["midword"]),
                "table_header_count": len(state["headers"]),
                "clipped_header_count": sum(h["scroll"] > h["client"] + 2 for h in state["headers"]),
            }

            if view == "v5":
                migration_cells = page.locator("#migration tbody td").all_inner_texts()
                percent_cells = sum("%" in value for value in migration_cells)
                if percent_cells < 24:
                    errors.append(f"Tab 04 migration matrix has only {percent_cells} populated rate cells; expected at least 24")
                report["tabs"][view]["migration_percent_cells"] = percent_cells
                quarterly_rows = page.locator("#quarterlyCohorts tbody tr").count()
                quarterly_text = page.locator("#quarterlyCohorts").inner_text()
                if quarterly_rows < 8 or "2019-Q1" not in quarterly_text or "2020-Q4" not in quarterly_text:
                    errors.append("Quarterly origination matrix is missing its observed quarter/split cohorts")
                if re.search(r"\bMOB\b|months.on.book", quarterly_text, re.I):
                    errors.append("Quarterly matrix contains an unsupported MOB claim")
                report["tabs"][view]["quarterly_rows"] = quarterly_rows
                cohort_audit = page.evaluate("window.__quarterlyAudit")
                if cohort_audit != {"row_count": 8, "empty_rows": 0, "first": "2019-Q1", "last": "2020-Q4"}:
                    errors.append(f"Quarterly matrix integrity assertion failed: {cohort_audit}")
                if "0.0772" not in state["text"] or "0.0069" not in state["text"] or "0.5006" not in state["text"] or "0.5001" not in state["text"]:
                    errors.append("M4-NoDrift remediation proof does not expose reconciled PSI and KS values")
                for token in ("150,591", "926 F", "190 M", "0.60", "0.74"):
                    if token not in state["text"]:
                        errors.append(f"Fairness evidence missing {token}")

            if view == "v1":
                for token in ("151,707", "75.16%", "0.8832%", "140.13M"):
                    if token not in state["text"]:
                        errors.append(f"v1 missing verified KPI {token}")
                page.locator("#metric").select_option("stability")
                if page.locator("#timelineLegend").inner_text().count("Gini") != 2:
                    errors.append("Timeline KS/Gini mode does not render four series")
                page.locator("#metric").select_option("volume")
                if "Approval" not in page.locator("#timelineLegend").inner_text():
                    errors.append("Timeline volume mode lacks approval metric")
                page.locator("#metric").select_option("rate")

            if view == "v2":
                initial_dsr = page.locator("#dsrGauge").text_content() or ""
                placeholder_verified = False
                for index in (0, 1, 2, 4, 3):
                    page.locator("#casepills button").nth(index).click()
                    page.locator("#loadCase").click()
                    page.wait_for_timeout(70)
                    parity = page.evaluate("window.__lastParity")
                    label = page.evaluate("i=>D.cases[i].case", index)
                    if not parity:
                        errors.append(f"{label} did not expose Python parity")
                        continue
                    report["parity"].append({"case": label, **parity})
                    maximum = max(value for key, value in parity.items() if key.endswith("Abs"))
                    if maximum >= 1e-6 or not parity.get("passed"):
                        errors.append(f"{label} Python/browser parity error {maximum}")

                    if index == 3:
                        score = page.locator("#scoreGauge").text_content() or ""
                        dsr = page.locator("#dsrGauge").text_content() or ""
                        income = page.evaluate("D.cases[3].raw_features.maininc_215A")
                        if "759" not in score or "688" in score:
                            errors.append(f"Case 4 score gauge did not update to 759: {score}")
                        if dsr == initial_dsr or (income is None and "Unverified" not in dsr):
                            errors.append(f"Case 4 DSR gauge did not refresh to match the available income data: {dsr}")
                        if income is not None and ("PTI" not in dsr or "Total DSR" not in dsr):
                            errors.append(f"Case 4 PTI/DSR labels are missing: {dsr}")
                        requested = page.locator("#liveWaterfall .case-step strong").filter(has_text="Requested").bounding_box()
                        if overlaps(requested, page.locator("#recourseCard").bounding_box()) or overlaps(requested, page.locator("#parityCard").bounding_box()):
                            errors.append("Requested waterfall label overlaps recourse card or parity badge")
                        if page.locator("#parityBadge").inner_text() != "Python Parity PASS · Exact Match (< 1e-9)" or "parity-fail" in (page.locator("#parityBadge").get_attribute("class") or ""):
                            errors.append("Case 4 freshly loaded spotlight is not green exact-match Python parity")
                        steps = page.locator("#liveWaterfall .case-step").all_inner_texts()
                        if len(steps) != 4 or "95,993" not in steps[1] or "95,993" not in steps[2] or not re.search(r"(?:^|\n)0\s*\nActive Final Limit(?:\D|$)", steps[3]):
                            errors.append(f"Case 4 live challenger waterfall does not match Stage 4: {steps}")
                        if "Champion shadow limit: 95,993" not in page.locator("#liveWaterfall").inner_text():
                            errors.append("Case 4 waterfall is missing its 95,993 champion shadow limit")
                        if "Stage 4 shadow limit 95,993" not in page.locator("#liveout").inner_text() or "127,990" in page.locator("#liveout").inner_text():
                            errors.append("Case 4 spotlight limit does not match the 95,993 Stage 4 artifact")
                    # Spotlight 3 (case index 2) has a confirmed missing income field.
                    # Some other missing model fields have no editable control by design.
                    if index == 2:
                        placeholder_verified = True
                        if page.locator('#income[placeholder="Income Unverified"]').count() != 1:
                            errors.append(f"{label} missing income lacks the explicit placeholder")

                before = page.locator("#liveout").inner_text()
                page.locator("#f_bureau_active_max_dpd").fill("90")
                after = page.locator("#liveout").inner_text()
                if before == after:
                    errors.append("What-If input did not update live dual-model results")
                page.locator("#f_bureau_active_max_dpd").fill(str(page.evaluate("D.scorecard.defaults.bureau_active_max_dpd")))
                if page.locator("#liveWaterfall .case-step").count() != 4:
                    errors.append("Live four-bar waterfall missing")
                if "Actionable recourse" not in page.locator("#recourse").inner_text():
                    errors.append("Live actionable recourse card is empty")
                if not placeholder_verified:
                    errors.append("QA did not verify the missing-input placeholder on a spotlight with missing values")
                page.locator("details.technical-accordion").evaluate("e=>e.open=true")
                page.locator('[data-api-mode="json"]').click()
                page.wait_for_timeout(150)
                api_payload = page.locator("#apiJsonTrace").inner_text()
                if not page.locator("#apiTrace").is_visible() or '"features"' not in api_payload:
                    errors.append("API JSON inspector did not render its live request payload")
                if '"monthly_income": null' not in api_payload or '"pti": null' not in api_payload or '"total_dsr": null' not in api_payload:
                    errors.append("Unverified income was not kept null in the API trace")
                if index == 3 and ('"observed_annuity": 9066' not in api_payload):
                    errors.append("Case 4 observed annuity 9,066 was not preserved in API trace")
                report["tabs"][view]["api_json_payload_chars"] = len(api_payload)

            if view == "v3":
                if "151,707" not in state["text"] or page.locator("#workflowDag .branch-node").count() != 4:
                    errors.append("Workflow DAG does not expose the full OOT_Monitor pipeline")
                if page.locator("#riskMatrix tbody tr").count() != 5:
                    errors.append("2D policy matrix is missing its ten risk/affordability cells")
                report["tabs"][view]["decision_matrix_cells"] = page.locator("#riskMatrix tbody tr").count()
                counts=page.locator("#workflowDag [data-count]").evaluate_all("es=>es.map(e=>Number(e.dataset.count))")
                if counts != [9928,141779,27758,114021] or not (151707 > counts[0] and counts[1] > counts[2] and counts[1] == counts[2]+counts[3]):
                    errors.append(f"Workflow branching counts do not reconcile monotonically: {counts}")
                if "affordability ceiling" not in state["text"] or "1.02" not in state["text"]:
                    errors.append("R02 affordability-ceiling MRM explanation is missing")
                if page.locator("#trafficShare").count():
                    errors.append("Retrospective Challenger traffic-share slider remains visible")

            if view == "v4":
                swap_headers = page.locator("#swapMatrix thead th")
                expected_headers = ["Policy", "Swap Segment", "Apps", "Bads", "Obs. Bad %", "Champ. ENP", "Chall. ENP", "Net ENP Delta"]
                if swap_headers.all_inner_texts() != expected_headers:
                    errors.append(f"Swap table headers are not the expected 8-column layout: {swap_headers.all_inner_texts()}")
                if swap_headers.evaluate_all("es=>es.some(e=>getComputedStyle(e).whiteSpace!=='nowrap'||e.scrollHeight>e.clientHeight+2)"):
                    errors.append("Tab 03 swap table header wraps or clips a word")
                rows = page.locator("#swapMatrix tbody tr")
                if rows.count() != 8:
                    errors.append(f"Expected eight 2x2 swap-set rows, found {rows.count()}")
                for mode in ("Iso-Approval", "Iso-Risk"):
                    counts = []
                    for row in rows.all():
                        cells = row.locator("td").all_inner_texts()
                        if cells and cells[0] == mode:
                            counts.append(int(cells[2].replace(",", "")))
                    if sum(counts) != 151707:
                        errors.append(f"{mode} swap-set counts do not reconcile to 151,707: {sum(counts)}")
                page.locator("#isoRiskTarget").fill("1.0")
                page.locator("#solveCutoffs").click()
                if "Valid frontier gap" not in page.locator("#solverResult").inner_text():
                    errors.append("Valid-only risk-appetite solver did not return feasible operating points")

            if view == "v7":
                for token in ("0.21 s / 333.9 MB", "16.73 s / 542.0 MB", "8.82 s / 491.5 MB", "train_base.parquet", "Malaysia Production & Regulatory Mapping", "MFRS 9 ECL bridge"):
                    if token not in state["text"]:
                        errors.append(f"Tab 07 missing Step 1/regulatory evidence {token}")
                if page.locator("#sqlTabs button").count() != 4:
                    errors.append("SQL lineage viewer missing source files")
                if "Release Build: FinGuard Risk OS v2.0 · Stage 6 Blind-Verified" not in state["text"]:
                    errors.append("Tab 06 deterministic release metadata is missing")
                clip = page.locator("#integrityPanel .integrity-detail .tablewrap").evaluate("e=>({r:[...e.querySelectorAll('tbody tr')].map(x=>({top:x.getBoundingClientRect().top,bottom:x.getBoundingClientRect().bottom})),b:e.getBoundingClientRect()})")
                if not clip["r"] or clip["r"][0]["top"] < clip["b"]["top"] - 1 or clip["r"][-1]["bottom"] > clip["b"]["bottom"] + 1:
                    errors.append("Tab 06 invariant table rows are clipped vertically")

            if view == "v6":
                if page.locator("#woeFeature option").count() != 19:
                    errors.append("WOE Explorer does not list all 19 Champion features")
                report["tabs"][view]["woe_feature_count"] = page.locator("#woeFeature option").count()
                if "WOE = ln(%Good / %Bad)" not in state["text"] or "Pop %" not in page.locator("#woePopulation").text_content() or "Bad Rate %" not in page.locator("#woePopulation").text_content():
                    errors.append("WOE explorer is missing its formula or dual-axis labels")
                first = page.locator("#woeFeature").input_value()
                page.locator("#woeFeature").select_option(index=1)
                if page.locator("#woeFeature").input_value() == first or not page.locator("#woePopulation").locator("text").count():
                    errors.append("WOE feature selection did not redraw its empirical diagnostic")

        page.keyboard.press("Control+k")
        page.locator("#paletteSearch").fill("SQL 01")
        page.locator("#paletteResults button").first.click()
        if page.locator(".view.active").get_attribute("id") != "v7":
            errors.append("Ctrl+K palette failed to jump to SQL lineage")

        page.locator('[data-view="v2"]').click()
        page.locator("button", has_text="Open Decision Review & Audit Trace").click()
        page.wait_for_timeout(200)
        if page.locator("#memoContent .memo").count() != 1:
            errors.append("Underwriting memo did not render")
        memo_text = page.locator("#memoContent").inner_text()
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in memo_text:
                errors.append(f"Memo contains forbidden text {forbidden!r}")
        if "Active Challenger Policy Final Limit" not in memo_text or "CASE ID" not in memo_text:
            errors.append("Formal memo is missing the structured dual-model decision fields")
        if page.locator(".memo-banner").count() != 1 or page.locator(".memo-meta > div").count() != 5:
            errors.append("Formal memo is missing its committee banner or three-column metadata")
        if "SOURCE MODE" not in memo_text.upper() or "INPUT SNAPSHOT SHA-256" not in memo_text.upper() or re.search(r"\b\d{1,2} [A-Z][a-z]{2} 20\d{2}\b", memo_text):
            errors.append("Formal memo metadata is not deterministic")
        if page.locator("#memoModal .memo-modal-actions #memoPrint").count() != 1 or page.locator("#memoModal .memo-modal-actions #memoClose").count() != 1:
            errors.append("Memo print and close controls are not in the header action bar")
        memo_actions_above = page.evaluate("(()=>{const a=document.querySelector('#memoModal .memo-modal-actions')?.getBoundingClientRect(),c=document.querySelector('#memoContent')?.getBoundingClientRect();return !!a&&!!c&&a.top<c.top})()")
        if not memo_actions_above:
            errors.append("Memo header action bar is clipped below its content")

        # Enforce the QA gate before writing any of the seven screenshot files.
        if errors or console:
            report["js_errors"] = errors
            report["console_errors"] = console
            report["status"] = "FAIL"
            (ROOT / "reports" / "finguard_browser_qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            browser.close()
            print(json.dumps(report, indent=2, ensure_ascii=False))
            raise SystemExit(1)

        memo_path = OUT / "modal_underwriting_memo.png"
        page.locator("#memoModal .modal-panel").screenshot(path=str(memo_path), animations="disabled")
        report["memo_screenshot"] = {"path": str(memo_path), "bytes": memo_path.stat().st_size}
        page.locator("#memoClose").click()

        for view, name in list(VIEWS.items())[2:]:
            page.locator(f'[data-view="{view}"]').click()
            page.wait_for_timeout(250)
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(100)
            path = OUT / name
            page.screenshot(path=str(path), full_page=True, animations="disabled", timeout=120000)
            report["tabs"][view]["screenshot"] = str(path)
            report["tabs"][view]["bytes"] = path.stat().st_size
            report["screenshots"].append(str(path))

        report["js_errors"] = errors
        report["console_errors"] = console
        report["status"] = "PASS"
        (ROOT / "reports" / "finguard_browser_qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        browser.close()
        isolated.cleanup()


if __name__ == "__main__":
    main()
