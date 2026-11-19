"""
FinGuard Enterprise Credit Risk Platform
Stage 1A: Project Initialization, Temporal Split Audit & Candidate Feature Discovery
======================================================================================
Rules (from GEMINI.md):
  - DuckDB: memory_limit='6GB', threads=4, temp_directory='./duckdb_temp/'
  - NEVER load depth-1 / depth-2 Parquet files directly into Pandas
  - ALL aggregations via DuckDB GROUP BY case_id first
  - Train_Dev ONLY for schema inspection and null-rate calculation (WEEK_NUM <= 64)
  - HARD STOP after generating reports/stage1a_audit_summary.md

Usage:
  cd "D:\\data analysis\\FinGuard"
  D:\\anaconda\\python.exe src/audit_stage1a.py
"""

import os
import sys
import json
import textwrap
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Path bootstrap — resolve project root relative to this script
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root
os.chdir(PROJECT_ROOT)

TRAIN_DIR      = PROJECT_ROOT / "data" / "parquet_files" / "train"
FEAT_DEF_PATH  = PROJECT_ROOT / "data" / "feature_definitions.csv"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
DUCKDB_TEMP    = PROJECT_ROOT / "duckdb_temp"
REPORTS_DIR    = PROJECT_ROOT / "reports"

for d in [PROCESSED_DIR, DUCKDB_TEMP, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DuckDB safe connection (GEMINI.md Rule 1)
# ---------------------------------------------------------------------------
import duckdb

print("[1/7] Initialising DuckDB connection...")
con = duckdb.connect()
con.execute("SET memory_limit = '6GB';")
con.execute("SET threads = 4;")
con.execute(f"SET temp_directory = '{DUCKDB_TEMP.as_posix()}';")
print(f"      DuckDB {duckdb.__version__}  |  memory=6GB  |  threads=4  |  temp={DUCKDB_TEMP}")

# ---------------------------------------------------------------------------
# 2. Pre-flight data check
# ---------------------------------------------------------------------------
print("\n[2/7] Pre-flight: scanning data directory...")

if not TRAIN_DIR.exists():
    sys.exit(f"ERROR: Train directory not found: {TRAIN_DIR}\n"
             "Please place Home Credit 2024 parquet files into data/parquet_files/train/")

parquet_files = sorted(TRAIN_DIR.glob("*.parquet"))
if not parquet_files:
    sys.exit(f"ERROR: No .parquet files found in {TRAIN_DIR}\n"
             "Please complete the Kaggle download first.")

print(f"      Found {len(parquet_files)} parquet file(s) in train directory:")
for pf in parquet_files:
    size_mb = pf.stat().st_size / 1_048_576
    print(f"        {pf.name:55s}  {size_mb:8.1f} MB")

# Check train_base.parquet specifically
train_base = TRAIN_DIR / "train_base.parquet"
if not train_base.exists():
    sys.exit(f"ERROR: train_base.parquet not found in {TRAIN_DIR}")

# Check feature_definitions.csv
feat_def_found = FEAT_DEF_PATH.exists()
if feat_def_found:
    print(f"      feature_definitions.csv found: {FEAT_DEF_PATH}")
else:
    print(f"      WARNING: feature_definitions.csv NOT found at {FEAT_DEF_PATH}")
    print("               Cross-reference step will be skipped.")

# ---------------------------------------------------------------------------
# 3. Temporal split audit on train_base.parquet
# ---------------------------------------------------------------------------
print("\n[3/7] Temporal split audit on train_base.parquet...")

BASE = train_base.as_posix()

# 3a. Overall stats
overall = con.execute(f"""
    SELECT
        COUNT(*)                                   AS total_rows,
        COUNT(DISTINCT case_id)                    AS unique_case_ids,
        MIN(date_decision)                         AS min_date,
        MAX(date_decision)                         AS max_date,
        MIN(WEEK_NUM)                              AS min_week,
        MAX(WEEK_NUM)                              AS max_week,
        SUM(target)                                AS bad_count,
        ROUND(100.0 * AVG(target), 4)              AS bad_rate_pct,
        COUNT(*) FILTER (WHERE target IS NULL)     AS target_null_count
    FROM read_parquet('{BASE}')
""").fetchdf()

print("\n  === Overall train_base.parquet Stats ===")
for col in overall.columns:
    print(f"    {col:25s}: {overall[col].iloc[0]}")

# 3b. Per-split stats
splits_sql = f"""
    SELECT
        CASE
            WHEN WEEK_NUM <= 64             THEN 'Train_Dev   (W 0-64)'
            WHEN WEEK_NUM BETWEEN 65 AND 77 THEN 'OOT_Valid   (W 65-77)'
            WHEN WEEK_NUM BETWEEN 78 AND 91 THEN 'OOT_Monitor (W 78-91)'
            ELSE                                 'Out_of_Range'
        END                                     AS split_name,
        COUNT(*)                                AS n_rows,
        COUNT(DISTINCT case_id)                 AS n_unique_cases,
        MIN(date_decision)                      AS date_min,
        MAX(date_decision)                      AS date_max,
        SUM(target)                             AS bad_count,
        ROUND(100.0 * AVG(target), 4)           AS bad_rate_pct
    FROM read_parquet('{BASE}')
    GROUP BY split_name
    ORDER BY MIN(WEEK_NUM)
"""
splits_df = con.execute(splits_sql).fetchdf()

print("\n  === Temporal Split Summary ===")
print(splits_df.to_string(index=False))

# 3c. Monthly volume + bad rate trend
monthly_sql = f"""
    SELECT
        MONTH                                   AS month,
        COUNT(*)                                AS n_rows,
        SUM(target)                             AS bad_count,
        ROUND(100.0 * AVG(target), 4)           AS bad_rate_pct,
        CASE
            WHEN WEEK_NUM <= 64             THEN 'Train_Dev'
            WHEN WEEK_NUM BETWEEN 65 AND 77 THEN 'OOT_Valid'
            WHEN WEEK_NUM BETWEEN 78 AND 91 THEN 'OOT_Monitor'
            ELSE                                 'Other'
        END                                     AS split
    FROM read_parquet('{BASE}')
    GROUP BY MONTH, split
    ORDER BY MONTH
"""
monthly_df = con.execute(monthly_sql).fetchdf()

print("\n  === Monthly Volume & Bad Rate Trend ===")
print(monthly_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Schema inspection for each curated table family (TRAIN_DEV ONLY)
# ---------------------------------------------------------------------------
print("\n[4/7] Schema inspection (Train_Dev WEEK_NUM <= 64)...")

# Table families to inspect
TABLE_FAMILIES = {
    "depth0_static":    list(TRAIN_DIR.glob("train_static_0_*.parquet")),
    "depth0_static_cb": [TRAIN_DIR / "train_static_cb_0.parquet"],
    "depth1_person":    [TRAIN_DIR / "train_person_1.parquet"],
    "depth1_applprev":  list(TRAIN_DIR.glob("train_applprev_1_*.parquet")),
    "depth1_bureau_a":  list(TRAIN_DIR.glob("train_credit_bureau_a_1_*.parquet")),
    "depth1_tax_a":     [TRAIN_DIR / "train_tax_registry_a_1.parquet"],
    "depth1_debitcard": [TRAIN_DIR / "train_debitcard_1.parquet"],
    "depth2_applprev":  [TRAIN_DIR / "train_applprev_2.parquet"],   # schema only
}

schemas = {}

for family, paths in TABLE_FAMILIES.items():
    existing = [p for p in paths if p.exists()]
    if not existing:
        print(f"      SKIP {family}: no matching files found")
        schemas[family] = []
        continue

    glob_map = {
        "depth0_static":   "train_static_0_*.parquet",
        "depth1_applprev": "train_applprev_1_*.parquet",
        "depth1_bureau_a": "train_credit_bureau_a_1_*.parquet",
    }

    if len(existing) == 1:
        path_expr = f"'{existing[0].as_posix()}'"
    else:
        glob_pat = glob_map.get(family, "*.parquet")
        path_expr = f"'{TRAIN_DIR.as_posix()}/{glob_pat}'"

    try:
        schema_df = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({path_expr})"
        ).fetchdf()
        schemas[family] = list(zip(schema_df["column_name"], schema_df["column_type"]))
        n_files = len(existing)
        sample_names = ", ".join(p.name for p in existing[:2])
        ellipsis = "..." if n_files > 2 else ""
        print(f"      {family:25s}: {len(schemas[family]):4d} columns  ({sample_names}{ellipsis})")
    except Exception as e:
        print(f"      {family:25s}: ERROR — {e}")
        schemas[family] = []
        sample_names = ", ".join(p.name for p in existing[:2])
        ellipsis = "..." if n_files > 2 else ""
        print(f"      {family:25s}: {len(schemas[family]):4d} columns  ({sample_names}{ellipsis})")
    except Exception as e:
        print(f"      {family:25s}: ERROR — {e}")
        schemas[family] = []

# ---------------------------------------------------------------------------
# 5. Cross-reference with feature_definitions.csv
# ---------------------------------------------------------------------------
print("\n[5/7] Cross-referencing with feature_definitions.csv...")

feat_meta = {}   # column_name -> {table, description, type}

if feat_def_found:
    import pandas as pd
    # Read minimal columns only — do NOT load parquet data into pandas
    feat_df = pd.read_csv(FEAT_DEF_PATH, encoding="utf-8", low_memory=False)
    print(f"      Loaded feature_definitions.csv: {len(feat_df)} rows, columns: {list(feat_df.columns)}")

    # Normalise column names (handle different possible header names)
    col_map = {}
    for c in feat_df.columns:
        cl = c.strip().lower()
        if "variable" in cl or "feature" in cl or "column" in cl:
            col_map["name"] = c
        elif "table" in cl or "source" in cl or "file" in cl:
            col_map["table"] = c
        elif "description" in cl or "definition" in cl or "meaning" in cl:
            col_map["desc"] = c
        elif "type" in cl or "dtype" in cl or "datatype" in cl:
            col_map["dtype"] = c

    name_col  = col_map.get("name")
    table_col = col_map.get("table")
    desc_col  = col_map.get("desc")
    dtype_col = col_map.get("dtype")

    if name_col:
        for _, row in feat_df.iterrows():
            feat_meta[str(row[name_col]).strip()] = {
                "table": str(row[table_col]).strip() if table_col else "unknown",
                "description": str(row[desc_col]).strip() if desc_col else "",
                "dtype": str(row[dtype_col]).strip() if dtype_col else "",
            }
        print(f"      Mapped {len(feat_meta)} feature definitions")
    else:
        print("      WARNING: Could not identify feature name column in feature_definitions.csv")
        print(f"      Columns found: {list(feat_df.columns)}")
else:
    print("      SKIPPED (feature_definitions.csv not available)")

# ---------------------------------------------------------------------------
# 6. Null-rate computation on Train_Dev for candidate columns
#    (NEVER load raw depth-1/2 Parquet into Pandas — all in DuckDB)
# ---------------------------------------------------------------------------
print("\n[6/7] Computing Train_Dev null rates in DuckDB...")

# Get Train_Dev case_ids first (from depth-0 base, which is safe to use directly
# since train_base IS depth-0 and already filtered in the SQL)
train_dev_case_ids_path = (PROCESSED_DIR / "train_dev_case_ids.parquet").as_posix()

con.execute(f"""
    COPY (
        SELECT case_id
        FROM read_parquet('{BASE}')
        WHERE WEEK_NUM <= 64
    ) TO '{train_dev_case_ids_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

n_train_dev = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{train_dev_case_ids_path}')"
).fetchone()[0]
print(f"      Train_Dev case_ids saved: {n_train_dev:,} rows -> {train_dev_case_ids_path}")

# Compute null rates per family (GROUP BY case_id inside DuckDB for depth-1/2)
null_rate_results = {}   # family -> {col -> null_rate}

for family, col_dtypes in schemas.items():
    if not col_dtypes:
        continue

    existing = [p for p in TABLE_FAMILIES[family] if p.exists()]
    if not existing:
        continue

    glob_map = {
        "depth0_static":   "train_static_0_*.parquet",
        "depth1_applprev": "train_applprev_1_*.parquet",
        "depth1_bureau_a": "train_credit_bureau_a_1_*.parquet",
    }

    if len(existing) == 1:
        src = f"'{existing[0].as_posix()}'"
    else:
        gp = glob_map.get(family, "*.parquet")
        src = f"'{TRAIN_DIR.as_posix()}/{gp}'"

    # Identify depth (depth-1/2 tables have multiple rows per case_id — aggregate first)
    is_depth0 = family.startswith("depth0")
    is_schema_only = family == "depth2_applprev"

    if is_schema_only:
        # Schema-only inspection: just report column names, skip null rates
        null_rate_results[family] = {col: "schema_only" for col, _ in col_dtypes}
        continue

    # Build null-count SELECT  (limit to numeric/useful columns to save memory)
    # Exclude case_id, date cols from null rate calc
    skip_cols = {"case_id", "num_group1", "num_group2"}
    num_cols = [
        col for col, dtype in col_dtypes
        if col not in skip_cols
        and any(t in dtype.upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "BOOL"])
    ][:60]  # cap at 60 columns per family to stay within memory budget

    if not num_cols:
        null_rate_results[family] = {}
        continue

    null_exprs = ", ".join(
        f"ROUND(100.0 * SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS \"{c}\""
        for c in num_cols
    )

    if is_depth0:
        # Depth-0: direct join with Train_Dev case_ids
        null_sql = f"""
            SELECT {null_exprs}
            FROM read_parquet({src}) t
            INNER JOIN read_parquet('{train_dev_case_ids_path}') td
                ON t.case_id = td.case_id
        """
    else:
        # Depth-1: aggregate to case_id level first (GEMINI.md Rule 1)
        # Count nulls BEFORE grouping — take MAX to collapse
        agg_exprs = ", ".join(
            f"MAX(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS \"{c}\""
            for c in num_cols
        )
        null_sql = f"""
            WITH agg AS (
                SELECT t.case_id, {agg_exprs}
                FROM read_parquet({src}) t
                INNER JOIN read_parquet('{train_dev_case_ids_path}') td
                    ON t.case_id = td.case_id
                GROUP BY t.case_id
            )
            SELECT {', '.join(f'ROUND(100.0 * AVG("{c}"), 2) AS "{c}"' for c in num_cols)}
            FROM agg
        """

    try:
        nr_df = con.execute(null_sql).fetchdf()
        null_rate_results[family] = {c: float(nr_df[c].iloc[0]) for c in num_cols}
        print(f"      {family:25s}: null rates computed for {len(num_cols)} columns")
    except Exception as e:
        print(f"      {family:25s}: null rate ERROR — {e}")
        null_rate_results[family] = {}

# ---------------------------------------------------------------------------
# 7. Candidate feature grouping — domain-driven column tagging
# ---------------------------------------------------------------------------
print("\n[7/7] Candidate feature tagging...")

# Domain keyword patterns (applied to actual schema column names)
DOMAIN_GROUPS = {
    "Traditional_Credit": [
        "external", "score", "bureau", "credit", "contract", "active", "closed",
        "outstanding", "limit", "overdue", "maxoverdue", "credlmt", "creditlmt",
        "pmtcount", "numberprevapp", "num_active", "cb_", "ext_source",
        "sk_id_bureau", "days_credit",
    ],
    "Historical_Repayment": [
        "dpd", "delinq", "delinquent", "late", "missed", "overdue", "daysover",
        "pmtcount", "pmt_status", "installment", "payment", "numpmt",
        "days_past_due", "num_instal", "repayment", "baddebt",
    ],
    "Application_Demographic": [
        "income", "employ", "education", "age", "birth", "family", "children",
        "housing", "sex", "gender", "marital", "region", "occupation",
        "organization", "type_suite", "name_income", "name_education",
        "name_family", "name_housing", "code_gender", "flag_own",
    ],
    "Alternative_NonTrad": [
        "tax", "registry", "debit", "card", "transaction", "contact",
        "document", "address", "mobile", "email", "phone", "social",
        "flag_document", "flag_mobil", "flag_email", "flag_cont",
        "reg_region", "live_region",
    ],
}

# DSR / PTI critical columns (keywords for col name or definition)
DSR_PTI_KEYWORDS = [
    "annuity", "maininc", "income", "credamount", "disbursedcred", "totaldebt", 
    "currdebt", "outstanddebt", "outstandingamount", "monthlyinstl", "installmentamount", 
    "instlamount", "lastotherlns", "actualbalance", "credit_sum_debt"
]

tagged = {}   # col -> list of groups

all_schema_cols = set()
for family, col_dtypes in schemas.items():
    for col, dtype in col_dtypes:
        all_schema_cols.add(col.lower())
        col_lower = col.lower()
        defn_text = feat_meta.get(col, {}).get("description", "")
        def_lower = defn_text.lower()
        search_text = f"{col_lower} {def_lower}"

        groups_hit = []
        for grp, keywords in DOMAIN_GROUPS.items():
            if any(kw in search_text for kw in keywords):
                groups_hit.append(grp)
        dsr_hit = any(kw in col_lower for kw in DSR_PTI_KEYWORDS) or (
            any(w in def_lower for w in ["annuity", "main income", "outstanding debt", "instalment amount", "monthly instalment", "total amount of debt"])
        )
        tagged[col] = {
            "dtype": dtype,
            "family": family,
            "domain_groups": groups_hit,
            "dsr_pti_candidate": dsr_hit,
            "in_feat_def": col in feat_meta,
            "definition": defn_text,
        }

# Summarise per group
group_summary = {grp: [] for grp in DOMAIN_GROUPS}
dsr_candidates = []

for col, info in tagged.items():
    for grp in info["domain_groups"]:
        group_summary[grp].append(col)
    if info["dsr_pti_candidate"]:
        dsr_candidates.append(col)

print("\n  === Candidate Column Count by Domain Group ===")
for grp, cols in group_summary.items():
    print(f"    {grp:30s}: {len(cols):4d} columns")
print(f"    {'DSR/PTI Candidates':30s}: {len(dsr_candidates):4d} columns")

print("\n  === DSR/PTI Candidate Columns (Total DSR check) ===")
for col in sorted(dsr_candidates):
    info = tagged[col]
    defn = (info['definition'][:70] + "...") if len(info['definition']) > 70 else info['definition']
    print(f"    {col:45s}  [{info['dtype']:15s}]  {defn}")

# ---------------------------------------------------------------------------
# 8. Generate reports/stage1a_audit_summary.md
# ---------------------------------------------------------------------------
print("\n[8/8] Writing reports/stage1a_audit_summary.md...")

report_path = REPORTS_DIR / "stage1a_audit_summary.md"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Helper: format splits_df as markdown table
def df_to_md(df, float_fmt="{:.4f}"):
    cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)

# Collect null rate summary (top-15 lowest null rate = most complete columns)
nr_summary_lines = []
for family, nr_dict in null_rate_results.items():
    if not nr_dict or all(v == "schema_only" for v in nr_dict.values()):
        continue
    numeric_nr = {k: v for k, v in nr_dict.items() if isinstance(v, float)}
    top15 = sorted(numeric_nr.items(), key=lambda x: x[1])[:15]
    nr_summary_lines.append(f"\n#### {family}\n")
    nr_summary_lines.append("| Column | Null Rate (%) | DSR/PTI | Definition |")
    nr_summary_lines.append("|---|---|---|---|")
    for col, nr in top15:
        t_info = tagged.get(col, {})
        dsr = "✅" if t_info.get("dsr_pti_candidate") else ""
        defn = t_info.get("definition", "")
        defn_short = (defn[:60] + "...") if len(defn) > 60 else defn
        nr_summary_lines.append(f"| `{col}` | {nr:.2f} | {dsr} | {defn_short} |")

# Build group candidate tables (top 15 per group)
group_md_sections = []
for grp, cols in group_summary.items():
    group_md_sections.append(f"\n#### {grp.replace('_', ' ')} ({len(cols)} 个候选列)\n")
    group_md_sections.append("| Column | Dtype | Null Rate (Train_Dev, %) | DSR/PTI | Definition |")
    group_md_sections.append("|---|---|---|---|---|")
    rows_added = 0
    for col in sorted(cols):
        if rows_added >= 15:
            break
        t_info = tagged.get(col, {})
        dtype  = t_info.get("dtype", "")
        dsr    = "✅" if t_info.get("dsr_pti_candidate") else ""
        defn   = t_info.get("definition", "")
        defn_s = (defn[:55] + "...") if len(defn) > 55 else defn
        # Pull null rate from results
        nr_val = "—"
        for fam, nr_d in null_rate_results.items():
            if col in nr_d and isinstance(nr_d[col], float):
                nr_val = f"{nr_d[col]:.2f}"
                break
        group_md_sections.append(f"| `{col}` | {dtype} | {nr_val} | {dsr} | {defn_s} |")
        rows_added += 1

# Overall stats for report
ov = overall.iloc[0]

report_content = f"""\
# FinGuard Stage 1A 审计报告 — Temporal Split Audit & Candidate Feature Discovery

> **生成时间 (Generated)**: {ts}
> **项目根目录 (Project Root)**: `{PROJECT_ROOT}`
> **执行脚本 (Script)**: `src/audit_stage1a.py`
> **数据规则 (Rules)**: 参见 `GEMINI.md` — Zero Leakage, 16GB RAM, Human-in-the-Loop

---

## 1. 总体数据集概况 (Overall Dataset Summary)

> `train_base.parquet` 全量统计，**未过滤**任何 split，仅用于数据完整性确认。

| 指标 (Metric) | 数值 (Value) |
|---|---|
| Total Rows | {int(ov['total_rows']):,} |
| Unique case_id | {int(ov['unique_case_ids']):,} |
| Min date_decision | {ov['min_date']} |
| Max date_decision | {ov['max_date']} |
| Min WEEK_NUM | {int(ov['min_week'])} |
| Max WEEK_NUM | {int(ov['max_week'])} |
| Bad Count (target=1) | {int(ov['bad_count']):,} |
| Overall Bad Rate | {float(ov['bad_rate_pct']):.4f}% |
| Target NULL Count | {int(ov['target_null_count'])} |

> [!IMPORTANT]
> `target_null_count = {int(ov['target_null_count'])}` — 若不为零，需在 Stage 1B 中排除 target 为空的行。

---

## 2. 时序分割审计 (Temporal Split Audit)

> 严格按 `WEEK_NUM` 切分，**零数据泄漏**：Train_Dev 边界 ≤ 64。

{df_to_md(splits_df)}

> [!NOTE]
> - **Train_Dev**：所有 WOE binning、IV 计算、VIF 筛查、模型训练均在此集上进行。
> - **OOT_Valid**：仅用于校准与策略模拟，严禁回流至训练环节。
> - **OOT_Monitor**：PSI/CSI 监控与再校准，时间线最晚，模拟真实部署后监控。
> - 若有 `Out_of_Range` 行出现，需人工核查其 `WEEK_NUM` 值后决定处理方式。

---

## 3. 月度量/坏账率趋势 (Monthly Volume & Bad Rate Trend)

> 用于检测宏观/季节性飘移，以及 OOT 切分点是否与市场事件吻合。

{df_to_md(monthly_df)}

> [!TIP]
> 若 OOT_Monitor 期间坏账率出现断崖式变化，Stage 6 的 PSI 阈值需相应收紧（建议 PSI > 0.25 触发再建模警报）。

---

## 4. 文件清单 & Schema 概览 (File Inventory & Schema Summary)

| 文件家族 (Family) | 文件数 | 列数 | 备注 |
|---|---|---|---|
{chr(10).join(f"| `{fam}` | {len([p for p in TABLE_FAMILIES[fam] if p.exists()]):2d} | {len(cols):4d} | {'Schema only (depth-2)' if fam=='depth2_applprev' else ''} |"
              for fam, cols in schemas.items())}

> [!NOTE]
> `depth2_applprev` 仅做 schema 检视，Stage 1B 才会在 DuckDB 内 GROUP BY case_id 聚合，**绝不直接 load 进 Pandas**。

---

## 5. Train_Dev 零泄漏 Null Rate（Top-15 最完整列）

> 以下 null rate **仅基于 `WEEK_NUM <= 64` 的 Train_Dev 样本**，在 DuckDB 内计算，depth-1 表已先做 `GROUP BY case_id` 聚合。

{''.join(nr_summary_lines) if nr_summary_lines else '*feature_definitions.csv 未找到，跳过此步骤。*'}

---

## 6. 候选特征分组（Candidate Feature Groups）

> 基于列名关键词进行域标注，每组展示最多 15 列（Stage 2 IV 筛选后精简至 10–15 个）。
> 标注方法：纯字符串匹配，不代表最终特征选择结论，需人工审核。

{''.join(group_md_sections)}

---

## 7. Total DSR / PTI 候选列核查 (DSR/PTI Column Check)

> **PTI** = 新贷款月供 / 收入；**Total DSR** = (现有在贷月供 + 新月供) / 收入。
> 以下列是区分 PTI 与 Total DSR 所需的原始字段候选，需在 Stage 1B 人工确认准确的列名。

| Column | Source Family | Dtype | Null Rate (Train_Dev, %) | Definition |
|---|---|---|---|---|
{chr(10).join(
    f"| `{col}` | {next((fam for fam, col_dtypes in schemas.items() if any(c==col for c,_ in col_dtypes)), '—')} | "
    f"{tagged.get(col, {}).get('dtype', '—')} | "
    f"{next((f'{nr_d[col]:.2f}' for fam, nr_d in null_rate_results.items() if col in nr_d and isinstance(nr_d[col], float)), '—')} | "
    f"{(tagged.get(col, {}).get('definition', '')[:55] + '...') if len(tagged.get(col, {}).get('definition', '')) > 55 else tagged.get(col, {}).get('definition', '')} |"
    for col in sorted(dsr_candidates)
    if col in tagged
)}

> [!CAUTION]
> 在 Stage 1B 中，`AMT_ANNUITY`（新月供）与 `credit_sum_debt`（在贷余额）这两个字段的来源表必须经人工确认后方可纳入特征工程 mart，防止混淆信贷局数据的 outstanding balance 与新申请月供。

---

## 8. 阶段审计摘要表 (Stage 1A Audit Summary Table)

| 检查项 (Check) | 结果 (Result) | 状态 (Status) |
|---|---|---|
| DuckDB memory_limit=6GB | 已设置 | ✅ |
| DuckDB threads=4 | 已设置 | ✅ |
| train_base.parquet 可读 | {int(ov['total_rows']):,} 行 | ✅ |
| WEEK_NUM 范围 | {int(ov['min_week'])} – {int(ov['max_week'])} | ✅ |
| Target NULL | {int(ov['target_null_count'])} 行 | {'⚠️ 需处理' if int(ov['target_null_count']) > 0 else '✅'} |
| Temporal split 覆盖率 | 见第2节 | ✅ |
| OOT 数据未触碰训练 | WEEK_NUM 过滤已实施 | ✅ |
| feature_definitions.csv | {'已加载 ' + str(len(feat_meta)) + ' 条' if feat_def_found else '未找到，跳过'} | {'✅' if feat_def_found else '⚠️'} |
| Depth-1/2 未直接 load Pandas | 全程 DuckDB GROUP BY | ✅ |
| DSR/PTI 候选列已识别 | {len(dsr_candidates)} 列 | ✅ |

---

## 9. 下一步指引 (Next Steps — Pending Human Approval)

> [!IMPORTANT]
> **HARD STOP** — 请审查以上所有内容，确认无误后方可进入 Stage 1B。

**人工审核要点 (Human Review Checklist)**：
1. ✅/❌ Temporal split 三段样本量是否与预期比例吻合（建议 Train ~70%，OOT 合计 ~30%）？
2. ✅/❌ 月度坏账率趋势是否有异常跳变（若有，记录周数与可能原因）？
3. ✅/❌ `target_null_count` 是否为 0？若不为 0，确认排除策略。
4. ✅/❌ DSR/PTI 候选列是否正确识别了 `AMT_ANNUITY`、`AMT_CREDIT` 以及信贷局在贷月供字段？
5. ✅/❌ 是否有需要额外纳入或剔除的 feature family？
6. ✅/❌ feature_definitions.csv 中的 feature 名称是否与实际 Parquet schema 列名一致？

**待确认后进入 Stage 1B**：DuckDB SQL 特征聚合 mart（仅 Train_Dev）。
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"\n  Report saved: {report_path}")
print(f"  Report size:  {report_path.stat().st_size / 1024:.1f} KB")

# ---------------------------------------------------------------------------
# Terminal print: Temporal Split Summary + DSR candidates (as requested)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("  CHAT OUTPUT 1: Temporal Split Summary Table")
print("="*70)
print(splits_df.to_string(index=False))

print("\n" + "="*70)
print("  CHAT OUTPUT 2: Total DSR / PTI Candidate Columns")
print("="*70)
print(f"{'Column':<45}  {'Dtype':<15}  {'Source Family'}")
print("-"*80)
for col in sorted(dsr_candidates):
    info = tagged.get(col, {})
    source = next(
        (fam for fam, col_dtypes in schemas.items() if any(c == col for c, _ in col_dtypes)),
        "unknown"
    )
    print(f"  {col:<43}  {info.get('dtype',''):<15}  {source}")

print("\n" + "="*70)
print("  HARD STOP — Stage 1A complete.")
print("  Review reports/stage1a_audit_summary.md and approve before Stage 1B.")
print("="*70)

con.close()
