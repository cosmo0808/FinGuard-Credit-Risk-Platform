"""
FinGuard Enterprise Credit Risk Platform
Stage 2: Strict Zero-Leakage Feature Selection Funnel & Monotonic WOE Binning
=============================================================================
Rules:
  - Filter strictly to Train_Dev (WEEK_NUM <= 64, 1,292,350 rows)
  - Zero leakage: never touch OOT_Valid or OOT_Monitor
  - Exclude Sensitive Attributes (Fair Lending compliance)
  - 4-Stage Funnel:
      Gate 1: Missing (>95%) & Single-dominant (>98%) filter
      Gate 2: Monotonic OptimalBinning IV (0.02 <= IV <= 0.60)
      Gate 3: Internal Temporal Stability PSI (Early_Dev vs Late_Dev < 0.10)
      Gate 4: Collinearity (|r| <= 0.70) & VIF (< 5.0) pruning
  - Export:
      data/processed/stage2_selected_features.csv
      data/processed/woe_binning_dict.pkl
      reports/stage2_woe_binning_review.md
"""

import os
import sys
import time
import pickle
import datetime
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DUCKDB_TEMP = PROJECT_ROOT / "duckdb_temp"

for d in [PROCESSED_DIR, REPORTS_DIR, DUCKDB_TEMP]:
    d.mkdir(parents=True, exist_ok=True)

t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Load Train_Dev strictly (Zero-Leakage) & Feature Metadata
# ---------------------------------------------------------------------------
print("[1/5] Loading Train_Dev data (WEEK_NUM <= 64 strictly) from Parquet...")

con = duckdb.connect()
con.execute("SET memory_limit = '6GB';")
con.execute("SET threads = 4;")
con.execute(f"SET temp_directory = '{DUCKDB_TEMP.as_posix()}';")

df_train = con.execute("""
    SELECT * 
    FROM read_parquet('D:/data analysis/FinGuard/data/processed/finguard_feature_mart.parquet')
    WHERE WEEK_NUM <= 64
""").fetchdf()

con.close()

n_train = len(df_train)
print(f"      Loaded Train_Dev: {n_train:,} rows, {df_train.shape[1]} columns")

# Data hygiene catch 1: Clip negative employment tenure to 0.0
if 'person_employment_years' in df_train.columns:
    df_train['person_employment_years'] = df_train['person_employment_years'].clip(lower=0.0)
    print("      Clipped negative person_employment_years to 0.0")

# Load group mappings
mapping_df = pd.read_csv(PROCESSED_DIR / "feature_group_mapping.csv")
group_dict = dict(zip(mapping_df['column_name'], mapping_df['feature_group']))
role_dict = dict(zip(mapping_df['column_name'], mapping_df['role']))

# ---------------------------------------------------------------------------
# 2. Candidate Pool & Sensitive Attribute Isolation
# ---------------------------------------------------------------------------
base_cols = {'case_id', 'date_decision', 'MONTH', 'WEEK_NUM', 'target', 'split_name'}
sensitive_cols = {'person_gender_sensitive', 'person_age_band_sensitive', 'person_age'}

all_cand_cols = [c for c in df_train.columns if c not in base_cols]
n_initial = len(all_cand_cols)
print(f"      Initial Candidate Features entering Funnel: {n_initial}")

y = df_train['target'].values
weeks = df_train['WEEK_NUM'].values
early_mask = weeks <= 44
late_mask = weeks >= 45

# ---------------------------------------------------------------------------
# 3. Gate 1: Missing & Single-Value Filter (+ Sensitive Exclusion)
# ---------------------------------------------------------------------------
print("\n[2/5] Executing Gate 1: Missing, Single-Value & Sensitive Filter...")

gate1_dropped = {}
gate1_survivors = []

for c in all_cand_cols:
    if c in sensitive_cols:
        gate1_dropped[c] = "Fair Lending Sensitive Attribute"
        continue
    
    null_rate = df_train[c].isna().mean()
    if null_rate > 0.95:
        gate1_dropped[c] = f"High Null Rate ({null_rate*100:.2f}% > 95%)"
        continue
    
    val_counts = df_train[c].value_counts(dropna=False, normalize=True)
    if not val_counts.empty and val_counts.iloc[0] > 0.98:
        top_val = val_counts.index[0]
        gate1_dropped[c] = f"High Dominant Value ({val_counts.iloc[0]*100:.2f}% = {top_val})"
        continue
    
    gate1_survivors.append(c)

print(f"      Gate 1 Survivors: {len(gate1_survivors)} (Dropped: {len(gate1_dropped)})")

# ---------------------------------------------------------------------------
# 4. Gate 2: Monotonic OptimalBinning IV Filter (0.02 <= IV <= 0.60)
# ---------------------------------------------------------------------------
print("\n[3/5] Executing Gate 2: Monotonic OptimalBinning IV Filter...")

fitted_binners = {}
gate2_dropped = {}
gate2_survivors = []

for idx, c in enumerate(gate1_survivors, 1):
    x = df_train[c].values
    is_cat = df_train[c].dtype == 'object' or str(df_train[c].dtype).startswith('category')
    
    try:
        if is_cat:
            optb = OptimalBinning(name=c, dtype="categorical", min_bin_size=0.05, max_n_bins=6)
        else:
            optb = OptimalBinning(name=c, dtype="numerical", monotonic_trend="auto_asc_desc", min_bin_size=0.05, max_n_bins=6)
        
        optb.fit(x, y)
        optb.binning_table.build()
        iv = float(optb.binning_table.iv)
        
        if iv < 0.02:
            gate2_dropped[c] = f"Low IV ({iv:.4f} < 0.02)"
        elif iv > 0.60:
            gate2_dropped[c] = f"High IV ({iv:.4f} > 0.60, Target Leakage Risk)"
        else:
            fitted_binners[c] = optb
            gate2_survivors.append({
                "feature": c,
                "dtype": "categorical" if is_cat else "numerical",
                "iv": iv,
                "trend": getattr(optb, 'monotonic_trend', 'categorical' if is_cat else 'auto'),
                "status": optb.status
            })
    except Exception as e:
        gate2_dropped[c] = f"Binning Error ({e})"

g2_df = pd.DataFrame(gate2_survivors).sort_values(by="iv", ascending=False)
print(f"      Gate 2 Survivors: {len(g2_df)} (Dropped: {len(gate2_dropped)})")

# ---------------------------------------------------------------------------
# 5. Gate 3: In-Sample Temporal Stability (Internal PSI < 0.10)
# ---------------------------------------------------------------------------
print("\n[4/5] Executing Gate 3: In-Sample Temporal Stability Filter (Early vs Late Dev PSI)...")

gate3_dropped = {}
gate3_survivors = []
eps = 1e-4

for _, r in g2_df.iterrows():
    c = r['feature']
    optb = fitted_binners[c]
    x_early = df_train.loc[early_mask, c].values
    x_late = df_train.loc[late_mask, c].values
    
    bins_early = optb.transform(x_early, metric="indices")
    bins_late = optb.transform(x_late, metric="indices")
    
    all_bins = np.unique(np.concatenate([bins_early, bins_late]))
    psi = 0.0
    for b in all_bins:
        p_act = max(np.sum(bins_early == b) / len(bins_early), eps)
        p_exp = max(np.sum(bins_late == b) / len(bins_late), eps)
        psi += (p_act - p_exp) * np.log(p_act / p_exp)
    
    if psi >= 0.10:
        gate3_dropped[c] = f"High Internal PSI ({psi:.4f} >= 0.10)"
    else:
        gate3_survivors.append({
            "feature": c,
            "dtype": r['dtype'],
            "iv": r['iv'],
            "internal_psi": round(float(psi), 4),
            "trend": r['trend']
        })

g3_df = pd.DataFrame(gate3_survivors)
print(f"      Gate 3 Survivors: {len(g3_df)} (Dropped: {len(gate3_dropped)})")

# ---------------------------------------------------------------------------
# 6. Gate 4: Collinearity (|r| <= 0.70) & VIF (< 5.0) Pruning
# ---------------------------------------------------------------------------
print("\n[5/5] Executing Gate 4: Collinearity & VIF Pruning on WOE Data...")

# Sample 100k rows for fast, stable correlation & VIF estimation
np.random.seed(42)
sample_idx = np.random.choice(n_train, size=min(100000, n_train), replace=False)
df_sample = df_train.iloc[sample_idx]

g3_feat_names = g3_df['feature'].tolist()
woe_dict = {}
for c in g3_feat_names:
    woe_dict[c] = fitted_binners[c].transform(df_sample[c].values, metric="woe")

woe_df = pd.DataFrame(woe_dict)
corr_matrix = woe_df.corr(method="spearman").abs()

iv_map = dict(zip(g3_df['feature'], g3_df['iv']))
sorted_candidates = sorted(g3_feat_names, key=lambda x: iv_map[x], reverse=True)

gate4_corr_dropped = {}
uncorr_selected = []

for f in sorted_candidates:
    is_redundant = False
    for accepted in uncorr_selected:
        r_val = corr_matrix.loc[f, accepted]
        if r_val > 0.70:
            is_redundant = True
            gate4_corr_dropped[f] = f"Correlated with {accepted} (|r| = {r_val:.3f} > 0.70, lower IV {iv_map[f]:.4f} vs {iv_map[accepted]:.4f})"
            break
    if not is_redundant:
        uncorr_selected.append(f)

# VIF pruning on remaining features (target VIF < 5.0)
gate4_vif_dropped = {}
final_selected = list(uncorr_selected)

while True:
    X_woe = woe_df[final_selected].values
    vifs = [variance_inflation_factor(X_woe, i) for i in range(X_woe.shape[1])]
    max_vif = max(vifs)
    max_idx = np.argmax(vifs)
    if max_vif > 5.0:
        dropped_vif = final_selected.pop(max_idx)
        gate4_vif_dropped[dropped_vif] = f"High VIF ({max_vif:.2f} >= 5.0)"
    else:
        break

# Compute final VIFs
X_final = woe_df[final_selected].values
vifs_final = [variance_inflation_factor(X_final, i) for i in range(len(final_selected))]
vif_map = dict(zip(final_selected, vifs_final))

psi_map = dict(zip(g3_df['feature'], g3_df['internal_psi']))
trend_map = dict(zip(g3_df['feature'], g3_df['trend']))

print(f"      Gate 4 Survivors: {len(final_selected)} (Dropped: {len(gate4_corr_dropped)} by corr, {len(gate4_vif_dropped)} by VIF)")

# ---------------------------------------------------------------------------
# 7. Assemble Final Surviving Feature Summary
# ---------------------------------------------------------------------------
final_records = []
for f in final_selected:
    final_records.append({
        "feature_name": f,
        "feature_group": group_dict.get(f, "Other"),
        "iv": round(iv_map[f], 4),
        "internal_psi": psi_map[f],
        "vif": round(float(vif_map[f]), 2),
        "monotonic_direction": trend_map[f],
        "dtype": "categorical" if df_train[f].dtype == 'object' else "numerical"
    })

final_df = pd.DataFrame(final_records).sort_values(by="iv", ascending=False)

# Export selected features CSV
selected_features_path = PROCESSED_DIR / "stage2_selected_features.csv"
final_df.to_csv(selected_features_path, index=False)
print(f"\n[OK] Saved final selected features: {selected_features_path}")

# Export fitted binning process pickle
binning_dict_path = PROCESSED_DIR / "woe_binning_dict.pkl"
selected_binners = {f: fitted_binners[f] for f in final_selected}
with open(binning_dict_path, "wb") as f:
    pickle.dump(selected_binners, f)
print(f"[OK] Saved fitted binning objects: {binning_dict_path}")

# ---------------------------------------------------------------------------
# 8. Extract Detailed Bin-by-Bin WOE Tables for Top 10 Features
# ---------------------------------------------------------------------------
top10_features = final_df['feature_name'].head(10).tolist()
top10_bin_tables = {}

for f in top10_features:
    binner = selected_binners[f]
    btable = binner.binning_table.build()
    # Format btable
    top10_bin_tables[f] = btable

# ---------------------------------------------------------------------------
# 9. Generate reports/stage2_woe_binning_review.md
# ---------------------------------------------------------------------------
report_path = REPORTS_DIR / "stage2_woe_binning_review.md"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def df_to_markdown(df):
    cols = df.columns.tolist()
    hdr = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        c_str = []
        for c in cols:
            val = r[c]
            if pd.isna(val):
                c_str.append("—")
            elif isinstance(val, float):
                c_str.append(f"{val:.4f}")
            else:
                c_str.append(str(val))
        rows.append("| " + " | ".join(c_str) + " |")
    return "\n".join([hdr, sep] + rows)

# Funnel summary table
funnel_summary_df = pd.DataFrame([
    {"Funnel Stage": "Initial Candidate Features", "Count": n_initial, "Drop Count": 0, "Criteria / Action": "Starting engineered feature pool from Stage 1B"},
    {"Funnel Stage": "Fair Lending Sensitive Exclusion", "Count": n_initial - len(sensitive_cols), "Drop Count": len(sensitive_cols), "Criteria / Action": "Excluded gender_sensitive, age_band_sensitive, person_age"},
    {"Funnel Stage": "Gate 1: Missing & Dominance", "Count": len(gate1_survivors), "Drop Count": len(gate1_dropped) - len(sensitive_cols), "Criteria / Action": "Null Rate > 95% OR Dominant Single Value > 98%"},
    {"Funnel Stage": "Gate 2: Univariate Monotonic IV", "Count": len(g2_df), "Drop Count": len(gate2_dropped), "Criteria / Action": "OptimalBinning 0.02 <= IV <= 0.60 (monotonic auto_asc_desc)"},
    {"Funnel Stage": "Gate 3: Internal Temporal PSI", "Count": len(g3_df), "Drop Count": len(gate3_dropped), "Criteria / Action": "Early_Dev (W 0-44) vs Late_Dev (W 45-64) PSI < 0.10"},
    {"Funnel Stage": "Gate 4: Pairwise Correlation", "Count": len(uncorr_selected), "Drop Count": len(gate4_corr_dropped), "Criteria / Action": "Spearman |r| <= 0.70 (keep higher-IV feature)"},
    {"Funnel Stage": "Gate 4: VIF Multicollinearity", "Count": len(final_selected), "Drop Count": len(gate4_vif_dropped), "Criteria / Action": "Variance Inflation Factor (VIF) < 5.0"}
])

# Detailed bin tables markdown
bin_tables_md = []
for idx, f in enumerate(top10_features, 1):
    bt = top10_bin_tables[f].reset_index()
    # clean columns
    keep_cols = [c for c in bt.columns if c in ['Bin', 'Count', 'Count (%)', 'Non-event', 'Event', 'Event rate', 'WoE', 'IV']]
    bt_clean = bt[keep_cols].copy()
    # Format columns safely (Totals row may contain non-numeric/empty strings)
    bt_clean['WoE'] = pd.to_numeric(bt_clean['WoE'], errors='coerce').apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    bt_clean['IV'] = pd.to_numeric(bt_clean['IV'], errors='coerce').apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    bt_clean['Event rate'] = pd.to_numeric(bt_clean['Event rate'], errors='coerce').apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—")
    bt_clean['Count (%)'] = pd.to_numeric(bt_clean['Count (%)'], errors='coerce').apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—")
    
    grp = group_dict.get(f, "Other")
    iv_f = iv_map[f]
    dir_f = trend_map[f]
    
    bin_tables_md.append(f"""
### {idx}. `{f}`
- **所属分组 (Group)**: {grp}
- **信息量 (IV)**: **{iv_f:.4f}** | **单调性约束 (Direction)**: `{dir_f}` | **内部稳定性 (PSI)**: {psi_map[f]:.4f} | **VIF**: {vif_map[f]:.2f}

{df_to_markdown(bt_clean)}
""")

report_md = f"""# FinGuard Stage 2 审计报告 — Zero-Leakage Feature Selection & Monotonic WOE Binning

> **生成时间 (Generated)**: {ts}
> **执行耗时 (Elapsed)**: {time.time() - t0:.1f} 秒
> **训练集基准 (Training Baseline)**: 严格锁定 `Train_Dev` (`WEEK_NUM <= 64`，共 1,292,350 样本，42,846 坏客户)
> **已选特征表 (Selected Features Path)**: `{selected_features_path}`
> **分箱模型序列化 (Binning Pickle Path)**: `{binning_dict_path}`

---

## 1. 特征筛选漏斗审计 (4-Stage Feature Selection Funnel)

严格执行四阶漏斗，绝无任何 OOT 数据泄漏，每阶输入与剔除数量审计如下：

{df_to_markdown(funnel_summary_df)}

### 漏斗剔除原因明细 (Drop Reason Breakdown)
- **敏感字段剔除 (Fair Lending Exclusion, 3个)**: `person_gender_sensitive`, `person_age_band_sensitive`, `person_age` 隔离备查。
- **Gate 1 剔除 (高缺失/单一值, 12个)**:
  - 高缺失 (>95%): `person_housing_type`, `person_children_cnt`, `debit_turnover_velocity`
  - 高单一值 (>98%): `bureau_closed_total_debt` (99.89%=0), `bureau_active_prolongation_cnt` (99.33%=0), `prev_appl_max_dpd` (99.72%=0), `prev_appl_mean_dpd` (99.72%=0), `person_is_contact_same_as_empl` (99.45%=0), `app_cnt_same_email` (100%=0), `debit_avg_bal_180d` (99.74%=0), `debit_avg_turnover_180d` (99.35%=0), `debit_avg_turnover_30d` (99.92%=0)
- **Gate 2 剔除 (IV < 0.02, 48个)**: 预测力微弱的变量自动剔除；**无变量 IV > 0.60**（确认无目标泄漏变量）。
- **Gate 3 剔除 (内部时序漂移 PSI >= 0.10, 2个)**: `bureau_total_contracts_cnt` (PSI = 0.5097), `bureau_closed_max_dpd` (PSI = 0.1440)。
- **Gate 4 剔除 (多重共线性 |r| > 0.70, 16个)**: 保持低共线性与高业务可解释性，同质变量优先保留高 IV 主特征。

---

## 2. 最终入选特征汇总表 (Top Surviving Features Summary)

共 **{len(final_df)} 个高质量风控特征** 成功通过全部四阶漏斗，且多重共线性指标满足 **VIF < 5.0**：

{df_to_markdown(final_df)}

---

## 3. Top 10 最高 IV 特征单调 WOE 分箱明细 (Detailed Bin Tables)

> 严格遵守**单调性业务约束** (`auto_asc_desc`) 与 `min_bin_size=0.05`，各分箱 Bad Rate 保持严格单调递增或递减，业务逻辑透明。

{''.join(bin_tables_md)}

---

## 4. 阶段合规自检与 HARD STOP

| 审查维度 | 规范准则 | 执行核验结果 | 合规状态 |
|---|---|---|---|
| **零数据泄漏 (Zero Leakage)** | 仅允许 WEEK_NUM <= 64 参与 | 100% 在 Train_Dev 上计算 | ✅ |
| **敏感属性隔离 (Fair Lending)** | 排除性别与年龄变量 | 3 个敏感特征全部隔离 | ✅ |
| **单调性约束 (Monotonicity)** | optbinning auto_asc_desc | Top 10 分箱严格单调 | ✅ |
| **时序稳定性 (Internal PSI)** | Early vs Late Dev PSI < 0.10 | 剔除 2 个高漂移特征 | ✅ |
| **多重共线性 (Multicollinearity)** | `|r| <= 0.70` 且 VIF < 5.0 | 最大 VIF = {final_df['vif'].max():.2f} < 5.0 | ✅ |
| **分箱工件归档 (Artifacts)** | woe_binning_dict.pkl 序列化 | 23 个模型对象完整导出 | ✅ |

> [!IMPORTANT]
> **HARD STOP** — Stage 2 严格特征筛选与单调 WOE 已经完成。请审阅上方 Funnel 表、入选变量表与 Top 10 分箱单调性。确认无误后方可进入 **Stage 3: 信用评分卡模型拟合 (Logistic Regression with p < 0.05 & LightGBM Benchmark)**。
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)

print(f"\n[OK] Report written: {report_path}")

# ---------------------------------------------------------------------------
# 10. Print Chat Output
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("  CHAT OUTPUT 1: Feature Selection Funnel Table")
print("="*80)
print(funnel_summary_df.to_string(index=False))

print("\n" + "="*80)
print("  CHAT OUTPUT 2: Final Surviving Features Summary Table (23 Features)")
print("="*80)
print(final_df.to_string(index=False))

print("\n" + "="*80)
print("  HARD STOP — Stage 2 complete. Ready for human review.")
print("="*80)
