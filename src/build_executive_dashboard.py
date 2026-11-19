"""Build FinGuard executive dashboard exports from the six approved Power BI marts only."""

from __future__ import annotations

from pathlib import Path
import html
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "powerbi"
OUT_DIR = ROOT / "dashboard" / "python_boards"
HTML_PATH = ROOT / "dashboard" / "FinGuard_Interactive_Command_Center.html"

FILES = {
    "temporal": "pbi_01_temporal_cohort_monitoring.csv",
    "cutoff": "pbi_02_cutoff_tradeoff_curve.csv",
    "concentration": "pbi_03_portfolio_concentration.csv",
    "stress": "pbi_04_stress_scenarios.csv",
    "psi_fairness": "pbi_05_psi_csi_and_fairness.csv",
    "migration": "pbi_06_delinquency_migration.csv",
}

NAVY = "#1B365D"
NAVY_2 = "#24476E"
TEAL = "#008080"
TEAL_DARK = "#006B6B"
BLUE = "#4C78A8"
GOLD = "#F2B84B"
CORAL = "#D9534F"
SLATE = "#627D98"
PALE = "#F4F8FB"
GRID = "#D9E2EC"
INK = "#243B53"
MUTED = "#627D98"
MODEL_COLORS = {"Champion Scorecard": BLUE, "Challenger LightGBM M4": TEAL_DARK}
SCENARIO_ORDER = ["S1_Baseline", "S2_Controlled_Growth", "S3_Mild_Downturn", "S4_Severe_Tail_Shock"]
SCENARIO_LABELS = {"S1_Baseline": "S1 Baseline", "S2_Controlled_Growth": "S2 Growth", "S3_Mild_Downturn": "S3 Downturn", "S4_Severe_Tail_Shock": "S4 Severe"}
TIER_ORDER = ["Tier A: Super Prime", "Tier B: Prime", "Tier C: Standard", "Tier D: High Risk", "Tier E: Decline"]
SEVERITY_ORDER = ["No prior application", "0 DPD", "1-10 DPD", "11-30 DPD", "31-89 DPD", "90+ DPD", "Prior contract: DPD unavailable"]


def load_marts() -> dict[str, pd.DataFrame]:
    """Read only the six approved pbi_01..pbi_06 summary CSVs."""
    tables = {}
    for key, filename in FILES.items():
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required summary mart missing: {path}")
        tables[key] = pd.read_csv(path)
    return tables


def fmt_m(value: float, _position: int | None = None) -> str:
    return f"{value / 1e3:.0f}k"


def prefer_all_rows(df: pd.DataFrame, grain_keys: list[str]) -> pd.DataFrame:
    """Use one ALL row per period/split when present; otherwise retain disjoint tier rows."""
    selected = []
    for _, group in df.groupby(grain_keys, sort=False, dropna=False):
        is_all = group["Risk_Tier"].astype(str).str.strip().str.upper().eq("ALL")
        selected.append(group.loc[is_all] if is_all.any() else group)
    return pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()


def build_frames(t: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | dict]:
    temporal = t["temporal"].copy()
    m4 = temporal.loc[temporal["Model"].eq("Challenger LightGBM M4")].copy()
    portfolio_all = temporal.loc[temporal["Risk_Tier"].eq("ALL_PORTFOLIO")].copy()
    portfolio = portfolio_all.loc[portfolio_all["Model"].eq("Challenger LightGBM M4")].copy()
    monthly = portfolio_all.loc[portfolio_all["Grain"].eq("Monthly")].copy()
    monthly["Month"] = pd.to_datetime(monthly["Period"].astype(str), format="%Y%m")
    monthly = monthly.rename(columns={"KS": "KS_Portfolio", "Gini": "Gini_Portfolio"})

    weekly_rows = portfolio.loc[portfolio["Grain"].eq("Weekly")].copy()
    weekly = weekly_rows.rename(columns={"KS": "KS_Portfolio", "Gini": "Gini_Portfolio"}).copy()
    weekly["Week"] = pd.to_numeric(weekly["Period"])

    cutoff = t["cutoff"].copy()
    cutoff = cutoff.loc[cutoff["Split"].eq("OOT_Monitor")].sort_values("Target_Eligible_Approval_Pct")

    concentration = t["concentration"].copy()
    segment_rows = concentration.loc[concentration["Dimension"].eq("Portfolio_Segment")].copy()
    mode_summary = segment_rows.groupby(["Model", "Mode"], as_index=False).agg(
        Approved_Cases=("Approved_Count", "sum"),
        Recommended_Volume=("Total_Recommended_Volume", "sum"),
        Bad_Count_Proxy=("Observed_Bad_Rate_Pct", lambda s: np.nan),
    )
    bad_weighted = segment_rows.assign(_bad_weight=segment_rows["Observed_Bad_Rate_Pct"] * segment_rows["Approved_Count"]).groupby(["Model", "Mode"], as_index=False).agg(
        _bad_weight=("_bad_weight", "sum"), _n=("Approved_Count", "sum"))
    mode_summary = mode_summary.drop(columns="Bad_Count_Proxy").merge(bad_weighted, on=["Model", "Mode"])
    mode_summary["Observed_Bad_Rate_Pct"] = mode_summary["_bad_weight"] / mode_summary["_n"]

    # Each case appears in one disjoint concentration cell; shares are recorded once per tier/driver.
    binding = concentration.loc[concentration["Dimension"].eq("Tier_x_Binding_Driver") & concentration["Mode"].eq("Iso-Risk")].copy()

    stress = t["stress"].loc[t["stress"]["Risk_Tier"].eq("Portfolio Total")].copy()
    stress = stress.loc[stress["Scenario"].isin(SCENARIO_ORDER)]
    stress["Scenario_Label"] = stress["Scenario"].map(SCENARIO_LABELS)

    psi = t["psi_fairness"].copy()
    csi = psi.loc[psi["Record_Type"].eq("Feature CSI") & psi["Split"].eq("Train_Dev vs OOT_Monitor")].copy()
    csi["Metric"] = pd.to_numeric(csi["Metric"], errors="coerce")
    csi = csi.nlargest(15, "Metric").sort_values("Metric")
    age = psi.loc[psi["Record_Type"].eq("Fairness") & psi["Split"].eq("OOT_Monitor") & psi["Group"].isin(["<25", "25-34", "35-49", "50+"]) & psi["Model"].eq("Challenger LightGBM M4")].copy()
    age["Group"] = pd.Categorical(age["Group"], ["<25", "25-34", "35-49", "50+"], ordered=True)
    age = age.sort_values("Group")

    migration = t["migration"].loc[t["migration"]["Model"].eq("Challenger LightGBM M4")].copy()
    migration["Risk_Tier"] = pd.Categorical(migration["Risk_Tier"], TIER_ORDER, ordered=True)
    migration["Prior_Delinquency_Severity"] = pd.Categorical(migration["Prior_Delinquency_Severity"], SEVERITY_ORDER, ordered=True)
    migration = migration.sort_values(["Risk_Tier", "Prior_Delinquency_Severity"])

    # Portfolio KPIs derive from the M4 weekly case counts, once across its mutually exclusive risk tiers.
    split_kpis = weekly.groupby("Split", as_index=False).agg(Applications=("Applications", "sum"), Approvals=("Approvals", "sum"))
    split_kpis["Bad_Rate_Pct"] = weekly_rows.groupby("Split").apply(
        lambda g: 100 * g["Bads"].sum() / g["Applications"].sum(), include_groups=False).reindex(split_kpis["Split"]).to_numpy()
    champion = mode_summary.loc[mode_summary["Model"].eq("Challenger LightGBM M4") & mode_summary["Mode"].eq("Iso-Risk")].iloc[0]
    baseline = stress.loc[stress["Model"].eq("Challenger LightGBM M4") & stress["Scenario"].eq("S1_Baseline")].iloc[0]
    monitor_apps = float(split_kpis.set_index("Split").loc["OOT_Monitor", "Applications"])
    kpis = {"applications": monitor_apps,
            "iso_risk_approval_pct": 100 * float(champion["Approved_Cases"]) / monitor_apps,
            "iso_risk_bad_pct": float(champion["Observed_Bad_Rate_Pct"]),
            "iso_risk_enp_m": float(baseline["Model_Implied_Expected_Net_Profit"]) / 1e6}
    return {"monthly": monthly, "weekly": weekly, "cutoff": cutoff, "mode_summary": mode_summary,
            "binding": binding, "stress": stress, "csi": csi, "age": age, "migration": migration,
            "split_kpis": split_kpis, "kpis": kpis}


def style_mpl(fig: plt.Figure, title: str, subtitle: str, page: int) -> None:
    fig.patch.set_facecolor(PALE)
    fig.text(0.04, 0.958, "FIN GUARD  /  EXECUTIVE RISK INTELLIGENCE", color=TEAL_DARK, fontsize=10, fontweight="bold")
    fig.text(0.04, 0.916, title, color=NAVY, fontsize=25, fontweight="bold")
    fig.text(0.04, 0.882, subtitle, color=MUTED, fontsize=10.5)
    fig.text(0.96, 0.958, f"PAGE 0{page}  ·  03", color=SLATE, fontsize=9, ha="right", fontweight="bold")


def style_ax(ax: plt.Axes, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor("white")
    ax.set_title(title, loc="left", color=NAVY, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.85)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)


def save_page(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUT_DIR / filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


def page1(frames: dict) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=120, layout=None)
    style_mpl(fig, "Portfolio pulse", "Monthly performance, booking activity and monthly ranking stability  |  2019–2020", 1)
    k=frames["kpis"]
    kpis = [
        ("OOT_MONITOR APPLICATIONS", f"{k['applications']:,.0f}", "Weeks 78–91"),
        ("ISORISK APPROVAL RATE", f"{k['iso_risk_approval_pct']:.2f}%", "Frozen policy replay"),
        ("ISORISK BAD RATE", f"{k['iso_risk_bad_pct']:.4f}%", "Observed Monitor outcomes"),
        ("ISORISK SIMULATED ENP", f"{k['iso_risk_enp_m']:.2f}M", "S1 model-implied profit"),
    ]
    for i, (label, value, foot) in enumerate(kpis):
        x = 0.04 + i * 0.235
        fig.text(x, 0.825, label, color=SLATE, fontsize=8, fontweight="bold")
        fig.text(x, 0.778, value, color=NAVY, fontsize=22, fontweight="bold")
        fig.text(x, 0.748, foot, color=MUTED, fontsize=8)
    ax1 = fig.add_axes([0.055, 0.46, 0.43, 0.23])
    style_ax(ax1, "Monthly actual bad rate vs recalibrated PD", "Percent")
    for split, color in [("Train_Dev", BLUE), ("OOT_Valid", GOLD), ("OOT_Monitor", TEAL_DARK)]:
        g = frames["monthly"].loc[frames["monthly"]["Model"].eq("Challenger LightGBM M4") & frames["monthly"]["Split"].eq(split)]
        if not g.empty:
            ax1.plot(g["Month"], g["Actual_Bad_Rate_Pct"], color=color, lw=1.8, marker="o", ms=3, label=f"Actual · {split}")
            ax1.plot(g["Month"], g["Recalibrated_PD_Pct"], color=color, lw=1.35, ls="--", alpha=.85, label=f"Recal PD · {split}")
    ax1.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax1.tick_params(axis="x", rotation=25)
    ax2 = fig.add_axes([0.535, 0.46, 0.41, 0.23])
    style_ax(ax2, "Weekly volume and approvals", "Applications")
    colors = {"Train_Dev": BLUE, "OOT_Valid": GOLD, "OOT_Monitor": TEAL_DARK}
    for split, g in frames["weekly"].groupby("Split"):
        ax2.plot(g["Week"], g["Applications"], color=colors[split], lw=1.25, alpha=.48)
        ax2.plot(g["Week"], g["Approvals"], color=colors[split], lw=1.8, label=f"{split} approvals")
    ax2.yaxis.set_major_formatter(FuncFormatter(fmt_m))
    ax2.legend(frameon=False, fontsize=7, ncol=2)
    ax3 = fig.add_axes([0.055, 0.12, 0.89, 0.23])
    style_ax(ax3, "Monthly OOT_Monitor portfolio discrimination", "Full-cohort metric")
    for model,color in [("Champion Scorecard",BLUE),("Challenger LightGBM M4",TEAL_DARK)]:
        g=frames["monthly"].loc[frames["monthly"]["Model"].eq(model) & frames["monthly"]["Split"].eq("OOT_Monitor")].sort_values("Month")
        if not g.empty:
            label="Champion" if model=="Champion Scorecard" else "LightGBM M4"
            ax3.plot(g["Month"],g["KS_Portfolio"],color=color,lw=1.8,marker="o",ms=3,label=f"KS · {label}")
            ax3.plot(g["Month"],g["Gini_Portfolio"],color=color,lw=1.4,ls="--",marker="s",ms=2.5,label=f"Gini · {label}")
    ax3.tick_params(axis="x",rotation=20); ax3.legend(frameon=False,ncol=4,fontsize=7,loc="upper left")
    fig.text(.055,.055,"Source: Power BI marts. ALL_PORTFOLIO rows use the full cohort for rates, KS and Gini.",color=MUTED,fontsize=8)
    save_page(fig,"exec_page1_portfolio_pulse.png")


def page2(frames: dict) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=120)
    style_mpl(fig, "Strategy & stress", "Operating-point trade-offs and assumption-based portfolio economics", 2)
    ax1 = fig.add_axes([.06, .56, .42, .27]); style_ax(ax1, "OOT_Monitor cutoff: approval vs observed bad rate", "Observed bad rate (%)")
    for (model, split), g in frames["cutoff"].groupby(["Model", "Split"]):
        ax1.plot(g["Approval_Rate_Pct"], g["Observed_Bad_Rate_Pct"], color=MODEL_COLORS[model], ls="-" if split == "OOT_Monitor" else "--", lw=1.8, label=f"{model.replace(' LightGBM M4',' M4')} · {split}")
    ax1.set_xlabel("Approval rate (%)", color=MUTED); ax1.legend(frameon=False, fontsize=7)
    modes = frames["mode_summary"].copy()
    ax2 = fig.add_axes([.55, .56, .39, .27]); style_ax(ax2, "Iso-Approval vs Iso-Risk · Monitor replay", "Approved applications")
    labels = ["Champion\nIso-Approval", "Champion\nIso-Risk", "M4\nIso-Approval", "M4\nIso-Risk"]
    order = [("Champion Scorecard", "Iso-Approval"), ("Champion Scorecard", "Iso-Risk"), ("Challenger LightGBM M4", "Iso-Approval"), ("Challenger LightGBM M4", "Iso-Risk")]
    m = modes.set_index(["Model", "Mode"])
    vals = [m.loc[k, "Approved_Cases"] for k in order]
    cols = [BLUE, BLUE, TEAL, TEAL]
    bars = ax2.bar(range(4), vals, color=cols, width=.64)
    ax2.set_xticks(range(4), labels, fontsize=8); ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for b, k in zip(bars, order): ax2.text(b.get_x()+b.get_width()/2, b.get_height()+900, f"{m.loc[k,'Recommended_Volume']/1e9:.2f}B", ha="center", fontsize=7, color=INK)
    ax3 = fig.add_axes([.07, .13, .86, .29]); style_ax(ax3, "S1–S4 portfolio profit comparison · Monitor", "Currency Units (M)")
    stress = frames["stress"].copy()
    x = np.arange(len(SCENARIO_ORDER)); width = .18
    for j, model in enumerate(["Champion Scorecard", "Challenger LightGBM M4"]):
        g = stress.loc[stress["Model"].eq(model)].set_index("Scenario").reindex(SCENARIO_ORDER)
        base=(j*2-1.5)*width
        expected=ax3.bar(x+base,g["Model_Implied_Expected_Net_Profit"]/1e6,width,color=MODEL_COLORS[model],label=f"{model} · expected")
        proxy=ax3.bar(x+base+width,g["Realized_Net_Profit_Stress_Proxy"]/1e6,width,color=MODEL_COLORS[model],alpha=.45,hatch="//",label=f"{model} · stress proxy")
        for idx, label in [(0, "140.1M"), (3, "51.0M" if model=="Champion Scorecard" else "42.5M")]:
            if (idx==0 and model=="Challenger LightGBM M4") or idx==3:
                b=expected[idx]; ax3.annotate(label,(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,3),textcoords="offset points",ha="center",fontsize=7,color=INK,fontweight="bold")
    ax3.set_xticks(x, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax3.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,1.30),ncol=4,fontsize=7)
    ax3.text(.01, -.23, "Expected bars use model PD. Hatched bars are stress-scaled observed-default proxies, not realized downturn P&L.", transform=ax3.transAxes, fontsize=8, color=MUTED)
    save_page(fig,"exec_page2_strategy_stress.png")


def page3(frames: dict) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=120)
    style_mpl(fig, "Exposure, stability & fairness", "Decision drivers, feature drift, prior-delinquency migration and age-band diagnostics", 3)
    ax_top=fig.add_axes([.055,.58,.89,.22]); style_ax(ax_top,"Iso-Risk binding-driver share by model and risk tier","Share (%)")
    driver_colors={"Affordability":TEAL_DARK,"Requested Amount":BLUE,"Risk_Behavior":GOLD,"Unverified_Income_Cap":CORAL}
    xpositions=np.arange(8)
    xlabels=[f"{tier.replace('Tier ','')}\n{('Champion' if model=='Champion Scorecard' else 'M4')}" for model in ["Champion Scorecard","Challenger LightGBM M4"] for tier in TIER_ORDER[:4]]
    bottoms=np.zeros(8)
    for driver,color in driver_colors.items():
        values=[]
        for model in ["Champion Scorecard","Challenger LightGBM M4"]:
            for tier in TIER_ORDER[:4]:
                rows=frames["binding"].loc[frames["binding"]["Model"].eq(model)&frames["binding"]["Risk_Tier"].eq(tier)&frames["binding"]["Binding_Driver"].eq(driver),"Binding_Driver_Share_Pct"]
                values.append(float(rows.iloc[0]) if not rows.empty else 0.0)
        ax_top.bar(xpositions,values,bottom=bottoms,color=color,width=.68,label=driver)
        bottoms+=np.asarray(values)
    for idx,total in enumerate(bottoms):
        if total==0: ax_top.text(idx,2,"No\nportfolio",ha="center",va="bottom",fontsize=6,color=MUTED)
    ax_top.set_xticks(xpositions,xlabels,fontsize=7); ax_top.set_ylim(0,100); ax_top.set_xlabel("")
    ax_top.legend(frameon=False,fontsize=7,ncol=4,loc="upper center",bbox_to_anchor=(.5,-.20))
    ax2=fig.add_axes([.085,.12,.25,.32]); style_ax(ax2,"Top 15 feature CSI","CSI")
    csi=frames["csi"]
    c= [CORAL if v>.25 else GOLD if v>=.10 else TEAL_DARK for v in csi["Metric"]]
    labels=csi["Feature"].replace({
        "has_tax_registry_record":"tax_registry",
        "person_education_level":"education_level",
        "tax_verified_income_ratio":"tax_income_ratio",
        "person_family_status":"family_status",
        "person_employment_years":"employment_years",
        "person_income_type":"income_type",
        "bureau_queries_120d":"queries_120d",
        "bureau_queries_30d":"queries_30d",
        "prev_appl_total_pmt_count":"prior_pmt_count",
        "static_cnt_pmts_months_24m":"paid_months_24m",
        "static_num_inst_paid_early":"paid_early_count",
        "static_avg_dpd_tol_closure24m":"avg_dpd_24m",
        "static_days_overdue_tol":"days_overdue",
        "static_avg_dbd_dpd_last24m":"avg_dbd_24m",
        "static_last_rejected_amount":"last_rejected_amount",
        "static_pct_inst_paid_late_10d":"paid_late_10d",
        "prev_appl_avg_tenor":"prior_avg_tenor",
        "app_cnt_last_30d":"apps_last_30d",
    })
    ax2.barh(labels,csi["Metric"],color=c)
    ax2.axvline(.10,color=GOLD,ls="--",lw=1); ax2.axvline(.25,color=CORAL,ls="--",lw=1)
    ax2.tick_params(axis="y",labelsize=6.2); ax2.text(.99,.98,"0.10 / 0.25 references",transform=ax2.transAxes,ha="right",va="top",fontsize=6.3,color=MUTED)
    ax3=fig.add_axes([.385,.12,.32,.32]); style_ax(ax3,"Prior DPD → current default","Prior delinquency severity")
    mig=frames["migration"].copy(); matrix=mig.pivot(index="Risk_Tier",columns="Prior_Delinquency_Severity",values="Current_Default_Rate_Pct").reindex(index=TIER_ORDER,columns=SEVERITY_ORDER)
    arr=matrix.to_numpy(dtype=float); cmap=plt.get_cmap("YlOrRd").copy(); cmap.set_bad("#E8EDF2")
    image=ax3.imshow(np.ma.masked_invalid(arr),aspect="auto",cmap=cmap,vmin=0,vmax=np.nanpercentile(arr,90))
    ax3.set_xticks(range(len(matrix.columns)),[str(x).replace("Prior contract: ","").replace("No prior application","No prior app") for x in matrix.columns],rotation=32,ha="right",fontsize=6.5)
    ax3.set_yticks(range(len(matrix.index)),[str(x).replace("Tier ","") for x in matrix.index],fontsize=7)
    for yi in range(arr.shape[0]):
        for xi in range(arr.shape[1]):
            if np.isfinite(arr[yi,xi]): ax3.text(xi,yi,f"{arr[yi,xi]:.1f}%",ha="center",va="center",fontsize=6,color="white" if arr[yi,xi] > np.nanpercentile(arr,65) else INK)
            else: ax3.text(xi,yi,"—",ha="center",va="center",fontsize=7,color=SLATE)
    cb=fig.colorbar(image,ax=ax3,fraction=.035,pad=.02); cb.set_label("Observed default rate (%)",fontsize=7); cb.ax.tick_params(labelsize=6)
    ageax=fig.add_axes([.75,.12,.205,.32]); style_ax(ageax,"Age AIR vs bad rate")
    age=frames["age"].sort_values("Group").copy(); ypos=np.arange(len(age))
    ageax.barh(ypos,age["AIR"],color=TEAL_DARK,alpha=.88,height=.58,label="AIR")
    ageax.axvline(.80,color=CORAL,ls="--",lw=1,label="AIR 0.80")
    ageax.set_yticks(ypos,age["Group"].astype(str),fontsize=6.5); ageax.invert_yaxis(); ageax.set_xlim(0,1.12); ageax.set_xlabel("AIR",fontsize=7); ageax.set_ylabel("")
    rateax=ageax.twiny(); rateax.plot(age["Observed_Bad_Rate_Pct"],ypos,color=GOLD,marker="o",lw=1.4,label="Observed Bad Rate %")
    rateax.set_xlim(0,8); rateax.set_xlabel("Observed Bad Rate %",fontsize=7,labelpad=2); rateax.tick_params(colors=MUTED,labelsize=6)
    for yi,(_,r) in enumerate(age.iterrows()):
        if str(r["Group"]) in ("<25","50+"):
            rateax.annotate(f"{r['Observed_Bad_Rate_Pct']:.2f}%",(r["Observed_Bad_Rate_Pct"],yi),xytext=(4,0),textcoords="offset points",va="center",fontsize=6.2,color=INK)
    fig.text(.055,.065,"AIR is diagnostic only. Prior-DPD table is cross-sectional, not a monthly roll-rate/MOB curve.",color=MUTED,fontsize=8)
    save_page(fig,"exec_page3_stability_fairness.png")


def plotly_theme(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(title={"text": title, "x": 0, "xanchor": "left"},
                      height=height, paper_bgcolor="white", plot_bgcolor="white",
                      font={"family":"Aptos, Segoe UI, sans-serif","color":INK,"size":12},
                      margin={"l":55,"r":25,"t":60,"b":55},
                      legend={"orientation":"h","y":-0.22},
                      hoverlabel={"bgcolor":NAVY,"font":{"color":"white"}})
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def interactive_figures(frames: dict) -> list[tuple[str,str,go.Figure]]:
    figs=[]
    # Page 1: monthly quality, weekly flows, stability.
    f=go.Figure()
    for split,color in [("Train_Dev",BLUE),("OOT_Valid",GOLD),("OOT_Monitor",TEAL_DARK)]:
        g=frames["monthly"].loc[frames["monthly"]["Split"].eq(split)]
        f.add_trace(go.Scatter(x=g["Month"],y=g["Actual_Bad_Rate_Pct"],mode="lines+markers",name=f"Actual · {split}",line={"color":color,"width":2},hovertemplate="%{x|%b %Y}<br>Actual: %{y:.2f}%<extra></extra>"))
        f.add_trace(go.Scatter(x=g["Month"],y=g["Recalibrated_PD_Pct"],mode="lines",name=f"Recal PD · {split}",line={"color":color,"dash":"dash","width":1.6},hovertemplate="%{x|%b %Y}<br>Recal PD: %{y:.2f}%<extra></extra>"))
    figs.append(("p1","Monthly actual bad rate vs recalibrated PD",plotly_theme(f,"Monthly actual bad rate vs recalibrated PD")))
    f=go.Figure()
    for split,color in [("Train_Dev",BLUE),("OOT_Valid",GOLD),("OOT_Monitor",TEAL_DARK)]:
        g=frames["weekly"].loc[frames["weekly"]["Split"].eq(split)]
        f.add_trace(go.Scatter(x=g["Week"],y=g["Applications"],mode="lines",name=f"Applications · {split}",line={"color":color,"width":1.4,"dash":"dot"},hovertemplate="Week %{x}<br>Applications: %{y:,}<extra></extra>"))
        f.add_trace(go.Scatter(x=g["Week"],y=g["Approvals"],mode="lines",name=f"Approvals · {split}",line={"color":color,"width":2},hovertemplate="Week %{x}<br>Approvals: %{y:,}<extra></extra>"))
    figs.append(("p1","Weekly volume and approvals",plotly_theme(f,"Weekly volume and approvals")))
    f=go.Figure()
    monitor=frames["monthly"].loc[frames["monthly"]["Split"].eq("OOT_Monitor")]
    for model,color in [("Champion Scorecard",BLUE),("Challenger LightGBM M4",TEAL_DARK)]:
        g=monitor.loc[monitor["Model"].eq(model)].sort_values("Month")
        label="Champion" if model=="Champion Scorecard" else "LightGBM M4"
        f.add_trace(go.Scatter(x=g["Month"],y=g["KS_Portfolio"],mode="lines+markers",name=f"KS · {label}",line={"color":color,"width":2},hovertemplate="%{x|%b %Y}<br>Portfolio KS %{y:.3f}<extra></extra>"))
        f.add_trace(go.Scatter(x=g["Month"],y=g["Gini_Portfolio"],mode="lines+markers",name=f"Gini · {label}",line={"color":color,"width":1.5,"dash":"dash"},hovertemplate="%{x|%b %Y}<br>Portfolio Gini %{y:.3f}<extra></extra>"))
    figs.append(("p1","Monthly OOT_Monitor portfolio KS/Gini",plotly_theme(f,"Monthly OOT_Monitor portfolio KS/Gini · both models")))
    # Page 2: strategy and profit.
    f=go.Figure()
    for (model,split),g in frames["cutoff"].groupby(["Model","Split"]):
        f.add_trace(go.Scatter(x=g["Approval_Rate_Pct"],y=g["Observed_Bad_Rate_Pct"],mode="lines+markers",name=f"{model} · {split}",line={"color":MODEL_COLORS[model],"dash":"solid" if split=="OOT_Monitor" else "dash"},customdata=g[["PD_Cutoff","Approved_Count"]],hovertemplate="Approval %{x:.1f}%<br>Bad rate %{y:.2f}%<br>PD cutoff %{customdata[0]:.4f}<br>Approved %{customdata[1]:,.0f}<extra></extra>"))
    figs.append(("p2","Cutoff approval vs bad-rate trade-off",plotly_theme(f,"OOT_Monitor cutoff: approval vs observed bad rate")))
    m=frames["mode_summary"].copy(); m["Short_Model"]=m["Model"].map({"Champion Scorecard":"Champion","Challenger LightGBM M4":"LightGBM M4"})
    f=make_subplots(rows=1,cols=2,subplot_titles=("Approved applications","Recommended volume"))
    for mode,color in [("Iso-Approval",BLUE),("Iso-Risk",TEAL_DARK)]:
        g=m.loc[m["Mode"].eq(mode)]
        f.add_trace(go.Bar(x=g["Short_Model"],y=g["Approved_Cases"],name=mode,marker_color=color,hovertemplate="%{x}<br>%{y:,.0f} cases<extra></extra>"),row=1,col=1)
        f.add_trace(go.Bar(x=g["Short_Model"],y=g["Recommended_Volume"]/1e9,name=mode,showlegend=False,marker_color=color,hovertemplate="%{x}<br>Volume %{y:.2f}B<extra></extra>"),row=1,col=2)
    f.update_layout(barmode="group",legend={"orientation":"h","y":-0.24})
    figs.append(("p2","Iso-Approval vs Iso-Risk comparison",plotly_theme(f,"Decision mode comparison · OOT_Monitor",460)))
    f=go.Figure()
    stress=frames["stress"]
    for model,color in MODEL_COLORS.items():
        g=stress.loc[stress["Model"].eq(model)].set_index("Scenario").reindex(SCENARIO_ORDER)
        f.add_trace(go.Bar(x=[SCENARIO_LABELS[s] for s in SCENARIO_ORDER],y=g["Model_Implied_Expected_Net_Profit"]/1e6,name=f"{model} · implied ENP",marker_color=color,hovertemplate="%{x}<br>Expected profit: %{y:.1f}M currency units<extra></extra>"))
        f.add_trace(go.Bar(x=[SCENARIO_LABELS[s] for s in SCENARIO_ORDER],y=g["Realized_Net_Profit_Stress_Proxy"]/1e6,name=f"{model} · stress proxy",marker={"color":MODEL_COLORS[model],"opacity":.5,"pattern":{"shape":"/"}},customdata=g["Stress_PD_Multiplier"],hovertemplate="%{x}<br>Stress proxy: %{y:.1f}M currency units<br>Multiplier %{customdata:.2f}x<extra></extra>"))
    f.update_layout(barmode="group",legend={"orientation":"h","y":-0.24},bargap=.18,bargroupgap=.08)
    figs.append(("p2","S1–S4 stress profit",plotly_theme(f,"S1–S4 portfolio stress comparison · proxy values are not realized downturn P&L")))
    # Page 3: drivers and governance.
    f=make_subplots(rows=1,cols=2,subplot_titles=("Champion","LightGBM M4"),shared_yaxes=True)
    for col,(model,title) in enumerate([("Champion Scorecard","Champion"),("Challenger LightGBM M4","LightGBM M4")],1):
        g=frames["binding"].loc[frames["binding"]["Model"].eq(model)]
        for driver,color in [("Affordability",TEAL_DARK),("Requested Amount",BLUE),("Risk_Behavior",GOLD),("Unverified_Income_Cap",CORAL)]:
            q=g.loc[g["Binding_Driver"].eq(driver)]
            f.add_trace(go.Bar(x=[x.replace("Tier ","") for x in q["Risk_Tier"]],y=q["Binding_Driver_Share_Pct"],name=driver,legendgroup=driver,showlegend=col==1,marker_color=color,hovertemplate="%{x}<br>Share %{y:.1f}%<extra></extra>"),row=1,col=col)
    f.update_layout(barmode="stack",legend={"orientation":"h","y":-0.24})
    figs.append(("p3","Risk-tier binding driver shares",plotly_theme(f,"Iso-Risk binding-driver share by risk tier",460)))
    f=go.Figure(); csi=frames["csi"]
    colors=[CORAL if x>.25 else GOLD if x>=.10 else TEAL_DARK for x in csi["Metric"]]
    f.add_trace(go.Bar(x=csi["Metric"],y=csi["Feature"],orientation="h",marker_color=colors,customdata=csi["Split"],hovertemplate="%{y}<br>CSI %{x:.3f}<br>%{customdata}<extra></extra>"))
    f.add_vline(x=.10,line_dash="dash",line_color=GOLD,annotation_text="0.10"); f.add_vline(x=.25,line_dash="dash",line_color=CORAL,annotation_text="0.25")
    figs.append(("p3","Top 15 CSI drift",plotly_theme(f,"Top 15 feature CSI · Train_Dev vs OOT_Monitor",520)))
    f.update_layout(margin={"l":220,"r":35,"t":65,"b":50})
    mig=frames["migration"]
    matrix=mig.pivot(index="Risk_Tier",columns="Prior_Delinquency_Severity",values="Current_Default_Rate_Pct").reindex(index=TIER_ORDER,columns=SEVERITY_ORDER)
    count=mig.pivot(index="Risk_Tier",columns="Prior_Delinquency_Severity",values="Application_Count").reindex(index=TIER_ORDER,columns=SEVERITY_ORDER)
    z=matrix.to_numpy(dtype=float); counts=count.to_numpy(dtype=float)
    text=np.where(np.isfinite(z),np.char.mod("%.1f%%",np.nan_to_num(z)),"—")
    f=go.Figure(go.Heatmap(z=z,x=[str(s) for s in matrix.columns],y=[str(s) for s in matrix.index],colorscale=[[0,"#fff5eb"],[1,"#b30000"]],colorbar={"title":"Bad rate %"},customdata=counts,text=text,texttemplate="%{text}",hovertemplate="%{y}<br>%{x}<br>Current bad rate %{z:.2f}%<br>Applications %{customdata:,.0f}<extra></extra>"))
    figs.append(("p3","Prior-DPD migration heatmap",plotly_theme(f,"Cross-sectional prior DPD severity → current default",500)))
    age=frames["age"].sort_values("Group")
    f=go.Figure()
    f.add_trace(go.Bar(y=age["Group"],x=age["AIR"],orientation="h",name="AIR",marker_color=TEAL_DARK,customdata=age["Sample_Count"],hovertemplate="Age %{y}<br>AIR %{x:.3f}<br>Sample %{customdata:,.0f}<extra></extra>"))
    f.add_trace(go.Scatter(y=age["Group"],x=age["Observed_Bad_Rate_Pct"],xaxis="x2",mode="lines+markers+text",text=[f"{v:.2f}%" if g in ("<25","50+") else "" for g,v in zip(age["Group"].astype(str),age["Observed_Bad_Rate_Pct"])],textposition="middle right",name="Observed Bad Rate %",line={"color":GOLD,"width":2},marker={"color":GOLD,"size":8},hovertemplate="Age %{y}<br>Observed bad rate %{x:.2f}%<extra></extra>"))
    f.update_layout(xaxis={"title":"AIR","range":[0,1.12]},xaxis2={"title":"Observed Bad Rate %","overlaying":"x","side":"top","range":[0,max(7,float(age["Observed_Bad_Rate_Pct"].max())+1)]})
    f.add_vline(x=.80,line_dash="dash",line_color=CORAL,annotation_text="AIR 0.80 diagnostic")
    figs.append(("p3","Age-band AIR vs bad rate",plotly_theme(f,"Age-band AIR vs observed bad rate · OOT_Monitor")))
    return figs


def write_interactive_html(frames: dict) -> None:
    sections = interactive_figures(frames)
    chunks=[]
    for i,(page,title,fig) in enumerate(sections):
        chunks.append(pio.to_html(fig,full_html=False,include_plotlyjs=True if i==0 else False,config={"displaylogo":False,"responsive":True,"scrollZoom":True}))
    tabs=[]
    ids=["p1","p2","p3"]
    names=["Page 1 Portfolio Pulse","Page 2 Strategy & Stress Lab","Page 3 Exposure, Stability & Fairness"]
    for i,(ident,name) in enumerate(zip(ids,names)):
        tabs.append(f'<button class="tab{" active" if i==0 else ""}" role="tab" aria-selected="{str(i==0).lower()}" aria-controls="panel-{ident}" id="tab-{ident}" onclick="showPage(\'{ident}\')">{html.escape(name)}</button>')
    panels=[]
    for ident,name in zip(ids,names):
        related=[chunk for (page,_,_),chunk in zip(sections,chunks) if page==ident]
        panels.append(f'<section class="page" id="panel-{ident}" role="tabpanel" aria-labelledby="tab-{ident}" {"" if ident=="p1" else "hidden"}><h2>{html.escape(name)}</h2><div class="grid">'+"".join(f'<article class="chart">{chunk}</article>' for chunk in related)+"</div></section>")
    k=frames["kpis"]
    page=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FinGuard Interactive Command Center</title>
<style>
:root{{--navy:{NAVY};--teal:{TEAL};--bg:#f3f7fa;--ink:#243b53;--muted:#627d98;--line:#d9e2ec;--white:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header{{background:linear-gradient(115deg,var(--navy),#17445b);color:white;padding:30px max(24px,calc((100vw - 1440px)/2)) 24px}}
.eyebrow{{font-size:11px;letter-spacing:.16em;color:#77e0d3;font-weight:700}}h1{{margin:10px 0 4px;font-size:30px}}header p{{margin:0;color:#c7d8e3;font-size:14px}}
.wrap{{max-width:1440px;margin:auto;padding:20px 24px 42px}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}}
.kpi{{background:white;border-radius:10px;padding:17px 20px;box-shadow:0 2px 8px #102a4310}}.kpi span{{display:block;color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.08em}}.kpi strong{{display:block;color:var(--navy);font-size:25px;margin-top:5px}}
nav{{display:flex;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:18px}}.tab{{border:0;background:transparent;padding:13px 16px;color:var(--muted);font-weight:600;cursor:pointer;border-bottom:3px solid transparent}}.tab.active{{color:var(--navy);border-color:var(--teal)}}
.page h2{{font-size:19px;color:var(--navy);margin:10px 0 16px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.chart{{background:white;border-radius:10px;padding:8px 8px 2px;box-shadow:0 2px 8px #102a430c;min-width:0}}
.note{{color:var(--muted);font-size:12px;margin-top:16px;line-height:1.5}}@media(max-width:820px){{.grid{{grid-template-columns:1fr}}.kpis{{grid-template-columns:1fr}}h1{{font-size:24px}}.tab{{padding:11px 8px;font-size:12px}}}}
</style></head><body><header><div class="eyebrow">FIN GUARD · CREDIT RISK INTELLIGENCE</div><h1>Interactive Command Center</h1><p>Portfolio performance, decision strategy, drift and fairness · historical retrospective analysis</p></header>
<main class="wrap"><div class="kpis"><div class="kpi"><span>OOT_MONITOR APPLICATIONS</span><strong>{k['applications']:,.0f}</strong></div><div class="kpi"><span>ISORISK APPROVAL</span><strong>{k['iso_risk_approval_pct']:.2f}%</strong></div><div class="kpi"><span>ISORISK BAD RATE</span><strong>{k['iso_risk_bad_pct']:.4f}%</strong></div><div class="kpi"><span>ISORISK SIMULATED ENP</span><strong>{k['iso_risk_enp_m']:.2f}M currency units</strong></div></div>
<nav role="tablist" aria-label="Dashboard pages">{"".join(tabs)}</nav>{"".join(panels)}
<p class="note">Metrics are aggregated from the supplied Power BI marts. Stress proxy values are scenario calculations, not observed downturn profit. AIR is a fairness diagnostic, not a legal determination. Prior-DPD heatmap is cross-sectional and does not represent monthly MOB migration.</p></main>
<script>function showPage(id){{document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!=='panel-'+id);document.querySelectorAll('.tab').forEach(t=>{{let on=t.id==='tab-'+id;t.classList.toggle('active',on);t.setAttribute('aria-selected',String(on));}});window.dispatchEvent(new Event('resize'));}}</script></body></html>'''
    HTML_PATH.write_text(page,encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.titleweight":"bold","figure.facecolor":PALE,"savefig.facecolor":PALE})
    tables=load_marts()
    frames=build_frames(tables)
    page1(frames); page2(frames); page3(frames)
    write_interactive_html(frames)
    print(f"Read exactly {len(FILES)} approved Power BI CSVs from {DATA_DIR}")
    for filename in ["exec_page1_portfolio_pulse.png","exec_page2_strategy_stress.png","exec_page3_stability_fairness.png"]:
        print(f"PNG {OUT_DIR / filename} ({(OUT_DIR / filename).stat().st_size:,} bytes)")
    print(f"HTML {HTML_PATH} ({HTML_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        main()
