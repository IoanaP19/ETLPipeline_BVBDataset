"""
dashboard.py — BVB Interactive Business Intelligence Dashboard
==============================================================
Bachelor Thesis: "An Applied Perspective on ETL: Python-Based Use Cases
                  for Business Analytics"

How to run:
    streamlit run dashboard.py

Prerequisites:
    Run `python main.py` at least once so outputs/bvb_warehouse.db exists.

Tabs:
    1. Market Analytics     — cumulative returns, Sharpe ratio, SMA, drawdown
    2. ETL Health Monitor   — warehouse stats, row counts, pipeline log
    3. Data Governance      — quarantine review: approve or reject anomalies,
                              before/after impact visualisation
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR     = Path(__file__).parent
DB_PATH      = ROOT_DIR / "outputs" / "bvb_warehouse.db"
LOG_PATH     = ROOT_DIR / "outputs" / "pipeline.log"
CFG_PATH     = ROOT_DIR / "config.json"
TRADING_DAYS = 252

# Table names — must match src/load.py
TABLE_MARKET = "market_data"
TABLE_AUDIT  = "data_audit_log"


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BVB BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; }
    [data-testid="metric-container"] {
        background: rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_config() -> dict:
    if CFG_PATH.exists():
        with CFG_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    return {"analysis": {"risk_free_rate": 0.065}}

cfg             = load_config()
DEFAULT_RF_RATE = cfg.get("analysis", {}).get("risk_free_rate", 0.065)
VOL_THRESHOLD   = cfg.get("data_quality", {}).get("volatility_threshold", 0.15)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_market_data() -> pd.DataFrame:
    """
    Read vetted, clean rows from the market_data table.
    Cached for 1 hour. Refresh button clears this cache.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {TABLE_MARKET} ORDER BY ticker, date",
                conn, parse_dates=["date"],
            )
        except Exception:
            return pd.DataFrame()
    return df


def load_audit_log() -> pd.DataFrame:
    """
    Read all rows from data_audit_log.
    NOT cached — must reflect live human decisions immediately.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {TABLE_AUDIT} ORDER BY ticker, date",
                conn, parse_dates=["date"],
            )
        except Exception:
            return pd.DataFrame()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Governance action helpers — write directly to SQLite
# ─────────────────────────────────────────────────────────────────────────────

def approve_anomaly(ticker: str, date_str: str) -> None:
    """
    Approve a quarantined row as a genuine market move.

    Actions:
      1. Read the full row from data_audit_log
      2. Insert it into market_data
      3. Update resolution_status to APPROVED in data_audit_log

    This is an atomic operation: either both writes succeed or neither does.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()

        # Read the row we are approving
        cur.execute(
            f"SELECT * FROM {TABLE_AUDIT} WHERE ticker=? AND date=?",
            (ticker, date_str),
        )
        row = cur.fetchone()
        if not row:
            return

        cols = [d[0] for d in cur.description]
        row_dict = dict(zip(cols, row))

        # Columns that exist in market_data (governance cols not included)
        gov_cols = {"reason", "resolution_status", "flagged_at"}

        # Get market_data columns to know what we can insert
        cur.execute(f"PRAGMA table_info({TABLE_MARKET})")
        market_cols = {r[1] for r in cur.fetchall()}

        insert_dict = {k: v for k, v in row_dict.items()
                       if k in market_cols and k not in gov_cols}

        placeholders = ", ".join(insert_dict.keys())
        values_ph    = ", ".join(["?" for _ in insert_dict])

        cur.execute(
            f"INSERT OR IGNORE INTO {TABLE_MARKET} ({placeholders}) VALUES ({values_ph})",
            list(insert_dict.values()),
        )

        # Mark as approved in audit log
        cur.execute(
            f"UPDATE {TABLE_AUDIT} SET resolution_status='APPROVED' "
            f"WHERE ticker=? AND date=?",
            (ticker, date_str),
        )
        conn.commit()


def reject_anomaly(ticker: str, date_str: str) -> None:
    """
    Reject a quarantined row as a data error / corporate action artefact.
    Updates resolution_status to REJECTED — row stays excluded from analytics.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"UPDATE {TABLE_AUDIT} SET resolution_status='REJECTED' "
            f"WHERE ticker=? AND date=?",
            (ticker, date_str),
        )
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────

df_all   = load_market_data()
df_audit = load_audit_log()

if df_all.empty:
    st.error(
        "### Warehouse not found\n"
        "Run the pipeline first:\n\n```bash\npython main.py\n```\n\nThen refresh."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Global alert — shown ABOVE everything if anomalies need review
# ─────────────────────────────────────────────────────────────────────────────

pending = df_audit[df_audit["resolution_status"] == "PENDING_REVIEW"] if not df_audit.empty else pd.DataFrame()

if not pending.empty:
    st.warning(
        f"⚠️  **Attention: {len(pending)} data anomaly(ies) require manual validation.** "
        f"Go to the **Data Governance** tab to review and approve or reject each one.",
        icon="⚠️",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 BVB BI Dashboard")
    st.markdown("*Bachelor Thesis · 2025*")
    st.divider()

    all_tickers = sorted(df_all["ticker"].unique().tolist())
    sel_tickers = st.multiselect(
        label="Ticker",
        options=all_tickers,
        default=all_tickers,
        help="Select one or more BVB-listed companies.",
    )

    date_min = df_all["date"].min().date()
    date_max = df_all["date"].max().date()
    date_range = st.slider(
        label="Date range",
        min_value=date_min,
        max_value=date_max,
        value=(date_min, date_max),
        format="MMM YYYY",
    )
    d_start = pd.Timestamp(date_range[0])
    d_end   = pd.Timestamp(date_range[1])

    rf_pct = st.slider(
        label="Risk-free rate (%)",
        min_value=0.0, max_value=15.0,
        value=round(DEFAULT_RF_RATE * 100, 2),
        step=0.25, format="%.2f%%",
        help=f"Default = {DEFAULT_RF_RATE*100:.2f}% (NBR 2024, from config.json).",
    )
    risk_free = rf_pct / 100

    st.divider()
    if st.button("🔄  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Warehouse: {date_min} → {date_max}")
    st.caption(f"{len(df_all):,} vetted rows · {df_all['ticker'].nunique()} tickers")
    if not pending.empty:
        st.caption(f"⚠️  {len(pending)} anomalies pending review")


# ─────────────────────────────────────────────────────────────────────────────
# Apply sidebar filters
# ─────────────────────────────────────────────────────────────────────────────

if not sel_tickers:
    st.warning("Select at least one ticker in the sidebar.")
    st.stop()

df = df_all[
    df_all["ticker"].isin(sel_tickers) &
    (df_all["date"] >= d_start) &
    (df_all["date"] <= d_end)
].copy()

if df.empty:
    st.warning("No data matches your filters.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("BVB Business Intelligence Dashboard")
st.markdown(
    f"**Bursa de Valori București** &nbsp;·&nbsp; "
    f"{d_start.strftime('%d %b %Y')} — {d_end.strftime('%d %b %Y')} &nbsp;·&nbsp; "
    f"{len(sel_tickers)} ticker(s) &nbsp;·&nbsp; rf = {rf_pct:.2f}%",
    unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sharpe helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe(group: pd.DataFrame, rf: float) -> float:
    rets = group["daily_return"].dropna()
    px   = group["close"].dropna()
    if len(rets) < 20 or len(px) < 2:
        return 0.0
    ann_ret = (px.iloc[-1] / px.iloc[0]) ** (TRADING_DAYS / max(len(group), 1)) - 1
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_market, tab_health, tab_gov = st.tabs([
    "📈  Market Analytics",
    "🔬  ETL Health Monitor",
    "🛡️  Data Governance",
])


# =============================================================================
#  TAB 1 — MARKET ANALYTICS
# =============================================================================

with tab_market:

    # Sharpe metric cards
    st.subheader("Sharpe Ratio  ·  live recalculation")
    st.caption(f"Recalculated with rf = **{rf_pct:.2f}%** · adjust the sidebar slider to update.")

    sharpe_values = {
        ticker: compute_sharpe(grp.sort_values("date"), risk_free)
        for ticker, grp in df.groupby("ticker")
    }

    cols = st.columns(max(len(sharpe_values), 1))
    for i, (ticker, sharpe) in enumerate(
        sorted(sharpe_values.items(), key=lambda x: x[1], reverse=True)
    ):
        company = df[df["ticker"] == ticker]["company"].iloc[0]
        cols[i].metric(
            label=f"{ticker}  ·  {company}",
            value=f"{sharpe:.3f}",
            delta=f"{'above' if sharpe >= 0 else 'below'} rf",
            delta_color="normal" if sharpe >= 0 else "inverse",
        )

    st.divider()

    # Cumulative return chart
    st.subheader("Cumulative Return")
    st.caption("Vetted data only (anomalies excluded until approved) · hover for values · click legend to isolate")

    cum_df = (
        df[["date", "ticker", "company", "cumulative_return"]]
        .assign(
            ret_pct=lambda d: d["cumulative_return"] * 100,
            label=lambda d: d["ticker"] + "  " + d["company"],
        )
        .sort_values(["ticker", "date"])
    )

    fig_cum = px.line(
        cum_df, x="date", y="ret_pct", color="label",
        labels={"date": "Date", "ret_pct": "Cumulative Return (%)", "label": "Company"},
        title=f"BVB Cumulative Returns  ·  {d_start.strftime('%d %b %Y')} — {d_end.strftime('%d %b %Y')}",
        template="plotly_white",
    )
    fig_cum.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)", line_width=1)
    fig_cum.update_traces(line_width=2)
    fig_cum.update_layout(
        height=420, hovermode="x unified", yaxis_ticksuffix="%",
        legend_title_text="Ticker · Company",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    st.divider()

    # SMA chart (single ticker only)
    sma_cols = [c for c in df.columns if c.startswith("SMA_")]
    if sma_cols and len(sel_tickers) == 1:
        ticker_sel = sel_tickers[0]
        sma_df     = df[df["ticker"] == ticker_sel].sort_values("date")
        st.subheader(f"Price & Moving Averages  ·  {ticker_sel}")
        price_long = (
            sma_df[["date", "close"] + sma_cols]
            .melt(id_vars="date", var_name="Series", value_name="Price (RON)")
        )
        fig_sma = px.line(
            price_long, x="date", y="Price (RON)", color="Series",
            title=f"{ticker_sel} — Adjusted Close & Moving Averages",
            template="plotly_white",
            color_discrete_map={"close": "#378ADD", "SMA_50": "#EF9F27", "SMA_200": "#E94560"},
        )
        fig_sma.update_layout(height=360, hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sma, use_container_width=True)
        st.divider()
    elif sma_cols and len(sel_tickers) > 1:
        st.info("Select a single ticker to view the SMA chart.")
        st.divider()

    # Drawdown chart
    if "drawdown" in df.columns:
        st.subheader("Rolling Maximum Drawdown")
        dd_df = (
            df[["date", "ticker", "company", "drawdown"]]
            .assign(
                dd_pct=lambda d: d["drawdown"] * 100,
                label=lambda d: d["ticker"] + "  " + d["company"],
            )
        )
        fig_dd = px.line(
            dd_df, x="date", y="dd_pct", color="label",
            labels={"date": "Date", "dd_pct": "Drawdown (%)", "label": "Company"},
            title="Rolling Maximum Drawdown (%)",
            template="plotly_white",
        )
        fig_dd.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)", line_width=1)
        fig_dd.update_layout(
            height=340, hovermode="x unified", yaxis_ticksuffix="%",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_dd, use_container_width=True)


# =============================================================================
#  TAB 2 — ETL HEALTH MONITOR
# =============================================================================

with tab_health:

    st.subheader("Warehouse Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows (market_data)",   f"{len(df_all):,}")
    c2.metric("Unique tickers",             str(df_all["ticker"].nunique()))
    c3.metric("Date span",                  f"{(df_all['date'].max() - df_all['date'].min()).days} days")
    c4.metric("Latest data point",          df_all["date"].max().strftime("%d %b %Y"))

    if not df_audit.empty:
        a1, a2, a3 = st.columns(3)
        a1.metric("Quarantined rows (total)",   len(df_audit))
        a2.metric("Pending review",             len(df_audit[df_audit["resolution_status"] == "PENDING_REVIEW"]))
        a3.metric("Resolved (approved+rejected)",
                  len(df_audit[df_audit["resolution_status"] != "PENDING_REVIEW"]))

    st.divider()
    st.subheader("Rows per Ticker")

    ticker_counts = (
        df_all.groupby(["ticker", "company", "sector"])
        .agg(rows=("date", "count"), first_date=("date", "min"), last_date=("date", "max"))
        .reset_index().sort_values("rows", ascending=False)
    )
    ticker_counts["first_date"] = ticker_counts["first_date"].dt.strftime("%Y-%m-%d")
    ticker_counts["last_date"]  = ticker_counts["last_date"].dt.strftime("%Y-%m-%d")

    fig_bar = px.bar(
        ticker_counts, x="ticker", y="rows", color="sector", text="rows",
        title="Vetted rows per ticker (market_data)", template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(ticker_counts, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Pipeline Log  ·  last 20 lines")

    if not LOG_PATH.exists():
        st.info("No log file yet. Run `python main.py` first.")
    else:
        lines    = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        log_text = "\n".join(lines[-20:])
        st.code(log_text, language="log")
        st.caption(f"Lines {max(1, len(lines)-19)}–{len(lines)} of {len(lines):,} total")


# =============================================================================
#  TAB 3 — DATA GOVERNANCE
# =============================================================================

with tab_gov:

    st.subheader("Data Governance — Anomaly Review")
    st.markdown(
        f"Rows where **|daily return| > {VOL_THRESHOLD*100:.0f}%** (BVB circuit-breaker limit) "
        f"are quarantined and excluded from analytics until manually reviewed here. "
        f"Each row is either confirmed as a real market event or rejected as a data error."
    )

    # Reload audit log fresh every time this tab renders (not cached)
    df_audit = load_audit_log()

    if df_audit.empty:
        st.success("✅  No rows in the quarantine log. The pipeline has not flagged any anomalies yet.")
        st.stop()

    # ── Summary counts ────────────────────────────────────────────────────────
    n_pending  = len(df_audit[df_audit["resolution_status"] == "PENDING_REVIEW"])
    n_approved = len(df_audit[df_audit["resolution_status"] == "APPROVED"])
    n_rejected = len(df_audit[df_audit["resolution_status"] == "REJECTED"])

    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("⏳ Pending review",  n_pending,  delta=None)
    gc2.metric("✅ Approved",        n_approved, delta=None)
    gc3.metric("❌ Rejected",        n_rejected, delta=None)

    st.divider()

    # ── Pending review table ──────────────────────────────────────────────────
    pending_df = df_audit[df_audit["resolution_status"] == "PENDING_REVIEW"].copy()

    if pending_df.empty:
        st.success("✅  All anomalies have been reviewed. No pending items.")
    else:
        st.subheader(f"⏳ Pending Review  ({len(pending_df)} rows)")
        st.caption(
            "For each flagged row, review the reason and decide: "
            "**Approve** if you believe it is a real BVB market event, "
            "**Reject** if you believe it is an API error or split artefact."
        )

        for idx, row in pending_df.iterrows():
            ticker   = row["ticker"]
            date_str = str(row["date"])[:10]
            ret_pct  = row["daily_return"] * 100 if pd.notna(row["daily_return"]) else 0
            reason   = row.get("reason", "—")
            company  = row.get("company", ticker)

            with st.expander(
                f"**{ticker}** · {company} · {date_str} · "
                f"return = **{ret_pct:+.2f}%**",
                expanded=True,
            ):
                col_info, col_chart = st.columns([1, 2])

                with col_info:
                    st.markdown(f"**Ticker:**   {ticker}")
                    st.markdown(f"**Date:**     {date_str}")
                    st.markdown(f"**Close:**    {row.get('close', '—')} RON")
                    st.markdown(f"**Return:**   {ret_pct:+.2f}%")
                    st.markdown(f"**Reason:**   {reason}")
                    st.markdown(f"**Flagged:**  {row.get('flagged_at', '—')}")

                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button(
                            "✅ Approve as Market Move",
                            key=f"approve_{ticker}_{date_str}",
                            use_container_width=True,
                            help="Move this row to market_data — it will appear in all analytics.",
                        ):
                            approve_anomaly(ticker, date_str)
                            st.cache_data.clear()
                            st.success(f"{ticker} {date_str} approved ✅")
                            st.rerun()

                    with b_col2:
                        if st.button(
                            "❌ Flag as API Error / Split",
                            key=f"reject_{ticker}_{date_str}",
                            use_container_width=True,
                            help="Mark as a data error — excluded from all analytics permanently.",
                        ):
                            reject_anomaly(ticker, date_str)
                            st.success(f"{ticker} {date_str} rejected ❌")
                            st.rerun()

                # ── Before / After impact chart ───────────────────────────────
                with col_chart:
                    st.markdown("**Impact visualisation — Before vs After governance**")
                    st.caption(
                        "**After** (current) = cumulative return in vetted market_data, "
                        "anomaly excluded.  "
                        "**Before** = what the return would look like if this row were included."
                    )

                    ticker_clean = df_all[df_all["ticker"] == ticker].sort_values("date").copy()

                    if len(ticker_clean) >= 2 and pd.notna(row["daily_return"]):
                        # --- AFTER: current clean series (anomaly excluded) ---
                        after_series = ticker_clean[["date", "close"]].copy()
                        after_series["version"] = "After governance (current)"

                        # --- BEFORE: insert the anomaly row back, recompute ---
                        anomaly_row = pd.DataFrame([{
                            "date":  pd.Timestamp(date_str),
                            "close": row["close"],
                        }])
                        before_df = (
                            pd.concat([ticker_clean[["date", "close"]], anomaly_row])
                            .sort_values("date")
                            .drop_duplicates(subset=["date"])
                            .reset_index(drop=True)
                        )
                        before_df["version"] = "Before governance (includes anomaly)"

                        # Recompute cumulative return for both versions
                        for version_df in [after_series, before_df]:
                            prices = version_df["close"]
                            version_df["cumulative_ret_%"] = (prices / prices.iloc[0] - 1) * 100

                        combined = pd.concat([after_series, before_df], ignore_index=True)

                        fig_impact = px.line(
                            combined,
                            x="date",
                            y="cumulative_ret_%",
                            color="version",
                            labels={
                                "date":               "Date",
                                "cumulative_ret_%":   "Cumulative Return (%)",
                                "version":            "",
                            },
                            template="plotly_white",
                            color_discrete_map={
                                "After governance (current)":           "#16a34a",
                                "Before governance (includes anomaly)": "#dc2626",
                            },
                        )

                        # Mark the anomaly date with a vertical line
                        fig_impact.add_vline(
                            x=pd.Timestamp(date_str).timestamp() * 1000,
                            line_dash="dot",
                            line_color="orange",
                            line_width=2,
                            annotation_text="Anomaly date",
                            annotation_position="top right",
                        )
                        fig_impact.add_hline(
                            y=0, line_dash="dash",
                            line_color="rgba(0,0,0,0.15)", line_width=1,
                        )
                        fig_impact.update_layout(
                            height=280,
                            yaxis_ticksuffix="%",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=0, r=0, t=30, b=0),
                        )
                        st.plotly_chart(fig_impact, use_container_width=True)
                    else:
                        st.info("Not enough data to render the before/after chart for this ticker.")

    # ── Resolved rows table (collapsed by default) ────────────────────────────
    resolved_df = df_audit[df_audit["resolution_status"] != "PENDING_REVIEW"]
    if not resolved_df.empty:
        st.divider()
        with st.expander(f"Resolved anomalies ({len(resolved_df)} rows)", expanded=False):
            display_cols = ["date", "ticker", "company", "close",
                            "daily_return", "reason", "resolution_status", "flagged_at"]
            show_cols    = [c for c in display_cols if c in resolved_df.columns]
            disp         = resolved_df[show_cols].copy()
            if "daily_return" in disp.columns:
                disp["daily_return"] = (disp["daily_return"] * 100).round(2)
                disp = disp.rename(columns={"daily_return": "daily_return_%"})
            st.dataframe(disp, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Data: Yahoo Finance · Prices dividend-adjusted · "
    f"Quarantine threshold: {VOL_THRESHOLD*100:.0f}% (BVB circuit-breaker) · "
    f"rf = {rf_pct:.2f}% · "
    f"Warehouse: market_data ({len(df_all):,} rows) | "
    f"data_audit_log ({len(df_audit):,} rows)"
)
