"""
comparison.py — Data Governance Impact Analysis
================================================
Bachelor Thesis: "An Applied Perspective on ETL: Python-Based Use Cases
                  for Business Analytics"

PURPOSE
-------
Generates the empirical evidence for the thesis research question:

    "What is the quantifiable impact of automated data governance on
     the accuracy of business analytics metrics for BVB-listed securities?"

HOW IT WORKS
------------
The script runs the full transformation pipeline TWICE on the same
raw data, changing only one variable between the two runs:

  Run A — Governance DISABLED
    All rows kept, including those with |daily_return| > 15%.
    This simulates a naive ETL pipeline with no quality controls.

  Run B — Governance ENABLED (your production pipeline)
    Rows with |daily_return| > 15% are quarantined and excluded.
    This is the current src/transform.py behaviour.

For each run it computes: total return, annualised return, annualised
volatility, Sharpe ratio, and maximum drawdown per ticker.

The difference between Run A and Run B for each metric is the
measurable contribution of the governance layer.

OUTPUTS (all saved to outputs/)
-------
  Table4_governance_impact.xlsx   — main comparison table for thesis
  Table4_quarantine_detail.xlsx   — detail of every quarantined row
  figure5_sharpe_comparison.png   — bar chart: Sharpe with vs without governance
  figure6_return_comparison.png   — bar chart: cumulative return with vs without

HOW TO RUN
----------
  python comparison.py

Run AFTER python main.py (needs the raw data already extracted,
but does NOT overwrite the warehouse — read-only analysis).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Add project root to path so src/ imports work ────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.extract   import run as extract_run
from src.transform import (
    _remove_price_spikes,
    _remove_return_outliers,
    _add_moving_averages,
    _add_max_drawdown,
    _flag_anomalies,
    _split_quarantine,
)
from src.logger import get_logger

log = get_logger("comparison")

OUT_DIR = ROOT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = ROOT_DIR / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Shared transformation steps (identical in both runs)
# Steps 1–9: cleaning, returns, volatility, SMA, drawdown
# ─────────────────────────────────────────────────────────────────────────────

def _apply_shared_steps(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply all transformation steps that are IDENTICAL in both runs.
    This ensures the only difference between Run A and Run B is the
    quarantine step — a clean controlled experiment.
    """
    dq_cfg  = cfg["data_quality"]
    ana_cfg = cfg["analysis"]
    tdays   = ana_cfg["trading_days_per_year"]

    df = df.copy()

    # Step 1 — basic cleaning
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    if "volume" in df.columns:
        df = df[df["volume"] > 0]
    df = df.drop_duplicates(subset=["date", "ticker"])

    # Step 2 — price spike filter
    df = _remove_price_spikes(df, dq_cfg)

    # Step 3 — daily returns
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()

    # Step 4 — hard outlier removal at 25%
    df = _remove_return_outliers(df, dq_cfg)

    # Step 5 — cumulative return
    df["cumulative_return"] = (
        df.groupby("ticker")["daily_return"]
          .transform(lambda x: (1 + x.fillna(0)).cumprod() - 1)
    )

    # Step 6 — rolling volatility
    df["volatility_30d"] = (
        df.groupby("ticker")["daily_return"]
          .transform(lambda x: x.rolling(30).std() * np.sqrt(tdays))
    )

    # Step 7 — calendar fields
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    # Step 8 — moving averages
    df = _add_moving_averages(df, ana_cfg.get("sma_windows", [50, 200]))

    # Step 9 — drawdown
    df = _add_max_drawdown(df)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Run A — Governance DISABLED
# ─────────────────────────────────────────────────────────────────────────────

def run_without_governance(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Naive pipeline — no quarantine step.
    All rows with |daily_return| in 15–25% range are KEPT.
    This is what an analyst gets if they trust the data blindly.
    """
    log.info("RUN A — Governance DISABLED (naive pipeline)")
    df = _apply_shared_steps(df_raw, cfg)
    # Step 10 is skipped — no flagging, no splitting
    # Step 11 is skipped — all rows go to analytics
    log.info(f"  Run A complete: {len(df):,} rows retained (including anomalies)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Run B — Governance ENABLED
# ─────────────────────────────────────────────────────────────────────────────

def run_with_governance(df_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Production pipeline — quarantine step active.
    Rows with |daily_return| > 15% are removed from analytics.
    Returns (df_clean, df_quarantine).
    """
    log.info("RUN B — Governance ENABLED (production pipeline)")
    df = _apply_shared_steps(df_raw, cfg)
    # Step 10 — flag anomalies
    df = _flag_anomalies(df, cfg["data_quality"])
    # Step 11 — split
    df_clean, df_quarantine = _split_quarantine(df)
    log.info(
        f"  Run B complete: {len(df_clean):,} clean rows | "
        f"{len(df_quarantine):,} quarantined"
    )
    return df_clean, df_quarantine


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, cfg: dict, label: str) -> pd.DataFrame:
    """
    Compute per-ticker analytics metrics from a DataFrame.
    Returns a tidy DataFrame with one row per ticker.
    """
    rf    = cfg["analysis"]["risk_free_rate"]
    tdays = cfg["analysis"]["trading_days_per_year"]
    rows  = []

    for ticker, g in df.groupby("ticker"):
        g    = g.sort_values("date")
        rets = g["daily_return"].dropna()
        px   = g["close"].dropna()

        if len(rets) < 10 or len(px) < 2:
            continue

        # Total return over the full period
        total_ret = px.iloc[-1] / px.iloc[0] - 1

        # Annualised return: geometric scaling to one year
        ann_ret = (px.iloc[-1] / px.iloc[0]) ** (tdays / max(len(g), 1)) - 1

        # Annualised volatility
        ann_vol = rets.std() * np.sqrt(tdays)

        # Sharpe ratio
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0

        # Max drawdown (worst peak-to-trough)
        if "drawdown" in g.columns:
            max_dd = g["drawdown"].min()
        else:
            roll_max = px.cummax()
            max_dd   = ((px - roll_max) / roll_max).min()

        rows.append({
            "ticker":        ticker,
            "company":       g["company"].iloc[0],
            "sector":        g["sector"].iloc[0],
            "run":           label,
            "rows_used":     len(g),
            "total_ret_%":   round(total_ret * 100, 2),
            "ann_ret_%":     round(ann_ret * 100, 2),
            "ann_vol_%":     round(ann_vol * 100, 2),
            "sharpe_ratio":  round(sharpe, 4),
            "max_dd_%":      round(max_dd * 100, 2),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(
    metrics_a: pd.DataFrame,
    metrics_b: pd.DataFrame,
    df_quarantine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Run A and Run B metrics side by side and compute the delta.
    Also adds a 'rows_quarantined' column for each ticker.

    This produces the thesis Table 4.
    """
    a = metrics_a.set_index("ticker")
    b = metrics_b.set_index("ticker")

    # Count quarantined rows per ticker
    if df_quarantine.empty:
        q_counts = pd.Series(0, index=a.index, name="rows_quarantined")
    else:
        q_counts = df_quarantine.groupby("ticker").size().rename("rows_quarantined")

    comparison_rows = []
    for ticker in sorted(a.index):
        if ticker not in b.index:
            continue

        row_a = a.loc[ticker]
        row_b = b.loc[ticker]
        quarantined = int(q_counts.get(ticker, 0))

        comparison_rows.append({
            # Identity
            "Ticker":              ticker,
            "Company":             row_a["company"],
            "Sector":              row_a["sector"],
            "Rows quarantined":    quarantined,

            # Total return
            "Total ret — no gov (%)":   row_a["total_ret_%"],
            "Total ret — gov (%)":      row_b["total_ret_%"],
            "Δ Total ret (pp)":         round(row_b["total_ret_%"] - row_a["total_ret_%"], 2),

            # Annualised return
            "Ann. ret — no gov (%)":    row_a["ann_ret_%"],
            "Ann. ret — gov (%)":       row_b["ann_ret_%"],
            "Δ Ann. ret (pp)":          round(row_b["ann_ret_%"] - row_a["ann_ret_%"], 2),

            # Annualised volatility
            "Ann. vol — no gov (%)":    row_a["ann_vol_%"],
            "Ann. vol — gov (%)":       row_b["ann_vol_%"],
            "Δ Ann. vol (pp)":          round(row_b["ann_vol_%"] - row_a["ann_vol_%"], 2),

            # Sharpe ratio
            "Sharpe — no gov":          row_a["sharpe_ratio"],
            "Sharpe — gov":             row_b["sharpe_ratio"],
            "Δ Sharpe":                 round(row_b["sharpe_ratio"] - row_a["sharpe_ratio"], 4),

            # Max drawdown
            "Max DD — no gov (%)":      row_a["max_dd_%"],
            "Max DD — gov (%)":         row_b["max_dd_%"],
            "Δ Max DD (pp)":            round(row_b["max_dd_%"] - row_a["max_dd_%"], 2),
        })

    return pd.DataFrame(comparison_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Chart generators
# ─────────────────────────────────────────────────────────────────────────────

def _chart_sharpe_comparison(comparison: pd.DataFrame) -> None:
    """
    Figure 5 — Side-by-side Sharpe ratio: with governance vs without.
    Saved to outputs/figure5_sharpe_comparison.png
    """
    tickers  = comparison["Ticker"].tolist()
    x        = np.arange(len(tickers))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 1.4), 5))

    bars_a = ax.bar(x - width / 2, comparison["Sharpe — no gov"],
                    width, label="No Governance (Run A)",
                    color="#dc2626", alpha=0.82, edgecolor="white")
    bars_b = ax.bar(x + width / 2, comparison["Sharpe — gov"],
                    width, label="With Governance (Run B)",
                    color="#16a34a", alpha=0.82, edgecolor="white")

    # Value labels on bars
    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=10)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(
        "Figure 5. Impact of Data Governance on Sharpe Ratio\n"
        "(BVB 2024–2025, rf = 6.5%)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = OUT_DIR / "figure5_sharpe_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def _chart_return_comparison(comparison: pd.DataFrame) -> None:
    """
    Figure 6 — Side-by-side total return: with governance vs without.
    Saved to outputs/figure6_return_comparison.png
    """
    tickers = comparison["Ticker"].tolist()
    x       = np.arange(len(tickers))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 1.4), 5))

    bars_a = ax.bar(x - width / 2, comparison["Total ret — no gov (%)"],
                    width, label="No Governance (Run A)",
                    color="#dc2626", alpha=0.82, edgecolor="white")
    bars_b = ax.bar(x + width / 2, comparison["Total ret — gov (%)"],
                    width, label="With Governance (Run B)",
                    color="#16a34a", alpha=0.82, edgecolor="white")

    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=10)
    ax.set_ylabel("Total Return (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title(
        "Figure 6. Impact of Data Governance on Total Return\n"
        "(BVB 2024–2025, dividend-adjusted)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = OUT_DIR / "figure6_return_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Excel export
# ─────────────────────────────────────────────────────────────────────────────

def export_excel(
    comparison: pd.DataFrame,
    metrics_a:  pd.DataFrame,
    metrics_b:  pd.DataFrame,
    df_quarantine: pd.DataFrame,
) -> str:
    """
    Write the full comparison to a timestamped Excel file.

    Sheets:
      Table4 — Governance Impact   : the main thesis comparison table
      Run A — No Governance        : full metrics, no quarantine
      Run B — With Governance      : full metrics, with quarantine
      Quarantine Detail            : every flagged row with its reason
      Summary Stats                : aggregate impact across all tickers
    """
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"Table4_governance_impact_{ts}.xlsx"

    # Aggregate summary row
    n_quarantined = len(df_quarantine) if not df_quarantine.empty else 0
    total_rows_a  = metrics_a["rows_used"].sum()

    summary = pd.DataFrame([{
        "Metric":                           "Across all tickers",
        "Total rows (no governance)":       total_rows_a,
        "Total rows (with governance)":     metrics_b["rows_used"].sum(),
        "Rows quarantined":                 n_quarantined,
        "Quarantine rate (%)":              round(n_quarantined / max(total_rows_a, 1) * 100, 3),
        "Avg Sharpe — no gov":              round(metrics_a["sharpe_ratio"].mean(), 4),
        "Avg Sharpe — gov":                 round(metrics_b["sharpe_ratio"].mean(), 4),
        "Avg Sharpe Δ":                     round(
            metrics_b["sharpe_ratio"].mean() - metrics_a["sharpe_ratio"].mean(), 4
        ),
        "Avg ann. vol — no gov (%)":        round(metrics_a["ann_vol_%"].mean(), 2),
        "Avg ann. vol — gov (%)":           round(metrics_b["ann_vol_%"].mean(), 2),
    }])

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:

            # Table 4 — the main thesis table
            comparison.to_excel(
                writer, sheet_name="Table4 — Governance Impact", index=False
            )

            # Raw metrics per run
            metrics_a.drop(columns=["run"], errors="ignore").to_excel(
                writer, sheet_name="Run A — No Governance", index=False
            )
            metrics_b.drop(columns=["run"], errors="ignore").to_excel(
                writer, sheet_name="Run B — With Governance", index=False
            )

            # Quarantine detail — every flagged row
            if not df_quarantine.empty:
                detail_cols = [
                    "date", "ticker", "company", "sector",
                    "close", "daily_return", "reason",
                ]
                detail = df_quarantine[
                    [c for c in detail_cols if c in df_quarantine.columns]
                ].copy()
                if "daily_return" in detail.columns:
                    detail["daily_return_%"] = (detail["daily_return"] * 100).round(2)
                    detail = detail.drop(columns=["daily_return"])
                detail.to_excel(
                    writer, sheet_name="Quarantine Detail", index=False
                )
            else:
                pd.DataFrame([{"note": "No rows quarantined in this run."}]).to_excel(
                    writer, sheet_name="Quarantine Detail", index=False
                )

            # Summary stats
            summary.to_excel(
                writer, sheet_name="Summary Stats", index=False
            )

        log.info(f"Excel saved: {path.name}")
        return str(path)

    except PermissionError:
        # OneDrive / Excel lock fallback
        fallback = Path.home() / "Desktop" / f"Table4_governance_impact_{ts}.xlsx"
        with pd.ExcelWriter(fallback, engine="openpyxl") as writer:
            comparison.to_excel(writer, sheet_name="Table4 — Governance Impact", index=False)
        log.warning(f"outputs/ locked — saved to Desktop: {fallback.name}")
        return str(fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(comparison: pd.DataFrame, df_quarantine: pd.DataFrame) -> None:
    """
    Print a clean summary table to the terminal so you can read the
    findings immediately without opening Excel.
    """
    n_q   = len(df_quarantine) if not df_quarantine.empty else 0
    total = comparison["Rows quarantined"].sum()

    print("\n" + "═" * 72)
    print("  GOVERNANCE IMPACT ANALYSIS — BVB 2024–2025")
    print("═" * 72)
    print(f"\n  Rows quarantined total : {n_q}")
    print(f"  Tickers affected       : {comparison[comparison['Rows quarantined'] > 0]['Ticker'].nunique()}")

    print("\n  ── Sharpe Ratio ────────────────────────────────────────────────")
    print(f"  {'Ticker':<10} {'No Gov':>10} {'Gov':>10} {'Δ':>10}  {'Rows Q':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*8}")
    for _, row in comparison.iterrows():
        delta = row["Δ Sharpe"]
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(
            f"  {row['Ticker']:<10} "
            f"{row['Sharpe — no gov']:>10.4f} "
            f"{row['Sharpe — gov']:>10.4f} "
            f"{arrow}{abs(delta):>9.4f}  "
            f"{int(row['Rows quarantined']):>8}"
        )

    print("\n  ── Total Return ────────────────────────────────────────────────")
    print(f"  {'Ticker':<10} {'No Gov %':>10} {'Gov %':>10} {'Δ pp':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in comparison.iterrows():
        delta = row["Δ Total ret (pp)"]
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(
            f"  {row['Ticker']:<10} "
            f"{row['Total ret — no gov (%)']:>9.2f}% "
            f"{row['Total ret — gov (%)']:>9.2f}% "
            f"{arrow}{abs(delta):>8.2f}pp"
        )

    print("\n  ── Interpretation ──────────────────────────────────────────────")
    max_sharpe_delta = comparison["Δ Sharpe"].abs().max()
    max_sharpe_ticker = comparison.loc[comparison["Δ Sharpe"].abs().idxmax(), "Ticker"]
    max_ret_delta    = comparison["Δ Total ret (pp)"].abs().max()
    max_ret_ticker   = comparison.loc[comparison["Δ Total ret (pp)"].abs().idxmax(), "Ticker"]
    print(
        f"  Largest Sharpe distortion : {max_sharpe_ticker} by {max_sharpe_delta:.4f} units\n"
        f"  Largest return distortion : {max_ret_ticker} by {max_ret_delta:.2f} percentage points\n"
        f"\n  These figures are the empirical evidence that data governance\n"
        f"  materially affects analytics outcomes on BVB data."
    )
    print("═" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   GOVERNANCE IMPACT COMPARISON — starting            ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    cfg = load_config()

    # ── 1. Extract raw data (shared by both runs) ─────────────────────────────
    log.info("Extracting raw data (used by both runs) ...")
    df_raw = extract_run(cfg)

    # ── 2. Run A — no governance ──────────────────────────────────────────────
    df_a = run_without_governance(df_raw, cfg)

    # ── 3. Run B — with governance ────────────────────────────────────────────
    df_b, df_quarantine = run_with_governance(df_raw, cfg)

    # ── 4. Compute metrics for each run ───────────────────────────────────────
    log.info("Computing metrics for Run A ...")
    metrics_a = compute_metrics(df_a, cfg, label="No Governance")

    log.info("Computing metrics for Run B ...")
    metrics_b = compute_metrics(df_b, cfg, label="With Governance")

    # ── 5. Build comparison table ─────────────────────────────────────────────
    log.info("Building comparison table ...")
    comparison = build_comparison_table(metrics_a, metrics_b, df_quarantine)

    # ── 6. Print terminal summary ─────────────────────────────────────────────
    print_summary(comparison, df_quarantine)

    # ── 7. Generate charts ────────────────────────────────────────────────────
    log.info("Generating comparison charts ...")
    _chart_sharpe_comparison(comparison)
    _chart_return_comparison(comparison)

    # ── 8. Export Excel ───────────────────────────────────────────────────────
    log.info("Exporting Excel report ...")
    excel_path = export_excel(comparison, metrics_a, metrics_b, df_quarantine)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   GOVERNANCE IMPACT COMPARISON — complete            ║")
    log.info("║                                                      ║")
    log.info("║  Outputs saved to outputs/:                          ║")
    log.info("║    Table4_governance_impact_*.xlsx                   ║")
    log.info("║    figure5_sharpe_comparison.png                     ║")
    log.info("║    figure6_return_comparison.png                     ║")
    log.info("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
