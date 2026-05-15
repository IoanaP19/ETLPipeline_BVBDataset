"""
src/analyse.py
--------------
STEP 4 — ANALYSE  &  STEP 5 — REPORT

Computes three business analytics use cases and persists results.
No static charts are produced here. All visualisation is handled
by dashboard.py, which queries bvb_warehouse.db directly.

Use Case 1 — Price & Return Analysis
    Q: Which BVB stocks performed best in 2024–2025?
    Output: summary DataFrame → Table1 in Excel

Use Case 2 — Risk & Volatility
    Q: Which stocks offered the best risk-adjusted return?
    Output: summary DataFrame → Table2 in Excel

Use Case 3 — Sector Rotation
    Q: Which sector led each quarter?
    Output: pivot DataFrame → Table3 in Excel

Report
    All three summary tables + a data-quality log are written to a
    single timestamped Excel file. The Streamlit dashboard reads
    directly from bvb_warehouse.db, not from these Excel tables.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)


def run(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Execute all three use cases and write the Excel report.

    Parameters
    ----------
    df  : transformed DataFrame from transform.run()
    cfg : full configuration dictionary loaded from config.json

    Returns
    -------
    dict with keys:
        uc1        : pd.DataFrame — return ranking per ticker
        uc2        : pd.DataFrame — risk metrics per ticker
        uc3        : pd.DataFrame — sector rotation pivot table
        excel_path : str          — absolute path of the saved Excel file
    """
    log.info("=" * 55)
    log.info("STEP 4 — ANALYSE")
    log.info("=" * 55)

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ana_cfg   = cfg["analysis"]
    risk_free = ana_cfg["risk_free_rate"]
    tdays     = ana_cfg["trading_days_per_year"]

    uc1 = _use_case_1_returns(df)
    uc2 = _use_case_2_risk(df, risk_free, tdays)
    uc3 = _use_case_3_sectors(df)

    log.info("=" * 55)
    log.info("STEP 5 — REPORT")
    log.info("=" * 55)

    dq_threshold = cfg["data_quality"]["dq_log_threshold"]
    excel_path   = _write_excel(df, uc1, uc2, uc3, dq_threshold, out_dir)

    return {"uc1": uc1, "uc2": uc2, "uc3": uc3, "excel_path": excel_path}


# -----------------------------------------------------------------------------
# Use Case 1 — Price & Return Analysis
# -----------------------------------------------------------------------------

def _use_case_1_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total return per ticker over the full period.

    Returns a DataFrame ranked by total_return_%. The cumulative_return
    column in bvb_warehouse.db is available as a time series for the
    dashboard to chart interactively.
    """
    log.info("UC1 — Price & Return Analysis ...")

    summary = (
        df.groupby(["ticker", "company", "sector"])
        .apply(
            lambda g: pd.Series({
                "start_price_RON": round(g["close"].iloc[0], 2),
                "end_price_RON":   round(g["close"].iloc[-1], 2),
                "total_return_%":  round(g["cumulative_return"].iloc[-1] * 100, 2),
            }),
            include_groups=False,
        )
        .reset_index()
        .sort_values("total_return_%", ascending=False)
        .reset_index(drop=True)
    )

    log.info(f"UC1 complete — {len(summary)} tickers ranked by return:")
    for _, row in summary.iterrows():
        log.info(
            f"  {row['ticker']:<10} {row['company']:<28} "
            f"{row['total_return_%']:>+7.2f}%"
        )

    return summary


# -----------------------------------------------------------------------------
# Use Case 2 — Risk & Volatility
# -----------------------------------------------------------------------------

def _use_case_2_risk(
    df: pd.DataFrame,
    risk_free: float,
    tdays: int,
) -> pd.DataFrame:
    """
    Compute annualised return, volatility, Sharpe ratio, and max drawdown
    per ticker over the full period.

    The drawdown column in bvb_warehouse.db contains the rolling daily
    drawdown time series; here we reduce it to a single worst-case
    summary value for the Excel report.
    """
    log.info("UC2 — Risk & Volatility ...")

    rows = []
    for ticker, g in df.groupby("ticker"):
        g    = g.sort_values("date")
        rets = g["daily_return"].dropna()
        px   = g["close"].dropna()

        if len(rets) < 20:
            log.debug(f"UC2: skipping {ticker} — fewer than 20 observations")
            continue

        # Annualised return: compound price change scaled to one year
        ann_ret = (px.iloc[-1] / px.iloc[0]) ** (tdays / len(g)) - 1

        # Annualised volatility: daily std × √trading_days
        ann_vol = rets.std() * np.sqrt(tdays)

        # Sharpe ratio: excess return above risk-free rate per unit of risk
        sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0.0

        # Max drawdown: worst peak-to-trough loss across the full period.
        # Use the pre-computed drawdown column if available (added in transform),
        # otherwise recompute it here as a fallback.
        if "drawdown" in g.columns:
            max_dd = g["drawdown"].min()
        else:
            roll_max = px.cummax()
            max_dd   = ((px - roll_max) / roll_max).min()

        rows.append({
            "ticker":       ticker,
            "company":      g["company"].iloc[0],
            "ann_return_%": round(ann_ret * 100, 2),
            "ann_vol_%":    round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_dd_%":     round(max_dd * 100, 2),
        })

    summary = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)

    log.info(f"UC2 complete — risk metrics for {len(summary)} tickers:")
    for _, row in summary.iterrows():
        log.info(
            f"  {row['ticker']:<10} Sharpe={row['sharpe_ratio']:>+6.3f}  "
            f"Vol={row['ann_vol_%']:>5.1f}%  MaxDD={row['max_dd_%']:>6.1f}%"
        )

    return summary


# -----------------------------------------------------------------------------
# Use Case 3 — Sector Rotation
# -----------------------------------------------------------------------------

def _use_case_3_sectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the average monthly return per sector per quarter.

    Returns a pivot table (sectors as rows, quarters as columns).
    The dashboard reads sector and quarter directly from bvb_warehouse.db
    to build the interactive sector rotation chart.
    """
    log.info("UC3 — Sector Rotation ...")

    sector_q = (
        df.sort_values("date")
        .groupby(["sector", "quarter"])
        .apply(
            # avg daily return × 21 trading days ≈ average monthly return
            lambda g: round(g["daily_return"].mean() * 21 * 100, 2),
            include_groups=False,
        )
        .reset_index(name="avg_monthly_ret_%")
    )

    pivot = sector_q.pivot(
        index="sector", columns="quarter", values="avg_monthly_ret_%"
    )

    log.info("UC3 complete — sector rotation pivot (avg monthly return %):")
    for line in pivot.to_string().splitlines():
        log.info(f"  {line}")

    return pivot


# -----------------------------------------------------------------------------
# Report — Excel export
# -----------------------------------------------------------------------------

def _write_excel(
    df: pd.DataFrame,
    uc1: pd.DataFrame,
    uc2: pd.DataFrame,
    uc3: pd.DataFrame,
    dq_threshold: float,
    out_dir: Path,
) -> str:
    """
    Write all three summary tables plus a data-quality log to a single
    timestamped Excel file.

    Timestamped filename prevents PermissionError when a previous run's
    file is still open in Excel. Falls back to Desktop if the outputs/
    folder is locked by OneDrive sync.
    """
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = out_dir / f"bvb_etl_report_{ts}.xlsx"

    try:
        excel_path.open("ab").close()
    except PermissionError:
        excel_path = Path.home() / "Desktop" / f"bvb_etl_report_{ts}.xlsx"
        log.warning(f"outputs/ locked by OS — saving to Desktop: {excel_path}")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # Table 1 — Return ranking (UC1)
        uc1.to_excel(writer, sheet_name="Table1 — Returns", index=False)

        # Table 2 — Risk metrics (UC2)
        uc2.to_excel(writer, sheet_name="Table2 — Risk", index=False)

        # Table 3 — Sector rotation pivot (UC3)
        uc3.to_excel(writer, sheet_name="Table3 — Sectors")

        # Data Quality Log — top 30 largest daily moves that passed all filters.
        # Useful for the thesis to show which real market events drove
        # extreme returns (e.g. earnings surprises, dividend announcements).
        dq = df.copy()
        dq["abs_return"] = dq["daily_return"].abs()
        suspicious = (
            dq[dq["abs_return"] > dq_threshold]
            [["date", "ticker", "company", "close", "daily_return", "abs_return"]]
            .copy()
        )
        suspicious["daily_return_%"] = (suspicious["daily_return"] * 100).round(2)
        suspicious = (
            suspicious
            .drop(columns=["daily_return"])
            .sort_values("abs_return", ascending=False)
            .head(30)
            .drop(columns=["abs_return"])
        )
        suspicious.to_excel(writer, sheet_name="Data Quality Log", index=False)

    log.info(f"Excel report saved → {excel_path}")
    return str(excel_path)
