"""
src/transform.py
----------------
STEP 2 — TRANSFORM

Cleans raw price data, engineers all financial features, and implements
the Quarantine pattern for Data Governance.

Pipeline (in order):
  1. Basic cleaning       — nulls, zero prices, zero volume, duplicates
  2. Price spike filter   — removes physically impossible price jumps
  3. Daily returns        — pct_change() grouped by ticker
  4. Return outlier cap   — hard-removes |return| > 25% (API errors)
  5. Cumulative return    — compounded total return from start of period
  6. Rolling volatility   — annualised 30-day standard deviation
  7. Calendar fields      — year, month, quarter for grouping
  8. Moving averages      — SMA_50 and SMA_200 on adjusted close price
  9. Maximum drawdown     — rolling % drop from the highest cumulative peak
 10. Quarantine flagging  — flags is_anomaly for |return| > volatility_threshold
 11. Quarantine split     — separates df_final and df_quarantine
 12. Sanity check         — logs year-by-year return table

QUARANTINE LOGIC (steps 10–11):
  The BVB has a 15% daily circuit-breaker limit. Any move at or beyond this
  threshold is ambiguous: it could be a genuine market event (e.g. a takeover
  bid, major earnings surprise) OR it could be a silent data error from the
  upstream provider (yfinance) that slipped past the hard 25% removal filter.

  Instead of silently removing or silently keeping these rows, we quarantine
  them. A human analyst can then review each one in the dashboard and either:
    - Approve it  → row moves to market_data for analytics
    - Reject it   → row stays in data_audit_log, excluded from analytics

  This "Human-in-the-Loop" pattern is standard practice in production ETL
  pipelines for financial data, and directly applicable to emerging markets
  where data quality is inconsistent.

Return value:
  tuple[pd.DataFrame, pd.DataFrame]
    df_final      — rows where is_anomaly == False  → loaded into market_data
    df_quarantine — rows where is_anomaly == True   → loaded into data_audit_log
"""

import numpy as np
import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)


def run(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full transformation pipeline.

    Parameters
    ----------
    df  : raw DataFrame from extract.run()
    cfg : full configuration dictionary from config.json

    Returns
    -------
    tuple (df_final, df_quarantine)
        df_final      : clean rows ready for analytics
        df_quarantine : anomalous rows pending human review
    """
    dq_cfg  = cfg["data_quality"]
    ana_cfg = cfg["analysis"]
    tdays   = ana_cfg["trading_days_per_year"]

    log.info("=" * 55)
    log.info("STEP 2 — TRANSFORM")
    log.info("=" * 55)

    df = df.copy()

    # ── 1. Basic cleaning ─────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    if "volume" in df.columns:
        df = df[df["volume"] > 0]
    df = df.drop_duplicates(subset=["date", "ticker"])
    log.info(f"Basic cleaning: {before - len(df)} rows removed — {len(df):,} remain")

    # ── 2. Price spike filter ─────────────────────────────────────────────────
    df = _remove_price_spikes(df, dq_cfg)

    # ── 3. Daily returns ──────────────────────────────────────────────────────
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    log.debug("Daily returns computed via pct_change()")

    # ── 4. Return outlier cap (hard remove at 25%) ────────────────────────────
    # This removes absolute garbage — API decimal errors, feed outages, etc.
    # The 15% quarantine (step 10) handles the ambiguous zone below this cap.
    df = _remove_return_outliers(df, dq_cfg)

    # ── 5. Cumulative return ──────────────────────────────────────────────────
    df["cumulative_return"] = (
        df.groupby("ticker")["daily_return"]
          .transform(lambda x: (1 + x.fillna(0)).cumprod() - 1)
    )
    log.debug("Cumulative returns computed")

    # ── 6. Rolling 30-day annualised volatility ───────────────────────────────
    df["volatility_30d"] = (
        df.groupby("ticker")["daily_return"]
          .transform(lambda x: x.rolling(30).std() * np.sqrt(tdays))
    )
    log.debug("Rolling 30-day annualised volatility computed")

    # ── 7. Calendar fields ────────────────────────────────────────────────────
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    # ── 8. Moving averages ────────────────────────────────────────────────────
    df = _add_moving_averages(df, ana_cfg.get("sma_windows", [50, 200]))

    # ── 9. Maximum drawdown ───────────────────────────────────────────────────
    df = _add_max_drawdown(df)

    # ── 10. Quarantine flagging ───────────────────────────────────────────────
    df = _flag_anomalies(df, dq_cfg)

    # ── 11. Split clean vs quarantine ─────────────────────────────────────────
    df_final, df_quarantine = _split_quarantine(df)

    log.info(f"TRANSFORM complete — {df_final.shape[0]:,} clean rows × {df_final.shape[1]} columns")
    log.info(f"QUARANTINE      — {len(df_quarantine):,} rows flagged for review")
    log.debug(f"Columns: {df_final.columns.tolist()}")

    # ── 12. Sanity check ──────────────────────────────────────────────────────
    _print_sanity_check(df_final)

    return df_final, df_quarantine


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 helpers — existing cleaning steps (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _remove_price_spikes(df: pd.DataFrame, dq_cfg: dict) -> pd.DataFrame:
    """
    Remove rows where closing price deviates more than spike_ratio_upper×
    from the 30-day rolling median. Catches data entry errors (e.g. missing
    decimal point turning 20.00 RON into 200.0 RON).
    """
    upper  = dq_cfg["spike_ratio_upper"]
    lower  = dq_cfg["spike_ratio_lower"]
    window = dq_cfg["spike_window_days"]
    minp   = dq_cfg["spike_min_periods"]

    cleaned, total_removed = [], 0

    for ticker, grp in df.groupby("ticker"):
        grp         = grp.sort_values("date").copy()
        rolling_med = grp["close"].rolling(window=window, min_periods=minp, center=True).median()
        ratio       = grp["close"] / rolling_med
        is_spike    = (ratio > upper) | (ratio < lower)
        n           = int(is_spike.sum())

        if n > 0:
            total_removed += n
            for _, row in grp[is_spike].iterrows():
                log.warning(
                    f"[Price spike] {ticker} on {row['date'].date()} — "
                    f"price={row['close']:.2f} RON "
                    f"(ratio={ratio[is_spike].iloc[0]:.1f}×) — REMOVED"
                )
        cleaned.append(grp[~is_spike])

    if total_removed:
        log.warning(f"Price spike filter: {total_removed} rows removed total")
    else:
        log.info("Price spike filter: no spikes detected")

    return pd.concat(cleaned, ignore_index=True)


def _remove_return_outliers(df: pd.DataFrame, dq_cfg: dict) -> pd.DataFrame:
    """
    Hard-remove rows where |daily_return| > return_outlier_cap (default 25%).
    These are almost certainly feed errors, not real market events.
    Rows in the 15–25% zone are handled by the quarantine step instead.
    """
    cap  = dq_cfg["return_outlier_cap"]
    mask = df["daily_return"].abs() > cap
    n    = int(mask.sum())

    if n:
        log.warning(f"Return outlier filter: hard-removing {n} rows with |return| > {cap*100:.0f}%")
        for _, row in df[mask].iterrows():
            log.warning(
                f"  {row['ticker']} on {row['date'].date()} — "
                f"return={row['daily_return']*100:+.1f}%  close={row['close']:.2f} RON"
            )
        df = df[~mask].copy()
    else:
        log.info(f"Return outlier filter: no returns exceed {cap*100:.0f}%")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 helpers — feature engineering (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _add_moving_averages(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """SMA_50 and SMA_200 — NaN for warm-up rows, intentionally."""
    for w in windows:
        col    = f"SMA_{w}"
        df[col] = (
            df.groupby("ticker")["close"]
              .transform(lambda x: x.rolling(window=w, min_periods=w).mean())
        )
        valid = df.groupby("ticker")[col].count().mean()
        log.info(f"SMA_{w}: computed — first {w-1} rows per ticker are NaN, "
                 f"then {valid:.0f} valid values on average")
    return df


def _add_max_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling drawdown = (price − cummax) / cummax. Always ≤ 0, never NaN."""
    df["drawdown"] = df.groupby("ticker")["close"].transform(
        lambda x: (x - x.cummax()) / x.cummax()
    )
    worst = df.groupby("ticker")["drawdown"].min()
    log.info("Drawdown computed — worst peak-to-trough per ticker:")
    for ticker, val in worst.items():
        log.info(f"  {ticker}: {val*100:.1f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 NEW helpers — quarantine detection and split
# ─────────────────────────────────────────────────────────────────────────────

def _flag_anomalies(df: pd.DataFrame, dq_cfg: dict) -> pd.DataFrame:
    """
    Step 10 — Quarantine Detection.

    Adds a boolean column `is_anomaly` to every row.

    Logic:
      is_anomaly = True   when |daily_return| > volatility_threshold (default 15%)
      is_anomaly = False  for all other rows

    The 15% threshold is the BVB daily price movement limit (circuit-breaker).
    Any move at or above this level is ambiguous — it may be genuine (e.g. a
    takeover announcement) or a silent data error that passed the 25% hard cap.
    Flagging it for human review is the conservative and auditable choice.

    The threshold is read from config.json → data_quality → volatility_threshold
    so it can be changed without modifying any Python code.
    """
    threshold = dq_cfg.get("volatility_threshold", 0.15)

    # Rows with no computed return (first day of each ticker) are never anomalous
    df["is_anomaly"] = (
        df["daily_return"].abs() > threshold
    ) & df["daily_return"].notna()

    n_flagged = int(df["is_anomaly"].sum())

    if n_flagged:
        log.warning(
            f"Quarantine: {n_flagged} rows flagged (|return| > {threshold*100:.0f}%) — "
            f"pending human review in data_audit_log"
        )
        for _, row in df[df["is_anomaly"]].iterrows():
            log.warning(
                f"  [ANOMALY] {row['ticker']} on {row['date'].date()} — "
                f"return={row['daily_return']*100:+.2f}%  close={row['close']:.2f} RON"
            )
    else:
        log.info(f"Quarantine: no anomalies detected above {threshold*100:.0f}% threshold")

    return df


def _split_quarantine(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 11 — Quarantine Split.

    Splits the fully-enriched DataFrame into two non-overlapping subsets:

    df_final
        Rows where is_anomaly == False.
        Safe for analytics — no ambiguous extreme moves.
        Loaded into the `market_data` table.
        The `is_anomaly` column is dropped (it is always False here, so
        it carries no information and would just confuse downstream queries).

    df_quarantine
        Rows where is_anomaly == True.
        Loaded into `data_audit_log` with resolution_status='PENDING_REVIEW'.
        A human analyst reviews these in the dashboard and either approves
        (moves to market_data) or rejects (marks as data error).

        A `reason` column is added describing why each row was flagged,
        using the actual return value so the analyst has immediate context.

    Guarantee: every row appears in exactly one of the two DataFrames.
    """
    mask         = df["is_anomaly"].fillna(False)

    # ── df_final ──────────────────────────────────────────────────────────────
    df_final = (
        df[~mask]
        .drop(columns=["is_anomaly"])
        .reset_index(drop=True)
    )

    # ── df_quarantine ──────────────────────────────────────────────────────────
    df_quarantine = df[mask].copy().reset_index(drop=True)

    if not df_quarantine.empty:
        # Human-readable reason column — shows the actual value that triggered flagging
        df_quarantine["reason"] = df_quarantine["daily_return"].apply(
            lambda r: f"Extreme Volatility Detected: {r*100:+.2f}%"
            if pd.notna(r) else "Extreme Volatility Detected: unknown"
        )
        df_quarantine = df_quarantine.drop(columns=["is_anomaly"])

    log.info(
        f"Split complete — "
        f"df_final: {len(df_final):,} rows | "
        f"df_quarantine: {len(df_quarantine):,} rows"
    )
    return df_final, df_quarantine


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _print_sanity_check(df: pd.DataFrame) -> None:
    """Year-by-year return table for quick verification."""
    log.info("Return sanity check (year-by-year, clean data only):")
    rows = []
    for ticker, g in df.groupby("ticker"):
        g       = g.sort_values("date")
        company = g["company"].iloc[0]

        g24   = g[g["year"] == 2024]
        ret24 = (g24["close"].iloc[-1] / g24["close"].iloc[0] - 1) * 100 if len(g24) > 1 else float("nan")

        g25   = g[g["year"] == 2025]
        ret25 = (g25["close"].iloc[-1] / g25["close"].iloc[0] - 1) * 100 if len(g25) > 1 else float("nan")

        total = g["cumulative_return"].iloc[-1] * 100

        rows.append({
            "ticker":    ticker,
            "company":   company,
            "2024 (%)":  f"{ret24:+.1f}" if not pd.isna(ret24) else "—",
            "2025 (%)":  f"{ret25:+.1f}" if not pd.isna(ret25) else "—",
            "total (%)": f"{total:+.1f}",
            "rows":      len(g),
        })

    table = pd.DataFrame(rows).to_string(index=False)
    for line in table.splitlines():
        log.info(f"  {line}")
    log.info("  NOTE: returns include dividends (auto_adjust=True).")
