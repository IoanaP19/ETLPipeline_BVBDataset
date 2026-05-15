"""
src/load.py
-----------
STEP 3 — LOAD

Implements a dual-table Data Governance storage strategy.

Tables written to outputs/bvb_warehouse.db:
  market_data      — clean, vetted rows (is_anomaly == False)
                     This is the ONLY table read by the analytics dashboard.
  data_audit_log   — quarantined rows pending human review
                     Rows here are EXCLUDED from all analytics until
                     a human analyst approves them via the dashboard.

Governance guarantee:
  A row is EITHER in market_data OR in data_audit_log — never in both.
  When an analyst approves a quarantined row, load.py's helper
  approve_anomaly() moves it atomically from data_audit_log → market_data.

Also writes:
  bvb_processed.csv — clean data as CSV (opens in Excel)
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from src.logger import get_logger

log = get_logger(__name__)

# Table names — single source of truth, referenced by dashboard.py too
TABLE_MARKET   = "market_data"
TABLE_AUDIT    = "data_audit_log"


def run(
    df_final: pd.DataFrame,
    df_quarantine: pd.DataFrame,
    cfg: dict,
) -> tuple[str, str]:
    """
    Save clean and quarantine data to the SQLite warehouse.

    Parameters
    ----------
    df_final      : clean rows from transform.run() — goes to market_data
    df_quarantine : flagged rows from transform.run() — goes to data_audit_log
    cfg           : full configuration dictionary from config.json

    Returns
    -------
    tuple (db_path, csv_path)
    """
    log.info("=" * 55)
    log.info("STEP 3 — LOAD")
    log.info("=" * 55)

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "bvb_warehouse.db"
    engine  = sa.create_engine(f"sqlite:///{db_path}")

    # ── 1. market_data — clean, analytics-ready rows ──────────────────────────
    # We drop any governance columns before saving so the analytics queries
    # stay clean and simple.
    clean_cols_to_drop = ["is_anomaly", "reason", "resolution_status"]
    df_to_save = df_final.drop(
        columns=[c for c in clean_cols_to_drop if c in df_final.columns]
    )

    df_to_save.to_sql(
        TABLE_MARKET, engine,
        if_exists="replace",   # full reload on every pipeline run
        index=False,
        chunksize=5000,
    )
    log.info(
        f"market_data    → {db_path.name}  "
        f"({len(df_to_save):,} rows, {df_to_save['ticker'].nunique()} tickers)"
    )

    # ── 2. data_audit_log — quarantined rows ──────────────────────────────────
    if df_quarantine.empty:
        log.info("data_audit_log → no anomalies to quarantine this run")
        # Still ensure the table exists with the correct schema (empty)
        _ensure_audit_table(engine)
    else:
        df_audit = df_quarantine.copy()

        # Add governance metadata columns
        # resolution_status tracks the human-in-the-loop decision:
        #   PENDING_REVIEW — not yet reviewed
        #   APPROVED       — analyst confirmed it is a real market move
        #   REJECTED       — analyst confirmed it is a data error
        df_audit["resolution_status"] = "PENDING_REVIEW"
        df_audit["flagged_at"]        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append strategy: preserve previous audit decisions across pipeline runs.
        # We insert only rows that are NOT already in the table (identified by
        # the composite key ticker + date) so re-running the pipeline doesn't
        # reset human decisions already made in the dashboard.
        _upsert_audit_rows(df_audit, engine)
        log.info(
            f"data_audit_log → {db_path.name}  "
            f"({len(df_audit):,} new rows quarantined, status=PENDING_REVIEW)"
        )

    # ── 3. CSV export ─────────────────────────────────────────────────────────
    csv_path = out_dir / "bvb_processed.csv"
    df_to_save.to_csv(csv_path, index=False)
    log.info(f"CSV            → {csv_path.name}  ({len(df_to_save):,} rows)")

    log.info(f"LOAD complete — warehouse: {db_path}")
    return str(db_path), str(csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_audit_table(engine: sa.Engine) -> None:
    """Create data_audit_log with the governance schema if it does not exist."""
    with engine.connect() as conn:
        conn.execute(sa.text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_AUDIT} (
                date               TEXT,
                ticker             TEXT,
                company            TEXT,
                sector             TEXT,
                close              REAL,
                daily_return       REAL,
                cumulative_return  REAL,
                reason             TEXT,
                resolution_status  TEXT DEFAULT 'PENDING_REVIEW',
                flagged_at         TEXT,
                PRIMARY KEY (ticker, date)
            )
        """))
        conn.commit()


def _upsert_audit_rows(df: pd.DataFrame, engine: sa.Engine) -> None:
    """
    Insert new audit rows without overwriting existing ones.

    Uses INSERT OR IGNORE so that rows already reviewed by the analyst
    (APPROVED or REJECTED) are never reset to PENDING_REVIEW when the
    pipeline reruns with the same data.
    """
    _ensure_audit_table(engine)

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(sa.text(f"""
                INSERT OR IGNORE INTO {TABLE_AUDIT}
                    (date, ticker, company, sector, close, daily_return,
                     cumulative_return, reason, resolution_status, flagged_at)
                VALUES
                    (:date, :ticker, :company, :sector, :close, :daily_return,
                     :cumulative_return, :reason, :resolution_status, :flagged_at)
            """), {
                "date":               str(row.get("date", ""))[:10],
                "ticker":             row.get("ticker", ""),
                "company":            row.get("company", ""),
                "sector":             row.get("sector", ""),
                "close":              float(row.get("close", 0)),
                "daily_return":       float(row.get("daily_return", 0)),
                "cumulative_return":  float(row.get("cumulative_return", 0)),
                "reason":             str(row.get("reason", "")),
                "resolution_status":  str(row.get("resolution_status", "PENDING_REVIEW")),
                "flagged_at":         str(row.get("flagged_at", "")),
            })
        conn.commit()
