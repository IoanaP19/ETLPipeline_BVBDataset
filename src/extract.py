"""
src/extract.py
--------------
STEP 1 — EXTRACT

Downloads daily dividend-adjusted closing prices for BVB-listed companies
from Yahoo Finance using the yfinance library.

Key design decisions:
  - auto_adjust=True  : prices are adjusted for dividends and stock splits,
                        giving a smooth return series without artificial jumps.
  - Long format output: one row per (ticker, date) — easier to filter and join.
  - Configuration is read from config.json, not hardcoded.
"""

import pandas as pd
import yfinance as yf

from src.logger import get_logger

log = get_logger(__name__)


def run(cfg: dict) -> pd.DataFrame:
    """
    Download price data for all tickers defined in config.json.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary loaded from config.json.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        date, ticker, company, sector, open, high, low, close, volume
    """
    companies  = cfg["companies"]
    start_date = cfg["pipeline"]["start_date"]
    end_date   = cfg["pipeline"]["end_date"]

    log.info("=" * 55)
    log.info("STEP 1 — EXTRACT")
    log.info(f"Period  : {start_date} → {end_date}")
    log.info(f"Tickers : {list(companies.keys())}")
    log.info("=" * 55)

    all_frames = []

    for ticker, meta in companies.items():
        company_name = meta["name"]
        sector       = meta["sector"]
        log.info(f"Downloading {ticker} — {company_name} ...")

        raw = _download_ticker(ticker, start_date, end_date)

        if raw is None or raw.empty:
            log.warning(f"No data returned for {ticker} — skipping.")
            continue

        raw["ticker"]  = ticker
        raw["company"] = company_name
        raw["sector"]  = sector

        log.info(
            f"  {ticker} OK — {len(raw)} rows | "
            f"price range {raw['close'].min():.2f}–{raw['close'].max():.2f} RON"
        )
        all_frames.append(raw)

    if not all_frames:
        raise RuntimeError("Extract failed: no data downloaded for any ticker.")

    df = pd.concat(all_frames, ignore_index=True)

    # Keep only the columns we need, in a consistent order
    keep = ["date", "ticker", "company", "sector",
            "open", "high", "low", "close", "volume"]
    df   = df[[c for c in keep if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info(f"EXTRACT complete — {len(df):,} rows, {df['ticker'].nunique()} tickers")
    return df


def _download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download a single ticker from Yahoo Finance and return a clean flat
    DataFrame with a lowercase 'date' column.

    Handles the yfinance API change in v0.2.40+ where multi_level_index
    was introduced. Falls back gracefully for older versions.
    """
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            # auto_adjust=True: prices corrected for dividends and splits.
            # Without this, a dividend payment looks like a fake price crash,
            # which inflates cumulative returns by the dividend amount each time.
            auto_adjust=True,
            progress=False,
            multi_level_index=False,   # yfinance >= 0.2.40: force flat columns
        )
    except TypeError:
        # Older yfinance does not support multi_level_index kwarg
        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False,
        )
    except Exception as exc:
        log.error(f"Download failed for {ticker}: {exc}")
        return None

    if raw is None or raw.empty:
        return None

    # Flatten MultiIndex columns if still present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(col[0]).lower().replace(" ", "_") for col in raw.columns]
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]

    # Date lives in the index — move it to a regular column
    raw = raw.reset_index()

    # yfinance names the date column differently across versions
    for candidate in ["date", "Date", "Datetime", "datetime", "index"]:
        if candidate in raw.columns:
            raw.rename(columns={candidate: "date"}, inplace=True)
            break
    else:
        # Last resort: first column is almost always the date
        raw.rename(columns={raw.columns[0]: "date"}, inplace=True)
        log.debug(f"{ticker}: date column guessed as '{raw.columns[0]}'")

    # With auto_adjust=True, 'Close' IS the adjusted close — no separate column
    if "close" not in raw.columns:
        log.warning(f"{ticker}: 'close' column not found — available: {raw.columns.tolist()}")
        return None

    return raw
