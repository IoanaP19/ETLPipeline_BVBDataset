# BVB ETL Pipeline — Bachelor Thesis

> **"An Applied Perspective on ETL: Python-Based Use Cases for Business Analytics"**

## Project structure

```
bvb_etl_pro/
├── main.py            ← run this to execute the full pipeline
├── dashboard.py       ← run this to open the Streamlit dashboard
├── config.json        ← all settings live here, no code changes needed
├── requirements.txt   ← pip dependencies
├── src/
│   ├── logger.py      ← logging: console + outputs/pipeline.log
│   ├── extract.py     ← Step 1: download BVB prices from Yahoo Finance
│   ├── transform.py   ← Step 2: clean data, compute returns & risk metrics
│   ├── load.py        ← Step 3: save to SQLite + CSV
│   └── analyse.py     ← Steps 4–5: charts, tables, Excel report
└── outputs/           ← generated on first run (gitignored)
    ├── bvb_warehouse.db
    ├── bvb_processed.csv
    ├── pipeline.log
    ├── figure1_cumulative_returns.png
    ├── figure2_sharpe_ratio.png
    ├── figure3_monthly_heatmap.png
    ├── figure4_sector_rotation.png
    └── bvb_etl_report_YYYYMMDD_HHMMSS.xlsx
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full ETL pipeline
python main.py

# 3. Open the interactive dashboard
streamlit run dashboard.py
```

## Configuration

All settings are in `config.json` — edit here, no Python needed:

| Key | What it controls |
|-----|-----------------|
| `pipeline.start_date` | Data start date |
| `pipeline.end_date` | Data end date |
| `companies` | Tickers, company names, sectors |
| `data_quality.spike_ratio_upper` | Price spike filter threshold (default 3×) |
| `data_quality.return_outlier_cap` | Max allowed daily return (default 25%) |
| `analysis.risk_free_rate` | NBR rate used for Sharpe ratio |

## Architecture improvements (v3)

| # | Improvement | How |
|---|-------------|-----|
| 1 | **Logging** | `logging` module → console + `outputs/pipeline.log` |
| 2 | **Config** | All hardcoded values moved to `config.json` |
| 3 | **Modular** | `src/extract.py`, `src/transform.py`, `src/load.py`, `src/analyse.py` |
| 4 | **Dependencies** | `requirements.txt` with pinned minimum versions |
| 5 | **Dashboard** | `dashboard.py` → Streamlit reading from SQLite |
