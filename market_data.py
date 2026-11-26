# ------------------------------------------
# MARKET DATA (Part 2 deliverables, read-only)
# ------------------------------------------
# Offline mode only: every number comes from the Part 2 files in data/part2/, so the TAs
# can reproduce the mystery-client result. Per the Part 2 handoff, nothing is re-downloaded.
#   insample_returns_cad.csv  weekly log returns, 2021-04-12 .. 2026-03-30  (estimation only)
#   mu_annualised.csv         mean weekly log return x 52
#   Q_annualised.csv          weekly covariance x 52
#   outsample_prices_cad.csv  daily CAD prices, 2026-04-01 .. 2026-11-18 (execution prices + backtest)
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "part2")

# Asset metadata, copied from Part 3 notebook (Section 1, META) + Part 2 Table 1 (asset type).
# Fossil / AI flags drive ESG exclusions (Part 2: is_fossil_fuel / is_ai_company).
_META = [
    # ticker,   name,                            sector,             asset class,    type,   exch,     ccy,  fossil, ai
    ("RY.TO",   "Royal Bank of Canada",          "Financials",       "Equity",       "Stock", "TSX",    "CAD", 0, 0),
    ("TD.TO",   "TD Bank",                       "Financials",       "Equity",       "Stock", "TSX",    "CAD", 0, 0),
    ("CNR.TO",  "CN Rail",                       "Industrials",      "Equity",       "Stock", "TSX",    "CAD", 0, 0),
    ("ENB.TO",  "Enbridge",                      "Energy",           "Equity",       "Stock", "TSX",    "CAD", 1, 0),
    ("XIU.TO",  "iShares S&P/TSX 60 ETF",        "Broad Market",     "Equity",       "ETF",   "TSX",    "CAD", 0, 0),
    ("XESG.TO", "iShares ESG MSCI Canada ETF",   "Broad Market",     "Equity",       "ETF",   "TSX",    "CAD", 0, 0),
    ("ZAG.TO",  "BMO Aggregate Bond ETF",        "Fixed Income",     "Fixed Income", "ETF",   "TSX",    "CAD", 0, 0),
    ("XRE.TO",  "iShares Canadian REIT ETF",     "Real Estate",      "Equity",       "ETF",   "TSX",    "CAD", 0, 0),
    ("AAPL",    "Apple",                         "Technology",       "Equity",       "Stock", "NASDAQ", "USD", 0, 1),
    ("MSFT",    "Microsoft",                     "Technology",       "Equity",       "Stock", "NASDAQ", "USD", 0, 1),
    ("GOOGL",   "Alphabet",                      "Technology",       "Equity",       "Stock", "NASDAQ", "USD", 0, 1),
    ("JPM",     "JPMorgan Chase",                "Financials",       "Equity",       "Stock", "NYSE",   "USD", 0, 0),
    ("JNJ",     "Johnson & Johnson",             "Healthcare",       "Equity",       "Stock", "NYSE",   "USD", 0, 0),
    ("PG",      "Procter & Gamble",              "Consumer Staples", "Equity",       "Stock", "NYSE",   "USD", 0, 0),
    ("BRK-B",   "Berkshire Hathaway B",          "Diversified",      "Equity",       "Stock", "NYSE",   "USD", 0, 0),
    ("XOM",     "ExxonMobil",                    "Energy",           "Equity",       "Stock", "NYSE",   "USD", 1, 0),
    ("NEE",     "NextEra Energy",                "Utilities",        "Equity",       "Stock", "NYSE",   "USD", 0, 0),
    ("SPY",     "SPDR S&P 500 ETF",              "Broad Market",     "Equity",       "ETF",   "NYSE",   "USD", 0, 0),
    ("QQQ",     "Invesco Nasdaq-100 ETF",        "Technology",       "Equity",       "ETF",   "NASDAQ", "USD", 0, 1),
    ("VTI",     "Vanguard Total Market ETF",     "Broad Market",     "Equity",       "ETF",   "NYSE",   "USD", 0, 0),
    ("ICLN",    "iShares Global Clean Energy",   "Clean Energy",     "Equity",       "ETF",   "NYSE",   "USD", 0, 0),
    ("AGG",     "iShares Core US Bond ETF",      "Fixed Income",     "Fixed Income", "ETF",   "NYSE",   "USD", 0, 0),
    ("GLD",     "SPDR Gold Shares",              "Commodities",      "Commodity",    "ETF",   "NYSE",   "USD", 0, 0),
    ("ACWV",    "iShares MSCI ACWI Min Vol ETF", "Broad Market",     "Equity",       "ETF",   "NYSE",   "USD", 0, 0),
]
META_COLUMNS = ["Ticker", "Name", "Sector", "AssetClass", "Type", "Exchange", "Ccy", "Fossil", "AI"]


def _read(name: str, **kw) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - copy the Part 2 data files into data/part2/.")
    return pd.read_csv(path, index_col=0, **kw)


@lru_cache(maxsize=1)
def load() -> dict:
    """Returns the Part 2 inputs, all in the same ticker order (mu_annualised.csv order)."""
    mu = _read("mu_annualised.csv")["mu"]
    tickers = list(mu.index)
    Q = _read("Q_annualised.csv").loc[tickers, tickers]
    ret_ins = _read("insample_returns_cad.csv", parse_dates=True)[tickers]
    px_out = _read("outsample_prices_cad.csv", parse_dates=True)[tickers]
    meta = pd.DataFrame(_META, columns=META_COLUMNS).set_index("Ticker").loc[tickers]

    # integrity checks (same as Part 3, Section 1)
    assert np.allclose(ret_ins.mean() * 52, mu.values), "mu does not match the in-sample returns"
    assert np.allclose(ret_ins.cov() * 52, Q.values), "Q does not match the in-sample returns"
    assert ret_ins.index.max() < px_out.index.min(), "in-sample and out-of-sample periods overlap"
    return {"tickers": tickers, "mu": mu, "Q": Q, "ret_ins": ret_ins, "px_out": px_out, "meta": meta}
