"""
data_loader.py
==============
Data loading and preprocessing utilities.

Supports:
  - Online download via yfinance (primary)
  - Local CSV loading (Kaggle Stock Market Dataset)

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker universe (50 liquid NASDAQ / S&P 500 names across 10 sectors)
# ---------------------------------------------------------------------------
UNIVERSE: List[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC", "CSCO", "ORCL", "QCOM", "TXN",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "V", "MA",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "AMGN",
    # Consumer Staples
    "WMT", "COST", "PG", "KO", "PEP",
    # Consumer Discretionary
    "AMZN", "MCD", "NKE", "HD", "TGT",
    # Communication
    "META", "NFLX", "DIS", "VZ",
    # Energy
    "XOM", "CVX", "COP",
    # Industrials
    "BA", "CAT", "GE", "HON",
    # Utilities
    "NEE", "DUK",
    # Real Estate
    "AMT", "PLD",
]

SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology",  "MSFT": "Technology",  "GOOGL": "Technology",
    "NVDA": "Technology",  "AMD":  "Technology",   "INTC":  "Technology",
    "CSCO": "Technology",  "ORCL": "Technology",   "QCOM":  "Technology",
    "TXN":  "Technology",
    "JPM":  "Financials",  "BAC":  "Financials",   "GS":    "Financials",
    "MS":   "Financials",  "WFC":  "Financials",   "BLK":   "Financials",
    "V":    "Financials",  "MA":   "Financials",
    "JNJ":  "Healthcare",  "PFE":  "Healthcare",   "MRK":   "Healthcare",
    "ABBV": "Healthcare",  "UNH":  "Healthcare",   "AMGN":  "Healthcare",
    "WMT":  "Consumer Staples", "COST": "Consumer Staples",
    "PG":   "Consumer Staples", "KO":   "Consumer Staples",
    "PEP":  "Consumer Staples",
    "AMZN": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE":  "Consumer Discretionary", "HD":  "Consumer Discretionary",
    "TGT":  "Consumer Discretionary",
    "META": "Communication", "NFLX": "Communication",
    "DIS":  "Communication", "VZ":   "Communication",
    "XOM":  "Energy",   "CVX":  "Energy",   "COP":  "Energy",
    "BA":   "Industrials", "CAT": "Industrials",
    "GE":   "Industrials", "HON": "Industrials",
    "NEE":  "Utilities", "DUK":  "Utilities",
    "AMT":  "Real Estate", "PLD": "Real Estate",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_prices(
    tickers: List[str] = UNIVERSE,
    start: str = "2015-01-01",
    end: str = "2024-01-01",
    save_path: Optional[str] = "data/prices.csv",
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.

    Steps
    -----
    1. Fetch OHLCV with auto-adjustment (splits + dividends).
    2. Keep only 'Close'.
    3. Drop tickers with > 10 % missing values.
    4. Forward-fill remaining gaps (≤ 5 consecutive days).
    5. Drop any remaining NaN rows.

    Parameters
    ----------
    tickers   : list of ticker symbols
    start,end : ISO date strings
    save_path : optional CSV cache path

    Returns
    -------
    prices : pd.DataFrame  shape (T × N)
    """
    log.info(f"Downloading {len(tickers)} tickers  [{start} → {end}]")

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    # ── Quality filter ────────────────────────────────────────────────────── #
    missing = prices.isna().mean()
    keep    = missing[missing < 0.10].index.tolist()
    dropped = set(tickers) - set(keep)
    if dropped:
        log.warning(f"Dropped {len(dropped)} tickers (>10 % missing): {sorted(dropped)}")

    prices = prices[keep].ffill(limit=5).dropna()
    log.info(f"Clean dataset: {prices.shape[0]} days × {prices.shape[1]} tickers")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(save_path)
        log.info(f"Prices cached → {save_path}")

    return prices


def load_prices(path: str = "data/prices.csv") -> pd.DataFrame:
    """Load a previously cached price CSV."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    log.info(f"Loaded prices {df.shape} from {path}")
    return df


def load_kaggle_prices(kaggle_dir: str) -> pd.DataFrame:
    """
    Load prices from the Kaggle Stock Market Dataset.

    Expected layout: one CSV per ticker in `kaggle_dir/`, with at least
    a 'Date' column and an 'Adj Close' (or 'Close') column.

    Parameters
    ----------
    kaggle_dir : str  path to the unzipped Kaggle folder

    Returns
    -------
    prices : pd.DataFrame  (date × ticker)
    """
    folder = Path(kaggle_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {kaggle_dir}")

    dfs: Dict[str, pd.Series] = {}
    for fp in sorted(folder.glob("*.csv")):
        ticker = fp.stem.upper()
        try:
            df  = pd.read_csv(fp, parse_dates=["Date"], index_col="Date")
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            dfs[ticker] = df[col].rename(ticker)
        except Exception as exc:
            log.debug(f"Skip {fp.name}: {exc}")

    if not dfs:
        raise ValueError(f"No valid CSVs found in {kaggle_dir}")

    prices = pd.concat(dfs, axis=1).sort_index()
    missing = prices.isna().mean()
    keep    = missing[missing < 0.10].index.tolist()
    prices  = prices[keep].ffill(limit=5).dropna()
    log.info(f"Kaggle dataset loaded: {prices.shape}")
    return prices


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    train_end: str = "2020-12-31",
    val_end:   str = "2022-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calendar-based train / validation / test split.

    ⚠  No random shuffling — temporal order preserved to avoid data leakage.
       This is the correct protocol for financial time series
       (cf. TimeSeriesSplit in scikit-learn).

    Parameters
    ----------
    df        : DataFrame with DatetimeIndex
    train_end : last date included in training  (ISO string)
    val_end   : last date included in validation (ISO string)

    Returns
    -------
    train, val, test  DataFrames
    """
    train = df.loc[:train_end]
    val   = df.loc[train_end:val_end].iloc[1:]
    test  = df.loc[val_end:].iloc[1:]

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        log.info(
            f"{name:5s}: {split.index[0].date()} → {split.index[-1].date()}"
            f"  ({len(split):>4} obs)"
        )
    return train, val, test


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def align_dataframes(*dfs: pd.DataFrame) -> List[pd.DataFrame]:
    """Align multiple DataFrames on their common date index."""
    common = dfs[0].index
    for df in dfs[1:]:
        common = common.intersection(df.index)
    return [df.loc[common] for df in dfs]


def get_sector_series(tickers: List[str]) -> pd.Series:
    """Return a Series mapping ticker → sector for the given tickers."""
    return pd.Series({t: SECTOR_MAP.get(t, "Unknown") for t in tickers},
                     name="sector")
