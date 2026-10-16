"""
Historical Multi-Asset Market Data Ingestion Module
===================================================
Fetches real, split- and dividend-adjusted daily OHLCV prices for liquid
macroeconomic benchmark ETF proxies via Yahoo Finance (yfinance):
  - US_EQUITY:  SPY (SPDR S&P 500 ETF Trust)
  - US_BOND:    TLT (iShares 20+ Year Treasury Bond ETF)
  - GOLD:       GLD (SPDR Gold Shares)
  - USD_INDEX:  UUP (Invesco DB US Dollar Index Bullish Fund)
  - CRUDE_OIL:  USO (United States Oil Fund)

Provides rigorous data validation:
  1. Price positivity and non-zero checks
  2. Bar consistency: High >= max(Open, Close), Low <= min(Open, Close)
  3. Continuous return calculation: daily_return = (Close_t - Close_{t-1}) / Close_{t-1}
  4. Rolling 5-day return volatility calculation
  5. 15-day lookback buffer to prevent boundary NaN losses
  6. Explicit failure reporting (NEVER silently substitutes synthetic data)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Add project root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ASSET_CLASSES, DATA_DIR, RAW_DIR  # noqa: E402

logger = logging.getLogger("CWT-MarketData")

# Benchmark ETF Proxy Mapping
DEFAULT_ETF_MAP: Dict[str, str] = {
    "US_EQUITY": "SPY",
    "US_BOND": "TLT",
    "GOLD": "GLD",
    "USD_INDEX": "UUP",
    "CRUDE_OIL": "USO",
}


class MarketDataFetcher:
    """
    Downloads, validates, and normalizes real historical market prices
    for the quantitative macro strategy engine.
    """

    def __init__(
        self,
        etf_map: Optional[Dict[str, str]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.etf_map = dict(etf_map or DEFAULT_ETF_MAP)
        self.cache_dir = Path(cache_dir) if cache_dir else RAW_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _validate_bar_invariants(self, df: pd.DataFrame, asset: str) -> None:
        """
        Verify that price bars satisfy basic physical market properties:
          - Prices are strictly positive
          - High is >= max(Open, Close)
          - Low is <= min(Open, Close)
        """
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError(f"[{asset}] Non-positive prices detected in downloaded market data.")

        # Relaxed floating point precision check (allow 1e-4 tolerance for rounding)
        tol = 1e-4
        invalid_high = (df["high"] + tol < df[["open", "close"]].max(axis=1)).sum()
        invalid_low = (df["low"] - tol > df[["open", "close"]].min(axis=1)).sum()

        if invalid_high > 0:
            logger.warning(f"[{asset}] {invalid_high} rows had High slightly below Open/Close; adjusting to max.")
            df["high"] = df[["high", "open", "close"]].max(axis=1)

        if invalid_low > 0:
            logger.warning(f"[{asset}] {invalid_low} rows had Low slightly above Open/Close; adjusting to min.")
            df["low"] = df[["low", "open", "close"]].min(axis=1)

    def fetch_ticker_data(
        self,
        asset_class: str,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch real daily adjusted bars for an individual ticker using yfinance.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for real market data ingestion. "
                "Install it via: pip install yfinance"
            ) from e

        logger.info(
            f"Downloading real price history for {asset_class} ({ticker}) "
            f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
        )

        # Download using auto_adjust=True to get dividend/split-adjusted OHLC
        raw_df = yf.download(
            tickers=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if raw_df is None or raw_df.empty:
            raise RuntimeError(
                f"Failed to download price history for {asset_class} ({ticker}). "
                "The source returned an empty DataFrame. Check internet connectivity and ticker status."
            )

        # Handle multi-level columns if returned by yfinance
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            if "Price" in df.columns.names or len(df.columns.levels) > 1:
                # Select the ticker level or drop ticker level
                try:
                    df = df.xs(ticker, axis=1, level="Ticker")
                except (KeyError, ValueError):
                    df.columns = [col[0].lower() for col in df.columns]

        # Standardize column names to lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]

        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"[{asset_class}] Downloaded data missing required price columns. Available: {list(df.columns)}"
            )

        # Ensure index is datetime and sorted
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Convert timezone to America/New_York trading session date
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        # Deduplicate index
        df = df[~df.index.duplicated(keep="last")]

        # Validate bar consistency
        self._validate_bar_invariants(df, asset_class)

        # Calculate accurate daily percentage return
        df["daily_return"] = df["close"].pct_change()

        # Calculate 5-day rolling volatility (sample standard deviation of daily returns)
        df["volatility_5d"] = df["daily_return"].rolling(window=5, min_periods=3).std()

        df["asset_class"] = asset_class
        df["ticker"] = ticker
        df["date"] = df.index.strftime("%Y-%m-%d")

        return df

    def fetch_market_data(
        self,
        days_back: int = 180,
        end_date: Optional[Union[datetime, str]] = None,
        buffer_days: int = 15,
        allow_synthetic_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch aligned historical market data across all target assets.

        Args:
            days_back: Number of calendar days of history required.
            end_date: Anchor end date (defaults to current UTC date).
            buffer_days: Extra lead days downloaded to compute 5-day rolling volatility
                         and first-day returns without dropping boundary observations.
            allow_synthetic_fallback: If True, falls back to synthetic generation upon
                                     critical failure, with explicit warning logs and tags.
                                     Default is False (strict quantitative rigor).

        Returns:
            List of validated dictionary records matching the MarketData schema.
        """
        if end_date is None:
            end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        elif isinstance(end_date, str):
            end_dt = pd.to_datetime(end_date).to_pydatetime()
        else:
            end_dt = end_date

        # Normalize datetime anchors to timezone-naive
        if hasattr(end_dt, "tzinfo") and end_dt.tzinfo is not None:
            end_dt = end_dt.replace(tzinfo=None)

        start_dt = end_dt - timedelta(days=days_back)
        buffer_start_dt = start_dt - timedelta(days=buffer_days)

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        logger.info(
            f"Initiating real market data ingestion: {len(self.etf_map)} assets, "
            f"target window {start_str} to {end_str} "
            f"(with {buffer_days}-day lead buffer from {buffer_start_dt.strftime('%Y-%m-%d')})."
        )

        all_records: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}

        for asset, ticker in self.etf_map.items():
            try:
                df = self.fetch_ticker_data(
                    asset_class=asset,
                    ticker=ticker,
                    start_date=buffer_start_dt,
                    end_date=end_dt,
                )

                # Filter strictly to target window [start_str, end_str]
                target_mask = (df["date"] >= start_str) & (df["date"] <= end_str)
                target_df = df[target_mask].copy()

                if target_df.empty:
                    raise RuntimeError(f"No observations remain for {asset} within target window.")

                # If first row's daily_return is NaN (due to start boundary), compute from preceding buffer row
                if pd.isna(target_df["daily_return"].iloc[0]):
                    first_idx = target_df.index[0]
                    prev_rows = df.loc[df.index < first_idx]
                    if not prev_rows.empty:
                        p_prev = float(prev_rows["close"].iloc[-1])
                        p_curr = float(target_df["close"].iloc[0])
                        target_df.loc[first_idx, "daily_return"] = (p_curr - p_prev) / p_prev

                for _, row in target_df.iterrows():
                    ret_val = float(row["daily_return"]) if pd.notna(row["daily_return"]) else 0.0
                    vol_val = float(row["volatility_5d"]) if pd.notna(row["volatility_5d"]) else None

                    record = {
                        "date": str(row["date"]),
                        "asset_class": asset,
                        "open_price": round(float(row["open"]), 2),
                        "high_price": round(float(row["high"]), 2),
                        "low_price": round(float(row["low"]), 2),
                        "close_price": round(float(row["close"]), 2),
                        "daily_return": round(ret_val, 6),
                        "volatility_5d": round(vol_val, 6) if vol_val is not None else None,
                        "source": f"yfinance/{ticker}",
                    }
                    all_records.append(record)

                logger.info(
                    f"Successfully processed {len(target_df)} real daily bars for {asset} ({ticker}). "
                    f"Close price range: ${target_df['close'].min():.2f} - ${target_df['close'].max():.2f}"
                )

            except Exception as e:
                logger.error(f"Error fetching real price history for {asset} ({ticker}): {e}")
                errors[asset] = str(e)

        if errors:
            err_msg = (
                f"Market data ingestion failed for {len(errors)}/{len(self.etf_map)} assets: {errors}. "
                "Real historical data is required for genuine quantitative walk-forward validation."
            )
            if not allow_synthetic_fallback:
                raise RuntimeError(err_msg)
            else:
                logger.warning(
                    f"CRITICAL WARNING: {err_msg} Falling back to DEMO SYNTHETIC generator as requested."
                )
                from main import _generate_market_data
                synthetic_data = _generate_market_data(days_back=days_back)
                for item in synthetic_data:
                    item["source"] = "DEMO_SYNTHETIC_FALLBACK"
                return synthetic_data

        # Save snapshot to raw directory for reproducibility
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.cache_dir / f"market_data_{timestamp}.json"
        try:
            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2)
            logger.info(f"Saved market data snapshot ({len(all_records)} rows) to {snapshot_file}")
        except Exception as e:
            logger.warning(f"Failed to cache market data snapshot: {e}")

        # Sort chronologically by date and asset
        all_records.sort(key=lambda x: (x["date"], x["asset_class"]))
        return all_records


def get_historical_market_data(
    days_back: int = 180,
    allow_synthetic: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to retrieve historical market data matching the
    existing _generate_market_data interface.
    """
    fetcher = MarketDataFetcher()
    return fetcher.fetch_market_data(
        days_back=days_back,
        allow_synthetic_fallback=allow_synthetic,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_historical_market_data(days_back=30)
    print(f"Retrieved {len(data)} real market records. Sample record:")
    if data:
        print(json.dumps(data[0], indent=2))
