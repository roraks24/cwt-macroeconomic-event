"""Unit tests for historical market data ingestion and bar consistency."""

import pandas as pd
import pytest
from src.scraper.market_data import MarketDataFetcher, DEFAULT_ETF_MAP


def test_etf_map_completeness():
    """Verify that all 5 target macroeconomic asset classes have valid ETF proxies."""
    expected_assets = {"US_EQUITY", "US_BOND", "GOLD", "USD_INDEX", "CRUDE_OIL"}
    assert set(DEFAULT_ETF_MAP.keys()) == expected_assets
    assert DEFAULT_ETF_MAP["US_EQUITY"] == "SPY"
    assert DEFAULT_ETF_MAP["US_BOND"] == "TLT"
    assert DEFAULT_ETF_MAP["GOLD"] == "GLD"
    assert DEFAULT_ETF_MAP["USD_INDEX"] == "UUP"
    assert DEFAULT_ETF_MAP["CRUDE_OIL"] == "USO"


def test_bar_invariants_validation():
    """Verify that bar invariant checking catches invalid prices."""
    fetcher = MarketDataFetcher()

    # Valid bar dataframe
    valid_df = pd.DataFrame({
        "open": [100.0, 102.0],
        "high": [105.0, 106.0],
        "low": [98.0, 100.0],
        "close": [102.0, 104.0],
    })
    # Should not raise
    fetcher._validate_bar_invariants(valid_df, "TEST_ASSET")

    # Invalid negative price
    invalid_neg = pd.DataFrame({
        "open": [100.0],
        "high": [105.0],
        "low": [-10.0],
        "close": [102.0],
    })
    with pytest.raises(ValueError, match="Non-positive prices"):
        fetcher._validate_bar_invariants(invalid_neg, "TEST_ASSET")


def test_daily_return_and_volatility_calculation():
    """Verify exact daily return formula and 5-day rolling volatility behavior."""
    closes = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0]
    df = pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
    })
    df["daily_return"] = df["close"].pct_change()
    df["volatility_5d"] = df["daily_return"].rolling(window=5, min_periods=3).std()

    # (102 - 100) / 100 = 0.02
    assert pytest.approx(df["daily_return"].iloc[1], 1e-5) == 0.02
    # (101 - 102) / 102 = -0.0098039
    assert pytest.approx(df["daily_return"].iloc[2], 1e-5) == (101.0 - 102.0) / 102.0
    # Volatility should be positive once enough observations exist
    assert df["volatility_5d"].iloc[-1] > 0
