"""Unit tests for DatabaseManager operations, deduplication, and table joins."""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from src.database.db_manager import DatabaseManager, MacroEvent, NewsArticle, MarketData


@pytest.fixture
def db():
    """Fixture providing an isolated in-memory SQLite database instance."""
    return DatabaseManager(db_url="sqlite:///:memory:")


def test_insert_and_deduplicate_macro_events(db):
    """Verify that macro events are inserted and duplicates on (event_date, event_name) are rejected."""
    events = [
        {
            "event_date": "2026-03-01 13:30:00",
            "event_name": "US Non-Farm Payrolls",
            "country": "US",
            "actual_value": 225000.0,
            "forecast_value": 200000.0,
            "surprise": 25000.0,
            "impact": "high",
            "source": "apify",
        },
        {
            "event_date": "2026-03-02 15:00:00",
            "event_name": "ISM Manufacturing PMI",
            "country": "US",
            "actual_value": 51.5,
            "forecast_value": 50.0,
            "surprise": 1.5,
            "impact": "medium",
            "source": "apify",
        },
    ]

    # First insert: both should be inserted
    inserted_first = db.insert_macro_events(events)
    assert inserted_first == 2

    # Verify query returns 2 records
    df = db.get_macro_events()
    assert len(df) == 2
    assert "US Non-Farm Payrolls" in df["event_name"].values

    # Second insert with exact same events: 0 new inserted
    inserted_second = db.insert_macro_events(events)
    assert inserted_second == 0

    # Insert 1 new event and 1 duplicate: only 1 new inserted
    mixed_events = [
        events[0],  # Duplicate
        {
            "event_date": "2026-03-03 14:00:00",
            "event_name": "Fed Interest Rate Decision",
            "country": "US",
            "actual_value": 5.25,
            "forecast_value": 5.25,
            "surprise": 0.0,
            "impact": "high",
            "source": "apify",
        },
    ]
    inserted_mixed = db.insert_macro_events(mixed_events)
    assert inserted_mixed == 1

    df_total = db.get_macro_events()
    assert len(df_total) == 3


def test_insert_and_deduplicate_news_articles(db):
    """Verify that news articles are inserted and deduplicated on source_url."""
    articles = [
        {
            "published_date": "2026-03-01 10:00:00",
            "title": "Fed Signals Patience on Rate Cuts",
            "content": "Federal Reserve officials indicated they will wait for further inflation data.",
            "source_url": "https://reuters.com/markets/rates-2026",
            "sentiment_score": 0.15,
            "relevance_score": 0.90,
            "query_topic": "interest_rates",
        },
        {
            "published_date": "2026-03-01 12:00:00",
            "title": "US Job Growth Exceeds Forecasts",
            "content": "Strong labor market figures point to continued economic resilience.",
            "source_url": "https://bloomberg.com/news/jobs-report-2026",
            "sentiment_score": 0.65,
            "relevance_score": 0.95,
            "query_topic": "labor_market",
        },
    ]

    inserted_first = db.insert_news_articles(articles)
    assert inserted_first == 2

    df_news = db.get_news_articles()
    assert len(df_news) == 2

    # Duplicate URL insertion should be prevented
    inserted_second = db.insert_news_articles(articles)
    assert inserted_second == 0

    df_news_after = db.get_news_articles()
    assert len(df_news_after) == 2


def test_insert_and_deduplicate_market_data(db):
    """Verify market data ingestion and deduplication on (date, asset_class)."""
    bars = [
        {
            "date": "2026-03-01 00:00:00",
            "asset_class": "US_EQUITY",
            "open_price": 500.0,
            "high_price": 505.0,
            "low_price": 498.0,
            "close_price": 503.0,
            "daily_return": 0.006,
            "volatility_5d": 0.008,
        },
        {
            "date": "2026-03-01 00:00:00",
            "asset_class": "US_BOND",
            "open_price": 95.0,
            "high_price": 96.0,
            "low_price": 94.5,
            "close_price": 95.8,
            "daily_return": 0.002,
            "volatility_5d": 0.005,
        },
    ]

    inserted = db.insert_market_data(bars)
    assert inserted == 2

    # Re-inserting the same date and asset class should be skipped
    inserted_dup = db.insert_market_data(bars)
    assert inserted_dup == 0

    df_market = db.get_market_data()
    assert len(df_market) == 2


def test_get_joined_dataset_calendar_alignment(db):
    """Verify that get_joined_dataset accurately aligns market bars and macro events on calendar date."""
    events = [
        {
            "event_date": "2026-03-01 13:30:00",
            "event_name": "US Non-Farm Payrolls",
            "country": "US",
            "actual_value": 225000.0,
            "forecast_value": 200000.0,
            "surprise": 25000.0,
            "impact": "high",
        }
    ]
    bars = [
        {
            "date": "2026-03-01 00:00:00",
            "asset_class": "US_EQUITY",
            "open_price": 500.0,
            "high_price": 505.0,
            "low_price": 498.0,
            "close_price": 503.0,
            "daily_return": 0.006,
            "volatility_5d": 0.008,
        },
        {
            "date": "2026-03-02 00:00:00",
            "asset_class": "US_EQUITY",
            "open_price": 503.0,
            "high_price": 507.0,
            "low_price": 501.0,
            "close_price": 506.0,
            "daily_return": 0.0059,
            "volatility_5d": 0.007,
        },
    ]

    db.insert_macro_events(events)
    db.insert_market_data(bars)

    joined = db.get_joined_dataset(asset_class="US_EQUITY")
    assert not joined.empty
    assert len(joined) == 2
    assert "close_price" in joined.columns
    assert "event_name" in joined.columns

    # First row (2026-03-01) should have the joined macro event
    row_01 = joined[joined["calendar_date"] == pd.to_datetime("2026-03-01").date()].iloc[0]
    assert row_01["event_name"] == "US Non-Farm Payrolls"
    assert row_01["surprise"] == 25000.0

    # Second row (2026-03-02) should have NaN for macro event (left join)
    row_02 = joined[joined["calendar_date"] == pd.to_datetime("2026-03-02").date()].iloc[0]
    assert pd.isna(row_02["event_name"])
