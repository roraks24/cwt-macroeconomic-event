"""Unit tests for Macro Calendar, News Scrapers, and Exa integration."""

import pytest
from src.scraper.apify_calendar import MacroCalendarScraper
from src.scraper.tavily_news import EconomicNewsScraper
from src.scraper.exa_news import ExaNewsScraper


def test_parse_event_value_percentages_and_units():
    """Verify that MacroCalendarScraper parses various numerical formats and units accurately."""
    scraper = MacroCalendarScraper(api_token=None)

    assert scraper._parse_event_value("3.2%") == 3.2
    assert scraper._parse_event_value("250K") == 250000.0
    assert scraper._parse_event_value("1.5M") == 1500000.0
    assert scraper._parse_event_value("$4.2B") == 4200000000.0
    assert scraper._parse_event_value("10.5") == 10.5
    assert scraper._parse_event_value(-15.2) == -15.2
    assert scraper._parse_event_value("N/A") is None
    assert scraper._parse_event_value("-") is None
    assert scraper._parse_event_value(None) is None


def test_synthetic_calendar_generation_structure():
    """Verify that fallback calendar generation creates compliant schema dictionaries."""
    scraper = MacroCalendarScraper(api_token=None)
    events = scraper._generate_synthetic_data(days_back=30)

    assert isinstance(events, list)
    assert len(events) > 0
    first = events[0]
    required_keys = {"event_date", "event_name", "country", "actual_value", "forecast_value", "surprise", "impact", "source"}
    assert required_keys.issubset(first.keys())
    assert first["country"] == "US"
    assert first["impact"] in {"high", "medium", "low"}


def test_news_sentiment_enrichment_bounds():
    """Verify that sentiment scoring yields polarity scores bounded in [-1.0, 1.0]."""
    scraper = EconomicNewsScraper(api_key=None)
    articles = [
        {"title": "US GDP surges with robust economic growth and strong employment", "content": "Rally continues as inflation cools."},
        {"title": "Recession fears deepen as manufacturing slumps and job cuts rise", "content": "Default risk escalates amid severe crisis."},
        {"title": "Federal Reserve maintains interest rates at balanced target", "content": "Neutral policy meeting concludes."},
    ]
    enriched = scraper.enrich_with_sentiment(articles)

    assert len(enriched) == 3
    for art in enriched:
        assert "sentiment_score" in art
        score = art["sentiment_score"]
        assert -1.0 <= score <= 1.0

    # First should be positive, second negative
    assert enriched[0]["sentiment_score"] > 0
    assert enriched[1]["sentiment_score"] < 0


def test_exa_news_offline_handling_when_no_token():
    """Verify that ExaNewsScraper reports unavailable cleanly when no API key is set."""
    exa = ExaNewsScraper(api_key="")
    assert exa.is_available is False
    res = exa.search_economic_news("US Inflation")
    assert res == []


def test_exa_news_search_parsing_with_mocked_response(monkeypatch):
    """Verify that ExaNewsScraper parses valid API responses into standard article format."""
    import requests

    fake_response_data = {
        "results": [
            {
                "title": "Fed Signals Rate Pause as Inflation Cools",
                "text": "The Federal Reserve indicated interest rates may remain steady following CPI prints.",
                "url": "https://example.com/fed-cpi-pause",
                "publishedDate": "2026-10-14T14:30:00Z",
                "score": 0.92,
            },
            {
                "title": "US Nonfarm Payrolls Beat Expectations",
                "text": "Labor market resilience continued in September with 250k jobs added.",
                "url": "https://example.com/nfp-september-beat",
                "publishedDate": "2026-10-10T12:30:00Z",
                "score": 0.88,
            },
        ]
    }

    class MockResponse:
        status_code = 200

        def json(self):
            return fake_response_data

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResponse())

    exa = ExaNewsScraper(api_key="mock_test_key_12345")
    assert exa.is_available is True

    articles = exa.search_economic_news("Federal Reserve Policy", num_results=2)
    assert len(articles) == 2
    assert articles[0]["title"] == "Fed Signals Rate Pause as Inflation Cools"
    assert "Federal Reserve" in articles[0]["content"]
    assert articles[0]["source_url"] == "https://example.com/fed-cpi-pause"
    assert articles[0]["source"] == "exa.ai"
    assert articles[0]["relevance_score"] == 0.92
    assert "created_at" in articles[0]

