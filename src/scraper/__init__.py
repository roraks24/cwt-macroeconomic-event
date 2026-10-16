"""Scraper package for economic calendars and macro news."""

from src.scraper.apify_calendar import MacroCalendarScraper
from src.scraper.tavily_news import EconomicNewsScraper
from src.scraper.market_data import MarketDataFetcher, get_historical_market_data

__all__ = ["MacroCalendarScraper", "EconomicNewsScraper", "MarketDataFetcher", "get_historical_market_data"]
