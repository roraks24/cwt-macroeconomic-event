"""Scraper package for economic calendars and macro news."""

from src.scraper.apify_calendar import MacroCalendarScraper
from src.scraper.tavily_news import EconomicNewsScraper

__all__ = ["MacroCalendarScraper", "EconomicNewsScraper"]
