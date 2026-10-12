"""
Macroeconomic Calendar Scraper using Apify actors (ForexFactory) with multi-level fallback.
Primary Actor: xtracto/forexfactory-calendar
Backup Actor:  bovi/forex-economic-calendar-scraper
Synthetic fallback when API token is unavailable or network fails.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add base directory to sys.path to import project config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (  # noqa: E402
    APIFY_API_TOKEN,
    APIFY_CALENDAR_PRIMARY,
    APIFY_CALENDAR_BACKUP,
    LOOKBACK_DAYS,
    MACRO_EVENTS,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None  # type: ignore


class MacroCalendarScraper:
    """Scrapes macroeconomic calendar data via ForexFactory Apify actors with synthetic fallback."""

    PRIMARY_ACTOR_ID = APIFY_CALENDAR_PRIMARY or "xtracto/forexfactory-calendar"
    BACKUP_ACTOR_ID = APIFY_CALENDAR_BACKUP or "bovi/forex-economic-calendar-scraper"

    def __init__(self, api_token: str | None = None) -> None:
        """Initialize the Apify client with token from argument or config."""
        self.api_token = api_token or APIFY_API_TOKEN
        self.client: ApifyClient | None = None

        if self.api_token and ApifyClient is not None:
            try:
                self.client = ApifyClient(self.api_token)
                logger.info("ApifyClient initialized successfully.")
            except Exception as e:
                logger.warning("Failed to initialize ApifyClient: %s. Fallback will be used.", e)
        else:
            if not self.api_token:
                logger.info("No Apify token provided; synthetic generation will be used by default.")
            if ApifyClient is None:
                logger.warning("apify-client package not installed. Fallback will be used.")

    def _parse_event_value(self, value_str: Any) -> float | None:
        """
        Parse raw numeric string representations with units into float values.
        Handles percentages ('3.2%'), thousands ('256K'), millions ('1.5M'),
        billions ('4.1B'), trillions ('$1.2T'), and currency symbols.
        """
        if value_str is None:
            return None
        if isinstance(value_str, (int, float)):
            return float(value_str)

        cleaned = str(value_str).strip()
        if not cleaned or cleaned in {"-", "--", "N/A", "null", "None"}:
            return None

        # Remove currency symbols and comma separators
        cleaned = re.sub(r"[\$,€£¥]", "", cleaned).replace(",", "").strip()

        multiplier = 1.0
        if cleaned.endswith("%"):
            multiplier = 1.0  # Keep percentage as raw numerical point (e.g., 3.2 for 3.2%)
            cleaned = cleaned[:-1].strip()
        elif cleaned.upper().endswith("K"):
            multiplier = 1e3
            cleaned = cleaned[:-1].strip()
        elif cleaned.upper().endswith("M"):
            multiplier = 1e6
            cleaned = cleaned[:-1].strip()
        elif cleaned.upper().endswith("B"):
            multiplier = 1e9
            cleaned = cleaned[:-1].strip()
        elif cleaned.upper().endswith("T"):
            multiplier = 1e12
            cleaned = cleaned[:-1].strip()

        try:
            return float(cleaned) * multiplier
        except (ValueError, TypeError):
            match = re.search(r"[-+]?\d*\.?\d+", cleaned)
            if match:
                try:
                    return float(match.group(0)) * multiplier
                except ValueError:
                    return None
            return None

    def _get_dataset_items(self, run: Any) -> list[dict]:
        """Extract dataset items safely from an Apify Run object or dict."""
        dataset_id = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)
        if not dataset_id and isinstance(run, dict):
            dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")

        if not dataset_id:
            logger.warning("No defaultDatasetId found in Apify run.")
            return []

        dataset = self.client.dataset(dataset_id)
        return list(dataset.iterate_items())

    def _scrape_with_primary_actor(self, start_dt: datetime, end_dt: datetime, max_items: int = 500) -> list[dict]:
        """
        Scrape using primary actor 'xtracto/forexfactory-calendar'.
        Conforms strictly to its OpenAPI schema:
        https://api.apify.com/v2/actors/xtracto~forexfactory-calendar/builds/default/openapi.json
        """
        logger.info(
            "Calling primary Apify actor %s for period %s to %s",
            self.PRIMARY_ACTOR_ID,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )
        run_input = {
            "dateRange": "range",
            "rangeFrom": start_dt.strftime("%Y-%m-%d"),
            "rangeTo": end_dt.strftime("%Y-%m-%d"),
            "currencies": ["USD"],
            "minImpact": "low",
            "upcomingOnly": False,
            "maxItems": max_items,
        }

        run = self.client.actor(self.PRIMARY_ACTOR_ID).call(run_input=run_input)
        raw_items = self._get_dataset_items(run)
        logger.info("Primary actor returned %d raw records.", len(raw_items))

        parsed_events: list[dict] = []
        for item in raw_items:
            event_name = item.get("title") or item.get("prefixedName") or item.get("name") or item.get("event")
            if not event_name:
                continue

            # Standardize timestamp to UTC ISO format
            raw_date = item.get("datetimeISO") or item.get("dateline") or item.get("date") or item.get("time")
            try:
                if isinstance(raw_date, (int, float)):
                    dt = datetime.fromtimestamp(raw_date, tz=timezone.utc).isoformat()
                elif raw_date:
                    dt = pd.to_datetime(raw_date).tz_localize(None).isoformat()
                else:
                    dt = datetime.now(timezone.utc).isoformat()
            except Exception:
                dt = datetime.now(timezone.utc).isoformat()

            actual_val = self._parse_event_value(item.get("actual"))
            forecast_val = self._parse_event_value(item.get("forecast") or item.get("consensus"))
            previous_val = self._parse_event_value(item.get("previous"))

            surprise_val = None
            if actual_val is not None and forecast_val is not None:
                surprise_val = round(actual_val - forecast_val, 4)
            elif item.get("actualBetterWorse") in {1, -1}:
                surprise_val = float(item["actualBetterWorse"])

            impact_raw = str(item.get("impact") or item.get("impactTitle") or "medium").lower()
            if "high" in impact_raw or "3" in impact_raw:
                impact = "high"
            elif "low" in impact_raw or "1" in impact_raw:
                impact = "low"
            else:
                impact = "medium"

            parsed_events.append(
                {
                    "event_date": dt,
                    "event_name": event_name.strip(),
                    "country": item.get("country", "US"),
                    "currency": item.get("currency", "USD"),
                    "actual_value": actual_val,
                    "forecast_value": forecast_val,
                    "previous_value": previous_val,
                    "surprise": surprise_val,
                    "impact": impact,
                    "source": f"apify/{self.PRIMARY_ACTOR_ID}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        return parsed_events

    def _scrape_with_backup_actor(self, max_events: int = 500) -> list[dict]:
        """
        Scrape using backup actor 'bovi/forex-economic-calendar-scraper'.
        Conforms strictly to its OpenAPI schema:
        https://api.apify.com/v2/actors/bovi~forex-economic-calendar-scraper/builds/default/openapi.json
        """
        logger.info("Attempting backup Apify actor %s...", self.BACKUP_ACTOR_ID)
        run_input = {
            "range": "this_month",
            "currencies": ["USD"],
            "impactLevels": ["low", "medium", "high"],
            "includeDetail": True,
            "maxEvents": max_events,
        }

        run = self.client.actor(self.BACKUP_ACTOR_ID).call(run_input=run_input)
        raw_items = self._get_dataset_items(run)
        logger.info("Backup actor returned %d raw records.", len(raw_items))

        parsed_events: list[dict] = []
        for item in raw_items:
            event_name = item.get("name") or item.get("title") or item.get("event")
            if not event_name:
                continue

            raw_date = item.get("date") or item.get("datetime") or item.get("time")
            try:
                if raw_date:
                    dt = pd.to_datetime(raw_date).tz_localize(None).isoformat()
                else:
                    dt = datetime.now(timezone.utc).isoformat()
            except Exception:
                dt = datetime.now(timezone.utc).isoformat()

            actual_val = self._parse_event_value(item.get("actual"))
            forecast_val = self._parse_event_value(item.get("forecast"))
            previous_val = self._parse_event_value(item.get("previous"))

            surprise_val = None
            if actual_val is not None and forecast_val is not None:
                surprise_val = round(actual_val - forecast_val, 4)

            impact_raw = str(item.get("impact") or "medium").lower()
            if "high" in impact_raw:
                impact = "high"
            elif "low" in impact_raw:
                impact = "low"
            else:
                impact = "medium"

            parsed_events.append(
                {
                    "event_date": dt,
                    "event_name": event_name.strip(),
                    "country": item.get("country", "US"),
                    "currency": item.get("currency", "USD"),
                    "actual_value": actual_val,
                    "forecast_value": forecast_val,
                    "previous_value": previous_val,
                    "surprise": surprise_val,
                    "impact": impact,
                    "source": f"apify/{self.BACKUP_ACTOR_ID}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        return parsed_events

    def _generate_synthetic_data(self, days_back: int = 180) -> list[dict]:
        """
        Generate realistic synthetic macroeconomic events for backtesting and demo.
        Covers CPI, Core CPI, Non-Farm Payrolls, FOMC, GDP, etc.
        """
        logger.info("Generating synthetic macroeconomic events for last %d days.", days_back)
        np.random.seed(42)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        event_definitions = [
            {"event_name": "CPI m/m", "frequency_days": 30, "base_forecast": 3.1, "volatility": 0.25, "impact": "high", "country": "US"},
            {"event_name": "Core CPI m/m", "frequency_days": 30, "base_forecast": 3.3, "volatility": 0.15, "impact": "high", "country": "US"},
            {"event_name": "Non-Farm Employment Change", "frequency_days": 30, "base_forecast": 180000.0, "volatility": 45000.0, "impact": "high", "country": "US"},
            {"event_name": "Unemployment Rate", "frequency_days": 30, "base_forecast": 3.9, "volatility": 0.2, "impact": "high", "country": "US"},
            {"event_name": "FOMC Statement & Fed Interest Rate Decision", "frequency_days": 45, "base_forecast": 5.25, "volatility": 0.0, "impact": "high", "country": "US"},
            {"event_name": "GDP Growth Rate q/q", "frequency_days": 90, "base_forecast": 2.4, "volatility": 0.4, "impact": "high", "country": "US"},
            {"event_name": "Retail Sales m/m", "frequency_days": 30, "base_forecast": 0.4, "volatility": 0.3, "impact": "medium", "country": "US"},
            {"event_name": "ISM Manufacturing PMI", "frequency_days": 30, "base_forecast": 49.5, "volatility": 1.2, "impact": "medium", "country": "US"},
            {"event_name": "Initial Jobless Claims", "frequency_days": 7, "base_forecast": 220000.0, "volatility": 15000.0, "impact": "medium", "country": "US"},
            {"event_name": "Consumer Sentiment", "frequency_days": 30, "base_forecast": 78.0, "volatility": 3.5, "impact": "low", "country": "US"},
        ]

        events: list[dict] = []
        current_cursor = start_date

        while current_cursor <= end_date:
            for edef in event_definitions:
                if np.random.uniform(0, 1) < (1.0 / (edef["frequency_days"] / 2.0)):
                    forecast = float(edef["base_forecast"] + np.random.normal(0, edef["volatility"] * 0.5))
                    surprise_factor = float(np.random.normal(0, edef["volatility"]))
                    actual = forecast + surprise_factor
                    previous = float(edef["base_forecast"] + np.random.normal(0, edef["volatility"] * 0.5))
                    event_time = current_cursor.replace(hour=12, minute=30, second=0, microsecond=0)

                    events.append(
                        {
                            "event_date": event_time.isoformat(),
                            "event_name": edef["event_name"],
                            "country": edef["country"],
                            "currency": "USD",
                            "actual_value": round(actual, 4),
                            "forecast_value": round(forecast, 4),
                            "previous_value": round(previous, 4),
                            "surprise": round(actual - forecast, 4),
                            "impact": edef["impact"],
                            "source": "synthetic_generator",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            current_cursor += timedelta(days=2)

        events.sort(key=lambda x: x["event_date"])
        logger.info("Generated %d synthetic events.", len(events))
        return events

    def scrape_economic_calendar(self, days_back: int = 180) -> list[dict]:
        """
        Scrape economic calendar using primary actor (xtracto/forexfactory-calendar),
        falling back to backup actor (bovi/forex-economic-calendar-scraper),
        and finally falling back to synthetic generator if both fail.
        """
        if not self.client:
            logger.warning("ApifyClient not configured. Using synthetic economic calendar data.")
            return self._generate_synthetic_data(days_back=days_back)

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days_back)

        # 1. Try Primary Actor
        try:
            events = self._scrape_with_primary_actor(start_dt=start_dt, end_dt=end_dt)
            if events:
                logger.info("Successfully scraped %d events from primary actor %s", len(events), self.PRIMARY_ACTOR_ID)
                return events
            logger.warning("Primary actor returned empty results. Falling back to backup actor.")
        except Exception as e:
            logger.error("Primary Apify actor %s failed: %s. Trying backup actor...", self.PRIMARY_ACTOR_ID, e)

        # 2. Try Backup Actor
        try:
            events = self._scrape_with_backup_actor()
            if events:
                logger.info("Successfully scraped %d events from backup actor %s", len(events), self.BACKUP_ACTOR_ID)
                return events
            logger.warning("Backup actor returned empty results.")
        except Exception as e:
            logger.error("Backup Apify actor %s failed: %s", self.BACKUP_ACTOR_ID, e)

        # 3. Fallback to Synthetic
        logger.warning("All Apify calendar actors failed. Falling back to synthetic calendar data.")
        return self._generate_synthetic_data(days_back=days_back)

    def save_raw_data(self, events: list[dict], filepath: str | Path | None = None) -> Path:
        """Save raw scraped events to a JSON file."""
        if filepath is None:
            filename = f"macro_calendar_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            filepath = RAW_DIR / filename
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d events to %s", len(events), filepath)
        return filepath

    def run(self, days_back: int = LOOKBACK_DAYS) -> list[dict]:
        """
        Execute full calendar pipeline:
        1. Scrape or synthetically generate economic calendar events
        2. Save raw output to raw data directory
        3. Return list of event dictionaries
        """
        logger.info("Running MacroCalendarScraper pipeline (days_back=%d)...", days_back)
        events = self.scrape_economic_calendar(days_back=days_back)
        self.save_raw_data(events)
        return events


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = MacroCalendarScraper()
    data = scraper.run(days_back=30)
    print(f"Scraped/Generated {len(data)} events. Sample:")
    if data:
        print(data[0])
