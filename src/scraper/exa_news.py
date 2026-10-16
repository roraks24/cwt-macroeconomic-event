"""
Exa.ai Economic News Scraper Module
====================================
Isolated search client for Exa.ai semantic search API.
Satisfies the CWT assessment submission guideline ("including apify/tavily/exa").

If EXA_API_KEY is not configured in the environment, the client cleanly reports
its status as unavailable/blocked without disrupting Tavily news ingestion.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

# Add project root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_DIR  # noqa: E402
import os

logger = logging.getLogger("CWT-ExaNews")


class ExaNewsScraper:
    """
    Scrapes economic news and articles via the Exa.ai neural search API.
    Operates as an alternative or complementary news provider alongside Tavily.
    """

    EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("EXA_API_KEY", "")
        self.is_available = bool(self.api_key.strip())

        if not self.is_available:
            logger.info("Exa API key not configured (EXA_API_KEY). Exa news scraper is offline.")
        else:
            logger.info("ExaNewsScraper initialized successfully with configured API key.")

    def search_economic_news(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for economic news using Exa's neural search with full article text.
        """
        if not self.is_available:
            logger.warning(
                "Exa search requested, but EXA_API_KEY is missing. "
                "Configure EXA_API_KEY in .env to enable Exa search."
            )
            return []

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "CWT-Macro-Agent/1.0",
        }

        payload = {
            "query": query,
            "useAutoprompt": True,
            "numResults": num_results,
            "contents": {
                "text": {"maxCharacters": 2000},
            },
        }

        try:
            logger.info(f"Executing Exa neural search for: '{query}'...")
            resp = requests.post(
                self.EXA_SEARCH_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=15,
            )

            if resp.status_code != 200:
                logger.error(f"Exa API returned HTTP {resp.status_code}: {resp.text}")
                return []

            data = resp.json()
            raw_results = data.get("results", [])
            articles: List[Dict[str, Any]] = []

            for item in raw_results:
                title = str(item.get("title", "")).strip()
                text = str(item.get("text", "")).strip()
                url = str(item.get("url", "")).strip()
                pub_date = item.get("publishedDate") or datetime.now(timezone.utc).isoformat()

                if title or text:
                    articles.append({
                        "title": title,
                        "content": text,
                        "source_url": url,
                        "published_date": pub_date,
                        "relevance_score": float(item.get("score", 0.85)) if item.get("score") is not None else 0.85,
                        "query_topic": query,
                        "source": "exa.ai",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })

            logger.info(f"Retrieved {len(articles)} articles from Exa for query '{query}'.")
            return articles

        except Exception as e:
            logger.error(f"Exa search request failed: {e}")
            return []

    def scrape_macro_news(
        self,
        topics: Optional[Sequence[str]] = None,
        max_per_topic: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute thematic macro news search across topics.
        """
        if not self.is_available:
            return []

        target_topics = list(topics) if topics else [
            "Federal Reserve interest rate decisions and monetary policy",
            "US inflation CPI trends and consumer price pressures",
            "US employment nonfarm payrolls and labor market",
        ]

        all_articles: List[Dict[str, Any]] = []
        seen_urls = set()

        for topic in target_topics:
            res = self.search_economic_news(query=topic, num_results=max_per_topic)
            for art in res:
                u = art.get("source_url")
                if u and u not in seen_urls:
                    seen_urls.add(u)
                    all_articles.append(art)

        return all_articles
