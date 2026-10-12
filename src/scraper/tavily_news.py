"""
Economic News Scraper using Tavily Search API with TextBlob sentiment analysis.
Supports deep historical macroeconomic news queries and optional Apify Google News scraping.
Includes automatic synthetic fallback when API keys are absent or network fails.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

# Add base directory to sys.path to import project config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (  # noqa: E402
    APIFY_API_TOKEN,
    APIFY_NEWS_HEADLINES,
    APIFY_NEWS_FULL_TEXT,
    RAW_DIR,
    TAVILY_API_KEY,
)

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None  # type: ignore

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None  # type: ignore


class EconomicNewsScraper:
    """Scrapes macroeconomic news via Tavily search and enriches with NLP sentiment scores."""

    DEFAULT_TOPICS = [
        "US Federal Reserve interest rates monetary policy FOMC",
        "US inflation CPI PPI consumer price index trends",
        "US employment jobs non-farm payrolls unemployment rate",
        "US GDP economic growth quarterly recession soft landing",
        "US retail sales consumer spending economic momentum",
        "US treasury yields bond market central bank policy",
        "US manufacturing ISM PMI economic activity expansion",
        "US trade balance tariffs global currency exchange rates",
    ]

    def __init__(self, api_key: str | None = None, apify_token: str | None = None) -> None:
        """Initialize TavilyClient and optional ApifyClient."""
        self.api_key = api_key or TAVILY_API_KEY
        self.apify_token = apify_token or APIFY_API_TOKEN
        self.client: TavilyClient | None = None
        self.apify_client: ApifyClient | None = None

        if self.api_key and TavilyClient is not None:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("TavilyClient initialized successfully.")
            except Exception as e:
                logger.warning("Failed to initialize TavilyClient: %s. Fallback will be used.", e)
        else:
            if not self.api_key:
                logger.info("No Tavily API key provided; synthetic news generator will be used if needed.")
            if TavilyClient is None:
                logger.warning("tavily-python library not installed.")

        if self.apify_token and ApifyClient is not None:
            try:
                self.apify_client = ApifyClient(self.apify_token)
                logger.info("ApifyClient available for optional news scraping.")
            except Exception:
                pass

    def search_economic_news(
        self,
        query: str,
        max_results: int = 15,
        topic: str = "news",
        days: int | None = None,
    ) -> list[dict]:
        """
        Search for economic news articles using the Tavily search API.
        Extracts title, content, URL, publication date, and relevance score.
        """
        if not self.client:
            logger.warning("Tavily client is not available. Skipping live search.")
            return []

        try:
            logger.info("Querying Tavily: '%s' (topic=%s, max_results=%d)", query, topic, max_results)
            kwargs = {
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "topic": topic,
                "include_answer": False,
            }
            if days:
                kwargs["days"] = days

            response = self.client.search(**kwargs)
            raw_results = response.get("results", [])
            articles: list[dict] = []

            for item in raw_results:
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                url = item.get("url", "").strip()
                pub_date = item.get("published_date")

                # Parse and standardize publication date to UTC ISO
                if pub_date:
                    try:
                        parsed_dt = pd.to_datetime(pub_date).tz_localize(None).isoformat()
                    except Exception:
                        parsed_dt = datetime.now(timezone.utc).isoformat()
                else:
                    parsed_dt = datetime.now(timezone.utc).isoformat()

                score = item.get("score", 0.85)

                if title or content:
                    articles.append(
                        {
                            "title": title,
                            "content": content,
                            "source_url": url,
                            "published_date": parsed_dt,
                            "relevance_score": float(score) if score is not None else 0.8,
                            "query_topic": query,
                            "source": "tavily",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            logger.info("Retrieved %d articles from Tavily for query '%s'.", len(articles), query)
            return articles

        except Exception as e:
            logger.error("Error executing Tavily search for '%s': %s", query, e)
            return []

    def scrape_historical_news(
        self,
        topics: Sequence[str] | None = None,
        days_back: int = 180,
        max_per_topic: int = 10,
    ) -> list[dict]:
        """
        Scrape historical macroeconomic news across topics covering the lookback window.
        Executes targeted thematic queries to capture macro narratives across time.
        """
        if not self.client:
            logger.info("Tavily client unavailable; generating synthetic news.")
            return self._generate_synthetic_news(days_back=days_back)

        target_topics = list(topics) if topics else self.DEFAULT_TOPICS
        all_articles: list[dict] = []
        seen_urls: set[str] = set()

        for topic in target_topics:
            articles = self.search_economic_news(
                query=f"{topic} report economic indicators",
                max_results=max_per_topic,
                topic="news",
            )
            for art in articles:
                url = art.get("source_url")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                all_articles.append(art)

        # If Tavily returned fewer than 5 articles, supplement with synthetic history to ensure robust model training
        if len(all_articles) < 5:
            logger.warning("Live Tavily returned %d articles. Supplementing with synthetic news history.", len(all_articles))
            synthetic = self._generate_synthetic_news(days_back=days_back)
            all_articles.extend(synthetic)

        logger.info("Total deduplicated articles gathered: %d", len(all_articles))
        return all_articles

    def scrape_apify_news(self, query: str = "US Federal Reserve economic news", max_items: int = 20) -> list[dict]:
        """
        Optional: Scrape news headlines using Apify Google News Scraper actor.
        Actor: gazidev/google-news-scraper or nourishing_courier/google-news-scraper.
        """
        if not self.apify_client:
            logger.warning("Apify client unavailable for news scraping.")
            return []

        actor_id = APIFY_NEWS_HEADLINES or "gazidev/google-news-scraper"
        try:
            logger.info("Scraping news headlines via Apify actor %s for '%s'...", actor_id, query)
            run = self.apify_client.actor(actor_id).call(
                run_input={"query": query, "maxItems": max_items, "language": "en"}
            )
            dataset_id = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)
            if not dataset_id and isinstance(run, dict):
                dataset_id = run.get("defaultDatasetId")

            if not dataset_id:
                return []

            dataset = self.apify_client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            articles = []

            for it in items:
                title = it.get("title") or it.get("headline")
                if not title:
                    continue
                url = it.get("link") or it.get("url") or ""
                pub_date = it.get("datetime") or it.get("date") or datetime.now(timezone.utc).isoformat()

                try:
                    dt = pd.to_datetime(pub_date).tz_localize(None).isoformat()
                except Exception:
                    dt = datetime.now(timezone.utc).isoformat()

                articles.append(
                    {
                        "title": title.strip(),
                        "content": it.get("snippet", "") or it.get("description", "") or title,
                        "source_url": url,
                        "published_date": dt,
                        "relevance_score": 0.85,
                        "query_topic": query,
                        "source": f"apify/{actor_id}",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            logger.info("Retrieved %d articles via Apify news actor.", len(articles))
            return articles
        except Exception as e:
            logger.warning("Apify news scraper error: %s", e)
            return []

    def _generate_synthetic_news(self, days_back: int = 180) -> list[dict]:
        """
        Generate realistic macroeconomic news articles across time for demo or backtesting.
        """
        logger.info("Generating realistic synthetic economic news dataset for past %d days.", days_back)
        now = datetime.now(timezone.utc)

        templates = [
            ("Federal Reserve Holds Rates Steady, Signals Cautious Stance on Future Cuts",
             "The Federal Reserve maintained benchmark interest rates within the 5.25%-5.50% range. Chairman Powell emphasized that inflation remains moderately above target, and the FOMC requires greater confidence in disinflation before initiating easing cycles. Bond yields edged higher while equity indices showed resilience.",
             0.95, "Federal Reserve interest rates monetary policy FOMC"),
            ("US Headline CPI Cools to 3.1% Year-Over-Year, Bolstering Soft Landing Hopes",
             "Consumer price index data revealed a noticeable decelerating trend in headline inflation, driven by sharp reductions in gasoline and automotive prices. Core CPI registered at 3.3%, confirming persistent progress toward the central bank's target and lifting investor optimism.",
             0.92, "US inflation CPI PPI consumer price index"),
            ("Non-Farm Payrolls Surge Exceeds Consensus, Labor Market Retains Robust Momentum",
             "The US economy added 215,000 jobs last month, substantially outperforming analyst estimates. Average hourly earnings ticked up 0.3%, indicating steady wage growth without reigniting severe wage-price spiral pressures. Unemployment remained steady at 3.8%.",
             0.89, "US employment jobs non-farm payrolls unemployment rate"),
            ("Q3 GDP Growth Revised Upward to 2.8% on Strong Consumer Spending",
             "Revised national economic accounts indicate gross domestic product expanded at a robust 2.8% annualized rate. Resilient consumer discretionary expenditures and corporate technology investments offset sluggish residential real estate activity.",
             0.88, "US GDP growth economic recession soft landing"),
            ("Global Trade Balances Stabilize as Supply Chains Modernize Amid Tariff Talks",
             "The US trade deficit narrowed slightly as goods exports rose while import volumes leveled. Corporate supply chain managers reported improved delivery timelines, though ongoing geopolitical tensions and maritime shipping rates continue to be closely monitored.",
             0.84, "US trade balance tariffs global supply chain"),
            ("Producer Price Index (PPI) Drops Unexpectedly, Alleviating Upstream Cost Pressures",
             "Wholesale prices fell 0.1% month-over-month, surprising economists who expected a modest rise. Energy and intermediate raw material declines contributed directly to producer margin expansion.",
             0.87, "US inflation CPI PPI consumer price index"),
            ("ISM Manufacturing PMI Expands Above 50 Mark for First Time in Months",
             "US factory activity snapped a prolonged contraction streak, fueled by expanding new export orders and inventory replenishment. Input cost gauges moderated, signaling balanced factory gate price trends.",
             0.86, "US manufacturing ISM PMI economic activity expansion"),
            ("Treasury Yields Retreat as Core PCE Inflation Hits Three-Year Low",
             "Benchmark 10-year Treasury yields slipped below key resistance levels following favorable personal consumption expenditures readings. Equity futures gained as traders raised odds of policy accommodation.",
             0.91, "US treasury yields bond market central bank policy"),
        ]

        synthetic_data: list[dict] = []
        step_days = max(1, days_back // len(templates))

        for idx, (title, content, score, topic) in enumerate(templates):
            art_dt = (now - timedelta(days=idx * step_days + 1)).isoformat()
            synthetic_data.append(
                {
                    "title": title,
                    "content": content,
                    "source_url": f"https://finance.example.com/macro-news-{idx}",
                    "published_date": art_dt,
                    "relevance_score": score,
                    "query_topic": topic,
                    "source": "synthetic_generator",
                    "created_at": now.isoformat(),
                }
            )

        return synthetic_data

    def analyze_sentiment(self, text: str) -> dict[str, float]:
        """
        Compute sentiment polarity score (-1.0 to +1.0) and subjectivity (0.0 to 1.0) using TextBlob.
        Includes domain financial lexicon heuristic fallback if TextBlob is unavailable.
        """
        if not text or not text.strip():
            return {"polarity": 0.0, "subjectivity": 0.0}

        if TextBlob is not None:
            try:
                blob = TextBlob(text)
                return {
                    "polarity": round(float(blob.sentiment.polarity), 4),
                    "subjectivity": round(float(blob.sentiment.subjectivity), 4),
                }
            except Exception as e:
                logger.debug("TextBlob analysis error: %s. Using heuristic fallback.", e)

        # Fallback financial sentiment heuristic
        pos_terms = {
            "growth", "expand", "expansion", "gain", "surge", "beat", "rally", "resilient",
            "optimism", "robust", "increase", "bullish", "strong", "cools", "eased", "moderated",
            "disinflation", "recovery", "outperformed", "steady", "upward"
        }
        neg_terms = {
            "recession", "decline", "drop", "slump", "miss", "bearish", "fall", "cooling",
            "weak", "crisis", "default", "inflationary", "loss", "down", "struggle", "downturn",
            "contraction", "headwind", "restrictive", "deficit", "sluggish"
        }

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return {"polarity": 0.0, "subjectivity": 0.0}

        pos_count = sum(1 for w in words if w in pos_terms)
        neg_count = sum(1 for w in words if w in neg_terms)
        total_matched = pos_count + neg_count

        if total_matched == 0:
            return {"polarity": 0.0, "subjectivity": 0.0}

        score = (pos_count - neg_count) / float(total_matched)
        polarity = round(float(max(min(score, 1.0), -1.0)), 4)
        subjectivity = round(float(min(total_matched / float(len(words)) * 5.0, 1.0)), 4)
        return {"polarity": polarity, "subjectivity": subjectivity}

    def enrich_with_sentiment(self, articles: list[dict]) -> list[dict]:
        """Compute sentiment scores for each article based on combined title and content."""
        enriched: list[dict] = []
        for article in articles:
            combined_text = f"{article.get('title', '')}. {article.get('content', '')}"
            sentiment = self.analyze_sentiment(combined_text)
            art_copy = dict(article)
            art_copy["sentiment_score"] = sentiment["polarity"]
            art_copy["sentiment_subjectivity"] = sentiment["subjectivity"]
            enriched.append(art_copy)
        return enriched

    def save_raw_data(self, articles: list[dict], filepath: str | Path | None = None) -> Path:
        """Save raw scraped and sentiment-enriched articles to a JSON file."""
        if filepath is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filepath = RAW_DIR / f"macro_news_{timestamp}.json"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d news articles to %s", len(articles), filepath)
        return filepath

    def run(self, topics: Sequence[str] | None = None, days_back: int = 180) -> list[dict]:
        """
        Execute full pipeline:
        1. Scrape historical economic news via Tavily
        2. Enrich articles with sentiment analysis
        3. Save raw output to raw data directory
        4. Return list of enriched article dictionaries
        """
        logger.info("Running EconomicNewsScraper pipeline (days_back=%d)...", days_back)
        articles = self.scrape_historical_news(topics=topics, days_back=days_back)
        enriched_articles = self.enrich_with_sentiment(articles)
        self.save_raw_data(enriched_articles)
        return enriched_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = EconomicNewsScraper()
    results = scraper.run()
    print(f"Scraped and enriched {len(results)} articles. Sample:")
    if results:
        print(f"Title: {results[0]['title']}")
        print(f"Sentiment: {results[0]['sentiment_score']}")
