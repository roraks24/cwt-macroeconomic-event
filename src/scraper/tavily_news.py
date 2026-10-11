"""
Economic News Scraper using Tavily Search API with TextBlob sentiment analysis.
Includes automatic synthetic fallback when API keys are absent.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

# Add base directory to sys.path to import project config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_DIR, TAVILY_API_KEY  # noqa: E402

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None  # type: ignore


class EconomicNewsScraper:
    """Scrapes macroeconomic news via Tavily search and enriches with sentiment scores."""

    DEFAULT_TOPICS = [
        "Federal Reserve interest rates monetary policy FOMC",
        "US inflation CPI PPI consumer price index",
        "US employment jobs non-farm payrolls unemployment rate",
        "US GDP growth economic recession soft landing",
        "US trade balance tariffs global supply chain",
    ]

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize TavilyClient with provided key or environment variable."""
        self.api_key = api_key or TAVILY_API_KEY
        self.client: TavilyClient | None = None

        if self.api_key and TavilyClient is not None:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("TavilyClient initialized successfully.")
            except Exception as e:
                logger.warning("Failed to initialize TavilyClient: %s. Fallback will be used.", e)
        else:
            if not self.api_key:
                logger.info("No Tavily API key provided; synthetic news generator will be used.")
            if TavilyClient is None:
                logger.warning("tavily-python library not installed. Fallback will be used.")

    def search_economic_news(self, query: str, max_results: int = 20) -> list[dict]:
        """
        Search for economic news articles using the Tavily search API.
        Extracts title, content, URL, publication date, and relevance score.
        """
        if not self.client:
            logger.warning("Tavily client is not available. Skipping live search.")
            return []

        try:
            logger.info("Querying Tavily for: '%s' (max_results=%d)", query, max_results)
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
            )

            raw_results = response.get("results", [])
            articles: list[dict] = []
            for item in raw_results:
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                url = item.get("url", "").strip()
                pub_date = item.get("published_date") or datetime.now(timezone.utc).isoformat()
                score = item.get("score", 0.85)

                if title or content:
                    articles.append(
                        {
                            "title": title,
                            "content": content,
                            "source_url": url,
                            "published_date": pub_date,
                            "relevance_score": float(score) if score is not None else 0.8,
                            "query_topic": query,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            logger.info("Retrieved %d articles from Tavily for query '%s'.", len(articles), query)
            return articles

        except Exception as e:
            logger.error("Error executing Tavily search for '%s': %s", query, e)
            return []

    def _generate_synthetic_news(self) -> list[dict]:
        """
        Generate realistic macroeconomic news articles for demo, backtesting, or fallback.
        """
        logger.info("Generating realistic synthetic economic news dataset.")
        now = datetime.now(timezone.utc)

        synthetic_data = [
            {
                "title": "Federal Reserve Holds Rates Steady, Signals Cautious Stance on Future Cuts",
                "content": (
                    "The Federal Reserve maintained benchmark interest rates within the 5.25%-5.50% range. "
                    "Chairman Powell emphasized that inflation remains moderately above target, and the FOMC "
                    "requires greater confidence in disinflation before initiating easing cycles. Bond yields "
                    "edged higher while equity indices showed resilience."
                ),
                "source_url": "https://www.reuters.com/markets/us/fed-holds-rates-steady-fomc-outlook",
                "published_date": (now - timedelta(days=1)).isoformat(),
                "relevance_score": 0.95,
                "query_topic": "Federal Reserve interest rates monetary policy FOMC",
            },
            {
                "title": "US Headline CPI Cools to 3.1% Year-Over-Year, Bolstering Soft Landing Hopes",
                "content": (
                    "Consumer price index data revealed a noticeable decelerating trend in headline inflation, "
                    "driven by sharp reductions in gasoline and automotive prices. Core CPI registered at 3.3%, "
                    "confirming persistent progress toward the central bank's target and lifting investor optimism."
                ),
                "source_url": "https://www.bloomberg.com/news/articles/us-cpi-cools-soft-landing-inflation",
                "published_date": (now - timedelta(days=2)).isoformat(),
                "relevance_score": 0.92,
                "query_topic": "US inflation CPI PPI consumer price index",
            },
            {
                "title": "Non-Farm Payrolls Surge Exceeds Consensus, Labor Market Retains Robust Momentum",
                "content": (
                    "The US economy added 215,000 jobs last month, substantially outperforming analyst estimates. "
                    "Average hourly earnings ticked up 0.3%, indicating steady wage growth without reigniting "
                    "severe wage-price spiral pressures. Unemployment remained steady at 3.8%."
                ),
                "source_url": "https://www.wsj.com/economy/jobs/us-nonfarm-payrolls-labor-market-report",
                "published_date": (now - timedelta(days=3)).isoformat(),
                "relevance_score": 0.89,
                "query_topic": "US employment jobs non-farm payrolls unemployment rate",
            },
            {
                "title": "Q3 GDP Growth Revised Upward to 2.8% on Strong Consumer Spending",
                "content": (
                    "Revised national economic accounts indicate gross domestic product expanded at a robust "
                    "2.8% annualized rate. Resilient consumer discretionary expenditures and corporate technology "
                    "investments offset sluggish residential real estate activity."
                ),
                "source_url": "https://www.ft.com/content/us-gdp-growth-revised-consumer-spending",
                "published_date": (now - timedelta(days=4)).isoformat(),
                "relevance_score": 0.88,
                "query_topic": "US GDP growth economic recession soft landing",
            },
            {
                "title": "Global Trade Balances Stabilize as Supply Chains Modernize Amid Tariff Talks",
                "content": (
                    "The US trade deficit narrowed slightly as goods exports rose while import volumes leveled. "
                    "Corporate supply chain managers reported improved delivery timelines, though ongoing geopolitical "
                    "tensions and maritime shipping rates continue to be closely monitored."
                ),
                "source_url": "https://www.cnbc.com/markets/global-trade-balance-tariffs-supply-chains",
                "published_date": (now - timedelta(days=5)).isoformat(),
                "relevance_score": 0.84,
                "query_topic": "US trade balance tariffs global supply chain",
            },
            {
                "title": "Producer Price Index (PPI) Drops Unexpectedly, Alleviating Upstream Cost Pressures",
                "content": (
                    "Wholesale prices fell 0.1% month-over-month, surprising economists who expected a modest rise. "
                    "Energy and intermediate raw material declines contributed directly to producer margin expansion."
                ),
                "source_url": "https://www.marketwatch.com/story/us-producer-prices-drop-unexpectedly",
                "published_date": (now - timedelta(days=6)).isoformat(),
                "relevance_score": 0.87,
                "query_topic": "US inflation CPI PPI consumer price index",
            },
        ]

        for item in synthetic_data:
            item["created_at"] = now.isoformat()

        return synthetic_data

    def scrape_macro_news(
        self, topics: Sequence[str] | None = None, days_back: int = 7
    ) -> list[dict]:
        """
        Scrape news across multiple macroeconomic topics and deduplicate articles by URL.
        If live scraping fails or client is unavailable, falls back to synthetic articles.
        """
        if not self.client:
            logger.info("Tavily client unavailable; generating synthetic news.")
            return self._generate_synthetic_news()

        target_topics = list(topics) if topics else self.DEFAULT_TOPICS
        all_articles: list[dict] = []
        seen_urls: set[str] = set()

        for topic in target_topics:
            query = f"{topic} economic news past {days_back} days"
            topic_articles = self.search_economic_news(query=query, max_results=10)

            for art in topic_articles:
                url = art.get("source_url")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                art["query_topic"] = topic
                all_articles.append(art)

        if not all_articles:
            logger.warning("No articles gathered from live search; falling back to synthetic dataset.")
            return self._generate_synthetic_news()

        logger.info("Total deduplicated articles gathered: %d", len(all_articles))
        return all_articles

    def analyze_sentiment(self, text: str) -> float:
        """
        Compute sentiment polarity score (-1.0 to +1.0) using TextBlob.
        Includes rule-based fallback if TextBlob is not installed or encountered error.
        """
        if not text or not text.strip():
            return 0.0

        if TextBlob is not None:
            try:
                blob = TextBlob(text)
                return round(float(blob.sentiment.polarity), 4)
            except Exception as e:
                logger.debug("TextBlob analysis error: %s. Using heuristic fallback.", e)

        # Fallback heuristic using domain sentiment lexicon
        pos_terms = {
            "growth", "expand", "gain", "surge", "beat", "rally", "resilient",
            "optimism", "robust", "increase", "bullish", "strong", "cools", "eased"
        }
        neg_terms = {
            "recession", "decline", "drop", "slump", "miss", "bearish", "fall",
            "weak", "crisis", "default", "inflationary", "loss", "down", "struggle"
        }

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        pos_count = sum(1 for w in words if w in pos_terms)
        neg_count = sum(1 for w in words if w in neg_terms)
        total_matched = pos_count + neg_count

        if total_matched == 0:
            return 0.0

        score = (pos_count - neg_count) / float(total_matched)
        return round(float(np_bound := max(min(score, 1.0), -1.0)), 4)

    def enrich_with_sentiment(self, articles: list[dict]) -> list[dict]:
        """Compute sentiment scores for each article based on combined title and content."""
        enriched: list[dict] = []
        for article in articles:
            combined_text = f"{article.get('title', '')}. {article.get('content', '')}"
            score = self.analyze_sentiment(combined_text)
            art_copy = dict(article)
            art_copy["sentiment_score"] = score
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

    def run(self, topics: Sequence[str] | None = None) -> list[dict]:
        """
        Execute full pipeline:
        1. Scrape or synthetically generate macroeconomic news
        2. Enrich articles with sentiment analysis
        3. Save raw output to raw data directory
        4. Return list of enriched article dictionaries
        """
        logger.info("Running EconomicNewsScraper pipeline...")
        articles = self.scrape_macro_news(topics=topics)
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
