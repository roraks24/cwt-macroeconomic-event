"""
Quantitative Sentiment & Macro Lexicon Scorer
=============================================
Deterministic, rule-based financial NLP scorer that extracts macroeconomic sentiment,
monetary policy (hawk/dove) stance, economic regime classification, and asset tilts
using domain-specific financial lexicons.

100% offline, reproducible, and instantaneous. Zero external LLM API dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Add project root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ASSET_CLASSES, DATA_DIR, PROCESSED_DIR  # noqa: E402

logger = logging.getLogger("CWT-SentimentScorer")


class QuantSentimentScorer:
    """
    Algorithmic financial NLP scorer.
    Evaluates news articles against macroeconomic domain lexicons to produce
    structured sentiment polarity, policy stance, and multi-asset directional tilts.
    """

    # Comprehensive Loughran-McDonald & Macroeconomic Lexicons
    BULLISH_KEYWORDS = [
        "surge", "rally", "gain", "beat", "expansion", "growth", "rebound", "boost",
        "record high", "outperform", "resilient", "recovery", "optimism", "uptick",
        "cooling inflation", "disinflation", "rate cut", "monetary easing", "stimulus",
        "hiring gain", "productivity", "strong demand", "consumer confidence", "upbeat",
        "easing pressures", "soft landing", "solid", "advancement", "upside",
    ]

    BEARISH_KEYWORDS = [
        "slump", "plunge", "decline", "fall", "miss", "contraction", "recession", "drop",
        "downturn", "crisis", "stagflation", "weakness", "layoff", "job cut", "turmoil",
        "inflation surge", "rate hike", "tightening", "restrictive", "hard landing",
        "liquidity crunch", "bank failure", "default", "downgrade", "headwind", "drag",
        "sluggish", "pessimism", "deteriorate", "slowdown", "selloff",
    ]

    HAWKISH_KEYWORDS = [
        "rate hike", "interest rate increase", "tightening", "sticky inflation",
        "higher for longer", "hawkish", "wage pressures", "demand overheating",
        "restrictive policy", "balance sheet reduction", "quantitative tightening",
        "inflation risk", "overheating", "fed hawkish", "curb demand", "persistent inflation",
    ]

    DOVISH_KEYWORDS = [
        "rate cut", "interest rate reduction", "monetary easing", "disinflation",
        "cooling labor", "dovish", "rate pause", "accommodation", "liquidity support",
        "fed dovish", "pivot", "policy normalization", "labor cooling", "subdued inflation",
        "economic slack", "sub-trend growth",
    ]

    INFLATION_KEYWORDS = [
        "cpi", "ppi", "pce", "inflation", "cost of living", "price index", "wage growth",
        "commodity prices", "energy costs", "food prices", "shelter inflation",
    ]

    GROWTH_KEYWORDS = [
        "gdp", "pmi", "retail sales", "industrial production", "factory orders",
        "consumer spending", "capex", "capital expenditure", "ism manufacturing",
    ]

    def __init__(self, cache_file: Optional[Union[str, Path]] = None) -> None:
        """Initialize the quantitative sentiment scorer."""
        self.cache_file = Path(cache_file) if cache_file else DATA_DIR / "sentiment_scorer_cache.json"
        self._cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load persistent cache if available."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read cache %s: %s", self.cache_file, e)
        return {}

    def _save_cache(self) -> None:
        """Save persistent cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning("Could not save cache %s: %s", self.cache_file, e)

    def _cache_key(self, text: str) -> str:
        """Generate deterministic cache key for text content."""
        clean_text = re.sub(r"\s+", " ", text.strip().lower())
        return str(abs(hash(clean_text)))

    def score_article(
        self,
        title: str,
        content: str = "",
        date_str: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate an individual article against financial and macroeconomic lexicons.

        Returns:
            Dict containing sentiment_score, hawk_dove_score, macro_regime,
            inflation_pressure, growth_momentum, surprise_impact_score, asset_biases,
            key_drivers, and confidence.
        """
        text = f"{title}. {content}".strip()
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        lower_text = text.lower()

        # Count keyword occurrences
        bull_hits = sum(1 for kw in self.BULLISH_KEYWORDS if kw in lower_text)
        bear_hits = sum(1 for kw in self.BEARISH_KEYWORDS if kw in lower_text)
        hawk_hits = sum(1 for kw in self.HAWKISH_KEYWORDS if kw in lower_text)
        dove_hits = sum(1 for kw in self.DOVISH_KEYWORDS if kw in lower_text)
        inf_hits = sum(1 for kw in self.INFLATION_KEYWORDS if kw in lower_text)
        gro_hits = sum(1 for kw in self.GROWTH_KEYWORDS if kw in lower_text)

        # 1. Sentiment Polarity [-1.0, 1.0]
        tot_sent = bull_hits + bear_hits
        if tot_sent > 0:
            sentiment_score = (bull_hits - bear_hits) / float(tot_sent)
        else:
            sentiment_score = 0.0

        # 2. Hawk-Dove Monetary Stance [-1.0, 1.0]
        tot_policy = hawk_hits + dove_hits
        if tot_policy > 0:
            hawk_dove_score = (hawk_hits - dove_hits) / float(tot_policy)
        else:
            hawk_dove_score = 0.0

        # 3. Inflation Pressure [0.0, 1.0]
        inflation_pressure = min(max(0.40 + (hawk_hits - dove_hits) * 0.15 + inf_hits * 0.05, 0.0), 1.0)

        # 4. Growth Momentum [-1.0, 1.0]
        growth_momentum = min(max(sentiment_score * 0.70 + (gro_hits * 0.05), -1.0), 1.0)

        # 5. Surprise Impact Score [0.0, 1.0]
        total_density = bull_hits + bear_hits + hawk_hits + dove_hits
        surprise_impact = min(max(0.30 + total_density * 0.08, 0.20), 0.95)

        # 6. Macro Regime Classification
        if sentiment_score >= 0.15 and hawk_dove_score <= 0.0:
            regime = "disinflationary_expansion"
        elif sentiment_score < -0.15 and hawk_dove_score > 0.10:
            regime = "stagflation"
        elif sentiment_score < -0.15:
            regime = "recessionary_contraction"
        elif sentiment_score >= 0.15 and hawk_dove_score > 0.0:
            regime = "inflationary_expansion"
        else:
            regime = "neutral_transition"

        # 7. Asset-Class Directional Biases [-1.0, 1.0]
        asset_biases = {
            "US_EQUITY": round(float(np.clip(sentiment_score * 0.85 - hawk_dove_score * 0.30, -1.0, 1.0)), 4),
            "US_BOND": round(float(np.clip(-hawk_dove_score * 0.80 - sentiment_score * 0.20, -1.0, 1.0)), 4),
            "GOLD": round(float(np.clip(-hawk_dove_score * 0.50 + (1.0 - sentiment_score) * 0.30, -1.0, 1.0)), 4),
            "USD_INDEX": round(float(np.clip(hawk_dove_score * 0.75 - sentiment_score * 0.15, -1.0, 1.0)), 4),
            "CRUDE_OIL": round(float(np.clip(growth_momentum * 0.70 + inflation_pressure * 0.25, -1.0, 1.0)), 4),
        }

        # 8. Detected Key Themes
        detected_themes = []
        if hawk_hits > dove_hits and hawk_hits > 0:
            detected_themes.append("Monetary Restraint / Hawkish Policy")
        elif dove_hits > hawk_hits and dove_hits > 0:
            detected_themes.append("Policy Normalization / Rate Easing")
        if inf_hits > 0:
            detected_themes.append("Inflation Dynamics & Price Pressures")
        if gro_hits > 0:
            detected_themes.append("Macro Real Activity & Growth")
        if bull_hits > bear_hits and bull_hits > 0:
            detected_themes.append("Corporate & Demand Resilience")
        elif bear_hits > bull_hits and bear_hits > 0:
            detected_themes.append("Cyclical Slowdown & Valuation Friction")

        if not detected_themes:
            detected_themes = ["Balanced Macroeconomic Conditions"]

        confidence = round(float(np.clip(0.60 + total_density * 0.05, 0.60, 0.95)), 3)

        result = {
            "sentiment_score": round(float(np.clip(sentiment_score, -1.0, 1.0)), 4),
            "hawk_dove_score": round(float(np.clip(hawk_dove_score, -1.0, 1.0)), 4),
            "macro_regime": regime,
            "inflation_pressure": round(inflation_pressure, 4),
            "growth_momentum": round(growth_momentum, 4),
            "surprise_impact_score": round(surprise_impact, 4),
            "asset_biases": asset_biases,
            "key_drivers": detected_themes[:3],
            "confidence": confidence,
        }

        self._cache[key] = result
        return result

    def score_articles_batch(
        self,
        articles: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Enrich a collection of articles with quantitative sentiment metrics.
        Executes instantly in-memory with zero external network requests.
        """
        if not articles:
            return []

        logger.info("Executing algorithmic sentiment scoring across %d news articles...", len(articles))
        enriched: List[Dict[str, Any]] = []

        for idx, art in enumerate(articles):
            title = art.get("title", "")
            content = art.get("content", "")
            dt_str = art.get("published_date", "")

            scores = self.score_article(title=title, content=content, date_str=dt_str)
            item = dict(art)
            item["sentiment_score"] = scores["sentiment_score"]
            item["hawk_dove_score"] = scores["hawk_dove_score"]
            item["macro_regime"] = scores["macro_regime"]
            item["inflation_pressure"] = scores["inflation_pressure"]
            item["growth_momentum"] = scores["growth_momentum"]
            item["surprise_impact_score"] = scores["surprise_impact_score"]
            item["asset_biases"] = scores["asset_biases"]
            item["key_drivers"] = scores["key_drivers"]
            item["confidence"] = scores["confidence"]

            # Retain backward-compatible alias keys
            item["llm_sentiment_score"] = scores["sentiment_score"]
            item["llm_hawk_dove_score"] = scores["hawk_dove_score"]
            item["llm_macro_regime"] = scores["macro_regime"]
            item["llm_inflation_pressure"] = scores["inflation_pressure"]
            item["llm_growth_momentum"] = scores["growth_momentum"]
            item["llm_surprise_impact"] = scores["surprise_impact_score"]
            item["llm_asset_biases"] = scores["asset_biases"]
            item["llm_key_drivers"] = scores["key_drivers"]
            item["llm_confidence"] = scores["confidence"]

            enriched.append(item)

        self._save_cache()
        logger.info("Successfully scored %d articles algorithmically.", len(enriched))
        return enriched


# Alias for backward compatibility
LLMScorer = QuantSentimentScorer
