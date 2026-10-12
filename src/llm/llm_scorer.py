"""
LLM Macroeconomic Scorer Module
===============================
Translates unstructured economic news (from Tavily) and macroeconomic releases
(from Apify) into standardized, high-density quantitative JSON scores using OpenAI.

JSON Output Schema:
  - sentiment_score: [-1.0, 1.0] (bearish to bullish)
  - hawk_dove_score: [-1.0, 1.0] (dovish/easing to hawkish/tightening)
  - macro_regime: 'disinflationary_expansion' | 'inflationary_expansion' | 'stagflation' | 'recessionary_contraction' | 'neutral_transition'
  - inflation_pressure: [-1.0, 1.0] (cooling/deflationary to surging)
  - growth_momentum: [-1.0, 1.0] (contracting to accelerating)
  - surprise_impact_score: [0.0, 1.0] (expected market re-pricing potency)
  - asset_biases: Dict[str, float] for US_EQUITY, US_BOND, GOLD, USD_INDEX, CRUDE_OIL
  - key_drivers: List[str]
  - confidence: [0.0, 1.0]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    ASSET_CLASSES,
    DATA_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger("CWT-LLMScorer")

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class LLMScorer:
    """
    Quantitative macroeconomic scorer powered by LLM JSON completions.
    Extracts multi-dimensional regime, sentiment, and hawk-dove features using Groq / OpenAI.
    """

    CACHE_FILE = DATA_DIR / "llm_scorer_cache.json"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = LLM_TEMPERATURE,
        provider: str | None = None,
    ) -> None:
        self.provider = (provider or LLM_PROVIDER or "groq").lower()
        self.temperature = temperature
        self.client = None
        self._quota_exhausted: bool = False
        self._cache: dict[str, dict] = self._load_cache()

        if self.provider == "groq":
            groq_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
            self.model = model or GROQ_MODEL or "openai/gpt-oss-120b"
            if groq_key and Groq is not None:
                try:
                    self.client = Groq(api_key=groq_key)
                    logger.info("LLMScorer Groq client initialized with model: %s", self.model)
                except Exception as e:
                    logger.warning("Failed to initialize Groq client: %s. Heuristic fallback active.", e)
            elif groq_key and OpenAI is not None:
                try:
                    self.client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
                    logger.info("LLMScorer OpenAI-compatible Groq client initialized with model: %s", self.model)
                except Exception as e:
                    logger.warning("Failed to initialize Groq OpenAI-compatible client: %s.", e)
            else:
                logger.info("Groq API key not found. Heuristic scoring active.")
        else:
            openai_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
            self.model = model or OPENAI_MODEL or "gpt-4o-mini"
            if openai_key and OpenAI is not None:
                try:
                    self.client = OpenAI(api_key=openai_key)
                    logger.info("LLMScorer OpenAI client initialized with model: %s", self.model)
                except Exception as e:
                    logger.warning("Failed to initialize OpenAI client in LLMScorer: %s. Rule-based fallback active.", e)
            else:
                logger.info("OpenAI API key not detected or library missing. Using heuristic quant scoring.")

    def _load_cache(self) -> dict[str, dict]:
        """Load previously scored articles from persistent JSON cache."""
        if self.CACHE_FILE.exists():
            try:
                with open(self.CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug("Failed to read cache file: %s", e)
        return {}

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        try:
            self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.debug("Failed to save cache file: %s", e)

    def _cache_key(self, text: str) -> str:
        """Generate deterministic MD5 hash key for caching."""
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def score_article(self, title: str, content: str = "", date_str: str = "") -> dict[str, Any]:
        """
        Score a single macroeconomic news article or release using LLM JSON completion.
        Returns validated structured dictionary conforming to the quant scoring schema.
        """
        combined = f"{title}. {content}".strip()
        ckey = self._cache_key(combined)

        if ckey in self._cache:
            return self._cache[ckey]

        # If LLM client is available and quota is not exhausted, prompt for structured JSON
        if self.client and not self._quota_exhausted:
            try:
                score_dict = self._call_llm_scorer(title, content, date_str)
                self._cache[ckey] = score_dict
                self._save_cache()
                return score_dict
            except Exception as e:
                err_msg = str(e)
                if "quota" in err_msg.lower() or "credit" in err_msg.lower() or "429" in err_msg:
                    logger.warning("OpenAI quota unavailable (%s). Switching instantly to quant heuristic scorer.", err_msg[:60])
                    self._quota_exhausted = True
                else:
                    logger.warning("LLM call failed for article '%s': %s. Falling back to heuristic scorer.", title[:40], err_msg)

        # Fallback to deterministic rule-based quantitative scorer
        score_dict = self._fallback_rule_scorer(title, content)
        self._cache[ckey] = score_dict
        self._save_cache()
        return score_dict

    def _call_llm_scorer(self, title: str, content: str, date_str: str) -> dict[str, Any]:
        """Execute OpenAI API call requesting JSON output format."""
        system_prompt = (
            "You are an expert Institutional Quantitative Macroeconomist and Portfolio Strategist. "
            "Analyze the provided economic news release or macro narrative and return a strictly valid JSON "
            "object with quantitative macroeconomic scores."
        )

        user_prompt = f"""Evaluate this macroeconomic release/news:
Date: {date_str or 'Recent'}
Title: {title}
Content: {content[:1000]}

Return a JSON object with EXACTLY this structure:
{{
  "sentiment_score": <float between -1.0 (extreme bearish for markets) and 1.0 (extreme bullish)>,
  "hawk_dove_score": <float between -1.0 (ultra-dovish / rate cuts) and 1.0 (ultra-hawkish / rate hikes)>,
  "macro_regime": <one of: "disinflationary_expansion", "inflationary_expansion", "stagflation", "recessionary_contraction", "neutral_transition">,
  "inflation_pressure": <float between -1.0 (sharp disinflation) and 1.0 (surging inflation)>,
  "growth_momentum": <float between -1.0 (severe contraction) and 1.0 (strong growth)>,
  "surprise_impact_score": <float between 0.0 and 1.0 (market re-pricing potential)>,
  "asset_biases": {{
    "US_EQUITY": <float -1.0 to 1.0>,
    "US_BOND": <float -1.0 to 1.0>,
    "GOLD": <float -1.0 to 1.0>,
    "USD_INDEX": <float -1.0 to 1.0>,
    "CRUDE_OIL": <float -1.0 to 1.0>
  }},
  "key_drivers": [<list of 2-3 short driver phrases>],
  "confidence": <float between 0.0 and 1.0>
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=1000,
        )

        raw_json = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_json)
        return self._normalize_score_schema(parsed)

    def _normalize_score_schema(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and clamp numeric bounds on LLM JSON output."""
        def _clamp(val: Any, low: float, high: float, default: float) -> float:
            try:
                f = float(val)
                return max(min(f, high), low)
            except (ValueError, TypeError):
                return default

        sentiment = _clamp(data.get("sentiment_score"), -1.0, 1.0, 0.0)
        hawk_dove = _clamp(data.get("hawk_dove_score"), -1.0, 1.0, 0.0)
        inflation = _clamp(data.get("inflation_pressure"), -1.0, 1.0, 0.0)
        growth = _clamp(data.get("growth_momentum"), -1.0, 1.0, 0.0)
        impact = _clamp(data.get("surprise_impact_score"), 0.0, 1.0, 0.5)
        conf = _clamp(data.get("confidence"), 0.0, 1.0, 0.8)

        valid_regimes = {
            "disinflationary_expansion",
            "inflationary_expansion",
            "stagflation",
            "recessionary_contraction",
            "neutral_transition",
        }
        regime = str(data.get("macro_regime", "neutral_transition")).lower()
        if regime not in valid_regimes:
            regime = "neutral_transition"

        raw_biases = data.get("asset_biases") or {}
        asset_biases = {}
        for a in ASSET_CLASSES:
            asset_biases[a] = _clamp(raw_biases.get(a), -1.0, 1.0, 0.0)

        drivers = data.get("key_drivers")
        if not isinstance(drivers, list):
            drivers = ["Macro condition shift"]

        return {
            "sentiment_score": round(sentiment, 4),
            "hawk_dove_score": round(hawk_dove, 4),
            "macro_regime": regime,
            "inflation_pressure": round(inflation, 4),
            "growth_momentum": round(growth, 4),
            "surprise_impact_score": round(impact, 4),
            "asset_biases": asset_biases,
            "key_drivers": [str(d)[:60] for d in drivers[:3]],
            "confidence": round(conf, 4),
        }

    def _fallback_rule_scorer(self, title: str, content: str) -> dict[str, Any]:
        """Deterministic financial lexicon scorer if LLM is unavailable."""
        text = f"{title}. {content}".lower()

        bullish_kw = ["gain", "surge", "beat", "rally", "growth", "cooling inflation", "rate cut", "easing", "expansion", "resilient"]
        bearish_kw = ["recession", "slump", "miss", "inflation surge", "rate hike", "hawkish", "contraction", "downturn", "crisis"]
        hawkish_kw = ["rate hike", "tightening", "sticky inflation", "higher for longer", "hawkish", "wage pressures"]
        dovish_kw = ["rate cut", "easing", "disinflation", "cooling labor", "dovish", "pause"]

        b_score = sum(1 for w in bullish_kw if w in text)
        r_score = sum(1 for w in bearish_kw if w in text)
        h_score = sum(1 for w in hawkish_kw if w in text)
        d_score = sum(1 for w in dovish_kw if w in text)

        tot_sent = b_score + r_score
        tot_hawk = h_score + d_score

        sentiment = (b_score - r_score) / float(tot_sent) if tot_sent > 0 else 0.0
        hawk_dove = (h_score - d_score) / float(tot_hawk) if tot_hawk > 0 else 0.0

        if sentiment > 0.2 and hawk_dove <= 0:
            regime = "disinflationary_expansion"
        elif sentiment < -0.2 and hawk_dove > 0:
            regime = "stagflation"
        elif sentiment < -0.2:
            regime = "recessionary_contraction"
        elif sentiment > 0.2:
            regime = "inflationary_expansion"
        else:
            regime = "neutral_transition"

        return {
            "sentiment_score": round(max(min(sentiment, 1.0), -1.0), 4),
            "hawk_dove_score": round(max(min(hawk_dove, 1.0), -1.0), 4),
            "macro_regime": regime,
            "inflation_pressure": round(hawk_dove * 0.5, 4),
            "growth_momentum": round(sentiment * 0.6, 4),
            "surprise_impact_score": 0.5,
            "asset_biases": {
                "US_EQUITY": round(sentiment * 0.8, 4),
                "US_BOND": round(-hawk_dove * 0.7, 4),
                "GOLD": round(-sentiment * 0.3 + hawk_dove * -0.2, 4),
                "USD_INDEX": round(hawk_dove * 0.8, 4),
                "CRUDE_OIL": round(sentiment * 0.5, 4),
            },
            "key_drivers": ["Rule-based financial lexicon alignment"],
            "confidence": 0.75,
        }

    def score_articles_batch(self, articles: Sequence[dict[str, Any]], max_workers: int = 1) -> list[dict[str, Any]]:
        """
        Enrich a collection of articles with full LLM JSON scores.
        Appends all structured score attributes directly to each article dictionary.
        """
        logger.info("Scoring %d news articles with LLMScorer (%s)...", len(articles), self.model)
        enriched: list[dict[str, Any]] = []

        for idx, art in enumerate(articles):
            title = art.get("title", "")
            content = art.get("content", "")
            dt_str = art.get("published_date", "")

            scores = self.score_article(title=title, content=content, date_str=dt_str)
            item = dict(art)
            item["llm_sentiment_score"] = scores["sentiment_score"]
            item["llm_hawk_dove_score"] = scores["hawk_dove_score"]
            item["llm_macro_regime"] = scores["macro_regime"]
            item["llm_inflation_pressure"] = scores["inflation_pressure"]
            item["llm_growth_momentum"] = scores["growth_momentum"]
            item["llm_surprise_impact"] = scores["surprise_impact_score"]
            item["llm_asset_biases"] = scores["asset_biases"]
            item["llm_key_drivers"] = scores["key_drivers"]
            item["llm_confidence"] = scores["confidence"]

            # Also maintain legacy sentiment_score field for downstream backward compatibility
            item["sentiment_score"] = scores["sentiment_score"]
            enriched.append(item)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(articles):
                logger.info("Scored %d / %d articles.", idx + 1, len(articles))

        return enriched


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scorer = LLMScorer()
    sample = scorer.score_article(
        title="Fed Approves 25 bps Rate Cut as Core Inflation Hits Three-Year Low",
        content="Chairman Powell signaled further easing may follow if labor market deceleration persists.",
    )
    print("\n--- LLM Scorer Output Sample (JSON) ---")
    print(json.dumps(sample, indent=2))
