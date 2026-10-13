"""
Quantitative Macro Sentiment Scorer (LLM-Free Architecture)
===========================================================
Re-exports QuantSentimentScorer for backward compatibility.
All scoring is deterministic, algorithmic, and executes offline without LLM APIs.
"""

from src.sentiment.sentiment_scorer import QuantSentimentScorer

# Backward compatible alias
LLMScorer = QuantSentimentScorer

__all__ = ["QuantSentimentScorer", "LLMScorer"]
