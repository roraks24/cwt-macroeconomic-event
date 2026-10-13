"""
CWT Macro Economics Sentiment Agent - Quantitative Analytics Modules (LLM-Free).
Provides backward-compatible exports for QuantSentimentScorer and MacroStrategyEngine.
"""

from src.sentiment.sentiment_scorer import QuantSentimentScorer, LLMScorer
from src.strategy.strategy_engine import MacroStrategyEngine, LLMAgent

__all__ = ["QuantSentimentScorer", "LLMScorer", "MacroStrategyEngine", "LLMAgent"]
