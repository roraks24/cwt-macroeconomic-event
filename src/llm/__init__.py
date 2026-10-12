"""
CWT Macro Economics Sentiment Agent - LLM Modules
Includes:
  - LLMScorer: Structured JSON extraction of macroeconomic sentiment, regime, and hawk-dove metrics.
  - LLMAgent: Executive macroeconomic outlook synthesis and portfolio trade decision reasoning.
"""

from src.llm.llm_scorer import LLMScorer
from src.llm.llm_agent import LLMAgent

__all__ = ["LLMScorer", "LLMAgent"]
