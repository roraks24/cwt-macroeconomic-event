"""
Quantitative Macro Strategy Engine (LLM-Free Architecture)
==========================================================
Re-exports MacroStrategyEngine for backward compatibility.
All macro outlooks and trade decisions are generated deterministically without LLM APIs.
"""

from src.strategy.strategy_engine import MacroStrategyEngine

# Backward compatible alias
LLMAgent = MacroStrategyEngine

__all__ = ["MacroStrategyEngine", "LLMAgent"]
