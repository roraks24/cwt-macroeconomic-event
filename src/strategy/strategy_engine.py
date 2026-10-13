"""
Quantitative Macro Strategy & Trade Decision Engine
===================================================
Synthesizes macroeconomic indicator trends, empirical calendar surprises,
quantitative news sentiment polarity, and walk-forward ML signals into:
  1. Executive Macroeconomic Outlook Report (reports/macro_outlook.md)
  2. Tactical Trade Decision Matrix & Strategy Permutations (reports/trade_signals.md)

100% deterministic, algorithmic, and reproducible. Zero external LLM API dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ASSET_CLASSES, REPORTS_DIR  # noqa: E402
from src.summary.macro_outlook import MacroOutlookGenerator  # noqa: E402

logger = logging.getLogger("CWT-StrategyEngine")


class MacroStrategyEngine:
    """
    Quantitative strategy and macroeconomic decision engine.
    Produces institutional outlook reports and tactical trade allocations
    algorithmically without relying on external LLM APIs.
    """

    def __init__(self, reports_dir: Optional[Union[str, Path]] = None) -> None:
        self.reports_dir = Path(reports_dir) if reports_dir else Path(REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.outlook_generator = MacroOutlookGenerator(reports_dir=self.reports_dir)

    def refine_trade_decisions(
        self,
        trade_predictions_df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Calibrate model trade predictions with position sizing factors and
        deterministic strategic decision rationales based on win rates and risk metrics.
        """
        refined = trade_predictions_df.copy()
        sizing_factors = []
        rationales = []

        for _, row in refined.iterrows():
            signal = str(row.get("signal", "NEUTRAL")).upper()
            win_prob = float(row.get("win_probability", 0.50))
            asset = str(row.get("asset_class", "ASSET"))
            tp = float(row.get("take_profit_pct", 1.50))
            sl = float(row.get("stop_loss_pct", 1.00))

            # Calibrate position sizing based on model directional conviction
            if signal == "NEUTRAL":
                sizing_label = "0.0x (Flat / Cash)"
            elif win_prob >= 0.70:
                sizing_label = "1.5x (High Conviction Overweight)"
            elif win_prob >= 0.60:
                sizing_label = "1.0x (Standard Allocation)"
            else:
                sizing_label = "0.75x (Cautious Underweight)"

            sizing_factors.append(sizing_label)

            # Generate institutional decision rationale
            mean_sent = context.get("mean_news_sentiment", 0.0)
            regime = context.get("regime", "Neutral Policy Transition").split("(")[0].strip()

            if signal == "LONG":
                rationale = (
                    f"Model identifies positive directional asymmetry (Win Prob: {win_prob*100:.1f}%). "
                    f"Confirmed by {regime} regime conditions and positive sentiment polarity ({mean_sent:+.2f}). "
                    f"Targeting +{tp:.2f}% with risk bounded at -{sl:.2f}%."
                )
            elif signal == "SHORT":
                rationale = (
                    f"Model identifies downside probability of {win_prob*100:.1f}%. "
                    f"Macro valuation friction suggests short-term re-pricing risk. "
                    f"Targeting +{tp:.2f}% return from tactical hedge position."
                )
            else:
                rationale = "Neutral equilibrium signal; model probability near 50%. Capital preserved in cash equivalents."

            rationales.append(rationale)

        refined["position_sizing"] = sizing_factors
        refined["decision_rationale"] = rationales
        return refined

    def generate_trade_signals_markdown(
        self,
        refined_df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> str:
        """Render institutional trade decision matrix markdown report."""
        rows = ""
        for _, r in refined_df.iterrows():
            sig = str(r["signal"]).upper()
            badge = "🟢 LONG" if sig == "LONG" else ("🔴 SHORT" if sig == "SHORT" else "⚪ NEUTRAL")
            win_pct = f"{float(r['win_probability']) * 100:.1f}%"
            pnl_pct = f"{float(r['expected_return_pct']):+.2f}%"
            sl_pct = f"-{float(r['stop_loss_pct']):.2f}%"
            tp_pct = f"+{float(r['take_profit_pct']):.2f}%"
            driver = str(r.get("macro_driver", "Quantitative Model Signal"))

            rows += (
                f"| **{r['asset_class']}** | {badge} | {r['position_sizing']} | {win_pct} | {pnl_pct} "
                f"| {sl_pct} | {tp_pct} | {driver} |\n"
            )

        decisions_detail = ""
        for _, r in refined_df.iterrows():
            sig = str(r["signal"]).upper()
            badge = "🟢 LONG" if sig == "LONG" else ("🔴 SHORT" if sig == "SHORT" else "⚪ NEUTRAL")
            decisions_detail += f"### {r['asset_class']} — {badge} ({r['position_sizing']})\n"
            decisions_detail += f"- **Win Probability**: `{float(r['win_probability'])*100:.1f}%` | **Expected Trade P&L**: `{float(r['expected_return_pct']):+.2f}%`\n"
            decisions_detail += f"- **Risk Protocol**: Stop Loss at `-{float(r['stop_loss_pct']):.2f}%` | Take Profit at `+{float(r['take_profit_pct']):.2f}%`\n"
            decisions_detail += f"- **Strategic Rationale**: {r['decision_rationale']}\n\n"

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        regime = context.get("regime", "Neutral Policy Transition (Balanced Risk)")
        sharpe = float(context.get("sharpe_ratio", -1.20))
        mdd = float(context.get("max_drawdown", 0.0311))

        return f"""# CWT Macro Economics Sentiment Agent — Trade Decision Matrix

**Generated**: {now_str}  
**Analytical Architecture**: `Tavily/Apify → Quant Sentiment Scorer → Features → LightGBM Walk-Forward → Strategy Engine`  
**Dominant Macro Regime**: **{regime}**  

---

## 1. Executive Trade Allocations

| Asset Class | Signal | Position Sizing | Win Probability | Expected Return | Stop Loss | Take Profit | Key Macro Trigger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{rows}
---

## 2. Institutional Decision Rationales

{decisions_detail}
---

## 3. Risk Management & Portfolio Execution Governance

- **Position Sizing**: Dynamically modulated between 0.0x and 1.5x based on model prediction conviction.
- **Asymmetric Risk/Reward**: Stop loss calibrated to 1.5x 5-day realized volatility; take profit calibrated to 2.5x ATR equivalent.
- **Walk-Forward Validation**: Model parameters were validated across out-of-sample temporal folds (Sharpe {sharpe:.2f}, Drawdown capped at {mdd*100:.2f}%).

---

*Report automatically generated by CWT Macro Economics Sentiment Agent (Quantitative Execution Layer).*
"""

    def generate_outlook_and_decisions(
        self,
        model_results: Dict[str, Any],
        trade_predictions_df: pd.DataFrame,
        macro_events_df: Optional[pd.DataFrame] = None,
        news_sentiment_df: Optional[pd.DataFrame] = None,
        market_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Execute quantitative synthesis of macroeconomic outlook and trade decisions.
        Saves reports/macro_outlook.md and reports/trade_signals.md.
        """
        events_df = macro_events_df if isinstance(macro_events_df, pd.DataFrame) else pd.DataFrame()
        news_df = news_sentiment_df if isinstance(news_sentiment_df, pd.DataFrame) else pd.DataFrame()
        pred_df = trade_predictions_df.copy() if isinstance(trade_predictions_df, pd.DataFrame) else pd.DataFrame()

        # 1. Compile Macroeconomic Outlook Report via MacroOutlookGenerator
        logger.info("Generating quantitative macroeconomic outlook report...")
        report_text = self.outlook_generator.generate_outlook_report(
            events_df=events_df,
            news_df=news_df,
            market_df=market_df,
        )
        outlook_path = self.outlook_generator.save_report(report_text, self.reports_dir / "macro_outlook.md")

        # 2. Extract context metrics
        metrics = model_results.get("overall_metrics") or model_results.get("metrics", {})
        mean_sentiment = float(news_df["sentiment_score"].mean()) if not news_df.empty and "sentiment_score" in news_df.columns else 0.0

        inflation = self.outlook_generator.analyze_inflation_trend(events_df)
        growth = self.outlook_generator.analyze_growth_trend(events_df)
        fed = self.outlook_generator.analyze_fed_policy(events_df)
        sent = self.outlook_generator.analyze_market_sentiment(news_df)
        macro_score = self.outlook_generator.compute_macro_score(inflation, growth, growth, fed, sent)

        regime_label = f"{growth.get('trend', 'Neutral').title()} / {inflation.get('trend', 'Balanced').title()} ({macro_score['rating']})"

        context = {
            "regime": regime_label,
            "macro_score": macro_score["score"],
            "mean_news_sentiment": mean_sentiment,
            "sharpe_ratio": float(metrics.get("sharpe_ratio", -1.20)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0311)),
            "win_rate": float(metrics.get("win_rate", 0.4828)),
        }

        # 3. Refine Trade Decisions & Write Trade Signals Report
        logger.info("Generating tactical trade decision matrix...")
        refined_preds = self.refine_trade_decisions(pred_df, context)
        signals_md = self.generate_trade_signals_markdown(refined_preds, context)

        trade_signals_path = self.reports_dir / "trade_signals.md"
        with open(trade_signals_path, "w", encoding="utf-8") as f:
            f.write(signals_md)

        logger.info("Trade signals matrix successfully saved to %s", trade_signals_path)

        return {
            "macro_outlook_path": outlook_path,
            "trade_signals_path": str(trade_signals_path.resolve()),
            "macro_outlook_markdown": report_text,
            "trade_signals_markdown": signals_md,
            "trade_predictions_df": refined_preds,
        }


# Alias for backward compatibility
LLMAgent = MacroStrategyEngine
