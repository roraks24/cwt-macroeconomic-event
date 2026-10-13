# CWT Macro Economics Sentiment Agent -- Evaluation Report

**Generated**: 2026-10-13 13:24 UTC  
**End-to-End Pipeline**: `Tavily/Apify -> Quant Sentiment Scorer -> Features -> LightGBM Walk-Forward -> Metrics/Plotly -> Strategy Engine`

---

## 1. Executive Summary

This report evaluates the **CWT Quantitative Macroeconomic Trading Pipeline** (Pure Algorithmic / LLM-Free Architecture).
The pipeline ingests macroeconomic calendar surprises (ForexFactory via Apify), scores thematic economic news using an **Algorithmic Lexicon Scorer**, builds quantitative interaction features, validates trade forecasting using sliding **walk-forward validation**, and synthesizes executive decisions through a **Quantitative Strategy Engine**.

---

## 2. Quantitative Model Performance (Out-of-Sample)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | 47.76% | Directional forecast hit rate |
| **Annualized Sharpe Ratio** | 0.17 | Risk-adjusted excess return (annualized, 252d) |
| **Sortino Ratio** | 0.27 | Downside risk-penalized return ratio |
| **Maximum Drawdown (MDD)** | 2.96% | Peak-to-trough equity curve decline |
| **Win Rate** | 52.24% | Out-of-sample winning trade percentage |
| **Profit Factor** | 1.03 | Gross profits / Gross losses |
| **Cumulative Return** | 0.24% | Total walk-forward strategy return |
| **Total Evaluated Trades** | 67 | Out-of-sample execution instances |

---

## 3. Multi-Asset Strategy Matrix

| Asset Class | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate | Profit Factor | Cumulative Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **US_EQUITY** | 0.17 | 0.27 | 2.96% | 52.24% | 1.03 | 0.24% |
| **US_BOND** | 0.17 | 0.27 | 2.96% | 52.24% | 1.03 | 0.24% |
| **GOLD** | 0.17 | 0.27 | 2.96% | 52.24% | 1.03 | 0.24% |
| **USD_INDEX** | 0.17 | 0.27 | 2.96% | 52.24% | 1.03 | 0.24% |
| **CRUDE_OIL** | 0.17 | 0.27 | 2.96% | 52.24% | 1.03 | 0.24% |

---

## 4. Visual Deliverables & Interactive HTML Dashboard

- 🌐 **Interactive Web Dashboard**: [macro_dashboard.html](macro_dashboard.html) *(Single-file dashboard with hover tooltips, zoom, and zero setup)*

### Priority Charts (Plotly Interactive HTML & High-Res PNG):
1. **Sentiment Over Time with Event Markers**: [sentiment_timeline_events.html](charts/sentiment_timeline_events.html) | [PNG](charts/sentiment_timeline_events.png)
2. **Event-Study Dynamics (-3 to +5 Days)**: [event_study_dynamics.html](charts/event_study_dynamics.html) | [PNG](charts/event_study_dynamics.png)
3. **Strategy Matrix Heatmap (Event × Direction × Horizon)**: [strategy_matrix_heatmap.html](charts/strategy_matrix_heatmap.html) | [PNG](charts/strategy_matrix_heatmap.png)
4. **Equity Curve with Drawdown vs. SPY Benchmark**: [equity_curve_drawdown.html](charts/equity_curve_drawdown.html) | [PNG](charts/equity_curve_drawdown.png)
5. **Predicted vs. Realised P&L (Model Calibration)**: [predicted_vs_realised_pnl.html](charts/predicted_vs_realised_pnl.html) | [PNG](charts/predicted_vs_realised_pnl.png)
6. **Economic Surprise vs. Next-Day SPY Return**: [surprise_vs_next_day_return.html](charts/surprise_vs_next_day_return.html) | [PNG](charts/surprise_vs_next_day_return.png)
7. **Rolling 30-Trade Sharpe Ratio**: [rolling_sharpe.html](charts/rolling_sharpe.html) | [PNG](charts/rolling_sharpe.png)

---

## 5. Artifact Deliverables Summary

- **Macroeconomic Outlook**: [macro_outlook.md](reports/macro_outlook.md)
- **Trade Decision Matrix**: [trade_signals.md](reports/trade_signals.md)
- **Structured Trade Data**: [trade_predictions.json](reports/trade_predictions.json)
- **Model Evaluation Results**: [model_results.json](reports/model_results.json)

---

*Report automatically compiled by CWT Macro Economics Sentiment Agent.*
