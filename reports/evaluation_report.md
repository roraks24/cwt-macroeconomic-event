# CWT Macro Economics Sentiment Agent -- Evaluation Report

**Generated**: 2026-10-12 06:33 UTC  
**End-to-End Pipeline**: `Tavily/Apify -> LLM Scorer (JSON) -> Features -> LightGBM Walk-Forward -> Metrics -> LLM Decision Agent`

---

## 1. Executive Summary

This report evaluates the **CWT Quantitative Macroeconomic Trading Pipeline**.
The pipeline ingests macroeconomic calendar surprises (ForexFactory via Apify), scores thematic economic news using an **LLM Scorer (JSON)**, builds quantitative interaction features, validates trade forecasting using sliding **walk-forward validation**, and synthesizes executive decisions through an **LLM Reasoning Agent**.

---

## 2. Quantitative Model Performance (Out-of-Sample)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | 30.77% | Directional forecast hit rate |
| **Annualized Sharpe Ratio** | -0.88 | Risk-adjusted excess return (annualized, 252d) |
| **Sortino Ratio** | -1.10 | Downside risk-penalized return ratio |
| **Maximum Drawdown (MDD)** | 1.22% | Peak-to-trough equity curve decline |
| **Win Rate** | 46.15% | Out-of-sample winning trade percentage |
| **Profit Factor** | 0.86 | Gross profits / Gross losses |
| **Cumulative Return** | -0.22% | Total walk-forward strategy return |
| **Total Evaluated Trades** | 13 | Out-of-sample execution instances |

---

## 3. Multi-Asset Strategy Matrix

| Asset Class | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate | Profit Factor | Cumulative Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **US_EQUITY** | -0.88 | -1.10 | 1.22% | 46.15% | 0.86 | -0.22% |
| **US_BOND** | -0.88 | -1.10 | 1.22% | 46.15% | 0.86 | -0.22% |
| **GOLD** | -0.88 | -1.10 | 1.22% | 46.15% | 0.86 | -0.22% |
| **USD_INDEX** | -0.88 | -1.10 | 1.22% | 46.15% | 0.86 | -0.22% |
| **CRUDE_OIL** | -0.88 | -1.10 | 1.22% | 46.15% | 0.86 | -0.22% |

---

## 4. Visual Artifacts Generated

1. **`macro_sentiment_timeline.png`**
2. **`economic_surprise_heatmap.png`**
3. **`strategy_performance.png`**
4. **`walk_forward_predictions.png`**
5. **`feature_correlation_matrix.png`**

---

## 5. Artifact Deliverables Summary

- **Macroeconomic Outlook**: [macro_outlook.md](reports/macro_outlook.md)
- **Trade Decision Matrix**: [trade_signals.md](reports/trade_signals.md)
- **Structured Trade Data**: [trade_predictions.json](reports/trade_predictions.json)
- **Model Evaluation Results**: [model_results.json](reports/model_results.json)

---

*Report automatically compiled by CWT Macro Economics Sentiment Agent.*
