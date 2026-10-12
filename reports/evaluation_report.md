# CWT Macro Economics Sentiment Agent -- Evaluation Report

**Generated**: 2026-10-12 07:02 UTC  
**End-to-End Pipeline**: `Tavily/Apify -> LLM Scorer (JSON) -> Features -> LightGBM Walk-Forward -> Metrics -> LLM Decision Agent`

---

## 1. Executive Summary

This report evaluates the **CWT Quantitative Macroeconomic Trading Pipeline**.
The pipeline ingests macroeconomic calendar surprises (ForexFactory via Apify), scores thematic economic news using an **LLM Scorer (JSON)**, builds quantitative interaction features, validates trade forecasting using sliding **walk-forward validation**, and synthesizes executive decisions through an **LLM Reasoning Agent**.

---

## 2. Quantitative Model Performance (Out-of-Sample)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | 34.48% | Directional forecast hit rate |
| **Annualized Sharpe Ratio** | -1.20 | Risk-adjusted excess return (annualized, 252d) |
| **Sortino Ratio** | -1.68 | Downside risk-penalized return ratio |
| **Maximum Drawdown (MDD)** | 3.11% | Peak-to-trough equity curve decline |
| **Win Rate** | 48.28% | Out-of-sample winning trade percentage |
| **Profit Factor** | 0.82 | Gross profits / Gross losses |
| **Cumulative Return** | -1.08% | Total walk-forward strategy return |
| **Total Evaluated Trades** | 29 | Out-of-sample execution instances |

---

## 3. Multi-Asset Strategy Matrix

| Asset Class | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate | Profit Factor | Cumulative Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **US_EQUITY** | -1.20 | -1.68 | 3.11% | 48.28% | 0.82 | -1.08% |
| **US_BOND** | -1.20 | -1.68 | 3.11% | 48.28% | 0.82 | -1.08% |
| **GOLD** | -1.20 | -1.68 | 3.11% | 48.28% | 0.82 | -1.08% |
| **USD_INDEX** | -1.20 | -1.68 | 3.11% | 48.28% | 0.82 | -1.08% |
| **CRUDE_OIL** | -1.20 | -1.68 | 3.11% | 48.28% | 0.82 | -1.08% |

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
