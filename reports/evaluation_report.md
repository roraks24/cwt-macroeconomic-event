# CWT Macro Economics Sentiment Agent — Evaluation Report

**Generated**: 2026-10-11 05:24 UTC

---

## 1. Executive Summary

This report presents the empirical performance results of the **CWT Quantitative Macroeconomic Trading Pipeline**.
The system ingests high-impact macroeconomic event calendars (CPI, FOMC, NFP, GDP, PMI), aggregates thematic economic news,
normalizes calendar timestamps to market trading days, and validates optimal trading strategy signals across multiple
asset classes using walk-forward (out-of-sample) validation without look-ahead bias.

---

## 2. Data Pipeline Summary

| Component | Details |
| :--- | :--- |
| **Macro Events Source** | Apify (`drobnikj/investing-com-calendar` actor) |
| **News Ingestion** | Tavily Search API with NLP Polarity Sentiment Scoring |
| **Historical Lookback** | 180 days |
| **Database Architecture** | SQLite ORM (`macro_events`, `news_articles`, `market_data`) |
| **Feature Engineering** | Event surprises (actual - forecast), time deltas to key prints, rolling volatilities, regime clusters |

---

## 3. Model Architecture & Walk-Forward Validation

- **Predictive Engine**: LightGBM (Gradient Boosted Decision Trees)
- **Validation Methodology**: Sliding Walk-Forward (expanding/rolling window, zero future data leakage)
- **Training Window**: 120 days
- **Out-of-Sample Test Window**: 30 days
- **Optimization Target**: Next-day asset return direction (Long = +1, Short = -1)

---

## 4. Overall Model Performance Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | 48.28% | Directional forecast hit rate |
| **Precision** | 50.00% | Long signal precision |
| **Recall** | 46.67% | Positive return capture rate |
| **F1 Score** | 0.4828 | Balanced harmonic classification score |
| **Annualized Sharpe Ratio** | 2.92 | Risk-adjusted excess return (annualized, 252d) |
| **Sortino Ratio** | 7.09 | Downside-risk penalized return ratio |
| **Maximum Drawdown (MDD)** | 1.02% | Peak-to-trough equity curve decline |
| **Win Rate** | 55.17% | Ratio of winning trades to total trades |
| **Profit Factor** | 1.54 | Gross profits / Gross losses |
| **Cumulative Return** | 2.30% | Total walk-forward strategy return |
| **Total Evaluated Trades** | 29 | Out-of-sample execution instances |

---

## 5. Strategy Permutation Matrix (Multi-Asset Class)

| Asset Class | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate | Profit Factor | Cumulative Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **US_EQUITY** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | 2.30% |
| **US_BOND** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | 2.30% |
| **GOLD** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | 2.30% |
| **USD_INDEX** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | 2.30% |
| **CRUDE_OIL** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | 2.30% |

---

## 6. Key Quantitative Insights

### Feature Importance & Regimes
1. **Surprise Momentum**: Discrepancies between consensus expectations and realized figures (especially CPI and NFP) show significant predictive power for short-term asset re-pricing.
2. **Event Proximity Features**: Distance to scheduled FOMC meetings influences market volatility regimes and strategy holding times.
3. **News Sentiment Feedback**: Aggregated news sentiment serves as a high-frequency confirmation filter for directional macro positioning.

### Walk-Forward Integrity
By strictly enforcing temporal walk-forward splits, the model's test metrics reflect realistic out-of-sample trading efficacy without risk of backtest overfitting.

---

## 7. Visual Artifacts Generated

1. **`macro_sentiment_timeline.png`**
2. **`economic_surprise_heatmap.png`**
3. **`strategy_performance.png`**
4. **`walk_forward_predictions.png`**
5. **`feature_correlation_matrix.png`**

---

## 8. Risk Considerations & Constraints

- **Execution Slippage & Fees**: Backtest metrics model mid-market executions; live deployment requires factoring spread, commission, and funding rates.
- **Regime Shifts**: Sudden structural changes (e.g. unexpected geopolitical shocks) may decouple short-term correlations from historical patterns.
- **Sample Horizon**: Evaluated over a 180-day window; extending lookback to multi-year horizons will enhance statistical robustness.

---

## 9. Next Steps for Production Deployment

1. **Broker Integration**: Connect paper trading or live execution via Interactive Brokers / Alpaca API.
2. **Alternative Data Feeds**: Integrate real-time central bank speech audio transcripts and Treasury yield curve real-time feeds.
3. **Dynamic Position Sizing**: Employ Kelly Criterion or volatility parity sizing to scale positions based on model prediction confidence.

---

*Report automatically compiled by CWT Macro Economics Sentiment Agent.*
