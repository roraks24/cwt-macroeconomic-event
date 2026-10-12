# CWT Macro Economics Sentiment Agent — Trade Recommendations

**Generated**: 2026-10-12 05:44 UTC  
**Methodology**: LightGBM Multi-Asset Predictive Model conditioned on Joined ForexFactory Releases & Tavily News Sentiment.

---

## 1. Executive Trade Signals

| Asset Class | Signal | Win Probability | Expected Return | Stop Loss | Take Profit | Primary Macro Driver |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **US_EQUITY** | ⚪ NEUTRAL | 50.0% | +0.00% | -0.45% | +0.75% | Bullish News Sentiment (+0.14) |
| **US_BOND** | 🟢 LONG | 63.0% | +0.09% | -0.42% | +0.70% | Bullish News Sentiment (+0.14) |
| **GOLD** | 🔴 SHORT | 56.1% | +0.05% | -0.48% | +0.81% | Bullish News Sentiment (+0.14) |
| **USD_INDEX** | ⚪ NEUTRAL | 50.0% | +0.00% | -0.53% | +0.88% | Bullish News Sentiment (+0.14) |
| **CRUDE_OIL** | 🟢 LONG | 79.4% | +2.50% | -5.10% | +8.49% | Bullish News Sentiment (+0.14) |

---

## 2. Quantitative Model Architecture

- **Calendar Integration**: Daily actual-versus-forecast surprise metrics from Apify (`xtracto/forexfactory-calendar`).
- **News NLP**: Polarity scoring via Tavily Search API with TextBlob NLP analysis.
- **Risk Management**: Volatility-scaled dynamic stops (1.5x 5d ATR equivalent) and profit targets (2.5x ATR).

*Report generated automatically by CWT Macro Economics Sentiment Agent.*
