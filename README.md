# CWT Macro Economics Sentiment Agent

A Python-based machine learning pipeline that predicts optimal macro trading strategies by ingesting macroeconomic event data, economic news, and market data.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│  Apify Calendar  │    │  Tavily News     │
│  (180 days)      │    │  (Recent)        │
└────────┬────────┘    └────────┬────────┘
         │                      │
         ▼                      ▼
┌────────────────────────────────────────┐
│         SQLite Database                 │
│  (macro_events + news + market_data)    │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│       Feature Engineering               │
│  • Surprise indicators                  │
│  • Time-based features                  │
│  • Rolling statistics                   │
│  • Sentiment features                   │
│  • Macro regime classification          │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│    LightGBM / XGBoost Model             │
│    Walk-Forward Validation              │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│         Outputs                         │
│  • Macro Outlook Report (1-page)        │
│  • Sentiment Charts                     │
│  • Strategy Performance Metrics         │
│  • Evaluation Report                    │
└────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd cwt-macro-agent
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your Apify and Tavily API keys
```

### 3. Run the Pipeline

```bash
# Full pipeline with synthetic data (no API keys needed)
python main.py --synthetic

# Full pipeline with live API data
python main.py

# Individual phases
python main.py --scrape-only      # Only ingest data
python main.py --train-only       # Only train model
python main.py --report-only      # Only generate reports
```

## 📊 Pipeline Phases

### Phase 1: Data Ingestion
- **Apify**: Scrapes macroeconomic calendar (CPI, FOMC, NFP, GDP, etc.) for the last 180 days
- **Tavily**: Collects recent economic news with relevance scoring
- **Market Data**: Historical price data for US Equity, Bonds, Gold, USD Index, Crude Oil

### Phase 2: Feature Engineering & Macro Summary
- **Surprise Features**: Actual vs. forecast deviations for each macro indicator
- **Time Features**: Day of week, month, distance from key events (FOMC, NFP, CPI)
- **Rolling Features**: Moving averages, momentum, volatility regimes
- **Sentiment Features**: News sentiment aggregation and momentum
- **Macro Regime**: Classification (inflationary/deflationary, growth/contraction)
- **1-Page Macro Outlook**: Automated macro environment assessment

### Phase 3: ML Model & Walk-Forward Validation
- **Algorithm**: LightGBM gradient boosted decision trees
- **Validation**: Walk-forward (rolling window) — no look-ahead bias
- **Target**: Next-day return direction (long/short signal)
- **Metrics**: Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Profit Factor

### Phase 4: Visualization
- Sentiment timeline charts
- Economic surprise heatmaps
- Strategy performance with drawdown analysis
- Feature importance rankings
- Walk-forward prediction accuracy
- Macro regime transition diagrams
- Feature correlation matrices

## 📁 Project Structure

```
cwt-macro-agent/
├── main.py                           # Pipeline orchestrator
├── config.py                         # Configuration & settings
├── requirements.txt                  # Python dependencies
├── .env.example                      # API key template
├── data/
│   ├── raw/                          # Raw scraped data (JSON)
│   └── processed/                    # Feature matrices (CSV)
├── src/
│   ├── scraper/
│   │   ├── apify_calendar.py         # Macro event scraping
│   │   └── tavily_news.py            # Economic news scraping
│   ├── database/
│   │   └── db_manager.py             # SQLAlchemy database layer
│   ├── features/
│   │   └── feature_engineering.py    # Quantitative feature creation
│   ├── model/
│   │   └── walk_forward.py           # ML model & validation
│   ├── visualization/
│   │   └── charts.py                 # Plotly/Matplotlib charts
│   └── summary/
│       └── macro_outlook.py          # 1-page macro report
└── reports/
    ├── evaluation_report.md          # Model performance report
    ├── macro_outlook.md              # Macro environment summary
    ├── model_results.json            # Raw model metrics
    └── charts/                       # Generated visualizations
```

## 🔑 API Keys Required

| Service | Purpose | Get Key |
|---------|---------|---------|
| **Apify** | Macro event calendar scraping | [apify.com](https://apify.com/) |
| **Tavily** | Economic news search | [tavily.com](https://tavily.com/) |
| **OpenAI** (optional) | Enhanced macro summaries | [openai.com](https://openai.com/) |

> **Note**: The pipeline can run in `--synthetic` mode without any API keys for demonstration purposes.

## 📈 Model Performance Metrics

The evaluation report includes:
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score

## 🛠️ Technical Stack

- **Python 3.11+**
- **Data**: Pandas, NumPy, SQLAlchemy
- **ML**: LightGBM, XGBoost, scikit-learn
- **NLP**: TextBlob (sentiment analysis)
- **Viz**: Plotly, Matplotlib, Seaborn
- **APIs**: Apify Client, Tavily Python SDK

## 📝 License

This project was created as an internship assessment for CrowdWisdomTrading.

---

*Built by Rohit Saini*
