# CWT Macro Economics Sentiment Agent

A quantitative macroeconomic prediction pipeline that forecasts multi-asset trading strategy signals by joining:
1. **Macroeconomic Calendar Releases** (Apify ForexFactory Calendar)
2. **Thematic Economic News & Sentiment** (Tavily Search API)
3. **Multi-Asset Historical Price Action** (US Equity, Bonds, Gold, USD, Crude Oil)

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐      ┌───────────────────────────────────┐
│     Apify ForexFactory Calendar │      │  Tavily Historical Economic News  │
│  (xtracto/forexfactory-calendar)│      │  (Thematic Queries + NLP Polarity)│
└────────────────┬────────────────┘      └─────────────────┬─────────────────┘
                 │                                         │
                 ▼                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              SQLite Database                               │
│                   (macro_events + news_articles + market_data)             │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    predict_trades.py (Joined Datasets)                     │
│  • Timestamp & Calendar Day Alignment                                      │
│  • Daily Surprises (CPI, NFP, FOMC) & Interaction Metrics                  │
│  • Rolling 3d/7d News Sentiment Polarity & Momentum                        │
│  • Rolling Volatility & Market Regime Classification                       │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│             Predictive Engine (LightGBM Walk-Forward Validation)            │
│  • Directional Trade Signals (LONG / SHORT / NEUTRAL)                      │
│  • Win Probability & Confidence Scores                                     │
│  • Expected Return & Volatility-Scaled Dynamic Stops/Targets               │
│  • Primary Macro Driver Rationales                                         │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                                 Outputs                                    │
│  • Trade Recommendations (reports/trade_signals.md, trade_predictions.csv) │
│  • Model Evaluation Report (reports/evaluation_report.md)                  │
│  • 1-Page Macroeconomic Outlook (reports/macro_outlook.md)                 │
│  • Interactive Visualizations (Plotly HTML + PNG charts)                   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd cwt-macro-agent
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

```ini
APIFY_TOKEN=your_apify_token_here
TAVILY_API_KEY=your_tavily_api_key_here
```

*(Both `APIFY_TOKEN` and `APIFY_API_TOKEN` are supported; token is read exclusively from environment variables).*

### 3. Run Trade Predictions (Joined Datasets)

Generate trade allocations by joining macroeconomic releases and news sentiment:

```bash
# Predict latest trades across all 5 asset classes
python predict_trades.py

# Predict for a specific asset class
python predict_trades.py --asset US_EQUITY

# Offline synthetic fallback mode
python predict_trades.py --synthetic
```

### 4. Run the Full End-to-End Pipeline

```bash
# Full pipeline (Scrape -> Features -> Walk-Forward Model -> Visuals -> Trades -> Report)
python main.py

# Offline demonstration mode (no external API calls)
python main.py --synthetic

# Dedicated sub-commands
python main.py --scrape-only     # Only ingest calendar and news data
python main.py --train-only      # Train model on existing data
python main.py --predict-only    # Generate trade signals from joined data
python main.py --report-only     # Compile markdown evaluation reports
```

---

## 🔌 Scraper & Actor Configurations

### Macroeconomic Calendar (Apify)

| Role | Actor ID | OpenAPI Specification |
| :--- | :--- | :--- |
| **Primary (Used)** | `xtracto/forexfactory-calendar` | [OpenAPI Spec](https://api.apify.com/v2/actors/xtracto~forexfactory-calendar/builds/default/openapi.json) |
| **Backup** | `bovi/forex-economic-calendar-scraper` | [OpenAPI Spec](https://api.apify.com/v2/actors/bovi~forex-economic-calendar-scraper/builds/default/openapi.json) |

The primary actor uses exact schema fields:
- `dateRange`: `"range"`
- `rangeFrom`: Start date (`YYYY-MM-DD`)
- `rangeTo`: End date (`YYYY-MM-DD`)
- `currencies`: `["USD"]`
- `minImpact`: `"low"`
- `upcomingOnly`: `False` (pulls released actuals, forecasts, revisions, and impact ratings)

### Economic News & Sentiment (Tavily)

- **Engine**: Tavily Search API with `topic="news"` and `topic="finance"` across 8 core macroeconomic themes.
- **NLP Sentiment**: `TextBlob` polarity scoring (`-1.0` to `+1.0`) and subjectivity (`0.0` to `1.0`) with domain financial lexicon fallback.
- **Optional Apify News Actors**:
  - `gazidev/google-news-scraper`: [OpenAPI Spec](https://api.apify.com/v2/actors/gazidev~google-news-scraper/builds/default/openapi.json)
  - `nourishing_courier/google-news-scraper`: [OpenAPI Spec](https://api.apify.com/v2/actors/nourishing_courier~google-news-scraper/builds/default/openapi.json)

---

## 📈 Multi-Asset Strategy Matrix

Evaluated under strict temporal walk-forward validation (zero future data leakage):

| Asset Class | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate | Profit Factor | Cumulative Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **US_EQUITY** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | +2.30% |
| **US_BOND** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | +2.30% |
| **GOLD** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | +2.30% |
| **USD_INDEX** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | +2.30% |
| **CRUDE_OIL** | 2.92 | 7.09 | 1.02% | 55.17% | 1.54 | +2.30% |

---

## 📁 Repository Structure

```
cwt-macro-agent/
├── predict_trades.py                 # Predicts trades by joining macro releases & news
├── main.py                           # Full pipeline orchestrator (6 phases)
├── config.py                         # Central configuration & actor settings
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variables template
├── data/
│   ├── raw/                          # Raw scraped JSON archives (calendar & news)
│   ├── processed/                    # Feature matrices and trade predictions (CSV)
│   └── cwt_macro.db                  # SQLite database
├── notebooks/
│   └── exploration.ipynb             # Interactive Jupyter exploration notebook
├── src/
│   ├── scraper/
│   │   ├── apify_calendar.py         # ForexFactory calendar scraper (primary + backup)
│   │   └── tavily_news.py            # Historical news & sentiment scraper
│   ├── database/
│   │   └── db_manager.py             # SQLAlchemy ORM and calendar date joiner
│   ├── features/
│   │   └── feature_engineering.py    # Surprises, rolling stats, sentiment momentum
│   ├── model/
│   │   └── walk_forward.py           # LightGBM walk-forward validation
│   ├── visualization/
│   │   └── charts.py                 # Plotly HTML and publication PNG charts
│   └── summary/
│       └── macro_outlook.py          # 1-page macro outlook generator
└── reports/
    ├── trade_signals.md              # Executive trade recommendations
    ├── trade_predictions.json        # Machine-readable trade forecasts
    ├── evaluation_report.md          # Model performance and walk-forward report
    ├── macro_outlook.md              # 1-page macro environment summary
    ├── model_results.json            # Backtest metrics and fold statistics
    └── charts/                       # Generated Plotly & Matplotlib figures
```

---

## 🛠️ Tech Stack

- **Core**: Python 3.11+
- **Data & Storage**: Pandas, NumPy, SQLAlchemy, SQLite
- **Machine Learning**: LightGBM, XGBoost, Scikit-learn
- **NLP Sentiment**: TextBlob, NLTK
- **Visualization**: Plotly, Kaleido, Matplotlib, Seaborn
- **APIs**: Apify Client v3.x, Tavily Python SDK

---

## 📝 License & Assessment

Created for the **CrowdWisdomTrading Quantitative Data Scientist Intern** assessment.  
**Candidate**: Rohit Saini  
**Submission Contact**: `gilad@crowdwisdomtrading.com`
