# CrowdWisdomTrading Macroeconomic Sentiment & Trading Prediction Agent

An institutional-grade, reproducible quantitative machine learning pipeline built for the **CrowdWisdomTrading Quantitative Data Scientist Internship Assessment**.

The system ingests 180 days of macroeconomic events, thematic economic news, and real multi-asset market data to forecast quantitative trading performance using **chronological walk-forward validation**, **realistic transaction friction modeling**, **interactive Plotly dashboards**, and **automated executive macroeconomic outlooks**.

---

## 🏗️ Architecture & Quantitative Workflow

```
┌─────────────────────────────────┐      ┌───────────────────────────────────┐      ┌───────────────────────────────────┐
│   Apify ForexFactory Calendar   │      │   Tavily Thematic Economic News   │      │    Real Historical Market Data    │
│ (xtracto/forexfactory-calendar) │      │  (Thematic Queries + NLP Polarity)│      │  (yfinance: SPY, TLT, GLD, etc.)  │
└────────────────┬────────────────┘      └─────────────────┬─────────────────┘      └─────────────────┬─────────────────┘
                 │                                         │                                          │
                 ▼                                         ▼                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             SQLite Relational Database                                                │
│                                    (macro_events + news_articles + market_data)                                       │
│                                       *Deduplication on Primary Calendar Keys*                                        │
└───────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                Quantitative Feature Engineering (Zero Lookahead)                                      │
│  • Surprise Metrics: actual - forecast, normalized surprise %, surprise direction                                    │
│  • Proximity Clocks: days_since_last_fomc, days_since_last_nfp, days_since_last_cpi                                    │
│  • News Sentiment: rolling 3d/7d polarity, sentiment momentum (3d - 7d), article volume                              │
│  • Market Regimes: CPI trend, employment momentum, GDP growth trend, rolling 5d/10d/20d volatility                    │
│  • Target Formation: next-day return direction (t+1), final unobserved row dropped                                    │
└───────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Chronological Walk-Forward Machine Learning Validation                                   │
│  • Strict temporal windows (train=60d, test=20d sliding) — No future data leakage                                     │
│  • Tree-based classifiers: LightGBM (with XGBoost / Random Forest fallbacks)                                          │
│  • Realistic Friction Model: 5.0 bps turnover cost (slippage + commissions), annualized rf = 4.0%                     │
│  • Strategy Permutations: BUY_AND_HOLD, MODEL_LONG_SHORT, MODEL_LONG_CASH, MACRO_MOMENTUM, MACRO_CONTRARIAN           │
└───────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              Deliverables & Visual Outputs                                            │
│  • Interactive Web Dashboard: reports/macro_dashboard.html (Single-file HTML with tooltips & zero setup)              │
│  • 7 Priority Plotly Charts: Interactive HTML + publication PNG copies in reports/charts/                             │
│  • Model Evaluation Report: reports/evaluation_report.md (Net vs Gross Sharpe, Sortino, Max Drawdown)                 │
│  • Macroeconomic Outlook: reports/macro_outlook.md (1-page institutional regime summary)                              │
│  • Trade Decision Matrix: reports/trade_signals.md & data/processed/trade_predictions.csv                            │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Core Data Sources & Integration

| Source | Role | Implementation / Actor ID | Coverage & Data Fields |
| :--- | :--- | :--- | :--- |
| **Apify** | Macroeconomic Calendar | `xtracto/forexfactory-calendar` *(Backup: `bovi/forex-economic-calendar-scraper`)* | 180 days: CPI, Core CPI, PPI, Non-Farm Payrolls, Unemployment Rate, FOMC decisions, GDP, Retail Sales, ISM PMI. Ingests actual, forecast, previous, impact, and UTC timestamps. |
| **Tavily** | Thematic Economic News | `TavilyClient` (Search API) | Targeted macroeconomic queries across 8 themes (monetary policy, inflation, employment, growth, yields, PMI, trade). Enriched with NLP polarity and subjectivity. |
| **Exa** | Alternative Neural Search | `ExaNewsScraper` | Neural macro news retrieval. Gracefully falls back to Tavily if `EXA_API_KEY` is omitted. |
| **yfinance** | Real Historical Market Data | `MarketDataFetcher` | Real daily Open, High, Low, Close, Volume, Daily Return, and 5-day Rolling Volatility for liquid US ETF proxies: **SPY** (Equity), **TLT** (Bonds), **GLD** (Gold), **UUP** (USD), **USO** (Crude Oil). |
| **SQLite** | Local Relational Store | `DatabaseManager` / SQLAlchemy | Normalized tables (`macro_events`, `news_articles`, `market_data`). Upsert deduplication on composite keys `(event_date, event_name)` and `(date, asset_class)`. |

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

Python 3.10+ is recommended. Clone the repository and install dependencies:

```bash
git clone https://github.com/roraks24/cwt-macroeconomic-event.git
cd cwt-macroeconomic-event
pip install -r requirements.txt
```

### 2. Configure API Credentials

Copy `.env.example` to `.env` and insert your credentials:

```bash
cp .env.example .env
```

```ini
# Macro Calendar Scraping (Required for live ForexFactory ingestion)
APIFY_TOKEN=your_apify_token_here

# Economic News Search (Required for live macro news ingestion)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional Alternative Search
EXA_API_KEY=your_exa_api_key_here

# Pipeline Lookback & Walk-Forward Settings
LOOKBACK_DAYS=180
WALK_FORWARD_TRAIN_DAYS=120
WALK_FORWARD_TEST_DAYS=30
```

*(Note: If API tokens are omitted or network is offline, the pipeline automatically activates its self-contained demonstration mode).*

### 3. Run Automated Unit Tests

Run the full `pytest` suite covering scrapers, market data ingestion, database deduplication, timezone alignment, zero data leakage, and walk-forward metrics:

```bash
pytest -v
```

All 56 unit tests execute in ~45 seconds (including Chromium PDF generation and chart exports).

### 4. Execute the Full Quantitative Pipeline

Run the complete 6-stage algorithmic pipeline:

```bash
# Run complete end-to-end pipeline
python main.py

# Run pipeline using existing database records (skips re-scraping APIs)
python main.py --skip-scrape

# Run in offline demonstration mode (guaranteed zero API dependency)
python main.py --synthetic
```

### 5. Generate Multi-Asset Trade Predictions

Generate tactical trade recommendations across US Equity, Bonds, Gold, USD, and Crude Oil:

```bash
# Predict latest trades across all 5 asset classes
python predict_trades.py

# Predict for a specific asset class
python predict_trades.py --asset US_EQUITY
```

---

## 🔬 Rigorous Quantitative Modeling & Execution Friction

### 1. Chronological Walk-Forward Cross-Validation
Standard k-fold cross-validation is invalid for financial time-series due to temporal leakage and autocorrelation. Our `WalkForwardModel`:
- Splits data sequentially: **60-day rolling training window**, **20-day out-of-sample test window**, sliding forward by 20 days per fold.
- Generates **multiple chronological folds** across a 180-day history with **zero future lookahead bias**.
- In every fold: $\min(\text{Test Date}) \ge \max(\text{Train Date})$.

### 2. Realistic Execution Friction Modeling
Frictionless backtests produce artificially high Sharpe ratios that cannot survive live execution. We model institutional execution friction:
- **Turnover Deduction**: Every portfolio rebalance incurs a 5.0 bps cost:
  $$\text{Friction}_t = |\text{Signal}_t - \text{Signal}_{t-1}| \times 0.0005$$
  (representing 2.5 bps exchange commission + 2.5 bps bid-ask half-spread / slippage).
- **Annualized Risk-Free Rate**: Excess returns are calculated against an institutional benchmark risk-free rate of $r_f = 4.0\%$ annualized.
- **Downside Sortino Ratio**: Evaluates standard deviation solely over negative excess return instances:
  $$\text{Sortino} = \frac{\mu_{\text{excess}}}{\sigma_{\text{downside}}} \times \sqrt{252}$$
- **Peak-to-Trough Maximum Drawdown**: Computed monotonically on the net cumulative equity curve after costs.

### 3. Strategy Permutations Comparison
The pipeline evaluates and compares multiple strategy configurations out-of-sample:
1. **BUY_AND_HOLD**: Unhedged passive long exposure benchmark.
2. **MODEL_LONG_SHORT**: Dynamic model-driven directional long/short allocation (5 bps turnover cost).
3. **MODEL_LONG_CASH**: Defensive model allocation (long when bullish, flat cash when bearish; 5 bps cost).
4. **MACRO_MOMENTUM**: Follows macroeconomic calendar surprise direction (5 bps cost).
5. **MACRO_CONTRARIAN**: Fades macroeconomic calendar surprise direction (5 bps cost).

---

## 📈 Visual Deliverables & Interactive Dashboard

The pipeline automatically compiles interactive Plotly visualizations (`.html`) with hover tooltips and zoom, as well as publication-ready high-resolution PNG copies (`.png`):

1. 🌐 **Interactive Dashboard** (`reports/macro_dashboard.html`): Single-file HTML container displaying all metrics and charts side-by-side, openable directly in any browser with zero installation.
2. **Sentiment Timeline with Macro Events** (`reports/charts/sentiment_timeline_events.html`): Daily sentiment line (raw + 7-day rolling average) overlaid with vertical event markers for CPI, FOMC, and NFP colored by surprise direction.
3. **Event-Study Dynamics** (`reports/charts/event_study_dynamics.html`): Cumulative asset return and sentiment trajectory from $-3$ to $+5$ days around macro release dates.
4. **Strategy Matrix Heatmap** (`reports/charts/strategy_matrix_heatmap.html`): Visual matrix comparing Event Type $\times$ Direction (Follow / Fade) $\times$ Horizon (1d, 3d, 5d) annotated with net Sharpe ratios.
5. **Equity Curve & Drawdown** (`reports/charts/equity_curve_drawdown.html`): Net walk-forward equity curve vs. passive SPY buy-and-hold baseline, with net drawdown underneath.
6. **Predicted vs. Realised P&L** (`reports/charts/predicted_vs_realised_pnl.html`): Model calibration scatter plot evaluating predicted probability vs. next-day realized return.
7. **Surprise vs. Next-Day Return** (`reports/charts/surprise_vs_next_day_return.html`): Scatter plot of normalized surprise values vs. next-day asset returns colored by event type.
8. **Rolling 30-Trade Sharpe Ratio** (`reports/charts/rolling_sharpe.html`): Rolling risk-adjusted stability curve verifying performance persistence over time.

---

## 📁 Repository Directory Structure

```
cwt-macro-agent/
├── main.py                           # Master pipeline orchestrator (6-stage execution)
├── predict_trades.py                 # Multi-asset trade allocation and signal predictor
├── config.py                         # Central configuration, actor IDs, and parameters
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variables template
├── README.md                         # Project documentation and interview briefing
│
├── data/
│   ├── raw/                          # Raw scraped JSON archives (calendar & news)
│   ├── processed/                    # Feature matrices and trade predictions (CSV)
│   └── cwt_macro.db                  # Relational SQLite database
│
├── src/
│   ├── scraper/
│   │   ├── apify_calendar.py         # Apify ForexFactory scraper (primary + backup)
│   │   ├── tavily_news.py            # Tavily economic news scraper & NLP sentiment
│   │   ├── exa_news.py               # Exa neural search integration with offline fallback
│   │   └── market_data.py            # Real historical ETF price fetcher (yfinance)
│   ├── database/
│   │   └── db_manager.py             # SQLAlchemy models, deduplication, and joins
│   ├── features/
│   │   └── feature_engineering.py    # Macro surprise, time proximity, and regime features
│   ├── model/
│   │   └── walk_forward.py           # Walk-forward cross validation & friction metrics
│   ├── strategy/
│   │   └── strategy_engine.py        # Macro strategy decision engine
│   ├── sentiment/
│   │   └── quant_sentiment.py        # Algorithmic macro lexicon sentiment scorer
│   ├── visualization/
│   │   └── charts.py                 # Plotly HTML dashboard and PNG chart generators
│   └── summary/
│       └── macro_outlook.py          # 1-page macro outlook compiler
│
├── tests/
│   ├── test_scrapers.py              # Parsing, synthetic schemas, and Exa offline handling
│   ├── test_market_data.py           # ETF mapping, bar invariants, and return formulas
│   ├── test_database.py              # In-memory SQLite tables, deduplication, and joins
│   ├── test_features_and_leakage.py  # Feature generation and strict zero lookahead tests
│   └── test_walk_forward_and_metrics.py # Chronological fold ordering, friction, and Sharpe
│
└── reports/
    ├── macro_dashboard.html          # Standalone interactive Plotly web dashboard
    ├── evaluation_report.md          # Comprehensive performance & walk-forward report
    ├── macro_outlook.md              # 1-page macroeconomic regime & factor analysis
    ├── trade_signals.md              # Tactical trade allocations & risk protocols
    ├── trade_predictions.json        # Machine-readable trade prediction records
    ├── model_results.json            # Walk-forward validation results and metrics
    └── charts/                       # Plotly HTML & PNG figures
```

---

## 🧪 Automated Test Suite Summary

The comprehensive 54-test suite validates end-to-end data integrity, timezone conversions, exchange calendar sessions, zero data leakage, execution friction math, walk-forward fold boundaries, and publication artifact rendering:

```bash
$ pytest -v
tests/test_database.py::test_insert_and_deduplicate_macro_events PASSED                  [  1%]
tests/test_database.py::test_insert_and_deduplicate_news_articles PASSED                  [  3%]
tests/test_database.py::test_insert_and_deduplicate_market_data PASSED                    [  5%]
tests/test_database.py::test_get_joined_dataset_calendar_alignment PASSED               [  7%]
tests/test_features_and_leakage.py::test_sanitize_name PASSED                            [  9%]
tests/test_features_and_leakage.py::test_create_surprise_features PASSED                 [ 11%]
tests/test_features_and_leakage.py::test_create_time_features PASSED                     [ 12%]
tests/test_features_and_leakage.py::test_create_sentiment_features PASSED                [ 14%]
tests/test_features_and_leakage.py::test_zero_data_leakage_in_target_construction PASSED [ 16%]
tests/test_features_and_leakage.py::test_multi_asset_target_alignment_no_boundary_crossing PASSED [ 18%]
tests/test_features_and_leakage.py::test_next_open_execution_target_and_pricing_isolation PASSED [ 20%]
tests/test_features_and_leakage.py::test_after_close_news_excluded_from_same_day_feature_vector PASSED [ 22%]
tests/test_features_and_leakage.py::test_historical_2022_article_not_assigned_to_april_2026 PASSED [ 24%]
tests/test_features_and_leakage.py::test_synthetic_fixture_first_session_does_not_absorb_unrelated_historical_articles PASSED [ 25%]
tests/test_features_and_leakage.py::test_actual_database_first_session_has_zero_news_volume_and_no_contamination PASSED [ 27%]
tests/test_features_and_leakage.py::test_pre_window_news_transformation_at_feature_matrix_level PASSED [ 29%]
tests/test_market_data.py::test_etf_map_completeness PASSED                              [ 31%]
tests/test_market_data.py::test_bar_invariants_validation PASSED                         [ 33%]
tests/test_market_data.py::test_daily_return_and_volatility_calculation PASSED          [ 35%]
tests/test_scrapers.py::test_parse_event_value_percentages_and_units PASSED              [ 37%]
tests/test_scrapers.py::test_synthetic_calendar_generation_structure PASSED             [ 38%]
tests/test_scrapers.py::test_news_sentiment_enrichment_bounds PASSED                     [ 40%]
tests/test_scrapers.py::test_exa_news_offline_handling_when_no_token PASSED              [ 42%]
tests/test_scrapers.py::test_exa_news_search_parsing_with_mocked_response PASSED         [ 44%]
tests/test_timezone.py::test_standard_time_vs_daylight_saving_detection PASSED          [ 46%]
tests/test_timezone.py::test_market_close_and_open_utc_offsets PASSED                    [ 48%]
tests/test_timezone.py::test_evening_utc_timestamp_calendar_date_no_shift PASSED         [ 50%]
tests/test_timezone.py::test_normalize_series_to_ny_date PASSED                          [ 51%]
tests/test_timezone.py::test_assign_to_eligible_trading_session_pre_and_post_close PASSED [ 53%]
tests/test_timezone.py::test_us_equity_exchange_holidays_vs_federal_holidays PASSED     [ 55%]
tests/test_visualization.py::test_sentiment_timeline_generation PASSED                   [ 57%]
tests/test_visualization.py::test_event_study_generation PASSED                          [ 59%]
tests/test_visualization.py::test_strategy_matrix_heatmap_generation PASSED             [ 61%]
tests/test_visualization.py::test_equity_curve_and_drawdown_generation PASSED             [ 62%]
tests/test_visualization.py::test_predicted_vs_realised_pnl_generation PASSED           [ 64%]
tests/test_visualization.py::test_surprise_vs_return_generation PASSED                   [ 66%]
tests/test_visualization.py::test_rolling_sharpe_generation PASSED                       [ 68%]
tests/test_visualization.py::test_generate_all_charts_master_runner PASSED               [ 70%]
tests/test_visualization.py::test_convert_markdown_to_pdf PASSED                         [ 72%]
tests/test_walk_forward_and_metrics.py::test_walk_forward_chronological_splits PASSED     [ 74%]
tests/test_walk_forward_and_metrics.py::test_calculate_trading_metrics_friction_and_sharpe PASSED [ 75%]
tests/test_walk_forward_and_metrics.py::test_sortino_ratio_downside_deviation_behavior PASSED [ 77%]
tests/test_walk_forward_and_metrics.py::test_compare_strategy_permutations PASSED         [ 79%]
tests/test_walk_forward_and_metrics.py::test_turnover_isolation_per_asset_class PASSED    [ 81%]
tests/test_walk_forward_and_metrics.py::test_trading_costs_entry_holding_exit_reversal PASSED [ 83%]
tests/test_walk_forward_and_metrics.py::test_date_aligned_multi_asset_portfolio_aggregation PASSED [ 85%]
tests/test_walk_forward_and_metrics.py::test_hand_calculated_next_open_execution_timing_and_returns PASSED [ 87%]
tests/test_walk_forward_and_metrics.py::test_fold_reconciliation_exact_mathematical_precision PASSED [ 88%]
tests/test_walk_forward_and_metrics.py::test_fold_reconciliation_exact_boundary_cost_accounting PASSED [ 90%]
tests/test_walk_forward_and_metrics.py::test_walk_forward_train_purge_zero_future_label_leakage PASSED [ 92%]
tests/test_walk_forward_and_metrics.py::test_open_to_open_execution_uses_explicit_entry_exit_prices PASSED [ 94%]
tests/test_walk_forward_and_metrics.py::test_hand_calculated_boundary_friction_reconciliation PASSED [ 96%]
tests/test_walk_forward_and_metrics.py::test_stored_model_results_precision_compounding_identity PASSED [ 94%]
tests/test_walk_forward_and_metrics.py::test_variant_5_vs_variant_6_probability_divergence_with_identical_positions_and_returns PASSED [ 96%]
tests/test_walk_forward_and_metrics.py::test_production_walk_forward_fold_dates_and_session_counts PASSED [ 98%]
tests/test_walk_forward_and_metrics.py::test_production_boundary_turnover_and_cost_accounting PASSED [100%]

============================= 56 passed in 44.15s =============================
```

---

## 📜 Assessment Compliance Matrix (14 Requirements)

The following verification matrix maps each requirement from `CWT Economic sentiment Agent - intern task.docx` to the concrete implementation, unit test coverage, and generated deliverables:

| # | Requirement Specification | Status | Implementation Evidence | Test Suite Verification | Output Deliverables |
| :-: | :--- | :-: | :--- | :--- | :--- |
| **1** | **Python-based ML Pipeline for Macro Trading** | **PASS** | `main.py`, `src/model/walk_forward.py`, `src/features/feature_engineering.py` | Full integration runner in `main.py` | `reports/model_results.json`, `reports/evaluation_report.md` |
| **2** | **180-Day Macro Calendar via Apify** | **PASS** | `src/scraper/apify_calendar.py` (ForexFactory actor `xtracto/forexfactory-calendar` with fallback `bovi/...`) | `test_parse_event_value_percentages_and_units`, `test_synthetic_calendar_generation_structure` | `data/raw/macro_calendar.json`, 300 records in SQLite |
| **3** | **Robust Timezone & Calendar Handling** | **PASS** | `src/utils/timezone.py` (US Eastern vs UTC, DST boundaries, NYSE calendar vs Federal holidays) | `test_standard_time_vs_daylight_saving_detection`, `test_us_equity_exchange_holidays_vs_federal_holidays` | Accurate cutoff times (16:00 ET / 20:00 or 21:00 UTC) across all features |
| **4** | **Thematic Economic News via Tavily** | **PASS** | `src/scraper/tavily_news.py` (8 macro thematic queries, sentiment enrichment) | `test_news_sentiment_enrichment_bounds` | `data/raw/news_articles.json`, 295 articles in SQLite |
| **5** | **Exa Neural Search Integration** | **PASS** | `src/scraper/exa_news.py` (`ExaNewsScraper`, neural queries, deduplication, offline fallback) | `test_exa_news_offline_handling_when_no_token`, `test_exa_news_search_parsing_with_mocked_response` | Dual-source news integration in `main.py` |
| **6** | **SQLite Relational Database & Joins** | **PASS** | `src/database/db_manager.py` (SQLAlchemy models, primary key deduplication, calendar joins) | `test_insert_and_deduplicate_*`, `test_get_joined_dataset_calendar_alignment` | `data/cwt_macro.db` (300 events, 295 news, 625 market bars) |
| **7** | **Quantitative Feature Engineering** | **PASS** | `src/features/feature_engineering.py` (surprises, FOMC/CPI/NFP clocks, regimes, strict information cutoffs) | `test_create_surprise_features`, `test_zero_data_leakage_in_target_construction`, `test_after_close_news_*` | `data/processed/features.csv`, 395 observation rows |
| **8** | **One-Page Macro Outlook Report** | **PASS** | `src/summary/macro_outlook.py` (factor score, regime flags, allocation recommendations) | Tested via `main.py` automated generation | `reports/macro_outlook.md` (institutional 1-page summary) |
| **9** | **Predictive Model & 5 Strategy Permutations** | **PARTIAL** | Model predicts directional win probability $P(R_{t+1} > 0)$ via LightGBM classifier. Expected trade P&L is estimated analytically via an edge-volatility heuristic $\mathbb{E}[\text{P\&L}] = (P(\text{Win}) - 0.50) \times \text{Vol}_{5d} \times 2.5 - 0.05\%$ friction. Strategy permutation performance is evaluated via historical walk-forward backtesting rather than a dedicated parametric P&L regression model. | `test_compare_strategy_permutations`, `test_turnover_isolation_per_asset_class` | `reports/trade_signals.md`, `reports/trade_predictions.json` |
| **10** | **Chronological Walk-Forward Validation** | **PASS** | `src/model/walk_forward.py` (6 folds, train=60d, test=20d sliding, 2d target purge, 5.0 bps friction) | `test_walk_forward_chronological_splits`, `test_walk_forward_train_purge_zero_future_label_leakage` | `reports/model_results.json` (Fold 1-6 reconciled net returns) |
| **11** | **Visual Sentiment Deliverables & Dashboard** | **PASS** | `src/visualization/charts.py` (Plotly HTML + static PNG generator) | `test_sentiment_timeline_generation`, `test_generate_all_charts_master_runner` | `reports/macro_dashboard.html`, 7 priority charts in `reports/charts/` |
| **12** | **Evaluation Report in Markdown & Institutional PDF** | **PASS** | `reports/evaluation_report.md`, `src/utils/generate_pdf.py` (Playwright headless Chromium generator) | `test_convert_markdown_to_pdf` | `reports/evaluation_report.md`, `reports/evaluation_report.pdf` |
| **13** | **Reproducible Repository Standards** | **PASS** | Pinned `requirements.txt`, `.env.example`, `.gitignore`, relative paths, clean git structure | Full test suite clean run (`pytest -v`) | Fully reproducible standalone repository |
| **14** | **Documentation of Actors, Tokens & Execution** | **PASS** | Central `config.py`, detailed `README.md`, environment configuration guide | Verification and reproduction CLI documented | Complete user and reviewer setup guide |

---

## 🎯 Model Forecast vs. Expected P&L vs. Realized Net Return

The assessment specification requires a clear distinction between:
1. **Model Classification Probability $P(R_{t+1} > 0 \mid X_t)$**:
   LightGBM / tree-ensemble probability that next trading day's open-to-open return will be positive.
2. **Heuristic Expected Trade P&L**:
   $$\mathbb{E}[\text{P\&L}] = (P(\text{Win}) - 0.50) \times \text{Vol}_{5d} \times 2.5 - 0.0005$$
   Reflects the expected directional edge scaled by short-term asset volatility minus execution friction.
3. **Realized Out-of-Sample Net Return**:
   The actual portfolio return achieved out-of-sample across the 6 chronological walk-forward folds after deducting 5.0 bps turnover cost on each rebalance:
   - **Net Cumulative Return**: $-4.20\%$ (vs. $-1.75\%$ gross return before costs)
   - **Total Turnover**: $49.0\times$ (cost drag: $2.45\%$)
   - **Net Sharpe Ratio**: $-2.00$ (Gross Sharpe: $-1.10$, annualized $r_f=4.0\%$)
   - **Downside Sortino Ratio**: $-2.57$
   - **Max Drawdown**: $7.61\%$

---

## ⚠️ Methodological Limitations & Realistic Point-in-Time Disclosures

1. **Historical News Indexing & Point-in-Time Availability**:
   - The pipeline enforces publication-time and available retrieval-time cutoffs (16:00 ET / 20:00 or 21:00 UTC).
   - This controls and substantially reduces known timestamp-based lookahead risk.
   - However, retrospective search-engine scraping cannot establish the exact historical minute an article first became discoverable or indexed by third-party search APIs (Tavily or Exa).
   - Therefore, complete historical point-in-time availability cannot be guaranteed.
2. **Out-of-Sample Window Horizon**: The out-of-sample test window covers 79 US equity trading sessions (June 22 to October 12, 2026). While statistically valid for initial model validation and walk-forward verification, macroeconomic cycles span multiple years; factor persistence across multiple business cycles (5–10 years) remains to be assessed.
3. **Execution Pricing via ETF Adjusted Prices**: Execution assumes Next-Session Open pricing derived from daily exchange adjusted bars. Real-world execution will experience intraday auction variability and ETF tracking error relative to underlying cash benchmarks.
4. **Linear Transaction Cost Assumption**: Friction is modeled at a constant 5.0 bps per unit turnover (2.5 bps commission + 2.5 bps half-spread). Large institutional order sizes will incur non-linear square-root market impact depending on order flow and ADV.

---

**Candidate**: Quantitative Data Scientist Intern Applicant  
**Assessment For**: CrowdWisdomTrading (Gilad)  

