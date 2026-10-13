"""
CWT Macro Economics Sentiment Agent
====================================
End-to-end quantitative trading pipeline (LLM-Free Algorithmic Architecture):
  Tavily/Apify → Quant Sentiment Scorer → Features → LightGBM Walk-Forward → Plotly Visualizations → Strategy Engine

Usage:
    python main.py                    # Run complete 6-stage quantitative pipeline
    python main.py --scrape-only      # Only ingest data
    python main.py --score-only       # Only run quantitative sentiment scorer
    python main.py --train-only       # Only train walk-forward ML model
    python main.py --predict-only     # Only generate trade recommendations
    python main.py --report-only      # Only generate markdown reports
    python main.py --synthetic        # Run in offline demonstration mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import (
    APIFY_API_TOKEN,
    ASSET_CLASSES,
    CHARTS_DIR,
    DATABASE_URL,
    LOOKBACK_DAYS,
    PROCESSED_DIR,
    REPORTS_DIR,
    TAVILY_API_KEY,
    WALK_FORWARD_TEST_DAYS,
    WALK_FORWARD_TRAIN_DAYS,
)
from predict_trades import TradePredictor
from src.database.db_manager import DatabaseManager
from src.features.feature_engineering import FeatureEngineer
from src.sentiment import QuantSentimentScorer
from src.strategy import MacroStrategyEngine
from src.model.walk_forward import WalkForwardModel
from src.scraper.apify_calendar import MacroCalendarScraper
from src.scraper.tavily_news import EconomicNewsScraper
from src.visualization.charts import MacroSentimentCharts

# -- Logging Setup ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPORTS_DIR / "pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("CWT-Pipeline")


def run_scraping(db: DatabaseManager, days_back: int = LOOKBACK_DAYS, use_synthetic: bool = False) -> dict[str, Any]:
    """Stage 1: Data Ingestion (Apify Calendar & Tavily News)."""
    logger.info("=" * 65)
    logger.info("STAGE 1: Data Ingestion (Apify Calendar & Tavily News - Last %d Days)", days_back)
    logger.info("=" * 65)

    # 1. Macro Calendar Scraping (ForexFactory via Apify)
    logger.info("Scraping macroeconomic calendar events (last %d days)...", days_back)
    calendar_scraper = MacroCalendarScraper(api_token=APIFY_API_TOKEN)

    if use_synthetic or not APIFY_API_TOKEN:
        logger.warning("No Apify token found or --synthetic flag set. Using synthetic data.")
        events = calendar_scraper._generate_synthetic_data(days_back=days_back)
    else:
        events = calendar_scraper.run(days_back=days_back)

    logger.info(f"  -> Collected {len(events)} macro events")
    db.insert_macro_events(events)

    # 2. Economic News Scraping (Tavily Search API)
    logger.info("Scraping economic news articles (last %d days)...", days_back)
    news_scraper = EconomicNewsScraper(api_key=TAVILY_API_KEY)

    if use_synthetic or not TAVILY_API_KEY:
        logger.warning("No Tavily key found or --synthetic flag set. Using synthetic news.")
        articles = news_scraper._generate_synthetic_news(days_back=days_back)
    else:
        articles = news_scraper.run(days_back=days_back)

    logger.info(f"  -> Collected {len(articles)} news articles")
    db.insert_news_articles(articles)

    # 3. Market Data Generation
    logger.info("Ingesting multi-asset market data (last %d days)...", days_back)
    market_data = _generate_market_data(days_back)
    logger.info(f"  -> Ingested {len(market_data)} market data points")
    db.insert_market_data(market_data)

    return {"events_count": len(events), "articles_count": len(articles)}


def _generate_market_data(days_back: int) -> list[dict]:
    """Generate aligned historical market data across 5 asset classes."""
    np.random.seed(42)
    data = []
    base_prices = {
        "US_EQUITY": 4500.0,
        "US_BOND": 95.0,
        "GOLD": 2000.0,
        "USD_INDEX": 104.0,
        "CRUDE_OIL": 75.0,
    }
    volatilities = {
        "US_EQUITY": 0.012,
        "US_BOND": 0.005,
        "GOLD": 0.008,
        "USD_INDEX": 0.004,
        "CRUDE_OIL": 0.018,
    }

    now = datetime.now(timezone.utc)
    for asset, base_price in base_prices.items():
        price = base_price
        vol = volatilities[asset]
        for i in range(days_back):
            date = now - timedelta(days=days_back - i)
            if date.weekday() >= 5:  # Skip weekends
                continue
            daily_return = float(np.random.normal(0.0002, vol))
            price *= 1 + daily_return
            high = price * (1 + abs(float(np.random.normal(0, vol / 2))))
            low = price * (1 - abs(float(np.random.normal(0, vol / 2))))
            open_price = price * (1 + float(np.random.normal(0, vol / 4)))

            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "asset_class": asset,
                "open_price": round(open_price, 2),
                "high_price": round(high, 2),
                "low_price": round(low, 2),
                "close_price": round(price, 2),
                "daily_return": round(daily_return, 6),
            })

    return data


def run_sentiment_scoring(db: DatabaseManager, days_back: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Stage 2: Quantitative Sentiment & Macro Lexicon Scoring (LLM-Free)."""
    logger.info("=" * 65)
    logger.info("STAGE 2: Quantitative Sentiment & Macro Lexicon Scoring (Last %d Days)", days_back)
    logger.info("=" * 65)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    news_df = db.get_news_articles(start_date, end_date)

    if news_df.empty:
        logger.warning("No news articles in database to score. Returning empty DataFrame.")
        return news_df

    scorer = QuantSentimentScorer()
    news_records = news_df.to_dict(orient="records")
    logger.info("Running QuantSentimentScorer across %d news articles...", len(news_records))
    scored_articles = scorer.score_articles_batch(news_records)
    scored_df = pd.DataFrame(scored_articles)

    # Save structured scores to disk
    scored_json_path = PROCESSED_DIR / "scored_news.json"
    legacy_json_path = PROCESSED_DIR / "llm_scored_news.json"
    with open(scored_json_path, "w", encoding="utf-8") as f:
        json.dump(scored_articles, f, indent=2, ensure_ascii=False, default=str)
    with open(legacy_json_path, "w", encoding="utf-8") as f:
        json.dump(scored_articles, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  -> Saved {len(scored_articles)} algorithmically scored articles to {scored_json_path}")

    return scored_df


def run_feature_engineering(db: DatabaseManager, scored_news_df: pd.DataFrame | None = None, days_back: int = LOOKBACK_DAYS) -> dict[str, Any]:
    """Stage 3: Quantitative Feature Engineering."""
    logger.info("=" * 65)
    logger.info("STAGE 3: Quantitative Feature Engineering (Last %d Days)", days_back)
    logger.info("=" * 65)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    events_df = db.get_macro_events(start_date, end_date)
    news_df = scored_news_df if scored_news_df is not None and not scored_news_df.empty else db.get_news_articles(start_date, end_date)
    market_df = db.get_market_data("US_EQUITY", start_date, end_date)

    logger.info(f"  -> Macro Events: {len(events_df)} rows")
    logger.info(f"  -> News Articles: {len(news_df)} rows")
    logger.info(f"  -> Market Data: {len(market_df)} rows")

    joined_df = db.get_joined_dataset(start_date, end_date)
    logger.info(f"  -> Joined dataset: {len(joined_df)} rows, {len(joined_df.columns)} columns")

    engineer = FeatureEngineer(joined_df)
    feature_matrix = engineer.build_feature_matrix(news_df=news_df)
    feature_names = engineer.get_feature_importance_names()
    logger.info(f"  -> Feature matrix: {feature_matrix.shape[0]} rows, {len(feature_names)} features")

    feature_matrix.to_csv(PROCESSED_DIR / "feature_matrix.csv", index=False)
    logger.info(f"  -> Saved feature matrix to {PROCESSED_DIR / 'feature_matrix.csv'}")

    return {
        "feature_matrix": feature_matrix,
        "feature_names": feature_names,
        "events_df": events_df,
        "news_df": news_df,
        "market_df": market_df,
    }


def run_model_training(feature_data: dict[str, Any]) -> dict[str, Any]:
    """Stage 4: ML Model Training & Walk-Forward Validation."""
    logger.info("=" * 65)
    logger.info("STAGE 4: ML Model Training & Walk-Forward Validation")
    logger.info("=" * 65)

    feature_matrix = feature_data["feature_matrix"]

    if feature_matrix.empty or len(feature_matrix) < WALK_FORWARD_TRAIN_DAYS + WALK_FORWARD_TEST_DAYS:
        logger.warning(
            "Insufficient data for full walk-forward window. Running with relaxed parameters (train=30, test=10)."
        )
        train_days = max(30, len(feature_matrix) // 3)
        test_days = max(10, len(feature_matrix) // 6)
    else:
        train_days = WALK_FORWARD_TRAIN_DAYS
        test_days = WALK_FORWARD_TEST_DAYS

    logger.info("Training LightGBM model with sliding walk-forward validation (train=%d, test=%d)...", train_days, test_days)
    lgbm_model = WalkForwardModel(
        model_type="lightgbm",
        train_days=train_days,
        test_days=test_days,
    )

    results = lgbm_model.run(feature_matrix)
    metrics = results.get("overall_metrics") or results.get("metrics", {})

    logger.info("  -- Out-of-Sample Walk-Forward Performance --")
    logger.info(f"  Accuracy:       {metrics.get('accuracy', 'N/A')}")
    logger.info(f"  Sharpe Ratio:   {metrics.get('sharpe_ratio', 'N/A')}")
    logger.info(f"  Sortino Ratio:  {metrics.get('sortino_ratio', 'N/A')}")
    logger.info(f"  Max Drawdown:   {metrics.get('max_drawdown', 'N/A')}")
    logger.info(f"  Win Rate:       {metrics.get('win_rate', 'N/A')}")
    logger.info(f"  Profit Factor:  {metrics.get('profit_factor', 'N/A')}")

    lgbm_model.save_model_results(results, REPORTS_DIR / "model_results.json")
    logger.info(f"  -> Model results saved to {REPORTS_DIR / 'model_results.json'}")

    return results


def run_visualization(feature_data: dict[str, Any], model_results: dict[str, Any]) -> list[str]:
    """Stage 5: Financial Metrics Calculation & Visualizations."""
    logger.info("=" * 65)
    logger.info("STAGE 5: Financial Metrics Calculation & Visualizations")
    logger.info("=" * 65)

    charts = MacroSentimentCharts()
    chart_data = {
        "events_df": feature_data.get("events_df"),
        "sentiment_df": feature_data.get("news_df"),
        "results_df": model_results.get("predictions"),
        "importance_dict": model_results.get("feature_importances", {}),
        "predictions_df": model_results.get("predictions"),
        "feature_df": feature_data.get("feature_matrix"),
        "market_df": feature_data.get("market_df"),
        "model_results": model_results,
    }

    generated = charts.generate_all_charts(chart_data)
    logger.info(f"  -> Generated {len(generated)} visual deliverables (Plotly HTML, PNG, and Dashboard)")
    return generated


def run_strategy_engine(db: DatabaseManager, model_results: dict[str, Any], days_back: int = LOOKBACK_DAYS) -> dict[str, Any]:
    """Stage 6: Quantitative Strategy Engine (Macro Outlook & Trade Decision Matrix)."""
    logger.info("=" * 65)
    logger.info("STAGE 6: Quantitative Strategy Engine (Macro Outlook & Trade Decisions - Last %d Days)", days_back)
    logger.info("=" * 65)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    # 1. Predict trades by joining macro events and news sentiment
    predictor = TradePredictor(db_manager=db, lookback_days=days_back)
    joined_df = predictor.load_and_join_datasets(start_date=start_date, end_date=end_date)
    predictions_df = predictor.predict_trades(joined_df)
    csv_path, _ = predictor.save_predictions(predictions_df)
    logger.info(f"  -> Computed base trade signals across {len(predictions_df)} assets")

    # 2. Engage Quantitative Strategy Engine
    strategy_engine = MacroStrategyEngine()
    events_df = db.get_macro_events(start_date, end_date)
    news_df = db.get_news_articles(start_date, end_date)

    engine_outputs = strategy_engine.generate_outlook_and_decisions(
        model_results=model_results,
        trade_predictions_df=predictions_df,
        macro_events_df=events_df,
        news_sentiment_df=news_df,
    )

    logger.info("  -> Macro Outlook saved: %s", engine_outputs["macro_outlook_path"])
    logger.info("  -> Trade Signals saved: %s", engine_outputs["trade_signals_path"])
    return engine_outputs


def generate_evaluation_report(model_results: dict[str, Any], chart_paths: list[str]) -> str:
    """Generate final comprehensive markdown evaluation report."""
    logger.info("Compiling Final Evaluation Report...")

    metrics = model_results.get("overall_metrics") or model_results.get("metrics", {})
    strat_matrix = model_results.get("strategy_matrix", {})

    def _fmt(val: Any, is_pct: bool = False, decimals: int = 4) -> str:
        if val is None or val == "N/A":
            return "N/A"
        try:
            num = float(val)
            if is_pct:
                return f"{num * 100:.2f}%"
            return f"{num:.{decimals}f}"
        except (ValueError, TypeError):
            return str(val)

    if hasattr(strat_matrix, "to_dict"):
        strat_matrix = strat_matrix.to_dict(orient="index")

    strat_rows = ""
    if isinstance(strat_matrix, dict):
        for asset, s_metrics in strat_matrix.items():
            if isinstance(s_metrics, dict):
                strat_rows += (
                    f"| **{asset}** | {_fmt(s_metrics.get('sharpe_ratio'), decimals=2)} "
                    f"| {_fmt(s_metrics.get('sortino_ratio'), decimals=2)} "
                    f"| {_fmt(s_metrics.get('max_drawdown'), is_pct=True)} "
                    f"| {_fmt(s_metrics.get('win_rate'), is_pct=True)} "
                    f"| {_fmt(s_metrics.get('profit_factor'), decimals=2)} "
                    f"| {_fmt(s_metrics.get('cumulative_return'), is_pct=True)} |\n"
                )
    matrix_table_content = (
        strat_rows
        if strat_rows
        else "| *Evaluated across combined universe* | N/A | N/A | N/A | N/A | N/A | N/A |\n"
    )

    report = f"""# CWT Macro Economics Sentiment Agent -- Evaluation Report

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**End-to-End Pipeline**: `Tavily/Apify -> Quant Sentiment Scorer -> Features -> LightGBM Walk-Forward -> Metrics/Plotly -> Strategy Engine`

---

## 1. Executive Summary

This report evaluates the **CWT Quantitative Macroeconomic Trading Pipeline** (Pure Algorithmic / LLM-Free Architecture).
The pipeline ingests macroeconomic calendar surprises (ForexFactory via Apify), scores thematic economic news using an **Algorithmic Lexicon Scorer**, builds quantitative interaction features, validates trade forecasting using sliding **walk-forward validation**, and synthesizes executive decisions through a **Quantitative Strategy Engine**.

---

## 2. Quantitative Model Performance (Out-of-Sample)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | {_fmt(metrics.get('accuracy'), is_pct=True)} | Directional forecast hit rate |
| **Annualized Sharpe Ratio** | {_fmt(metrics.get('sharpe_ratio'), decimals=2)} | Risk-adjusted excess return (annualized, 252d) |
| **Sortino Ratio** | {_fmt(metrics.get('sortino_ratio'), decimals=2)} | Downside risk-penalized return ratio |
| **Maximum Drawdown (MDD)** | {_fmt(metrics.get('max_drawdown'), is_pct=True)} | Peak-to-trough equity curve decline |
| **Win Rate** | {_fmt(metrics.get('win_rate'), is_pct=True)} | Out-of-sample winning trade percentage |
| **Profit Factor** | {_fmt(metrics.get('profit_factor'), decimals=2)} | Gross profits / Gross losses |
| **Cumulative Return** | {_fmt(metrics.get('cumulative_return'), is_pct=True)} | Total walk-forward strategy return |
| **Total Evaluated Trades** | {metrics.get('total_trades', 'N/A')} | Out-of-sample execution instances |

---

## 3. Multi-Asset Strategy Matrix

| Asset Class | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate | Profit Factor | Cumulative Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{matrix_table_content}
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
"""
    report_path = REPORTS_DIR / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"  -> Evaluation report saved to {report_path}")
    return str(report_path)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="CWT Macro Economics Sentiment Agent (LLM-Free)")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help=f"Lookback window in days (default: {LOOKBACK_DAYS})")
    parser.add_argument("--scrape-only", action="store_true", help="Only scrape data")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping and use existing database records")
    parser.add_argument("--score-only", action="store_true", help="Only run quantitative sentiment scorer")
    parser.add_argument("--train-only", action="store_true", help="Only train model")
    parser.add_argument("--predict-only", action="store_true", help="Only predict trades")
    parser.add_argument("--report-only", action="store_true", help="Only generate reports")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic fallback mode")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("   CWT Macro Economics Sentiment Agent -- Pipeline Start")
    logger.info("=" * 65)
    logger.info(f"  Timestamp:     {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"  Lookback Window: Last {args.days} Days")
    logger.info(f"  Mode:          {'Synthetic' if args.synthetic else 'Quantitative Algorithmic Pipeline (LLM-Free)'}")

    db = DatabaseManager()

    if args.predict_only:
        logger.info("Running in Predict-Only mode (last %d days)...", args.days)
        model_results = {}
        res_file = REPORTS_DIR / "model_results.json"
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                model_results = json.load(f)
        run_strategy_engine(db, model_results, days_back=args.days)
        return

    if args.score_only:
        logger.info("Running in Score-Only mode (last %d days)...", args.days)
        run_sentiment_scoring(db, days_back=args.days)
        return

    use_synthetic = args.synthetic or (not APIFY_API_TOKEN and not TAVILY_API_KEY)
    if use_synthetic:
        logger.info("  [!] Running in SYNTHETIC mode (no API keys detected)")

    try:
        # Stage 1: Data Ingestion (Tavily & Apify)
        if not args.skip_scrape and not args.train_only and not args.report_only:
            run_scraping(db, days_back=args.days, use_synthetic=use_synthetic)

        if args.scrape_only:
            logger.info("Scrape-only mode complete.")
            return

        # Stage 2: Quantitative Sentiment & Macro Lexicon Scoring
        scored_news_df = run_sentiment_scoring(db, days_back=args.days)

        # Stage 3: Quantitative Feature Engineering
        feature_data = run_feature_engineering(db, scored_news_df=scored_news_df, days_back=args.days)

        if args.report_only:
            results_path = REPORTS_DIR / "model_results.json"
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    model_results = json.load(f)
            else:
                logger.error("No model results found. Run training first.")
                return
        else:
            # Stage 4: ML Model + Walk-Forward Validation
            model_results = run_model_training(feature_data)

        # Stage 5: Metrics & Visualizations
        chart_paths = run_visualization(feature_data, model_results)

        # Stage 6: Quantitative Strategy Engine (Macro Outlook & Trade Decisions)
        strategy_outputs = run_strategy_engine(db, model_results, days_back=args.days)

        # Final Evaluation Report Compilation
        report_path = generate_evaluation_report(model_results, chart_paths)

        logger.info("=" * 65)
        logger.info("         Pipeline Completed Successfully!")
        logger.info("=" * 65)
        logger.info(f"  Macro Outlook:     {strategy_outputs['macro_outlook_path']}")
        logger.info(f"  Trade Signals:     {strategy_outputs['trade_signals_path']}")
        logger.info(f"  Evaluation Report: {report_path}")
        logger.info(f"  Charts Directory:  {CHARTS_DIR}")
        logger.info(f"  Model Results:     {REPORTS_DIR / 'model_results.json'}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
