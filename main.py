"""
CWT Macro Economics Sentiment Agent
====================================
End-to-end quantitative trading pipeline with dual-LLM integration:
  Tavily/Apify → LLM Scorer (JSON) → Features → ML Model + Walk-Forward → Metrics → LLM Agent (Outlook & Decision)

Usage:
    python main.py                    # Run complete 6-stage pipeline
    python main.py --scrape-only      # Only ingest data
    python main.py --score-only       # Only run LLM scorer on news data
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
from src.llm import LLMAgent, LLMScorer
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


def run_scraping(db: DatabaseManager, use_synthetic: bool = False) -> dict[str, Any]:
    """Stage 1: Data Ingestion (Apify Calendar & Tavily News)."""
    logger.info("=" * 65)
    logger.info("STAGE 1: Data Ingestion (Apify Calendar & Tavily News)")
    logger.info("=" * 65)

    # 1. Macro Calendar Scraping (ForexFactory via Apify)
    logger.info("Scraping macroeconomic calendar events...")
    calendar_scraper = MacroCalendarScraper(api_token=APIFY_API_TOKEN)

    if use_synthetic or not APIFY_API_TOKEN:
        logger.warning("No Apify token found or --synthetic flag set. Using synthetic data.")
        events = calendar_scraper._generate_synthetic_data(days_back=LOOKBACK_DAYS)
    else:
        events = calendar_scraper.run(days_back=LOOKBACK_DAYS)

    logger.info(f"  -> Collected {len(events)} macro events")
    db.insert_macro_events(events)

    # 2. Economic News Scraping (Tavily Search API)
    logger.info("Scraping economic news articles...")
    news_scraper = EconomicNewsScraper(api_key=TAVILY_API_KEY)

    if use_synthetic or not TAVILY_API_KEY:
        logger.warning("No Tavily key found or --synthetic flag set. Using synthetic news.")
        articles = news_scraper._generate_synthetic_news(days_back=LOOKBACK_DAYS)
    else:
        articles = news_scraper.run(days_back=LOOKBACK_DAYS)

    logger.info(f"  -> Collected {len(articles)} news articles")
    db.insert_news_articles(articles)

    # 3. Market Data Generation
    logger.info("Ingesting multi-asset market data...")
    market_data = _generate_market_data(LOOKBACK_DAYS)
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


def run_llm_scoring(db: DatabaseManager) -> pd.DataFrame:
    """Stage 2: LLM Scorer (JSON Extraction of Sentiment, Regime, and Hawk-Dove)."""
    logger.info("=" * 65)
    logger.info("STAGE 2: LLM Scorer (Structured JSON Extraction)")
    logger.info("=" * 65)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    news_df = db.get_news_articles(start_date, end_date)

    if news_df.empty:
        logger.warning("No news articles in database to score. Returning empty DataFrame.")
        return news_df

    scorer = LLMScorer()
    news_records = news_df.to_dict(orient="records")
    logger.info("Running LLMScorer across %d news articles...", len(news_records))
    scored_articles = scorer.score_articles_batch(news_records)
    scored_df = pd.DataFrame(scored_articles)

    # Save structured scores to disk
    scored_json_path = PROCESSED_DIR / "llm_scored_news.json"
    with open(scored_json_path, "w", encoding="utf-8") as f:
        json.dump(scored_articles, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  -> Saved {len(scored_articles)} LLM-scored articles to {scored_json_path}")

    return scored_df


def run_feature_engineering(db: DatabaseManager, scored_news_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Stage 3: Quantitative Feature Engineering."""
    logger.info("=" * 65)
    logger.info("STAGE 3: Quantitative Feature Engineering")
    logger.info("=" * 65)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

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
    }

    generated = charts.generate_all_charts(chart_data)
    logger.info(f"  -> Generated {len(generated)} charts in {CHARTS_DIR}")
    return generated


def run_llm_agent(db: DatabaseManager, model_results: dict[str, Any]) -> dict[str, Any]:
    """Stage 6: LLM Agent (Macro Outlook Synthesis & Strategic Trade Decisions)."""
    logger.info("=" * 65)
    logger.info("STAGE 6: LLM Agent (Macro Outlook & Trade Decision Matrix)")
    logger.info("=" * 65)

    # 1. Predict trades by joining macro events and news sentiment
    predictor = TradePredictor(db_manager=db, lookback_days=LOOKBACK_DAYS)
    joined_df = predictor.load_and_join_datasets()
    predictions_df = predictor.predict_trades(joined_df)
    csv_path, _ = predictor.save_predictions(predictions_df)
    logger.info(f"  -> Computed base trade signals across {len(predictions_df)} assets")

    # 2. Engage LLM Reasoning Agent
    agent = LLMAgent()
    events_df = db.get_macro_events()
    news_df = db.get_news_articles()

    agent_outputs = agent.generate_outlook_and_decisions(
        model_results=model_results,
        trade_predictions_df=predictions_df,
        macro_events_df=events_df,
        news_sentiment_df=news_df,
    )

    logger.info("  -> Macro Outlook saved: %s", agent_outputs["macro_outlook_path"])
    logger.info("  -> Trade Signals saved: %s", agent_outputs["trade_signals_path"])
    return agent_outputs


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
**End-to-End Pipeline**: `Tavily/Apify -> LLM Scorer (JSON) -> Features -> LightGBM Walk-Forward -> Metrics -> LLM Decision Agent`

---

## 1. Executive Summary

This report evaluates the **CWT Quantitative Macroeconomic Trading Pipeline**.
The pipeline ingests macroeconomic calendar surprises (ForexFactory via Apify), scores thematic economic news using an **LLM Scorer (JSON)**, builds quantitative interaction features, validates trade forecasting using sliding **walk-forward validation**, and synthesizes executive decisions through an **LLM Reasoning Agent**.

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

## 4. Visual Artifacts Generated

"""
    for i, path in enumerate(chart_paths, 1):
        report += f"{i}. **`{Path(path).name}`**\n"

    report += f"""
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
    parser = argparse.ArgumentParser(description="CWT Macro Economics Sentiment Agent")
    parser.add_argument("--scrape-only", action="store_true", help="Only scrape data")
    parser.add_argument("--score-only", action="store_true", help="Only run LLM scorer")
    parser.add_argument("--train-only", action="store_true", help="Only train model")
    parser.add_argument("--predict-only", action="store_true", help="Only predict trades")
    parser.add_argument("--report-only", action="store_true", help="Only generate reports")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic fallback mode")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("   CWT Macro Economics Sentiment Agent -- Pipeline Start")
    logger.info("=" * 65)
    logger.info(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"  Mode: {'Synthetic' if args.synthetic else 'Live Dual-LLM Pipeline'}")

    db = DatabaseManager()

    if args.predict_only:
        logger.info("Running in Predict-Only mode...")
        model_results = {}
        res_file = REPORTS_DIR / "model_results.json"
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                model_results = json.load(f)
        run_llm_agent(db, model_results)
        return

    if args.score_only:
        logger.info("Running in Score-Only mode...")
        run_llm_scoring(db)
        return

    use_synthetic = args.synthetic or (not APIFY_API_TOKEN and not TAVILY_API_KEY)
    if use_synthetic:
        logger.info("  [!] Running in SYNTHETIC mode (no API keys detected)")

    try:
        # Stage 1: Data Ingestion (Tavily & Apify)
        if not args.train_only and not args.report_only:
            run_scraping(db, use_synthetic=use_synthetic)

        if args.scrape_only:
            logger.info("Scrape-only mode complete.")
            return

        # Stage 2: LLM Scorer (JSON Extraction)
        scored_news_df = run_llm_scoring(db)

        # Stage 3: Quantitative Feature Engineering
        feature_data = run_feature_engineering(db, scored_news_df=scored_news_df)

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

        # Stage 6: LLM Agent (Macro Outlook & Trade Decisions)
        agent_outputs = run_llm_agent(db, model_results)

        # Final Evaluation Report Compilation
        report_path = generate_evaluation_report(model_results, chart_paths)

        logger.info("=" * 65)
        logger.info("         Pipeline Completed Successfully!")
        logger.info("=" * 65)
        logger.info(f"  Macro Outlook:     {agent_outputs['macro_outlook_path']}")
        logger.info(f"  Trade Signals:     {agent_outputs['trade_signals_path']}")
        logger.info(f"  Evaluation Report: {report_path}")
        logger.info(f"  Charts Directory:  {CHARTS_DIR}")
        logger.info(f"  Model Results:     {REPORTS_DIR / 'model_results.json'}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
