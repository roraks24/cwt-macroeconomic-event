"""
CWT Macro Economics Sentiment Agent — Trade Prediction Module
==============================================================
Predicts optimal multi-asset trading strategy signals by joining:
  1. Macroeconomic Calendar Releases (Apify ForexFactory / Investing.com)
  2. Thematic Economic News & NLP Sentiment (Tavily Search API)
  3. Multi-Asset Market Data (US_EQUITY, US_BOND, GOLD, USD_INDEX, CRUDE_OIL)

Usage:
    python predict_trades.py                      # Predict latest trade recommendations
    python predict_trades.py --historical         # Backtest historical predictions on joined data
    python predict_trades.py --asset US_EQUITY    # Predict for specific asset
    python predict_trades.py --synthetic          # Run in offline synthetic mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ASSET_CLASSES,
    DATABASE_URL,
    LOOKBACK_DAYS,
    PROCESSED_DIR,
    REPORTS_DIR,
    WALK_FORWARD_TEST_DAYS,
    WALK_FORWARD_TRAIN_DAYS,
)
from src.database.db_manager import DatabaseManager
from src.features.feature_engineering import FeatureEngineer
from src.model.walk_forward import WalkForwardModel
from src.llm import LLMAgent, LLMScorer

logger = logging.getLogger("CWT-TradePredictor")


class TradePredictor:
    """
    Unifies macroeconomic event releases, news sentiment, and multi-asset price action
    into a joint dataset and forecasts trade signals with win rates, expected P&L,
    and macro driver rationales.
    """

    def __init__(self, db_manager: DatabaseManager | None = None, lookback_days: int = LOOKBACK_DAYS) -> None:
        self.db = db_manager or DatabaseManager()
        self.lookback_days = lookback_days

    def load_and_join_datasets(
        self,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Extract macroeconomic event releases, economic news sentiment, and market data,
        and join them on calendar trading date boundaries into a unified dataset.
        """
        end_dt = pd.to_datetime(end_date) if end_date else datetime.now(timezone.utc)
        start_dt = pd.to_datetime(start_date) if start_date else (end_dt - timedelta(days=self.lookback_days))

        logger.info("Loading macro events from database for %s to %s...", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        events_df = self.db.get_macro_events(start_dt, end_dt)

        logger.info("Loading news articles & sentiment from database...")
        news_df = self.db.get_news_articles(start_dt, end_dt)

        # Stage 2: LLM Scorer (JSON)
        if not news_df.empty:
            logger.info("Enriching news with LLMScorer (JSON)...")
            scorer = LLMScorer()
            news_records = news_df.to_dict(orient="records")
            enriched_news = scorer.score_articles_batch(news_records)
            news_df = pd.DataFrame(enriched_news)

        logger.info("Loading multi-asset market data from database...")
        market_df = self.db.get_market_data(start_date=start_dt, end_date=end_dt)

        if market_df.empty:
            logger.warning("Market data table is empty. Generating baseline market data...")
            from main import _generate_market_data
            synthetic_market = _generate_market_data(self.lookback_days)
            self.db.insert_market_data(synthetic_market)
            market_df = self.db.get_market_data(start_date=start_dt, end_date=end_dt)

        # ── 1. Aggregate Daily Macro Calendar Surprises ────────────────────────
        daily_macro = self._aggregate_daily_macro_events(events_df)

        # ── 2. Aggregate Daily News Sentiment & LLM Metrics ────────────────────
        daily_sentiment = self._aggregate_daily_news_sentiment(news_df)

        # ── 3. Join with Market Data across Assets ─────────────────────────────
        market_df["calendar_date"] = pd.to_datetime(market_df["date"]).dt.date

        # Left merge daily macro events onto market dates
        joined = pd.merge(market_df, daily_macro, on="calendar_date", how="left")

        # Left merge daily sentiment onto market dates
        joined = pd.merge(joined, daily_sentiment, on="calendar_date", how="left")

        # Fill missing values on non-event or non-news days with neutral values
        joined = self._impute_neutral_defaults(joined)

        joined.sort_values(by=["calendar_date", "asset_class"], inplace=True)
        joined.reset_index(drop=True, inplace=True)

        logger.info("Successfully constructed joined macro-sentiment-market dataset: %d rows, %d columns", len(joined), len(joined.columns))
        return joined

    def _aggregate_daily_macro_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate multiple intraday macro releases into structured daily surprise metrics."""
        if events_df.empty:
            return pd.DataFrame(columns=["calendar_date", "surprise_sum", "surprise_mean", "high_impact_events", "cpi_surprise", "nfp_surprise", "fomc_event"])

        df = events_df.copy()
        df["calendar_date"] = pd.to_datetime(df["event_date"]).dt.date

        records = []
        for c_date, group in df.groupby("calendar_date"):
            surprises = pd.to_numeric(group["surprise"], errors="coerce").dropna()
            high_impact_count = int((group["impact"].str.lower() == "high").sum())

            # Detect key specific indicator surprises
            cpi_rows = group[group["event_name"].str.contains("CPI|Consumer Price", case=False, na=False)]
            nfp_rows = group[group["event_name"].str.contains("Payrolls|Employment Change", case=False, na=False)]
            fomc_rows = group[group["event_name"].str.contains("FOMC|Fed Interest Rate", case=False, na=False)]

            cpi_surp = float(cpi_rows["surprise"].iloc[0]) if not cpi_rows.empty and pd.notna(cpi_rows["surprise"].iloc[0]) else 0.0
            nfp_surp = float(nfp_rows["surprise"].iloc[0]) if not nfp_rows.empty and pd.notna(nfp_rows["surprise"].iloc[0]) else 0.0
            fomc_flag = 1.0 if not fomc_rows.empty else 0.0

            records.append({
                "calendar_date": c_date,
                "surprise_sum": float(surprises.sum()) if not surprises.empty else 0.0,
                "surprise_mean": float(surprises.mean()) if not surprises.empty else 0.0,
                "high_impact_events": high_impact_count,
                "cpi_surprise": cpi_surp,
                "nfp_surprise": nfp_surp,
                "fomc_event": fomc_flag,
            })

        return pd.DataFrame(records)

    def _aggregate_daily_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news articles into daily polarity, LLM metrics, article counts, and rolling sentiment."""
        if news_df.empty:
            return pd.DataFrame(columns=["calendar_date", "news_sentiment", "llm_hawk_dove", "llm_growth", "llm_inflation", "article_count"])

        df = news_df.copy()
        df["calendar_date"] = pd.to_datetime(df["published_date"]).dt.date

        records = []
        for c_date, group in df.groupby("calendar_date"):
            sentiments = pd.to_numeric(group.get("sentiment_score"), errors="coerce").dropna()
            hawk_dove = pd.to_numeric(group.get("llm_hawk_dove_score"), errors="coerce").dropna()
            growth = pd.to_numeric(group.get("llm_growth_momentum"), errors="coerce").dropna()
            inflation = pd.to_numeric(group.get("llm_inflation_pressure"), errors="coerce").dropna()
            impact = pd.to_numeric(group.get("llm_surprise_impact"), errors="coerce").dropna()

            records.append({
                "calendar_date": c_date,
                "news_sentiment": float(sentiments.mean()) if not sentiments.empty else 0.0,
                "llm_hawk_dove": float(hawk_dove.mean()) if not hawk_dove.empty else 0.0,
                "llm_growth": float(growth.mean()) if not growth.empty else 0.0,
                "llm_inflation": float(inflation.mean()) if not inflation.empty else 0.0,
                "llm_impact": float(impact.mean()) if not impact.empty else 0.5,
                "article_count": len(group),
            })

        res_df = pd.DataFrame(records)
        res_df.sort_values(by="calendar_date", inplace=True)

        # Rolling 3-day and 7-day smoothing
        res_df["sentiment_3d_rolling"] = res_df["news_sentiment"].rolling(3, min_periods=1).mean()
        res_df["sentiment_7d_rolling"] = res_df["news_sentiment"].rolling(7, min_periods=1).mean()
        res_df["sentiment_momentum"] = res_df["sentiment_3d_rolling"] - res_df["sentiment_7d_rolling"]
        res_df["hawk_dove_7d"] = res_df["llm_hawk_dove"].rolling(7, min_periods=1).mean()
        return res_df

    def _impute_neutral_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill days without releases or news with forward-filled or neutral defaults."""
        fill_zero_cols = ["surprise_sum", "surprise_mean", "high_impact_events", "cpi_surprise", "nfp_surprise", "fomc_event", "article_count"]
        for col in fill_zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        sentiment_cols = [
            "news_sentiment", "sentiment_3d_rolling", "sentiment_7d_rolling", "sentiment_momentum",
            "llm_hawk_dove", "hawk_dove_7d", "llm_growth", "llm_inflation", "llm_impact"
        ]
        for col in sentiment_cols:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0.0)

        if "news_subjectivity" in df.columns:
            df["news_subjectivity"] = df["news_subjectivity"].ffill().fillna(0.5)

        return df

    def engineer_trade_features(self, joined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute quantitative model features from the joined dataset:
        - Surprise x Sentiment interactions
        - Proximity to key events
        - Moving average momentum & volatility regimes
        - Target variable (next-day directional return: 1 = UP, 0 = DOWN)
        """
        df = joined_df.copy()
        df["calendar_date"] = pd.to_datetime(df["calendar_date"])

        # Time features
        df["day_of_week"] = df["calendar_date"].dt.dayofweek
        df["is_month_end"] = df["calendar_date"].dt.is_month_end.astype(int)

        # Interaction feature: Surprise amplified by News Sentiment
        df["surprise_sentiment_interaction"] = df["surprise_sum"] * df["news_sentiment"]

        # Asset-level momentum & volatility features
        feature_dfs = []
        for asset, group in df.groupby("asset_class"):
            grp = group.sort_values(by="calendar_date").copy()

            # Technical returns & rolling stats
            grp["return_lag1"] = grp["daily_return"].shift(1).fillna(0.0)
            grp["return_roll_mean_5d"] = grp["daily_return"].rolling(5, min_periods=1).mean()
            grp["return_roll_std_5d"] = grp["daily_return"].rolling(5, min_periods=1).std().fillna(0.01)
            grp["vol_regime_high"] = (grp["return_roll_std_5d"] > grp["return_roll_std_5d"].quantile(0.66)).astype(int)

            # Binary target: 1 if next-day return > 0, else 0
            grp["target"] = (grp["daily_return"].shift(-1) > 0).astype(int)
            feature_dfs.append(grp)

        out_df = pd.concat(feature_dfs, ignore_index=True)
        out_df.sort_values(by=["calendar_date", "asset_class"], inplace=True)
        return out_df

    def predict_trades(
        self,
        joined_df: pd.DataFrame | None = None,
        asset_class: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate model trade signals (LONG / SHORT / NEUTRAL) for current market state,
        including win probabilities, expected P&L, and macro driver rationales.
        """
        if joined_df is None:
            joined_df = self.load_and_join_datasets()

        fe_df = self.engineer_trade_features(joined_df)

        assets_to_evaluate = [asset_class] if asset_class else ASSET_CLASSES
        predictions_list = []

        feature_cols = [
            "surprise_sum", "surprise_mean", "high_impact_events",
            "cpi_surprise", "nfp_surprise", "fomc_event",
            "news_sentiment", "sentiment_3d_rolling", "sentiment_momentum",
            "llm_hawk_dove", "hawk_dove_7d", "llm_growth", "llm_inflation", "llm_impact",
            "surprise_sentiment_interaction", "day_of_week",
            "return_lag1", "return_roll_mean_5d", "return_roll_std_5d", "vol_regime_high",
        ]

        # Use WalkForwardModel with LightGBM for robust predictions
        for asset in assets_to_evaluate:
            asset_rows = fe_df[fe_df["asset_class"] == asset].sort_values(by="calendar_date").copy()
            if len(asset_rows) < 15:
                continue

            # Train on history up to latest point
            train_data = asset_rows.iloc[:-1] if len(asset_rows) > 1 else asset_rows
            latest_row = asset_rows.iloc[-1]

            X_train = train_data[feature_cols].fillna(0.0)
            y_train = train_data["target"].values

            # Initialize LightGBM classifier
            try:
                import lightgbm as lgb
                clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1,
                )
                if len(np.unique(y_train)) > 1:
                    clf.fit(X_train, y_train)
                    X_latest = pd.DataFrame([latest_row[feature_cols].fillna(0.0)])
                    prob_up = float(clf.predict_proba(X_latest)[0][1])
                else:
                    prob_up = 0.55 if y_train[0] == 1 else 0.45
            except Exception as e:
                logger.warning("LightGBM fitting fallback for %s: %s", asset, e)
                # Heuristic fallback based on macro surprise + news sentiment
                raw_score = latest_row["news_sentiment"] * 0.5 + np.sign(latest_row["surprise_sum"]) * 0.2
                prob_up = float(1.0 / (1.0 + np.exp(-raw_score * 3.0)))

            # Signal thresholding
            if prob_up >= 0.53:
                signal = "LONG"
                confidence = round(prob_up, 4)
            elif prob_up <= 0.47:
                signal = "SHORT"
                confidence = round(1.0 - prob_up, 4)
            else:
                signal = "NEUTRAL"
                confidence = 0.50

            # Dynamic Volatility & Expected Return
            recent_vol = float(latest_row.get("return_roll_std_5d", 0.01))
            if signal == "LONG":
                expected_pnl_pct = round((prob_up - 0.5) * recent_vol * 100 * 2.5, 3)
            elif signal == "SHORT":
                expected_pnl_pct = round((0.5 - prob_up) * recent_vol * 100 * 2.5, 3)
            else:
                expected_pnl_pct = 0.0

            stop_loss_pct = round(recent_vol * 100 * 1.5, 2)
            take_profit_pct = round(recent_vol * 100 * 2.5, 2)

            # Determine primary macro driver
            drivers = []
            if abs(latest_row["cpi_surprise"]) > 0.05:
                drivers.append(f"CPI Surprise ({latest_row['cpi_surprise']:+.2f})")
            if abs(latest_row["nfp_surprise"]) > 1000:
                drivers.append(f"NFP Surprise ({latest_row['nfp_surprise']:+,.0f})")
            if latest_row["fomc_event"] > 0:
                drivers.append("FOMC Policy Event")
            if abs(latest_row["news_sentiment"]) > 0.1:
                sentiment_tone = "Bullish" if latest_row["news_sentiment"] > 0 else "Bearish"
                drivers.append(f"{sentiment_tone} News Sentiment ({latest_row['news_sentiment']:+.2f})")
            if not drivers:
                drivers.append("Technical Momentum & Mean Reversion")

            primary_driver = "; ".join(drivers)

            predictions_list.append({
                "trade_date": latest_row["calendar_date"].strftime("%Y-%m-%d") if hasattr(latest_row["calendar_date"], "strftime") else str(latest_row["calendar_date"]),
                "asset_class": asset,
                "signal": signal,
                "win_probability": round(prob_up if signal == "LONG" else (1.0 - prob_up if signal == "SHORT" else 0.5), 4),
                "expected_return_pct": expected_pnl_pct,
                "confidence_score": confidence,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "macro_driver": primary_driver,
            })

        pred_df = pd.DataFrame(predictions_list)
        return pred_df

    def save_predictions(self, pred_df: pd.DataFrame) -> tuple[Path, Path]:
        """Save trade predictions to CSV and JSON formats in processed and reports directories."""
        csv_path = PROCESSED_DIR / "trade_predictions.csv"
        json_path = REPORTS_DIR / "trade_predictions.json"

        pred_df.to_csv(csv_path, index=False)

        json_records = pred_df.to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "trades_count": len(pred_df),
                "recommendations": json_records,
            }, f, indent=2)

        logger.info("Saved trade predictions to %s and %s", csv_path, json_path)
        return csv_path, json_path

    def generate_trade_report(self, pred_df: pd.DataFrame) -> Path:
        """Create a Markdown trade recommendation summary report."""
        report_path = REPORTS_DIR / "trade_signals.md"

        rows = ""
        for _, r in pred_df.iterrows():
            badge = "🟢 LONG" if r["signal"] == "LONG" else ("🔴 SHORT" if r["signal"] == "SHORT" else "⚪ NEUTRAL")
            win_pct = f"{r['win_probability'] * 100:.1f}%"
            pnl_pct = f"{r['expected_return_pct']:+.2f}%"
            rows += (
                f"| **{r['asset_class']}** | {badge} | {win_pct} | {pnl_pct} "
                f"| -{r['stop_loss_pct']:.2f}% | +{r['take_profit_pct']:.2f}% | {r['macro_driver']} |\n"
            )

        content = f"""# CWT Macro Economics Sentiment Agent — Trade Recommendations

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Methodology**: LightGBM Multi-Asset Predictive Model conditioned on Joined ForexFactory Releases & Tavily News Sentiment.

---

## 1. Executive Trade Signals

| Asset Class | Signal | Win Probability | Expected Return | Stop Loss | Take Profit | Primary Macro Driver |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
{rows}
---

## 2. Quantitative Model Architecture

- **Calendar Integration**: Daily actual-versus-forecast surprise metrics from Apify (`xtracto/forexfactory-calendar`).
- **News NLP**: Polarity scoring via Tavily Search API with TextBlob NLP analysis.
- **Risk Management**: Volatility-scaled dynamic stops (1.5x 5d ATR equivalent) and profit targets (2.5x ATR).

*Report generated automatically by CWT Macro Economics Sentiment Agent.*
"""
        report_path.write_text(content, encoding="utf-8")
        logger.info("Trade signals markdown report saved to %s", report_path)
        return report_path


def main():
    """CLI execution for trade prediction."""
    parser = argparse.ArgumentParser(description="Predict trades by joining macro events and news sentiment")
    parser.add_argument("--asset", type=str, default=None, help="Filter for specific asset (e.g., US_EQUITY)")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="Lookback window in days")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic fallback data")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    predictor = TradePredictor(lookback_days=args.days)
    joined_df = predictor.load_and_join_datasets()
    preds = predictor.predict_trades(joined_df, asset_class=args.asset)

    print("\n" + "=" * 95)
    print("           CWT MACRO AGENT -- PREDICTED TRADES & STRATEGY ALLOCATIONS")
    print("=" * 95)
    headers = f"{'Asset Class':<15} {'Signal':<10} {'Win Rate':<12} {'Exp Return':<12} {'Risk / Reward':<16} {'Macro Driver'}"
    print(headers)
    print("-" * 95)

    for _, row in preds.iterrows():
        sig_str = f"[{row['signal']}]"
        win_str = f"{row['win_probability']*100:.1f}%"
        exp_str = f"{row['expected_return_pct']:+.2f}%"
        rr_str = f"-{row['stop_loss_pct']:.1f}% / +{row['take_profit_pct']:.1f}%"
        driver = (row['macro_driver'][:28] + "..") if len(row['macro_driver']) > 30 else row['macro_driver']
        print(f"{row['asset_class']:<15} {sig_str:<10} {win_str:<12} {exp_str:<12} {rr_str:<16} {driver}")
    print("=" * 95 + "\n")

    predictor.save_predictions(preds)

    # Stage 6: LLM Agent (Outlook and Decision)
    logger.info("Engaging LLM Agent for macroeconomic outlook synthesis and strategic trade decisions (lookback=%d days)...", args.days)
    agent = LLMAgent()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)
    events_df = predictor.db.get_macro_events(start_date, end_date)
    news_df = predictor.db.get_news_articles(start_date, end_date)

    model_results = {}
    results_path = REPORTS_DIR / "model_results.json"
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                model_results = json.load(f)
        except Exception:
            pass

    agent_res = agent.generate_outlook_and_decisions(
        model_results=model_results,
        trade_predictions_df=preds,
        macro_events_df=events_df,
        news_sentiment_df=news_df,
    )
    logger.info("LLM Agent successfully updated macro outlook and trade decision matrices.")


if __name__ == "__main__":
    main()
