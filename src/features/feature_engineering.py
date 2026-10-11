"""
CWT Macro Economics Sentiment Agent - Feature Engineering Module
Transforms raw macroeconomic events, asset returns, and sentiment data into ML-ready features.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to sys.path for central config imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    ASSET_CLASSES,
    MACRO_EVENTS,
    MODEL_PARAMS,
    PROCESSED_DIR,
    REPORTS_DIR,
    STRATEGY_SIGNALS,
    WALK_FORWARD_TEST_DAYS,
    WALK_FORWARD_TRAIN_DAYS,
)

logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """Sanitize strings for column names."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return clean.strip("_")


class FeatureEngineer:
    """
    Feature engineering pipeline for macroeconomic events and asset price data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize FeatureEngineer with raw joined dataset.

        Args:
            df: DataFrame containing at least event_date, event_name,
                actual_value, forecast_value, previous_value, surprise,
                impact, asset_class, close_price, daily_return.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty.")

        self.df = df.copy()

        # Standardize date column
        date_candidates = ["event_date", "date", "Date", "Event_Date"]
        date_col = next((col for col in date_candidates if col in self.df.columns), None)
        if date_col is None:
            raise ValueError(f"Could not find a valid date column in: {list(self.df.columns)}")

        if date_col != "event_date":
            self.df["event_date"] = self.df[date_col]

        self.df["event_date"] = pd.to_datetime(self.df["event_date"])
        self.df = self.df.sort_values("event_date").reset_index(drop=True)

        self._feature_names: List[str] = []

    def create_surprise_features(self) -> pd.DataFrame:
        """
        For each event, compute:
          surprise = actual - forecast
          surprise_pct = surprise / abs(forecast) if forecast != 0
          surprise_direction = sign(surprise)

        Pivots so each event type becomes columns indexed by event_date.

        Returns:
            pd.DataFrame: Pivot table of surprise metrics per event date.
        """
        logger.info("Computing macro surprise features...")
        event_df = self.df.copy()

        required_cols = {"actual_value", "forecast_value"}
        if not required_cols.issubset(event_df.columns):
            logger.warning("Missing actual_value or forecast_value. Attempting fallback calculation.")
            if "actual_value" not in event_df.columns:
                event_df["actual_value"] = np.nan
            if "forecast_value" not in event_df.columns:
                event_df["forecast_value"] = np.nan

        # Fill surprise if missing
        if "surprise" not in event_df.columns or event_df["surprise"].isna().all():
            event_df["surprise"] = event_df["actual_value"] - event_df["forecast_value"]

        # Compute percentage surprise
        denom = event_df["forecast_value"].abs()
        event_df["surprise_pct"] = np.where(
            (denom > 1e-6) & event_df["actual_value"].notna() & event_df["forecast_value"].notna(),
            event_df["surprise"] / denom,
            0.0,
        )

        # Compute surprise direction (-1, 0, 1)
        event_df["surprise_direction"] = np.sign(event_df["surprise"].fillna(0.0))

        # Filter events with valid names
        if "event_name" not in event_df.columns:
            return pd.DataFrame(index=pd.DatetimeIndex(event_df["event_date"].unique()))

        event_subset = event_df.dropna(subset=["event_name", "event_date"]).copy()
        event_subset["event_slug"] = event_subset["event_name"].apply(_sanitize_name)

        # Group by event_date and event_slug to handle duplicates on same day
        grouped = (
            event_subset.groupby(["event_date", "event_slug"])[
                ["surprise", "surprise_pct", "surprise_direction"]
            ]
            .mean()
            .reset_index()
        )

        pivoted = grouped.pivot(
            index="event_date",
            columns="event_slug",
            values=["surprise", "surprise_pct", "surprise_direction"],
        )

        # Flatten multi-index columns: {event_slug}_{metric}
        flat_cols = [f"{col[1]}_{col[0]}" for col in pivoted.columns]
        pivoted.columns = flat_cols

        return pivoted

    def create_time_features(self) -> pd.DataFrame:
        """
        Generate calendar and proximity time features:
          - day_of_week
          - month
          - quarter
          - is_month_end
          - days_since_last_fomc
          - days_since_last_nfp
          - days_since_last_cpi

        Returns:
            pd.DataFrame: Time-based features indexed by event_date.
        """
        logger.info("Computing calendar and time proximity features...")
        unique_dates = pd.Series(sorted(self.df["event_date"].unique()), name="event_date")
        time_df = pd.DataFrame({"event_date": unique_dates})

        dt = time_df["event_date"].dt
        time_df["day_of_week"] = dt.dayofweek
        time_df["month"] = dt.month
        time_df["quarter"] = dt.quarter
        time_df["is_month_end"] = dt.is_month_end.astype(int)

        # Extract dates for key recurring macro releases
        event_names = self.df.get("event_name", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
        fomc_dates = sorted(self.df.loc[event_names.str.contains("fomc|fed interest|fed funds"), "event_date"].unique())
        nfp_dates = sorted(self.df.loc[event_names.str.contains("non-farm|payroll|nfp"), "event_date"].unique())
        cpi_dates = sorted(self.df.loc[event_names.str.contains(r"\bcpi\b"), "event_date"].unique())

        def _days_since_last(current_date: pd.Timestamp, event_dates: List[pd.Timestamp]) -> float:
            prior = [d for d in event_dates if d < current_date]
            if not prior:
                return 365.0  # Default cap for days elapsed
            return float((current_date - max(prior)).days)

        time_df["days_since_last_fomc"] = time_df["event_date"].apply(lambda d: _days_since_last(d, fomc_dates))
        time_df["days_since_last_nfp"] = time_df["event_date"].apply(lambda d: _days_since_last(d, nfp_dates))
        time_df["days_since_last_cpi"] = time_df["event_date"].apply(lambda d: _days_since_last(d, cpi_dates))

        return time_df.set_index("event_date")

    def create_rolling_features(self) -> pd.DataFrame:
        """
        Compute rolling return, momentum, volatility, and surprise statistics:
          - Rolling 5d/10d/20d mean and std of daily_return
          - Rolling 5d/10d mean of surprise values
          - Momentum (return over last 5d, 10d, 20d)
          - Volatility regime (rolling 20d std categorized as low/medium/high)

        Returns:
            pd.DataFrame: Rolling features indexed by event_date (or multi-indexed if multiple assets).
        """
        logger.info("Computing rolling return and volatility features...")
        df = self.df.copy()

        # Aggregate daily returns per date if multiple assets or pivot
        features_list = []
        has_asset_class = "asset_class" in df.columns and df["asset_class"].notna().any()

        # If asset classes exist, group per asset class; otherwise treat whole dataframe
        grouped = df.groupby("asset_class") if has_asset_class else [("ALL", df)]

        for asset, group in grouped:
            g = group.sort_values("event_date").drop_duplicates(subset=["event_date"]).copy()
            g = g.set_index("event_date")

            prefix = f"{_sanitize_name(asset)}_" if has_asset_class else ""

            # Ensure daily_return exists
            if "daily_return" not in g.columns:
                if "close_price" in g.columns:
                    g["daily_return"] = g["close_price"].pct_change()
                else:
                    g["daily_return"] = 0.0

            asset_feats = pd.DataFrame(index=g.index)

            # Rolling return mean and std (5d, 10d, 20d)
            for window in [5, 10, 20]:
                asset_feats[f"{prefix}return_roll_mean_{window}d"] = g["daily_return"].rolling(window, min_periods=1).mean()
                asset_feats[f"{prefix}return_roll_std_{window}d"] = g["daily_return"].rolling(window, min_periods=1).std().fillna(0.0)

            # Momentum over 5d, 10d, 20d
            if "close_price" in g.columns and g["close_price"].notna().any():
                for window in [5, 10, 20]:
                    asset_feats[f"{prefix}momentum_{window}d"] = g["close_price"].pct_change(window).fillna(0.0)
            else:
                for window in [5, 10, 20]:
                    asset_feats[f"{prefix}momentum_{window}d"] = g["daily_return"].rolling(window, min_periods=1).sum().fillna(0.0)

            # Volatility regime based on rolling 20d std
            vol_20d = asset_feats[f"{prefix}return_roll_std_20d"]
            q_low = vol_20d.quantile(0.33) if len(vol_20d.dropna()) > 10 else 0.005
            q_high = vol_20d.quantile(0.66) if len(vol_20d.dropna()) > 10 else 0.015

            vol_regime = pd.Series(1, index=asset_feats.index)  # Default 1: medium
            vol_regime[vol_20d <= q_low] = 0  # low
            vol_regime[vol_20d >= q_high] = 2  # high

            asset_feats[f"{prefix}vol_regime_low"] = (vol_regime == 0).astype(int)
            asset_feats[f"{prefix}vol_regime_medium"] = (vol_regime == 1).astype(int)
            asset_feats[f"{prefix}vol_regime_high"] = (vol_regime == 2).astype(int)

            features_list.append(asset_feats)

        combined_rolling = pd.concat(features_list, axis=1)

        # Rolling 5d / 10d mean of surprise values across all events
        if "surprise" in df.columns:
            daily_surprise = df.groupby("event_date")["surprise"].mean()
            combined_rolling["macro_surprise_roll_5d"] = daily_surprise.rolling(5, min_periods=1).mean()
            combined_rolling["macro_surprise_roll_10d"] = daily_surprise.rolling(10, min_periods=1).mean()
        else:
            combined_rolling["macro_surprise_roll_5d"] = 0.0
            combined_rolling["macro_surprise_roll_10d"] = 0.0

        return combined_rolling

    def create_sentiment_features(self, news_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        If news_df is provided:
          - Rolling average sentiment score (3d, 7d)
          - Sentiment momentum (change in rolling sentiment)
          - News volume (count of articles per day)
        If not provided or empty, returns empty DataFrame with date index.

        Args:
            news_df: DataFrame with sentiment and date columns.

        Returns:
            pd.DataFrame: Sentiment features indexed by event_date.
        """
        logger.info("Computing sentiment features...")
        all_dates = pd.DatetimeIndex(sorted(self.df["event_date"].unique()))
        empty_res = pd.DataFrame(index=all_dates)

        if news_df is None or news_df.empty:
            return empty_res

        ndf = news_df.copy()
        # Find date column
        date_col = next((c for c in ["date", "event_date", "published_at", "published_date"] if c in ndf.columns), None)
        if date_col is None:
            return empty_res

        ndf["event_date"] = pd.to_datetime(ndf[date_col])

        # Find sentiment score column
        score_col = next((c for c in ["sentiment_score", "sentiment", "score", "polarity"] if c in ndf.columns), None)
        if score_col is None:
            ndf["sentiment_score"] = 0.0
        else:
            ndf["sentiment_score"] = pd.to_numeric(ndf[score_col], errors="coerce").fillna(0.0)

        # Aggregate per date
        daily_sentiment = ndf.groupby("event_date").agg(
            mean_sentiment=("sentiment_score", "mean"),
            news_volume=("sentiment_score", "count"),
        )

        daily_sentiment = daily_sentiment.reindex(all_dates).fillna(0.0)

        daily_sentiment["sentiment_roll_3d"] = daily_sentiment["mean_sentiment"].rolling(3, min_periods=1).mean()
        daily_sentiment["sentiment_roll_7d"] = daily_sentiment["mean_sentiment"].rolling(7, min_periods=1).mean()
        daily_sentiment["sentiment_momentum"] = daily_sentiment["sentiment_roll_3d"] - daily_sentiment["sentiment_roll_7d"]

        return daily_sentiment[["sentiment_roll_3d", "sentiment_roll_7d", "sentiment_momentum", "news_volume"]]

    def create_macro_regime_features(self) -> pd.DataFrame:
        """
        Classify macro regimes based on:
          - CPI trend (inflationary / deflationary / stable)
          - Employment trend (improving / worsening / stable)
          - Growth trend (expanding / contracting / stable)
        Creates one-hot encoded regime features.

        Returns:
            pd.DataFrame: Macro regime indicator columns indexed by event_date.
        """
        logger.info("Computing macro regime features...")
        unique_dates = pd.DatetimeIndex(sorted(self.df["event_date"].unique()))
        regime_df = pd.DataFrame(index=unique_dates)

        df = self.df.copy()
        event_names = df.get("event_name", pd.Series(dtype=str)).fillna("").astype(str).str.lower()

        # 1. CPI Trend
        cpi_mask = event_names.str.contains(r"\bcpi\b")
        if cpi_mask.any() and "actual_value" in df.columns:
            cpi_series = df.loc[cpi_mask].groupby("event_date")["actual_value"].last().reindex(unique_dates).ffill()
            cpi_change = cpi_series.diff(3).fillna(0.0)
            cpi_regime = np.where(cpi_change > 0.05, "inflationary", np.where(cpi_change < -0.05, "deflationary", "stable"))
        else:
            cpi_regime = np.repeat("stable", len(unique_dates))

        # 2. Employment Trend (NFP / Unemployment)
        emp_mask = event_names.str.contains("non-farm|payroll|unemployment")
        if emp_mask.any() and "actual_value" in df.columns:
            emp_series = df.loc[emp_mask].groupby("event_date")["actual_value"].last().reindex(unique_dates).ffill()
            emp_change = emp_series.diff(3).fillna(0.0)
            emp_regime = np.where(emp_change > 0.0, "improving", np.where(emp_change < -0.0, "worsening", "stable"))
        else:
            emp_regime = np.repeat("stable", len(unique_dates))

        # 3. Growth Trend (GDP / PMI / Retail Sales)
        growth_mask = event_names.str.contains("gdp|pmi|retail")
        if growth_mask.any() and "actual_value" in df.columns:
            growth_series = df.loc[growth_mask].groupby("event_date")["actual_value"].last().reindex(unique_dates).ffill()
            growth_change = growth_series.diff(3).fillna(0.0)
            growth_regime = np.where(growth_change > 0.0, "expanding", np.where(growth_change < -0.0, "contracting", "stable"))
        else:
            growth_regime = np.repeat("stable", len(unique_dates))

        # One-hot encoding
        regime_df["regime_cpi_inflationary"] = (cpi_regime == "inflationary").astype(int)
        regime_df["regime_cpi_deflationary"] = (cpi_regime == "deflationary").astype(int)
        regime_df["regime_cpi_stable"] = (cpi_regime == "stable").astype(int)

        regime_df["regime_emp_improving"] = (emp_regime == "improving").astype(int)
        regime_df["regime_emp_worsening"] = (emp_regime == "worsening").astype(int)
        regime_df["regime_emp_stable"] = (emp_regime == "stable").astype(int)

        regime_df["regime_growth_expanding"] = (growth_regime == "expanding").astype(int)
        regime_df["regime_growth_contracting"] = (growth_regime == "contracting").astype(int)
        regime_df["regime_growth_stable"] = (growth_regime == "stable").astype(int)

        return regime_df

    def build_feature_matrix(self, news_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge all feature sets on date, forward-fill, drop remaining NaNs,
        and add binary next-day return direction target (1 if positive, 0 if negative).

        Args:
            news_df: Optional news dataframe for sentiment features.

        Returns:
            pd.DataFrame: Feature matrix containing features, event_date, and target.
        """
        logger.info("Building full feature matrix...")

        surprise_feats = self.create_surprise_features()
        time_feats = self.create_time_features()
        rolling_feats = self.create_rolling_features()
        sentiment_feats = self.create_sentiment_features(news_df=news_df)
        regime_feats = self.create_macro_regime_features()

        # Base frame on unique event_dates
        all_dates = pd.DatetimeIndex(sorted(self.df["event_date"].unique()))
        base = pd.DataFrame(index=all_dates)

        # Merge on date index
        matrix = base.join(time_feats, how="left")
        if not surprise_feats.empty:
            matrix = matrix.join(surprise_feats, how="left")
        if not rolling_feats.empty:
            matrix = matrix.join(rolling_feats, how="left")
        if not sentiment_feats.empty:
            matrix = matrix.join(sentiment_feats, how="left")
        if not regime_feats.empty:
            matrix = matrix.join(regime_feats, how="left")

        # Forward-fill macro announcements and rolling variables
        matrix = matrix.ffill().fillna(0.0)

        # Build target: next-day return direction (1 if > 0, 0 otherwise)
        # Compute next day return from asset data
        df_sorted = self.df.sort_values("event_date").reset_index(drop=True)
        if "daily_return" in df_sorted.columns:
            # If multiple asset classes are present, compute target per asset or mean market target
            daily_ret = df_sorted.groupby("event_date")["daily_return"].mean()
        elif "close_price" in df_sorted.columns:
            daily_close = df_sorted.groupby("event_date")["close_price"].last()
            daily_ret = daily_close.pct_change()
        else:
            daily_ret = pd.Series(0.0, index=all_dates)

        daily_ret = daily_ret.reindex(matrix.index).fillna(0.0)
        target_series = (daily_ret.shift(-1) > 0).astype(int)

        matrix["target"] = target_series
        matrix["daily_return"] = daily_ret
        matrix["event_date"] = matrix.index

        # Drop the last row since its target shift(-1) is looking into unobserved future
        matrix = matrix.iloc[:-1].copy()

        # Ensure no remaining NaNs
        matrix = matrix.dropna()

        # Cache feature names
        exclude = {"event_date", "date", "target", "daily_return", "close_price", "asset_class"}
        self._feature_names = [col for col in matrix.columns if col not in exclude]

        logger.info(f"Feature matrix built successfully: {matrix.shape[0]} rows, {len(self._feature_names)} features.")
        return matrix.reset_index(drop=True)

    def get_feature_importance_names(self) -> List[str]:
        """
        Return list of feature column names excluding dates, targets, and metadata.

        Returns:
            List[str]: Clean list of feature names.
        """
        return list(self._feature_names)
