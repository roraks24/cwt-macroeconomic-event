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
from src.utils.timezone import assign_to_eligible_trading_session

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

        # Standardize date column (prioritize calendar_date or date to capture all market trading days)
        date_candidates = ["calendar_date", "date", "event_date", "Date", "Event_Date"]
        date_col = next((col for col in date_candidates if col in self.df.columns and self.df[col].notna().any()), None)
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
        if "event_name" in self.df.columns:
            event_names = self.df["event_name"].fillna("").astype(str).str.lower()
            fomc_dates = sorted(self.df.loc[event_names.str.contains("fomc|fed interest|fed funds"), "event_date"].unique())
            nfp_dates = sorted(self.df.loc[event_names.str.contains("non-farm|payroll|nfp"), "event_date"].unique())
            cpi_dates = sorted(self.df.loc[event_names.str.contains(r"\bcpi\b"), "event_date"].unique())
        else:
            fomc_dates, nfp_dates, cpi_dates = [], [], []

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

            # Volatility regime based on rolling 20d std (strictly backward-looking expanding quantile to prevent lookahead leakage)
            vol_20d = asset_feats[f"{prefix}return_roll_std_20d"]
            q_low = vol_20d.expanding(min_periods=10).quantile(0.33).fillna(0.006)
            q_high = vol_20d.expanding(min_periods=10).quantile(0.66).fillna(0.014)

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
        # Find publication date/time column
        pub_col = next((c for c in ["published_date", "published_at", "publication_time", "date", "event_date"] if c in ndf.columns), None)
        if pub_col is None:
            return empty_res

        # Distinguish retrieval/scrape time from publication time
        retrieval_col = next((c for c in ["created_at", "scraped_at", "retrieved_at"] if c in ndf.columns), None)

        min_date = all_dates[0]

        # Timezone-aware session assignment: articles published after 16:00 America/New_York (or retrieved after cutoff)
        # are strictly excluded from that day's feature vector and rolled forward to the next eligible trading session.
        eligible_dates = []
        for _, row in ndf.iterrows():
            pub_val = row[pub_col]
            ret_val = row[retrieval_col] if retrieval_col else None
            el_dt = assign_to_eligible_trading_session(pub_val, retrieved_at=ret_val, trading_dates=all_dates)
            eligible_dates.append(el_dt)

        ndf["event_date"] = pd.to_datetime(eligible_dates)

        # Check for legitimate pre-window lookback articles within the feature lookback horizon (up to 14 days before min_date)
        from src.utils.timezone import USEquityBusinessDay
        usb = USEquityBusinessDay
        lookback_cutoff = min_date - pd.Timedelta(days=14)
        pre_window_articles = ndf[(ndf["event_date"] < min_date) & (ndf["event_date"] >= lookback_cutoff)]

        if not pre_window_articles.empty:
            earliest_pre_date = pre_window_articles["event_date"].min()
            pre_dates = pd.bdate_range(start=earliest_pre_date, end=min_date, freq=usb).floor("D")[:-1]
            active_calendar = pre_dates.union(all_dates).sort_values()
        else:
            active_calendar = all_dates

        # Find sentiment score column
        score_col = next((c for c in ["sentiment_score", "sentiment", "score", "polarity"] if c in ndf.columns), None)
        if score_col is None:
            ndf["sentiment_score"] = 0.0
        else:
            ndf["sentiment_score"] = pd.to_numeric(ndf[score_col], errors="coerce").fillna(0.0)

        # Check for LLM Scorer structured fields
        llm_cols = ["llm_hawk_dove_score", "llm_growth_momentum", "llm_inflation_pressure", "llm_surprise_impact"]
        for lc in llm_cols:
            if lc in ndf.columns:
                ndf[lc] = pd.to_numeric(ndf[lc], errors="coerce").fillna(0.0)

        agg_dict = {
            "mean_sentiment": ("sentiment_score", "mean"),
            "news_volume": ("sentiment_score", "count"),
        }
        for lc in llm_cols:
            if lc in ndf.columns:
                agg_dict[f"mean_{lc}"] = (lc, "mean")

        daily_sentiment = ndf.groupby("event_date").agg(**agg_dict)
        # Reindex across active calendar so rolling windows incorporate legitimate pre-window lookback
        daily_sentiment = daily_sentiment.reindex(active_calendar).fillna(0.0)

        daily_sentiment["sentiment_roll_3d"] = daily_sentiment["mean_sentiment"].rolling(3, min_periods=1).mean()
        daily_sentiment["sentiment_roll_7d"] = daily_sentiment["mean_sentiment"].rolling(7, min_periods=1).mean()
        daily_sentiment["sentiment_momentum"] = daily_sentiment["sentiment_roll_3d"] - daily_sentiment["sentiment_roll_7d"]

        # Slice back strictly to the evaluation dates
        daily_sentiment = daily_sentiment.loc[all_dates]

        out_cols = ["sentiment_roll_3d", "sentiment_roll_7d", "sentiment_momentum", "news_volume"]
        for lc in llm_cols:
            if f"mean_{lc}" in daily_sentiment.columns:
                out_cols.append(f"mean_{lc}")

        return daily_sentiment[out_cols]

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
        macro_matrix = base.join(time_feats, how="left")
        if not surprise_feats.empty:
            macro_matrix = macro_matrix.join(surprise_feats, how="left")
        if not rolling_feats.empty:
            macro_matrix = macro_matrix.join(rolling_feats, how="left")
        if not sentiment_feats.empty:
            macro_matrix = macro_matrix.join(sentiment_feats, how="left")
        if not regime_feats.empty:
            macro_matrix = macro_matrix.join(regime_feats, how="left")

        # Forward-fill macro announcements and rolling variables
        macro_matrix = macro_matrix.ffill().fillna(0.0)

        has_assets = "asset_class" in self.df.columns and self.df["asset_class"].notna().any()
        if has_assets and len(self.df["asset_class"].unique()) > 1:
            panel_slices = []
            for asset in sorted(self.df["asset_class"].unique()):
                asset_sub = self.df[self.df["asset_class"] == asset].drop_duplicates(subset=["event_date"]).sort_values("event_date").copy()
                asset_sub = asset_sub.set_index("event_date").reindex(all_dates)

                # Next-Session Open Execution when open_price is available:
                # Features X_T observed up to Day T Close (16:00 EST).
                # Signal S_T generated post-close.
                # Entry is at Day T+1 Open: Open_{T+1} = open_series.shift(-1).
                # Exit/rebalance is at Day T+2 Open: Open_{T+2} = open_series.shift(-2).
                # Trade return = (Open_{T+2} - Open_{T+1}) / Open_{T+1}.
                has_open = "open_price" in asset_sub.columns and asset_sub["open_price"].notna().sum() > 2
                if has_open:
                    open_series = asset_sub["open_price"].astype(float)
                    trade_ret = (open_series.shift(-2) - open_series.shift(-1)) / open_series.shift(-1)
                    entry_price = open_series.shift(-1)
                    exit_price = open_series.shift(-2)
                    drop_n = 2
                elif "daily_return" in asset_sub.columns:
                    asset_ret = asset_sub["daily_return"].fillna(0.0)
                    trade_ret = asset_ret.shift(-1)
                    entry_price = pd.Series(1.0, index=all_dates)
                    exit_price = 1.0 + trade_ret
                    drop_n = 1
                elif "close_price" in asset_sub.columns:
                    close_series = asset_sub["close_price"].astype(float)
                    trade_ret = close_series.pct_change().shift(-1)
                    entry_price = close_series
                    exit_price = close_series.shift(-1)
                    drop_n = 1
                else:
                    trade_ret = pd.Series(0.0, index=all_dates)
                    entry_price = pd.Series(1.0, index=all_dates)
                    exit_price = pd.Series(1.0, index=all_dates)
                    drop_n = 1

                asset_target = (trade_ret > 0).astype(int)

                slice_df = macro_matrix.copy()
                slice_df["asset_class"] = asset
                slice_df["daily_return"] = trade_ret.fillna(0.0)
                slice_df["trade_return"] = trade_ret.fillna(0.0)
                slice_df["open_to_open_return"] = trade_ret.fillna(0.0)
                slice_df["target"] = asset_target
                slice_df["entry_price"] = entry_price
                slice_df["exit_price"] = exit_price
                slice_df["event_date"] = slice_df.index
                if "surprise" in asset_sub.columns and asset_sub["surprise"].notna().any():
                    slice_df["surprise"] = asset_sub["surprise"].fillna(0.0)
                elif "macro_surprise_roll_5d" in slice_df.columns:
                    slice_df["surprise"] = slice_df["macro_surprise_roll_5d"]
                else:
                    slice_df["surprise"] = 0.0

                # Drop the unobserved rows for this specific asset
                slice_df = slice_df.iloc[:-drop_n].copy()
                panel_slices.append(slice_df)

            matrix = pd.concat(panel_slices, axis=0).sort_values(by=["event_date", "asset_class"]).reset_index(drop=True)
        else:
            # Single asset series or aggregated fallback
            df_sorted = self.df.sort_values("event_date").reset_index(drop=True)
            has_open = "open_price" in df_sorted.columns and df_sorted["open_price"].notna().sum() > 2
            if has_open:
                daily_open = df_sorted.groupby("event_date")["open_price"].first().reindex(macro_matrix.index)
                trade_ret = (daily_open.shift(-2) - daily_open.shift(-1)) / daily_open.shift(-1)
                entry_price = daily_open.shift(-1)
                exit_price = daily_open.shift(-2)
                drop_n = 2
            elif "daily_return" in df_sorted.columns:
                daily_ret = df_sorted.groupby("event_date")["daily_return"].mean().reindex(macro_matrix.index).fillna(0.0)
                trade_ret = daily_ret.shift(-1)
                entry_price = pd.Series(1.0, index=macro_matrix.index)
                exit_price = 1.0 + trade_ret
                drop_n = 1
            elif "close_price" in df_sorted.columns:
                daily_close = df_sorted.groupby("event_date")["close_price"].last().reindex(macro_matrix.index)
                trade_ret = daily_close.pct_change().shift(-1)
                entry_price = daily_close
                exit_price = daily_close.shift(-1)
                drop_n = 1
            else:
                trade_ret = pd.Series(0.0, index=macro_matrix.index)
                entry_price = pd.Series(1.0, index=macro_matrix.index)
                exit_price = pd.Series(1.0, index=macro_matrix.index)
                drop_n = 1

            matrix = macro_matrix.copy()
            matrix["target"] = (trade_ret > 0).astype(int)
            matrix["daily_return"] = trade_ret.fillna(0.0)
            matrix["trade_return"] = trade_ret.fillna(0.0)
            matrix["open_to_open_return"] = trade_ret.fillna(0.0)
            matrix["entry_price"] = entry_price
            matrix["exit_price"] = exit_price
            matrix["event_date"] = matrix.index
            if "surprise" in df_sorted.columns:
                matrix["surprise"] = df_sorted.groupby("event_date")["surprise"].mean().reindex(matrix.index).fillna(0.0)
            elif "macro_surprise_roll_5d" in matrix.columns:
                matrix["surprise"] = matrix["macro_surprise_roll_5d"]

            if has_assets:
                matrix["asset_class"] = self.df["asset_class"].iloc[0]

            # Drop the unobserved rows
            matrix = matrix.iloc[:-drop_n].copy().reset_index(drop=True)

        # Ensure no remaining NaNs
        matrix = matrix.dropna()

        # Cache feature names (exclude dates, targets, returns, prices, and metadata)
        exclude = {
            "event_date", "date", "target", "daily_return", "trade_return", "open_to_open_return",
            "close_price", "open_price", "entry_price", "exit_price", "asset_class", "surprise"
        }
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
