"""
CWT Macro Economics Sentiment Agent - Walk-Forward Modeling Module
Implements time-series walk-forward cross validation, machine learning training,
backtesting, and quantitative strategy metrics evaluation.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


class WalkForwardModel:
    """
    Sequential walk-forward validation and backtesting pipeline for
    macroeconomic sentiment trading models.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        train_days: int = WALK_FORWARD_TRAIN_DAYS,
        test_days: int = WALK_FORWARD_TEST_DAYS,
        purge_periods: int = 2,
    ) -> None:
        """
        Initialize the WalkForwardModel.

        Args:
            model_type: 'lightgbm' or 'xgboost'.
            train_days: Number of calendar days for the rolling training window.
            test_days: Number of calendar days for the sequential test window.
            purge_periods: Number of final training periods to purge at the train/test
                           boundary to ensure forward targets (Open_{t+2} - Open_{t+1})/Open_{t+1}
                           have fully matured and do not leak into the test period.
        """
        self.model_type = model_type.lower()
        self.train_days = int(train_days)
        self.test_days = int(test_days)
        self.purge_periods = int(purge_periods)

        # Retrieve model hyper-parameters from config
        self.params: Dict[str, Any] = MODEL_PARAMS.get(self.model_type, {}).copy()

    def _create_model(self) -> Any:
        """
        Instantiate a fresh classifier according to model_type.

        Returns:
            An unfitted classifier instance (LGBMClassifier or XGBClassifier).
        """
        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(**self.params)
            except ImportError:
                logger.warning("LightGBM not installed. Falling back to sklearn RandomForestClassifier.")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(**self.params)
            except ImportError:
                logger.warning("XGBoost not installed. Falling back to sklearn GradientBoostingClassifier.")
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Supported: 'lightgbm', 'xgboost'.")

    def walk_forward_split(
        self,
        df: pd.DataFrame,
        date_col: str = "event_date",
        purge_periods: Optional[int] = None,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate sequential walk-forward train and test splits.
        Starting from earliest date + train_days, creates sequential windows
        where train = prior train_days of data, test = next test_days.
        Slides forward by test_days each iteration with ZERO future lookahead.
        Applies a boundary purge to train_fold to ensure that all forward-looking
        open-to-open targets are fully matured and observable by the training cutoff.

        Args:
            df: DataFrame containing date_col and features.
            date_col: Column name containing datetime stamps.
            purge_periods: Number of trailing trading dates to purge from train_fold
                           (defaults to self.purge_periods, 2).

        Returns:
            List of (train_df, test_df) tuples.
        """
        if df.empty:
            return []

        n_purge = self.purge_periods if purge_periods is None else int(purge_periods)

        df_sorted = df.copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
        df_sorted = df_sorted.sort_values(date_col).reset_index(drop=True)

        min_date = df_sorted[date_col].min()
        max_date = df_sorted[date_col].max()
        total_days = (max_date - min_date).days

        # Adaptive windowing for 180-day constrained data to guarantee at least 4-5 out-of-sample folds
        train_days = self.train_days
        test_days = self.test_days
        if total_days <= 200 and train_days > 75:
            train_days = 60
            test_days = 20
            logger.info(
                f"Data span is {total_days} days. Adapting walk-forward windows to "
                f"train={train_days}d, test={test_days}d to yield multiple out-of-sample folds."
            )

        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        current_train_start = min_date

        while True:
            train_end = current_train_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)

            if train_end >= max_date:
                break

            train_mask = (df_sorted[date_col] >= current_train_start) & (df_sorted[date_col] < train_end)
            test_mask = (df_sorted[date_col] >= train_end) & (df_sorted[date_col] < test_end)

            train_fold = df_sorted[train_mask].copy()
            test_fold = df_sorted[test_mask].copy()

            # Boundary Purge: Drop the final n_purge unique trading dates from train_fold
            # because targets for those dates require Open(t+1) and Open(t+2) prices that
            # mature on or after train_end (extending into test_fold).
            if n_purge > 0 and not train_fold.empty and not test_fold.empty:
                train_unique_dates = sorted(train_fold[date_col].unique())
                if len(train_unique_dates) > n_purge:
                    purge_cutoff_date = train_unique_dates[-n_purge]
                    train_fold = train_fold[train_fold[date_col] < purge_cutoff_date].copy()

            if not train_fold.empty and not test_fold.empty:
                splits.append((train_fold, test_fold))

            current_train_start += timedelta(days=test_days)

        logger.info(f"Generated {len(splits)} chronological walk-forward split folds (Purge: {n_purge} periods).")
        return splits

    def train_and_predict(
        self,
        feature_matrix: pd.DataFrame,
        target_col: str = "target",
        date_col: str = "event_date",
    ) -> Dict[str, Any]:
        """
        Execute walk-forward cross validation across feature matrix.
        For each split, trains model and records out-of-sample predictions.

        Args:
            feature_matrix: Preprocessed feature dataset.
            target_col: Name of target column (binary 0/1).
            date_col: Name of datetime stamp column.

        Returns:
            Dict containing:
              - 'predictions': DataFrame with date, actual, predicted, probability.
              - 'models': List of trained model instances.
              - 'feature_importances': Series of averaged feature importances across folds.
              - 'fold_metadata': Summary of date ranges and samples per fold.
        """
        splits = self.walk_forward_split(feature_matrix, date_col=date_col)
        if not splits:
            raise ValueError("Insufficient time history to construct any walk-forward train/test windows.")

        trained_models: List[Any] = []
        prediction_records: List[pd.DataFrame] = []
        fold_importances: List[pd.Series] = []
        fold_metadata: List[Dict[str, Any]] = []

        # Identify non-feature columns
        non_features = {
            target_col, date_col, "daily_return", "trade_return", "open_to_open_return",
            "close_price", "open_price", "entry_price", "exit_price", "asset_class", "surprise"
        }
        feature_cols = [c for c in feature_matrix.columns if c not in non_features and np.issubdtype(feature_matrix[c].dtype, np.number)]

        if not feature_cols:
            raise ValueError("No numeric feature columns found in feature_matrix.")

        current_positions: Optional[Dict[str, float]] = None

        for fold_idx, (train_df, test_df) in enumerate(splits, start=1):
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values.astype(int)

            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values.astype(int)

            # Skip fold if only 1 class in training set
            if len(np.unique(y_train)) < 2:
                logger.warning(f"Fold {fold_idx}: Only one class present in training fold. Skipping.")
                continue

            model = self._create_model()
            model.fit(X_train, y_train)
            trained_models.append(model)

            # Predictions
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred.astype(float)

            fold_preds = pd.DataFrame({
                date_col: test_df[date_col].values,
                "actual": y_test,
                "predicted": y_pred,
                "probability": y_prob,
            })

            # Retain execution prices, returns, asset_class, and surprise if present in test_df
            if "daily_return" in test_df.columns:
                fold_preds["daily_return"] = test_df["daily_return"].values
            if "trade_return" in test_df.columns:
                fold_preds["trade_return"] = test_df["trade_return"].values
            if "open_to_open_return" in test_df.columns:
                fold_preds["open_to_open_return"] = test_df["open_to_open_return"].values
            if "entry_price" in test_df.columns:
                fold_preds["entry_price"] = test_df["entry_price"].values
            if "exit_price" in test_df.columns:
                fold_preds["exit_price"] = test_df["exit_price"].values
            if "asset_class" in test_df.columns:
                fold_preds["asset_class"] = test_df["asset_class"].values
            if "surprise" in test_df.columns:
                fold_preds["surprise"] = test_df["surprise"].values

            fold_preds["fold"] = fold_idx
            prediction_records.append(fold_preds)

            # 1. Chained fold evaluation (carries continuous portfolio state across fold boundaries)
            f_metrics = self.calculate_trading_metrics(
                fold_preds,
                cost_bps=5.0,
                strategy_mode="long_short",
                initial_positions=current_positions,
            )
            current_positions = f_metrics.get("final_positions", None)

            # 2. Standalone fold evaluation (evaluating each fold starting independently from flat cash)
            f_standalone = self.calculate_trading_metrics(
                fold_preds,
                cost_bps=5.0,
                strategy_mode="long_short",
                initial_positions=None,
            )

            # 3. Calculate exact boundary friction cost incurred at this fold's start
            raw_b_cost = float(f_standalone.get("raw_costs_paid_pct", f_standalone.get("costs_paid_pct", 0.0)) - f_metrics.get("raw_costs_paid_pct", f_metrics.get("costs_paid_pct", 0.0)))
            f_boundary_cost = round(raw_b_cost, 6)

            f_acc = f_metrics.get("accuracy", 0.0)
            f_net = f_metrics.get("cumulative_return", 0.0)
            f_sharpe = f_metrics.get("sharpe_ratio", 0.0)
            raw_f_net = f_metrics.get("raw_cumulative_return", float(f_net))
            raw_f_standalone = f_standalone.get("raw_cumulative_return", float(f_standalone.get("cumulative_return", 0.0)))

            fold_metadata.append({
                "fold": fold_idx,
                "train_start": str(train_df[date_col].min())[:10],
                "train_end": str(train_df[date_col].max())[:10],
                "test_start": str(test_df[date_col].min())[:10],
                "test_end": str(test_df[date_col].max())[:10],
                "train_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
                "accuracy": round(f_acc, 4),
                "net_return": round(f_net, 4),
                "raw_net_return": raw_f_net,
                "sharpe_ratio": round(f_sharpe, 4),
                "standalone_net_return": round(float(f_standalone.get("cumulative_return", 0.0)), 4),
                "raw_standalone_net_return": raw_f_standalone,
                "boundary_friction_cost": f_boundary_cost,
                "raw_boundary_friction_cost": raw_b_cost,
                "assets": sorted(list(set(test_df["asset_class"].dropna().unique()))) if "asset_class" in test_df.columns else ["ALL"],
                "daily_net_returns": f_metrics.get("daily_net_returns", []),
                "standalone_daily_net_returns": f_standalone.get("daily_net_returns", []),
            })

            # Extract feature importance
            if hasattr(model, "feature_importances_"):
                fold_importances.append(pd.Series(model.feature_importances_, index=feature_cols))

        if not prediction_records:
            raise RuntimeError("Model training produced no valid out-of-sample predictions.")

        all_predictions = pd.concat(prediction_records, axis=0).reset_index(drop=True)

        # Average feature importances across all folds
        if fold_importances:
            avg_importances = pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        else:
            avg_importances = pd.Series(dtype=float)

        return {
            "predictions": all_predictions,
            "models": trained_models,
            "feature_importances": avg_importances,
            "fold_metadata": fold_metadata,
        }

    def calculate_trading_metrics(
        self,
        predictions_df: pd.DataFrame,
        daily_returns: Optional[pd.Series] = None,
        cost_bps: float = 5.0,
        risk_free_rate: float = 0.04,
        strategy_mode: str = "long_short",
        initial_positions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate classification performance and realistic net simulated trading strategy metrics.

        Trading execution & friction modeling:
          - 'long_short': Long when predicted == 1 (signal = +1), Short when predicted == 0 (signal = -1)
          - 'long_cash':  Long when predicted == 1 (signal = +1), Cash when predicted == 0 (signal = 0)
          - 'buy_hold':   Continuous long exposure (signal = +1)
          - Transaction costs & slippage: Deducted on position turnover (|signal_t - signal_{t-1}| * cost_bps)
          - Excess return: Net return minus daily risk-free rate (assumed 4% annual)
          - initial_positions: Optional dict of starting position weights per asset (defaults to cash 0.0).

        Metrics:
          - Accuracy, Precision, Recall, F1
          - Net Sharpe Ratio, Gross Sharpe Ratio
          - Sortino Ratio (downside deviation of excess returns)
          - Maximum Drawdown (peak-to-trough on net cumulative equity)
          - Win Rate, Profit Factor, Net Cumulative Return, Total Turnover
          - final_positions: Ending position state per asset for continuous multi-fold chaining.

        Args:
            predictions_df: DataFrame with 'actual' and 'predicted' columns.
            daily_returns: Optional Series of actual returns aligning with predictions_df.
            cost_bps: Transaction cost and slippage in basis points per 1.0 unit turnover (default 5 bps = 0.05%).
            risk_free_rate: Annualized risk-free rate for Sharpe/Sortino excess returns (default 4.0%).
            strategy_mode: 'long_short', 'long_cash', or 'buy_hold'.
            initial_positions: Optional dict mapping asset names to starting positions.

        Returns:
            Dict[str, Any]: Dictionary of trading performance metrics and final position state.
        """
        if predictions_df.empty:
            return {}

        y_true = predictions_df["actual"].values.astype(int)
        y_pred = predictions_df["predicted"].values.astype(int)

        # 1. Classification Metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # 2. Setup cost rates & risk-free rate
        cost_rate = float(cost_bps) / 10000.0
        rf_daily = float((1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0)

        date_col = next((c for c in ["event_date", "date", "calendar_date", "trade_date"] if c in predictions_df.columns), None)
        is_multi_asset = (
            "asset_class" in predictions_df.columns 
            and predictions_df["asset_class"].nunique() > 1
            and date_col is not None
        )

        if is_multi_asset:
            # Multi-asset portfolio evaluation: align by trading date with equal weights w_i = 1/N
            unique_dates = sorted(predictions_df[date_col].unique())
            assets = sorted(predictions_df["asset_class"].unique())
            weight = 1.0 / len(assets)
            prev_signals = dict(initial_positions) if initial_positions is not None else {asset: 0.0 for asset in assets}

            daily_records = []
            for dt in unique_dates:
                dt_df = predictions_df[predictions_df[date_col] == dt]
                day_gross = 0.0
                day_turnover = 0.0
                for asset in assets:
                    row = dt_df[dt_df["asset_class"] == asset]
                    if not row.empty:
                        yp = int(row["predicted"].values[0])
                        # Prioritize actual open-to-open return calculated from explicit entry and exit prices
                        if "entry_price" in row.columns and "exit_price" in row.columns and pd.notna(row["entry_price"].values[0]) and pd.notna(row["exit_price"].values[0]):
                            p_entry = float(row["entry_price"].values[0])
                            p_exit = float(row["exit_price"].values[0])
                            ret = (p_exit - p_entry) / p_entry if p_entry > 0 else 0.0
                        elif "trade_return" in row.columns and pd.notna(row["trade_return"].values[0]):
                            ret = float(row["trade_return"].values[0])
                        elif "open_to_open_return" in row.columns and pd.notna(row["open_to_open_return"].values[0]):
                            ret = float(row["open_to_open_return"].values[0])
                        elif "daily_return" in row.columns and pd.notna(row["daily_return"].values[0]):
                            ret = float(row["daily_return"].values[0])
                        else:
                            ret = 0.0
                        if strategy_mode == "buy_hold":
                            sig = 1.0
                        elif strategy_mode == "long_cash":
                            sig = 1.0 if yp == 1 else 0.0
                        else:  # 'long_short'
                            sig = 1.0 if yp == 1 else -1.0
                    else:
                        sig = prev_signals.get(asset, 0.0) if strategy_mode == "buy_hold" else 0.0
                        ret = 0.0

                    to = abs(sig - prev_signals[asset])
                    prev_signals[asset] = sig
                    day_gross += weight * sig * ret
                    day_turnover += weight * to

                day_friction = day_turnover * cost_rate
                day_net = day_gross - day_friction
                daily_records.append({
                    "date": dt,
                    "gross": day_gross,
                    "net": day_net,
                    "turnover": day_turnover,
                    "friction": day_friction,
                    "excess": day_net - rf_daily,
                })

            final_positions = {asset: prev_signals[asset] for asset in assets}
            p_df = pd.DataFrame(daily_records)
            gross_returns = p_df["gross"].values
            net_returns = p_df["net"].values
            excess_returns = p_df["excess"].values
            total_turnover = float(np.sum(p_df["turnover"]))
            total_friction_pct = float(np.sum(p_df["friction"]))
            total_portfolio_days = len(p_df)

        else:
            # Single asset evaluation (or multi-asset fallback when unindexed by date)
            if "entry_price" in predictions_df.columns and "exit_price" in predictions_df.columns and predictions_df["entry_price"].notna().any():
                p_entry = predictions_df["entry_price"].values.astype(float)
                p_exit = predictions_df["exit_price"].values.astype(float)
                actual_rets = np.where(p_entry > 0, (p_exit - p_entry) / p_entry, 0.0)
            elif "trade_return" in predictions_df.columns:
                actual_rets = predictions_df["trade_return"].values.astype(float)
            elif "open_to_open_return" in predictions_df.columns:
                actual_rets = predictions_df["open_to_open_return"].values.astype(float)
            elif daily_returns is not None and len(daily_returns) == len(predictions_df):
                actual_rets = daily_returns.values.astype(float)
            elif "daily_return" in predictions_df.columns:
                actual_rets = predictions_df["daily_return"].values.astype(float)
            else:
                actual_rets = np.where(y_true == 1, 0.01, -0.01)

            # Signal Formation
            if strategy_mode == "long_cash":
                signal = np.where(y_pred == 1, 1.0, 0.0)
            elif strategy_mode == "buy_hold":
                signal = np.ones(len(y_pred), dtype=float)
            else:  # 'long_short'
                signal = np.where(y_pred == 1, 1.0, -1.0)

            # Turnover & Friction
            if "asset_class" in predictions_df.columns and predictions_df["asset_class"].nunique() > 1:
                turnover_arr = np.zeros(len(signal), dtype=float)
                final_positions = {}
                for asset, grp_idx in predictions_df.groupby("asset_class").groups.items():
                    idx_arr = np.array(list(grp_idx))
                    sig_group = signal[idx_arr]
                    prev_sig = np.roll(sig_group, 1)
                    init_pos = initial_positions.get(asset, 0.0) if initial_positions else 0.0
                    prev_sig[0] = init_pos
                    turnover_arr[idx_arr] = np.abs(sig_group - prev_sig)
                    final_positions[asset] = float(sig_group[-1]) if len(sig_group) > 0 else 0.0
                turnover = turnover_arr
            else:
                prev_signal = np.roll(signal, 1)
                init_pos = initial_positions.get("ALL", 0.0) if initial_positions else (
                    list(initial_positions.values())[0] if initial_positions else 0.0
                )
                prev_signal[0] = init_pos
                turnover = np.abs(signal - prev_signal)
                final_positions = {"ALL": float(signal[-1]) if len(signal) > 0 else 0.0}

            total_turnover = float(np.sum(turnover))
            friction = turnover * cost_rate
            total_friction_pct = float(np.sum(friction))

            gross_returns = signal * actual_rets
            net_returns = gross_returns - friction
            excess_returns = net_returns - rf_daily
            total_portfolio_days = len(net_returns)

        # 5. Annualized Sharpe Ratio (Net and Gross)
        mean_excess = float(np.mean(excess_returns))
        std_excess = float(np.std(excess_returns))
        if std_excess > 1e-8:
            net_sharpe = float(mean_excess / std_excess * np.sqrt(252))
        else:
            net_sharpe = 0.0 if mean_excess <= 0 else 999.0

        mean_gross_excess = float(np.mean(gross_returns - rf_daily))
        std_gross = float(np.std(gross_returns))
        if std_gross > 1e-8:
            gross_sharpe = float(mean_gross_excess / std_gross * np.sqrt(252))
        else:
            gross_sharpe = 0.0 if mean_gross_excess <= 0 else 999.0

        # 6. Annualized Sortino Ratio (Downside Deviation on Net Excess Returns)
        downside = np.minimum(0.0, excess_returns)
        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        if downside_std > 1e-8:
            sortino = float(mean_excess / downside_std * np.sqrt(252))
        else:
            sortino = 0.0 if mean_excess <= 0 else 999.0

        # 7. Cumulative Equity Curve & Maximum Drawdown (MDD)
        cum_ret = np.cumprod(1.0 + net_returns)
        cum_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - cum_max) / cum_max
        max_drawdown = float(np.abs(np.min(drawdown))) if len(drawdown) > 0 else 0.0
        total_cum_return = float(cum_ret[-1] - 1.0) if len(cum_ret) > 0 else 0.0

        # 8. Win Rate & Profit Factor
        winning_trades = net_returns[net_returns > 0]
        losing_trades = net_returns[net_returns < 0]
        win_rate = float(len(winning_trades) / len(net_returns)) if len(net_returns) > 0 else 0.0

        sum_gains = float(np.sum(winning_trades))
        sum_losses = float(np.abs(np.sum(losing_trades)))
        profit_factor = (sum_gains / sum_losses) if sum_losses > 1e-8 else (sum_gains if sum_gains > 0 else 1.0)

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "sharpe_ratio": round(net_sharpe, 4),
            "gross_sharpe_ratio": round(gross_sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "cumulative_return": round(total_cum_return, 4),
            "raw_cumulative_return": float(total_cum_return),
            "total_turnover": round(total_turnover, 2),
            "raw_total_turnover": float(total_turnover),
            "cost_bps": float(cost_bps),
            "costs_paid_pct": round(total_friction_pct, 4),
            "raw_costs_paid_pct": float(total_friction_pct),
            "total_trades": int(total_portfolio_days),
            "total_predictions": int(len(predictions_df)),
            "final_positions": final_positions,
            "daily_net_returns": [float(r) for r in net_returns],
            "daily_gross_returns": [float(r) for r in gross_returns],
            "daily_turnover": [float(to) for to in p_df["turnover"]] if is_multi_asset else [float(to) for to in turnover],
            "daily_friction": [float(fr) for fr in p_df["friction"]] if is_multi_asset else [float(fr) for fr in friction],
            "daily_dates": [str(d)[:10] for d in p_df["date"]] if is_multi_asset else [],
        }

    def generate_strategy_matrix(
        self,
        predictions_df: pd.DataFrame,
        asset_classes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute strategy performance metrics individually per asset class.

        Args:
            predictions_df: Out-of-sample predictions containing asset_class.
            asset_classes: List of target asset classes (defaults to config.ASSET_CLASSES).

        Returns:
            pd.DataFrame: Matrix of asset_class x metrics.
        """
        assets = asset_classes or ASSET_CLASSES
        results = {}

        if "asset_class" in predictions_df.columns:
            for asset in assets:
                sub_df = predictions_df[predictions_df["asset_class"] == asset]
                if not sub_df.empty:
                    results[asset] = self.calculate_trading_metrics(sub_df, cost_bps=5.0)
                else:
                    results[asset] = self.calculate_trading_metrics(predictions_df, cost_bps=5.0)
        else:
            for asset in assets:
                results[asset] = self.calculate_trading_metrics(predictions_df, cost_bps=5.0)

        return pd.DataFrame.from_dict(results, orient="index")

    def compare_strategy_permutations(
        self,
        predictions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Evaluate multiple defined quantitative strategy configurations:
          1. BUY_AND_HOLD: Benchmark buy-and-hold exposure (5 bps entry from cash)
          2. MODEL_LONG_SHORT: Walk-forward model dynamic long/short (5 bps cost)
          3. MODEL_LONG_CASH:  Walk-forward model long/cash defensive (5 bps cost)
          4. MACRO_MOMENTUM:   Follow macro surprise direction (5 bps cost)
          5. MACRO_CONTRARIAN: Fade macro surprise direction (5 bps cost)

        Returns:
            pd.DataFrame: Comparison matrix of strategy permutations.
        """
        permutations: Dict[str, Dict[str, Any]] = {}

        # 1. Buy & Hold Benchmark (5 bps entry cost from cash)
        bnh = self.calculate_trading_metrics(predictions_df, cost_bps=5.0, strategy_mode="buy_hold")
        bnh["description"] = "Passive 100% long buy-and-hold benchmark (5 bps entry from cash)"
        permutations["BUY_AND_HOLD"] = bnh

        # 2. Model Dynamic Long/Short (5 bps cost)
        mls = self.calculate_trading_metrics(predictions_df, cost_bps=5.0, strategy_mode="long_short")
        mls["description"] = "Model-guided dynamic long/short with 5 bps turnover cost"
        permutations["MODEL_LONG_SHORT"] = mls

        # 3. Model Long / Cash Defensive (5 bps cost)
        mlc = self.calculate_trading_metrics(predictions_df, cost_bps=5.0, strategy_mode="long_cash")
        mlc["description"] = "Model-guided long or flat cash with 5 bps turnover cost"
        permutations["MODEL_LONG_CASH"] = mlc

        # 4. Macro Momentum Follow (if surprise is available in predictions_df)
        if "surprise" in predictions_df.columns or "surprise_mean" in predictions_df.columns:
            surp_col = "surprise" if "surprise" in predictions_df.columns else "surprise_mean"
            mom_preds = predictions_df.copy()
            mom_preds["predicted"] = (mom_preds[surp_col] > 0).astype(int)
            mm = self.calculate_trading_metrics(mom_preds, cost_bps=5.0, strategy_mode="long_short")
            mm["description"] = "Macro surprise momentum: long on positive surprise, short on negative"
            permutations["MACRO_MOMENTUM_FOLLOW"] = mm

            fade_preds = predictions_df.copy()
            fade_preds["predicted"] = (fade_preds[surp_col] <= 0).astype(int)
            mf = self.calculate_trading_metrics(fade_preds, cost_bps=5.0, strategy_mode="long_short")
            mf["description"] = "Macro surprise contrarian fade: short on positive surprise, long on negative"
            permutations["MACRO_CONTRARIAN_FADE"] = mf

        return pd.DataFrame.from_dict(permutations, orient="index")

    def run(
        self,
        feature_matrix: pd.DataFrame,
        target_col: str = "target",
        date_col: str = "event_date",
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end walk-forward modeling pipeline:
          1. Split feature matrix
          2. Train and generate test predictions
          3. Evaluate classification and trading metrics
          4. Compute asset-level strategy matrix

        Args:
            feature_matrix: Matrix of processed features.
            target_col: Target column name.
            date_col: Timestamp column name.

        Returns:
            Dict with all results, models, and evaluation metrics.
        """
        logger.info(f"Running WalkForwardModel pipeline ({self.model_type})...")

        tp_results = self.train_and_predict(
            feature_matrix=feature_matrix,
            target_col=target_col,
            date_col=date_col,
        )

        preds_df = tp_results["predictions"]
        overall_metrics = self.calculate_trading_metrics(preds_df)
        strategy_matrix = self.generate_strategy_matrix(preds_df)
        strategy_permutations = self.compare_strategy_permutations(preds_df)

        results = {
            "model_type": self.model_type,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "overall_metrics": overall_metrics,
            "metrics": overall_metrics,
            "strategy_matrix": strategy_matrix,
            "strategy_permutations": strategy_permutations,
            "fold_metadata": tp_results.get("fold_metadata", []),
            "feature_importances": tp_results["feature_importances"],
            "predictions": preds_df,
            "models": tp_results["models"],
        }

        logger.info(
            f"Walk-forward run complete. Net Sharpe: {overall_metrics.get('sharpe_ratio')}, "
            f"Gross Sharpe: {overall_metrics.get('gross_sharpe_ratio')}, "
            f"Sortino: {overall_metrics.get('sortino_ratio')}, "
            f"Max DD: {overall_metrics.get('max_drawdown')}, "
            f"Folds: {len(tp_results.get('fold_metadata', []))}"
        )
        return results

    def save_model_results(self, results: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """
        Serialize results dictionary to a clean JSON report.

        Args:
            results: Results dictionary from run().
            filepath: Destination path for the JSON file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable: Dict[str, Any] = {
            "model_type": results.get("model_type", self.model_type),
            "train_days": results.get("train_days", self.train_days),
            "test_days": results.get("test_days", self.test_days),
            "overall_metrics": results.get("overall_metrics", {}),
            "fold_metadata": results.get("fold_metadata", []),
        }

        # Handle strategy_matrix
        strat_mat = results.get("strategy_matrix")
        if isinstance(strat_mat, pd.DataFrame):
            serializable["strategy_matrix"] = strat_mat.to_dict(orient="index")
        elif isinstance(strat_mat, dict):
            serializable["strategy_matrix"] = strat_mat

        # Handle strategy_permutations
        strat_perm = results.get("strategy_permutations")
        if isinstance(strat_perm, pd.DataFrame):
            serializable["strategy_permutations"] = strat_perm.to_dict(orient="index")
        elif isinstance(strat_perm, dict):
            serializable["strategy_permutations"] = strat_perm

        # Handle feature importances
        fi = results.get("feature_importances")
        if isinstance(fi, pd.Series):
            serializable["feature_importances"] = fi.to_dict()
        elif isinstance(fi, dict):
            serializable["feature_importances"] = fi

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Model results saved successfully to {path}")
