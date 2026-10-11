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
    ) -> None:
        """
        Initialize the WalkForwardModel.

        Args:
            model_type: 'lightgbm' or 'xgboost'.
            train_days: Number of calendar days for the rolling training window.
            test_days: Number of calendar days for the sequential test window.
        """
        self.model_type = model_type.lower()
        self.train_days = int(train_days)
        self.test_days = int(test_days)

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
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate sequential walk-forward train and test splits.
        Starting from earliest date + train_days, creates sequential windows
        where train = prior train_days of data, test = next test_days.
        Slides forward by test_days each iteration.

        Args:
            df: DataFrame containing date_col and features.
            date_col: Column name containing datetime stamps.

        Returns:
            List of (train_df, test_df) tuples.
        """
        if df.empty:
            return []

        df_sorted = df.copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
        df_sorted = df_sorted.sort_values(date_col).reset_index(drop=True)

        min_date = df_sorted[date_col].min()
        max_date = df_sorted[date_col].max()

        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        current_train_start = min_date

        while True:
            train_end = current_train_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)

            if train_end >= max_date:
                break

            train_mask = (df_sorted[date_col] >= current_train_start) & (df_sorted[date_col] < train_end)
            test_mask = (df_sorted[date_col] >= train_end) & (df_sorted[date_col] < test_end)

            train_fold = df_sorted[train_mask]
            test_fold = df_sorted[test_mask]

            if not train_fold.empty and not test_fold.empty:
                splits.append((train_fold.copy(), test_fold.copy()))

            current_train_start += timedelta(days=self.test_days)

        logger.info(f"Generated {len(splits)} walk-forward split folds.")
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
        """
        splits = self.walk_forward_split(feature_matrix, date_col=date_col)
        if not splits:
            raise ValueError("Insufficient time history to construct any walk-forward train/test windows.")

        trained_models: List[Any] = []
        prediction_records: List[pd.DataFrame] = []
        fold_importances: List[pd.Series] = []

        # Identify non-feature columns
        non_features = {target_col, date_col, "daily_return", "close_price", "asset_class"}
        feature_cols = [c for c in feature_matrix.columns if c not in non_features and np.issubdtype(feature_matrix[c].dtype, np.number)]

        if not feature_cols:
            raise ValueError("No numeric feature columns found in feature_matrix.")

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

            # Retain daily_return and asset_class if present in test_df
            if "daily_return" in test_df.columns:
                fold_preds["daily_return"] = test_df["daily_return"].values
            if "asset_class" in test_df.columns:
                fold_preds["asset_class"] = test_df["asset_class"].values

            fold_preds["fold"] = fold_idx
            prediction_records.append(fold_preds)

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
        }

    def calculate_trading_metrics(
        self,
        predictions_df: pd.DataFrame,
        daily_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Calculate classification performance and simulated trading strategy metrics.

        Trading rule:
          - Go long when predicted == 1 (signal = +1)
          - Go short when predicted == 0 (signal = -1)
          - strategy_return = signal * actual_daily_return

        Metrics:
          - Accuracy, Precision, Recall, F1
          - Sharpe Ratio, Sortino Ratio, Max Drawdown
          - Win Rate, Profit Factor, Cumulative Return

        Args:
            predictions_df: DataFrame with 'actual' and 'predicted' columns.
            daily_returns: Optional Series of actual returns aligning with predictions_df.

        Returns:
            Dict[str, float]: Dictionary of trading performance metrics.
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

        # 2. Daily returns alignment
        if daily_returns is not None and len(daily_returns) == len(predictions_df):
            actual_rets = daily_returns.values.astype(float)
        elif "daily_return" in predictions_df.columns:
            actual_rets = predictions_df["daily_return"].values.astype(float)
        else:
            # Fallback return approximation: +1% on positive day, -1% on negative day
            actual_rets = np.where(y_true == 1, 0.01, -0.01)

        # Strategy return simulation: Long = +1, Short = -1
        signal = np.where(y_pred == 1, 1.0, -1.0)
        strategy_returns = signal * actual_rets

        # 3. Financial Metrics
        mean_ret = float(np.mean(strategy_returns))
        std_ret = float(np.std(strategy_returns))

        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0.0

        # Downside Deviation for Sortino
        neg_rets = strategy_returns[strategy_returns < 0]
        downside_std = float(np.std(neg_rets)) if len(neg_rets) > 0 else 0.0
        sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 1e-8 else 0.0

        # Cumulative Return & Max Drawdown
        cum_ret = np.cumprod(1.0 + strategy_returns)
        cum_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - cum_max) / cum_max
        max_drawdown = float(np.abs(np.min(drawdown))) if len(drawdown) > 0 else 0.0
        total_cum_return = float(cum_ret[-1] - 1.0) if len(cum_ret) > 0 else 0.0

        # Win Rate & Profit Factor
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        win_rate = float(len(winning_trades) / len(strategy_returns)) if len(strategy_returns) > 0 else 0.0

        sum_gains = float(np.sum(winning_trades))
        sum_losses = float(np.abs(np.sum(losing_trades)))
        profit_factor = (sum_gains / sum_losses) if sum_losses > 1e-8 else (sum_gains if sum_gains > 0 else 1.0)

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "cumulative_return": round(total_cum_return, 4),
            "total_trades": int(len(strategy_returns)),
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
                    results[asset] = self.calculate_trading_metrics(sub_df)
                else:
                    results[asset] = self.calculate_trading_metrics(predictions_df)
        else:
            for asset in assets:
                results[asset] = self.calculate_trading_metrics(predictions_df)

        return pd.DataFrame.from_dict(results, orient="index")

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

        results = {
            "model_type": self.model_type,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "overall_metrics": overall_metrics,
            "metrics": overall_metrics,
            "strategy_matrix": strategy_matrix,
            "feature_importances": tp_results["feature_importances"],
            "predictions": preds_df,
            "models": tp_results["models"],
        }

        logger.info(f"Walk-forward run complete. Sharpe: {overall_metrics.get('sharpe_ratio')}, Acc: {overall_metrics.get('accuracy')}")
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
        }

        # Handle strategy_matrix
        strat_mat = results.get("strategy_matrix")
        if isinstance(strat_mat, pd.DataFrame):
            serializable["strategy_matrix"] = strat_mat.to_dict(orient="index")
        elif isinstance(strat_mat, dict):
            serializable["strategy_matrix"] = strat_mat

        # Handle feature importances
        fi = results.get("feature_importances")
        if isinstance(fi, pd.Series):
            serializable["feature_importances"] = fi.to_dict()
        elif isinstance(fi, dict):
            serializable["feature_importances"] = fi

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Model results saved successfully to {path}")
