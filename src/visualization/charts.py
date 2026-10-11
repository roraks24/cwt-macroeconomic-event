"""
Macro Sentiment Charts Module for CWT Macro Economics Sentiment Agent.
Produces publication-quality interactive and static charts for macro events,
sentiment timelines, strategy performances, and regime analysis.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server execution
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Ensure base directory is in sys.path to import project config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ASSET_CLASSES, CHARTS_DIR, MACRO_EVENTS, REPORTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)


class MacroSentimentCharts:
    """
    Visualization engine generating macroeconomic, sentiment, and strategy performance charts.
    Supports both static PNG (via Matplotlib/Seaborn) and interactive HTML (via Plotly).
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the chart generator with destination directory and visual styles.

        Args:
            output_dir: Destination folder path for charts. Defaults to CHARTS_DIR from config.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(CHARTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._configure_styles()

    def _configure_styles(self) -> None:
        """Set up professional dark theme styling for matplotlib and seaborn."""
        style_candidates = ["seaborn-v0_8-darkgrid", "seaborn-darkgrid", "dark_background"]
        for style in style_candidates:
            if style in plt.style.available:
                plt.style.use(style)
                break

        # Custom dark-theme palette and font properties
        plt.rcParams.update({
            "figure.facecolor": "#111726",
            "axes.facecolor": "#1a2234",
            "axes.edgecolor": "#2d3748",
            "axes.labelcolor": "#e2e8f0",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "grid.color": "#2d3748",
            "grid.alpha": 0.6,
            "figure.autolayout": True,
            "font.family": "sans-serif",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
        })

    def _save_plotly(self, fig: go.Figure, filename_stem: str) -> str:
        """
        Save a Plotly figure as HTML and attempt to export PNG.
        Returns the primary output filepath (PNG if created, else HTML).
        """
        html_path = self.output_dir / f"{filename_stem}.html"
        png_path = self.output_dir / f"{filename_stem}.png"

        try:
            fig.write_html(str(html_path), include_plotlyjs="cdn")
        except Exception as e:
            logger.warning("Failed to save HTML chart %s: %s", html_path, e)

        saved_png = False
        try:
            fig.write_image(str(png_path), width=1200, height=650, scale=2)
            saved_png = True
        except Exception as e:
            logger.info("Plotly static image export skipped or unavailable for %s: %s", png_path.name, e)

        return str(png_path if saved_png else html_path)

    def plot_sentiment_timeline(
        self,
        sentiment_df: pd.DataFrame,
        title: str = "Macro Sentiment Over Time"
    ) -> str:
        """
        Plot rolling sentiment score over time with color-coded positive and negative zones.
        Saves as both HTML and PNG.

        Args:
            sentiment_df: DataFrame with date column and sentiment score column.
            title: Title for the chart.

        Returns:
            Saved file path.
        """
        if sentiment_df.empty:
            logger.warning("sentiment_df is empty, generating empty timeline.")
            fig = go.Figure()
            fig.update_layout(title=f"{title} (No Data Available)")
            return self._save_plotly(fig, "sentiment_timeline")

        df = sentiment_df.copy()

        # Identify date column
        date_col = None
        for col in ["date", "event_date", "published_date", "timestamp"]:
            if col in df.columns:
                date_col = col
                break
        if not date_col:
            date_col = df.columns[0]

        # Identify sentiment score column
        score_col = None
        for col in ["sentiment_score", "sentiment", "score", "macro_sentiment"]:
            if col in df.columns:
                score_col = col
                break
        if not score_col:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            score_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[-1]

        df[date_col] = pd.to_datetime(df[date_col])
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        df = df.sort_values(by=date_col)

        # Rolling 7-day or 5-period average if sufficient points
        rolling_col = f"{score_col}_rolling"
        window = min(7, max(3, len(df) // 10)) if len(df) > 10 else 1
        df[rolling_col] = df[score_col].rolling(window=window, min_periods=1).mean()

        y_min = min(float(df[score_col].min()), -1.0)
        y_max = max(float(df[score_col].max()), 1.0)
        y_pad = (y_max - y_min) * 0.1
        y_min -= y_pad
        y_max += y_pad

        fig = go.Figure()

        # Positive / negative background color zones
        fig.add_hrect(
            y0=0,
            y1=y_max,
            fillcolor="rgba(34, 197, 94, 0.12)",
            layer="below",
            line_width=0,
            annotation_text="Positive Sentiment Zone",
            annotation_position="top left",
            annotation_font=dict(color="#22c55e", size=10),
        )
        fig.add_hrect(
            y0=y_min,
            y1=0,
            fillcolor="rgba(239, 68, 68, 0.12)",
            layer="below",
            line_width=0,
            annotation_text="Negative Sentiment Zone",
            annotation_position="bottom left",
            annotation_font=dict(color="#ef4444", size=10),
        )

        # Zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1.5)

        # Raw scores
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[score_col],
                mode="markers",
                name="Daily Sentiment",
                marker=dict(
                    color=df[score_col],
                    colorscale="RdYlGn",
                    cmin=-1,
                    cmax=1,
                    size=6,
                    opacity=0.6,
                ),
            )
        )

        # Rolling trend line
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[rolling_col],
                mode="lines",
                name=f"Trend ({window}-Period Rolling)",
                line=dict(color="#38bdf8", width=3),
            )
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#f8fafc")),
            paper_bgcolor="#111726",
            plot_bgcolor="#1a2234",
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="#2d3748",
                color="#cbd5e1",
            ),
            yaxis=dict(
                title="Sentiment Score",
                showgrid=True,
                gridcolor="#2d3748",
                color="#cbd5e1",
                range=[y_min, y_max],
            ),
            legend=dict(
                font=dict(color="#f8fafc"),
                bgcolor="rgba(26, 34, 52, 0.8)",
                bordercolor="#2d3748",
            ),
            margin=dict(l=60, r=40, t=60, b=50),
        )

        return self._save_plotly(fig, "macro_sentiment_timeline")

    def plot_surprise_heatmap(
        self,
        events_df: pd.DataFrame,
        title: str = "Economic Surprise Heatmap"
    ) -> str:
        """
        Seaborn/matplotlib heatmap where rows = event types, columns = months, values = surprise magnitude.
        Color-coded red (negative surprise) to green (positive surprise).

        Args:
            events_df: DataFrame with event names, dates, and surprise values.
            title: Plot title.

        Returns:
            Saved PNG file path.
        """
        out_path = self.output_dir / "economic_surprise_heatmap.png"
        fig, ax = plt.subplots(figsize=(14, 8), facecolor="#111726")
        ax.set_facecolor("#1a2234")

        if events_df.empty:
            ax.text(0.5, 0.5, "No Economic Event Surprise Data Available",
                    ha="center", va="center", color="#94a3b8", fontsize=14)
            ax.set_title(title, color="#f8fafc", pad=20)
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            return str(out_path)

        df = events_df.copy()

        # Date normalization
        date_col = next((c for c in ["event_date", "date", "created_at"] if c in df.columns), None)
        event_col = next((c for c in ["event_name", "event", "name", "indicator"] if c in df.columns), None)
        surprise_col = next((c for c in ["surprise", "surprise_value", "surprise_magnitude"] if c in df.columns), None)

        if not surprise_col and "actual_value" in df.columns and "forecast_value" in df.columns:
            df["surprise"] = pd.to_numeric(df["actual_value"], errors="coerce") - pd.to_numeric(df["forecast_value"], errors="coerce")
            surprise_col = "surprise"

        if not date_col or not event_col or not surprise_col:
            ax.text(0.5, 0.5, "Required columns missing for Surprise Heatmap",
                    ha="center", va="center", color="#ef4444", fontsize=14)
            ax.set_title(title, color="#f8fafc", pad=20)
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            return str(out_path)

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, surprise_col, event_col])
        df["month_year"] = df[date_col].dt.to_period("M").astype(str)
        df[surprise_col] = pd.to_numeric(df[surprise_col], errors="coerce")

        # Pivot to Event x Month matrix
        pivot = df.pivot_table(
            index=event_col,
            columns="month_year",
            values=surprise_col,
            aggfunc="mean"
        )

        if pivot.empty:
            ax.text(0.5, 0.5, "Insufficient observations for heatmap matrix",
                    ha="center", va="center", color="#94a3b8", fontsize=14)
        else:
            # Sort columns chronologically
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)

            # Cap values for robust visual contrast
            abs_max = max(float(np.nanpercentile(np.abs(pivot.values), 95)), 1.0) if not np.all(np.isnan(pivot.values)) else 1.0

            cmap = sns.diverging_palette(10, 140, as_cmap=True)  # Red to Green
            sns.heatmap(
                pivot,
                cmap=cmap,
                center=0,
                vmin=-abs_max,
                vmax=abs_max,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                linecolor="#111726",
                cbar_kws={"label": "Surprise Magnitude (Actual - Forecast)"},
                ax=ax,
            )
            # Format heatmap colorbar text
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.label.set_color("#e2e8f0")
            cbar.ax.tick_params(colors="#cbd5e1")

        ax.set_title(title, color="#f8fafc", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Month", color="#cbd5e1", labelpad=10)
        ax.set_ylabel("Macroeconomic Event", color="#cbd5e1", labelpad=10)
        plt.xticks(rotation=45, ha="right", color="#cbd5e1")
        plt.yticks(rotation=0, color="#cbd5e1")

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def plot_strategy_performance(
        self,
        results_df: pd.DataFrame,
        title: str = "Strategy Performance"
    ) -> str:
        """
        Plotly chart with cumulative returns over time for each asset class strategy,
        plus an integrated drawdown subplot below. Saves as HTML and PNG.

        Args:
            results_df: DataFrame with date, cumulative returns or returns per asset class.
            title: Chart title.

        Returns:
            Saved file path.
        """
        if results_df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (No Results)")
            return self._save_plotly(fig, "strategy_performance")

        df = results_df.copy()
        date_col = next((c for c in ["date", "timestamp", "trade_date"] if c in df.columns), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Cumulative Strategy Returns", "Portfolio Drawdown (%)"),
            row_heights=[0.7, 0.3],
        )

        palette = [
            "#38bdf8", "#4ade80", "#f59e0b", "#ec4899",
            "#a855f7", "#6366f1", "#14b8a6", "#fb7185"
        ]

        # Determine asset columns or grouped structure
        if "asset_class" in df.columns and "return" in df.columns:
            asset_groups = df.groupby("asset_class")
            for idx, (asset, group) in enumerate(asset_groups):
                color = palette[idx % len(palette)]
                group = group.sort_values(by=date_col)
                cum_ret = (1 + group["return"]).cumprod() - 1
                running_max = cum_ret.cummax()
                drawdown = (cum_ret - running_max) / (1 + running_max) * 100

                fig.add_trace(
                    go.Scatter(
                        x=group[date_col],
                        y=cum_ret * 100,
                        mode="lines",
                        name=f"{asset}",
                        line=dict(color=color, width=2.2),
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=group[date_col],
                        y=drawdown,
                        mode="lines",
                        name=f"{asset} DD",
                        showlegend=False,
                        line=dict(color=color, width=1.5),
                        fill="tozeroy",
                        fillcolor=f"rgba{matplotlib.colors.to_rgba(color, 0.15)}",
                    ),
                    row=2, col=1,
                )
        else:
            # Columns representing different strategies/assets
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != date_col]
            for idx, col in enumerate(numeric_cols):
                color = palette[idx % len(palette)]
                series = df[col]
                # If values are not already cumulative, convert or check
                if (series.abs().max() < 1.0) and not (series > 0).all():
                    cum_ret = (1 + series).cumprod() - 1
                else:
                    cum_ret = series / series.iloc[0] - 1 if series.iloc[0] != 0 else series

                running_max = cum_ret.cummax()
                drawdown = (cum_ret - running_max) / (1 + np.maximum(running_max, 0)) * 100

                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=cum_ret * 100,
                        mode="lines",
                        name=str(col),
                        line=dict(color=color, width=2.2),
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=drawdown,
                        mode="lines",
                        name=f"{col} DD",
                        showlegend=False,
                        line=dict(color=color, width=1.2),
                        fill="tozeroy",
                    ),
                    row=2, col=1,
                )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#f8fafc")),
            paper_bgcolor="#111726",
            plot_bgcolor="#1a2234",
            legend=dict(
                font=dict(color="#f8fafc"),
                bgcolor="rgba(26, 34, 52, 0.8)",
                bordercolor="#2d3748",
            ),
            margin=dict(l=60, r=40, t=70, b=50),
        )
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1, gridcolor="#2d3748", color="#cbd5e1")
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, gridcolor="#2d3748", color="#cbd5e1")
        fig.update_xaxes(gridcolor="#2d3748", color="#cbd5e1")

        return self._save_plotly(fig, "strategy_performance")

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 20,
        title: str = "Top Feature Importances"
    ) -> str:
        """
        Horizontal bar chart of top N features by importance. Saves as PNG.

        Args:
            importance_dict: Dictionary mapping feature names to importance scores.
            top_n: Maximum features to display.
            title: Chart title.

        Returns:
            Saved PNG file path.
        """
        out_path = self.output_dir / "feature_importance.png"
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#111726")
        ax.set_facecolor("#1a2234")

        if not importance_dict:
            ax.text(0.5, 0.5, "No Feature Importance Data Available",
                    ha="center", va="center", color="#94a3b8", fontsize=14)
            ax.set_title(title, color="#f8fafc", pad=20)
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            return str(out_path)

        # Sort and take top N
        sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [x[0] for x in reversed(sorted_feats)]
        scores = [x[1] for x in reversed(sorted_feats)]

        colors = sns.color_palette("viridis", len(scores))
        bars = ax.barh(features, scores, color=colors, edgecolor="#2d3748", height=0.65)

        # Annotate values at the bar tips
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (max(scores) * 0.01 if scores else 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center",
                ha="left",
                color="#cbd5e1",
                fontsize=9,
            )

        ax.set_title(f"{title} (Top {len(features)})", color="#f8fafc", fontsize=15, weight="bold", pad=15)
        ax.set_xlabel("Importance Metric", color="#cbd5e1", labelpad=10)
        ax.set_ylabel("Macro & Sentiment Feature", color="#cbd5e1", labelpad=10)
        ax.grid(axis="x", color="#2d3748", linestyle="--", alpha=0.7)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def plot_walk_forward_results(
        self,
        predictions_df: pd.DataFrame,
        title: str = "Walk-Forward Predictions"
    ) -> str:
        """
        Time series plot showing actual vs predicted signals over time,
        with accuracy per fold annotated. Saves as PNG.

        Args:
            predictions_df: DataFrame with actual, predicted, date, and fold columns.
            title: Chart title.

        Returns:
            Saved PNG file path.
        """
        out_path = self.output_dir / "walk_forward_predictions.png"
        fig, ax = plt.subplots(figsize=(13, 6), facecolor="#111726")
        ax.set_facecolor("#1a2234")

        if predictions_df.empty:
            ax.text(0.5, 0.5, "No Walk-Forward Prediction Data Available",
                    ha="center", va="center", color="#94a3b8", fontsize=14)
            ax.set_title(title, color="#f8fafc", pad=20)
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            return str(out_path)

        df = predictions_df.copy()
        date_col = next((c for c in ["date", "timestamp", "event_date"] if c in df.columns), df.columns[0])
        actual_col = next((c for c in ["actual", "y_true", "target"] if c in df.columns), None)
        pred_col = next((c for c in ["predicted", "pred", "y_pred"] if c in df.columns), None)
        fold_col = next((c for c in ["fold", "fold_id", "split"] if c in df.columns), None)

        if not actual_col or not pred_col:
            ax.text(0.5, 0.5, "Predictions DataFrame missing 'actual' or 'predicted' columns",
                    ha="center", va="center", color="#ef4444", fontsize=13)
            ax.set_title(title, color="#f8fafc", pad=20)
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            return str(out_path)

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)

        # Plot actual vs predicted
        ax.plot(df[date_col], df[actual_col], label="Actual Signal", color="#38bdf8", lw=2, alpha=0.9)
        ax.plot(df[date_col], df[pred_col], label="Predicted Signal", color="#f59e0b", lw=1.8, linestyle="--", alpha=0.9)

        # Annotate folds if present
        if fold_col and fold_col in df.columns:
            folds = df.groupby(fold_col)
            for fold_name, fgroup in folds:
                fgroup_sorted = fgroup.sort_values(by=date_col)
                start_d = fgroup_sorted[date_col].iloc[0]
                end_d = fgroup_sorted[date_col].iloc[-1]
                mid_d = start_d + (end_d - start_d) / 2

                # Compute fold accuracy or correlation
                is_correct = (fgroup[actual_col] == fgroup[pred_col])
                acc = is_correct.mean() * 100 if len(is_correct) > 0 else 0.0

                ax.axvline(start_d, color="#475569", linestyle=":", alpha=0.8)
                y_pos = ax.get_ylim()[1] if ax.get_ylim()[1] != ax.get_ylim()[0] else 1.0
                ax.annotate(
                    f"Fold {fold_name}\nAcc: {acc:.1f}%",
                    xy=(mid_d, y_pos * 0.88),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec="#38bdf8", alpha=0.85),
                    color="#f8fafc",
                    fontsize=8.5,
                )

        overall_acc = (df[actual_col] == df[pred_col]).mean() * 100
        ax.set_title(f"{title} (Overall Accuracy: {overall_acc:.1f}%)", color="#f8fafc", fontsize=15, weight="bold", pad=15)
        ax.set_xlabel("Date", color="#cbd5e1", labelpad=10)
        ax.set_ylabel("Signal", color="#cbd5e1", labelpad=10)
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")
        ax.grid(True, color="#2d3748", alpha=0.6)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def plot_macro_regime_transitions(
        self,
        regime_df: pd.DataFrame,
        title: str = "Macro Regime Transitions"
    ) -> str:
        """
        Stacked area chart or state diagram showing macroeconomic regime transitions over time.
        (e.g., Inflationary / Deflationary, Growth / Contraction). Saves as PNG.

        Args:
            regime_df: DataFrame with dates and regime probabilities or categories.
            title: Chart title.

        Returns:
            Saved PNG file path.
        """
        out_path = self.output_dir / "macro_regime_transitions.png"
        fig, ax = plt.subplots(figsize=(13, 6), facecolor="#111726")
        ax.set_facecolor("#1a2234")

        if regime_df.empty:
            ax.text(0.5, 0.5, "No Macro Regime Data Available",
                    ha="center", va="center", color="#94a3b8", fontsize=14)
            ax.set_title(title, color="#f8fafc", pad=20)
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            return str(out_path)

        df = regime_df.copy()
        date_col = next((c for c in ["date", "timestamp", "period"] if c in df.columns), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)

        # Check if columns represent probability distributions across regimes
        regime_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]

        if len(regime_cols) >= 2:
            # Stacked area chart of probabilities
            palette = ["#22c55e", "#eab308", "#ef4444", "#3b82f6"]
            colors = [palette[i % len(palette)] for i in range(len(regime_cols))]

            # Normalize to 100% stack if needed
            row_sums = df[regime_cols].sum(axis=1)
            row_sums = row_sums.replace(0, 1)
            norm_df = df[regime_cols].div(row_sums, axis=0) * 100

            ax.stackplot(
                df[date_col],
                norm_df.T,
                labels=regime_cols,
                colors=colors,
                alpha=0.85,
                edgecolor="#1e293b",
                linewidth=0.5,
            )
            ax.set_ylabel("Regime Probability (%)", color="#cbd5e1", labelpad=10)
            ax.set_ylim(0, 100)
            ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")
        else:
            # Categorical state sequence
            cat_col = regime_cols[0] if regime_cols else df.columns[-1]
            unique_states = df[cat_col].unique()
            state_map = {s: i for i, s in enumerate(unique_states)}
            df["state_code"] = df[cat_col].map(state_map)

            ax.step(df[date_col], df["state_code"], where="mid", color="#38bdf8", lw=2.5)
            ax.fill_between(df[date_col], df["state_code"], step="mid", alpha=0.25, color="#38bdf8")
            ax.set_yticks(list(state_map.values()))
            ax.set_yticklabels(list(state_map.keys()), color="#cbd5e1")
            ax.set_ylabel("Macro Regime State", color="#cbd5e1", labelpad=10)

        ax.set_title(title, color="#f8fafc", fontsize=15, weight="bold", pad=15)
        ax.set_xlabel("Date", color="#cbd5e1", labelpad=10)
        ax.grid(True, color="#2d3748", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def plot_correlation_matrix(
        self,
        feature_df: pd.DataFrame,
        title: str = "Feature Correlation Matrix"
    ) -> str:
        """
        Seaborn heatmap of feature correlations. Saves as PNG.

        Args:
            feature_df: DataFrame with numeric features.
            title: Chart title.

        Returns:
            Saved PNG file path.
        """
        out_path = self.output_dir / "feature_correlation_matrix.png"
        fig, ax = plt.subplots(figsize=(14, 8), facecolor="#111726")
        ax.set_facecolor("#1a2234")

        numeric_df = feature_df.select_dtypes(include=[np.number]) if not feature_df.empty else pd.DataFrame()

        if numeric_df.empty or numeric_df.shape[1] < 2:
            ax.text(0.5, 0.5, "Insufficient numeric features for Correlation Matrix",
                    ha="center", va="center", color="#94a3b8", fontsize=14)
            ax.set_title(title, color="#f8fafc", pad=20)
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            return str(out_path)

        # Cap features to top 25 to preserve readability
        if numeric_df.shape[1] > 25:
            numeric_df = numeric_df.iloc[:, :25]

        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=corr.shape[1] <= 15,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            linecolor="#111726",
            cbar_kws={"label": "Pearson Correlation Coefficient", "shrink": 0.8},
            ax=ax,
        )

        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.yaxis.label.set_color("#e2e8f0")
            cbar.ax.tick_params(colors="#cbd5e1")

        ax.set_title(title, color="#f8fafc", fontsize=16, weight="bold", pad=20)
        plt.xticks(rotation=45, ha="right", color="#cbd5e1")
        plt.yticks(rotation=0, color="#cbd5e1")

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def generate_all_charts(self, data: Dict[str, Any]) -> List[str]:
        """
        Takes a dict with data structures and triggers all plot routines available.

        Expected keys in data:
            - 'sentiment_df': DataFrame
            - 'events_df': DataFrame
            - 'results_df': DataFrame
            - 'importance_dict': dict
            - 'predictions_df': DataFrame
            - 'regime_df': DataFrame
            - 'feature_df': DataFrame

        Returns:
            List of generated chart file paths.
        """
        generated_paths: List[str] = []

        # 1. Sentiment timeline
        if "sentiment_df" in data and isinstance(data["sentiment_df"], pd.DataFrame):
            try:
                path = self.plot_sentiment_timeline(data["sentiment_df"])
                generated_paths.append(path)
            except Exception as e:
                logger.error("Failed to generate sentiment timeline: %s", e)

        # 2. Economic surprise heatmap
        if "events_df" in data and isinstance(data["events_df"], pd.DataFrame):
            try:
                path = self.plot_surprise_heatmap(data["events_df"])
                generated_paths.append(path)
            except Exception as e:
                logger.error("Failed to generate surprise heatmap: %s", e)

        # 3. Strategy performance
        if "results_df" in data and isinstance(data["results_df"], pd.DataFrame):
            try:
                path = self.plot_strategy_performance(data["results_df"])
                generated_paths.append(path)
            except Exception as e:
                logger.error("Failed to generate strategy performance: %s", e)

        # 4. Feature importance
        if "importance_dict" in data and isinstance(data["importance_dict"], dict):
            try:
                path = self.plot_feature_importance(data["importance_dict"])
                generated_paths.append(path)
            except Exception as e:
                logger.error("Failed to generate feature importance: %s", e)

        # 5. Walk-forward predictions
        if "predictions_df" in data and isinstance(data["predictions_df"], pd.DataFrame):
            try:
                path = self.plot_walk_forward_results(data["predictions_df"])
                generated_paths.append(path)
            except Exception as e:
                logger.error("Failed to generate walk-forward results: %s", e)

        # 6. Macro regime transitions
        if "regime_df" in data and isinstance(data["regime_df"], pd.DataFrame):
            try:
                path = self.plot_macro_regime_transitions(data["regime_df"])
                generated_paths.append(path)
            except Exception as e:
                logger.error("Failed to generate macro regime transitions: %s", e)

        # 7. Correlation matrix
        if "feature_df" in data and isinstance(data["feature_df"], pd.DataFrame):
            try:
                path = self.plot_correlation_matrix(data["feature_df"])
                generated_paths.append(path)
            except Exception as e:
                logger.error("Failed to generate correlation matrix: %s", e)

        return generated_paths
