from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.visualization.charts import MacroSentimentCharts


@pytest.fixture
def mock_viz_fixtures():
    """Create mock data fixtures matching expected schemas across all chart methods."""
    dates = pd.date_range("2026-03-01", periods=60, freq="D")
    np.random.seed(42)

    # 1. Market DataFrame (SPY / US_EQUITY)
    prices = [100.0]
    for _ in range(59):
        prices.append(prices[-1] * (1.0 + np.random.normal(0.0005, 0.01)))
    
    market_df = pd.DataFrame({
        "date": dates,
        "asset_class": "US_EQUITY",
        "open_price": prices,
        "high_price": [p * 1.01 for p in prices],
        "low_price": [p * 0.99 for p in prices],
        "close_price": prices,
        "daily_return": pd.Series(prices).pct_change().fillna(0.0),
    })

    # 2. Sentiment DataFrame
    sentiment_df = pd.DataFrame({
        "published_date": dates,
        "sentiment_score": np.random.uniform(-0.5, 0.5, size=60),
    })

    # 3. Macro Events DataFrame
    events_df = pd.DataFrame([
        {
            "event_date": dates[10],
            "event_name": "US Non-Farm Payrolls",
            "impact": "high",
            "actual_value": 220000.0,
            "forecast_value": 200000.0,
            "surprise": 20000.0,
        },
        {
            "event_date": dates[25],
            "event_name": "US Consumer Price Index (CPI)",
            "impact": "high",
            "actual_value": 3.2,
            "forecast_value": 3.0,
            "surprise": 0.2,
        },
        {
            "event_date": dates[40],
            "event_name": "FOMC Interest Rate Decision",
            "impact": "high",
            "actual_value": 5.25,
            "forecast_value": 5.25,
            "surprise": 0.0,
        },
    ])

    # 4. Predictions DataFrame
    predictions_df = pd.DataFrame({
        "event_date": dates,
        "actual": (market_df["daily_return"] > 0).astype(int),
        "predicted": (market_df["daily_return"].shift(-1) > 0).fillna(0).astype(int),
        "probability": np.random.uniform(0.3, 0.8, size=60),
        "daily_return": market_df["daily_return"],
        "surprise": np.random.normal(0.0, 0.2, size=60),
    })

    # 5. Model Results Dictionary
    model_results = {
        "metrics": {
            "accuracy": 0.58,
            "sharpe_ratio": 1.45,
            "sortino_ratio": 2.10,
            "max_drawdown": 0.045,
            "win_rate": 0.55,
            "profit_factor": 1.35,
            "cumulative_return": 0.082,
        },
        "predictions": predictions_df.to_dict(orient="records"),
        "strategy_matrix": {
            "US_EQUITY": {"sharpe_ratio": 1.25, "win_rate": 0.54, "max_drawdown": 0.05},
            "US_BOND": {"sharpe_ratio": 0.95, "win_rate": 0.52, "max_drawdown": 0.03},
            "GOLD": {"sharpe_ratio": 1.60, "win_rate": 0.58, "max_drawdown": 0.04},
        },
    }

    return market_df, sentiment_df, events_df, predictions_df, model_results


def test_sentiment_timeline_generation(tmp_path, mock_viz_fixtures):
    """Verify that sentiment timeline with macro event markers produces valid HTML."""
    _, sentiment_df, events_df, _, _ = mock_viz_fixtures
    charts = MacroSentimentCharts(output_dir=tmp_path / "charts", reports_dir=tmp_path)

    fig, html_path, png_path = charts.plot_sentiment_timeline_with_events(sentiment_df, events_df)

    assert fig is not None
    assert Path(html_path).exists()
    content = Path(html_path).read_text(encoding="utf-8")
    assert "plotly" in content.lower()


def test_event_study_generation(tmp_path, mock_viz_fixtures):
    """Verify event-study dynamics chart generation."""
    market_df, sentiment_df, events_df, _, _ = mock_viz_fixtures
    charts = MacroSentimentCharts(output_dir=tmp_path / "charts", reports_dir=tmp_path)

    fig, html_path, _ = charts.plot_event_study(events_df, market_df, sentiment_df)
    assert fig is not None
    assert Path(html_path).exists()


def test_strategy_matrix_heatmap_generation(tmp_path, mock_viz_fixtures):
    """Verify strategy matrix heatmap generation with numbers in cells."""
    market_df, _, events_df, _, _ = mock_viz_fixtures
    charts = MacroSentimentCharts(output_dir=tmp_path / "charts", reports_dir=tmp_path)

    fig, html_path, _ = charts.plot_strategy_matrix_heatmap(events_df, market_df)
    assert fig is not None
    assert Path(html_path).exists()


def test_equity_curve_and_drawdown_generation(tmp_path, mock_viz_fixtures):
    """Verify walk-forward equity curve vs benchmark and drawdown chart."""
    market_df, _, _, predictions_df, _ = mock_viz_fixtures
    charts = MacroSentimentCharts(output_dir=tmp_path / "charts", reports_dir=tmp_path)

    fig, html_path, _ = charts.plot_equity_curve_with_drawdown(predictions_df, market_df)
    assert fig is not None
    assert Path(html_path).exists()


def test_predicted_vs_realised_pnl_generation(tmp_path, mock_viz_fixtures):
    """Verify model calibration scatter chart generation."""
    _, _, _, predictions_df, _ = mock_viz_fixtures
    charts = MacroSentimentCharts(output_dir=tmp_path / "charts", reports_dir=tmp_path)

    fig, html_path, _ = charts.plot_predicted_vs_realised_pnl(predictions_df)
    assert fig is not None
    assert Path(html_path).exists()


def test_surprise_vs_return_generation(tmp_path, mock_viz_fixtures):
    """Verify surprise vs next-day return scatter plot."""
    market_df, _, events_df, _, _ = mock_viz_fixtures
    charts = MacroSentimentCharts(output_dir=tmp_path / "charts", reports_dir=tmp_path)

    fig, html_path, _ = charts.plot_surprise_vs_next_day_return(events_df, market_df)
    assert fig is not None
    assert Path(html_path).exists()


def test_rolling_sharpe_generation(tmp_path, mock_viz_fixtures):
    """Verify 30-trade rolling Sharpe stability chart."""
    _, _, _, predictions_df, _ = mock_viz_fixtures
    charts = MacroSentimentCharts(output_dir=tmp_path / "charts", reports_dir=tmp_path)

    fig, html_path, _ = charts.plot_rolling_sharpe(predictions_df, window=15)
    assert fig is not None
    assert Path(html_path).exists()


def test_generate_all_charts_master_runner(tmp_path, mock_viz_fixtures):
    """Verify that generate_all_charts produces deliverables and the unified HTML dashboard."""
    market_df, sentiment_df, events_df, predictions_df, model_results = mock_viz_fixtures
    charts = MacroSentimentCharts(output_dir=tmp_path / "charts", reports_dir=tmp_path)

    data = {
        "market_df": market_df,
        "sentiment_df": sentiment_df,
        "events_df": events_df,
        "predictions_df": predictions_df,
        "model_results": model_results,
    }

    paths = charts.generate_all_charts(data)
    assert len(paths) > 0

    dashboard_path = tmp_path / "macro_dashboard.html"
    assert dashboard_path.exists()
    html_text = dashboard_path.read_text(encoding="utf-8")
    assert "Macroeconomic Trading & Sentiment Dashboard" in html_text


def test_convert_markdown_to_pdf(tmp_path):
    """Verify that convert_markdown_to_pdf produces a valid styled PDF document."""
    from src.utils.generate_pdf import convert_markdown_to_pdf
    sample_md = tmp_path / "sample_report.md"
    sample_md.write_text("# Test Title\n\n| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |\n\n> Note: test callout\n", encoding="utf-8")
    pdf_out = tmp_path / "sample_report.pdf"
    res = convert_markdown_to_pdf(sample_md, pdf_out)
    assert res is not None
    assert pdf_out.exists()
    assert pdf_out.stat().st_size > 1000

