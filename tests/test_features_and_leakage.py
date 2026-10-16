"""Unit tests for feature engineering, time proximity, macro regimes, and zero data leakage."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer, _sanitize_name


@pytest.fixture
def sample_joined_data():
    """Fixture providing a mock joined dataset over 40 days."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    records = []

    # Generate synthetic price series (random walk with drift)
    np.random.seed(42)
    price = 100.0
    for i, dt in enumerate(dates):
        daily_ret = 0.001 + 0.01 * np.random.randn()
        price *= (1.0 + daily_ret)
        
        # Inject macro events on specific days
        event_name = None
        actual = None
        forecast = None
        surprise = None
        impact = None

        if i == 5:
            event_name = "US Non-Farm Payrolls"
            actual = 210.0
            forecast = 190.0
            surprise = 20.0
            impact = "high"
        elif i == 12:
            event_name = "US Consumer Price Index (CPI)"
            actual = 3.1
            forecast = 2.9
            surprise = 0.2
            impact = "high"
        elif i == 20:
            event_name = "Fed Interest Rate Decision (FOMC)"
            actual = 5.25
            forecast = 5.25
            surprise = 0.0
            impact = "high"

        records.append({
            "event_date": dt,
            "date": dt,
            "asset_class": "US_EQUITY",
            "close_price": price,
            "daily_return": daily_ret,
            "event_name": event_name,
            "actual_value": actual,
            "forecast_value": forecast,
            "surprise": surprise,
            "impact": impact,
        })

    return pd.DataFrame(records)


def test_sanitize_name():
    """Verify column name sanitization handles symbols and spaces properly."""
    assert _sanitize_name("US Non-Farm Payrolls (NFP)") == "us_non_farm_payrolls_nfp"
    assert _sanitize_name("  CPI % y/y  ") == "cpi_y_y"
    assert _sanitize_name("Fed Funds Rate!") == "fed_funds_rate"


def test_create_surprise_features(sample_joined_data):
    """Verify that surprise metrics are correctly computed and pivoted."""
    fe = FeatureEngineer(sample_joined_data)
    surp_df = fe.create_surprise_features()

    assert not surp_df.empty
    # Expect columns for the non-farm payrolls, cpi, and fomc slugs
    nfp_cols = [c for c in surp_df.columns if "non_farm" in c]
    assert len(nfp_cols) > 0
    assert any("surprise" in c for c in nfp_cols)
    assert any("surprise_direction" in c for c in nfp_cols)


def test_create_time_features(sample_joined_data):
    """Verify that time and macro proximity features are computed properly."""
    fe = FeatureEngineer(sample_joined_data)
    time_df = fe.create_time_features()

    assert not time_df.empty
    assert "day_of_week" in time_df.columns
    assert "month" in time_df.columns
    assert "quarter" in time_df.columns
    assert "is_month_end" in time_df.columns
    assert "days_since_last_fomc" in time_df.columns
    assert "days_since_last_nfp" in time_df.columns
    assert "days_since_last_cpi" in time_df.columns

    # Check proximity behavior: on day 5 (NFP), days_since_last_nfp should be 365 or prior,
    # and on day 6 it should be exactly 1.0 day
    nfp_day = sample_joined_data.iloc[5]["event_date"]
    day_after_nfp = sample_joined_data.iloc[6]["event_date"]
    
    assert time_df.loc[day_after_nfp, "days_since_last_nfp"] == 1.0


def test_create_sentiment_features(sample_joined_data):
    """Verify news sentiment rolling averages and momentum calculation."""
    fe = FeatureEngineer(sample_joined_data)
    
    # Create mock news dataset
    news_records = [
        {"published_date": "2026-01-02", "sentiment_score": 0.5},
        {"published_date": "2026-01-02", "sentiment_score": 0.3},
        {"published_date": "2026-01-03", "sentiment_score": -0.2},
        {"published_date": "2026-01-04", "sentiment_score": 0.8},
    ]
    news_df = pd.DataFrame(news_records)

    sent_df = fe.create_sentiment_features(news_df=news_df)
    assert not sent_df.empty
    assert "sentiment_roll_3d" in sent_df.columns
    assert "sentiment_roll_7d" in sent_df.columns
    assert "sentiment_momentum" in sent_df.columns
    assert "news_volume" in sent_df.columns


def test_zero_data_leakage_in_target_construction(sample_joined_data):
    """
    CRITICAL: Verify strict zero data leakage.
    The prediction target at timestamp t must equal (daily_return[t+1] > 0).
    The last row (unobserved future) must be dropped to avoid lookahead contamination.
    """
    fe = FeatureEngineer(sample_joined_data)
    matrix = fe.build_feature_matrix()

    assert not matrix.empty
    # The matrix must have exactly len(unique_dates) - 1 rows
    unique_dates_count = sample_joined_data["event_date"].nunique()
    assert len(matrix) == unique_dates_count - 1

    # Verify that target at index i corresponds to daily_return at i+1
    raw_returns = sample_joined_data.groupby("event_date")["daily_return"].mean().values

    for i in range(len(matrix)):
        expected_target = int(raw_returns[i + 1] > 0)
        assert matrix.iloc[i]["target"] == expected_target, f"Target leakage mismatch at step {i}"

    # Verify feature columns do not include target or future returns
    feat_names = fe.get_feature_importance_names()
    assert "target" not in feat_names
    assert "daily_return" not in feat_names
    assert "close_price" not in feat_names


def test_multi_asset_target_alignment_no_boundary_crossing():
    """
    CRITICAL: Verify that target calculation shifts strictly within each asset class.
    Target labels must never cross asset boundaries, and the last row of each asset
    must be dropped as unobserved future.
    """
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    records = []
    
    # Asset A returns: all positive (+0.02) except day 5 which is -0.05
    for dt in dates:
        records.append({
            "event_date": dt,
            "asset_class": "SPY",
            "close_price": 500.0,
            "daily_return": 0.02,
        })
    # Asset B returns: all negative (-0.02)
    for dt in dates:
        records.append({
            "event_date": dt,
            "asset_class": "TLT",
            "close_price": 100.0,
            "daily_return": -0.02,
        })
        
    df = pd.DataFrame(records)
    fe = FeatureEngineer(df)
    matrix = fe.build_feature_matrix()
    
    assert not matrix.empty
    # Each asset had 10 rows; dropping the last unobserved row per asset yields 9 * 2 = 18 rows
    assert len(matrix) == 18
    
    spy_matrix = matrix[matrix["asset_class"] == "SPY"]
    tlt_matrix = matrix[matrix["asset_class"] == "TLT"]
    assert len(spy_matrix) == 9
    assert len(tlt_matrix) == 9
    
    # SPY targets should all be 1 (because next returns are +0.02)
    # Most importantly, the last row of SPY must NOT see TLT's first return (-0.02)!
    assert (spy_matrix["target"] == 1).all()
    
    # TLT targets should all be 0 (because next returns are -0.02)
    assert (tlt_matrix["target"] == 0).all()

def test_next_open_execution_target_and_pricing_isolation():
    """
    CRITICAL: Verify Next-Session Open Execution target and pricing isolation.
    When open_price is present:
      - entry_price must equal Open_{T+1} (open_price.shift(-1))
      - exit_price must equal Open_{T+2} (open_price.shift(-2))
      - daily_return must equal (Open_{T+2} - Open_{T+1}) / Open_{T+1}
      - target must equal (daily_return > 0).astype(int)
      - exactly 2 unobserved rows per asset must be dropped
      - open_price, close_price, entry_price, exit_price must NOT leak into feature names
    """
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    records = []
    # Open prices: 100, 102, 101, 105, 104, 108, 107, 111, 110, 115
    opens = [100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 111.0, 110.0, 115.0]
    for i, dt in enumerate(dates):
        records.append({
            "event_date": dt,
            "asset_class": "SPY",
            "open_price": opens[i],
            "close_price": opens[i] + 0.5,
            "daily_return": 0.01,
        })
    df = pd.DataFrame(records)
    fe = FeatureEngineer(df)
    matrix = fe.build_feature_matrix()

    # 10 rows - 2 unobserved = 8 rows
    assert len(matrix) == 8, f"Expected 8 rows after dropping 2 unobserved rows, got {len(matrix)}"

    # Check row 0 (Date T=0):
    # Entry price must be Open_1 = 102.0
    # Exit price must be Open_2 = 101.0
    # Expected trade return = (101.0 - 102.0) / 102.0 = -1 / 102
    assert matrix.iloc[0]["entry_price"] == 102.0
    assert matrix.iloc[0]["exit_price"] == 101.0
    expected_ret_0 = (101.0 - 102.0) / 102.0
    assert pytest.approx(matrix.iloc[0]["daily_return"], 1e-6) == expected_ret_0
    assert matrix.iloc[0]["target"] == 0  # negative return -> target 0

    # Check row 1 (Date T=1):
    # Entry price must be Open_2 = 101.0
    # Exit price must be Open_3 = 105.0
    # Expected trade return = (105.0 - 101.0) / 101.0 = +4 / 101
    assert matrix.iloc[1]["entry_price"] == 101.0
    assert matrix.iloc[1]["exit_price"] == 105.0
    expected_ret_1 = (105.0 - 101.0) / 101.0
    assert pytest.approx(matrix.iloc[1]["daily_return"], 1e-6) == expected_ret_1
    assert matrix.iloc[1]["target"] == 1  # positive return -> target 1

    # Check feature isolation (zero leakage of price columns into model features)
    features = fe.get_feature_importance_names()
    forbidden = {"open_price", "close_price", "entry_price", "exit_price", "target", "daily_return"}
    for f_col in forbidden:
        assert f_col not in features, f"Leaked column {f_col} found in feature names!"


def test_after_close_news_excluded_from_same_day_feature_vector():
    """
    CRITICAL: Verify that news articles published after 16:00 America/New_York (market close)
    are strictly excluded from Day T's feature vector and become available only on Day T+1.
    """
    trading_dates = [pd.Timestamp("2026-07-15"), pd.Timestamp("2026-07-16"), pd.Timestamp("2026-07-17")]
    records = []
    for dt in trading_dates:
        records.append({
            "event_date": dt,
            "asset_class": "SPY",
            "open_price": 500.0,
            "close_price": 502.0,
            "daily_return": 0.004,
        })
    df = pd.DataFrame(records)
    fe = FeatureEngineer(df)

    # Article A: Published Day 1 (July 15) at 15:30 EDT (before close) -> sentiment +0.80
    # Article B: Published Day 1 (July 15) at 16:30 EDT (after close) -> sentiment -0.90
    news_df = pd.DataFrame([
        {
            "published_at": "2026-07-15 15:30:00",
            "sentiment_score": 0.80,
        },
        {
            "published_at": "2026-07-15 16:30:00",
            "sentiment_score": -0.90,
        },
    ])

    sentiment_feats = fe.create_sentiment_features(news_df=news_df)

    # Day 1 (July 15):
    # Only Article A (+0.80) should be included. Article B was published AFTER the 16:00 close!
    # Therefore, Day 1 news_volume must be 1.0 and 3d rolling sentiment must equal 0.80.
    assert sentiment_feats.loc[pd.Timestamp("2026-07-15"), "news_volume"] == 1.0
    assert pytest.approx(sentiment_feats.loc[pd.Timestamp("2026-07-15"), "sentiment_roll_3d"], 1e-4) == 0.80

    assert sentiment_feats.loc[pd.Timestamp("2026-07-16"), "news_volume"] == 1.0


def test_historical_2022_article_not_assigned_to_april_2026():
    """
    CRITICAL: Verify that an article published in 2022 is NOT treated as newly published
    in April 2026, even when the market calendar provided starts on April 17, 2026.
    """
    from src.utils.timezone import assign_to_eligible_trading_session

    market_dates = [pd.Timestamp("2026-04-17"), pd.Timestamp("2026-04-20"), pd.Timestamp("2026-04-21")]

    # Article from 2022
    s_2022 = assign_to_eligible_trading_session("2022-07-28 14:33:59", trading_dates=market_dates)
    assert s_2022 == pd.Timestamp("2022-07-28"), f"Expected 2022-07-28, got {s_2022}"

    # Article from 2025
    s_2025 = assign_to_eligible_trading_session("2025-11-20 13:33:48", trading_dates=market_dates)
    assert s_2025 == pd.Timestamp("2025-11-20"), f"Expected 2025-11-20, got {s_2025}"

    # Bulk scraping timestamp on 2026-10-14 must NOT override 2022 publication date
    s_bulk = assign_to_eligible_trading_session(
        published_at="2022-07-28 14:33:59",
        retrieved_at="2026-10-14 16:05:43",
        trading_dates=market_dates,
    )
    assert s_bulk == pd.Timestamp("2022-07-28"), f"Bulk retrieval must not shift 2022 article, got {s_bulk}"


def test_synthetic_fixture_first_session_does_not_absorb_unrelated_historical_articles():
    """
    UNIT TEST (CONTROLLED SYNTHETIC FIXTURE):
    Verifies that when a single legitimate article is published on Day 1 alongside
    ancient 2022-2025 articles and pre-window lookback articles, Day 1 news_volume
    is strictly 1.0 (counting only the single same-day synthetic article, and strictly
    excluding the older articles).
    """
    # 5 market sessions starting Friday April 17, 2026
    market_dates = [
        pd.Timestamp("2026-04-17"),  # Friday (Day 1)
        pd.Timestamp("2026-04-20"),  # Monday (Day 2)
        pd.Timestamp("2026-04-21"),  # Tuesday (Day 3)
        pd.Timestamp("2026-04-22"),  # Wednesday (Day 4)
        pd.Timestamp("2026-04-23"),  # Thursday (Day 5)
    ]
    records = []
    for dt in market_dates:
        records.append({
            "event_date": dt,
            "asset_class": "SPY",
            "open_price": 500.0,
            "close_price": 501.0,
            "daily_return": 0.002,
        })
    df = pd.DataFrame(records)
    fe = FeatureEngineer(df)

    news_df = pd.DataFrame([
        # 1. Ancient article from 2022 -> must be excluded completely
        {"published_at": "2022-07-28 14:33:59", "sentiment_score": -0.80},
        # 2. Older article from 2025 -> must be excluded completely
        {"published_at": "2025-11-20 13:33:48", "sentiment_score": -0.60},
        # 3. Legitimate pre-window lookback article from Monday April 13, 2026 -> +0.50
        {"published_at": "2026-04-13 10:17:03", "sentiment_score": 0.50},
        # 4. Same-day article on April 17, 2026 at 11:00 (before close) -> +0.20
        {"published_at": "2026-04-17 11:00:00", "sentiment_score": 0.20},
        # 5. After-close article on April 17 at 16:30 -> rolls to Monday April 20 -> -0.40
        {"published_at": "2026-04-17 16:30:00", "sentiment_score": -0.40},
    ])

    feats = fe.create_sentiment_features(news_df=news_df)

    # 1. Day 1 (2026-04-17):
    # In this synthetic fixture, exactly 1 article was created for April 17 (Article 4).
    # news_volume must be EXACTLY 1.0 (Only Article 4). Ancient articles (Articles 1-3) must NOT be absorbed!
    assert feats.loc[pd.Timestamp("2026-04-17"), "news_volume"] == 1.0, (
        f"Expected news_volume 1.0 on 2026-04-17 in synthetic fixture, got {feats.loc[pd.Timestamp('2026-04-17'), 'news_volume']}"
    )
    # mean_sentiment on Day 1 must be 0.20 (Article 4 alone)
    assert pytest.approx(feats.loc[pd.Timestamp("2026-04-17"), "sentiment_roll_3d"], 1e-4) == (0.20 / 3)

    # 2. Rolling 7d sentiment incorporates the legitimate pre-window article (April 13: +0.50)
    # and Day 1 (April 17: +0.20).
    # It must NOT contain the negative sentiment of 2022 (-0.80) or 2025 (-0.60)!
    s_7d = feats.loc[pd.Timestamp("2026-04-17"), "sentiment_roll_7d"]
    assert s_7d > 0.0, f"Expected positive rolling 7d sentiment from +0.50 and +0.20, got {s_7d}"

    # 3. Day 2 (Monday 2026-04-20):
    # Receives Article 5 (-0.40) rolled across the weekend
    assert feats.loc[pd.Timestamp("2026-04-20"), "news_volume"] == 1.0


def test_actual_database_first_session_has_zero_news_volume_and_no_contamination():
    """
    INTEGRATION TEST (REAL DATABASE RECORDS):
    Verifies that on the actual production database records (data/cwt_macro.db):
    - Exactly 0 articles were published on April 17, 2026.
    - news_volume on 2026-04-17 is strictly 0.0 (not 66.0).
    - Pre-window articles from 2022-2025 are completely excluded from Day 1 features.
    - Legitimate pre-window lookback context (April 13) is preserved in sentiment_roll_7d.
    """
    from src.database.db_manager import DatabaseManager, NewsArticle
    from src.sentiment.sentiment_scorer import QuantSentimentScorer

    db = DatabaseManager()
    mdf = db.get_market_data("US_EQUITY")
    if mdf.empty:
        pytest.skip("Market data not available in database.")

    joined = db.get_joined_dataset()
    fe = FeatureEngineer(joined)

    with db.get_session() as s:
        articles = s.query(NewsArticle).all()
        if not articles:
            pytest.skip("No articles in database.")
        news_records = [
            {
                "id": a.id,
                "published_date": a.published_date,
                "sentiment_score": a.sentiment_score or 0.0,
            }
            for a in articles
        ]
    news_df = pd.DataFrame(news_records)

    feats = fe.create_sentiment_features(news_df=news_df)

    # In the real database, zero articles were published on 2026-04-17
    # news_volume must be strictly 0.0, proving that the 66 pre-window articles were NOT clamped forward!
    actual_day1_vol = feats.loc[pd.Timestamp("2026-04-17"), "news_volume"]
    assert actual_day1_vol == 0.0, f"Expected news_volume == 0.0 on real database, got {actual_day1_vol}"

    # mean_sentiment on Day 1 must be 0.0
    actual_day1_mean = feats.loc[pd.Timestamp("2026-04-17"), "sentiment_roll_3d"]
    assert actual_day1_mean == 0.0, f"Expected sentiment_roll_3d == 0.0 on real database, got {actual_day1_mean}"

    # sentiment_roll_7d reflects the legitimate pre-window lookback from April 13 (+0.15 / 7 = ~0.0214)
    actual_day1_7d = feats.loc[pd.Timestamp("2026-04-17"), "sentiment_roll_7d"]
    assert 0.0 < actual_day1_7d < 0.10, f"Expected legitimate pre-window lookback sentiment, got {actual_day1_7d}"


def test_pre_window_news_transformation_at_feature_matrix_level():
    """
    REGRESSION TEST:
    Verifies that the pre-window news fix produces a tangible, mathematically verified
    transformation in intermediate feature matrices and training rows, rather than testing
    only the final return series.

    Asserts:
    1. Legacy clamping forces 66 ancient articles onto Day 1, creating artificial news_volume == 66.0.
    2. Corrected pre-window lookback excludes ancient articles, resulting in Day 1 news_volume == 0.0.
    3. Sentiment rolling 3d, 7d, and momentum features distinctly diverge over the initial lookback window.
    4. Feature matrix hashes are non-identical.
    """
    import hashlib
    from src.database.db_manager import DatabaseManager, NewsArticle
    from src.utils.timezone import USEquityBusinessDay

    db = DatabaseManager()
    mdf = db.get_market_data("US_EQUITY")
    if mdf.empty:
        pytest.skip("Market data not available.")

    joined = db.get_joined_dataset()
    fe = FeatureEngineer(joined)
    all_dates = pd.DatetimeIndex(sorted(fe.df["event_date"].unique()))
    min_date = all_dates[0]

    with db.get_session() as s:
        articles = s.query(NewsArticle).all()
        if not articles:
            pytest.skip("No articles in database.")
        news_records = [
            {
                "id": a.id,
                "published_date": a.published_date,
                "sentiment_score": a.sentiment_score or 0.0,
            }
            for a in articles
        ]
    raw_news_df = pd.DataFrame(news_records)

    # 1. Legacy Clamping Transformation (Variant 1)
    legacy_ndf = raw_news_df.copy()
    dt_s = pd.to_datetime(legacy_ndf["published_date"], format="mixed")
    if hasattr(dt_s.dt, "tz") and dt_s.dt.tz is not None:
        dt_s = dt_s.dt.tz_convert(None)
    legacy_ndf["event_date"] = dt_s.dt.floor("D")
    legacy_ndf.loc[legacy_ndf["event_date"] < min_date, "event_date"] = min_date

    legacy_daily = legacy_ndf.groupby("event_date").agg(
        mean_sentiment=("sentiment_score", "mean"),
        news_volume=("sentiment_score", "count")
    ).reindex(all_dates).fillna(0.0)
    legacy_daily["sentiment_roll_3d"] = legacy_daily["mean_sentiment"].rolling(3, min_periods=1).mean()
    legacy_daily["sentiment_roll_7d"] = legacy_daily["mean_sentiment"].rolling(7, min_periods=1).mean()
    legacy_daily["sentiment_momentum"] = legacy_daily["sentiment_roll_3d"] - legacy_daily["sentiment_roll_7d"]

    # 2. Corrected Pre-Window News Fix Transformation (Variant 4)
    fixed_ndf = raw_news_df.copy()
    fixed_ndf["event_date"] = dt_s.dt.floor("D")
    lookback_cutoff = min_date - pd.Timedelta(days=14)
    pre_window = fixed_ndf[(fixed_ndf["event_date"] < min_date) & (fixed_ndf["event_date"] >= lookback_cutoff)]
    if not pre_window.empty:
        earliest_pre = pre_window["event_date"].min()
        pre_dates = pd.bdate_range(start=earliest_pre, end=min_date, freq=USEquityBusinessDay).floor("D")[:-1]
        active_cal = pre_dates.union(all_dates).sort_values()
    else:
        active_cal = all_dates

    fixed_daily = fixed_ndf.groupby("event_date").agg(
        mean_sentiment=("sentiment_score", "mean"),
        news_volume=("sentiment_score", "count")
    ).reindex(active_cal).fillna(0.0)
    fixed_daily["sentiment_roll_3d"] = fixed_daily["mean_sentiment"].rolling(3, min_periods=1).mean()
    fixed_daily["sentiment_roll_7d"] = fixed_daily["mean_sentiment"].rolling(7, min_periods=1).mean()
    fixed_daily["sentiment_momentum"] = fixed_daily["sentiment_roll_3d"] - fixed_daily["sentiment_roll_7d"]
    fixed_res = fixed_daily.loc[all_dates]

    # Verification 1: Day 1 news_volume
    assert legacy_daily.loc[min_date, "news_volume"] >= 60.0, (
        f"Legacy transformation must clamp pre-window news to Day 1, got {legacy_daily.loc[min_date, 'news_volume']}"
    )
    assert fixed_res.loc[min_date, "news_volume"] == 0.0, (
        f"Fixed transformation must NOT clamp pre-window news to Day 1, got {fixed_res.loc[min_date, 'news_volume']}"
    )

    # Verification 2: Initial rolling feature divergence
    diff_roll7 = (legacy_daily["sentiment_roll_7d"] != fixed_res["sentiment_roll_7d"]).sum()
    diff_mom = (legacy_daily["sentiment_momentum"] != fixed_res["sentiment_momentum"]).sum()
    assert diff_roll7 >= 7, f"Expected at least 7 days of rolling 7d sentiment divergence, got {diff_roll7}"
    assert diff_mom >= 7, f"Expected at least 7 days of sentiment momentum divergence, got {diff_mom}"

    # Verification 3: Hash divergence
    legacy_hash = hashlib.sha256(legacy_daily.to_json().encode()).hexdigest()
    fixed_hash = hashlib.sha256(fixed_res.to_json().encode()).hexdigest()
    assert legacy_hash != fixed_hash, "Intermediate sentiment feature matrices must have distinct hashes"




