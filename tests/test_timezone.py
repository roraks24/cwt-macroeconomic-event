"""Unit tests for America/New_York timezone handling across Standard Time (EST) and Daylight Saving Time (EDT)."""

from datetime import datetime, timezone
import pandas as pd
import pytest

from src.utils.timezone import (
    NY_TZ,
    get_ny_market_close,
    get_ny_market_open,
    is_daylight_saving,
    normalize_series_to_ny_date,
    to_new_york_date,
    to_new_york_datetime,
)


def test_standard_time_vs_daylight_saving_detection():
    """Verify that winter (EST) and summer (EDT) are correctly differentiated."""
    # January is Eastern Standard Time (EST, UTC-5)
    winter_date = datetime(2026, 1, 15, 12, 0)
    assert not is_daylight_saving(winter_date), "January should be Eastern Standard Time (EST)"
    ny_winter = to_new_york_datetime(winter_date)
    assert ny_winter.tzname() == "EST"
    assert ny_winter.utcoffset().total_seconds() == -5 * 3600

    # July is Eastern Daylight Time (EDT, UTC-4)
    summer_date = datetime(2026, 7, 15, 12, 0)
    assert is_daylight_saving(summer_date), "July should be Eastern Daylight Time (EDT)"
    ny_summer = to_new_york_datetime(summer_date)
    assert ny_summer.tzname() == "EDT"
    assert ny_summer.utcoffset().total_seconds() == -4 * 3600


def test_market_close_and_open_utc_offsets():
    """
    Verify 16:00 America/New_York market close converts to:
      - 21:00 UTC in winter (EST, UTC-5)
      - 20:00 UTC in summer (EDT, UTC-4)
    and 09:30 open converts to:
      - 14:30 UTC in winter
      - 13:30 UTC in summer
    """
    # Winter (EST)
    winter_close = get_ny_market_close("2026-01-15")
    assert winter_close.hour == 16
    assert winter_close.minute == 0
    winter_close_utc = winter_close.astimezone(timezone.utc)
    assert winter_close_utc.hour == 21, "16:00 EST must convert to 21:00 UTC"

    winter_open = get_ny_market_open("2026-01-15")
    assert winter_open.hour == 9
    assert winter_open.minute == 30
    winter_open_utc = winter_open.astimezone(timezone.utc)
    assert winter_open_utc.hour == 14 and winter_open_utc.minute == 30, "09:30 EST must convert to 14:30 UTC"

    # Summer (EDT)
    summer_close = get_ny_market_close("2026-07-15")
    assert summer_close.hour == 16
    assert summer_close.minute == 0
    summer_close_utc = summer_close.astimezone(timezone.utc)
    assert summer_close_utc.hour == 20, "16:00 EDT must convert to 20:00 UTC (not 21:00)"

    summer_open = get_ny_market_open("2026-07-15")
    assert summer_open.hour == 9
    assert summer_open.minute == 30
    summer_open_utc = summer_open.astimezone(timezone.utc)
    assert summer_open_utc.hour == 13 and summer_open_utc.minute == 30, "09:30 EDT must convert to 13:30 UTC"


def test_evening_utc_timestamp_calendar_date_no_shift():
    """
    CRITICAL: Verify that an event published at 23:30 UTC in summer
    (which is 19:30 EDT on the same day) maps to the correct New York calendar date
    and does NOT leak into the next day.
    """
    # 2026-07-15 23:30:00 UTC is 19:30:00 EDT on 2026-07-15
    utc_evening = pd.to_datetime("2026-07-15 23:30:00").tz_localize("UTC")
    ny_date = to_new_york_date(utc_evening)
    assert str(ny_date) == "2026-07-15", f"Expected 2026-07-15, got {ny_date}"

    # Early morning UTC: 2026-01-15 03:00:00 UTC is 22:00:00 EST on 2026-01-14
    utc_early = pd.to_datetime("2026-01-15 03:00:00").tz_localize("UTC")
    ny_date_early = to_new_york_date(utc_early)
    assert str(ny_date_early) == "2026-01-14", f"Expected 2026-01-14, got {ny_date_early}"


def test_normalize_series_to_ny_date():
    """Verify series of UTC timestamps correctly normalizes to New York calendar dates."""
    dates_utc = pd.Series([
        pd.to_datetime("2026-07-15 14:00:00+00:00"),  # 10:00 EDT -> July 15
        pd.to_datetime("2026-07-15 23:45:00+00:00"),  # 19:45 EDT -> July 15
        pd.to_datetime("2026-01-15 02:00:00+00:00"),  # 21:00 EST (Jan 14) -> Jan 14
    ])
    ny_normalized = normalize_series_to_ny_date(dates_utc)
    expected = pd.Series([
        pd.Timestamp("2026-07-15"),
        pd.Timestamp("2026-07-15"),
        pd.Timestamp("2026-01-14"),
    ])
    pd.testing.assert_series_equal(ny_normalized, expected)


def test_assign_to_eligible_trading_session_pre_and_post_close():
    """
    Verify that information items published shortly before 16:00 America/New_York
    are included in that day's trading session, while items published shortly after
    the 16:00 cutoff are excluded and rolled forward to the next session.
    """
    from src.utils.timezone import assign_to_eligible_trading_session

    trading_calendar = [
        pd.Timestamp("2026-07-15"),  # Wednesday
        pd.Timestamp("2026-07-16"),  # Thursday
        pd.Timestamp("2026-07-17"),  # Friday
        pd.Timestamp("2026-07-20"),  # Monday
    ]

    # 1. Summer EDT (UTC-4): Market close is 16:00 EDT (20:00 UTC)
    # 15:55 EDT (19:55 UTC) -> Before close -> Eligible on 2026-07-15
    ts_pre_close = "2026-07-15 15:55:00"
    session_pre = assign_to_eligible_trading_session(ts_pre_close, trading_dates=trading_calendar)
    assert session_pre == pd.Timestamp("2026-07-15"), f"Expected 2026-07-15, got {session_pre}"

    # 16:05 EDT (20:05 UTC) -> After close -> Must roll to 2026-07-16
    ts_post_close = "2026-07-15 16:05:00"
    session_post = assign_to_eligible_trading_session(ts_post_close, trading_dates=trading_calendar)
    assert session_post == pd.Timestamp("2026-07-16"), f"Expected 2026-07-16, got {session_post}"

    # 2. Winter EST (UTC-5): Market close is 16:00 EST (21:00 UTC)
    winter_calendar = [pd.Timestamp("2026-01-14"), pd.Timestamp("2026-01-15")]
    # 20:55 UTC = 15:55 EST (before close) -> Eligible on 2026-01-14
    ts_winter_pre = "2026-01-14 20:55:00+00:00"
    session_w_pre = assign_to_eligible_trading_session(ts_winter_pre, trading_dates=winter_calendar)
    assert session_w_pre == pd.Timestamp("2026-01-14"), f"Expected 2026-01-14, got {session_w_pre}"

    # 21:05 UTC = 16:05 EST (after close) -> Rolled to 2026-01-15
    ts_winter_post = "2026-01-14 21:05:00+00:00"
    session_w_post = assign_to_eligible_trading_session(ts_winter_post, trading_dates=winter_calendar)
    assert session_w_post == pd.Timestamp("2026-01-15"), f"Expected 2026-01-15, got {session_w_post}"

    # 3. Friday after close rolls across weekend to Monday
    ts_friday_post = "2026-07-17 17:30:00"
    session_friday = assign_to_eligible_trading_session(ts_friday_post, trading_dates=trading_calendar)
    assert session_friday == pd.Timestamp("2026-07-20"), f"Expected Monday 2026-07-20, got {session_friday}"

    # 4. Distinction between publication time and retrieval time
    # Published at 15:30 (before close), but scraper retrieved at 16:30 (after close)
    session_late_retrieval = assign_to_eligible_trading_session(
        published_at="2026-07-15 15:30:00",
        retrieved_at="2026-07-15 16:30:00",
        trading_dates=trading_calendar,
    )
    assert session_late_retrieval == pd.Timestamp("2026-07-16"), (
        f"Retrieval after cutoff must defer availability to 2026-07-16, got {session_late_retrieval}"
    )


def test_us_equity_exchange_holidays_vs_federal_holidays():
    """
    CRITICAL: Verify that USEquityExchangeHolidayCalendar correctly distinguishes
    stock exchange holidays (Good Friday) from federal bank holidays when equity
    exchanges remain OPEN (Columbus Day, Veterans Day).
    """
    from src.utils.timezone import assign_to_eligible_trading_session, USEquityExchangeHolidayCalendar

    cal = USEquityExchangeHolidayCalendar()
    holidays_2026 = cal.holidays("2026-01-01", "2026-12-31")

    # 1. Good Friday 2026 (April 3, 2026): Stock exchange is CLOSED
    assert pd.Timestamp("2026-04-03") in holidays_2026
    s_good_friday = assign_to_eligible_trading_session("2026-04-03 10:00:00")
    # Must roll across the weekend to Monday April 6, 2026
    assert s_good_friday == pd.Timestamp("2026-04-06"), f"Good Friday must roll to Monday, got {s_good_friday}"

    # 2. Columbus Day 2026 (October 12, 2026): US equity market is OPEN
    assert pd.Timestamp("2026-10-12") not in holidays_2026
    s_columbus = assign_to_eligible_trading_session("2026-10-12 10:00:00")
    assert s_columbus == pd.Timestamp("2026-10-12"), f"Columbus Day is an active trading session, got {s_columbus}"

    # 3. Veterans Day 2026 (November 11, 2026): US equity market is OPEN
    assert pd.Timestamp("2026-11-11") not in holidays_2026
    s_veterans = assign_to_eligible_trading_session("2026-11-11 10:00:00")
    assert s_veterans == pd.Timestamp("2026-11-11"), f"Veterans Day is an active trading session, got {s_veterans}"

    # 4. Memorial Day 2026 (Monday May 25, 2026): Stock exchange CLOSED -> Tuesday May 26
    assert pd.Timestamp("2026-05-25") in holidays_2026
    s_memorial = assign_to_eligible_trading_session("2026-05-25 10:00:00")
    assert s_memorial == pd.Timestamp("2026-05-26"), f"Memorial Day must roll to Tuesday, got {s_memorial}"

    # 5. Labor Day 2026 (Monday September 7, 2026): Stock exchange CLOSED -> Tuesday September 8
    assert pd.Timestamp("2026-09-07") in holidays_2026
    s_labor = assign_to_eligible_trading_session("2026-09-07 10:00:00")
    assert s_labor == pd.Timestamp("2026-09-08"), f"Labor Day must roll to Tuesday, got {s_labor}"

    # 6. Thanksgiving Day 2026 (Thursday November 26, 2026): Stock exchange CLOSED -> Friday November 27
    assert pd.Timestamp("2026-11-26") in holidays_2026
    s_thanksgiving = assign_to_eligible_trading_session("2026-11-26 10:00:00")
    assert s_thanksgiving == pd.Timestamp("2026-11-27"), f"Thanksgiving must roll to Friday, got {s_thanksgiving}"


