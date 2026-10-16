"""
Timezone normalization utilities using America/New_York (IANA).
Provides accurate conversions for both Eastern Standard Time (EST, UTC-5)
and Eastern Daylight Time (EDT, UTC-4), avoiding fixed-offset errors.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Union
from zoneinfo import ZoneInfo
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    nearest_workday,
    USMartinLutherKingJr,
    USPresidentsDay,
    GoodFriday,
    USMemorialDay,
    USLaborDay,
    USThanksgivingDay,
)
from pandas.tseries.offsets import CustomBusinessDay

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = timezone.utc


class USEquityExchangeHolidayCalendar(AbstractHolidayCalendar):
    """
    US Equity Exchange (NYSE / NASDAQ) Official Trading Holiday Calendar.

    Exchange holidays:
      1. New Year's Day (Jan 1, observed nearest workday)
      2. Martin Luther King, Jr. Day (third Monday in Jan)
      3. Washington's Birthday / Presidents' Day (third Monday in Feb)
      4. Good Friday (Friday before Easter - exchange closed, NOT a federal holiday)
      5. Memorial Day (last Monday in May)
      6. Juneteenth National Independence Day (June 19, observed nearest workday)
      7. Independence Day (July 4, observed nearest workday)
      8. Labor Day (first Monday in Sep)
      9. Thanksgiving Day (fourth Thursday in Nov)
     10. Christmas Day (Dec 25, observed nearest workday)

    Note: Columbus Day / Indigenous Peoples' Day and Veterans Day are US Federal
    holidays where bond markets and banks close, but US Equity Exchanges remain OPEN.
    They are intentionally excluded from exchange holidays.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2021-06-18",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


USEquityBusinessDay = CustomBusinessDay(calendar=USEquityExchangeHolidayCalendar())


def to_new_york_datetime(dt_val: Any) -> datetime:
    """
    Convert any timestamp/string/datetime to timezone-aware America/New_York datetime.
    """
    if dt_val is None:
        return datetime.now(NY_TZ)

    ts = pd.to_datetime(dt_val)
    if ts.tzinfo is None:
        # Default naive datetime to New York
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    return ts.to_pydatetime()


def to_new_york_date(dt_val: Any) -> date:
    """
    Extract the calendar date in America/New_York.
    Prevents UTC timestamps in evening/night from shifting to the next calendar date.
    """
    if dt_val is None:
        return datetime.now(NY_TZ).date()

    ts = pd.to_datetime(dt_val)
    if ts.tzinfo is None:
        return ts.date()
    ny_ts = ts.tz_convert(NY_TZ)
    return ny_ts.date()


def is_daylight_saving(dt_val: Any = None) -> bool:
    """
    Check if a given date or current time is in Eastern Daylight Time (EDT, UTC-4).
    Returns False if Eastern Standard Time (EST, UTC-5).
    """
    if dt_val is None:
        dt = datetime.now(NY_TZ)
    else:
        dt = to_new_york_datetime(dt_val)
    return bool(dt.dst())


def get_ny_market_close(dt_val: Any) -> datetime:
    """
    Get the 16:00:00 America/New_York market close timestamp for a given date.
    Aware of EST (21:00 UTC) vs EDT (20:00 UTC).
    """
    d = to_new_york_date(dt_val)
    return datetime.combine(d, time(16, 0, 0), tzinfo=NY_TZ)


def get_ny_market_open(dt_val: Any) -> datetime:
    """
    Get the 09:30:00 America/New_York market open timestamp for a given date.
    Aware of EST (14:30 UTC) vs EDT (13:30 UTC).
    """
    d = to_new_york_date(dt_val)
    return datetime.combine(d, time(9, 30, 0), tzinfo=NY_TZ)


def normalize_series_to_ny_date(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas datetime Series to America/New_York calendar dates.
    """
    dt_series = pd.to_datetime(series)
    if dt_series.dt.tz is None:
        return dt_series.dt.tz_localize(NY_TZ).dt.tz_convert(NY_TZ).dt.floor("D").dt.tz_localize(None)
    return dt_series.dt.tz_convert(NY_TZ).dt.floor("D").dt.tz_localize(None)


def parse_to_ny_iso(val: Any) -> str:
    """
    Parse any datetime/string and return an ISO string normalized to America/New_York session date.
    Prevents UTC evening/night timestamps from jumping to the next calendar date.
    """
    if not val:
        return datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M:%S")
    try:
        ts = pd.to_datetime(val)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(NY_TZ).tz_localize(None)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M:%S")


def assign_to_eligible_trading_session(
    published_at: Any,
    retrieved_at: Any = None,
    trading_dates: Any = None,
) -> pd.Timestamp:
    """
    Determine the earliest trading session date on which an information item
    (news article, economic announcement) is eligible to be incorporated into
    the model's feature vector.

    The daily information cutoff is strictly 16:00:00 America/New_York (regular market close).

    Rules:
      1. Effective availability timestamp is max(published_at, retrieved_at).
         (Only applies retrieved_at if contemporaneous <= 3 days, avoiding bulk
         scraping backfills from overriding historical publication dates by months or years).
      2. If effective time has no specific time-of-day (date-only or midnight):
         it is treated as released during business hours before 16:00 on that date.
      3. If effective time is published on or before 16:00:00 New York on Day D,
         it is eligible for trading session D.
      4. If effective time is published strictly after 16:00:00 New York on Day D,
         it is EXCLUDED from Day D features and rolled forward to Day D+1.
      5. If the resulting date falls on a weekend, it rolls forward to Monday.
      6. If a calendar of actual market trading dates is supplied:
         - If candidate date >= min(trading_dates), snaps to earliest trading date >= candidate.
         - If candidate date < min(trading_dates), snaps to min(trading_dates) ONLY if candidate
           falls in the immediate non-trading gap (weekend/holiday) directly before min(trading_dates)
           with no intervening trading sessions.
         - Otherwise, candidate date maps to its own historical trading session and is never clamped
           forward across historical months/years into the active calendar.
      7. Non-trading days and market holidays are resolved via USFederalHolidayCalendar.
    """
    if published_at is None or pd.isna(published_at):
        return pd.Timestamp(to_new_york_date(None))

    ts_pub = pd.to_datetime(published_at)
    if ts_pub.tzinfo is None:
        ts_pub_ny = to_new_york_datetime(ts_pub)
    else:
        ts_pub_ny = ts_pub.tz_convert(NY_TZ).to_pydatetime()

    effective_ts = ts_pub_ny
    if retrieved_at is not None and pd.notna(retrieved_at):
        ts_ret = pd.to_datetime(retrieved_at)
        if ts_ret.tzinfo is None:
            ts_ret_ny = to_new_york_datetime(ts_ret)
        else:
            ts_ret_ny = ts_ret.tz_convert(NY_TZ).to_pydatetime()
        # Only apply retrieval timestamp if contemporaneous (within 3 calendar days)
        # to prevent bulk-download scraping backfills from overriding historical publication dates
        if (ts_ret_ny.date() - ts_pub_ny.date()).days <= 3:
            effective_ts = max(effective_ts, ts_ret_ny)

    cal_date = effective_ts.date()
    has_time = not (
        effective_ts.hour == 0
        and effective_ts.minute == 0
        and effective_ts.second == 0
        and effective_ts.microsecond == 0
    )

    if has_time:
        close_cutoff = datetime.combine(cal_date, time(16, 0, 0), tzinfo=NY_TZ)
        if effective_ts > close_cutoff:
            eligible_date = cal_date + pd.Timedelta(days=1).to_pytimedelta()
        else:
            eligible_date = cal_date
    else:
        eligible_date = cal_date

    # Weekend roll-forward
    if eligible_date.weekday() == 5:  # Saturday
        eligible_date = eligible_date + pd.Timedelta(days=2).to_pytimedelta()
    elif eligible_date.weekday() == 6:  # Sunday
        eligible_date = eligible_date + pd.Timedelta(days=1).to_pytimedelta()

    res_ts = pd.Timestamp(eligible_date)

    usb = USEquityBusinessDay

    if trading_dates is not None and len(trading_dates) > 0:
        td_series = pd.Series(pd.to_datetime(list(trading_dates)).tz_localize(None).floor("D")).sort_values().drop_duplicates()
        min_td = td_series.iloc[0]
        if res_ts >= min_td:
            future_tds = td_series[td_series >= res_ts]
            if not future_tds.empty:
                res_ts = future_tds.iloc[0]
            else:
                own_b = pd.bdate_range(start=res_ts, periods=1, freq=usb)
                res_ts = own_b[0].floor("D")
        else:
            # res_ts is strictly before min_td
            b_days = pd.bdate_range(res_ts, min_td, freq=usb)
            if len(b_days) == 1:
                # Immediate non-trading gap right before min_td
                res_ts = min_td
            else:
                # Article was published on an earlier trading date/period; map to its own business day
                own_b = pd.bdate_range(start=res_ts, periods=1, freq=usb)
                res_ts = own_b[0].floor("D")
    else:
        own_b = pd.bdate_range(start=res_ts, periods=1, freq=usb)
        res_ts = own_b[0].floor("D")

    return res_ts


