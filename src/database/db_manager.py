"""
Database Manager for CWT Macro Economics Sentiment Agent.
Provides SQLAlchemy ORM models and operations for macro events, news, and market data.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Sequence

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Ensure base directory is in sys.path to import project config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATABASE_URL  # noqa: E402

logger = logging.getLogger(__name__)

Base = declarative_base()


class MacroEvent(Base):
    """SQLAlchemy model representing macroeconomic calendar events."""

    __tablename__ = "macro_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_date = Column(DateTime, nullable=False, index=True)
    event_name = Column(String(255), nullable=False, index=True)
    country = Column(String(50), default="US", nullable=False)
    actual_value = Column(Float, nullable=True)
    forecast_value = Column(Float, nullable=True)
    previous_value = Column(Float, nullable=True)
    surprise = Column(Float, nullable=True)
    impact = Column(String(20), nullable=True)  # 'high', 'medium', 'low'
    source = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class NewsArticle(Base):
    """SQLAlchemy model representing economic news articles."""

    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    published_date = Column(DateTime, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source_url = Column(String(1000), nullable=True, index=True)
    sentiment_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    query_topic = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class MarketData(Base):
    """SQLAlchemy model representing multi-asset market prices and metrics."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    asset_class = Column(String(50), nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    daily_return = Column(Float, nullable=True)
    volatility_5d = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DatabaseManager:
    """Manages database connection, schema initialization, and CRUD operations."""

    def __init__(self, db_url: str | None = None) -> None:
        """Initialize database engine and create tables if they do not exist."""
        self.db_url = db_url or DATABASE_URL
        self.engine = create_engine(self.db_url, echo=False, future=True)
        self.SessionFactory = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False
        )
        self.create_tables()

    def create_tables(self) -> None:
        """Create all defined tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables verified/created successfully.")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager providing a transactional database session."""
        session: Session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _parse_datetime(val: str | datetime | None) -> datetime | None:
        """Helper to parse datetime from string or datetime object."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return pd.to_datetime(val).to_pydatetime()
            except Exception:
                return None
        return None

    def insert_macro_events(self, events: Sequence[dict]) -> int:
        """Bulk insert macroeconomic events."""
        if not events:
            return 0

        inserted_count = 0
        with self.get_session() as session:
            for item in events:
                event_date = self._parse_datetime(item.get("event_date"))
                if not event_date:
                    continue

                actual = item.get("actual_value")
                forecast = item.get("forecast_value")
                surprise = item.get("surprise")

                # Compute surprise if not provided but actual and forecast exist
                if surprise is None and actual is not None and forecast is not None:
                    try:
                        surprise = float(actual) - float(forecast)
                    except (ValueError, TypeError):
                        surprise = None

                event = MacroEvent(
                    event_date=event_date,
                    event_name=str(item.get("event_name", "Unknown")),
                    country=str(item.get("country", "US")),
                    actual_value=float(actual) if actual is not None else None,
                    forecast_value=float(forecast) if forecast is not None else None,
                    previous_value=(
                        float(item["previous_value"])
                        if item.get("previous_value") is not None
                        else None
                    ),
                    surprise=float(surprise) if surprise is not None else None,
                    impact=item.get("impact"),
                    source=item.get("source", "apify"),
                    created_at=self._parse_datetime(item.get("created_at"))
                    or datetime.utcnow(),
                )
                session.add(event)
                inserted_count += 1

        logger.info("Inserted %d macro events.", inserted_count)
        return inserted_count

    def insert_news_articles(self, articles: Sequence[dict]) -> int:
        """Bulk insert economic news articles."""
        if not articles:
            return 0

        inserted_count = 0
        with self.get_session() as session:
            for item in articles:
                pub_date = self._parse_datetime(
                    item.get("published_date") or item.get("created_at")
                ) or datetime.utcnow()

                article = NewsArticle(
                    published_date=pub_date,
                    title=str(item.get("title", "")),
                    content=str(item.get("content", "")),
                    source_url=item.get("source_url") or item.get("url"),
                    sentiment_score=(
                        float(item["sentiment_score"])
                        if item.get("sentiment_score") is not None
                        else None
                    ),
                    relevance_score=(
                        float(item["relevance_score"])
                        if item.get("relevance_score") is not None
                        else None
                    ),
                    query_topic=item.get("query_topic") or item.get("topic"),
                    created_at=datetime.utcnow(),
                )
                session.add(article)
                inserted_count += 1

        logger.info("Inserted %d news articles.", inserted_count)
        return inserted_count

    def insert_market_data(self, data: Sequence[dict]) -> int:
        """Bulk insert multi-asset market data rows."""
        if not data:
            return 0

        inserted_count = 0
        with self.get_session() as session:
            for item in data:
                m_date = self._parse_datetime(item.get("date"))
                if not m_date:
                    continue

                market_row = MarketData(
                    date=m_date,
                    asset_class=str(item.get("asset_class", "US_EQUITY")),
                    open_price=float(item.get("open_price", 0.0)),
                    high_price=float(item.get("high_price", 0.0)),
                    low_price=float(item.get("low_price", 0.0)),
                    close_price=float(item.get("close_price", 0.0)),
                    daily_return=(
                        float(item["daily_return"])
                        if item.get("daily_return") is not None
                        else None
                    ),
                    volatility_5d=(
                        float(item["volatility_5d"])
                        if item.get("volatility_5d") is not None
                        else None
                    ),
                    created_at=datetime.utcnow(),
                )
                session.add(market_row)
                inserted_count += 1

        logger.info("Inserted %d market data rows.", inserted_count)
        return inserted_count

    def get_macro_events(
        self,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Query macroeconomic events in a given date range as a DataFrame."""
        with self.get_session() as session:
            query = session.query(MacroEvent)
            if start_date:
                query = query.filter(MacroEvent.event_date >= self._parse_datetime(start_date))
            if end_date:
                query = query.filter(MacroEvent.event_date <= self._parse_datetime(end_date))
            query = query.order_by(MacroEvent.event_date.asc())

            results = [
                {
                    "id": r.id,
                    "event_date": r.event_date,
                    "event_name": r.event_name,
                    "country": r.country,
                    "actual_value": r.actual_value,
                    "forecast_value": r.forecast_value,
                    "previous_value": r.previous_value,
                    "surprise": r.surprise,
                    "impact": r.impact,
                    "source": r.source,
                    "created_at": r.created_at,
                }
                for r in query.all()
            ]

        return pd.DataFrame(results)

    def get_news_articles(
        self,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Query news articles in a given date range as a DataFrame."""
        with self.get_session() as session:
            query = session.query(NewsArticle)
            if start_date:
                query = query.filter(NewsArticle.published_date >= self._parse_datetime(start_date))
            if end_date:
                query = query.filter(NewsArticle.published_date <= self._parse_datetime(end_date))
            query = query.order_by(NewsArticle.published_date.asc())

            results = [
                {
                    "id": r.id,
                    "published_date": r.published_date,
                    "title": r.title,
                    "content": r.content,
                    "source_url": r.source_url,
                    "sentiment_score": r.sentiment_score,
                    "relevance_score": r.relevance_score,
                    "query_topic": r.query_topic,
                    "created_at": r.created_at,
                }
                for r in query.all()
            ]

        return pd.DataFrame(results)

    def get_market_data(
        self,
        asset_class: str | None = None,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Query market data for an asset class across a date range as a DataFrame."""
        with self.get_session() as session:
            query = session.query(MarketData)
            if asset_class:
                query = query.filter(MarketData.asset_class == asset_class)
            if start_date:
                query = query.filter(MarketData.date >= self._parse_datetime(start_date))
            if end_date:
                query = query.filter(MarketData.date <= self._parse_datetime(end_date))
            query = query.order_by(MarketData.date.asc())

            results = [
                {
                    "id": r.id,
                    "date": r.date,
                    "asset_class": r.asset_class,
                    "open_price": r.open_price,
                    "high_price": r.high_price,
                    "low_price": r.low_price,
                    "close_price": r.close_price,
                    "daily_return": r.daily_return,
                    "volatility_5d": r.volatility_5d,
                    "created_at": r.created_at,
                }
                for r in query.all()
            ]

        return pd.DataFrame(results)

    def get_joined_dataset(
        self,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Join macro events with market data on calendar date, producing the training dataset.
        Normalizes timestamps to date boundaries and performs a merge.
        """
        macro_df = self.get_macro_events(start_date, end_date)
        market_df = self.get_market_data(start_date=start_date, end_date=end_date)

        if macro_df.empty or market_df.empty:
            logger.warning("Either macro events or market data is empty. Returning available rows.")
            if not macro_df.empty:
                return macro_df
            return market_df

        # Normalize timestamps to date for accurate calendar joining
        macro_df["calendar_date"] = pd.to_datetime(macro_df["event_date"]).dt.date
        market_df["calendar_date"] = pd.to_datetime(market_df["date"]).dt.date

        # Join datasets
        joined_df = pd.merge(
            market_df,
            macro_df,
            on="calendar_date",
            how="left",
            suffixes=("_market", "_macro"),
        )
        joined_df.sort_values(by="calendar_date", inplace=True)
        return joined_df.reset_index(drop=True)
