"""
Macro Outlook Generator Module for CWT Macro Economics Sentiment Agent.
Analyzes macroeconomic indicators, news sentiment, and policy trends to synthesize
a composite macro score and generate executive markdown outlook reports.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Ensure base directory is in sys.path to import project config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ASSET_CLASSES, MACRO_EVENTS, REPORTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)


class MacroOutlookGenerator:
    """
    Synthesizes economic events and sentiment into high-level macroeconomic analysis,
    calculates composite regime scores, and produces comprehensive outlook reports.
    """

    def __init__(self, reports_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the outlook generator.

        Args:
            reports_dir: Directory where reports are saved. Defaults to REPORTS_DIR from config.
        """
        self.reports_dir = Path(reports_dir) if reports_dir else Path(REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_events_df(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Helper to standardize column names and timestamps for event data."""
        if events_df.empty:
            return pd.DataFrame()

        df = events_df.copy()
        # Find date column
        date_col = next((c for c in ["event_date", "date", "created_at"] if c in df.columns), None)
        if date_col:
            df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(by="event_date")

        # Standardize event_name
        name_col = next((c for c in ["event_name", "event", "name", "indicator"] if c in df.columns), None)
        if name_col and name_col != "event_name":
            df["event_name"] = df[name_col]

        # Standardize actual_value
        val_col = next((c for c in ["actual_value", "actual", "value"] if c in df.columns), None)
        if val_col and val_col != "actual_value":
            df["actual_value"] = pd.to_numeric(df[val_col], errors="coerce")
        elif "actual_value" in df.columns:
            df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce")

        return df

    def analyze_inflation_trend(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Filter CPI/PPI/PCE events and compute trend (rising/falling/stable), latest value, and 3-month average.

        Args:
            events_df: Macroeconomic events DataFrame.

        Returns:
            Dictionary with 'trend', 'latest', 'average', and 'assessment'.
        """
        default_res = {
            "trend": "stable",
            "latest": None,
            "average": None,
            "indicators": {},
            "assessment": "Inflation data is limited or unavailable; maintaining a neutral assessment.",
        }

        if events_df.empty:
            return default_res

        df = self._normalize_events_df(events_df)
        if "event_name" not in df.columns or "actual_value" not in df.columns:
            return default_res

        inflation_patterns = ["CPI", "PPI", "PCE", "INFLATION"]
        mask = df["event_name"].astype(str).str.upper().apply(
            lambda name: any(pat in name for pat in inflation_patterns)
        )
        sub_df = df[mask].dropna(subset=["actual_value"])

        if sub_df.empty:
            return default_res

        # Sort chronologically
        sub_df = sub_df.sort_values(by="event_date") if "event_date" in sub_df.columns else sub_df

        latest_val = float(sub_df["actual_value"].iloc[-1])

        # Filter recent 3 months or last 3 releases
        if "event_date" in sub_df.columns and not sub_df["event_date"].isnull().all():
            max_date = sub_df["event_date"].max()
            recent_3m = sub_df[sub_df["event_date"] >= (max_date - pd.DateOffset(days=90))]
            avg_val = float(recent_3m["actual_value"].mean()) if not recent_3m.empty else float(sub_df["actual_value"].tail(3).mean())
        else:
            avg_val = float(sub_df["actual_value"].tail(3).mean())

        # Determine trend
        prev_vals = sub_df["actual_value"].tail(4).values
        if len(prev_vals) >= 2:
            delta = prev_vals[-1] - prev_vals[0]
            if delta > 0.15:
                trend = "rising"
            elif delta < -0.15:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Assessment text
        if trend == "rising":
            assessment = f"Inflationary pressures are accelerating. Latest print of {latest_val:.2f}% sits above recent averages ({avg_val:.2f}%), indicating ongoing price stickiness."
        elif trend == "falling":
            assessment = f"Disinflation trend remains intact. Latest reading of {latest_val:.2f}% moderates below prior averages ({avg_val:.2f}%), supporting monetary easing leeway."
        else:
            assessment = f"Inflation metrics remain largely anchored around {latest_val:.2f}% (3-month average: {avg_val:.2f}%), exhibiting balanced baseline conditions."

        return {
            "trend": trend,
            "latest": round(latest_val, 2),
            "average": round(avg_val, 2),
            "assessment": assessment,
        }

    def analyze_employment_trend(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Filter NFP, Unemployment, and Jobless Claims events and compute employment trajectory.

        Args:
            events_df: Macroeconomic events DataFrame.

        Returns:
            Dictionary with 'trend', 'latest', 'average', and 'assessment'.
        """
        default_res = {
            "trend": "stable",
            "latest": None,
            "average": None,
            "assessment": "Labor market data is insufficient for a conclusive directional trend.",
        }

        if events_df.empty:
            return default_res

        df = self._normalize_events_df(events_df)
        if "event_name" not in df.columns or "actual_value" not in df.columns:
            return default_res

        emp_patterns = ["NON-FARM", "PAYROLL", "UNEMPLOYMENT", "JOBLESS", "NFP"]
        mask = df["event_name"].astype(str).str.upper().apply(
            lambda name: any(pat in name for pat in emp_patterns)
        )
        sub_df = df[mask].dropna(subset=["actual_value"])

        if sub_df.empty:
            return default_res

        sub_df = sub_df.sort_values(by="event_date") if "event_date" in sub_df.columns else sub_df

        # If surprise is available, check positive vs negative surprises
        surprise_col = next((c for c in ["surprise", "surprise_value"] if c in sub_df.columns), None)
        latest_val = float(sub_df["actual_value"].iloc[-1])

        # 3-month average
        if "event_date" in sub_df.columns and not sub_df["event_date"].isnull().all():
            max_date = sub_df["event_date"].max()
            recent_3m = sub_df[sub_df["event_date"] >= (max_date - pd.DateOffset(days=90))]
            avg_val = float(recent_3m["actual_value"].mean()) if not recent_3m.empty else float(sub_df["actual_value"].tail(3).mean())
        else:
            avg_val = float(sub_df["actual_value"].tail(3).mean())

        # Trend calculation (account for surprise if available)
        recent_surprises = sub_df[surprise_col].tail(3).dropna() if surprise_col else []
        if len(recent_surprises) > 0 and recent_surprises.mean() > 0:
            trend = "strengthening"
        elif len(recent_surprises) > 0 and recent_surprises.mean() < 0:
            trend = "cooling"
        else:
            tail_vals = sub_df["actual_value"].tail(3).values
            if len(tail_vals) >= 2:
                trend = "strengthening" if tail_vals[-1] > tail_vals[0] else ("cooling" if tail_vals[-1] < tail_vals[0] else "stable")
            else:
                trend = "stable"

        if trend == "strengthening":
            assessment = f"Labor market displays resilient momentum with hiring exceeding baseline projections (latest print: {latest_val:.2f}, avg: {avg_val:.2f})."
        elif trend == "cooling":
            assessment = f"Labor demand demonstrates softening conditions with prints trending below estimates (latest: {latest_val:.2f}, avg: {avg_val:.2f}), reducing wage-push pressures."
        else:
            assessment = f"Employment conditions remain stable and balanced around recent averages ({avg_val:.2f})."

        return {
            "trend": trend,
            "latest": round(latest_val, 2),
            "average": round(avg_val, 2),
            "assessment": assessment,
        }

    def analyze_growth_trend(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Filter GDP, PMI, and Retail Sales events to assess economic growth trajectory.

        Args:
            events_df: Macroeconomic events DataFrame.

        Returns:
            Dictionary with 'trend', 'latest', 'average', and 'assessment'.
        """
        default_res = {
            "trend": "stable",
            "latest": None,
            "average": None,
            "assessment": "Macroeconomic activity indicators reflect steady baseline activity.",
        }

        if events_df.empty:
            return default_res

        df = self._normalize_events_df(events_df)
        if "event_name" not in df.columns or "actual_value" not in df.columns:
            return default_res

        growth_patterns = ["GDP", "PMI", "RETAIL SALES", "INDUSTRIAL PRODUCTION"]
        mask = df["event_name"].astype(str).str.upper().apply(
            lambda name: any(pat in name for pat in growth_patterns)
        )
        sub_df = df[mask].dropna(subset=["actual_value"])

        if sub_df.empty:
            return default_res

        sub_df = sub_df.sort_values(by="event_date") if "event_date" in sub_df.columns else sub_df
        latest_val = float(sub_df["actual_value"].iloc[-1])

        if "event_date" in sub_df.columns and not sub_df["event_date"].isnull().all():
            max_date = sub_df["event_date"].max()
            recent_3m = sub_df[sub_df["event_date"] >= (max_date - pd.DateOffset(days=90))]
            avg_val = float(recent_3m["actual_value"].mean()) if not recent_3m.empty else float(sub_df["actual_value"].tail(3).mean())
        else:
            avg_val = float(sub_df["actual_value"].tail(3).mean())

        # Check PMI expansion (>50) or GDP positive growth
        if avg_val > 50.0 or (latest_val > avg_val and latest_val > 0):
            trend = "expanding"
        elif avg_val < 48.0 or (latest_val < avg_val and latest_val < 0):
            trend = "contracting"
        else:
            trend = "stable"

        if trend == "expanding":
            assessment = f"Real economic activity is expanding robustly, supported by favorable PMI and GDP trajectory (latest: {latest_val:.2f}, 3M avg: {avg_val:.2f})."
        elif trend == "contracting":
            assessment = f"Macro aggregate demand shows signs of contraction or cyclical slowdown (latest: {latest_val:.2f}, 3M avg: {avg_val:.2f})."
        else:
            assessment = f"Economic activity demonstrates moderate, steady-state growth near trend (latest: {latest_val:.2f}, 3M avg: {avg_val:.2f})."

        return {
            "trend": trend,
            "latest": round(latest_val, 2),
            "average": round(avg_val, 2),
            "assessment": assessment,
        }

    def analyze_fed_policy(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Filter FOMC and Federal Reserve Rate decisions to determine monetary policy stance.

        Args:
            events_df: Macroeconomic events DataFrame.

        Returns:
            Dictionary with 'stance', 'latest_rate', 'trend', and 'assessment'.
        """
        default_res = {
            "stance": "neutral",
            "latest_rate": None,
            "trend": "hold",
            "assessment": "Federal Reserve maintains a data-dependent, neutral monetary stance.",
        }

        if events_df.empty:
            return default_res

        df = self._normalize_events_df(events_df)
        if "event_name" not in df.columns:
            return default_res

        fed_patterns = ["FOMC", "FED", "INTEREST RATE", "FUNDS RATE", "RATE DECISION"]
        mask = df["event_name"].astype(str).str.upper().apply(
            lambda name: any(pat in name for pat in fed_patterns)
        )
        sub_df = df[mask]

        if sub_df.empty:
            return default_res

        sub_df = sub_df.sort_values(by="event_date") if "event_date" in sub_df.columns else sub_df

        latest_rate = None
        if "actual_value" in sub_df.columns and not sub_df["actual_value"].dropna().empty:
            latest_rate = float(sub_df["actual_value"].dropna().iloc[-1])

        # Analyze surprise or change relative to previous
        if "previous_value" in sub_df.columns and "actual_value" in sub_df.columns:
            valid_diff = sub_df.dropna(subset=["actual_value", "previous_value"])
            if not valid_diff.empty:
                last_row = valid_diff.iloc[-1]
                rate_change = float(last_row["actual_value"]) - float(last_row["previous_value"])
                if rate_change > 0:
                    stance = "hawkish"
                    trend = "hiking"
                elif rate_change < 0:
                    stance = "dovish"
                    trend = "cutting"
                else:
                    stance = "neutral"
                    trend = "hold"
            else:
                stance = "neutral"
                trend = "hold"
        else:
            stance = "neutral"
            trend = "hold"

        if stance == "hawkish":
            assessment = f"The FOMC leans hawkish, prioritizing inflation suppression via restrictive benchmark rates (policy rate: {latest_rate if latest_rate is not None else 'N/A'}%)."
        elif stance == "dovish":
            assessment = f"The Federal Reserve demonstrates dovish inclinations, pivoting toward monetary easing and liquidity support."
        else:
            assessment = f"The Fed maintains an observational 'higher-for-longer' or neutral policy posture, preserving flexibility across upcoming FOMC cycles."

        return {
            "stance": stance,
            "latest_rate": round(latest_rate, 2) if latest_rate is not None else None,
            "trend": trend,
            "assessment": assessment,
        }

    def analyze_market_sentiment(self, news_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate news sentiment scores, key themes, and distribution.

        Args:
            news_df: News articles DataFrame with sentiment scores and topics.

        Returns:
            Dictionary with 'overall_sentiment', 'average_score', 'key_themes',
            'sentiment_distribution', and 'assessment'.
        """
        default_res = {
            "overall_sentiment": "neutral",
            "average_score": 0.0,
            "key_themes": ["macro stability", "liquidity conditions"],
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "assessment": "Sentiment across financial commentary remains balanced without decisive directional skew.",
        }

        if news_df.empty:
            return default_res

        df = news_df.copy()
        score_col = next((c for c in ["sentiment_score", "score", "sentiment"] if c in df.columns), None)

        if not score_col or df[score_col].dropna().empty:
            return default_res

        scores = pd.to_numeric(df[score_col], errors="coerce").dropna()
        avg_score = float(scores.mean())

        # Distribution
        pos_count = int((scores > 0.1).sum())
        neg_count = int((scores < -0.1).sum())
        neu_count = int(((scores >= -0.1) & (scores <= 0.1)).sum())

        total = len(scores)
        distribution = {
            "positive": round((pos_count / total) * 100, 1) if total > 0 else 0.0,
            "neutral": round((neu_count / total) * 100, 1) if total > 0 else 0.0,
            "negative": round((neg_count / total) * 100, 1) if total > 0 else 0.0,
        }

        # Stance classification
        if avg_score >= 0.15:
            overall_sentiment = "bullish"
        elif avg_score <= -0.15:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"

        # Key themes extraction
        themes = []
        topic_col = next((c for c in ["query_topic", "topic", "category"] if c in df.columns), None)
        if topic_col:
            top_topics = df[topic_col].dropna().value_counts().head(3).index.tolist()
            themes.extend([str(t) for t in top_topics])

        if not themes and "title" in df.columns:
            # Simple keyword frequency on titles
            title_text = " ".join(df["title"].dropna().astype(str)).lower()
            keywords = ["inflation", "rates", "earnings", "labor", "dollar", "yields", "recession"]
            found = [k for k in keywords if k in title_text]
            themes.extend(found[:3])

        if not themes:
            themes = ["economic resilience", "interest rate trajectory"]

        assessment = (
            f"News sentiment averages {avg_score:+.2f} ({overall_sentiment.upper()}), "
            f"with {distribution['positive']}% positive, {distribution['neutral']}% neutral, "
            f"and {distribution['negative']}% negative coverage focusing on {', '.join(themes)}."
        )

        return {
            "overall_sentiment": overall_sentiment,
            "average_score": round(avg_score, 3),
            "key_themes": themes,
            "sentiment_distribution": distribution,
            "assessment": assessment,
        }

    def compute_macro_score(
        self,
        inflation: Dict[str, Any],
        employment: Dict[str, Any],
        growth: Dict[str, Any],
        fed_policy: Dict[str, Any],
        sentiment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine all analyses into a composite macro score ranging from -100 to +100.
        Positive = bullish macro environment, negative = bearish.

        Weights:
            - Growth (PMI/GDP): 30%
            - Employment: 20%
            - Inflation (Disinflation bullish, acceleration bearish): 20%
            - Fed Policy (Accommodative/Dovish bullish, Hawkish restrictive bearish): 15%
            - Market Sentiment: 15%

        Returns:
            Dict containing score, rating, and component breakdown.
        """
        # 1. Growth component (-100 to +100)
        growth_trend = inflation.get("trend", "stable")
        g_trend = growth.get("trend", "stable")
        if g_trend == "expanding":
            growth_sub = 80.0
        elif g_trend == "contracting":
            growth_sub = -80.0
        else:
            growth_sub = 10.0

        # 2. Employment component
        e_trend = employment.get("trend", "stable")
        if e_trend == "strengthening":
            emp_sub = 75.0
        elif e_trend == "cooling":
            emp_sub = -60.0
        else:
            emp_sub = 15.0

        # 3. Inflation component (Disinflation is bullish for financial assets, rising is bearish)
        i_trend = inflation.get("trend", "stable")
        if i_trend == "falling":
            inf_sub = 70.0
        elif i_trend == "rising":
            inf_sub = -75.0
        else:
            inf_sub = 0.0

        # 4. Fed Policy component
        stance = fed_policy.get("stance", "neutral")
        if stance == "dovish":
            fed_sub = 80.0
        elif stance == "hawkish":
            fed_sub = -70.0
        else:
            fed_sub = 10.0

        # 5. Market Sentiment component (-100 to +100)
        raw_sentiment = sentiment.get("average_score", 0.0)
        # Scaled to [-100, 100]
        sent_sub = float(np.clip(raw_sentiment * 100.0, -100.0, 100.0))

        # Weighted calculation
        composite_score = (
            (growth_sub * 0.30)
            + (emp_sub * 0.20)
            + (inf_sub * 0.20)
            + (fed_sub * 0.15)
            + (sent_sub * 0.15)
        )
        composite_score = float(np.clip(composite_score, -100.0, 100.0))

        # Rating categorization
        if composite_score >= 50.0:
            rating = "Very Bullish"
        elif composite_score >= 15.0:
            rating = "Bullish"
        elif composite_score <= -50.0:
            rating = "Very Bearish"
        elif composite_score <= -15.0:
            rating = "Bearish"
        else:
            rating = "Neutral"

        return {
            "score": round(composite_score, 1),
            "rating": rating,
            "breakdown": {
                "growth": round(growth_sub, 1),
                "employment": round(emp_sub, 1),
                "inflation": round(inf_sub, 1),
                "fed_policy": round(fed_sub, 1),
                "sentiment": round(sent_sub, 1),
            },
        }

    def generate_outlook_report(
        self,
        events_df: pd.DataFrame,
        news_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate a comprehensive 1-page markdown report synthesizing macroeconomic status,
        composite score, sub-pillar assessments, risk vectors, and recommended asset allocations.

        Args:
            events_df: Macro events data.
            news_df: News sentiment data.
            market_df: Optional market performance data.

        Returns:
            Complete Markdown report string.
        """
        # Run individual analytics
        inflation = self.analyze_inflation_trend(events_df)
        employment = self.analyze_employment_trend(events_df)
        growth = self.analyze_growth_trend(events_df)
        fed_policy = self.analyze_fed_policy(events_df)
        sentiment = self.analyze_market_sentiment(news_df)

        macro_score_dict = self.compute_macro_score(
            inflation, employment, growth, fed_policy, sentiment
        )
        score = macro_score_dict["score"]
        rating = macro_score_dict["rating"]
        breakdown = macro_score_dict["breakdown"]

        today_str = datetime.utcnow().strftime("%B %d, %Y")

        # Determine Asset Class Position Recommendations
        if score >= 35.0:
            overweights = ["US_EQUITY (Cyclicals/Tech)", "CRUDE_OIL"]
            underweights = ["US_BOND (Long Duration)", "USD_INDEX"]
            neutrals = ["GOLD"]
            posture_summary = "Pro-cyclical, risk-on positioning capturing expansionary momentum and corporate profitability."
        elif score >= 10.0:
            overweights = ["US_EQUITY (Quality Growth)", "GOLD"]
            underweights = ["USD_INDEX"]
            neutrals = ["US_BOND", "CRUDE_OIL"]
            posture_summary = "Constructive risk allocation balancing quality equities with selective precious metal hedges."
        elif score <= -35.0:
            overweights = ["USD_INDEX (Cash Preservation)", "US_BOND (Treasuries)"]
            underweights = ["US_EQUITY (High Beta)", "CRUDE_OIL"]
            neutrals = ["GOLD"]
            posture_summary = "Defensive capital preservation stance favoring liquidity, sovereign duration, and drawdown insulation."
        elif score <= -10.0:
            overweights = ["US_BOND (Defensive Curve)", "GOLD"]
            underweights = ["US_EQUITY (Cyclicals)"]
            neutrals = ["USD_INDEX", "CRUDE_OIL"]
            posture_summary = "Cautious defensive tilt with elevated fixed income duration and gold reserves."
        else:
            overweights = ["GOLD", "US_BOND (Short-to-Intermediate Duration)"]
            underweights = ["CRUDE_OIL"]
            neutrals = ["US_EQUITY", "USD_INDEX"]
            posture_summary = "Balanced market-neutral posture prioritizing carry, diversification, and non-directional volatility."

        # Key Risks
        key_risks = [
            "**Policy Lag Overreach**: Continued restrictive interest rates delaying economic recovery or inducing liquidity stress in credit markets.",
            "**Commodity Volatility**: Geopolitical supply disruptions re-igniting energy and agricultural input costs, halting disinflation.",
            "**Cross-Asset Volatility Spillovers**: High equity-bond correlation challenging traditional risk parity portfolio allocations.",
        ]

        # Markdown Report Compilation
        report = f"""# CWT Macroeconomics Sentiment & Outlook Report
**Date of Assessment:** {today_str}  
**Analytical Engine:** CWT Macro Intelligence System  

---

## 1. Executive Summary
The macroeconomic environment currently registers a **{rating.upper()}** posture with a Composite Macro Score of **{score:+.1f} / 100**. Economic indicators demonstrate {growth.get('trend', 'stable')} growth characteristics coupled with {inflation.get('trend', 'stable')} inflationary pressures. Portfolio positioning favors a {posture_summary}

---

## 2. Macro Score & Regime Overview

| Macro Composite Score | Environment Rating | Dominant Regime | Risk Bias |
| :---: | :---: | :---: | :---: |
| **{score:+.1f} / 100** | **{rating}** | **{growth.get('trend', 'Neutral').title()} / {inflation.get('trend', 'Anchored').title()}** | **{'Upside Growth' if score > 0 else 'Downside Contraction'}** |

### Factor Pillar Decomposition
- **Economic Growth (30% weight):** `{breakdown['growth']:+.1f} pts`
- **Labor Market (20% weight):** `{breakdown['employment']:+.1f} pts`
- **Inflation Trajectory (20% weight):** `{breakdown['inflation']:+.1f} pts`
- **Monetary Policy Stance (15% weight):** `{breakdown['fed_policy']:+.1f} pts`
- **Market Sentiment & News (15% weight):** `{breakdown['sentiment']:+.1f} pts`

---

## 3. Inflation Analysis
- **Trend:** {inflation['trend'].title()}  
- **Latest Reading:** {inflation['latest'] if inflation['latest'] is not None else 'N/A'}%  
- **3-Month Rolling Average:** {inflation['average'] if inflation['average'] is not None else 'N/A'}%  
- **Assessment:** {inflation['assessment']}  

---

## 4. Employment Analysis
- **Labor Market Momentum:** {employment['trend'].title()}  
- **Recent Indicator Print:** {employment['latest'] if employment['latest'] is not None else 'N/A'}  
- **3-Month Trajectory:** {employment['average'] if employment['average'] is not None else 'N/A'}  
- **Assessment:** {employment['assessment']}  

---

## 5. Growth & Activity
- **Activity Trajectory:** {growth['trend'].title()}  
- **Latest Growth/PMI Metric:** {growth['latest'] if growth['latest'] is not None else 'N/A'}  
- **Recent Benchmark Average:** {growth['average'] if growth['average'] is not None else 'N/A'}  
- **Assessment:** {growth['assessment']}  

---

## 6. Federal Reserve Policy Outlook
- **Policy Stance:** {fed_policy['stance'].title()}  
- **Benchmark Policy Rate:** {fed_policy['latest_rate'] if fed_policy['latest_rate'] is not None else 'Historical Range'}%  
- **Anticipated Trajectory:** {fed_policy['trend'].title()}  
- **Assessment:** {fed_policy['assessment']}  

---

## 7. Market Sentiment & Thematic Narrative
- **Overall Sentiment Tone:** {sentiment['overall_sentiment'].upper()} (Mean Score: {sentiment['average_score']:+.2f})  
- **Distribution:** {sentiment['sentiment_distribution']['positive']}% Positive | {sentiment['sentiment_distribution']['neutral']}% Neutral | {sentiment['sentiment_distribution']['negative']}% Negative  
- **Key Discourse Themes:** {', '.join(sentiment['key_themes'])}  
- **Assessment:** {sentiment['assessment']}  

---

## 8. Key Macroeconomic Risks
1. {key_risks[0]}
2. {key_risks[1]}
3. {key_risks[2]}

---

## 9. Recommended Tactical Positioning

| Strategic Stance | Asset Classes | Tactical Rationale |
| :--- | :--- | :--- |
| **Overweight (OW)** | {', '.join(overweights)} | High risk-adjusted asymmetry aligned with current macro score of {score:+.1f}. |
| **Neutral (N)** | {', '.join(neutrals)} | Hold market weight; balance cyclical upside against duration risk. |
| **Underweight (UW)** | {', '.join(underweights)} | Headwinds from prevailing macro regime and monetary policy conditions. |

*Generated automatically by CWT Macro Economics Sentiment Agent.*
"""
        return report.strip()

    def save_report(self, report_text: str, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Save markdown report to destination file path.

        Args:
            report_text: Markdown content.
            filepath: Target file path. Defaults to reports/macro_outlook.md.

        Returns:
            Resolved absolute filepath string.
        """
        out_path = Path(filepath) if filepath else self.reports_dir / "macro_outlook.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Macro Outlook report successfully saved to %s", out_path)
        return str(out_path.resolve())

    def run(
        self,
        events_df: pd.DataFrame,
        news_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Execute full end-to-end report generation pipeline.

        Args:
            events_df: DataFrame of macro calendar events.
            news_df: DataFrame of news sentiment articles.
            market_df: Optional DataFrame of multi-asset market data.

        Returns:
            Path to the saved markdown report.
        """
        logger.info("Executing MacroOutlookGenerator pipeline...")
        report_md = self.generate_outlook_report(events_df, news_df, market_df)
        saved_path = self.save_report(report_md)
        return saved_path
