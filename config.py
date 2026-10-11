"""
CWT Macro Economics Sentiment Agent - Configuration
Central configuration loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

# Create directories
for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Database ───────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'cwt_macro.db'}")

# ── Pipeline Settings ──────────────────────────────────────────────────────
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))
WALK_FORWARD_TRAIN_DAYS = int(os.getenv("WALK_FORWARD_TRAIN_DAYS", "120"))
WALK_FORWARD_TEST_DAYS = int(os.getenv("WALK_FORWARD_TEST_DAYS", "30"))

# ── Macroeconomic Events of Interest ──────────────────────────────────────
MACRO_EVENTS = [
    "CPI", "Core CPI", "PPI",
    "Non-Farm Payrolls", "Unemployment Rate", "Initial Jobless Claims",
    "FOMC", "Fed Interest Rate Decision", "Fed Funds Rate",
    "GDP", "GDP Growth Rate",
    "Retail Sales", "Consumer Confidence", "Consumer Sentiment",
    "ISM Manufacturing PMI", "ISM Services PMI",
    "Industrial Production", "Durable Goods Orders",
    "Housing Starts", "Building Permits",
    "Trade Balance", "Current Account",
    "PCE Price Index", "Core PCE",
]

# ── Trading Strategy Universe ─────────────────────────────────────────────
ASSET_CLASSES = ["US_EQUITY", "US_BOND", "GOLD", "USD_INDEX", "CRUDE_OIL"]
STRATEGY_SIGNALS = ["LONG", "SHORT", "NEUTRAL"]

# ── Model Configuration ──────────────────────────────────────────────────
MODEL_PARAMS = {
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": 0,
    },
}
