"""Central configuration for the Bitcoin modelling stack."""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("BTC_CACHE_DIR", BASE_DIR / ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------
TICKER = os.environ.get("BTC_TICKER", "BTC-USD")

# Yahoo rate-limits aggressively; we rotate hosts and back off.
YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
YAHOO_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
YAHOO_HISTORY_START = "2014-09-01"

# Seconds a cached payload stays fresh before we re-hit the network.
DAILY_CACHE_TTL = int(os.environ.get("BTC_DAILY_TTL", 15 * 60))
HOURLY_CACHE_TTL = int(os.environ.get("BTC_HOURLY_TTL", 10 * 60))
OPTIONS_CACHE_TTL = int(os.environ.get("BTC_OPTIONS_TTL", 5 * 60))

DERIBIT_BASE = os.environ.get("DERIBIT_BASE", "https://www.deribit.com/api/v2/public/")

# --------------------------------------------------------------------------
# Forecast horizons (in calendar days -- BTC trades 24/7 so a bar is a day)
# --------------------------------------------------------------------------
HORIZONS = {
    "24h": 1,
    "1w": 7,
}
HORIZON_LABELS = {
    "24h": "Next 24 hours",
    "1w": "Next 7 days",
}

# --------------------------------------------------------------------------
# Monte Carlo / model settings
# --------------------------------------------------------------------------
N_PATHS = int(os.environ.get("BTC_N_PATHS", 40_000))
RANDOM_SEED = int(os.environ.get("BTC_SEED", 20260827))

# Annualisation factor for a 24/7 market.
DAYS_PER_YEAR = 365.0

# Risk-free rate fallback when the Deribit forward curve is unavailable.
FALLBACK_RATE = float(os.environ.get("BTC_RATE", 0.04))

# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------
BACKTEST_DAYS = int(os.environ.get("BTC_BACKTEST_DAYS", 1095))   # ~3 years of test points
BACKTEST_MIN_TRAIN = int(os.environ.get("BTC_MIN_TRAIN", 730))   # ~2 years of warm-up
BACKTEST_REFIT_EVERY = int(os.environ.get("BTC_REFIT_EVERY", 30))  # refit cadence in days
BACKTEST_COST_BPS = float(os.environ.get("BTC_COST_BPS", 10.0))  # round-trip cost, basis points

# Run the walk-forward backtest in a background thread at startup when no
# cached result exists. Off by default: it is a ~30 minute job on a machine with
# a couple of spare cores and proportionally longer on a small instance, so it
# should only be enabled where that CPU is actually available.
BACKTEST_ON_START = os.environ.get("BTC_BACKTEST_ON_START", "0") == "1"

# How often the web process regenerates the walk-forward artefact once it has
# one. Render does not allow a disk on a cron job, so a scheduled job cannot
# hand a file to the web service -- the web service owns the artefact itself.
# 0 disables regeneration (generate once, then leave it alone).
BACKTEST_REFRESH_HOURS = float(os.environ.get("BTC_BACKTEST_REFRESH_HOURS", 24))

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
OPTIONS_MAX_DTE = int(os.environ.get("BTC_OPT_MAX_DTE", 10))     # "next week" window
OPTIONS_MIN_DTE = int(os.environ.get("BTC_OPT_MIN_DTE", 1))
OPTIONS_MONEYNESS_BAND = float(os.environ.get("BTC_OPT_BAND", 0.25))  # +/-25% strikes kept

# --------------------------------------------------------------------------
# Dashboard refresh
# --------------------------------------------------------------------------
REFRESH_INTERVAL_SECONDS = int(os.environ.get("BTC_REFRESH_SECONDS", 300))
