"""Risk limits and broker settings.

Defaults are deliberately conservative and every one of them can be raised by
the operator -- it is their money and their call. What cannot be bypassed by
configuration alone is the set of hard ceilings at the bottom of this file:
those exist so that a typo in an environment variable cannot turn a $1,000
account into a margin call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LEDGER_DIR = Path(os.environ.get("TRADER_LEDGER_DIR", BASE_DIR / "ledger"))

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE = "https://api.alpaca.markets"
ALPACA_DATA_BASE = "https://data.alpaca.markets"

# --- hard ceilings: not configurable -------------------------------------
# A single fat-fingered env var should not be able to concentrate the whole
# account into one name or lever it up.
HARD_MAX_POSITION_PCT = 0.35
HARD_MAX_GROSS_EXPOSURE = 1.5
HARD_MIN_HOLDINGS = 3


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class RiskLimits:
    """Pre-trade constraints. Every order is checked against all of them."""

    # Fraction of equity allowed in any single position.
    max_position_pct: float = field(default_factory=lambda: _f("TRADER_MAX_POSITION_PCT", 0.12))
    # Total long exposure as a fraction of equity. 1.0 = fully invested, no margin.
    max_gross_exposure: float = field(default_factory=lambda: _f("TRADER_MAX_GROSS", 0.98))
    # Halt trading for the session if the account is down this much from its
    # recorded high-water mark.
    max_drawdown_pct: float = field(default_factory=lambda: _f("TRADER_MAX_DRAWDOWN", 0.25))
    # Skip any order smaller than this; below it, spread and rounding dominate.
    min_order_notional: float = field(default_factory=lambda: _f("TRADER_MIN_ORDER", 5.0))
    # Do not rebalance a position whose weight is already within this band of
    # its target. Churning a 0.4% drift costs more in spread than it corrects.
    rebalance_band: float = field(default_factory=lambda: _f("TRADER_REBALANCE_BAND", 0.25))
    # Refuse to trade a name whose quoted spread is wider than this.
    max_spread_pct: float = field(default_factory=lambda: _f("TRADER_MAX_SPREAD", 0.01))

    def clamp(self) -> list[str]:
        """Apply the hard ceilings, reporting anything that had to be reduced."""
        notes = []
        if self.max_position_pct > HARD_MAX_POSITION_PCT:
            notes.append(f"max_position_pct {self.max_position_pct:.0%} -> "
                         f"{HARD_MAX_POSITION_PCT:.0%} (hard ceiling)")
            self.max_position_pct = HARD_MAX_POSITION_PCT
        if self.max_gross_exposure > HARD_MAX_GROSS_EXPOSURE:
            notes.append(f"max_gross_exposure {self.max_gross_exposure:.2f} -> "
                         f"{HARD_MAX_GROSS_EXPOSURE:.2f} (hard ceiling)")
            self.max_gross_exposure = HARD_MAX_GROSS_EXPOSURE
        return notes


@dataclass
class BrokerConfig:
    key_id: str = ""
    secret_key: str = ""
    live: bool = False

    @classmethod
    def from_env(cls, live: bool = False) -> "BrokerConfig":
        return cls(
            key_id=os.environ.get("ALPACA_KEY_ID", ""),
            secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
            live=live,
        )

    @property
    def base_url(self) -> str:
        return ALPACA_LIVE_BASE if self.live else ALPACA_PAPER_BASE

    @property
    def configured(self) -> bool:
        return bool(self.key_id and self.secret_key)
