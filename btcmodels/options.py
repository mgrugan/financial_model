"""Options analytics for the week-ahead BTC book.

Pricing uses **Black-76 on the forward** rather than Black-Scholes on spot.
Crypto options are quoted and hedged against the dated future for their own
expiry, and Deribit publishes that forward per expiry, so using it removes the
basis error that shows up as a fake skew when spot is used instead.

Two distinct numbers appear throughout and must not be confused:

``mark`` / greeks
    The **market's** risk-neutral valuation, computed from Deribit's mark IV.
    This is what the contract costs and how it will behave.

``model_ev``
    The discounted expected payoff under a **forecasting model's real-world**
    terminal distribution.  This is not an arbitrage-free price -- it is what
    the option is worth *if that model's view of the world is correct*.  The
    difference between the two is the model's claimed edge, and it is only as
    good as the model's out-of-sample record on the backtest page.

A note on contract mechanics: Deribit BTC options are inverse and cash-settled
in BTC, one BTC per contract, so a premium quoted at 0.01 BTC costs
``0.01 * index`` in dollars. The USD-linear greeks below are the standard
sensitivities a directional trader wants; a delta-hedged book also carries the
BTC-denominated exposure that inverse settlement creates.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

from .base import ModelForecast
from .config import DAYS_PER_YEAR

log = logging.getLogger(__name__)

SQRT_2PI = np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Black-76
# ---------------------------------------------------------------------------
def _d1_d2(F, K, T, sigma):
    F = np.asarray(F, dtype="float64")
    K = np.asarray(K, dtype="float64")
    T = np.clip(np.asarray(T, dtype="float64"), 1e-8, None)
    sigma = np.clip(np.asarray(sigma, dtype="float64"), 1e-6, None)
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol_t**2) / vol_t
    return d1, d1 - vol_t, vol_t


def black76_price(F, K, T, sigma, r, option_type: str = "call"):
    d1, d2, _ = _d1_d2(F, K, T, sigma)
    disc = np.exp(-np.asarray(r, dtype="float64") * np.clip(T, 1e-8, None))
    if option_type == "call":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_greeks(F, K, T, sigma, r, option_type: str = "call",
                   spot: float | None = None) -> dict[str, np.ndarray]:
    """Full greek set, expressed per 1 BTC of notional in USD.

    Sensitivities are taken with respect to **spot**, which is what a trader
    actually hedges.  With ``F = S*exp(rT)`` the discount factor and the forward's
    own spot sensitivity cancel, so delta reduces to ``N(d1)`` exactly as in
    Black-Scholes with no dividend.
    """
    F = np.asarray(F, dtype="float64")
    K = np.asarray(K, dtype="float64")
    T = np.clip(np.asarray(T, dtype="float64"), 1e-8, None)
    sigma = np.clip(np.asarray(sigma, dtype="float64"), 1e-6, None)
    r = np.asarray(r, dtype="float64")

    d1, d2, vol_t = _d1_d2(F, K, T, sigma)
    pdf = np.exp(-0.5 * d1**2) / SQRT_2PI
    disc = np.exp(-r * T)
    S = F * disc if spot is None else np.asarray(spot, dtype="float64")

    is_call = option_type == "call"
    sign = 1.0 if is_call else -1.0

    delta = sign * norm.cdf(sign * d1)
    gamma = pdf / np.clip(S * vol_t, 1e-12, None)
    vega = S * pdf * np.sqrt(T)                       # per 1.00 of vol
    theta_year = (
        -S * pdf * sigma / (2.0 * np.sqrt(T))
        - sign * r * K * disc * norm.cdf(sign * d2)
    )
    rho = sign * K * T * disc * norm.cdf(sign * d2)

    # Second-order: how delta itself moves with vol and time.
    vanna = -pdf * d2 / sigma
    volga = vega * d1 * d2 / sigma
    charm = -pdf * (2 * r * T - d2 * vol_t) / (2 * T * vol_t)

    price = black76_price(F, K, T, sigma, r, option_type)
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100.0,                          # per 1 volatility point
        "theta": theta_year / DAYS_PER_YEAR,           # per calendar day
        "theta_year": theta_year,
        "rho": rho / 100.0,                            # per 1% rate move
        "vanna": vanna / 100.0,
        "volga": volga / 10_000.0,
        "charm": charm / DAYS_PER_YEAR,                # delta lost per day
        "d1": d1,
        "d2": d2,
        "rn_prob_itm": norm.cdf(sign * d2),            # risk-neutral P(finish ITM)
    }


def implied_vol(price: float, F: float, K: float, T: float, r: float,
                option_type: str = "call") -> float:
    """Invert Black-76 for volatility (Brent, with an intrinsic-value guard)."""
    disc = np.exp(-r * T)
    intrinsic = disc * max(F - K, 0.0) if option_type == "call" else disc * max(K - F, 0.0)
    upper = disc * (F if option_type == "call" else K)
    if not np.isfinite(price) or price <= intrinsic + 1e-9 or price >= upper - 1e-9:
        return float("nan")

    def objective(sigma: float) -> float:
        return float(black76_price(F, K, T, sigma, r, option_type)) - price

    try:
        return float(optimize.brentq(objective, 1e-4, 12.0, maxiter=120, xtol=1e-8))
    except (ValueError, RuntimeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Chain enrichment
# ---------------------------------------------------------------------------
def enrich_chain(chain: pd.DataFrame, rate: float, spot: float,
                 moneyness_band: float = 0.25,
                 delta_band: tuple[float, float] = (0.04, 0.96)) -> pd.DataFrame:
    """Attach greeks (from market mark IV) to each live contract.

    Contracts outside ``delta_band`` are dropped.  The deep wings of a crypto
    book are quoted but not really traded: a 0.76-moneyness call carries a
    reported 130% mark IV, delta pinned at 1.000 and gamma of order 1e-10.  It
    is a forward in disguise, its "implied volatility" is an artefact of a stale
    quote, and any expected-value edge computed on it is just the model's drift
    restated.  Screening on delta rather than on strike distance removes them
    without also removing legitimately far strikes when volatility is high.
    """
    if chain.empty:
        return chain

    df = chain.copy()
    df = df[
        df["mark_iv"].notna()
        & (df["mark_iv"] > 0.05) & (df["mark_iv"] < 4.0)
        & (df["moneyness"] > 1 - moneyness_band)
        & (df["moneyness"] < 1 + moneyness_band)
        & (df["mark_usd"] > 0)
    ].copy()
    if df.empty:
        return df

    rows = []
    for option_type, block in df.groupby("option_type"):
        greeks = black76_greeks(block["forward"].to_numpy(), block["strike"].to_numpy(),
                                block["years"].to_numpy(), block["mark_iv"].to_numpy(),
                                rate, option_type, spot=spot)
        enriched = block.copy()
        for name, values in greeks.items():
            enriched[name] = values
        enriched["model_price_check"] = greeks["price"]
        rows.append(enriched)

    out = pd.concat(rows).sort_values(["expiry", "option_type", "strike"])
    abs_delta = out["delta"].abs()
    out["tradeable"] = (abs_delta >= delta_band[0]) & (abs_delta <= delta_band[1])
    out = out[out["tradeable"]].copy()
    if out.empty:
        return out
    out["theta_pct_of_premium"] = out["theta"] / out["mark_usd"].replace(0, np.nan) * 100.0
    out["breakeven"] = np.where(out["option_type"] == "call",
                                out["strike"] + out["mark_usd"],
                                out["strike"] - out["mark_usd"])
    out["spread_pct"] = (out["ask_usd"] - out["bid_usd"]) / out["mark_usd"].replace(0, np.nan) * 100.0
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model-based valuation
# ---------------------------------------------------------------------------
def scale_distribution(forecast: ModelForecast, target_days: float) -> np.ndarray:
    """Re-express a model's terminal distribution at a different horizon.

    A model forecasts 1 or 7 days; an expiry sits somewhere in between.  The
    log-return sample is rescaled by ``sqrt(t/h)`` for the diffusive part and
    ``t/h`` for the drift, which is the standard square-root-of-time
    convention and preserves the sample's skew and kurtosis rather than
    replacing them with a normal approximation.
    """
    logret = forecast.log_returns
    ratio = max(target_days, 1e-6) / max(forecast.horizon_days, 1e-6)
    if abs(ratio - 1.0) < 1e-9:
        return forecast.terminal_prices
    drift = float(np.mean(logret))
    centred = logret - drift
    scaled = drift * ratio + centred * np.sqrt(ratio)
    return forecast.spot * np.exp(scaled)


def value_chain_under_model(chain: pd.DataFrame, forecast: ModelForecast,
                            rate: float, spot: float,
                            max_samples: int = 20_000) -> pd.DataFrame:
    """Expected payoff of every contract under one model's own distribution.

    Vectorised across strikes: the terminal sample is rescaled once per distinct
    expiry and then broadcast against every strike at that expiry.  Looping row
    by row costs about a second per model, and this runs for all eleven models
    on every refresh.
    """
    if chain.empty:
        return chain

    out = chain.copy()
    rng = np.random.default_rng(12345)
    columns = ["model_ev", "model_p_itm", "model_p_profit", "model_edge_usd"]
    for col in columns:
        out[col] = np.nan

    for (dte, option_type), block in out.groupby(["dte", "option_type"], sort=False):
        terminal = scale_distribution(forecast, float(dte))
        if terminal.size > max_samples:
            terminal = rng.choice(terminal, size=max_samples, replace=False)

        strikes = block["strike"].to_numpy(dtype="float64")[None, :]
        grid = terminal[:, None]
        if option_type == "call":
            payoff = np.maximum(grid - strikes, 0.0)
            itm = (grid > strikes).mean(axis=0)
        else:
            payoff = np.maximum(strikes - grid, 0.0)
            itm = (grid < strikes).mean(axis=0)

        disc = np.exp(-rate * block["years"].to_numpy(dtype="float64"))
        fair = payoff.mean(axis=0) * disc
        premium = block["mark_usd"].to_numpy(dtype="float64")
        p_profit = (payoff > premium[None, :]).mean(axis=0)

        out.loc[block.index, "model_ev"] = fair
        out.loc[block.index, "model_p_itm"] = itm
        out.loc[block.index, "model_p_profit"] = p_profit
        out.loc[block.index, "model_edge_usd"] = fair - premium

    out["model_edge_pct"] = out["model_edge_usd"] / out["mark_usd"].replace(0, np.nan) * 100.0
    # Edge per dollar at risk, which is the number that should drive selection:
    # a 200% edge on a $40 lottery ticket is not better than 30% on a $900 call.
    out["model_edge_per_premium"] = out["model_edge_usd"] / out["mark_usd"].replace(0, np.nan)
    out["model_iv"] = [
        implied_vol(float(row["model_ev"]), float(row["forward"]), float(row["strike"]),
                    float(row["years"]), rate, str(row["option_type"]))
        for _, row in out.iterrows()
    ]
    out["iv_gap_pts"] = (out["model_iv"] - out["mark_iv"]) * 100.0
    return out


def atm_implied_vol(chain: pd.DataFrame, expiry: Any | None = None) -> float:
    """Mark IV of the strikes nearest the forward, in annualised percent."""
    if chain.empty:
        return float("nan")
    df = chain if expiry is None else chain[chain["expiry"] == expiry]
    if df.empty:
        return float("nan")
    distance = (df["strike"] / df["forward"] - 1.0).abs()
    nearest = df.loc[distance.nsmallest(min(6, len(df))).index]
    return float(nearest["mark_iv"].median()) * 100.0


def volatility_context(chain: pd.DataFrame, forecast: ModelForecast,
                       expiry: Any | None = None) -> dict[str, float]:
    """Model volatility against what the option market is charging.

    This comparison belongs next to every expected-value number on the page.
    Buying an option because a model says it is cheap is, mechanically, a bet
    that realised volatility will beat implied -- and if the model's volatility
    runs above the market's across the whole surface, every contract will look
    cheap and the "edge" is one volatility opinion, not a set of independent
    findings.
    """
    market_iv = atm_implied_vol(chain, expiry)
    model_vol = forecast.annualised_vol
    return {
        "model_vol_pct": model_vol,
        "market_atm_iv_pct": market_iv,
        "vol_gap_pts": model_vol - market_iv,
        "model_richer": bool(np.isfinite(market_iv) and model_vol > market_iv),
    }


def theta_decay_schedule(row: pd.Series, rate: float, spot: float,
                         n_points: int = 12) -> pd.DataFrame:
    """Value of a contract as expiry approaches, holding spot and IV fixed.

    Isolating time decay this way is the only way to see it: in live prices it
    is buried under moves in spot and vol many times its size.
    """
    dte = float(row["dte"])
    days = np.linspace(0.0, dte, n_points)
    remaining = np.clip(dte - days, 1e-6, None) / DAYS_PER_YEAR
    prices = black76_price(float(row["forward"]), float(row["strike"]), remaining,
                           float(row["mark_iv"]), rate, str(row["option_type"]))
    greeks = black76_greeks(float(row["forward"]), float(row["strike"]), remaining,
                            float(row["mark_iv"]), rate, str(row["option_type"]), spot=spot)
    start = float(prices[0])
    return pd.DataFrame({
        "days_elapsed": days,
        "days_to_expiry": np.clip(dte - days, 0.0, None),
        "value": prices,
        "pct_of_today": prices / max(start, 1e-9) * 100.0,
        "cumulative_decay_usd": prices - start,
        "theta_per_day": greeks["theta"],
        "gamma": greeks["gamma"],
        "delta": greeks["delta"],
    })


# ---------------------------------------------------------------------------
# Per-model trade suggestions
# ---------------------------------------------------------------------------
DELTA_TARGETS = {
    "ITM": 0.70,
    "ATM": 0.50,
    "OTM": 0.28,
}


def suggest_trades(valued: pd.DataFrame, forecast: ModelForecast,
                   expiry: pd.Timestamp | None = None,
                   min_open_interest: float = 1.0) -> list[dict[str, Any]]:
    """Pick one ITM, one ATM and one OTM contract in the model's own direction.

    Direction follows the model's P(up).  Within that direction, the contract
    closest to each target delta is chosen, subject to a liquidity floor --
    a theoretical edge on a contract with no open interest is not a trade.
    """
    if valued.empty:
        return []

    df = valued
    if expiry is not None:
        df = df[df["expiry"] == expiry]
    if df.empty:
        return []

    bullish = forecast.p_up >= 0.5
    side = "call" if bullish else "put"
    book = df[df["option_type"] == side].copy()
    liquid = book[book["open_interest"] >= min_open_interest]
    if len(liquid) >= 3:
        book = liquid
    if book.empty:
        return []

    book["abs_delta"] = book["delta"].abs()
    quantiles = forecast.quantile_prices()
    suggestions = []

    for label, target in DELTA_TARGETS.items():
        pick = book.iloc[(book["abs_delta"] - target).abs().argsort()].iloc[0]
        dte = float(pick["dte"])

        # Target price: the model's own median for a directional hold, and its
        # upper/lower quartile as the stretch objective.
        scaled = scale_distribution(forecast, dte)
        target_mid = float(np.median(scaled))
        target_stretch = float(np.quantile(scaled, 0.75 if bullish else 0.25))

        suggestions.append({
            "moneyness_label": label,
            "instrument": pick["instrument"],
            "option_type": side,
            "strike": float(pick["strike"]),
            "expiry": pick["expiry"],
            "dte": dte,
            "premium_usd": float(pick["mark_usd"]),
            "premium_btc": float(pick["mark_btc"]),
            "premium_pct_of_spot": float(pick["mark_usd"]) / float(pick["index_price"]) * 100.0,
            "breakeven": float(pick["breakeven"]),
            "breakeven_move_pct": (float(pick["breakeven"]) / float(pick["index_price"]) - 1.0) * 100.0,
            "delta": float(pick["delta"]),
            "gamma": float(pick["gamma"]),
            "theta_usd_per_day": float(pick["theta"]),
            "theta_pct_per_day": float(pick["theta_pct_of_premium"]),
            "vega": float(pick["vega"]),
            "rho": float(pick["rho"]),
            "vanna": float(pick["vanna"]),
            "volga": float(pick["volga"]),
            "charm": float(pick["charm"]),
            "market_iv_pct": float(pick["mark_iv"]) * 100.0,
            "model_iv_pct": float(pick["model_iv"]) * 100.0 if np.isfinite(pick["model_iv"]) else float("nan"),
            "iv_gap_pts": float(pick["iv_gap_pts"]) if np.isfinite(pick["iv_gap_pts"]) else float("nan"),
            "rn_prob_itm_pct": float(pick["rn_prob_itm"]) * 100.0,
            "model_prob_itm_pct": float(pick["model_p_itm"]) * 100.0,
            "model_prob_profit_pct": float(pick["model_p_profit"]) * 100.0,
            "model_ev_usd": float(pick["model_ev"]),
            "model_edge_usd": float(pick["model_edge_usd"]),
            "model_edge_pct": float(pick["model_edge_pct"]),
            "open_interest": float(pick["open_interest"]),
            "target_price_median": target_mid,
            "target_price_stretch": target_stretch,
            "model_q25": quantiles["q25"],
            "model_q50": quantiles["q50"],
            "model_q75": quantiles["q75"],
            "model_q95": quantiles["q95"],
            "max_loss_usd": float(pick["mark_usd"]),
        })
    return suggestions


def vertical_spread(valued: pd.DataFrame, forecast: ModelForecast,
                    expiry: pd.Timestamp | None = None) -> dict[str, Any] | None:
    """A debit vertical in the model's direction.

    Buying the ~50 delta and selling the ~25 delta caps the payoff but cuts the
    premium, and with it the theta bill -- which matters here because every
    expiry in the window is under ten days out and decay dominates.
    """
    if valued.empty:
        return None
    df = valued if expiry is None else valued[valued["expiry"] == expiry]
    bullish = forecast.p_up >= 0.5
    side = "call" if bullish else "put"
    book = df[(df["option_type"] == side) & (df["open_interest"] > 0)].copy()
    if len(book) < 2:
        return None

    book["abs_delta"] = book["delta"].abs()
    long_leg = book.iloc[(book["abs_delta"] - 0.50).abs().argsort()].iloc[0]
    short_pool = book[book["strike"] > long_leg["strike"]] if bullish else book[book["strike"] < long_leg["strike"]]
    if short_pool.empty:
        return None
    short_leg = short_pool.iloc[(short_pool["abs_delta"] - 0.25).abs().argsort()].iloc[0]

    debit = float(long_leg["mark_usd"] - short_leg["mark_usd"])
    width = abs(float(short_leg["strike"] - long_leg["strike"]))
    if debit <= 0:
        return None

    scaled = scale_distribution(forecast, float(long_leg["dte"]))
    if bullish:
        payoff = np.clip(scaled - float(long_leg["strike"]), 0, width)
    else:
        payoff = np.clip(float(long_leg["strike"]) - scaled, 0, width)
    disc = float(np.exp(-0.0 * long_leg["years"]))

    return {
        "structure": f"{'Bull call' if bullish else 'Bear put'} spread",
        "long_instrument": long_leg["instrument"],
        "short_instrument": short_leg["instrument"],
        "long_strike": float(long_leg["strike"]),
        "short_strike": float(short_leg["strike"]),
        "expiry": long_leg["expiry"],
        "dte": float(long_leg["dte"]),
        "net_debit_usd": debit,
        "max_profit_usd": width - debit,
        "max_loss_usd": debit,
        "risk_reward": (width - debit) / debit if debit > 0 else float("nan"),
        "breakeven": float(long_leg["strike"]) + (debit if bullish else -debit),
        "net_delta": float(long_leg["delta"] - short_leg["delta"]),
        "net_gamma": float(long_leg["gamma"] - short_leg["gamma"]),
        "net_theta_usd_per_day": float(long_leg["theta"] - short_leg["theta"]),
        "net_vega": float(long_leg["vega"] - short_leg["vega"]),
        "model_ev_usd": float(np.mean(payoff)) * disc,
        "model_edge_usd": float(np.mean(payoff)) * disc - debit,
        "model_prob_profit_pct": float(np.mean(payoff > debit)) * 100.0,
    }
