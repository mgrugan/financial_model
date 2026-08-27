"""Shared contract that every forecasting model implements.

The design goal is that a Monte-Carlo diffusion and a gradient-boosted
classifier are directly comparable and are both usable for option analytics.
So every model must emit the same thing: a **distribution of terminal prices**
at the horizon.  Direction probability, target prices, confidence and option
expected values are all read off that one object.
"""

from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR

log = logging.getLogger(__name__)

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# ---------------------------------------------------------------------------
# Market context -- expensive shared state computed once per refresh
# ---------------------------------------------------------------------------
@dataclass
class MarketContext:
    """State shared by all models so volatility is not re-fitted nine times."""

    daily: pd.DataFrame
    logret: np.ndarray
    spot: float
    as_of: dt.datetime
    sigma_daily: float                        # current conditional daily vol
    sigma_horizon: dict[int, float]           # horizon days -> stdev of horizon log-return
    std_resid: np.ndarray                     # standardised residuals for bootstrapping
    cond_var: np.ndarray                      # filtered conditional variance path
    garch_params: dict[str, float] = field(default_factory=dict)
    features: "pd.DataFrame | None" = None    # precomputed causal feature matrix

    def horizon_sigma(self, horizon_days: int) -> float:
        if horizon_days in self.sigma_horizon:
            return self.sigma_horizon[horizon_days]
        return self.sigma_daily * np.sqrt(horizon_days)


def build_context(daily: pd.DataFrame, horizons: tuple[int, ...] = (1, 7),
                  features: "pd.DataFrame | None" = None,
                  garch_params: dict[str, float] | None = None) -> MarketContext:
    """Fit the volatility backbone once and cache it for every model.

    ``features`` and ``garch_params`` let a caller supply work that has already
    been done -- a causal feature matrix built over the full history, and GARCH
    dynamics estimated at the last refit.  Both are how the walk-forward
    backtest avoids re-deriving identical quantities a thousand times over.
    """
    frame = daily[~daily.get("is_partial", pd.Series(False, index=daily.index)).astype(bool)]
    close = frame["close"]
    logret = np.diff(np.log(close.to_numpy()))

    sigma_daily = float(np.std(logret[-63:], ddof=1))
    sigma_horizon = {h: sigma_daily * np.sqrt(h) for h in horizons}
    std_resid = (logret - logret.mean()) / max(np.std(logret, ddof=1), 1e-12)
    cond_var = np.full(logret.size, sigma_daily**2)
    params: dict[str, float] = {}

    try:
        fit = fit_garch(logret[-GARCH_LOOKBACK:], horizons=horizons, params=garch_params)
        sigma_daily = fit["sigma_daily"]
        sigma_horizon = fit["sigma_horizon"]
        std_resid = fit["std_resid"]
        cond_var = fit["cond_var"]
        params = {
            k: fit[k] for k in
            ("omega", "alpha", "beta", "nu", "mu", "persistence",
             "variance_targeted", "loglik", "half_life_days")
        }
        params["long_run_vol_annual_pct"] = float(
            np.sqrt(fit["long_run_var"] * DAYS_PER_YEAR) * 100.0)
        params["sample_vol_annual_pct"] = float(
            np.sqrt(fit["sample_var"] * DAYS_PER_YEAR) * 100.0)
    except Exception as exc:  # pragma: no cover - falls back to realised vol
        log.warning("GARCH context fit failed (%s); using realised volatility", exc)

    std_resid = std_resid[np.isfinite(std_resid)]
    if std_resid.size < 100:
        std_resid = np.random.default_rng(0).standard_normal(1000)

    return MarketContext(
        daily=frame,
        logret=logret,
        spot=float(close.iloc[-1]),
        as_of=close.index[-1].to_pydatetime(),
        sigma_daily=sigma_daily,
        sigma_horizon=sigma_horizon,
        std_resid=std_resid,
        cond_var=cond_var,
        garch_params=params,
        features=features.reindex(frame.index) if features is not None else None,
    )


# ---------------------------------------------------------------------------
# GARCH backbone (shared by the context and the standalone GARCH model)
# ---------------------------------------------------------------------------
GARCH_LOOKBACK = 1460          # 4 years: long enough to identify, short enough
                               # to exclude the 2017-2021 volatility regime


def fit_garch(rets: np.ndarray, horizons: tuple[int, ...] = (1, 7),
              variance_targeting: bool = True,
              params: dict[str, float] | None = None) -> dict[str, Any]:
    """Fit GARCH(1,1) with Student-t errors and forecast the variance path.

    Two departures from a naive ``arch_model(...).fit()``:

    * **Bounded window.**  Fitted on the whole 12-year history the persistence
      estimates to exactly 1.0 -- an IGARCH degeneracy where the long-run
      variance does not exist and shocks never decay -- and the level is pulled
      up by the 2017-2021 regime.  Four years keeps the parameters identified.
    * **Variance targeting.**  With a free intercept and fat-tailed data the
      implied unconditional volatility came out near 61% against 47% realised
      over the same window.  Following Engle-Mezrich, omega is instead pinned to
      ``sample_var * (1 - alpha - beta)`` so the model's long-run level matches
      the sample by construction, and only the dynamics are estimated.
    """
    rets = np.asarray(rets, dtype="float64")
    sample_var = float(np.var(rets, ddof=1))

    if params is not None:
        # Reuse previously estimated dynamics and only roll the recursion
        # forward.  Used by the backtester: re-estimating the same three
        # parameters on every one of a thousand days is pure waste, while the
        # filtered variance genuinely must update daily.
        alpha, beta = float(params["alpha"]), float(params["beta"])
        mu, nu = float(params["mu"]), float(params["nu"])
        omega = float(params["omega"])
        loglik = float(params.get("loglik", np.nan))
    else:
        from arch import arch_model

        res = arch_model(pd.Series(rets * 100.0), vol="GARCH", p=1, q=1,
                         dist="t", mean="Constant").fit(disp="off", show_warning=False)
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        mu = float(res.params["mu"]) / 100.0
        nu = float(np.clip(res.params.get("nu", 5.0), 2.1, 60.0))
        omega = float(res.params["omega"]) / 10_000.0
        loglik = float(res.loglikelihood)

    persistence = alpha + beta
    targeted = False
    if variance_targeting and persistence < 0.999 and params is None:
        omega = sample_var * (1.0 - persistence)
        targeted = True
    elif params is not None:
        targeted = bool(params.get("variance_targeted", False))

    # Re-run the recursion so the filtered path is consistent with the intercept.
    # Vectorising this is not possible -- it is inherently sequential -- but it
    # is a single pass over a few thousand floats.
    resid = rets - mu
    cond_var = np.empty(rets.size)
    var = sample_var
    for i, e in enumerate(resid):
        cond_var[i] = var
        var = omega + alpha * e * e + beta * var
    next_var = var

    var_path = np.empty(max(horizons))
    v = next_var
    long_run = omega / max(1.0 - persistence, 1e-9)
    for i in range(max(horizons)):
        var_path[i] = v
        v = omega + persistence * v
    std_resid = resid / np.clip(np.sqrt(cond_var), 1e-12, None)

    return {
        "omega": omega, "alpha": alpha, "beta": beta, "nu": nu, "mu": mu,
        "persistence": persistence, "variance_targeted": targeted,
        "sample_var": sample_var, "long_run_var": long_run,
        "cond_var": cond_var, "std_resid": std_resid,
        "next_var": next_var, "var_path": var_path,
        "sigma_horizon": {h: float(np.sqrt(var_path[:h].sum())) for h in horizons},
        "sigma_daily": float(np.sqrt(var_path[0])),
        "loglik": loglik,
        "half_life_days": float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else np.inf,
    }


# ---------------------------------------------------------------------------
# Forecast object
# ---------------------------------------------------------------------------
@dataclass
class ModelForecast:
    """A model's full view of the horizon, not just an up/down call."""

    model_key: str
    model_name: str
    family: str
    horizon_key: str
    horizon_days: int
    spot: float
    terminal_prices: np.ndarray
    p_up: float
    p_up_stderr: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # -- derived statistics ------------------------------------------------
    @property
    def log_returns(self) -> np.ndarray:
        return np.log(self.terminal_prices / self.spot)

    @property
    def expected_price(self) -> float:
        return float(np.mean(self.terminal_prices))

    @property
    def median_price(self) -> float:
        return float(np.median(self.terminal_prices))

    @property
    def expected_return(self) -> float:
        """Expected simple return over the horizon, in percent."""
        return float(np.mean(self.terminal_prices) / self.spot - 1.0) * 100.0

    @property
    def median_return(self) -> float:
        return float(np.median(self.terminal_prices) / self.spot - 1.0) * 100.0

    @property
    def sigma_horizon(self) -> float:
        """Standard deviation of the horizon log return."""
        return float(np.std(self.log_returns, ddof=1))

    @property
    def annualised_vol(self) -> float:
        return self.sigma_horizon * np.sqrt(DAYS_PER_YEAR / self.horizon_days) * 100.0

    @property
    def direction(self) -> str:
        return "UP" if self.p_up >= 0.5 else "DOWN"

    @property
    def edge(self) -> float:
        """Distance from a coin flip, on 0-1."""
        return abs(self.p_up - 0.5) * 2.0

    @property
    def confidence_label(self) -> str:
        edge = self.edge
        if edge < 0.04:
            return "Coin-flip"
        if edge < 0.12:
            return "Weak"
        if edge < 0.25:
            return "Moderate"
        if edge < 0.45:
            return "Strong"
        return "Very strong"

    def quantile_prices(self, qs: tuple[float, ...] = QUANTILES) -> dict[str, float]:
        values = np.quantile(self.terminal_prices, qs)
        return {f"q{int(q * 100):02d}": float(v) for q, v in zip(qs, values)}

    def prob_above(self, level: float) -> float:
        return float(np.mean(self.terminal_prices > level))

    def prob_below(self, level: float) -> float:
        return float(np.mean(self.terminal_prices < level))

    def prob_move_beyond(self, pct: float) -> float:
        """P(|return| exceeds ``pct`` percent)."""
        move = np.abs(self.terminal_prices / self.spot - 1.0)
        return float(np.mean(move > pct / 100.0))

    def expected_shortfall(self, alpha: float = 0.05) -> float:
        """Mean return in the worst ``alpha`` tail, in percent."""
        rets = self.terminal_prices / self.spot - 1.0
        cutoff = np.quantile(rets, alpha)
        tail = rets[rets <= cutoff]
        return float(tail.mean()) * 100.0 if tail.size else float(cutoff) * 100.0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "family": self.family,
            "horizon_key": self.horizon_key,
            "horizon_days": self.horizon_days,
            "spot": self.spot,
            "p_up": self.p_up,
            "p_down": 1.0 - self.p_up,
            "p_up_stderr": self.p_up_stderr,
            "direction": self.direction,
            "edge": self.edge,
            "confidence_label": self.confidence_label,
            "expected_price": self.expected_price,
            "median_price": self.median_price,
            "expected_return_pct": self.expected_return,
            "median_return_pct": self.median_return,
            "sigma_horizon": self.sigma_horizon,
            "annualised_vol_pct": self.annualised_vol,
            "expected_shortfall_5pct": self.expected_shortfall(0.05),
            "prob_move_5pct": self.prob_move_beyond(5.0),
            "prob_move_10pct": self.prob_move_beyond(10.0),
            "notes": self.notes,
            "diagnostics": self.diagnostics,
        }
        payload.update(self.quantile_prices())
        return payload


# ---------------------------------------------------------------------------
# Distribution helpers shared by the models
# ---------------------------------------------------------------------------
def tilt_to_probability(logret_samples: np.ndarray, p_up: float) -> np.ndarray:
    """Shift a zero-drift return sample so that ``P(return > 0) == p_up``.

    This is what lets a *classifier* produce a full price distribution: the
    shape (fat tails, volatility clustering) comes from the stochastic-volatility
    backbone, while the machine-learning model supplies only the location.  The
    shift is exact by construction -- ``c = -Quantile(samples, 1 - p_up)``.
    """
    p_up = float(np.clip(p_up, 1e-4, 1 - 1e-4))
    shift = -np.quantile(logret_samples, 1.0 - p_up)
    return logret_samples + shift


def bootstrap_innovations(
    context: MarketContext, horizon_days: int, n_paths: int, rng: np.random.Generator,
    sigma_override: float | None = None, block: int = 1,
) -> np.ndarray:
    """Zero-mean horizon log returns with realistic tails.

    Standardised GARCH residuals are resampled (optionally in blocks, to retain
    short-run autocorrelation) and rescaled to the model's horizon volatility.
    """
    resid = context.std_resid
    sigma_h = sigma_override if sigma_override is not None else context.horizon_sigma(horizon_days)
    sigma_step = sigma_h / np.sqrt(horizon_days)

    if block <= 1:
        draws = rng.choice(resid, size=(n_paths, horizon_days), replace=True)
    else:
        n_blocks = int(np.ceil(horizon_days / block))
        starts = rng.integers(0, max(len(resid) - block, 1), size=(n_paths, n_blocks))
        offsets = np.arange(block)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
        draws = resid[np.clip(idx, 0, len(resid) - 1)][:, :horizon_days]

    draws = draws - draws.mean()
    draws = draws / max(np.std(draws, ddof=1), 1e-12)
    total = (draws * sigma_step).sum(axis=1)
    # Normalise to an exact martingale in price space.  With nu ~ 3 tails the
    # closed-form -0.5*sigma^2 Ito term is not accurate enough, so the drift is
    # solved numerically from the sample itself.
    return total - np.log(np.mean(np.exp(total)))


def prob_stderr(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1.0 - p), 1e-12) / max(n, 1)))


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------
class BaseModel(ABC):
    """Interface shared by stochastic, machine-learning and neural models."""

    key: str = "base"
    name: str = "Base"
    family: str = "stochastic"
    description: str = ""
    method: str = ""

    def __init__(self) -> None:
        self.fitted_ = False
        self.fit_info_: dict[str, Any] = {}

    @abstractmethod
    def fit(self, context: MarketContext) -> "BaseModel":
        """Calibrate on everything up to ``context.as_of``."""

    @abstractmethod
    def simulate(self, context: MarketContext, horizon_days: int, n_paths: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        """Return ``(terminal_prices, diagnostics)`` for the horizon."""

    def forecast(self, context: MarketContext, horizon_key: str, horizon_days: int,
                 n_paths: int, rng: np.random.Generator) -> ModelForecast:
        if not self.fitted_:
            self.fit(context)
        terminal, diagnostics = self.simulate(context, horizon_days, n_paths, rng)
        terminal = np.asarray(terminal, dtype="float64")
        terminal = terminal[np.isfinite(terminal) & (terminal > 0)]
        if terminal.size == 0:
            raise RuntimeError(f"{self.key}: simulation produced no valid paths")

        p_up = float(np.mean(terminal > context.spot))
        stderr = diagnostics.pop("p_up_stderr", None)
        if stderr is None:
            stderr = prob_stderr(p_up, terminal.size)

        return ModelForecast(
            model_key=self.key,
            model_name=self.name,
            family=self.family,
            horizon_key=horizon_key,
            horizon_days=horizon_days,
            spot=context.spot,
            terminal_prices=terminal,
            p_up=p_up,
            p_up_stderr=float(stderr),
            diagnostics={**self.fit_info_, **diagnostics},
            notes=self.description,
        )

    # Models that predict direction directly override this so the backtester can
    # score them without running a full simulation.
    def predict_proba_up(self, context: MarketContext, horizon_days: int,
                         n_paths: int, rng: np.random.Generator) -> tuple[float, float]:
        terminal, _ = self.simulate(context, horizon_days, n_paths, rng)
        p = float(np.mean(terminal > context.spot))
        return p, prob_stderr(p, len(terminal))
