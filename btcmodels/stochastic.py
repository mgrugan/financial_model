"""Stochastic (Brownian-motion family) models.

Each one relaxes a different Black-Scholes assumption:

===========================  ======================================================
Model                        Assumption it drops
===========================  ======================================================
Geometric Brownian motion    -- (the baseline: constant drift, constant vol, normal)
Merton jump diffusion        continuity of paths (adds Poisson jumps)
GARCH(1,1)-t                 constant volatility (adds clustering + fat tails)
Heston                       deterministic volatility (vol becomes its own SDE)
Markov regime switching      a single regime (bull/bear states with own drift+vol)
===========================  ======================================================

All of them are calibrated on log returns and simulated forward to the horizon,
returning terminal prices so they plug into the same analytics as the ML models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .base import GARCH_LOOKBACK, BaseModel, MarketContext, fit_garch
from .config import DAYS_PER_YEAR

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Geometric Brownian motion
# ---------------------------------------------------------------------------
class GBMModel(BaseModel):
    key = "gbm"
    name = "Geometric Brownian Motion"
    family = "Stochastic"
    method = "Maximum-likelihood drift/diffusion, closed-form terminal law"
    description = (
        "The Black-Scholes baseline: dS = mu*S*dt + sigma*S*dW. Log returns are "
        "i.i.d. normal, so the horizon distribution is lognormal in closed form. "
        "It ignores volatility clustering and jumps, which is exactly why the "
        "other models exist -- it is the control in this experiment."
    )

    def __init__(self, lookback: int = 365, shrink_drift: float = 0.5) -> None:
        super().__init__()
        self.lookback = lookback
        self.shrink_drift = shrink_drift
        self.mu_ = 0.0
        self.sigma_ = 0.0

    def fit(self, context: MarketContext) -> "GBMModel":
        rets = context.logret[-self.lookback:]
        mu_hat = float(np.mean(rets))
        sigma = float(np.std(rets, ddof=1))
        # Drift is the least identifiable parameter in a diffusion (its standard
        # error only shrinks with calendar span, not sample count), so it is
        # shrunk toward zero rather than trusted at face value.
        self.mu_ = mu_hat * self.shrink_drift
        self.sigma_ = sigma
        self.fitted_ = True
        self.fit_info_ = {
            "mu_daily": self.mu_,
            "sigma_daily": self.sigma_,
            "mu_annual_pct": self.mu_ * DAYS_PER_YEAR * 100.0,
            "sigma_annual_pct": self.sigma_ * np.sqrt(DAYS_PER_YEAR) * 100.0,
            "drift_tstat": mu_hat / (sigma / np.sqrt(len(rets))),
            "lookback_days": self.lookback,
        }
        return self

    def simulate(self, context, horizon_days, n_paths, rng):
        drift = (self.mu_ - 0.5 * self.sigma_**2) * horizon_days
        diffusion = self.sigma_ * np.sqrt(horizon_days) * rng.standard_normal(n_paths)
        terminal = context.spot * np.exp(drift + diffusion)
        return terminal, {"drift_term": drift, "sigma_horizon": self.sigma_ * np.sqrt(horizon_days)}


# ---------------------------------------------------------------------------
# 2. Merton jump diffusion
# ---------------------------------------------------------------------------
class MertonJumpModel(BaseModel):
    key = "merton"
    name = "Merton Jump Diffusion"
    family = "Stochastic"
    method = "Threshold jump detection + Poisson-normal jump calibration"
    description = (
        "GBM plus a compound Poisson jump term: between jumps the price diffuses "
        "normally, and jumps arrive at rate lambda with normally distributed size. "
        "Bitcoin realises several standard-deviation moves per month that a pure "
        "diffusion assigns essentially zero probability, so the jump term carries "
        "most of the tail risk that matters for out-of-the-money options."
    )

    def __init__(self, lookback: int = 730, jump_threshold: float = 3.0) -> None:
        super().__init__()
        self.lookback = lookback
        self.jump_threshold = jump_threshold
        self.mu_ = 0.0
        self.sigma_ = 0.0
        self.lambda_ = 0.0
        self.jump_mean_ = 0.0
        self.jump_std_ = 0.0

    def fit(self, context: MarketContext) -> "MertonJumpModel":
        rets = context.logret[-self.lookback:]
        # Iteratively separate the diffusive core from the jump tail: anything
        # beyond `jump_threshold` robust sigmas is reclassified as a jump.
        scale = float(stats.median_abs_deviation(rets, scale="normal"))
        mask = np.abs(rets - np.median(rets)) < self.jump_threshold * max(scale, 1e-9)
        for _ in range(8):
            core = rets[mask]
            if core.size < 30:
                break
            centre, spread = float(np.mean(core)), float(np.std(core, ddof=1))
            new_mask = np.abs(rets - centre) < self.jump_threshold * spread
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

        core, jumps = rets[mask], rets[~mask]
        self.sigma_ = float(np.std(core, ddof=1)) if core.size > 2 else float(np.std(rets, ddof=1))
        self.mu_ = float(np.mean(core)) * 0.5 if core.size > 2 else 0.0
        self.lambda_ = float(jumps.size) / max(rets.size, 1)          # jumps per day
        self.jump_mean_ = float(np.mean(jumps)) if jumps.size > 1 else 0.0
        self.jump_std_ = float(np.std(jumps, ddof=1)) if jumps.size > 2 else 2.0 * self.sigma_

        self.fitted_ = True
        self.fit_info_ = {
            "sigma_diffusive_annual_pct": self.sigma_ * np.sqrt(DAYS_PER_YEAR) * 100.0,
            "lambda_per_year": self.lambda_ * DAYS_PER_YEAR,
            "n_jumps_detected": int(jumps.size),
            "jump_mean_pct": self.jump_mean_ * 100.0,
            "jump_std_pct": self.jump_std_ * 100.0,
            "jump_variance_share": (
                self.lambda_ * (self.jump_mean_**2 + self.jump_std_**2)
                / max(self.lambda_ * (self.jump_mean_**2 + self.jump_std_**2) + self.sigma_**2, 1e-12)
            ),
        }
        return self

    def simulate(self, context, horizon_days, n_paths, rng):
        # Martingale-preserving compensator for the jump component.
        kappa = np.exp(self.jump_mean_ + 0.5 * self.jump_std_**2) - 1.0
        drift = (self.mu_ - 0.5 * self.sigma_**2 - self.lambda_ * kappa) * horizon_days
        diffusion = self.sigma_ * np.sqrt(horizon_days) * rng.standard_normal(n_paths)

        n_jumps = rng.poisson(self.lambda_ * horizon_days, size=n_paths)
        jump_total = np.zeros(n_paths)
        active = n_jumps > 0
        if active.any():
            counts = n_jumps[active]
            jump_total[active] = (
                counts * self.jump_mean_
                + np.sqrt(counts) * self.jump_std_ * rng.standard_normal(counts.size)
            )

        terminal = context.spot * np.exp(drift + diffusion + jump_total)
        return terminal, {
            "expected_jumps": float(self.lambda_ * horizon_days),
            "paths_with_jump_pct": float(np.mean(active) * 100.0),
        }


# ---------------------------------------------------------------------------
# 3. GARCH(1,1) with Student-t innovations
# ---------------------------------------------------------------------------
class GARCHModel(BaseModel):
    key = "garch"
    name = "GARCH(1,1)-t Volatility"
    family = "Stochastic"
    method = "Quasi-MLE conditional variance, simulated forward with Student-t shocks"
    description = (
        "Volatility is no longer a constant but a recursion: today's variance is "
        "a weighted blend of yesterday's variance and yesterday's squared shock. "
        "That reproduces the volatility clustering Bitcoin actually shows, and "
        "the Student-t innovations keep the tails heavy. The near-unit "
        "persistence (alpha+beta close to 1) means shocks to volatility decay "
        "very slowly, so the current regime dominates the week-ahead forecast."
    )

    def __init__(self, lookback: int = GARCH_LOOKBACK) -> None:
        super().__init__()
        self.lookback = lookback
        self.omega_ = 0.0
        self.alpha_ = 0.08
        self.beta_ = 0.90
        self.nu_ = 5.0
        self.mu_ = 0.0
        self.last_var_ = 1e-4
        self.last_resid_ = 0.0

    def fit(self, context: MarketContext) -> "GARCHModel":
        rets = context.logret[-self.lookback:]
        fitted = False
        targeted = False
        try:
            fit = fit_garch(rets, horizons=(1, 7))
            self.omega_ = fit["omega"]
            self.alpha_ = fit["alpha"]
            self.beta_ = fit["beta"]
            self.nu_ = fit["nu"]
            self.mu_ = fit["mu"]
            self.last_var_ = fit["next_var"]
            self.last_resid_ = 0.0        # the recursion is already rolled forward
            targeted = fit["variance_targeted"]
            fitted = True
        except Exception as exc:  # pragma: no cover
            log.warning("GARCH fit failed (%s); using EWMA fallback", exc)
            lam = 0.94
            var = float(np.var(rets, ddof=1))
            for r in rets:
                var = lam * var + (1 - lam) * r * r
            self.omega_, self.alpha_, self.beta_ = 0.0, 1 - lam, lam
            self.mu_, self.nu_ = 0.0, 4.0
            self.last_var_, self.last_resid_ = var, float(rets[-1])

        persistence = self.alpha_ + self.beta_
        self.fitted_ = True
        self.fit_info_ = {
            "omega": self.omega_,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "persistence": persistence,
            "nu_tail_dof": self.nu_,
            "mu_daily": self.mu_,
            "current_vol_annual_pct": np.sqrt(self.last_var_ * DAYS_PER_YEAR) * 100.0,
            "long_run_vol_annual_pct": (
                np.sqrt(self.omega_ / max(1 - persistence, 1e-6) * DAYS_PER_YEAR) * 100.0
                if persistence < 0.9999 else np.nan
            ),
            "half_life_days": (np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else np.inf,
            "engine": "arch-QMLE" if fitted else "EWMA-fallback",
            "variance_targeted": targeted,
        }
        return self

    def simulate(self, context, horizon_days, n_paths, rng):
        scale = np.sqrt((self.nu_ - 2.0) / self.nu_)   # unit-variance Student-t
        # The starting variance is read from the context, which rolls the same
        # recursion forward every day.  Parameters are slow-moving and are
        # re-estimated on a refit schedule; the conditional variance is the fast
        # state and must be current, which is the entire point of the model.
        start_var = float(context.sigma_daily**2) if context.sigma_daily > 0 else self.last_var_
        var = np.full(n_paths, start_var)
        total = np.zeros(n_paths)

        for step in range(horizon_days):
            if step > 0:
                var = self.omega_ + self.alpha_ * resid**2 + self.beta_ * var
            vol = np.sqrt(var)
            shock = rng.standard_t(self.nu_, size=n_paths) * scale
            resid = vol * shock
            total += self.mu_ + resid

        # Remove the drift's Jensen term so the level is not biased by the tails.
        total = total - np.log(np.mean(np.exp(total - self.mu_ * horizon_days)))
        terminal = context.spot * np.exp(total)
        return terminal, {
            "vol_path_end_annual_pct": float(np.sqrt(np.mean(var) * DAYS_PER_YEAR) * 100.0),
            "start_vol_annual_pct": float(np.sqrt(start_var * DAYS_PER_YEAR) * 100.0),
        }


# ---------------------------------------------------------------------------
# 4. Heston stochastic volatility
# ---------------------------------------------------------------------------
class HestonModel(BaseModel):
    key = "heston"
    name = "Heston Stochastic Volatility"
    family = "Stochastic"
    method = "Method-of-moments calibration on realised variance, full-truncation Euler"
    description = (
        "Volatility gets its own mean-reverting square-root SDE, correlated with "
        "the price shock: dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW2, corr(dW1,dW2)=rho. "
        "A negative rho produces the volatility skew -- crashes come with a "
        "volatility spike -- which is what bends the option smile that a "
        "constant-volatility model cannot reproduce."
    )

    def __init__(self, lookback: int = 1095, ewma_lambda: float = 0.94,
                 shrink_drift: float = 0.5) -> None:
        super().__init__()
        self.lookback = lookback
        self.ewma_lambda = ewma_lambda      # only used if no filtered variance exists
        self.shrink_drift = shrink_drift
        self.v0_ = 1e-4
        self.kappa_ = 3.0
        self.theta_ = 1e-4
        self.xi_ = 0.5
        self.rho_ = -0.4
        self.mu_ = 0.0

    def _variance_proxy(self, context: MarketContext, rets: np.ndarray) -> np.ndarray:
        cond = getattr(context, "cond_var", None)
        if cond is not None and len(cond) >= rets.size and np.std(cond) > 0:
            return np.asarray(cond[-rets.size:], dtype="float64")
        lam = self.ewma_lambda
        rv = np.empty(rets.size)
        var = float(np.var(rets[:63], ddof=1)) if rets.size > 63 else float(np.var(rets, ddof=1))
        for i, r in enumerate(rets):
            var = lam * var + (1.0 - lam) * r * r
            rv[i] = var
        return rv

    def fit(self, context: MarketContext) -> "HestonModel":
        rets = context.logret[-self.lookback:]
        self.mu_ = float(np.mean(rets)) * self.shrink_drift

        # The latent variance proxy is the GARCH-filtered conditional variance
        # already computed for the shared context.  An EWMA is only a fallback:
        # its heavy smoothing suppresses the vol-of-vol and the leverage
        # correlation that this model exists to capture.
        rv = self._variance_proxy(context, rets)
        burn = min(126, rv.size // 4)
        rv, rets_aligned = rv[burn:], rets[burn:]

        self.theta_ = float(np.mean(rv))
        self.v0_ = float(context.sigma_daily**2)

        if rv.size > 60:
            centred = rv - self.theta_
            denom = float(np.dot(centred[:-1], centred[:-1]))
            phi = float(np.dot(centred[:-1], centred[1:]) / denom) if denom > 0 else 0.98
            self.kappa_ = float(-np.log(np.clip(phi, 0.5, 0.9995)))            # per day
            # Method of moments on the CIR stationary law: Var(v) = xi^2*theta/(2*kappa).
            # More stable than differencing, which amplifies filter noise.
            self.xi_ = float(np.clip(
                np.sqrt(2.0 * self.kappa_ * float(np.var(rv, ddof=1)) / max(self.theta_, 1e-12)),
                1e-3, 5.0))
            # Leverage: dv[i] is the variance update driven by rets_aligned[i+1].
            dv = np.diff(rv)
            shocks = rets_aligned[1:]
            if shocks.size == dv.size and np.std(dv) > 0:
                self.rho_ = float(np.clip(np.corrcoef(shocks, dv)[0, 1], -0.95, 0.5))

        feller = 2 * self.kappa_ * self.theta_ / max(self.xi_**2, 1e-12)
        self.fitted_ = True
        self.fit_info_ = {
            "v0_annual_vol_pct": np.sqrt(self.v0_ * DAYS_PER_YEAR) * 100.0,
            "theta_annual_vol_pct": np.sqrt(self.theta_ * DAYS_PER_YEAR) * 100.0,
            "kappa_per_year": self.kappa_ * DAYS_PER_YEAR,
            "xi_vol_of_var": self.xi_,
            "rho_leverage": self.rho_,
            "feller_ratio": feller,
            "var_half_life_days": np.log(2) / max(self.kappa_, 1e-6),
            "vol_of_vol_annual": self.xi_ * np.sqrt(DAYS_PER_YEAR),
            "mu_annual_pct": self.mu_ * DAYS_PER_YEAR * 100.0,
        }
        return self

    def simulate(self, context, horizon_days, n_paths, rng):
        steps_per_day = 4                      # sub-stepping keeps the Euler bias small
        dt = 1.0 / steps_per_day
        n_steps = horizon_days * steps_per_day

        # kappa/theta/xi/rho are estimated on a refit schedule; the *initial*
        # variance is current state and is taken from the context every time.
        v0 = float(context.sigma_daily**2) if context.sigma_daily > 0 else self.v0_
        v = np.full(n_paths, v0)
        log_price = np.zeros(n_paths)
        sqrt_dt = np.sqrt(dt)

        for _ in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = self.rho_ * z1 + np.sqrt(1 - self.rho_**2) * rng.standard_normal(n_paths)
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            log_price += -0.5 * v_pos * dt + sqrt_v * sqrt_dt * z1
            # Full-truncation scheme: the variance may go negative in the update
            # but is floored at zero everywhere it is used.
            v = v + self.kappa_ * (self.theta_ - v_pos) * dt + self.xi_ * sqrt_v * sqrt_dt * z2

        # Neutralise the sampling drift of the shock, then re-apply the model's
        # own estimated drift so the direction call is the model's, not an
        # artefact of the lognormal Jensen term.
        log_price = log_price - np.log(np.mean(np.exp(log_price)))
        log_price = log_price + self.mu_ * horizon_days
        terminal = context.spot * np.exp(log_price)
        return terminal, {
            "terminal_vol_annual_pct": float(np.sqrt(np.mean(np.maximum(v, 0)) * DAYS_PER_YEAR) * 100.0),
            "start_vol_annual_pct": float(np.sqrt(v0 * DAYS_PER_YEAR) * 100.0),
            "sub_steps": n_steps,
        }


# ---------------------------------------------------------------------------
# 5. Two-state Markov regime switching
# ---------------------------------------------------------------------------
class RegimeSwitchingModel(BaseModel):
    key = "regime"
    name = "Markov Regime-Switching"
    family = "Stochastic"
    method = "EM-fitted 2-state Gaussian HMM, simulated with regime transitions"
    description = (
        "Bitcoin alternates between a calm trending regime and a violent one. "
        "This fits a two-state hidden Markov model -- each state has its own "
        "drift and volatility, with a transition matrix governing switches -- "
        "and simulates forward from the current filtered state probability. "
        "It is the only model here whose forecast changes character depending "
        "on which regime the market is judged to be in right now."
    )

    def __init__(self, lookback: int = 1460, n_iter: int = 120,
                 shrink_drift: float = 0.5) -> None:
        super().__init__()
        self.lookback = lookback
        self.n_iter = n_iter
        self.shrink_drift = shrink_drift
        self.mu_ = np.array([0.0, 0.0])
        self.sigma_ = np.array([0.02, 0.05])
        self.trans_ = np.array([[0.97, 0.03], [0.05, 0.95]])
        self.state_prob_ = np.array([0.5, 0.5])

    def _em(self, x: np.ndarray) -> None:
        n = x.size
        # Seed the two states by splitting on rolling volatility.
        roll = pd.Series(x).rolling(21).std(ddof=1).bfill().to_numpy()
        calm = roll <= np.median(roll)
        mu = np.array([x[calm].mean(), x[~calm].mean()])
        sigma = np.array([max(x[calm].std(ddof=1), 1e-6), max(x[~calm].std(ddof=1), 1e-6)])
        trans = np.array([[0.95, 0.05], [0.08, 0.92]])
        pi = np.array([0.5, 0.5])

        for _ in range(self.n_iter):
            dens = np.stack([stats.norm.pdf(x, mu[k], sigma[k]) for k in range(2)], axis=1)
            dens = np.clip(dens, 1e-300, None)

            # Forward-backward with per-step scaling.
            alpha = np.zeros((n, 2))
            scale = np.zeros(n)
            alpha[0] = pi * dens[0]
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0]
            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ trans) * dens[t]
                scale[t] = max(alpha[t].sum(), 1e-300)
                alpha[t] /= scale[t]

            beta = np.zeros((n, 2))
            beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                beta[t] = trans @ (dens[t + 1] * beta[t + 1])
                beta[t] /= max(beta[t].sum(), 1e-300)

            gamma = alpha * beta
            gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)

            xi = np.zeros((2, 2))
            for t in range(n - 1):
                num = trans * np.outer(alpha[t], dens[t + 1] * beta[t + 1])
                xi += num / max(num.sum(), 1e-300)

            new_trans = xi / np.clip(xi.sum(axis=1, keepdims=True), 1e-300, None)
            weights = gamma.sum(axis=0)
            new_mu = (gamma * x[:, None]).sum(axis=0) / np.clip(weights, 1e-300, None)
            new_sigma = np.sqrt(
                (gamma * (x[:, None] - new_mu) ** 2).sum(axis=0) / np.clip(weights, 1e-300, None)
            )
            new_sigma = np.clip(new_sigma, 1e-6, None)

            if (np.max(np.abs(new_mu - mu)) < 1e-9 and np.max(np.abs(new_sigma - sigma)) < 1e-9):
                mu, sigma, trans = new_mu, new_sigma, new_trans
                break
            mu, sigma, trans, pi = new_mu, new_sigma, new_trans, gamma[0]

        # Order states so index 0 is always the calm one.
        order = np.argsort(sigma)
        self.mu_, self.sigma_ = mu[order], sigma[order]
        self.trans_ = trans[np.ix_(order, order)]
        self.trans_ /= self.trans_.sum(axis=1, keepdims=True)
        self.state_prob_ = gamma[-1][order]

    def fit(self, context: MarketContext) -> "RegimeSwitchingModel":
        x = context.logret[-self.lookback:]
        try:
            self._em(x)
        except Exception as exc:  # pragma: no cover
            log.warning("regime EM failed (%s); using volatility split", exc)
            roll = pd.Series(x).rolling(21).std(ddof=1).bfill().to_numpy()
            calm = roll <= np.median(roll)
            self.mu_ = np.array([x[calm].mean(), x[~calm].mean()])
            self.sigma_ = np.array([x[calm].std(ddof=1), x[~calm].std(ddof=1)])
            self.state_prob_ = np.array([1.0, 0.0]) if calm[-1] else np.array([0.0, 1.0])

        # Ergodic (long-run) distribution of the chain.
        eigvals, eigvecs = np.linalg.eig(self.trans_.T)
        stat = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
        stat = np.clip(stat / stat.sum(), 0.0, 1.0)

        self.fitted_ = True
        self.fit_info_ = {
            "calm_vol_annual_pct": self.sigma_[0] * np.sqrt(DAYS_PER_YEAR) * 100.0,
            "stress_vol_annual_pct": self.sigma_[1] * np.sqrt(DAYS_PER_YEAR) * 100.0,
            "calm_drift_annual_pct": self.mu_[0] * DAYS_PER_YEAR * 100.0,
            "stress_drift_annual_pct": self.mu_[1] * DAYS_PER_YEAR * 100.0,
            "p_currently_calm": float(self.state_prob_[0]),
            "p_stay_calm": float(self.trans_[0, 0]),
            "p_stay_stressed": float(self.trans_[1, 1]),
            "expected_calm_duration_days": float(1.0 / max(1 - self.trans_[0, 0], 1e-6)),
            "expected_stress_duration_days": float(1.0 / max(1 - self.trans_[1, 1], 1e-6)),
            "unconditional_p_calm": float(stat[0]),
        }
        return self

    def filtered_state(self, rets: np.ndarray, window: int = 400) -> np.ndarray:
        """Forward-filter the current regime probability under fixed parameters.

        The EM estimation is expensive and its parameters drift slowly, but the
        *filtered state* is the fast-moving part -- it is what tells you whether
        the market is calm or stressed today -- so it is recomputed on every
        call from the most recent returns.
        """
        x = np.asarray(rets[-window:], dtype="float64")
        if x.size < 10:
            return self.state_prob_
        dens = np.stack([stats.norm.pdf(x, self.mu_[k], self.sigma_[k]) for k in range(2)], axis=1)
        dens = np.clip(dens, 1e-300, None)
        alpha = self.state_prob_ * dens[0]
        alpha /= max(alpha.sum(), 1e-300)
        for t in range(1, x.size):
            alpha = (alpha @ self.trans_) * dens[t]
            alpha /= max(alpha.sum(), 1e-300)
        return np.clip(alpha, 0.0, 1.0)

    def simulate(self, context, horizon_days, n_paths, rng):
        state_prob = self.filtered_state(context.logret)
        state = (rng.random(n_paths) > state_prob[0]).astype(int)
        drift = np.zeros(n_paths)
        shock = np.zeros(n_paths)
        stress_days = np.zeros(n_paths)

        for _ in range(horizon_days):
            # Drift is halved: regime means are noisy and extrapolating them
            # fully would make the forecast a momentum bet in disguise.
            drift += self.shrink_drift * self.mu_[state]
            shock += self.sigma_[state] * rng.standard_normal(n_paths)
            stress_days += state
            switch = rng.random(n_paths)
            stay = np.where(state == 0, self.trans_[0, 0], self.trans_[1, 1])
            state = np.where(switch > stay, 1 - state, state)

        # Only the shock is de-meaned; the regime drift is the model's actual
        # directional view and must survive into the terminal distribution.
        shock = shock - np.log(np.mean(np.exp(shock)))
        terminal = context.spot * np.exp(drift + shock)
        return terminal, {
            "avg_stress_days": float(np.mean(stress_days)),
            "p_start_stressed": float(1 - state_prob[0]),
            "p_start_calm": float(state_prob[0]),
            "mean_drift_pct": float(np.mean(drift)) * 100.0,
        }
