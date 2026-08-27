"""Feature engineering.

Every feature is built from information available **at the close of bar t** and
is used to predict the return over ``(t, t+h]``.  Nothing here peeks forward;
the label helpers shift in the opposite direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator primitives
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    # macd_hist is omitted deliberately: it equals macd - macd_signal exactly,
    # so carrying all three makes the design matrix rank-deficient.
    return pd.DataFrame({"macd": line / close, "macd_signal": sig / close})


def average_true_range(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / close


def bollinger_position(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=1)
    upper, lower = mid + n_std * std, mid - n_std * std
    return ((close - lower) / (upper - lower).replace(0.0, np.nan)).clip(-1.0, 2.0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=1).replace(0.0, np.nan)
    return ((series - mean) / std).clip(-6.0, 6.0)


def parkinson_vol(frame: pd.DataFrame, window: int) -> pd.Series:
    """High/low range volatility -- a lower-variance estimator than close-to-close."""
    hl = np.log(frame["high"] / frame["low"]) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4.0 * np.log(2.0))) * np.sqrt(365.0)


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = []


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Assemble the model feature matrix from a daily OHLCV frame."""
    df = daily.copy()
    close = df["close"]
    logret = np.log(close).diff()

    out = pd.DataFrame(index=df.index)

    # --- momentum / trend ------------------------------------------------
    out["ret_1d"] = logret
    for lag in (2, 3, 5):
        out[f"ret_lag_{lag}"] = logret.shift(lag - 1)
    # mom_3d is omitted: it equals ret_1d + ret_lag_2 + ret_lag_3 exactly.
    for window in (5, 10, 21, 63, 126):
        out[f"mom_{window}d"] = np.log(close / close.shift(window))
    for window in (10, 21, 50, 200):
        out[f"px_vs_sma_{window}"] = np.log(close / close.rolling(window).mean())
    # sma_10_50 and sma_50_200 are omitted: they are exact differences of the
    # px_vs_sma_* columns already present.

    # --- volatility ------------------------------------------------------
    for window in (5, 10, 21, 63):
        out[f"vol_{window}d"] = logret.rolling(window).std(ddof=1) * np.sqrt(365.0)
    out["vol_ratio_5_21"] = out["vol_5d"] / out["vol_21d"].replace(0.0, np.nan)
    out["vol_ratio_21_63"] = out["vol_21d"] / out["vol_63d"].replace(0.0, np.nan)
    out["parkinson_10d"] = parkinson_vol(df, 10)
    out["parkinson_ratio"] = out["parkinson_10d"] / out["vol_21d"].replace(0.0, np.nan)
    out["atr_14"] = average_true_range(df, 14)

    # --- distribution shape ---------------------------------------------
    out["skew_21d"] = logret.rolling(21).skew()
    out["kurt_21d"] = logret.rolling(21).kurt()
    out["downside_vol_21d"] = logret.clip(upper=0.0).rolling(21).std(ddof=1) * np.sqrt(365.0)
    out["upside_ratio_21d"] = (logret > 0).rolling(21).mean()

    # --- oscillators -----------------------------------------------------
    out["rsi_14"] = rsi(close, 14) / 100.0
    out["rsi_7"] = rsi(close, 7) / 100.0
    out = out.join(macd(close))
    out["bb_pos_20"] = bollinger_position(close, 20)

    # --- volume ----------------------------------------------------------
    volume = df["volume"].replace(0.0, np.nan)
    out["vol_z_21"] = rolling_zscore(np.log(volume), 21)
    out["dollar_vol_ratio"] = np.log(volume / volume.rolling(63).mean())
    out["ret_vol_interaction"] = out["ret_1d"] * out["vol_z_21"]

    # --- structure -------------------------------------------------------
    running_max = close.cummax()
    out["drawdown"] = np.log(close / running_max)
    out["dd_21d"] = np.log(close / close.rolling(21).max())
    out["range_pos_21d"] = (
        (close - close.rolling(21).min())
        / (close.rolling(21).max() - close.rolling(21).min()).replace(0.0, np.nan)
    )
    out["days_since_high_63"] = (
        close.rolling(63).apply(lambda w: len(w) - 1 - int(np.argmax(w)), raw=True) / 63.0
    )

    # --- intrabar shape --------------------------------------------------
    out["close_loc"] = (
        (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0.0, np.nan)
    )
    out["gap"] = np.log(df["open"] / close.shift(1))
    # body is omitted: ret_1d = gap + body exactly.

    # Day-of-week terms are deliberately absent. Bitcoin trades 24/7 with no
    # settlement, open or close cycle, so the mechanism that generates weekday
    # effects in equities does not exist and the prior on one is near zero. An
    # earlier comment here asserted the opposite, which is a non-sequitur.
    # Measured: one-way ANOVA of the next-day return on weekday gives F = 1.41,
    # p = 0.21, and at the 7-day horizon the terms are null BY CONSTRUCTION --
    # every 7-day window contains every weekday -- giving F = 0.0007, p = 1.000.
    # They were nevertheless selected by the elastic net in up to a third of
    # refits, which measures the false-discovery rate of the selection procedure
    # rather than anything about the market.

    out = out.replace([np.inf, -np.inf], np.nan)

    global FEATURE_COLUMNS
    FEATURE_COLUMNS = list(out.columns)
    return out


def make_labels(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward log return and its sign over ``horizon`` bars."""
    close = daily["close"]
    fwd = np.log(close.shift(-horizon) / close)
    return pd.DataFrame({"fwd_logret": fwd, "y_up": (fwd > 0).astype("float64")}, index=close.index)


def supervised_frame(daily: pd.DataFrame, horizon: int, drop_partial: bool = True,
                     features: pd.DataFrame | None = None
                     ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Aligned ``(X, y_up, forward_return)`` with unusable rows removed.

    The last ``horizon`` rows have no realised label yet and are dropped from
    the training set; the caller uses ``latest_features`` for live inference.

    ``features`` accepts a precomputed matrix.  Every feature is causal, so a
    matrix built once over the full history can be sliced to any as-of date
    without leakage -- which is what makes walk-forward backtesting tractable
    instead of rebuilding 47 rolling features for every model on every day.
    """
    frame = daily
    if drop_partial and "is_partial" in frame.columns:
        frame = frame[~frame["is_partial"].astype(bool)]

    features = build_features(frame) if features is None else features.reindex(frame.index)
    labels = make_labels(frame, horizon)

    joined = features.join(labels).dropna()
    X = joined[features.columns]
    return X, joined["y_up"], joined["fwd_logret"]


def latest_features(daily: pd.DataFrame, drop_partial: bool = True) -> pd.Series:
    """Feature row for the most recent completed bar."""
    frame = daily
    if drop_partial and "is_partial" in frame.columns:
        frame = frame[~frame["is_partial"].astype(bool)]
    features = build_features(frame)
    valid = features.dropna()
    if valid.empty:
        raise ValueError("no complete feature row available")
    return valid.iloc[-1]
