"""Recurrent neural model, implemented directly in NumPy.

Everything else here sees each day as an independent row of features.  An LSTM
instead reads an ordered *window* of days and carries a learned state across it,
so it can represent things a flat model structurally cannot -- "volatility has
been compressing for two weeks and price just broke the range" is a statement
about a path, not about a point.

The cell is written out by hand rather than pulled from PyTorch. A CPU-only
Torch install is roughly a gigabyte, which does not fit a small Render
instance, and the whole network here is a few hundred parameters -- NumPy
trains it in seconds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .features import build_features, make_labels
from .supervised import SupervisedModel

log = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LSTMBinaryClassifier:
    """Single-layer LSTM -> dense -> sigmoid, trained with Adam and BPTT."""

    def __init__(self, n_features: int, hidden: int = 10, seed: int = 0,
                 lr: float = 0.004, l2: float = 5e-3, clip: float = 3.0) -> None:
        rng = np.random.default_rng(seed)
        H = hidden
        self.H = H
        self.F = n_features
        self.lr = lr
        self.l2 = l2
        self.clip = clip

        # Xavier for the input map; scaled-orthogonal for the recurrent map so
        # the state neither explodes nor dies over the window.
        self.Wx = rng.normal(0, np.sqrt(1.0 / n_features), (n_features, 4 * H))
        q = np.linalg.qr(rng.normal(0, 1, (H, H)))[0]
        self.Wh = np.concatenate([q, q, q, q], axis=1) * 0.9
        self.b = np.zeros(4 * H)
        self.b[H:2 * H] = 1.0            # forget-gate bias: remember by default
        self.Wy = rng.normal(0, np.sqrt(1.0 / H), (H, 1))
        self.by = np.zeros(1)

        self._params = ["Wx", "Wh", "b", "Wy", "by"]
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._t = 0

    # -- forward ----------------------------------------------------------
    def _forward(self, X: np.ndarray, cache: bool = True):
        B, L, _ = X.shape
        H = self.H
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        steps = [] if cache else None

        for t in range(L):
            x = X[:, t, :]
            z = x @ self.Wx + h @ self.Wh + self.b
            i = _sigmoid(z[:, :H])
            f = _sigmoid(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H])
            o = _sigmoid(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            if cache:
                steps.append((x, h, c, i, f, g, o, tc))
            h, c = h_new, c_new

        logit = (h @ self.Wy + self.by).ravel()
        return logit, h, steps

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logit, _, _ = self._forward(X, cache=False)
        return _sigmoid(logit)

    # -- backward ---------------------------------------------------------
    def _backward(self, X: np.ndarray, y: np.ndarray, logit: np.ndarray,
                  h_last: np.ndarray, steps: list) -> dict[str, np.ndarray]:
        B, L, _ = X.shape
        H = self.H
        p = _sigmoid(logit)
        dlogit = ((p - y) / B).reshape(B, 1)

        grads = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        grads["Wy"] = h_last.T @ dlogit
        grads["by"] = dlogit.sum(axis=0)

        dh = dlogit @ self.Wy.T
        dc = np.zeros((B, H))

        for t in range(L - 1, -1, -1):
            x, h_prev, c_prev, i, f, g, o, tc = steps[t]
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc * tc)
            di, dg, df = dc * g, dc * i, dc * c_prev
            dc_prev = dc * f

            dz = np.concatenate([
                di * i * (1.0 - i),
                df * f * (1.0 - f),
                dg * (1.0 - g * g),
                do * o * (1.0 - o),
            ], axis=1)

            grads["Wx"] += x.T @ dz
            grads["Wh"] += h_prev.T @ dz
            grads["b"] += dz.sum(axis=0)
            dh = dz @ self.Wh.T
            dc = dc_prev

        for key in ("Wx", "Wh", "Wy"):
            grads[key] += self.l2 * getattr(self, key)
        return grads

    def _adam_step(self, grads: dict[str, np.ndarray]) -> None:
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = min(1.0, self.clip / (total + 1e-12))
        for key in self._params:
            g = grads[key] * scale
            self._m[key] = b1 * self._m[key] + (1 - b1) * g
            self._v[key] = b2 * self._v[key] + (1 - b2) * g * g
            m_hat = self._m[key] / (1 - b1 ** self._t)
            v_hat = self._v[key] / (1 - b2 ** self._t)
            setattr(self, key, getattr(self, key) - self.lr * m_hat / (np.sqrt(v_hat) + eps))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self.predict_proba(X), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def state(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).copy() for k in self._params}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            setattr(self, k, v.copy())

    # -- training ---------------------------------------------------------
    @staticmethod
    def _auc(y: np.ndarray, score: np.ndarray) -> float:
        """Rank AUC via the Mann-Whitney identity, ties averaged."""
        pos, neg = int(y.sum()), int((1 - y).sum())
        if pos == 0 or neg == 0:
            return 0.5
        order = np.argsort(score, kind="mergesort")
        ranks = np.empty(len(score), dtype="float64")
        ranks[order] = np.arange(1, len(score) + 1, dtype="float64")
        sorted_score = score[order]
        # Average the ranks inside each tied group.
        start = 0
        for i in range(1, len(sorted_score) + 1):
            if i == len(sorted_score) or sorted_score[i] != sorted_score[start]:
                if i - start > 1:
                    ranks[order[start:i]] = ranks[order[start:i]].mean()
                start = i
        return float((ranks[y > 0].sum() - pos * (pos + 1) / 2.0) / (pos * neg))

    def fit(self, X: np.ndarray, y: np.ndarray, Xval: np.ndarray, yval: np.ndarray,
            epochs: int = 40, batch_size: int = 128, patience: int = 12,
            seed: int = 0) -> int:
        """Train with Adam, selecting the epoch by validation **AUC**.

        Validation log-loss is the wrong stopping signal here.  The base rate
        drifts between the training block and the held-out tail, so log-loss is
        dominated by a level shift the network cannot know about and rises from
        epoch one even while the ranking keeps improving.  AUC is invariant to
        that shift, and the probability level is set afterwards by the Platt
        calibrator, which is fitted on pooled out-of-sample folds.
        """
        rng = np.random.default_rng(seed + 991)
        n = len(X)
        best_auc, best_state, best_epoch = -1.0, self.state(), 0

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                if idx.size < 8:
                    continue
                xb, yb = X[idx], y[idx]
                logit, h_last, steps = self._forward(xb, cache=True)
                self._adam_step(self._backward(xb, yb, logit, h_last, steps))

            auc = self._auc(yval, self.predict_proba(Xval))
            if auc > best_auc + 1e-5:
                best_auc, best_state, best_epoch = auc, self.state(), epoch
            elif epoch - best_epoch >= patience:
                break

        self.load_state(best_state)
        self.best_val_auc = best_auc
        return best_epoch


# ---------------------------------------------------------------------------
# 10. LSTM sequence model
# ---------------------------------------------------------------------------
class LSTMModel(SupervisedModel):
    key = "lstm"
    name = "LSTM Recurrent Network"
    family = "Neural Network"
    method = "NumPy LSTM (hidden 10) over 24-day windows, Adam + BPTT, AUC early stopping, 3-seed ensemble"
    description = (
        "Reads the last 24 days as an ordered sequence and carries a learned "
        "memory state across it, so its input is the shape of the recent path "
        "rather than a snapshot of it. Gates let it hold information over the "
        "window and discard it when the regime turns -- the mechanism that "
        "makes it the natural counterpart to a volatility-clustering model like "
        "GARCH, learned from data instead of assumed."
    )

    # Deliberately narrow: a recurrent net over 47 correlated features and a few
    # thousand noisy windows would fit the noise long before the signal.
    SEQ_FEATURES = [
        "ret_1d", "mom_5d", "mom_21d", "px_vs_sma_10", "px_vs_sma_50",
        "vol_5d", "vol_21d", "vol_ratio_5_21", "rsi_14", "macd_hist",
        "bb_pos_20", "vol_z_21", "drawdown", "close_loc",
    ]
    seq_len = 24
    calib_folds = 2           # each fold is a full ensemble training run
    max_train_rows = 2900

    def __init__(self, n_seeds: int = 3, hidden: int = 10, epochs: int = 40) -> None:
        super().__init__()
        self.n_seeds = n_seeds
        self.hidden = hidden
        self.epochs = epochs

    # -- sequence construction ---------------------------------------------
    def _windows(self, features: pd.DataFrame) -> pd.DataFrame:
        """Flatten each 24-day window into one row so the shared training,
        calibration and weighting machinery works unchanged."""
        cols = [c for c in self.SEQ_FEATURES if c in features.columns]
        block = features[cols].dropna()
        values = block.to_numpy(dtype="float64")
        L = self.seq_len
        if len(values) < L:
            raise ValueError("not enough history for an LSTM window")

        n_windows = len(values) - L + 1
        strided = np.lib.stride_tricks.sliding_window_view(values, L, axis=0)
        # sliding_window_view gives (n, F, L); transpose to (n, L, F) then flatten.
        flat = np.ascontiguousarray(strided.transpose(0, 2, 1)).reshape(n_windows, L * len(cols))
        names = [f"{c}_t-{L - 1 - k}" for k in range(L) for c in cols]
        return pd.DataFrame(flat, index=block.index[L - 1:], columns=names)

    def _prepare(self, context, horizon_days: int):
        frame = context.daily
        if "is_partial" in frame.columns:
            frame = frame[~frame["is_partial"].astype(bool)]
        features = self._feature_frame(context).reindex(frame.index)
        windows = self._windows(features)
        labels = make_labels(frame, horizon_days)
        joined = windows.join(labels).dropna()
        X = joined[windows.columns]
        if self.max_train_rows:
            X = X.iloc[-self.max_train_rows:]
            joined = joined.iloc[-self.max_train_rows:]
        return X, joined["y_up"], joined["fwd_logret"]

    def _live_row(self, context) -> np.ndarray:
        frame = context.daily
        if "is_partial" in frame.columns:
            frame = frame[~frame["is_partial"].astype(bool)]
        windows = self._windows(self._feature_frame(context).reindex(frame.index))
        row = windows.iloc[[-1]]
        if self.feature_names_:
            row = row[self.feature_names_]
        return row.to_numpy(dtype="float64")

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        n_feat = len([c for c in self.SEQ_FEATURES])
        return X.reshape(len(X), self.seq_len, n_feat)

    # -- SupervisedModel hooks ---------------------------------------------
    def purge_gap(self, horizon_days: int) -> int:
        # A 24-day input window means the last training row's inputs overlap the
        # next block's inputs by up to seq_len - 1 days, on top of the label
        # overlap. The point-feature gap of `horizon_days` alone leaves roughly
        # 7% of a validation block contaminated.
        return horizon_days + self.seq_len - 1

    def _make_estimator(self, horizon_days: int):
        # Seeds vary with the refit and the horizon. Fixed seeds 0,1,2 made the
        # initialisation-induced component of the fit COMMON to every refit in a
        # walk-forward, so the whole backtest was conditional on one triple of
        # random initialisations rather than averaging over them.
        base = abs(hash((str(self._context_stamp), horizon_days))) % 100_000
        return {"seeds": [base + 7919 * i for i in range(self.n_seeds)],
                "horizon": horizon_days}

    def _fit_estimator(self, estimator, X, y, weights):
        seq = self._reshape(X)
        # Standardise per feature channel using training statistics only.
        mean = seq.reshape(-1, seq.shape[2]).mean(axis=0)
        std = np.clip(seq.reshape(-1, seq.shape[2]).std(axis=0), 1e-8, None)
        seq = (seq - mean) / std

        n_val = int(np.clip(len(seq) * 0.15, 100, len(seq) // 4))
        cut = len(seq) - n_val
        # Purge the boundary: training rows near `cut` have labels reaching into
        # the validation block AND input windows overlapping it by up to
        # seq_len - 1 days. Early stopping reads that block, so contamination
        # there inflates the very statistic used to choose when to stop.
        purge = self.purge_gap(int(estimator.get("horizon", 1)))
        train_end = max(cut - purge, 10)
        Xtr, ytr, Xva, yva = seq[:train_end], y[:train_end], seq[cut:], y[cut:]

        nets, epochs, aucs = [], [], []
        for seed in estimator["seeds"]:
            net = LSTMBinaryClassifier(seq.shape[2], hidden=self.hidden, seed=int(seed))
            best = net.fit(Xtr, ytr, Xva, yva, epochs=self.epochs, seed=int(seed))
            nets.append(net)
            epochs.append(best)
            aucs.append(getattr(net, "best_val_auc", 0.5))
        return {"nets": nets, "mean": mean, "std": std, "epochs": epochs, "val_aucs": aucs}

    def _member_probas(self, estimator, X: np.ndarray) -> np.ndarray:
        seq = (self._reshape(X) - estimator["mean"]) / estimator["std"]
        return np.stack([net.predict_proba(seq) for net in estimator["nets"]], axis=0)

    def _raw_proba(self, estimator, X):
        return self._member_probas(estimator, X).mean(axis=0)

    def _dispersion(self, estimator, X):
        members = self._member_probas(estimator, X)
        if members.shape[0] < 2:
            return None
        return float(np.std(members[:, 0], ddof=1) / np.sqrt(members.shape[0]))

    def _extra_info(self, estimator, horizon_days):
        return {
            "seq_len": self.seq_len,
            "hidden_units": self.hidden,
            "n_seeds": self.n_seeds,
            "n_seq_features": len(self.SEQ_FEATURES),
            "mean_best_epoch": float(np.mean(estimator["epochs"])),
            "mean_val_auc": float(np.mean(estimator["val_aucs"])),
            "n_parameters": int(4 * self.hidden * (len(self.SEQ_FEATURES) + self.hidden + 1)
                                + self.hidden + 1),
        }

    def feature_importance(self, horizon_days: int, top: int = 12):
        return None
