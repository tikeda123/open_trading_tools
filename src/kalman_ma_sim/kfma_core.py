"""
kfma_core.py -- Core utilities for Kalman‑MA based BTC strategy back‑testing.

This module intentionally contains ZERO side‑effects:
    • indicator calculation (Kalman MA model‑3)
    • signal generation
    • vectorised PnL evaluation
    • aggregate performance metrics
    • dataset integrity helper

All public functions are pure and deterministic, which makes the module
fully unit‑testable and reusable from notebooks / CLI / web apps alike.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────
TAKER_FEE: float = 0.00055   # round turn
SLIP_BPS:   float = 0.00010  # 1 bp
HOURS_PER_YEAR = 24 * 365
ROLL_WIN       = 24 * 7      # 1 week rolling window

_logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ────────────────────────────────────────────────────────────
class Side(IntEnum):
    """Position side (+1 long, −1 short, 0 flat)."""
    FLAT  = 0
    LONG  = 1
    SHORT = -1


@dataclass(frozen=True, slots=True)
class StrategyParam:
    """Hyper‑parameters to optimise."""
    atr_pct:  float
    slope_m:  int
    beta_kma: float
    sl_mult:  float
    tp_mult:  float


# ────────────────────────────────────────────────────────────
# KALMAN MA MODEL 3
# ────────────────────────────────────────────────────────────
def kalman_ma_model3(
    close: np.ndarray,
    atr:   np.ndarray | None = None,
    phi_params: tuple[float, float, float] = (1.0, 0.10, 1.0),
    h_params:   tuple[float, float]        = (1.5, 1.0),
    q_params:   tuple[float, float]        = (0.20, 0.10),
    r_param: float | None                  = None,
    p0_params: tuple[float, float]         = (5.0, 5.0),
    beta: float                            = 1.0,
) -> np.ndarray:
    """
    2‑state Kalman Moving Average (Benhamou 2016, model‑3).

    Parameters
    ----------
    close
        Price series.
    atr
        ATR series used to modulate process noise. Pass ``None`` to use fixed Q.
    *params
        Low‑level model parameters. Defaults provide a robust "follow‑fast"
        behaviour well suited for crypto.

    Returns
    -------
    np.ndarray
        Smoothed price series (length identical to *close*).
    """
    # ① Scale prices to O(1) to reduce floating‑point error
    scale = close[0]
    close_ = close / scale
    atr_   = None if atr is None else atr / scale

    n = len(close_)
    xhat = np.zeros((n, 2))      # state estimates
    yhat = np.zeros(n)           # MA output
    P    = np.zeros((n, 2, 2))   # covariance

    # Transition matrix Φ
    a11, a12, a22 = phi_params
    Phi = np.array([[a11, a12],
                    [0.0, a22]])

    # Observation vector H, normalised so that H.sum() == 1
    h1, h2 = h_params
    H = np.array([h1, h2], dtype=float)
    H /= H.sum()

    # Baseline process noise Q
    q1, q2 = q_params
    Q_base = np.array([[q1**2, q1 * q2],
                       [q1 * q2, q2**2]])

    # Helper – observation noise R_t
    def _obs_noise(idx: int) -> float:
        if atr_ is not None and not np.isnan(atr_[idx]):
            # scale by (ATR / price)²
            return (atr_[idx] / (close_[idx] + 1e-12)) ** 2
        return 1e-4 if r_param is None else r_param

    # Initialisation
    P[0]    = np.diag(p0_params)
    xhat[0] = close_[0]
    yhat[0] = close_[0]

    # Main filter loop
    for t in range(1, n):
        # Predict
        x_pred = Phi @ xhat[t - 1]

        if atr_ is not None and not np.isnan(atr_[t]):
            dyn_scale = beta * (atr_[t] / close_[t]) ** 2
            Q_t = Q_base * dyn_scale
        else:
            Q_t = Q_base
        P_pred = Phi @ P[t - 1] @ Phi.T + Q_t

        # Update
        S = H @ P_pred @ H.T + _obs_noise(t)
        K = (P_pred @ H) / S
        innovation = close_[t] - H @ x_pred
        xhat[t] = x_pred + K * innovation
        P[t]   = (np.eye(2) - np.outer(K, H)) @ P_pred

        yhat[t] = H @ xhat[t]

    return yhat * scale


# ────────────────────────────────────────────────────────────
# DATA CHECK & PRE‑PROCESS
# ────────────────────────────────────────────────────────────
def load_dataset(
    csv_path: str = "BTCUSDT_240_market_data_tech.csv",
    time_col: str = "start_at",
    resample_rule: str = "4h",
) -> pd.DataFrame:
    """
    Minimal I/O helper that:
        • drops duplicates
        • resamples to hourly bars with forward‑fill
        • adds ``rsi`` and ``kf_ma_mod3`` columns if absent
    The function **does not** mutate raw data on disk.
    """
    df = (
        pd.read_csv(csv_path, parse_dates=[time_col])
          .drop_duplicates(time_col)
          .set_index(time_col)
          .sort_index()
          .resample(resample_rule)
          .ffill()
    )

    _logger.info("Loaded %d rows from %s", len(df), csv_path)

    # Compute RSI if missing
    if "rsi" not in df.columns:
        rsi_len = 14
        delta = df["close"].diff()
        up   = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up   = up.rolling(rsi_len).mean()
        roll_down = down.rolling(rsi_len).mean().replace(0, np.nan)
        rs = roll_up / roll_down
        df["rsi"] = 100 - (100 / (1 + rs))

    # Compute Kalman MA if missing
    if "kf_ma_mod3" not in df.columns:
        df["kf_ma_mod3"] = kalman_ma_model3(df["close"].values, df["atr"].values)

    return df.copy()


# ────────────────────────────────────────────────────────────
# SIGNAL GENERATION & PNL
# ────────────────────────────────────────────────────────────
def build_signals(df: pd.DataFrame, p: StrategyParam) -> pd.DataFrame:
    """
    Construct long / short entry signals using:
        • Kalman MA vs 20‑SMA cross
        • slope, ATR & RSI filters
    Resulting boolean columns: ``long_entry`` & ``short_entry``.
    """
    out = df.copy()

    out["kma"] = kalman_ma_model3(df["close"].values, df["atr"].values, beta=p.beta_kma)
    out["sma"] = df["close"].rolling(20).mean()

    slope = out["kma"].diff(p.slope_m) / p.slope_m
    atr_q = df["atr"].rolling(20).quantile(p.atr_pct).shift()

    filt_atr = df["atr"] >= atr_q

    out["long_entry"] = (
        (out["kma"].shift(1) < out["sma"].shift(1)) &
        (out["kma"] >= out["sma"]) &
        (slope.shift(1) > 0) &
        filt_atr.shift(1)
    )
    out["short_entry"] = (
        (out["kma"].shift(1) > out["sma"].shift(1)) &
        (out["kma"] <= out["sma"]) &
        (slope.shift(1) < 0) &
        filt_atr.shift(1)
    )
    return out


def vectorised_pnl(df: pd.DataFrame, p: StrategyParam) -> pd.Series:
    """
    Fast %PnL evaluation – returns series with non‑zero values *only* on exit bars.
    """
    close = df["close"].to_numpy()
    atr   = df["atr"].to_numpy()

    entries_long  = df["long_entry"].to_numpy(dtype=bool)
    entries_short = df["short_entry"].to_numpy(dtype=bool)
    entries_any   = entries_long | entries_short
    dirs = np.where(entries_long, Side.LONG, np.where(entries_short, Side.SHORT, Side.FLAT))

    n = len(df)
    entry_px = np.full(n, np.nan)
    entry_atr = np.zeros(n)
    pos_dir = np.zeros(n, dtype=int)
    pnl_pct = np.zeros(n)

    last_dir = Side.FLAT
    last_px  = np.nan
    last_atr = 0.0

    for i in range(n):
        # New entry
        if entries_any[i] and last_dir == Side.FLAT:
            last_dir = dirs[i]
            last_px  = close[i]
            last_atr = atr[i]
            entry_px[i]  = last_px
            entry_atr[i] = last_atr
            pos_dir[i]   = last_dir
            continue

        pos_dir[i]   = last_dir
        entry_px[i]  = last_px
        entry_atr[i] = last_atr

        if last_dir == Side.FLAT:
            continue  # flat

        rr = (close[i] - last_px) / last_px * last_dir
        tp_hit = rr >= p.tp_mult * last_atr / last_px
        sl_hit = rr <= -p.sl_mult * last_atr / last_px
        exit_now = tp_hit or sl_hit or i == n - 1

        if exit_now:
            cost = TAKER_FEE * 2 + SLIP_BPS
            pnl_pct[i] = rr - cost
            last_dir = Side.FLAT

    return pd.Series(pnl_pct, index=df.index, name="pnl_pct")


# ────────────────────────────────────────────────────────────
# PERFORMANCE
# ────────────────────────────────────────────────────────────
def performance(pnl_pct: pd.Series, full_index: pd.DatetimeIndex) -> Dict[str, float | int | pd.Series]:
    """
    Aggregate performance statistics.
    """
    curve = pnl_pct.cumsum().reindex(full_index).ffill()
    dd = curve.cummax() - curve
    ret_h = curve.diff().fillna(0)

    sharpe = 0.0 if ret_h.std() == 0 else ret_h.mean() / ret_h.std() * np.sqrt(HOURS_PER_YEAR)
    roll_sharpe = ret_h.rolling(ROLL_WIN).apply(
        lambda x: 0.0 if x.std() == 0 else x.mean() / x.std() * np.sqrt(HOURS_PER_YEAR)
    )

    return {
        "tot": float(pnl_pct.sum()),
        "sharpe": float(sharpe),
        "maxdd": float(dd.max()),
        "n": int((pnl_pct != 0).sum()),
        "curve": curve,
        "dd": dd,
        "roll_sharpe": roll_sharpe,
    }
