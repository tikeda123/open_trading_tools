"""
run_backtest.py -- CLI wrapper for Kalman‑MA strategy back‑test & optimisation.

Usage example
-------------
python run_backtest.py --train 2022-01-01 2023-01-01 \
                       --test  2023-01-02 2023-07-01 \
                       --trials 500
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import optuna
from optuna.trial import Trial

from kfma_core import (
    StrategyParam,
    build_signals,
    load_dataset,
    performance,
    vectorised_pnl,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# OPTUNA
# ────────────────────────────────────────────────────────────
def objective(trial: Trial, df_train):
    """Composite score: Sharpe − 0.1·MaxDD + 0.001·N − penalty(<100 trades)."""
    param = StrategyParam(
        atr_pct=trial.suggest_float("atr_pct", 0.30, 0.70, step=0.01),
        slope_m=trial.suggest_int("slope_m", 2, 6),
        beta_kma=trial.suggest_float("beta_kma", 0.5, 2.0),
        sl_mult=trial.suggest_float("sl_mult", 1.0, 2.5, step=0.1),
        tp_mult=trial.suggest_float("tp_mult", 2.0, 5.0, step=0.1),
    )

    sig  = build_signals(df_train, param)
    pnl  = vectorised_pnl(sig, param)
    stats = performance(pnl, df_train.index)

    penalty = max(0, 100 - stats["n"]) * 0.01
    score = stats["sharpe"] - 0.1 * stats["maxdd"] + 0.001 * stats["n"] - penalty

    trial.set_user_attr("stats", stats)  # save full statistics
    if trial.number % 50 == 0:
        _logger.info(
            "Trial %d → %.3f  (Sharpe %.2f, MaxDD %.2f, N %d)",
            trial.number, score, stats["sharpe"], stats["maxdd"], stats["n"]
        )
    return score


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalman‑MA strategy back‑tester")
    parser.add_argument("--train", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--test",  nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--csv",   default="btc_kalman_ma_calc.csv")
    parser.add_argument("--trials", type=int, default=800)
    return parser.parse_args()


def _plot_timeseries(stats: dict, title: str, filename: str) -> None:
    """Create and save 3-panel timeseries plot to file."""
    # Set matplotlib to use non-interactive backend
    plt.ioff()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax1.plot(stats["curve"] * 100, label="Cumulative %")
    ax1.set_ylabel("CumRet %"); ax1.grid(); ax1.legend(loc="upper left")

    ax2.fill_between(stats["dd"].index, stats["dd"] * 100, color="red", alpha=.3)
    ax2.set_ylabel("DrawDown %"); ax2.grid()

    ax3.plot(stats["roll_sharpe"]); ax3.set_ylabel("Rolling Sharpe"); ax3.grid()
    ax3.set_xlabel("Datetime")

    fig.suptitle(title)
    fig.tight_layout()
    
    # Save to file in same directory
    filepath = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    _logger.info("Plot saved to: %s", filepath)


def run_backtest() -> None:
    args = parse_args()

    df_all = load_dataset(args.csv)

    df_train = df_all.loc[args.train[0]: args.train[1]]
    df_test  = df_all.loc[args.test [0]: args.test [1]]

    _logger.info("Rows → train %d, test %d", len(df_train), len(df_test))

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, df_train),
                   n_trials=args.trials,
                   n_jobs=-1,
                   show_progress_bar=True)

    best_param = StrategyParam(**study.best_params)  # type: ignore[arg-type]
    _logger.info("Best param set: %s", best_param)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for tag, df in (("TRAIN", df_train), ("TEST", df_test)):
        sig  = build_signals(df, best_param)
        pnl  = vectorised_pnl(sig, best_param)
        stats = performance(pnl, df.index)
        _logger.info("%s: tot %.4f / Sharpe %.2f / MaxDD %.2f / N %d",
                     tag, stats["tot"], stats["sharpe"], stats["maxdd"], stats["n"])
        
        # Create filename with timestamp and tag
        filename = f"backtest_{tag.lower()}_{timestamp}.png"
        _plot_timeseries(stats, f"{tag}  Equity / DD / Rolling Sharpe", filename)


if __name__ == "__main__":
    run_backtest()
