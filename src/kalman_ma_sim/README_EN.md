# Kalman Moving Average Trading Strategy Simulation

This directory contains an implementation and backtesting system for a Bitcoin moving average trading strategy using Kalman filters.

## Overview

This project constructs a trading strategy by calculating dynamic moving averages (Kalman MA) from Bitcoin price data using Kalman filters and generating crossover signals with SMA. It also includes hyperparameter optimization functionality using Optuna.

## File Structure

### Core Files

- **`kfma_core.py`**: Core functionality module (pure functions only)
  - Kalman filter moving average calculation
  - Signal generation
  - Vectorized PnL calculation
  - Performance metrics calculation
  - Dataset loading functionality

- **`run_backtest.py`**: CLI wrapper
  - Execution of backtesting and parameter optimization
  - Command-line argument processing
  - Result visualization

- **`btc_kalman_ma_calc.csv`**: Bitcoin price and indicator data

### Legacy Versions (ol/ directory)

Files showing development process history and feature evolution:

#### Basic Implementation
- **`kalman_ma.py`**: Basic Kalman MA implementation and data integrity checks
- **`kalman_ma_step0.py`**: Data preprocessing and cleaning
- **`kalman_ma_step1.py`**: Signal statistics profiling and fake-out analysis

#### Optimization Versions
- **`kalman_ma_opt.py`**: Initial Optuna optimization implementation
- **`kalman_ma_optv2.py`**: Improved version - fee/slippage consideration, risk-adjusted objective function
- **`kalman_ma_optv3.py`**: Latest version - Calmar ratio, PBO analysis, exponential penalties, fixed TP/SL grid

## Key Features

### 1. Kalman Filter Moving Average (Kalman MA Model 3)

Implementation of Benhamou (2016)'s 2-dimensional Kalman filter model:
- ATR-based dynamic process noise
- Price normalization for numerical stability
- 2-component state space model

### 2. Trading Signals

```python
# Long entry conditions
long_entry = (
    (kma.shift(1) < sma.shift(1)) &     # KMA < SMA in previous bar
    (kma >= sma) &                      # Cross confirmed in current bar
    (slope.shift(1) > 0) &              # KMA slope is positive
    atr_filter.shift(1)                 # ATR filter passed
)
```

### 3. Risk Management

- **Stop Loss**: Set as multiples of ATR
- **Take Profit**: Set as multiples of ATR
- **Trailing Stop**: ATR-based dynamic stops (v3)
- **Fees**: Taker fee 0.055%
- **Slippage**: 1bp

### 4. Performance Metrics

- **Sharpe Ratio**: Annualized risk-adjusted returns
- **Calmar Ratio**: Cumulative return / Maximum drawdown (v3)
- **Maximum Drawdown**: Maximum decline percentage
- **PBO (Probability of Backtest Overfitting)**: Overfitting probability (v3)
- **Win Rate**: Ratio of winning trades
- **Number of Trades**: Number of executed trades

### 5. Optuna Optimization

Optimization target parameters:
- `atr_pct`: ATR filter percentile (0.30-0.70)
- `slope_m`: KMA slope calculation period (2-6)
- `beta_kma`: Kalman filter ATR scale coefficient (0.5-2.0)
- `sl_mult`: Stop loss multiplier (1.0-2.5)
- `tp_mult`: Take profit multiplier (2.0-5.0)

Objective function (v3):
```
score = 0.5×Sharpe + 0.5×Calmar - PBO - penalty_trades
penalty_trades = exp(max(0, 100-trade_count)/15) - 1
```

## Usage

### Basic Execution

```bash
python run_backtest.py \
  --train 2024-01-01 2024-12-31 \
  --test 2025-01-01 2025-05-20 \
  --trials 600
```

When executed, the following graph files are automatically generated in the same directory:
- `backtest_train_YYYYMMDD_HHMMSS.png` - Training period analysis results
- `backtest_test_YYYYMMDD_HHMMSS.png` - Test period analysis results

### Parameters

- `--train START END`: Start and end dates for training period
- `--test START END`: Start and end dates for test period
- `--trials N`: Number of Optuna optimization trials (default: 800)
- `--csv FILE`: Data file path (default: btc_kalman_ma_calc.csv)

## Data Requirements

The CSV file must contain the following columns:
- `start_at`: Timestamp
- `close`: Closing price
- `sma`: Simple moving average
- `atr`: Average True Range
- `rsi`: RSI indicator (can be auto-calculated)

## Output

1. **Optimal Parameters**: Optimal parameter set for the training period
2. **Training Period Performance**: In-sample results
3. **Test Period Performance**: Out-of-sample results
4. **Visualization Graph Files**: Saved in high-resolution PNG format (300 DPI)

### Generated Graph Details

Each graph file contains time series analysis composed of 3 panels:

#### Panel 1: Cumulative Return Curve (Cumulative Return %)
- **Y-axis**: Cumulative return percentage (%)
- **Content**: Progression of strategy's cumulative performance
- **Interpretation**: 
  - Upward trend = Accumulating profits
  - Flat = Performance stagnation
  - Downward = Losses occurring

#### Panel 2: Drawdown (DrawDown %)
- **Y-axis**: Drawdown percentage (%)
- **Content**: Progression of decline from past highs
- **Display**: Red filled area
- **Interpretation**:
  - Deep drawdown = High risk
  - Near 0% = Making new highs
  - Maximum value = Maximum drawdown (MaxDD)

#### Panel 3: Rolling Sharpe Ratio (Rolling Sharpe)
- **Y-axis**: Sharpe ratio
- **Content**: Risk-adjusted returns in 7-day (168 hours) moving window
- **Color**: Purple line
- **Interpretation**:
  - 1.0 or higher = Excellent risk-adjusted returns
  - Near 0 = Returns not commensurate with risk
  - Negative = Performance below risk-free rate

### File Naming Convention

```
backtest_{period}_{timestamp}.png
```

Examples:
- `backtest_train_20250124_143025.png` - Training period results from execution at 14:30:25 on January 24, 2025
- `backtest_test_20250124_143025.png` - Test period results from same execution time

This naming convention allows managing execution history without file overwrites during multiple runs.

## Technical Details

### Kalman Filter Implementation

```python
def kalman_ma_model3(close, atr, beta=1.0):
    # 2-dimensional state space model
    # State: [trend component, smoothing component]
    # Observation: price data
    # Dynamic Q: ATR-based process noise
```

### PnL Calculation

Vectorized high-speed PnL calculation:
- Record price & ATR at entry point
- Continuous checking of TP/SL conditions
- Automatic deduction of fees & slippage
- Forced settlement of open positions at period end

### Backtest Reliability

- **Look-ahead bias prevention**: Use shift(1) for all signals
- **Realistic execution**: Consider fees & slippage
- **Statistical validation**: Overfitting detection through PBO analysis

## Important Notes

1. **Data Quality**: Requires consistent 1-hour intervals for hourly data
2. **Computation Time**: Optimization time is proportional to number of trials
3. **Memory Usage**: Sufficient RAM required for large data processing
4. **Performance**: Past performance does not guarantee future results

## Dependencies

- Python 3.11+
- pandas
- numpy
- optuna
- matplotlib
- scikit-learn (optional)

## History

- **v1 (kalman_ma.py)**: Basic Kalman MA implementation
- **v2 (kalman_ma_optv2.py)**: Fee consideration, multi-objective optimization
- **v3 (kalman_ma_optv3.py)**: Calmar ratio, PBO, fixed grid, trailing stops
- **Current (kfma_core.py + run_backtest.py)**: Modularization, pure function design

## ⚠️ Important Disclaimer

**This software is developed solely for educational and research purposes. It is not recommended for use in actual investment or trading.**

### Educational Purpose
- This code is intended for learning financial engineering, data science, and algorithm development
- It aims to provide academic value as an application example of Kalman filters
- The primary purpose is to improve programming skills and promote understanding of financial data analysis methods

### Investment Risk Warnings
1. **Past performance does not guarantee future results**
   - Backtest results are theoretical calculations based on historical data
   - Actual market results may differ significantly due to various factors

2. **High-risk nature of cryptocurrency trading**
   - Bitcoin and other cryptocurrencies have extremely high volatility
   - Prices can fluctuate dramatically in short periods and may fall significantly below initial investment
   - Regulatory environment changes may significantly impact the market

3. **Limitations of algorithmic trading**
   - Algorithms may not function during rapid market changes
   - Technical risks such as system failures and network delays exist
   - Intended execution may not be possible due to insufficient liquidity

### Technical Limitations
- This code is a simplified implementation lacking features necessary for serious trading systems
- Error handling, outlier processing, and real-time processing are limited
- Exchange APIs, order execution, and position management functions are not included

### Usage Precautions
- Code modifications and improvements may change results
- Parameter adjustments may cause overfitting
- The possibility of implementation errors cannot be excluded

### Limitation of Liability
**Developers and related parties assume no responsibility for:**
- Any losses arising from the use of this code
- Financial damages resulting from investment decisions
- Losses due to system failures or data errors
- Impact or damage to third parties

### Proper Investment Practices
If considering investment:
- Consult with financial advisors with professional expertise
- Invest only within the range of surplus funds that you can afford to lose
- Make decisions at your own risk after sufficient research and consideration
- Manage risk appropriately through diversified investment

### License and Terms of Use
By using this software, you are deemed to have agreed to the above disclaimers and limitations. When engaging in commercial use, redistribution, or modification, please check applicable license terms.

---

This system is developed for research and educational purposes and requires sufficient verification and risk management for actual trading use.