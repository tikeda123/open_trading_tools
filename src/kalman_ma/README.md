# Kalman Filter Moving Average Implementation

## Project Description

This project is **open-source educational code** that was originally shared on Twitter for learning purposes. The implementation demonstrates advanced Kalman filter techniques applied to financial time series, specifically focusing on cryptocurrency price data analysis.

**⚠️ Important Notice:**
- **Educational Purpose Only**: This code is intended for learning and research purposes
- **Not Investment Advice**: This is not financial advice and should not be used for actual trading decisions
- **No Warranty**: Use at your own risk - no guarantees on performance or accuracy
- **Learning Resource**: Designed to help understand Kalman filter applications in finance

This module implements advanced Kalman filter-based moving averages for financial time series analysis, with particular focus on cryptocurrency trading applications for educational exploration.

## Environment Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Quick Setup

#### Option 1: Using requirements.txt (Recommended)

```bash
# Clone or download the project files
# Navigate to the kalman_ma directory
cd path/to/kalman_ma/

# Install all required dependencies
pip install -r requirements.txt
```

#### Option 2: Manual Installation

```bash
# Install core dependencies
pip install numpy>=1.20.0 pandas>=1.3.0 matplotlib>=3.5.0
```

#### Option 3: Using Virtual Environment (Best Practice)

```bash
# Create a virtual environment
python -m venv kalman_ma_env

# Activate the virtual environment
# On Windows:
kalman_ma_env\Scripts\activate
# On macOS/Linux:
source kalman_ma_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# When done, deactivate the environment
deactivate
```

### Dependencies

**Core Requirements:**
```
numpy>=1.20.0      # Numerical computing and matrix operations
pandas>=1.3.0      # Data manipulation and CSV handling
matplotlib>=3.5.0  # Plotting and visualization
```

**Optional Dependencies:**
```
optuna>=2.10.0          # Bayesian optimization for parameter tuning
scikit-learn>=1.0.0     # Additional statistical tools
scipy>=1.7.0            # Scientific computing functions
statsmodels>=0.12.0     # Statistical analysis and backtesting
```

### Verification

After installation, verify your setup:

```python
# Test import of all required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print("✅ All dependencies installed successfully!")

# Run the main demonstration
python kalman_ma.py
```

### Troubleshooting Installation

#### Common Issues:

**1. Permission Errors:**
```bash
# Use --user flag to install for current user only
pip install --user -r requirements.txt
```

**2. Outdated pip:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**3. Python Version Conflicts:**
```bash
# Explicitly use Python 3
python3 -m pip install -r requirements.txt
```

**4. Missing System Dependencies (Linux/macOS):**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# macOS with Homebrew
brew install python3
```

### Development Setup

If you plan to contribute or modify the code:

```bash
# Install development dependencies
pip install pytest>=6.2.0 black>=21.0.0 flake8>=3.9.0

# Format code
black kalman_ma.py

# Run linting
flake8 kalman_ma.py

# Run tests (if available)
pytest
```

## Overview

The implementation provides three distinct Kalman filter approaches for price smoothing and trend following:

1. **ATR-Based Dynamic Kalman MA** - Volatility-adaptive single-component filter
2. **Fixed-Q Kalman MA** - Traditional single-component filter with constant parameters
3. **Enhanced Model 3** - Two-component system based on Benhamou (2016) research

### Model 3 Improvements for Practical Trading

Kalman MAモデルに改善を加えたモデル３を以下の論文をもとに、より現実的なトレードにおいて役立つように改良を加えました。

**参考文献：TREND WITHOUT HICCUPS - A KALMAN FILTER APPROACH**

https://arxiv.org/pdf/1808.03297

#### Key Enhancements Made:

1. **Two-Factor Structure**: Implementation of Benhamou's Model 3 with separate short-term and long-term components for better trend tracking
2. **Price Normalization**: Added numerical stability through price scaling to initial value = 1
3. **Dynamic ATR Integration**: Enhanced volatility adaptation using Average True Range
4. **Optimized Parameters**: Tuned parameters specifically for cryptocurrency trading scenarios
5. **Practical Trading Focus**: Balanced responsiveness vs. stability for real-world trading applications

## Mathematical Foundation

### Basic Kalman Filter Model

The fundamental state-space model uses a random walk assumption:

```
State Equation:    x_t = x_{t-1} + w_t
Observation Eq:    y_t = x_t + v_t

Where:
- w_t ~ N(0, Q) : Process noise
- v_t ~ N(0, R) : Observation noise
```

### Enhanced Model 3 Structure

Based on Eric Benhamou's research, Model 3 extends to a two-dimensional system:

```
State Vector:      X_t = [x_t^(1), x_t^(2)]'  (short-term, long-term)
Transition Matrix: Φ = [[a11, a12], [0, a22]]
Observation Eq:    y_t = h1*x_t^(1) + h2*x_t^(2) + v_t
```

## Usage Examples

### Basic Usage

```python
import numpy as np
import pandas as pd
from kalman_ma import kalman_ma, kalman_ma_atr, kalman_ma_model3

# Load your price data
df = pd.read_csv('btc_kalman_ma_calc.csv')
close_prices = df['close'].values
atr_values = df['atr'].values

# Method 1: Fixed-parameter Kalman MA
kf_basic = kalman_ma(close_prices, Q=1e-5, R=1e-2)

# Method 2: ATR-adaptive Kalman MA
kf_atr = kalman_ma_atr(atr_values, close_prices, beta=0.01)

# Method 3: Enhanced two-component Model 3
kf_model3 = kalman_ma_model3(close_prices, atr=atr_values)
```

### Running the Demo

```bash
# Make sure you have the sample data file
# btc_kalman_ma_calc.csv should be in the same directory

# Run the main demonstration
python kalman_ma.py

# This will:
# 1. Load Bitcoin price data
# 2. Calculate all three Kalman MA variants
# 3. Display a comparative plot
```

## File Structure

```
kalman_ma/
├── kalman_ma.py              # Main implementation
├── kalman_ma_english.py      # Fully documented English version
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── 分析結果.md               # Japanese analysis document
├── generate_graph.py         # Graph generation utility
├── btc_kalman_ma_calc.csv   # Sample Bitcoin data
├── kalman_ma_vs_close.png   # Sample comparison graph
└── kalman_ma_comparison.png # Generated comparison graph
```

## API Reference

### Function Signatures

#### kalman_ma_atr()
```python
def kalman_ma_atr(atr: np.ndarray, close: np.ndarray,
                  beta: float = 0.01, R: float = 1e-2) -> np.ndarray
```
Returns filtered price series with ATR-based dynamic adaptation.

#### kalman_ma()
```python
def kalman_ma(series: np.ndarray, Q: float = 1e-5, 
              R: float = 1e-2) -> np.ndarray
```
Returns filtered price series with fixed parameters.

#### kalman_ma_model3()
```python
def kalman_ma_model3(close: np.ndarray, atr: np.ndarray | None = None,
                     phi_params=(1.0, 0.10, 1.0), h_params=(1.5, 1.0),
                     q_params=(0.20, 0.10), r_param=None,
                     p0_params=(5.0, 5.0), beta: float = 1.0) -> np.ndarray
```
Returns enhanced two-component Kalman MA with advanced features.

## Performance Characteristics

### Tracking vs Smoothness Trade-off

| Method | Tracking Speed | Smoothness | Stability | Best Use Case |
|--------|---------------|------------|-----------|---------------|
| Fixed Q | Medium | High | High | Stable markets |
| ATR-based | High | Medium | Medium | Volatile markets |
| Model 3 | High | High | High | All conditions |

## Parameter Tuning Guidelines

**For More Responsiveness:**
```python
# Increase process noise
kf_responsive = kalman_ma(prices, Q=1e-4, R=1e-2)

# Or increase ATR influence
kf_atr_responsive = kalman_ma_atr(atr, prices, beta=0.05)

# Or adjust Model 3 Q parameters
kf_mod3_responsive = kalman_ma_model3(prices, q_params=(0.30, 0.15))
```

**For More Smoothing:**
```python
# Decrease process noise, increase observation noise
kf_smooth = kalman_ma(prices, Q=1e-6, R=1e-1)

# Reduce ATR influence
kf_atr_smooth = kalman_ma_atr(atr, prices, beta=0.005)
```

## Common Issues and Solutions

### Numerical Instability
**Problem**: Filter divergence or extreme values
**Solution**: 
- Enable price normalization in Model 3
- Reduce initial covariance values
- Check for data quality issues (NaN, outliers)

### Over-smoothing
**Problem**: Filter too slow to react to trend changes
**Solution**:
- Increase Q (process noise)
- Increase beta (ATR influence)
- Adjust h_params to favor short-term component

### Over-sensitivity
**Problem**: Filter too reactive, following noise
**Solution**:
- Decrease Q (process noise)  
- Increase R (observation noise)
- Reduce beta (ATR influence)

## License and Disclaimer

This implementation is provided for **educational and research purposes only**.

### Educational Use
- Originally shared on Twitter as open-source learning material
- Designed to demonstrate Kalman filter concepts in financial applications
- Suitable for academic study and algorithm research

### Important Disclaimers
- **Not for Production Trading**: This code is not intended for live trading or investment decisions
- **No Financial Advice**: This project does not constitute financial or investment advice
- **Research Only**: Use this code to understand algorithms, not to make trading decisions
- **No Liability**: Authors assume no responsibility for any losses from code usage

### Academic Citation
Please cite Benhamou (2016) when using Model 3 concepts in academic work:
- Benhamou, E. (2016). "Trend Without Hiccups – A Kalman Filter Approach"
- Paper URL: https://arxiv.org/pdf/1808.03297

#### Research Background

This implementation builds upon Eric Benhamou's comprehensive research comparing different Kalman filter models for financial time series. The original paper evaluates four different models:

- **Model 0**: Basic random walk (implemented as `kalman_ma`)
- **Model 1**: Random walk with drift
- **Model 2**: Mean reversion model
- **Model 3**: Two-factor model (implemented as `kalman_ma_model3`)
- **Model 4**: Model 3 + oscillator component

Benhamou's research demonstrated that **Model 3 significantly outperformed** simpler approaches across multiple asset classes, providing:
- Superior trend-following capabilities
- Reduced lag compared to traditional moving averages
- Better handling of market regime changes
- Improved signal quality for trading applications

Our implementation extends Model 3 with additional enhancements for cryptocurrency markets, including ATR-based dynamic adaptation and optimized parameters for high-frequency, volatile trading environments.

### Open Source
This code is shared freely for the educational benefit of the community. Feel free to study, modify, and learn from it, but please use responsibly and ethically.

## Contributing

Contributions welcome! Areas for improvement:
- Additional Kalman filter variants (Model 4 with oscillator component)
- Parameter auto-tuning algorithms
- Multi-timeframe implementations
- Real-time streaming capabilities

## Contact

For questions or contributions, please open an issue in the repository.