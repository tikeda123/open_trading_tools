# Kalman Filter Moving Average Implementation

## Project Description

This project is **open-source educational code** that was originally shared on Twitter for learning purposes. The implementation demonstrates advanced Kalman filter techniques applied to financial time series, specifically focusing on cryptocurrency price data analysis.

**⚠️ Important Notice:**
- **Educational Purpose Only**: This code is intended for learning and research purposes
- **Not Investment Advice**: This is not financial advice and should not be used for actual trading decisions
- **No Warranty**: Use at your own risk - no guarantees on performance or accuracy
- **Learning Resource**: Designed to help understand Kalman filter applications in finance

This module implements advanced Kalman filter-based moving averages for financial time series analysis, with particular focus on cryptocurrency trading applications for educational exploration.

## Overview

The implementation provides three distinct Kalman filter approaches for price smoothing and trend following:

1. **ATR-Based Dynamic Kalman MA** - Volatility-adaptive single-component filter
2. **Fixed-Q Kalman MA** - Traditional single-component filter with constant parameters
3. **Enhanced Model 3** - Two-component system based on Benhamou (2016) research

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

## Implementation Details

### 1. ATR-Based Dynamic Kalman MA

```python
def kalman_ma_atr(atr, close, beta=0.01, R=1e-2):
```

**Key Features:**
- Dynamic process noise adjustment based on Average True Range (ATR)
- Higher volatility periods → Larger Q → Faster adaptation
- Lower volatility periods → Smaller Q → More smoothing

**Parameters:**
- `atr`: Average True Range values for volatility measurement
- `close`: Price series to filter
- `beta`: ATR scaling coefficient (0.005-0.05 recommended)
- `R`: Fixed observation noise variance

**Dynamic Q Calculation:**
```python
Q_t = beta * (atr[t] / close[t])**2
```

### 2. Fixed-Q Kalman MA

```python
def kalman_ma(series, Q=1e-5, R=1e-2):
```

**Key Features:**
- Constant process and observation noise parameters
- Provides baseline comparison for dynamic methods
- Optimal for stable market conditions

**Parameters:**
- `series`: Price series to filter
- `Q`: Process noise variance (controls tracking vs smoothing)
- `R`: Observation noise variance (measurement uncertainty)

### 3. Enhanced Model 3 Kalman MA

```python
def kalman_ma_model3(close, atr=None, phi_params=(1.0, 0.10, 1.0), 
                     h_params=(1.5, 1.0), q_params=(0.20, 0.10),
                     r_param=None, p0_params=(5.0, 5.0), beta=1.0):
```

**Key Features:**
- Two-component state system (short-term + long-term)
- Price normalization for numerical stability
- Dynamic ATR-based noise adaptation
- Optimized parameters for tracking vs stability balance

**Advanced Parameters:**
- `phi_params`: State transition parameters (a11, a12, a22)
- `h_params`: Observation weights for components (normalized to sum=1)
- `q_params`: Process noise parameters for covariance matrix
- `r_param`: Fixed observation noise (None for ATR-based dynamic)
- `p0_params`: Initial covariance diagonal values
- `beta`: ATR scaling factor (0.5-2.0 recommended)

## Price Normalization Strategy

The Enhanced Model 3 includes price normalization to improve numerical stability:

```python
scale = close[0]           # Initial price as scaling factor
close_ = close / scale     # Normalize to initial value = 1
atr_ = atr / scale         # Scale ATR accordingly
# ... filtering on normalized data ...
return yhat * scale        # Denormalize results
```

**Benefits:**
- Consistent parameter behavior across different price levels
- Improved numerical stability for matrix operations
- Easier parameter optimization and tuning

## Dynamic Observation Noise

Model 3 implements ATR-dependent observation noise:

```python
def _obs_noise(idx):
    if atr_ is not None and not np.isnan(atr_[idx]):
        return (atr_[idx] / (close_[idx] + 1e-12))**2
    return 1e-4 if r_param is None else r_param
```

**Adaptive Behavior:**
- High volatility → Higher observation noise → Less trust in individual observations
- Low volatility → Lower observation noise → More trust in observations

## Usage Examples

### Basic Usage

```python
import numpy as np
import pandas as pd
from kalman_ma import kalman_ma, kalman_ma_atr, kalman_ma_model3

# Load your price data
df = pd.read_csv('price_data.csv')
close_prices = df['close'].values
atr_values = df['atr'].values

# Method 1: Fixed-parameter Kalman MA
kf_basic = kalman_ma(close_prices, Q=1e-5, R=1e-2)

# Method 2: ATR-adaptive Kalman MA
kf_atr = kalman_ma_atr(atr_values, close_prices, beta=0.01)

# Method 3: Enhanced two-component Model 3
kf_model3 = kalman_ma_model3(close_prices, atr=atr_values)
```

### Parameter Tuning Guidelines

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

## Performance Characteristics

### Tracking vs Smoothness Trade-off

| Method | Tracking Speed | Smoothness | Stability | Best Use Case |
|--------|---------------|------------|-----------|---------------|
| Fixed Q | Medium | High | High | Stable markets |
| ATR-based | High | Medium | Medium | Volatile markets |
| Model 3 | High | High | High | All conditions |

### Computational Complexity

- **Fixed Q**: O(n) - Linear time complexity
- **ATR-based**: O(n) - Linear with volatility calculation overhead
- **Model 3**: O(n) - Linear with 2×2 matrix operations per step

## Dependencies

```python
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.5.0  # For visualization only
```

## Installation

```bash
pip install numpy pandas matplotlib
```

## Research Background

This implementation is based on:

1. **Benhamou, E. (2016)**: "Trend Without Hiccups – A Kalman Filter Approach"
   - Model 3 two-factor approach
   - Parameter optimization strategies
   - Performance comparison across asset classes

2. **Classical Kalman Filtering Theory**:
   - Optimal state estimation under Gaussian assumptions
   - Recursive Bayesian filtering
   - Minimum mean square error estimation

## File Structure

```
kalman_ma/
├── kalman_ma.py              # Main implementation
├── kalman_ma_english.py      # Fully documented English version
├── README.md                 # This documentation
└── btc_kalman_ma_calc.csv   # Sample Bitcoin data
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

## Visualization

The module includes a demonstration function that creates comparative plots:

```python
if __name__ == "__main__":
    main()  # Runs demonstration with Bitcoin data
```

The demo generates plots comparing:
- Original BTC close prices
- Fixed-Q Kalman MA
- Simple Moving Average (SMA)
- Enhanced Model 3 Kalman MA

## Parameter Optimization

### Recommended Starting Values

**Conservative (Smooth) Settings:**
```python
# Fixed Q
kalman_ma(prices, Q=1e-6, R=1e-2)

# ATR-based
kalman_ma_atr(atr, prices, beta=0.005)

# Model 3
kalman_ma_model3(prices, q_params=(0.05, 0.02), beta=0.5)
```

**Aggressive (Responsive) Settings:**
```python
# Fixed Q
kalman_ma(prices, Q=1e-4, R=1e-3)

# ATR-based  
kalman_ma_atr(atr, prices, beta=0.05)

# Model 3
kalman_ma_model3(prices, q_params=(0.30, 0.15), beta=2.0)
```

### Optimization Strategies

1. **Grid Search**: Systematic parameter space exploration
2. **Bayesian Optimization**: Efficient parameter tuning with Optuna
3. **Walk-Forward Analysis**: Time-series specific validation
4. **Cross-Validation**: Multiple fold validation for robustness

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

## Testing and Validation

### Unit Tests
```python
# Test basic functionality
def test_kalman_ma_basic():
    prices = np.array([100, 101, 102, 101, 100])
    result = kalman_ma(prices)
    assert len(result) == len(prices)
    assert not np.any(np.isnan(result))

# Test ATR version
def test_kalman_ma_atr():
    prices = np.array([100, 101, 102, 101, 100])
    atr = np.array([1.0, 1.1, 1.2, 1.1, 1.0])
    result = kalman_ma_atr(atr, prices)
    assert len(result) == len(prices)
```

### Performance Metrics
- **Lag Analysis**: Compare phase delay vs SMA/EMA
- **Noise Reduction**: Signal-to-noise ratio improvement
- **Trend Following**: Directional accuracy during trend periods
- **Whipsaw Reduction**: False signal frequency analysis

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