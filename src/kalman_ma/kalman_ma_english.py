import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===== Path Configuration (Do not modify) =====
current_dir     = os.path.dirname(os.path.abspath(__file__))
workspace_root  = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(workspace_root)

# ---------------------------------------------------------------------
# 1️⃣ Kalman Filter Moving Average Implementation
# ---------------------------------------------------------------------
def kalman_ma_atr(atr: np.ndarray, close: np.ndarray,
                  beta: float = 0.01, R: float = 1e-2) -> np.ndarray:
    """
    Kalman Filter with dynamic Q parameter based on Average True Range (ATR).
    
    This implementation uses ATR to dynamically adjust the process noise (Q) based on market volatility.
    Higher volatility (larger ATR) results in higher Q values, allowing the filter to adapt more quickly
    to price changes during volatile periods.
    
    Args:
        atr: Average True Range values
        close: Closing prices
        beta: ATR scaling coefficient for dynamic Q calculation
        R: Observation noise variance (fixed)
    
    Returns:
        Filtered price series with reduced lag and adaptive behavior
    """
    n = len(close)
    xhat = np.zeros(n)   # State estimates (filtered values)
    P    = np.zeros(n)   # Error covariance matrix
    K    = np.zeros(n)   # Kalman gain

    # Initialization
    xhat[0] = close[0]   # Initialize first state with first price
    P[0]    = 1.0        # Start with high uncertainty to allow rapid adaptation

    for t in range(1, n):
        # Dynamic Q based on ATR (process noise adaptation)
        if not np.isnan(atr[t]):
            # Scale Q by ATR relative to price to normalize across different price levels
            # Higher ATR = more volatility = higher process noise = faster adaptation
            Q_t = beta * (atr[t] / close[t])**2  
        else:
            Q_t = 1e-5  # Default small Q when ATR is unavailable

        # --- Prediction Step ---
        # State equation: x_t = x_{t-1} + w_t (random walk model)
        x_pred = xhat[t-1]        
        P_pred = P[t-1] + Q_t     # Covariance prediction with dynamic Q

        # --- Update Step ---
        K[t]     = P_pred / (P_pred + R)           # Kalman gain calculation
        # State update: blend prediction with observation based on Kalman gain
        xhat[t]  = x_pred + K[t] * (close[t] - x_pred)
        # Covariance update
        P[t]     = (1 - K[t]) * P_pred

    return xhat

def kalman_ma(series: np.ndarray, Q: float = 1e-5, R: float = 1e-2) -> np.ndarray:
    """
    First-order state space model smoothed variable coefficient EMA (Kalman MA).
    
    This is a basic Kalman filter implementation using a random walk model with fixed
    process and observation noise parameters. It provides smoothing while maintaining
    responsiveness to price changes.
    
    Args:
        series: Price series to filter
        Q: Process noise variance (controls responsiveness vs smoothness trade-off)
        R: Observation noise variance (measurement uncertainty)
    
    Returns:
        Filtered series with optimal balance between lag and noise reduction
    """
    n = len(series)
    xhat = np.zeros(n)   # State estimates (filtered values)
    P    = np.zeros(n)   # Error covariance matrix
    K    = np.zeros(n)   # Kalman gain

    # Initialization
    xhat[0] = series[0]  # Initialize with first observation
    P[0]    = 1.0        # Start with high uncertainty for rapid convergence

    for t in range(1, n):
        # --- Prediction Step ---
        # Random walk state transition: x_t = x_{t-1} + w_t
        x_pred = xhat[t-1]        
        P_pred = P[t-1] + Q       # Forward covariance with constant process noise

        # --- Update Step ---
        # Optimal Kalman gain balances prediction and observation uncertainties
        K[t]     = P_pred / (P_pred + R)           
        xhat[t]  = x_pred + K[t] * (series[t] - x_pred)
        P[t]     = (1 - K[t]) * P_pred

    return xhat

# ---- Enhanced Version: Price Normalization with Stable Parameters ----
def kalman_ma_model3(
        close: np.ndarray,
        atr:   np.ndarray | None = None,
        # ---- ① Tracking-Priority Stable Parameters ----
        phi_params=(1.0, 0.10, 1.0),   # a11=a22=1 → Pure random walk for both components
        h_params=(1.5, 1.0),           # 60% : 40% weighting (normalized to 0.6/0.4)
        # Process noise 10x stronger, observation noise switched to dynamic
        q_params=(0.20, 0.10),         # >> 0.05, enhanced tracking capability
        r_param = None,                # Abandon fixed values, use ATR-based dynamic noise
        p0_params=(5.0, 5.0),          # Larger initial uncertainty for faster adaptation
        beta: float = 1.0              # ATR→Q scaling factor, range 0.5~2.0 recommended
    ) -> np.ndarray:
    """
    Two-dimensional Kalman MA based on Benhamou (2016) Model 3 (Enhanced Version).
    
    This implementation follows Eric Benhamou's research on Kalman filters for financial time series,
    specifically Model 3 which uses a two-factor approach with short-term and long-term components.
    The enhanced version includes price normalization for numerical stability and dynamic parameter
    adjustment based on market volatility (ATR).
    
    Model Structure:
    - State vector: X_t = [x_t^(1), x_t^(2)]' (short-term, long-term components)
    - Transition matrix: Φ = [[a11, a12], [0, a22]] 
    - Observation equation: y_t = h1*x_t^(1) + h2*x_t^(2) + v_t
    
    Key Features:
    - Price normalization to initial value = 1 for numerical stability
    - ATR-based dynamic process noise for volatility adaptation  
    - Dynamic observation noise based on relative ATR
    - Optimized parameters for tracking vs stability balance
    
    Args:
        close: Price series to filter
        atr: Average True Range for dynamic noise adjustment (optional)
        phi_params: State transition parameters (a11, a12, a22)
        h_params: Observation weights for short/long components (h1, h2)
        q_params: Process noise parameters (q1, q2) 
        r_param: Fixed observation noise (None for ATR-based dynamic)
        p0_params: Initial covariance diagonal values
        beta: ATR scaling factor for dynamic Q computation
    
    Returns:
        Filtered moving average with enhanced tracking and stability
    """
    # ---- ② Price Normalization: Scale down to initial value = 1 ----
    # This normalization improves numerical stability and makes parameter tuning
    # more consistent across different price levels and asset classes
    scale  = close[0]
    close_ = close / scale
    atr_   = None if atr is None else atr / scale

    n = len(close_)
    xhat = np.zeros((n, 2))           # State estimates [short-term, long-term]
    yhat = np.zeros(n)                # Moving average output values
    P    = np.zeros((n, 2, 2))        # Error covariance matrices (2x2 for each time step)

    # --- Matrix Definitions ---
    # State transition matrix Φ for two-component system
    a11, a12, a22 = phi_params
    Phi = np.array([[a11, a12],
                    [0.0, a22]])

    h1, h2 = h_params
    # Normalize observation weights so h1+h2=1 for easier interpretation
    # This ensures the output is a proper weighted average of the two components
    H = np.array([h1, h2], dtype=float)
    H /= H.sum()

    q1,  q2  = q_params
    Q_base   = np.array([[q1**2, q1*q2],
                         [q1*q2, q2**2]])
    # Observation noise: ATR-dependent or fixed
    # Dynamic observation noise adapts to market conditions:
    # - High volatility periods: higher observation noise (less trust in individual observations)
    # - Low volatility periods: lower observation noise (more trust in observations)
    def _obs_noise(idx:int):
        if atr_ is not None and not np.isnan(atr_[idx]):
            return (atr_[idx] / (close_[idx] + 1e-12))**2
        return 1e-4 if r_param is None else r_param

    P[0] = np.diag(p0_params)

    # Initialization: Set both components to the same normalized price
    # This provides a neutral starting point for the short/long-term decomposition
    xhat[0] = close_[0]
    yhat[0] = close_[0]

    # === Kalman Filtering Loop ===
    # Sequential processing of each time step with prediction and update phases
    for t in range(1, n):
        # --- Prediction Phase ---
        # Project current state estimate forward using transition model
        x_pred = Phi @ xhat[t-1]
        
        # Dynamic process noise Q based on ATR
        # Higher volatility (ATR) → larger Q → faster adaptation to new trends
        # Lower volatility (ATR) → smaller Q → more smoothing, less noise
        if atr_ is not None and not np.isnan(atr_[t]):
            # Scale process noise by relative volatility (ATR/price ratio squared)
            dyn_scale = beta * (atr_[t] / close_[t])**2
            Q_t = Q_base * dyn_scale
        else:
            # Use base process noise when ATR is unavailable
            Q_t = Q_base
            
        # Covariance prediction: propagate uncertainty and add process noise
        P_pred = Phi @ P[t-1] @ Phi.T + Q_t   # Q_t added directly to predicted covariance

        # --- Update Phase ---
        # Innovation covariance: uncertainty in the predicted observation
        S = H @ P_pred @ H.T + _obs_noise(t)        # Scalar value
        # Kalman gain: optimal weighting between prediction and observation
        K = (P_pred @ H) / S            # 2x1 vector
        y_t  = close_[t]
        # State update: blend prediction with observation using Kalman gain
        # Innovation (y_t - H @ x_pred) is the prediction error
        xhat[t] = x_pred + K * (y_t - H @ x_pred)
        # Covariance update: reduce uncertainty after incorporating observation
        P[t]   = (np.eye(2) - np.outer(K, H)) @ P_pred

        # Moving average output: weighted combination of short and long-term components
        yhat[t] = H @ xhat[t]           

    # ---- Return to Original Scale ----
    # Denormalize the filtered results back to the original price scale
    return yhat * scale



# ---------------------------------------------------------------------
def main():
    """
    Main function demonstrating Kalman filter moving averages on Bitcoin price data.
    
    This function:
    1. Loads BTC price data from CSV with OHLC, EMA, SMA, and ATR columns
    2. Calculates various Kalman filter moving averages:
       - ATR-based dynamic Kalman MA
       - Fixed-parameter Kalman MA  
       - Enhanced Model 3 two-component Kalman MA
    3. Displays comparative plots for visual analysis
    
    The demo focuses on a specific date range to show detailed behavior
    during different market conditions.
    """

    calc_df = pd.read_csv('btc_kalman_ma_calc.csv')
        # Convert date column to datetime type when loaded from CSV
    calc_df['start_at'] = pd.to_datetime(calc_df['start_at'])

    # Select only required columns (including EMA, SMA, ATR for comparison)
    calc_df = calc_df[['start_at', 'close', 'ema', 'sma', 'atr']].copy()
    calc_df.sort_values('start_at', inplace=True)       # Ensure chronological order

    # 2️⃣ Calculate ATR-based dynamic Kalman MA (computed over full dataset)
    # This version adapts its responsiveness based on market volatility
    calc_df['kf_ma_atr'] = kalman_ma_atr(
        calc_df['atr'].values,
        calc_df['close'].values,
        beta=0.01,    # ATR scaling coefficient for dynamic Q
        R=1e-2        # Fixed observation noise variance
    )
    
    # Calculate traditional fixed-Q Kalman MA for comparison
    # This version uses constant process and observation noise
    calc_df['kf_ma'] = kalman_ma(
        calc_df['close'].values,
        Q=1e-5,    # Process noise: controls tracking vs smoothing trade-off
        R=1e-2     # Observation noise: larger values = more smoothing for high-frequency data
    )
    
    # Calculate enhanced Model 3 Kalman MA with two-component structure
    # This version provides the best balance of tracking and stability
    calc_df['kf_ma_mod3'] = kalman_ma_model3(
        close=calc_df['close'].values,
        atr=calc_df['atr'].values       # Pass None if not using ATR-based dynamics
    )


    # Filter data for graph display (recent period only)
    # Focus on specific date range for detailed visual analysis
    display_start = datetime(2025, 3, 1)  # Graph display start date
    display_end = datetime(2025, 3,10)    # Graph display end date

    graph_df = calc_df[
        (calc_df['start_at'] >= display_start) &
        (calc_df['start_at'] <= display_end)
    ].copy()

    # 3️⃣ Graph Plotting
    # Create comparative visualization of different moving average methods
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(graph_df['start_at'], graph_df['close'],
            label='BTC Close', linewidth=1.0)
    ax.plot(graph_df['start_at'], graph_df['kf_ma'],
            label='Kalman MA (Fixed Q)', linewidth=1.4, alpha=0.7)
    ax.plot(graph_df['start_at'], graph_df['sma'],
            label='SMA', linewidth=1.2, alpha=0.8)
    ax.plot(graph_df['start_at'], graph_df['kf_ma_mod3'],
            label='Kalman MA Model 3', linewidth=1.8, color='green')

    # Axis formatting and labels
    ax.set_title('BTC/USDT Close vs. Kalman Moving Average', fontsize=14)
    ax.set_xlabel('Date & Time')
    ax.set_ylabel('Price (USDT)')
    ax.legend()

    # Format date/time axis for readability (showing hours for short timeframe)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # Major ticks every 12 hours
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))  # Format: MM/DD HH:MM
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))   # Minor ticks every 6 hours

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()