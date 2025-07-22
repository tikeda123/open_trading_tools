import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import the Kalman MA functions from the main module
from kalman_ma import kalman_ma_atr, kalman_ma, kalman_ma_model3

def generate_comparison_graph():
    """Generate and save a comparison graph of different Kalman MA methods"""
    
    # Load data
    calc_df = pd.read_csv('btc_kalman_ma_calc.csv')
    calc_df['start_at'] = pd.to_datetime(calc_df['start_at'])

    # Select required columns
    calc_df = calc_df[['start_at', 'close', 'ema', 'sma', 'atr']].copy()
    calc_df.sort_values('start_at', inplace=True)

    # Calculate different Kalman MAs
    calc_df['kf_ma_atr'] = kalman_ma_atr(
        calc_df['atr'].values,
        calc_df['close'].values,
        beta=0.01,
        R=1e-2
    )
    
    calc_df['kf_ma'] = kalman_ma(
        calc_df['close'].values,
        Q=1e-5,
        R=1e-2
    )
    
    calc_df['kf_ma_mod3'] = kalman_ma_model3(
        close=calc_df['close'].values,
        atr=calc_df['atr'].values
    )

    # Filter data for graph display
    display_start = datetime(2025, 3, 1)
    display_end = datetime(2025, 3, 10)

    graph_df = calc_df[
        (calc_df['start_at'] >= display_start) &
        (calc_df['start_at'] <= display_end)
    ].copy()

    # Create the graph
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the lines
    ax.plot(graph_df['start_at'], graph_df['close'],
            label='BTC Close', linewidth=2.0, color='blue', alpha=0.8)
    ax.plot(graph_df['start_at'], graph_df['kf_ma'],
            label='Kalman MA (固定Q)', linewidth=2.0, color='orange', alpha=0.8)
    ax.plot(graph_df['start_at'], graph_df['sma'],
            label='SMA', linewidth=1.8, color='lightgreen', alpha=0.7)
    ax.plot(graph_df['start_at'], graph_df['kf_ma_mod3'],
            label='Kalman MA Model 3', linewidth=2.5, color='green', alpha=0.9)

    # Formatting
    ax.set_title('BTC/USDT価格 vs Kalman Moving Average 比較\n(2025年3月1日-10日)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('日時', fontsize=12)
    ax.set_ylabel('価格 (USDT)', fontsize=12)
    
    # Legend with better positioning
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Format date/time axis
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Tight layout
    plt.tight_layout()

    # Save the graph
    plt.savefig('kalman_ma_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Graph saved as 'kalman_ma_comparison.png'")
    
    # Also show the graph
    plt.show()

if __name__ == "__main__":
    generate_comparison_graph()