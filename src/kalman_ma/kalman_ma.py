import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===== パス設定（変更不要） =====
current_dir     = os.path.dirname(os.path.abspath(__file__))
workspace_root  = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(workspace_root)

# ---------------------------------------------------------------------
# 1️⃣ カルマンフィルター Moving Average
# ---------------------------------------------------------------------
def kalman_ma_atr(atr: np.ndarray, close: np.ndarray,
                  beta: float = 0.01, R: float = 1e-2) -> np.ndarray:
    """ATRベースの動的Qパラメータを使用したカルマンフィルター"""
    n = len(close)
    xhat = np.zeros(n)   # 状態推定値
    P    = np.zeros(n)   # 誤差共分散
    K    = np.zeros(n)   # カルマンゲイン

    # 初期化
    xhat[0] = close[0]
    P[0]    = 1.0        # 不確実性大きめで開始

    for t in range(1, n):
        # ATRベースの動的Q
        if not np.isnan(atr[t]):
            Q_t = beta * (atr[t] / close[t])**2  # 価格比率でスケール統一
        else:
            Q_t = 1e-5  # ATRが計算できない場合はデフォルト値

        # --- 予測ステップ ---
        x_pred = xhat[t-1]        # 状態方程式: x_t = x_{t-1}
        P_pred = P[t-1] + Q_t     # 動的Qを使用

        # --- 更新ステップ ---
        K[t]     = P_pred / (P_pred + R)           # カルマンゲイン
        xhat[t]  = x_pred + K[t] * (close[t] - x_pred)
        P[t]     = (1 - K[t]) * P_pred

    return xhat

def kalman_ma(series: np.ndarray, Q: float = 1e-5, R: float = 1e-2) -> np.ndarray:
    """1次状態空間モデルで平滑化した可変係数 EMA（カルマンMA）"""
    n = len(series)
    xhat = np.zeros(n)   # 状態推定値
    P    = np.zeros(n)   # 誤差共分散
    K    = np.zeros(n)   # カルマンゲイン

    # 初期化
    xhat[0] = series[0]
    P[0]    = 1.0        # 不確実性大きめで開始

    for t in range(1, n):
        # --- 予測ステップ ---
        x_pred = xhat[t-1]        # 状態方程式: x_t = x_{t-1}
        P_pred = P[t-1] + Q       # 分散の前進

        # --- 更新ステップ ---
        K[t]     = P_pred / (P_pred + R)           # カルマンゲイン
        xhat[t]  = x_pred + K[t] * (series[t] - x_pred)
        P[t]     = (1 - K[t]) * P_pred

    return xhat



# ---------------------------------------------------------------------
def main():

    calc_df = pd.read_csv('btc_kalman_ma_calc.csv')
        # CSVから読み込んだ場合、日付カラムを datetime 型に変換
    calc_df['start_at'] = pd.to_datetime(calc_df['start_at'])

    # 必要列のみ（EMA、SMA、ATRも含める）
    calc_df = calc_df[['start_at', 'close', 'ema', 'sma', 'atr']].copy()
    calc_df.sort_values('start_at', inplace=True)       # 念のため時系列順

    # 2️⃣ ATRベースの動的カルマン MA を計算（全期間のデータで計算）
    calc_df['kf_ma_atr'] = kalman_ma_atr(
        calc_df['atr'].values,
        calc_df['close'].values,
        beta=0.01,    # ATRスケーリング係数
        R=1e-2        # 観測ノイズ
    )

    # 従来の固定Qカルマン MA も計算（比較用）
    calc_df['kf_ma'] = kalman_ma(
        calc_df['close'].values,
        Q=1e-5,    # ← 追従性を左右する。ボラティリティに応じて動的更新も可
        R=1e-2     # ← 観測ノイズ。高頻度データなら大きめが滑らか
    )


    # グラフ表示用データ（直近期間のみ）
    display_start = datetime(2025, 3, 1)  # グラフ表示開始日
    display_end = datetime(2025, 3,10)    # グラフ表示終了日

    graph_df = calc_df[
        (calc_df['start_at'] >= display_start) &
        (calc_df['start_at'] <= display_end)
    ].copy()

    # 3️⃣ グラフ描画
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(graph_df['start_at'], graph_df['close'],
            label='BTC Close', linewidth=1.0)
    ax.plot(graph_df['start_at'], graph_df['kf_ma'],
            label='Kalman MA (Fixed Q)', linewidth=1.4, alpha=0.7)
    ax.plot(graph_df['start_at'], graph_df['sma'],
            label='SMA', linewidth=1.2, alpha=0.8)

    # 軸フォーマット
    ax.set_title('BTC/USDT Close vs. Kalman Moving Average', fontsize=14)
    ax.set_xlabel('Date & Time')
    ax.set_ylabel('Price (USDT)')
    ax.legend()

    # 日付時間軸を見やすく（短期間なので時間も表示）
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # 12時間間隔
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))  # 月/日 時:分
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))   # 6時間間隔でマイナーティック

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
