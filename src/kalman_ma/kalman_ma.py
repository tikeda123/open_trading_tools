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

# ---- 改良版：価格を正規化し、安定パラメータを採用 ----
def kalman_ma_model3(
        close: np.ndarray,
        atr:   np.ndarray | None = None,
        # ---- ① 追従性を最優先する安定パラメータ ----
        phi_params=(1.0, 0.10, 1.0),   # a11=a22=1 → ランダムウォーク
        h_params=(1.5, 1.0),           # 60% : 40%（自然対数で 0.6/0.4）
        # プロセスノイズを 10 倍強め，観測ノイズは動的に切替
        q_params=(0.20, 0.10),         # >> 0.05, 追従性↑
        r_param = None,                # 固定値をやめ ATR ベースに
        p0_params=(5.0, 5.0),          # 初期不確実性を大きめ
        beta: float = 1.0              # ATR→Q スケール 0.5〜2.0 が目安
    ) -> np.ndarray:
    """
    Benhamou (2016) モデル3 に基づく 2 次元カルマン MA (改良版)
    close : 価格系列
    atr   : ATR (動的 Q 用)。無ければ固定 Q
    """
    # ---- ② 価格を初期値=1 にスケールダウン ----
    scale  = close[0]
    close_ = close / scale
    atr_   = None if atr is None else atr / scale

    n = len(close_)
    xhat = np.zeros((n, 2))           # 状態推定値
    yhat = np.zeros(n)                # MA として返す値
    P    = np.zeros((n, 2, 2))        # 誤差共分散

    # --- 行列定義 ---
    a11, a12, a22 = phi_params
    Phi = np.array([[a11, a12],
                    [0.0, a22]])

    h1, h2 = h_params
    # 求めやすいよう h1+h2=1 に正規化
    H = np.array([h1, h2], dtype=float)
    H /= H.sum()

    q1,  q2  = q_params
    Q_base   = np.array([[q1**2, q1*q2],
                         [q1*q2, q2**2]])
    # 観測ノイズ：ATR 依存 or 固定
    def _obs_noise(idx:int):
        if atr_ is not None and not np.isnan(atr_[idx]):
            return (atr_[idx] / (close_[idx] + 1e-12))**2
        return 1e-4 if r_param is None else r_param

    P[0] = np.diag(p0_params)

    # 初期化：両成分に同じ価格を入れておく
    xhat[0] = close_[0]
    yhat[0] = close_[0]

    # === フィルタリング ===
    for t in range(1, n):
        # --- 予測 ---
        x_pred = Phi @ xhat[t-1]
        
        # ATR による動的 Q
        if atr_ is not None and not np.isnan(atr_[t]):
            dyn_scale = beta * (atr_[t] / close_[t])**2
            Q_t = Q_base * dyn_scale
        else:
            Q_t = Q_base
            
        P_pred = Phi @ P[t-1] @ Phi.T + Q_t   # ← 先に Q_t を加算

        # --- 更新 ---
        S = H @ P_pred @ H.T + _obs_noise(t)        # スカラー
        K = (P_pred @ H) / S            # 2x1 ベクトル
        y_t  = close_[t]
        xhat[t] = x_pred + K * (y_t - H @ x_pred)
        P[t]   = (np.eye(2) - np.outer(K, H)) @ P_pred

        yhat[t] = H @ xhat[t]           # MA 出力

    # ---- 元スケールに戻して返す ----
    return yhat * scale



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
    
    # kalman_ma_model3 を計算
    calc_df['kf_ma_mod3'] = kalman_ma_model3(
        close=calc_df['close'].values,
        atr=calc_df['atr'].values       # ATR を使わない場合は None
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
    ax.plot(graph_df['start_at'], graph_df['kf_ma_mod3'],
            label='Kalman MA Model 3', linewidth=1.8, color='green')

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
