import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
# Use a non-interactive backend to avoid Tkinter issues when running headless
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import argparse

# Detection tuning constants (tweakable)
SHOULDER_TOL = 0.15                 # max relative difference between shoulders (10%)
HEAD_OVER_SHOULDER_PCT = 0.05      # head must be at least 7% above the higher shoulder
MAX_NECKLINE_ANGLE_DEG = 35        # maximum allowed angle for the neckline (degrees)
TIME_RATIO_MIN = 0.5                # min ratio of left-interval/right-interval
TIME_RATIO_MAX = 2.0                # max ratio of left-interval/right-interval
VOLUME_TREND_TOL = 0.95             # require vol_ls > vol_h*VOLUME_TREND_TOL and vol_h > vol_rs*VOLUME_TREND_TOL
VOLUME_SPIKE_MULT = 1.50           # breakout volume must exceed avg pattern volume * this multiplier
BASE_NBARS = 20                     # base lookback for swing detection
MIN_NBARS = 8
MAX_NBARS = 60
# Feature flags (enable/disable expensive checks)
USE_ZIGZAG = True                   # use ZigZag swing detection instead of rolling highs/lows
ZIGZAG_PCT = 0.03                   # 3% reversal threshold for ZigZag
USE_FRACTAL = False                 # use Williams Fractals for swing detection
REQUIRE_VOLUME_DIVERGENCE = False   # require bearish volume divergence during head formation
REQUIRE_ADX = False                 # require ADX threshold for preceding uptrend
MIN_ADX = 20
REQUIRE_RSI_OR_MACD = False         # require RSI < 50 or MACD histogram < 0 at breakout
REQUIRE_MA_CONFIRM = False          # require price below moving average on breakout
MA_CONFIRM_PERIOD = 50

def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data.reset_index()
    # Flatten MultiIndex columns (yfinance may return MultiIndex with ticker level)
    if isinstance(data.columns, pd.MultiIndex):
        new_cols = []
        for c in data.columns:
            # prefer first level if present, else second
            if c[0] and c[0] != '':
                new_cols.append(c[0])
            else:
                new_cols.append(c[1])
        data.columns = new_cols

    # Ensure 'Date' is a datetime dtype
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

    return data

def find_swing_points(data, N_bars=20):
    data['is_swing_high'] = (data['High'] == data['High'].rolling(window=N_bars*2+1, center=True).max())
    data['is_swing_low'] = (data['Low'] == data['Low'].rolling(window=N_bars*2+1, center=True).min())
    return data

def compute_atr(data, period=14):
    # True Range
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def compute_dynamic_nbars(data, base=BASE_NBARS, min_bars=MIN_NBARS, max_bars=MAX_NBARS):
    # Use ATR relative to median ATR to scale N_bars: higher volatility -> larger lookback
    try:
        atr = compute_atr(data, period=14)
        median_atr = np.nanmedian(atr)
        recent_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else median_atr
        if median_atr == 0 or np.isnan(median_atr):
            scale = 1.0
        else:
            scale = recent_atr / median_atr
        nbars = int(max(min_bars, min(max_bars, round(base * scale))))
        return nbars
    except Exception:
        return base

def compute_adx(data, period=14):
    # Implementation of ADX (simplified)
    high = data['High']
    low = data['Low']
    close = data['Close']

    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.rolling(window=period, min_periods=1).mean()
    return adx

def compute_rsi(data, period=14):
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - macd_signal
    return macd_line, macd_signal, hist

def compute_fractals(data, left=2, right=2):
    highs = data['High'].values
    lows = data['Low'].values
    n = len(data)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for i in range(left, n-right):
        if highs[i] == max(highs[i-left:i+right+1]):
            is_high[i] = True
        if lows[i] == min(lows[i-left:i+right+1]):
            is_low[i] = True
    data['is_swing_high'] = is_high
    data['is_swing_low'] = is_low
    return data

def compute_zigzag(data, pct=ZIGZAG_PCT):
    # Simple ZigZag implementation marking local extremes where price reverses by pct
    closes = data['Close'].values
    n = len(closes)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    if n == 0:
        data['is_swing_high'] = is_high
        data['is_swing_low'] = is_low
        return data

    last_extreme_idx = 0
    last_extreme_price = closes[0]
    last_type = 'unknown'  # 'high' or 'low'
    for i in range(1, n):
        change = (closes[i] - last_extreme_price) / last_extreme_price
        if last_type in ('unknown', 'low') and change >= pct:
            # new high
            is_high[i] = True
            last_extreme_idx = i
            last_extreme_price = closes[i]
            last_type = 'high'
        elif last_type in ('unknown', 'high') and change <= -pct:
            # new low
            is_low[i] = True
            last_extreme_idx = i
            last_extreme_price = closes[i]
            last_type = 'low'

    data['is_swing_high'] = is_high
    data['is_swing_low'] = is_low
    return data

def generate_swing_flags(data, method='rolling', N_bars=BASE_NBARS):
    if method == 'fractal':
        return compute_fractals(data)
    if method == 'zigzag':
        return compute_zigzag(data, pct=ZIGZAG_PCT)
    # default rolling
    return find_swing_points(data, N_bars=N_bars)

def check_preceding_uptrend(data, pattern_start_index, lookback_period=90, min_rise_percent=0.15):
    """Check if there was a significant uptrend before the pattern started."""
    # Coerce incoming index/label to a plain Python int (handles numpy scalars)
    try:
        p_idx = int(pattern_start_index)
    except Exception:
        # Fallback: try to call .item() if it's a numpy/pandas scalar-like
        try:
            p_idx = int(getattr(pattern_start_index, 'item', lambda: pattern_start_index)())
        except Exception:
            return False

    if p_idx < lookback_period:
        return False

    # Safe bounds for lookback slice
    start_pos = max(0, p_idx - lookback_period)
    lookback_data = data.iloc[start_pos: p_idx]
    if lookback_data.empty:
        return False

    # Extract scalar floats reliably (handle pandas/numpy scalars)
    start_price_raw = lookback_data['Close'].iloc[0]
    try:
        start_price = float(start_price_raw.item()) if hasattr(start_price_raw, 'item') else float(start_price_raw)
    except Exception:
        start_price = float(start_price_raw)

    # Use integer-location to get the row and coerce to float
    try:
        row = data.iloc[p_idx]
    except Exception:
        return False

    end_price_raw = row.get('High') if hasattr(row, 'get') else row['High']
    try:
        end_price = float(end_price_raw.item()) if hasattr(end_price_raw, 'item') else float(end_price_raw)
    except Exception:
        end_price = float(end_price_raw)

    return end_price > start_price * (1 + min_rise_percent)


def check_preceding_downtrend(data, pattern_start_index, lookback_period=90, min_fall_percent=0.15):
    """Check if there was a significant downtrend before the pattern started."""
    try:
        p_idx = int(pattern_start_index)
    except Exception:
        try:
            p_idx = int(getattr(pattern_start_index, 'item', lambda: pattern_start_index)())
        except Exception:
            return False

    if p_idx < lookback_period:
        return False

    start_pos = max(0, p_idx - lookback_period)
    lookback_data = data.iloc[start_pos: p_idx]
    if lookback_data.empty:
        return False

    try:
        start_price_raw = lookback_data['Close'].iloc[0]
        start_price = float(start_price_raw.item()) if hasattr(start_price_raw, 'item') else float(start_price_raw)
    except Exception:
        try:
            start_price = float(lookback_data['Close'].iloc[0])
        except Exception:
            return False

    try:
        row = data.iloc[p_idx]
    except Exception:
        return False

    end_price_raw = row.get('Low') if hasattr(row, 'get') else row['Low']
    try:
        end_price = float(end_price_raw.item()) if hasattr(end_price_raw, 'item') else float(end_price_raw)
    except Exception:
        try:
            end_price = float(end_price_raw)
        except Exception:
            return False

    return end_price < start_price * (1 - min_fall_percent)

def detect_head_and_shoulders(data, min_days=None, max_days=None):
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    patterns = []

    for i in range(len(swing_points_df) - 4):
        # Get indices of the 5 swing points
        p1_idx, t1_idx, p2_idx, t2_idx, p3_idx = swing_points_df['index'][i:i+5]

        # Ensure correct sequence of swings (High, Low, High, Low, High)
        if not (data['is_swing_high'][p1_idx] and data['is_swing_low'][t1_idx] and
                data['is_swing_high'][p2_idx] and data['is_swing_low'][t2_idx] and
                data['is_swing_high'][p3_idx]):
            continue

        # --- Get prices and dates for key points ---
        p1_high, p1_date = data.loc[p1_idx, ['High', 'Date']]
        t1_low, t1_date = data.loc[t1_idx, ['Low', 'Date']]
        p2_high, p2_date = data.loc[p2_idx, ['High', 'Date']]
        t2_low, t2_date = data.loc[t2_idx, ['Low', 'Date']]
        p3_high, p3_date = data.loc[p3_idx, ['High', 'Date']]

    # --- Rule 1: Preceding Uptrend ---
        if not check_preceding_uptrend(data, p1_idx):
            continue
        # optional ADX check
        if REQUIRE_ADX:
            try:
                adx = compute_adx(data)
                start_pos = max(0, p1_idx - 90)
                mean_adx = float(adx.iloc[start_pos:p1_idx].mean())
                if np.isnan(mean_adx) or mean_adx < MIN_ADX:
                    continue
            except Exception:
                continue

        # --- Rule 2: Head must be the highest peak and noticeably above shoulders ---
        higher_shoulder = max(p1_high, p3_high)
        if not (p2_high > p1_high and p2_high > p3_high):
            continue
        if not (p2_high > higher_shoulder * (1 + HEAD_OVER_SHOULDER_PCT)):
            # head not sufficiently above shoulders
            continue

        # --- Rule 3: Shoulders Proportionality ---
        if abs(p1_high - p3_high) / max(p1_high, p3_high) > SHOULDER_TOL:
            continue

        # --- Duration filter (optional) ---
        try:
            duration_days = (p3_date - p1_date).days
        except Exception:
            # If dates are malformed, skip
            continue
        if min_days is not None and duration_days < min_days:
            continue
        if max_days is not None and duration_days > max_days:
            continue

        # --- Rule 4: Neckline Calculation ---
        neckline_y = [t1_low, t2_low]
        neckline_x = [t1_date.toordinal(), t2_date.toordinal()]
        if neckline_x[1] - neckline_x[0] == 0: continue
        slope = (neckline_y[1] - neckline_y[0]) / (neckline_x[1] - neckline_x[0])
        intercept = neckline_y[0] - slope * neckline_x[0]
        # check neckline slope angle
        try:
            angle_deg = abs(np.degrees(np.arctan(slope)))
            if angle_deg > MAX_NECKLINE_ANGLE_DEG:
                continue
        except Exception:
            continue

        # --- Rule 5: Diminishing Volume ---
        # Force scalar floats for volume comparisons to avoid ambiguous Series truth-values
        try:
            vol_ls_val = data['Volume'].iloc[p1_idx:t1_idx+1].mean()
            vol_h_val = data['Volume'].iloc[t1_idx:t2_idx+1].mean()
            vol_rs_val = data['Volume'].iloc[t2_idx:p3_idx+1].mean()
        except Exception:
            # If slicing fails for any reason, skip this candidate
            continue

        try:
            vol_ls = float(vol_ls_val) if not np.isnan(vol_ls_val) else np.nan
            vol_h = float(vol_h_val) if not np.isnan(vol_h_val) else np.nan
            vol_rs = float(vol_rs_val) if not np.isnan(vol_rs_val) else np.nan
        except Exception:
            continue

        # If any volume measure is NaN, skip
        if np.isnan(vol_ls) or np.isnan(vol_h) or np.isnan(vol_rs):
            continue

        # stronger requirement on decreasing volume during formation
        if not (vol_ls > vol_h * VOLUME_TREND_TOL and vol_h > vol_rs * VOLUME_TREND_TOL):
            continue
        # optional volume divergence: lower volume on head vs left shoulder
        if REQUIRE_VOLUME_DIVERGENCE:
            try:
                if not (vol_h < vol_ls * 0.95):
                    continue
            except Exception:
                continue

        # --- Rule 6: Neckline Breakout Confirmation ---
        breakout_confirmed = False
        breakout_idx = -1
        for j in range(p3_idx + 1, len(data)):
            # use iloc for positional access
            date_j = data['Date'].iloc[j]
            neckline_price_at_j = slope * date_j.toordinal() + intercept
            # Check for a close below the neckline
            close_j = data['Close'].iloc[j]
            if close_j < neckline_price_at_j:
                # Check for volume spike on breakout
                avg_volume_pattern = float(data.iloc[p1_idx:p3_idx+1]['Volume'].mean())
                vol_j = float(data['Volume'].iloc[j])
                if vol_j > avg_volume_pattern * VOLUME_SPIKE_MULT:
                    breakout_confirmed = True
                    breakout_idx = j
                    break
            # Stop searching for breakout after a certain period
            if (data['Date'][j] - p3_date).days > 60:
                break
        
        if breakout_confirmed:
            pattern_data = {
                'P1': (p1_date, p1_high), 'T1': (t1_date, t1_low),
                'P2': (p2_date, p2_high), 'T2': (t2_date, t2_low),
                'P3': (p3_date, p3_high), 'breakout_idx': breakout_idx,
                'neckline_slope': slope, 'neckline_intercept': intercept
            }
            patterns.append(pattern_data)
            
    return patterns

def plot_pattern_zoom(df, pattern, stock_name, output_path):
    p1_date, p1_high = pattern['P1']
    t1_date, t1_low = pattern['T1']
    p2_date, p2_high = pattern['P2']
    t2_date, t2_low = pattern['T2']
    p3_date, p3_high = pattern['P3']
    breakout_idx = pattern['breakout_idx']
    
    start_date = p1_date - pd.Timedelta(days=30)
    end_date = df['Date'][breakout_idx] + pd.Timedelta(days=30)
    
    df_zoom = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    if df_zoom.empty: return

    # Create two stacked axes: price (top) and volume (bottom) sharing the x-axis
    fig, (ax_price, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                                          gridspec_kw={'height_ratios': [3, 1]})

    # Price line
    ax_price.plot(df_zoom['Date'], df_zoom['Close'], label='Close Price', color='navy')

    # Mark pattern points
    points_dates = [p1_date, p2_date, p3_date]
    points_prices = [p1_high, p2_high, p3_high]
    ax_price.scatter(points_dates, points_prices, color=['orange', 'red', 'orange'], s=120, zorder=5, label='Shoulders & Head')
    
    troughs_dates = [t1_date, t2_date]
    troughs_prices = [t1_low, t2_low]
    ax_price.scatter(troughs_dates, troughs_prices, color='blue', s=100, zorder=5, label='Troughs')

    # Draw Neckline
    slope = pattern['neckline_slope']
    intercept = pattern['neckline_intercept']
    neckline_x = [df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1]]
    neckline_y = [slope * d.toordinal() + intercept for d in neckline_x]
    ax_price.plot(neckline_x, neckline_y, 'g--', label='Neckline')

    # Mark Breakout on price
    breakout_date = df['Date'][breakout_idx]
    breakout_price = df['Close'][breakout_idx]
    ax_price.scatter(breakout_date, breakout_price, color='purple', s=150, zorder=6, marker='*', label='Breakout')

    # Volume bars on bottom axis
    if 'Volume' in df_zoom.columns:
        ax_vol.bar(df_zoom['Date'], df_zoom['Volume'], color='gray', alpha=0.6)
        # Highlight breakout volume bar
        try:
            ax_vol.bar(breakout_date, df.loc[breakout_idx, 'Volume'], color='purple', alpha=0.9)
        except Exception:
            pass

    # Formatting
    ax_price.set_xlim(df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1])
    ax_price.set_ylim(df_zoom['Low'].min() * 0.98, df_zoom['High'].max() * 1.02)
    ax_price.set_title(f"{stock_name} - Confirmed Head and Shoulders Pattern")
    ax_price.set_ylabel('Price')
    ax_vol.set_ylabel('Volume')
    ax_vol.set_xlabel('Date')

    ax_price.legend(loc='upper left')
    ax_price.grid(True, linestyle='--', alpha=0.6)
    ax_vol.grid(False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def detect_double_tops_and_bottoms(data, max_spacing_days=90):
    """Detect simple double tops and double bottoms using swing flags."""
    results = []
    sw = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    for i in range(len(sw) - 2):
        idx1, idx_mid, idx2 = sw['index'][i:i+3]
        # double top: High, Low, High
        if data.get('is_swing_high', False)[idx1] and data.get('is_swing_low', False)[idx_mid] and data.get('is_swing_high', False)[idx2]:
            p1_high, p1_date = data.loc[idx1, ['High', 'Date']]
            t_low, t_date = data.loc[idx_mid, ['Low', 'Date']]
            p2_high, p2_date = data.loc[idx2, ['High', 'Date']]

            # spacing check
            try:
                if (p2_date - p1_date).days > max_spacing_days:
                    continue
            except Exception:
                continue

            # Require preceding uptrend for a valid double-top
            if not check_preceding_uptrend(data, idx1):
                continue

            # shoulder similarity
            if abs(p1_high - p2_high) / max(p1_high, p2_high) > SHOULDER_TOL:
                continue

            # require peaks to be meaningfully above the trough (prominence)
            try:
                if not ((p1_high - t_low) / t_low > 0.03 and (p2_high - t_low) / t_low > 0.03):
                    continue
            except Exception:
                continue

            # neckline is trough t_low; look for breakout below trough
            breakout_idx = -1
            for j in range(idx2 + 1, len(data)):
                if data['Close'].iloc[j] < t_low:
                    # check for volume spike OR sustained close below neckline in next few days
                    avg_vol = float(data.iloc[idx1:idx2+1]['Volume'].mean()) if 'Volume' in data.columns else 0
                    vol_j = float(data['Volume'].iloc[j]) if 'Volume' in data.columns else 0
                    sustained = 0
                    look_ahead = min(len(data) - 1, j + 5)
                    for k in range(j, look_ahead + 1):
                        if data['Close'].iloc[k] < t_low:
                            sustained += 1
                    if (avg_vol > 0 and vol_j > avg_vol * VOLUME_SPIKE_MULT) or sustained >= 2:
                        breakout_idx = j
                        break
                if (data['Date'].iloc[j] - p2_date).days > 60:
                    break

            if breakout_idx == -1:
                # no validated breakout -> not a confirmed pattern
                continue

            results.append({'type': 'double_top', 'P1': (p1_date, p1_high), 'T': (t_date, t_low), 'P2': (p2_date, p2_high), 'breakout_idx': breakout_idx})

        # double bottom: Low, High, Low
        if data.get('is_swing_low', False)[idx1] and data.get('is_swing_high', False)[idx_mid] and data.get('is_swing_low', False)[idx2]:
            p1_low, p1_date = data.loc[idx1, ['Low', 'Date']]
            t_high, t_date = data.loc[idx_mid, ['High', 'Date']]
            p2_low, p2_date = data.loc[idx2, ['Low', 'Date']]

            try:
                if (p2_date - p1_date).days > max_spacing_days:
                    continue
            except Exception:
                continue

            # Require preceding downtrend for a valid double-bottom
            if not check_preceding_downtrend(data, idx1):
                continue

            if abs(p1_low - p2_low) / max(p1_low, p2_low) > SHOULDER_TOL:
                continue

            # require troughs to be meaningfully below the peak (prominence)
            try:
                if not ((t_high - p1_low) / t_high > 0.03 and (t_high - p2_low) / t_high > 0.03):
                    continue
            except Exception:
                continue

            # neckline is peak t_high; look for breakout above neckline
            breakout_idx = -1
            for j in range(idx2 + 1, len(data)):
                if data['Close'].iloc[j] > t_high:
                    avg_vol = float(data.iloc[idx1:idx2+1]['Volume'].mean()) if 'Volume' in data.columns else 0
                    vol_j = float(data['Volume'].iloc[j]) if 'Volume' in data.columns else 0
                    sustained = 0
                    look_ahead = min(len(data) - 1, j + 5)
                    for k in range(j, look_ahead + 1):
                        if data['Close'].iloc[k] > t_high:
                            sustained += 1
                    if (avg_vol > 0 and vol_j > avg_vol * VOLUME_SPIKE_MULT) or sustained >= 2:
                        breakout_idx = j
                        break
                if (data['Date'].iloc[j] - p2_date).days > 60:
                    break

            if breakout_idx == -1:
                continue

            results.append({'type': 'double_bottom', 'P1': (p1_date, p1_low), 'T': (t_date, t_high), 'P2': (p2_date, p2_low), 'breakout_idx': breakout_idx})

    return results


def plot_double_zoom(df, pattern, stock_name, output_path):
    p1_date, p1_price = pattern['P1']
    t_date, t_price = pattern['T']
    p2_date, p2_price = pattern['P2']
    breakout_idx = pattern.get('breakout_idx', -1)

    start_date = p1_date - pd.Timedelta(days=30)
    end_date = (df['Date'].iloc[breakout_idx] + pd.Timedelta(days=30)) if breakout_idx >= 0 else p2_date + pd.Timedelta(days=30)

    df_zoom = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    if df_zoom.empty: return

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3,1]})
    ax_price.plot(df_zoom['Date'], df_zoom['Close'], color='navy', label='Close')

    # mark points
    if pattern['type'] == 'double_top':
        ax_price.scatter([p1_date, p2_date], [p1_price, p2_price], color=['orange','orange'], s=120, zorder=5, label='Peaks')
        ax_price.scatter([t_date], [t_price], color='blue', s=100, zorder=5, label='Trough')
        # neckline
        ax_price.hlines(t_price, df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1], colors='g', linestyles='--', label='Neckline')
    else:
        ax_price.scatter([p1_date, p2_date], [p1_price, p2_price], color=['green','green'], s=120, zorder=5, label='Troughs')
        ax_price.scatter([t_date], [t_price], color='red', s=100, zorder=5, label='Peak')
        ax_price.hlines(t_price, df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1], colors='g', linestyles='--', label='Neckline')

    # breakout marker
    if breakout_idx >= 0:
        b_date = df['Date'].iloc[breakout_idx]
        b_price = df['Close'].iloc[breakout_idx]
        ax_price.scatter(b_date, b_price, color='purple', s=150, marker='*', label='Breakout')

    if 'Volume' in df_zoom.columns:
        ax_vol.bar(df_zoom['Date'], df_zoom['Volume'], color='gray', alpha=0.6)
        if breakout_idx >= 0:
            try:
                ax_vol.bar(b_date, df.loc[breakout_idx, 'Volume'], color='purple', alpha=0.9)
            except Exception:
                pass

    ax_price.set_ylabel('Price')
    ax_vol.set_ylabel('Volume')
    ax_vol.set_xlabel('Date')
    ax_price.legend()
    ax_price.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    # parse command line mode: strict (default), lenient (legacy), or both
    parser = argparse.ArgumentParser(description='Head and Shoulders detector')
    parser.add_argument('--mode', choices=['strict', 'lenient', 'both'], default='strict', help='detection mode')
    args = parser.parse_args()

    def apply_mode_settings(mode):
        # set globals to lenient/strict defaults
        if mode == 'lenient':
            globals()['SHOULDER_TOL'] = 0.15
            globals()['HEAD_OVER_SHOULDER_PCT'] = 0.0
            globals()['MAX_NECKLINE_ANGLE_DEG'] = 90
            globals()['VOLUME_TREND_TOL'] = 0.8
            globals()['VOLUME_SPIKE_MULT'] = 1.5
            globals()['USE_ZIGZAG'] = False
            globals()['USE_FRACTAL'] = False
            globals()['REQUIRE_VOLUME_DIVERGENCE'] = False
            globals()['REQUIRE_ADX'] = False
            globals()['REQUIRE_RSI_OR_MACD'] = False
            globals()['REQUIRE_MA_CONFIRM'] = False
        else:
            # strict: leave as defined at top (no change)
            pass

    modes_to_run = []
    if args.mode == 'both':
        modes_to_run = ['lenient', 'strict']
    else:
        modes_to_run = [args.mode]

    # List of ~25 popular NSE tickers to analyze (add or remove as needed)
    symbols = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'HINDUNILVR.NS',
        'ITC.NS', 'MARUTI.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'BAJFINANCE.NS',
        'HCLTECH.NS', 'WIPRO.NS', 'ONGC.NS', 'BPCL.NS', 'TATAMOTORS.NS',
        'TATASTEEL.NS', 'JSWSTEEL.NS', 'SUNPHARMA.NS', 'LTIM.NS', 'NESTLEIND.NS'
    ]

    start_date = '2015-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Run detection for multiple time windows and save results separately
    windows = {
        '1y': 365,
        '2y': 365*2,
        '3y': 365*3,
        '5y': 365*5,
    }

    for run_mode in modes_to_run:
        print(f"\n==== Running mode: {run_mode} ====")
        apply_mode_settings(run_mode)
        summary_records = []

        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            try:
                df = load_data(symbol, start_date, end_date)
            except Exception as e:
                print(f"Failed to download {symbol}: {e}")
                # short sleep to avoid hammering the API on repeated failures
                time.sleep(1)
                continue

            if df is None or df.empty:
                print(f"No data for {symbol}, skipping.")
                time.sleep(0.5)
                continue

            # Compute swing points once for full history (used to slice windows)
            df = df.reset_index(drop=True)
            df = find_swing_points(df, N_bars=20)

            out_base = os.path.join(os.getcwd(), 'PatternCharts', symbol)
            os.makedirs(out_base, exist_ok=True)

            for name, days in windows.items():
                df_slice = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=days)].copy()
                if df_slice.empty:
                    print(f"{symbol} - {name}: No data for window {name}")
                    continue

                # Reset index and compute dynamic swing points on the slice so indices are positional (0..n-1)
                df_slice = df_slice.reset_index(drop=True)
                nbars_slice = compute_dynamic_nbars(df_slice, base=BASE_NBARS)
                if USE_ZIGZAG:
                    df_slice = generate_swing_flags(df_slice, method='zigzag', N_bars=nbars_slice)
                elif USE_FRACTAL:
                    df_slice = generate_swing_flags(df_slice, method='fractal', N_bars=nbars_slice)
                else:
                    df_slice = generate_swing_flags(df_slice, method='rolling', N_bars=nbars_slice)

                patterns = detect_head_and_shoulders(df_slice)
                # detect double tops / bottoms
                double_patterns = detect_double_tops_and_bottoms(df_slice)
                # prepare output folders for this symbol/window
                base_folder = os.path.join(out_base, name)
                os.makedirs(base_folder, exist_ok=True)
                hns_folder = os.path.join(base_folder, 'HNS')
                dt_folder = os.path.join(base_folder, 'DT')
                db_folder = os.path.join(base_folder, 'DB')
                os.makedirs(hns_folder, exist_ok=True)
                os.makedirs(dt_folder, exist_ok=True)
                os.makedirs(db_folder, exist_ok=True)

                # Handle Head & Shoulders patterns
                if patterns:
                    print(f"{symbol} - {name}: {len(patterns)} confirmed pattern(s) detected")
                    for idx, pattern in enumerate(patterns, 1):
                        out_path = os.path.join(hns_folder, f'{symbol}_HNS_pattern_zoom_{name}_{idx}.png')
                        try:
                            plot_pattern_zoom(df_slice, pattern, symbol, out_path)
                        except Exception as e:
                            print(f"Failed to plot pattern for {symbol} {name} #{idx}: {e}")
                        # Record summary info for CSV
                        try:
                            p1_date, p1_high = pattern['P1']
                            t1_date, t1_low = pattern['T1']
                            p2_date, p2_high = pattern['P2']
                            t2_date, t2_low = pattern['T2']
                            p3_date, p3_high = pattern['P3']
                            breakout_idx = pattern.get('breakout_idx', -1)
                            breakout_date = df_slice['Date'].iloc[breakout_idx] if breakout_idx >= 0 and breakout_idx < len(df_slice) else None
                        except Exception:
                            # Fallback: skip recording malformed pattern
                            breakout_date = None
                            p1_date = p2_date = p3_date = None
                            p1_high = p2_high = p3_high = None
                            t1_date = t2_date = None
                            t1_low = t2_low = None
                        summary_records.append({
                            'pattern_type': 'head_and_shoulders',
                            'symbol': symbol,
                            'window': name,
                            'P1_date': pd.to_datetime(p1_date).strftime('%Y-%m-%d') if p1_date is not None else None,
                            'P1_high': p1_high,
                            'T1_date': pd.to_datetime(t1_date).strftime('%Y-%m-%d') if t1_date is not None else None,
                            'T1_low': t1_low,
                            'P2_date': pd.to_datetime(p2_date).strftime('%Y-%m-%d') if p2_date is not None else None,
                            'P2_high': p2_high,
                            'T2_date': pd.to_datetime(t2_date).strftime('%Y-%m-%d') if t2_date is not None else None,
                            'T2_low': t2_low,
                            'P3_date': pd.to_datetime(p3_date).strftime('%Y-%m-%d') if p3_date is not None else None,
                            'P3_high': p3_high,
                            'breakout_date': pd.to_datetime(breakout_date).strftime('%Y-%m-%d') if breakout_date is not None else None,
                            'image_path': out_path
                        })
                else:
                    print(f"{symbol} - {name}: No confirmed head and shoulders pattern detected.")

                # handle double patterns
                if double_patterns:
                    print(f"{symbol} - {name}: {len(double_patterns)} double pattern(s) detected")
                    for d_idx, dpat in enumerate(double_patterns, 1):
                        if dpat.get('type') == 'double_top':
                            out_path = os.path.join(dt_folder, f'{symbol}_DT_pattern_{name}_{d_idx}.png')
                        else:
                            out_path = os.path.join(db_folder, f'{symbol}_DB_pattern_{name}_{d_idx}.png')
                        try:
                            plot_double_zoom(df_slice, dpat, symbol, out_path)
                        except Exception as e:
                            print(f"Failed to plot double pattern for {symbol} {name} #{d_idx}: {e}")
                        # record
                        try:
                            p1_date, p1_price = dpat['P1']
                            t_date, t_price = dpat['T']
                            p2_date, p2_price = dpat['P2']
                            breakout_idx = dpat.get('breakout_idx', -1)
                            breakout_date = df_slice['Date'].iloc[breakout_idx] if breakout_idx >= 0 and breakout_idx < len(df_slice) else None
                        except Exception:
                            breakout_date = None
                            p1_date = p2_date = t_date = None
                            p1_price = p2_price = t_price = None
                        summary_records.append({
                            'pattern_type': dpat.get('type', 'double'),
                            'symbol': symbol,
                            'window': name,
                            'P1_date': pd.to_datetime(p1_date).strftime('%Y-%m-%d') if p1_date is not None else None,
                            'P1_price': p1_price,
                            'T_date': pd.to_datetime(t_date).strftime('%Y-%m-%d') if t_date is not None else None,
                            'T_price': t_price,
                            'P2_date': pd.to_datetime(p2_date).strftime('%Y-%m-%d') if p2_date is not None else None,
                            'P2_price': p2_price,
                            'breakout_date': pd.to_datetime(breakout_date).strftime('%Y-%m-%d') if breakout_date is not None else None,
                            'image_path': out_path
                        })

        # Be polite to the API
        time.sleep(1)

        # Save CSV summary of all detected patterns for this mode
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            csv_path = os.path.join(os.getcwd(), f'pattern_detection_summary_{run_mode}.csv')
            try:
                summary_df.to_csv(csv_path, index=False)
                print(f"\nSaved summary CSV to {csv_path}")
            except Exception as e:
                print(f"Failed to write summary CSV: {e}")
        else:
            print(f"\nNo patterns detected across all symbols in mode={run_mode}; no summary CSV created.")
