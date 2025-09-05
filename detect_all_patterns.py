#!/usr/bin/env python3
"""
Comprehensive Stock Pattern Detector
====================================
Unified script for detecting multiple chart patterns with extensive configuration options.

Supported Patterns:
- Head and Shoulders (HNS)
- Cup and Handle (CH)
- Double Top (DT)
- Double Bottom (DB)

Features:
- Multiple timeframes (1d to 5y)
- Configurable detection modes (strict/lenient)
- Volume confirmation options
- Preceding trend analysis
- Customizable output formats
- Extensive filtering options
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import argparse
import json
import random
from pathlib import Path

# Optional Plotly backend (interactive charts)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Pattern-specific tuning constants
HNS_CONFIG = {
    'strict': {
        'SHOULDER_TOL': 0.10,
        'HEAD_OVER_SHOULDER_PCT': 0.07,
        'MAX_NECKLINE_ANGLE_DEG': 25,
        'VOLUME_TREND_TOL': 0.95,
        'VOLUME_SPIKE_MULT': 1.75,
    },
    'lenient': {
        'SHOULDER_TOL': 0.15,
        'HEAD_OVER_SHOULDER_PCT': 0.03,
        'MAX_NECKLINE_ANGLE_DEG': 45,
        'VOLUME_TREND_TOL': 0.85,
        'VOLUME_SPIKE_MULT': 1.25,
    }
}

CH_CONFIG = {
    'strict': {
        'CUP_DEPTH_MIN': 0.15,
        'CUP_DEPTH_MAX': 0.40,
        'HANDLE_DEPTH_MAX': 0.20,
        'CUP_SYMMETRY_TOL': 0.25,
        'HANDLE_DURATION_MIN': 7,
        'HANDLE_DURATION_MAX': 60,
        'VOLUME_DECLINE_PCT': 0.85,
        'VOLUME_SPIKE_MULT': 1.75,
    },
    'lenient': {
        'CUP_DEPTH_MIN': 0.08,
        'CUP_DEPTH_MAX': 0.60,
        'HANDLE_DEPTH_MAX': 0.35,
        'CUP_SYMMETRY_TOL': 0.50,
        'HANDLE_DURATION_MIN': 3,
        'HANDLE_DURATION_MAX': 90,
        'VOLUME_DECLINE_PCT': 0.70,
        'VOLUME_SPIKE_MULT': 1.25,
    }
}

DT_CONFIG = {
    'strict': {
        'PEAK_SIMILARITY_TOL': 0.06,
        'MIN_PROMINENCE_PCT': 0.06,
        'MAX_SPACING_DAYS': 90,
        'MIN_SPACING_DAYS': 20,
        'VOLUME_DECLINE_PCT': 0.90,
        'VOLUME_SPIKE_MULT': 1.75,
        'NECKLINE_TOLERANCE': 0.015,
    },
    'lenient': {
    'PEAK_SIMILARITY_TOL': 0.06,
        'MIN_PROMINENCE_PCT': 0.03,
        'MAX_SPACING_DAYS': 150,
        'MIN_SPACING_DAYS': 10,
        'VOLUME_DECLINE_PCT': 0.75,
        'VOLUME_SPIKE_MULT': 1.25,
        'NECKLINE_TOLERANCE': 0.03,
    }
}

# Global settings
GLOBAL_CONFIG = {
    'BASE_NBARS': 20,
    'MIN_NBARS': 8,
    'MAX_NBARS': 60,
    'ZIGZAG_PCT': 0.03,
    'MIN_TREND_PCT': 0.15,
    'CUP_DURATION_MIN': 30,
    'CUP_DURATION_MAX': 365,
}

# Default symbol lists
DEFAULT_SYMBOLS = {
    'us_tech': ['AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD'],
    'us_large': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'JPM', 'JNJ', 'PG', 'KO', 'WMT'],
    'indian_popular': [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS',
        'BHARTIARTL.NS', 'HINDUNILVR.NS', 'ITC.NS', 'LT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS',
        'MARUTI.NS', 'ASIANPAINT.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'WIPRO.NS',
        'ONGC.NS', 'SUNPHARMA.NS', 'LTIM.NS', 'NESTLEIND.NS', 'TATAMOTORS.NS',
        'TATASTEEL.NS', 'JSWSTEEL.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS'
    ],
    'indian_banking': [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS',
        'BANKBARODA.NS', 'INDUSINDBK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS'
    ]
   
}


def get_random_symbols(count=None):
    """Get a random selection of symbols from all available symbol pools."""
    if count is None:
        count = random.randint(5, 10)
    
    # Combine all symbols from different categories
    all_symbols = []
    for category, symbols in DEFAULT_SYMBOLS.items():
        all_symbols.extend(symbols)
    
    # Remove duplicates while preserving order
    unique_symbols = list(dict.fromkeys(all_symbols))
    
    # Randomly select the requested number of symbols
    selected_count = min(count, len(unique_symbols))
    selected_symbols = random.sample(unique_symbols, selected_count)
    
    return selected_symbols

# Timeframe configurations
TIMEFRAMES = {
    '1d': 1,
    '1w': 7,
    '2w': 14,
    '1m': 30,
    '2m': 60,
    '3m': 90,
    '6m': 180,
    '1y': 365,
    '2y': 730,
    '3y': 1095,
    '5y': 1825,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_data(symbol, start_date, end_date):
    """Load stock data from yfinance."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        data = data.reset_index()
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            new_cols = []
            for c in data.columns:
                if c[0] and c[0] != '':
                    new_cols.append(c[0])
                else:
                    new_cols.append(c[1])
            data.columns = new_cols

        # Ensure 'Date' is datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])

        return data
    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        return None

def load_data_from_csv(symbol: str, data_dir: str | os.PathLike) -> pd.DataFrame | None:
    """Load stock data from a local CSV file.

    Expects columns including Date, Open, High, Low, Close, Volume (Adj Close optional).
    Will try to match filename case-insensitively and will also try stripping common suffixes
    like '.NS' when looking for a matching CSV.
    """
    try:
        base = symbol.strip()
        # Try removing common Yahoo suffix
        for suff in ('.NS', '.BS', '.L', '.HK', '.TO', '.AX'):
            if base.upper().endswith(suff):
                base = base[: -len(suff)]
                break
        # Candidate filenames to try
        candidates = [f"{base}.csv", f"{base.upper()}.csv", f"{base.lower()}.csv"]
        data_dir = str(data_dir)
        # Build case-insensitive map of files in dir once
        files = {}
        try:
            for fn in os.listdir(data_dir):
                files[fn.lower()] = os.path.join(data_dir, fn)
        except FileNotFoundError:
            return None
        path = None
        for cand in candidates:
            p = files.get(cand.lower())
            if p:
                path = p
                break
        if not path:
            return None
        df = pd.read_csv(path)
        # Normalize columns
        cols = {c.strip(): c for c in df.columns}
        # Some csvs might have lowercase headers
        def col(name: str):
            for k in list(cols.keys()):
                if k.lower() == name.lower():
                    return cols[k]
            return None
        date_col = col('Date') or col('date')
        open_col = col('Open') or col('open')
        high_col = col('High') or col('high')
        low_col = col('Low') or col('low')
        close_col = col('Close') or col('close')
        vol_col = col('Volume') or col('volume')
        if not (date_col and open_col and high_col and low_col and close_col and vol_col):
            return None
        df = df.rename(columns={
            date_col: 'Date',
            open_col: 'Open',
            high_col: 'High',
            low_col: 'Low',
            close_col: 'Close',
            vol_col: 'Volume',
        })
        # Parse dates
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception:
            # Try common formats fallback
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[df['Date'].notna()]
        # Ensure sorting
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception:
        return None

def find_swing_points(data, N_bars=20):
    """Find swing highs and lows using rolling windows."""
    data['is_swing_high'] = (data['High'] == data['High'].rolling(window=N_bars*2+1, center=True).max())
    data['is_swing_low'] = (data['Low'] == data['Low'].rolling(window=N_bars*2+1, center=True).min())
    return data

def compute_atr(data, period=14):
    """Compute Average True Range."""
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def compute_dynamic_nbars(data, base=20, min_bars=8, max_bars=60):
    """Compute dynamic N_bars based on volatility."""
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

def compute_zigzag(data, pct=0.03):
    """Compute ZigZag swing points."""
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
    last_type = 'unknown'
    
    for i in range(1, n):
        change = (closes[i] - last_extreme_price) / last_extreme_price
        if last_type in ('unknown', 'low') and change >= pct:
            is_high[i] = True
            last_extreme_idx = i
            last_extreme_price = closes[i]
            last_type = 'high'
        elif last_type in ('unknown', 'high') and change <= -pct:
            is_low[i] = True
            last_extreme_idx = i
            last_extreme_price = closes[i]
            last_type = 'low'

    data['is_swing_high'] = is_high
    data['is_swing_low'] = is_low
    return data

def compute_fractals(data, left=2, right=2):
    """Compute Williams Fractals."""
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

def generate_swing_flags(data, method='rolling', N_bars=20):
    """Generate swing point flags using specified method."""
    if method == 'fractal':
        return compute_fractals(data)
    elif method == 'zigzag':
        return compute_zigzag(data, pct=GLOBAL_CONFIG['ZIGZAG_PCT'])
    else:  # rolling
        return find_swing_points(data, N_bars=N_bars)

def check_preceding_trend(data, pattern_start_index, trend_type='up', lookback_period=90, min_change_percent=0.15):
    """Check for preceding trend before pattern formation."""
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
        start_price = float(lookback_data['Close'].iloc[0])
        if trend_type == 'up':
            end_price = float(data.iloc[p_idx]['High'])
            return end_price > start_price * (1 + min_change_percent)
        else:  # down
            end_price = float(data.iloc[p_idx]['Low'])
            return end_price < start_price * (1 - min_change_percent)
    except Exception:
        return False

# =============================================================================
# PATTERN DETECTION FUNCTIONS
# =============================================================================

def detect_head_and_shoulders(data, config, require_preceding_trend=True):
    """Detect Head and Shoulders patterns."""
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    patterns = []

    for i in range(len(swing_points_df) - 4):
        # Get indices of the 5 swing points
        p1_idx, t1_idx, p2_idx, t2_idx, p3_idx = swing_points_df['index'][i:i+5]

        # Ensure correct sequence (High, Low, High, Low, High)
        if not (data['is_swing_high'][p1_idx] and data['is_swing_low'][t1_idx] and
                data['is_swing_high'][p2_idx] and data['is_swing_low'][t2_idx] and
                data['is_swing_high'][p3_idx]):
            continue

        # Get prices and dates
        p1_high, p1_date = data.loc[p1_idx, ['High', 'Date']]
        t1_low, t1_date = data.loc[t1_idx, ['Low', 'Date']]
        p2_high, p2_date = data.loc[p2_idx, ['High', 'Date']]
        t2_low, t2_date = data.loc[t2_idx, ['Low', 'Date']]
        p3_high, p3_date = data.loc[p3_idx, ['High', 'Date']]

        # Check preceding uptrend
        if require_preceding_trend and not check_preceding_trend(data, p1_idx, 'up'):
            continue

        # Head must be highest and above shoulders
        higher_shoulder = max(p1_high, p3_high)
        if not (p2_high > p1_high and p2_high > p3_high):
            continue
        if not (p2_high > higher_shoulder * (1 + config['HEAD_OVER_SHOULDER_PCT'])):
            continue

        # Shoulder proportionality
        if abs(p1_high - p3_high) / max(p1_high, p3_high) > config['SHOULDER_TOL']:
            continue

        # Neckline calculation and angle check
        neckline_y = [t1_low, t2_low]
        neckline_x = [t1_date.toordinal(), t2_date.toordinal()]
        if neckline_x[1] - neckline_x[0] == 0:
            continue
        
        slope = (neckline_y[1] - neckline_y[0]) / (neckline_x[1] - neckline_x[0])
        intercept = neckline_y[0] - slope * neckline_x[0]
        
        try:
            angle_deg = abs(np.degrees(np.arctan(slope)))
            if angle_deg > config['MAX_NECKLINE_ANGLE_DEG']:
                continue
        except Exception:
            continue

        # Volume analysis
        try:
            vol_ls = float(data['Volume'].iloc[p1_idx:t1_idx+1].mean())
            vol_h = float(data['Volume'].iloc[t1_idx:t2_idx+1].mean())
            vol_rs = float(data['Volume'].iloc[t2_idx:p3_idx+1].mean())
            
            if np.isnan(vol_ls) or np.isnan(vol_h) or np.isnan(vol_rs):
                continue
                
            if not (vol_ls > vol_h * config['VOLUME_TREND_TOL'] and vol_h > vol_rs * config['VOLUME_TREND_TOL']):
                continue
        except Exception:
            continue

        # Look for neckline breakout
        breakout_confirmed = False
        breakout_idx = -1
        
        for j in range(p3_idx + 1, min(len(data), p3_idx + 90)):
            date_j = data['Date'].iloc[j]
            neckline_price_at_j = slope * date_j.toordinal() + intercept
            close_j = data['Close'].iloc[j]
            
            if close_j < neckline_price_at_j:
                avg_volume_pattern = float(data.iloc[p1_idx:p3_idx+1]['Volume'].mean())
                vol_j = float(data['Volume'].iloc[j])
                if vol_j > avg_volume_pattern * config['VOLUME_SPIKE_MULT']:
                    breakout_confirmed = True
                    breakout_idx = j
                    break

        if breakout_confirmed:
            pattern_data = {
                'type': 'head_and_shoulders',
                'P1': (p1_date, p1_high, p1_idx),
                'T1': (t1_date, t1_low, t1_idx),
                'P2': (p2_date, p2_high, p2_idx),
                'T2': (t2_date, t2_low, t2_idx),
                'P3': (p3_date, p3_high, p3_idx),
                'breakout': (data['Date'].iloc[breakout_idx], data['Close'].iloc[breakout_idx], breakout_idx),
                'neckline_slope': slope,
                'neckline_intercept': intercept,
                'duration': (p3_date - p1_date).days
            }
            patterns.append(pattern_data)

    return patterns

def detect_cup_and_handle(data, config, require_preceding_trend=True):
    """Detect Cup and Handle patterns."""
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    patterns = []

    # Get high swing points
    high_indices = swing_points_df[swing_points_df['index'].apply(
        lambda x: data.iloc[x].get('is_swing_high', False))].reset_index(drop=True)

    for i in range(len(high_indices) - 1):
        # Cup left rim
        left_rim_idx = high_indices.iloc[i]['index']
        left_rim_high = data.loc[left_rim_idx, 'High']
        left_rim_date = data.loc[left_rim_idx, 'Date']

        for j in range(i + 1, len(high_indices)):
            # Cup right rim
            right_rim_idx = high_indices.iloc[j]['index']
            right_rim_high = data.loc[right_rim_idx, 'High']
            right_rim_date = data.loc[right_rim_idx, 'Date']

            # Check cup duration
            try:
                cup_duration = (right_rim_date - left_rim_date).days
            except Exception:
                continue

            if cup_duration < GLOBAL_CONFIG['CUP_DURATION_MIN'] or cup_duration > GLOBAL_CONFIG['CUP_DURATION_MAX']:
                continue

            # Find cup bottom
            cup_section = data.iloc[left_rim_idx:right_rim_idx+1]
            if cup_section.empty:
                continue

            cup_bottom_idx = cup_section['Low'].idxmin()
            cup_bottom_low = data.loc[cup_bottom_idx, 'Low']
            cup_bottom_date = data.loc[cup_bottom_idx, 'Date']

            # Check preceding uptrend
            if require_preceding_trend and not check_preceding_trend(data, left_rim_idx, 'up'):
                continue

            # Cup depth check
            left_rim_price = max(left_rim_high, right_rim_high)
            cup_depth = (left_rim_price - cup_bottom_low) / left_rim_price
            if cup_depth < config['CUP_DEPTH_MIN'] or cup_depth > config['CUP_DEPTH_MAX']:
                continue


            # Cup symmetry check (enforce U-shape, not V)
            left_side_duration = (cup_bottom_date - left_rim_date).days
            right_side_duration = (right_rim_date - cup_bottom_date).days
            total_cup_duration = (right_rim_date - left_rim_date).days
            if left_side_duration == 0 or right_side_duration == 0:
                continue
            symmetry_ratio = min(left_side_duration, right_side_duration) / max(left_side_duration, right_side_duration)
            if symmetry_ratio < (1 - config['CUP_SYMMETRY_TOL']):
                continue
            # Enforce U-shape: both sides must be at least 30% of total cup duration (no sharp V)
            if left_side_duration < 0.3 * total_cup_duration or right_side_duration < 0.3 * total_cup_duration:
                continue

            # Rim height similarity
            rim_difference = abs(left_rim_high - right_rim_high) / max(left_rim_high, right_rim_high)
            if rim_difference > 0.05:
                continue


            # Look for handle formation and breakout
            handle_found = False
            handle_low_idx = -1
            handle_end_idx = -1

            for k in range(right_rim_idx + 1, min(len(data), right_rim_idx + config['HANDLE_DURATION_MAX'])):
                current_low = data.loc[k, 'Low']
                current_date = data.loc[k, 'Date']

                handle_depth = (right_rim_high - current_low) / right_rim_high
                handle_duration = (current_date - right_rim_date).days

                # Enforce: handle must not be too deep or too long
                if handle_depth > config['HANDLE_DEPTH_MAX']:
                    break
                if handle_duration > config['HANDLE_DURATION_MAX']:
                    break

                # Handle must not retrace more than 1/3 of cup height
                cup_height = right_rim_high - cup_bottom_low
                if cup_height > 0 and (right_rim_high - current_low) > (cup_height / 3):
                    break

                if handle_duration >= config['HANDLE_DURATION_MIN']:
                    # Look for breakout
                    for m in range(k + 1, min(len(data), k + 30)):
                        if data.loc[m, 'Close'] > right_rim_high:
                            try:
                                avg_pattern_volume = data.iloc[left_rim_idx:k]['Volume'].mean()
                                breakout_volume = data.loc[m, 'Volume']
                                if breakout_volume > avg_pattern_volume * config['VOLUME_SPIKE_MULT']:
                                    handle_found = True
                                    handle_low_idx = k
                                    handle_end_idx = m
                                    break
                            except Exception:
                                handle_found = True
                                handle_low_idx = k
                                handle_end_idx = m
                                break

                if handle_found:
                    break

            if handle_found:
                pattern_data = {
                    'type': 'cup_and_handle',
                    'left_rim': (left_rim_date, left_rim_high, left_rim_idx),
                    'cup_bottom': (cup_bottom_date, cup_bottom_low, cup_bottom_idx),
                    'right_rim': (right_rim_date, right_rim_high, right_rim_idx),
                    'handle_low': (data.loc[handle_low_idx, 'Date'], data.loc[handle_low_idx, 'Low'], handle_low_idx),
                    'breakout': (data.loc[handle_end_idx, 'Date'], data.loc[handle_end_idx, 'Close'], handle_end_idx),
                    'cup_duration': cup_duration,
                    'cup_depth': cup_depth
                }
                patterns.append(pattern_data)

    return patterns

def detect_double_patterns(data, config, pattern_type='both', require_preceding_trend=True):
    """Detect Double Top and/or Double Bottom patterns."""
    results = []
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    
    for i in range(len(swing_points_df) - 2):
        idx1, idx_mid, idx2 = swing_points_df['index'][i:i+3]
        
        # Double Top detection
        if (pattern_type in ('both', 'double_top') and 
            data.get('is_swing_high', False)[idx1] and 
            data.get('is_swing_low', False)[idx_mid] and 
            data.get('is_swing_high', False)[idx2]):
            
            p1_high, p1_date = data.loc[idx1, ['High', 'Date']]
            t_low, t_date = data.loc[idx_mid, ['Low', 'Date']]
            p2_high, p2_date = data.loc[idx2, ['High', 'Date']]

            try:
                spacing = (p2_date - p1_date).days
            except Exception:
                continue

            if spacing < config['MIN_SPACING_DAYS'] or spacing > config['MAX_SPACING_DAYS']:
                continue

            # Check preceding uptrend
            if require_preceding_trend and not check_preceding_trend(data, idx1, 'up'):
                continue

            # Peak similarity
            if abs(p1_high - p2_high) / max(p1_high, p2_high) > config['PEAK_SIMILARITY_TOL']:
                continue

            # Prominence check
            try:
                prominence1 = (p1_high - t_low) / t_low
                prominence2 = (p2_high - t_low) / t_low
                if prominence1 < config['MIN_PROMINENCE_PCT'] or prominence2 < config['MIN_PROMINENCE_PCT']:
                    continue
            except Exception:
                continue

            # Look for neckline breakout
            neckline_level = t_low
            breakout_idx = -1
            
            for j in range(idx2 + 1, min(len(data), idx2 + 90)):
                if data['Close'].iloc[j] < neckline_level * (1 - config['NECKLINE_TOLERANCE']):
                    try:
                        avg_vol = float(data.iloc[idx1:idx2+1]['Volume'].mean())
                        vol_j = float(data['Volume'].iloc[j])
                        if vol_j > avg_vol * config['VOLUME_SPIKE_MULT']:
                            breakout_idx = j
                            break
                    except Exception:
                        # Sustained break check
                        sustained = 0
                        look_ahead = min(len(data) - 1, j + 5)
                        for k in range(j, look_ahead + 1):
                            if data['Close'].iloc[k] < neckline_level:
                                sustained += 1
                        if sustained >= 2:
                            breakout_idx = j
                            break

            if breakout_idx != -1:
                results.append({
                    'type': 'double_top',
                    'P1': (p1_date, p1_high, idx1),
                    'T': (t_date, t_low, idx_mid),
                    'P2': (p2_date, p2_high, idx2),
                    'breakout': (data['Date'].iloc[breakout_idx], data['Close'].iloc[breakout_idx], breakout_idx),
                    'neckline_level': neckline_level,
                    'duration': spacing
                })

        # Double Bottom detection
        if (pattern_type in ('both', 'double_bottom') and 
            data.get('is_swing_low', False)[idx1] and 
            data.get('is_swing_high', False)[idx_mid] and 
            data.get('is_swing_low', False)[idx2]):
            
            p1_low, p1_date = data.loc[idx1, ['Low', 'Date']]
            t_high, t_date = data.loc[idx_mid, ['High', 'Date']]
            p2_low, p2_date = data.loc[idx2, ['Low', 'Date']]

            try:
                spacing = (p2_date - p1_date).days
            except Exception:
                continue

            if spacing < config['MIN_SPACING_DAYS'] or spacing > config['MAX_SPACING_DAYS']:
                continue

            # Check preceding downtrend
            if require_preceding_trend and not check_preceding_trend(data, idx1, 'down'):
                continue

            # Trough similarity
            if abs(p1_low - p2_low) / max(p1_low, p2_low) > config['PEAK_SIMILARITY_TOL']:
                continue

            # Prominence check
            try:
                prominence1 = (t_high - p1_low) / t_high
                prominence2 = (t_high - p2_low) / t_high
                if prominence1 < config['MIN_PROMINENCE_PCT'] or prominence2 < config['MIN_PROMINENCE_PCT']:
                    continue
            except Exception:
                continue

            # Look for neckline breakout
            neckline_level = t_high
            breakout_idx = -1
            
            for j in range(idx2 + 1, min(len(data), idx2 + 90)):
                if data['Close'].iloc[j] > neckline_level * (1 + config['NECKLINE_TOLERANCE']):
                    try:
                        avg_vol = float(data.iloc[idx1:idx2+1]['Volume'].mean())
                        vol_j = float(data['Volume'].iloc[j])
                        if vol_j > avg_vol * config['VOLUME_SPIKE_MULT']:
                            breakout_idx = j
                            break
                    except Exception:
                        # Sustained break check
                        sustained = 0
                        look_ahead = min(len(data) - 1, j + 5)
                        for k in range(j, look_ahead + 1):
                            if data['Close'].iloc[k] > neckline_level:
                                sustained += 1
                        if sustained >= 2:
                            breakout_idx = j
                            break

            if breakout_idx != -1:
                results.append({
                    'type': 'double_bottom',
                    'P1': (p1_date, p1_low, idx1),
                    'T': (t_date, t_high, idx_mid),
                    'P2': (p2_date, p2_low, idx2),
                    'breakout': (data['Date'].iloc[breakout_idx], data['Close'].iloc[breakout_idx], breakout_idx),
                    'neckline_level': neckline_level,
                    'duration': spacing
                })

    return results

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_hns_pattern(df, pattern, stock_name, output_path):
    """Plot Head and Shoulders pattern."""
    p1_date, p1_high, p1_idx = pattern['P1']
    t1_date, t1_low, t1_idx = pattern['T1']
    p2_date, p2_high, p2_idx = pattern['P2']
    t2_date, t2_low, t2_idx = pattern['T2']
    p3_date, p3_high, p3_idx = pattern['P3']
    breakout_date, breakout_price, breakout_idx = pattern['breakout']

    # Display the entire available dataset window instead of zooming
    df_zoom = df.copy()
    if df_zoom.empty:
        return

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                                          gridspec_kw={'height_ratios': [3, 1]})

    # Price line
    ax_price.plot(df_zoom['Date'], df_zoom['Close'], label='Close Price', color='navy')


    # Mark pattern points
    ax_price.scatter([p1_date, p2_date, p3_date], [p1_high, p2_high, p3_high], 
                    color=['orange', 'red', 'orange'], s=120, zorder=5, label='Shoulders & Head')
    ax_price.scatter([t1_date, t2_date], [t1_low, t2_low], 
                    color='blue', s=100, zorder=5, label='Troughs')

    # Draw neckline
    slope = pattern['neckline_slope']
    intercept = pattern['neckline_intercept']
    neckline_x = [df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1]]
    neckline_y = [slope * d.toordinal() + intercept for d in neckline_x]
    ax_price.plot(neckline_x, neckline_y, 'g--', label='Neckline')

    # Mark breakout
    ax_price.scatter(breakout_date, breakout_price, color='purple', s=150, 
                    zorder=6, marker='*', label='Breakout')

    # Volume bars
    if 'Volume' in df_zoom.columns:
        ax_vol.bar(df_zoom['Date'], df_zoom['Volume'], color='gray', alpha=0.6)
        try:
            ax_vol.bar(breakout_date, df.loc[breakout_idx, 'Volume'], color='purple', alpha=0.9)
        except Exception:
            pass

    ax_price.set_title(f"{stock_name} - Head and Shoulders Pattern")
    ax_price.set_ylabel('Price')
    ax_vol.set_ylabel('Volume')
    ax_vol.set_xlabel('Date')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_ch_pattern(df, pattern, stock_name, output_path):
    """Plot Cup and Handle pattern."""
    left_rim_date, left_rim_high, left_rim_idx = pattern['left_rim']
    cup_bottom_date, cup_bottom_low, cup_bottom_idx = pattern['cup_bottom']
    right_rim_date, right_rim_high, right_rim_idx = pattern['right_rim']
    handle_date, handle_low, handle_low_idx = pattern['handle_low']
    breakout_date, breakout_price, breakout_idx = pattern['breakout']

    # Display the entire available dataset window instead of zooming
    df_zoom = df.copy()
    if df_zoom.empty:
        return

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                                          gridspec_kw={'height_ratios': [3, 1]})

    # Price line
    ax_price.plot(df_zoom['Date'], df_zoom['Close'], label='Close Price', color='navy')


    # Mark Cup Rims (unique labels)
    ax_price.scatter([left_rim_date], [left_rim_high], color='blue', s=120, zorder=5, label='Left Rim', marker='o', edgecolors='black', linewidths=1.5)
    ax_price.scatter([right_rim_date], [right_rim_high], color='blue', s=120, zorder=5, label='Right Rim', marker='o', edgecolors='black', linewidths=1.5)
    # Mark Cup Bottom (single dot)
    ax_price.scatter([cup_bottom_date], [cup_bottom_low], color='red', s=120, zorder=6, label='Cup Bottom', marker='o', edgecolors='black', linewidths=1.5)

    # Mark Handle Low
    ax_price.scatter([handle_date], [handle_low], color='orange', s=100, zorder=6, label='Handle Low', marker='v', edgecolors='black', linewidths=1.2)

    # Draw Resistance Level: horizontal line from right rim
    resistance_level = max(left_rim_high, right_rim_high)
    ax_price.hlines(resistance_level, right_rim_date, df_zoom['Date'].iloc[-1],
                   colors='purple', linestyles='--', alpha=0.8, linewidth=2, label='Resistance Level')

    # Highlight Breakout point with a star symbol
    ax_price.scatter(breakout_date, breakout_price, color='green', s=180,
                    zorder=7, marker='*', label='Breakout')

    # --- Annotations for key points ---
    ax_price.annotate('Left Rim', (left_rim_date, left_rim_high), xytext=(-40, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=10)
    ax_price.annotate('Right Rim', (right_rim_date, right_rim_high), xytext=(20, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=10)
    ax_price.annotate('Cup Bottom', (cup_bottom_date, cup_bottom_low), xytext=(0, -40), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=10)
    ax_price.annotate('Handle', (handle_date, handle_low), xytext=(30, -30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='orange'), color='orange', fontsize=10)
    ax_price.annotate('Breakout', (breakout_date, breakout_price), xytext=(20, 40), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=10)

    # Volume bars
    if 'Volume' in df_zoom.columns:
        ax_vol.bar(df_zoom['Date'], df_zoom['Volume'], color='gray', alpha=0.6)
        # Make breakout volume bar thicker and green
        try:
            ax_vol.bar(breakout_date, df.loc[breakout_idx, 'Volume'], color='green', alpha=0.95, width=1.5, label='Breakout Volume')
        except Exception:
            pass

    ax_price.set_title(f"{stock_name} - Cup and Handle Pattern")
    ax_price.set_ylabel('Price')
    ax_vol.set_ylabel('Volume')
    ax_vol.set_xlabel('Date')
    # Remove duplicate legend entries
    handles, labels = ax_price.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_price.legend(unique.values(), unique.keys(), loc='upper left', fontsize=9)
    ax_price.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_double_pattern(df, pattern, stock_name, output_path):
    """Plot Double Top/Bottom pattern."""
    p1_date, p1_price, p1_idx = pattern['P1']
    t_date, t_price, t_idx = pattern['T']
    p2_date, p2_price, p2_idx = pattern['P2']
    breakout_date, breakout_price, breakout_idx = pattern['breakout']
    neckline_level = pattern['neckline_level']

    # Display the entire available dataset window instead of zooming
    df_zoom = df.copy()
    if df_zoom.empty:
        return

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), 
                                          gridspec_kw={'height_ratios': [3, 1]})
    
    ax_price.plot(df_zoom['Date'], df_zoom['Close'], color='navy', label='Close')


    # Mark pattern points and draw W/M shape with fill and dotted neckline
    if pattern['type'] == 'double_top':
        # M shape: extend from left context to P1
        # Find the last price before P1 in the zoomed data
        left_idx = df_zoom[df_zoom['Date'] < p1_date].index
        if len(left_idx) > 0:
            left_date = df_zoom.loc[left_idx[-1], 'Date']
            left_price = df_zoom.loc[left_idx[-1], 'Close']
            x_points = [left_date, p1_date, t_date, p2_date, breakout_date]
            y_points = [left_price, p1_price, t_price, p2_price, breakout_price]
        else:
            x_points = [p1_date, t_date, p2_date, breakout_date]
            y_points = [p1_price, t_price, p2_price, breakout_price]
        ax_price.scatter([p1_date, p2_date], [p1_price, p2_price], color=['red', 'red'], s=120, zorder=5, label='Double Top')
        ax_price.scatter([t_date], [t_price], color='blue', s=100, zorder=5, label='Valley')
        pattern_title = f"{stock_name} - Double Top Pattern"
    else:
        # W shape: extend from left context to P1
        left_idx = df_zoom[df_zoom['Date'] < p1_date].index
        if len(left_idx) > 0:
            left_date = df_zoom.loc[left_idx[-1], 'Date']
            left_price = df_zoom.loc[left_idx[-1], 'Close']
            x_points = [left_date, p1_date, t_date, p2_date, breakout_date]
            y_points = [left_price, p1_price, t_price, p2_price, breakout_price]
        else:
            x_points = [p1_date, t_date, p2_date, breakout_date]
            y_points = [p1_price, t_price, p2_price, breakout_price]
        ax_price.scatter([p1_date, p2_date], [p1_price, p2_price], color=['green', 'green'], s=120, zorder=5, label='Double Bottom')
        ax_price.scatter([t_date], [t_price], color='red', s=100, zorder=5, label='Peak')
        pattern_title = f"{stock_name} - Double Bottom Pattern"

    # Draw neckline (dotted)
    ax_price.hlines(neckline_level, df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1], 
                   colors='teal', linestyles='dotted', linewidth=2, label='Neckline')

    # Mark breakout
    ax_price.scatter(breakout_date, breakout_price, color='orange', s=150, 
                    marker='*', label='Breakout')

    # (Matplotlib) Keep original style; no extra right-leg line enforced

    # Volume bars
    if 'Volume' in df_zoom.columns:
        ax_vol.bar(df_zoom['Date'], df_zoom['Volume'], color='gray', alpha=0.6)
        try:
            ax_vol.bar(breakout_date, df.loc[breakout_idx, 'Volume'], color='orange', alpha=0.9)
        except Exception:
            pass

    ax_price.set_ylabel('Price')
    ax_vol.set_ylabel('Volume')
    ax_vol.set_xlabel('Date')
    ax_price.set_title(pattern_title)
    ax_price.legend()
    ax_price.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# PLOTLY PLOTTING FUNCTIONS (optional backend)
# =============================================================================

def _plotly_price_traces(df_zoom: pd.DataFrame, chart_type: str):
    """Return Plotly traces for the price chart based on chart_type.
    Supported: 'candle', 'line', 'ohlc'. Defaults to 'candle' on invalid input.
    """
    chart_type = (chart_type or 'candle').lower()
    x = df_zoom['Date']
    if chart_type == 'line':
        return [go.Scatter(x=x, y=df_zoom['Close'], mode='lines', name='Close',
                           line=dict(color='#1f77b4', width=2))]
    elif chart_type == 'ohlc':
        return [go.Ohlc(x=x,
                        open=df_zoom['Open'], high=df_zoom['High'], low=df_zoom['Low'], close=df_zoom['Close'],
                        name='OHLC')]
    else:  # 'candle'
        return [go.Candlestick(x=x,
                               open=df_zoom['Open'], high=df_zoom['High'], low=df_zoom['Low'], close=df_zoom['Close'],
                               name='Candles', increasing_line_color='#16a34a', decreasing_line_color='#dc2626')]

def _plotly_volume_trace(df_zoom: pd.DataFrame):
    """Create a volume bar trace color-coded by up/down days."""
    colors = np.where(df_zoom['Close'] >= df_zoom['Open'], '#16a34a', '#dc2626')
    return go.Bar(x=df_zoom['Date'], y=df_zoom['Volume'], name='Volume', marker_color=colors, opacity=0.6)

def plotly_hns_pattern(df, pattern, stock_name, output_path, chart_type='candle'):
    """Plot Head and Shoulders with Plotly, interactive with volume subplot and annotations."""
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError('Plotly is not available in this environment')

    p1_date, p1_high, p1_idx = pattern['P1']
    t1_date, t1_low, t1_idx = pattern['T1']
    p2_date, p2_high, p2_idx = pattern['P2']
    t2_date, t2_low, t2_idx = pattern['T2']
    p3_date, p3_high, p3_idx = pattern['P3']
    breakout_date, breakout_price, breakout_idx = pattern['breakout']

    # Display the entire available dataset window instead of zooming
    df_zoom = df.copy()
    if df_zoom.empty:
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.82, 0.18])

    for tr in _plotly_price_traces(df_zoom, chart_type):
        fig.add_trace(tr, row=1, col=1)

    # Neckline (dotted)
    slope = pattern['neckline_slope']
    intercept = pattern['neckline_intercept']
    def _neck_y(d):
        return slope * d.toordinal() + intercept
    nx = [df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1]]
    ny = [_neck_y(d) for d in nx]
    fig.add_trace(go.Scatter(x=nx, y=ny, mode='lines', name='Neckline',
                             line=dict(color='#0891b2', dash='dash', width=2)), row=1, col=1)

    # Outline legs similar to the reference (green)
    line_color = '#16a34a'
    fill_color = 'rgba(34, 197, 94, 0.18)'

    # Pre-context left leg into Left Shoulder, if we have data before P1 in the zoom window
    pre_idx = df_zoom[df_zoom['Date'] < p1_date].index
    if len(pre_idx) > 0:
        left_date = df_zoom.loc[pre_idx[-1], 'Date']
        left_price = df_zoom.loc[pre_idx[-1], 'Close']
        fig.add_trace(go.Scatter(x=[left_date, p1_date], y=[left_price, p1_high],
                                 mode='lines', name='Left Leg',
                                 line=dict(color=line_color, width=2), showlegend=False), row=1, col=1)

    # Main zigzag path P1 -> T1 -> P2 -> T2 -> P3
    outline_x = [p1_date, t1_date, p2_date, t2_date, p3_date]
    outline_y = [p1_high, t1_low, p2_high, t2_low, p3_high]
    fig.add_trace(go.Scatter(x=outline_x, y=outline_y, mode='lines', name='Pattern',
                             line=dict(color=line_color, width=2)), row=1, col=1)

    # Right leg from P3 to breakout
    fig.add_trace(go.Scatter(x=[p3_date, breakout_date], y=[p3_high, breakout_price],
                             mode='lines', name='Right Leg',
                             line=dict(color=line_color, width=2), showlegend=False), row=1, col=1)

    # Shaded regions between neckline and the edges (two polygons: P1-T1-P2 and P2-T2-P3)
    x_seg1 = [p1_date, t1_date, p2_date]
    neck_seg1 = [_neck_y(d) for d in x_seg1]
    price_seg1 = [p1_high, t1_low, p2_high]
    poly1_x = x_seg1 + list(reversed(x_seg1))
    poly1_y = neck_seg1 + list(reversed(price_seg1))
    fig.add_trace(go.Scatter(x=poly1_x, y=poly1_y, mode='lines', name='Region',
                             line=dict(color=line_color, width=1), fill='toself',
                             fillcolor=fill_color, showlegend=False), row=1, col=1)

    x_seg2 = [p2_date, t2_date, p3_date]
    neck_seg2 = [_neck_y(d) for d in x_seg2]
    price_seg2 = [p2_high, t2_low, p3_high]
    poly2_x = x_seg2 + list(reversed(x_seg2))
    poly2_y = neck_seg2 + list(reversed(price_seg2))
    fig.add_trace(go.Scatter(x=poly2_x, y=poly2_y, mode='lines', name='Region',
                             line=dict(color=line_color, width=1), fill='toself',
                             fillcolor=fill_color, showlegend=False), row=1, col=1)

    # Minimal, clear labels
    markers = [
        (p1_date, p1_high, 'Left Shoulder', line_color),
        (p2_date, p2_high, 'Head', line_color),
        (p3_date, p3_high, 'Right Shoulder', line_color),
        (breakout_date, breakout_price, 'Breakout', '#8b5cf6'),
    ]
    for x, y, label, color in markers:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', name=label,
                                 text=[label], textposition='top center',
                                 marker=dict(color=color, size=10, symbol='circle')), row=1, col=1)

    # Volume
    fig.add_trace(_plotly_volume_trace(df_zoom), row=2, col=1)

    fig.update_layout(
        title=f"{stock_name} - Head and Shoulders",
        template='plotly_white',
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode='x unified',
        showlegend=True,
        height=980,
    )
    rs_visible = (chart_type or 'candle').lower() in ('candle', 'ohlc')
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb', row=1, col=1, rangeslider_visible=rs_visible)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True, config={'responsive': True})

def plotly_ch_pattern(df, pattern, stock_name, output_path, chart_type='candle'):
    """Plot Cup and Handle with Plotly."""
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError('Plotly is not available in this environment')

    left_rim_date, left_rim_high, left_rim_idx = pattern['left_rim']
    cup_bottom_date, cup_bottom_low, cup_bottom_idx = pattern['cup_bottom']
    right_rim_date, right_rim_high, right_rim_idx = pattern['right_rim']
    handle_date, handle_low, handle_low_idx = pattern['handle_low']
    breakout_date, breakout_price, breakout_idx = pattern['breakout']

    # Display the entire available dataset window instead of zooming
    df_zoom = df.copy()
    if df_zoom.empty:
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.82, 0.18])

    for tr in _plotly_price_traces(df_zoom, chart_type):
        fig.add_trace(tr, row=1, col=1)

    resistance_level = max(left_rim_high, right_rim_high)
    fig.add_trace(go.Scatter(x=[df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1]],
                             y=[resistance_level, resistance_level], mode='lines',
                             name='Resistance', line=dict(color='#7c3aed', dash='dash')), row=1, col=1)

    markers = [
        (left_rim_date, left_rim_high, 'Left Rim', '#3b82f6'),
        (cup_bottom_date, cup_bottom_low, 'Cup Bottom', '#ef4444'),
        (right_rim_date, right_rim_high, 'Right Rim', '#3b82f6'),
        (handle_date, handle_low, 'Handle', '#f59e0b'),
        (breakout_date, breakout_price, 'Breakout', '#16a34a'),
    ]
    for x, y, label, color in markers:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', name=label,
                                 text=[label], textposition='top center',
                                 marker=dict(color=color, size=10, symbol='circle')), row=1, col=1)

    fig.add_trace(_plotly_volume_trace(df_zoom), row=2, col=1)

    fig.update_layout(
        title=f"{stock_name} - Cup and Handle",
        template='plotly_white',
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode='x unified',
        showlegend=True,
        height=980,
    )
    rs_visible = (chart_type or 'candle').lower() in ('candle', 'ohlc')
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb', row=1, col=1, rangeslider_visible=rs_visible)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True, config={'responsive': True})

def plotly_double_pattern(df, pattern, stock_name, output_path, chart_type='candle'):
    """Plot Double Top/Bottom with Plotly."""
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError('Plotly is not available in this environment')

    p1_date, p1_price, p1_idx = pattern['P1']
    t_date, t_price, t_idx = pattern['T']
    p2_date, p2_price, p2_idx = pattern['P2']
    breakout_date, breakout_price, breakout_idx = pattern['breakout']
    neckline_level = pattern['neckline_level']

    # Display the entire available dataset window instead of zooming
    df_zoom = df.copy()
    if df_zoom.empty:
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.82, 0.18])

    for tr in _plotly_price_traces(df_zoom, chart_type):
        fig.add_trace(tr, row=1, col=1)

    # Neckline (dotted)
    fig.add_trace(
        go.Scatter(
            x=[df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1]],
            y=[neckline_level, neckline_level],
            mode='lines',
            name='Neckline',
            line=dict(color='#0d9488', dash='dot', width=2)
        ),
        row=1, col=1
    )

    is_dt = pattern['type'] == 'double_top'
    label_main = 'Double Top' if is_dt else 'Double Bottom'
    c_main = '#ef4444' if is_dt else '#16a34a'
    fill_color = 'rgba(239, 68, 68, 0.15)' if is_dt else 'rgba(22, 163, 74, 0.15)'

    # Outline of the M/W (P1 -> T -> P2)
    fig.add_trace(
        go.Scatter(
            x=[p1_date, t_date, p2_date],
            y=[p1_price, neckline_level, p2_price],
            mode='lines',
            name='Pattern',
            line=dict(color=c_main, width=2)
        ),
        row=1, col=1
    )

    # Left-side line logic
    if not is_dt:
        # For Double Bottom: from last swing high before Bottom 1 -> Bottom 1
        try:
            p1_idx = pattern['P1'][2]
            # Find last swing high before p1_idx within df (not just zoom)
            seg = df.iloc[:p1_idx]
            if 'is_swing_high' in seg.columns:
                highs = seg[seg.get('is_swing_high', False)]
            else:
                # Fallback: local maxima by rolling window on High
                win = 5
                max_mask = seg['High'] == seg['High'].rolling(window=win, center=True, min_periods=1).max()
                highs = seg[max_mask]
            if not highs.empty:
                left_idx = highs.index[-1]
                left_date = df.loc[left_idx, 'Date']
                left_price = df.loc[left_idx, 'High']
                fig.add_trace(
                    go.Scatter(
                        x=[left_date, p1_date],
                        y=[left_price, p1_price],
                        mode='lines',
                        name='Left Leg',
                        line=dict(color=c_main, width=2),
                        showlegend=False,
                    ),
                    row=1, col=1
                )
        except Exception:
            pass
    else:
        # For Double Top: from last swing low before Top 1 -> Top 1
        try:
            p1_idx = pattern['P1'][2]
            seg = df.iloc[:p1_idx]
            if 'is_swing_low' in seg.columns:
                lows = seg[seg.get('is_swing_low', False)]
            else:
                win = 5
                min_mask = seg['Low'] == seg['Low'].rolling(window=win, center=True, min_periods=1).min()
                lows = seg[min_mask]
            if not lows.empty:
                left_idx = lows.index[-1]
                left_date = df.loc[left_idx, 'Date']
                left_price = df.loc[left_idx, 'Low']
                fig.add_trace(
                    go.Scatter(
                        x=[left_date, p1_date],
                        y=[left_price, p1_price],
                        mode='lines',
                        name='Left Leg',
                        line=dict(color=c_main, width=2),
                        showlegend=False,
                    ),
                    row=1, col=1
                )
        except Exception:
            pass

    # Add right-side line from P2 to breakout (to mirror reference chart)
    fig.add_trace(
        go.Scatter(
            x=[p2_date, breakout_date],
            y=[p2_price, breakout_price],
            mode='lines',
            name='Right Leg',
            line=dict(color=c_main, width=2),
            showlegend=False,
        ),
        row=1, col=1
    )

    # Shaded triangles to emphasize the pattern legs relative to neckline
    if is_dt:
        # Shade above neckline towards tops
        tri1_x = [p1_date, t_date, p1_date]
        tri1_y = [p1_price, neckline_level, neckline_level]
        tri2_x = [p2_date, t_date, p2_date]
        tri2_y = [p2_price, neckline_level, neckline_level]
    else:
        # Shade below neckline towards bottoms (W)
        tri1_x = [p1_date, t_date, p1_date]
        tri1_y = [p1_price, neckline_level, neckline_level]
        tri2_x = [p2_date, t_date, p2_date]
        tri2_y = [p2_price, neckline_level, neckline_level]

    fig.add_trace(go.Scatter(x=tri1_x, y=tri1_y, name='Pattern Zone',
                             mode='lines', line=dict(color=c_main, width=1),
                             fill='toself', fillcolor=fill_color, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=tri2_x, y=tri2_y, name='Pattern Zone',
                             mode='lines', line=dict(color=c_main, width=1),
                             fill='toself', fillcolor=fill_color, showlegend=False), row=1, col=1)

    # Markers and labels for points
    lbl1 = 'Top 1' if is_dt else 'Bottom 1'
    lbl2 = 'Top 2' if is_dt else 'Bottom 2'
    mid_lbl = 'Trough' if is_dt else 'Peak'
    mid_color = '#3b82f6'

    marks = [
        (p1_date, p1_price, lbl1, c_main),
        (t_date, t_price, mid_lbl, mid_color),
        (p2_date, p2_price, lbl2, c_main),
        (breakout_date, breakout_price, 'Breakout', '#f59e0b'),
    ]
    for x, y, label, color in marks:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', name=label,
                                 text=[label], textposition='top center',
                                 marker=dict(color=color, size=10, symbol='circle')), row=1, col=1)

    # Measured move target from neckline height
    if is_dt:
        height = abs(max(p1_price, p2_price) - neckline_level)
        target = neckline_level - height
    else:
        height = abs(neckline_level - min(p1_price, p2_price))
        target = neckline_level + height

    # Vertical dotted line from breakout to target + target annotation
    fig.add_shape(type='line', xref='x', yref='y',
                  x0=breakout_date, x1=breakout_date, y0=breakout_price, y1=target,
                  line=dict(color='#4b5563', width=1, dash='dot'), row=1, col=1)
    fig.add_annotation(x=breakout_date, y=target, text='Target', showarrow=True,
                       arrowhead=2, yshift=10, bgcolor='#e5e7eb', bordercolor='#9ca3af',
                       font=dict(color='#111827', size=11), row=1, col=1)

    # Volume
    fig.add_trace(_plotly_volume_trace(df_zoom), row=2, col=1)

    fig.update_layout(
        title=f"{stock_name} - {label_main}",
        template='plotly_white',
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode='x unified',
        showlegend=True,
        height=980,
    )
    rs_visible = (chart_type or 'candle').lower() in ('candle', 'ohlc')
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb', row=1, col=1, rangeslider_visible=rs_visible)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True, config={'responsive': True})
# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_symbol(symbol, timeframes, patterns, mode, swing_method, output_dir, 
                  require_preceding_trend, min_patterns, max_patterns_per_timeframe,
                  organize_by_date=False, charts_subdir='charts', reports_subdir='reports',
                  use_plotly=False, chart_type='candle',
                  start_date: str | None = None, end_date: str | None = None,
                  keep_best_only: bool = False,
                  data_source: str = 'live',
                  stock_data_dir: str | None = None):
    """Process a single symbol for pattern detection."""
    print(f"\nProcessing {symbol}...")
    
    using_date_range = bool(start_date and end_date)
    
    # Calculate date range
    if using_date_range:
        start_date_dl = start_date
        end_date_dl = end_date
    else:
        # Use the largest timeframe + buffer when no explicit date range
        max_days = max(TIMEFRAMES[tf] for tf in timeframes)
        start_date_dl = (datetime.now() - timedelta(days=max_days + 365)).strftime('%Y-%m-%d')
        end_date_dl = datetime.now().strftime('%Y-%m-%d')
    
    try:
        if (data_source or 'live').lower() == 'past':
            # Load from CSV directory
            ddir = stock_data_dir or os.environ.get('STOCK_DATA_DIR') or os.path.join(os.path.dirname(__file__), 'StockData')
            df = load_data_from_csv(symbol, ddir)
        else:
            df = load_data(symbol, start_date_dl, end_date_dl)
    except Exception as e:
        print(f"Failed to load data for {symbol}: {e}")
        return []

    if df is None or df.empty:
        print(f"No data for {symbol}, skipping.")
        return []

    df = df.reset_index(drop=True)
    
    # Get configurations
    hns_config = HNS_CONFIG[mode]
    ch_config = CH_CONFIG[mode]
    dt_config = DT_CONFIG[mode]
    
    all_results = []
    
    # Determine iteration plan
    if using_date_range:
        # Single pass: use the explicit window
        iter_plan = [(f"range_{start_date.replace('-','')}_to_{end_date.replace('-','')}", df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].copy())]
    else:
        iter_plan = []
        for timeframe in timeframes:
            days = TIMEFRAMES[timeframe]
            df_slice = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=days)].copy()
            iter_plan.append((timeframe, df_slice))

    for timeframe_label, df_slice in iter_plan:
        if df_slice.empty:
            if using_date_range:
                print(f"{symbol} - {start_date} to {end_date}: No data in this period")
            else:
                print(f"{symbol} - {timeframe_label}: No data for this timeframe")
            continue

        df_slice = df_slice.reset_index(drop=True)

        # Generate swing points
        nbars = compute_dynamic_nbars(df_slice, base=GLOBAL_CONFIG['BASE_NBARS'])
        df_slice = generate_swing_flags(df_slice, method=swing_method, N_bars=nbars)

    # Pattern detection
        timeframe_patterns = []
        pattern_counts = {'head_and_shoulders': 0, 'cup_and_handle': 0, 'double_top': 0, 'double_bottom': 0}

        # Head and Shoulders
        if 'head_and_shoulders' in patterns:
            hns_patterns = detect_head_and_shoulders(df_slice, hns_config, require_preceding_trend)
            try:
                from validator.validate_hns import validate_hns
                from validator.explain_patterns import explain_pattern, compute_weighted_score
            except ImportError:
                validate_hns = None
                explain_pattern = None
                compute_weighted_score = None
            for pattern in hns_patterns:
                if pattern_counts['head_and_shoulders'] >= max_patterns_per_timeframe:
                    break
                # Validate pattern
                if validate_hns:
                    validation = validate_hns(df_slice, pattern)
                    pattern['validation'] = validation
                # Attach rule-of-thumb explanation and measured target
                if 'explanation' not in pattern and 'P1' in pattern and 'P2' in pattern and 'P3' in pattern:
                    try:
                        if explain_pattern:
                            pattern['explanation'] = explain_pattern(df_slice, pattern)
                    except Exception:
                        pass
                # weighted score
                try:
                    if compute_weighted_score and 'explanation' in pattern:
                        ws, wmax = compute_weighted_score(pattern['explanation'])
                        pattern['weighted_score'] = float(ws)
                        pattern['weighted_max'] = float(wmax)
                except Exception:
                    pass
                pattern['symbol'] = symbol
                pattern['timeframe'] = timeframe_label
                timeframe_patterns.append(pattern)
                pattern_counts['head_and_shoulders'] += 1

        # Cup and Handle
        if 'cup_and_handle' in patterns:
            ch_patterns = detect_cup_and_handle(df_slice, ch_config, require_preceding_trend)
            # Import validator
            try:
                from validator.validate_cup_handle import validate_cup_handle
                from validator.explain_patterns import explain_pattern, compute_weighted_score
            except ImportError:
                validate_cup_handle = None
                explain_pattern = None
                compute_weighted_score = None
            for pattern in ch_patterns:
                if pattern_counts['cup_and_handle'] >= max_patterns_per_timeframe:
                    break
                # Validate pattern
                if validate_cup_handle:
                    validation = validate_cup_handle(df_slice, pattern)
                    pattern['validation'] = validation
                # Attach rule-of-thumb explanation and measured target
                try:
                    if explain_pattern:
                        pattern['explanation'] = explain_pattern(df_slice, pattern)
                except Exception:
                    pass
                # weighted score
                try:
                    if compute_weighted_score and 'explanation' in pattern:
                        ws, wmax = compute_weighted_score(pattern['explanation'])
                        pattern['weighted_score'] = float(ws)
                        pattern['weighted_max'] = float(wmax)
                except Exception:
                    pass
                pattern['symbol'] = symbol
                pattern['timeframe'] = timeframe_label
                timeframe_patterns.append(pattern)
                pattern_counts['cup_and_handle'] += 1

        # Double patterns
        if 'double_top' in patterns or 'double_bottom' in patterns:
            double_type = 'both'
            if 'double_top' in patterns and 'double_bottom' not in patterns:
                double_type = 'double_top'
            elif 'double_bottom' in patterns and 'double_top' not in patterns:
                double_type = 'double_bottom'

            double_patterns = detect_double_patterns(df_slice, dt_config, double_type, require_preceding_trend)
            for pattern in double_patterns:
                pattern_type = pattern['type']
                if pattern_counts[pattern_type] >= max_patterns_per_timeframe:
                    continue

                # Validate double pattern
                try:
                    from validator.validate_double_patterns import validate_double_pattern
                    from validator.explain_patterns import explain_pattern, compute_weighted_score
                except ImportError:
                    validate_double_pattern = None
                    explain_pattern = None
                    compute_weighted_score = None

                # Add validation
                if validate_double_pattern:
                    validation = validate_double_pattern(df_slice, pattern)
                    pattern['validation'] = validation
                # Attach rule-of-thumb explanation and measured target
                try:
                    if explain_pattern:
                        pattern['explanation'] = explain_pattern(df_slice, pattern)
                except Exception:
                    pass
                # weighted score
                try:
                    if compute_weighted_score and 'explanation' in pattern:
                        ws, wmax = compute_weighted_score(pattern['explanation'])
                        pattern['weighted_score'] = float(ws)
                        pattern['weighted_max'] = float(wmax)
                except Exception:
                    pass

                pattern['symbol'] = symbol
                pattern['timeframe'] = timeframe_label
                timeframe_patterns.append(pattern)
                pattern_counts[pattern_type] += 1

        # Optionally keep only the best pattern per timeframe to avoid overlap
        if keep_best_only and timeframe_patterns:
            def _priority_order(ptype: str) -> int:
                # Prefer H&S over Double patterns over CH
                if ptype == 'head_and_shoulders':
                    return 0
                if ptype in ('double_top', 'double_bottom'):
                    return 1
                if ptype == 'cup_and_handle':
                    return 2
                return 3

            def _score_key(p: dict) -> tuple:
                ws = float(p.get('weighted_score', 0.0))
                # normalize by max if available
                wmax = float(p.get('weighted_max', 1.0)) or 1.0
                norm = ws / wmax
                return (norm, -_priority_order(p.get('type', 'zzz')) * 1.0)

            best = sorted(timeframe_patterns, key=_score_key, reverse=True)[0]
            # adjust counts to reflect single kept pattern
            pattern_counts = {'head_and_shoulders': 0, 'cup_and_handle': 0, 'double_top': 0, 'double_bottom': 0}
            pattern_counts[best['type']] = 1
            timeframe_patterns = [best]

        # Create output directories and save charts
        if timeframe_patterns:
            for pattern in timeframe_patterns:
                pattern_type = pattern['type']

                # Create organized folder structure
                base_output = Path(output_dir)

                # Add date organization if requested
                if organize_by_date:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    charts_base = base_output / date_str / charts_subdir
                else:
                    charts_base = base_output / charts_subdir

                # Pattern type abbreviations for cleaner folder names
                pattern_abbrev = {
                    'head_and_shoulders': 'HNS',
                    'cup_and_handle': 'CH',
                    'double_top': 'DT',
                    'double_bottom': 'DB'
                }

                # Create nested structure:
                # charts/symbol/timeframe/pattern_type/<backend>/[chart_type]
                pattern_root = charts_base / symbol / timeframe_label / pattern_abbrev[pattern_type]
                backend_dir = 'plotly' if use_plotly else 'matplotlib'
                pattern_dir = pattern_root / backend_dir
                if use_plotly:
                    pattern_dir = pattern_dir / (chart_type.lower() if chart_type else 'candle')
                pattern_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename
                pattern_count = pattern_counts[pattern_type]
                if use_plotly:
                    filename = f"{symbol}_{pattern_type.upper()}_{timeframe_label}_{pattern_count}.html"
                else:
                    filename = f"{symbol}_{pattern_type.upper()}_{timeframe_label}_{pattern_count}.png"
                output_path = pattern_dir / filename

                # Plot pattern
                try:
                    if use_plotly:
                        if pattern_type == 'head_and_shoulders':
                            plotly_hns_pattern(df_slice, pattern, symbol, str(output_path), chart_type=chart_type)
                        elif pattern_type == 'cup_and_handle':
                            plotly_ch_pattern(df_slice, pattern, symbol, str(output_path), chart_type=chart_type)
                        elif pattern_type in ['double_top', 'double_bottom']:
                            plotly_double_pattern(df_slice, pattern, symbol, str(output_path), chart_type=chart_type)
                    else:
                        if pattern_type == 'head_and_shoulders':
                            plot_hns_pattern(df_slice, pattern, symbol, str(output_path))
                        elif pattern_type == 'cup_and_handle':
                            plot_ch_pattern(df_slice, pattern, symbol, str(output_path))
                        elif pattern_type in ['double_top', 'double_bottom']:
                            plot_double_pattern(df_slice, pattern, symbol, str(output_path))

                    pattern['image_path'] = str(output_path)
                except Exception as e:
                    print(f"Failed to plot {pattern_type} for {symbol} {timeframe_label}: {e}")
                    pattern['image_path'] = None

        # Report results
        total_patterns = sum(pattern_counts.values())
        if total_patterns > 0:
            pattern_summary = ", ".join([f"{k}: {v}" for k, v in pattern_counts.items() if v > 0])
            print(f"{symbol} - {timeframe_label}: {total_patterns} patterns detected ({pattern_summary})")
        else:
            print(f"{symbol} - {timeframe_label}: No patterns detected")

        all_results.extend(timeframe_patterns)
    
    # Log Cup and Handle scores if any (only if at least one has a score)
    try:
        from log.log_cup_handle_scores import log_cup_handle_scores
        ch_patterns_with_score = [p for p in all_results if p.get('type') == 'cup_and_handle' and 'validation' in p]
        if ch_patterns_with_score:
            ch_score_log = Path(output_dir) / 'reports' / 'score' / 'ch' / f'{symbol}_cup_handle_scores.csv'
            log_cup_handle_scores(ch_patterns_with_score, ch_score_log)
    except Exception:
        pass
    # Log HNS scores if any (only if at least one has a score)
    try:
        from log.log_hns_scores import log_hns_scores
        hns_patterns_with_score = [p for p in all_results if p.get('type') == 'head_and_shoulders' and 'validation' in p]
        if hns_patterns_with_score:
            hns_score_log = Path(output_dir) / 'reports' / 'score' / 'hns' / f'{symbol}_hns_scores.csv'
            log_hns_scores(hns_patterns_with_score, hns_score_log)
    except Exception:
        pass
    # Log Double Pattern scores if any (only if at least one has a score)
    try:
        from log.log_double_pattern_scores import log_double_pattern_scores
        double_patterns_with_score = [p for p in all_results if p.get('type') in ['double_top', 'double_bottom'] and 'validation' in p]
        if double_patterns_with_score:
            double_score_log = Path(output_dir) / 'reports' / 'score' / 'double' / f'{symbol}_double_pattern_scores.csv'
            log_double_pattern_scores(double_patterns_with_score, double_score_log)
    except Exception:
        pass
    return all_results


def cleanup_empty_directories(root_path):
    """Remove empty directories recursively to avoid creating empty folder structures."""
    root_path = Path(root_path)
    
    if not root_path.exists():
        return
    
    # Walk through directories bottom-up to remove empty ones
    for dir_path in sorted(root_path.rglob('*'), key=lambda x: len(x.parts), reverse=True):
        if dir_path.is_dir():
            try:
                # Try to remove if empty
                dir_path.rmdir()
                print(f"Removed empty directory: {dir_path}")
            except OSError:
                # Directory not empty, which is fine
                pass

def save_results(results, output_file, mode):
    """Save results to CSV."""
    if not results:
        print(f"No patterns detected; no summary CSV created for mode={mode}")
        return
    
    # Convert results to DataFrame format
    summary_records = []
    
    for pattern in results:
        record = {
            'pattern_type': pattern['type'],
            'symbol': pattern['symbol'],
            'timeframe': pattern['timeframe'],
            'mode': mode,
        }
        
        if pattern['type'] == 'head_and_shoulders':
            p1_date, p1_high, _ = pattern['P1']
            t1_date, t1_low, _ = pattern['T1']
            p2_date, p2_high, _ = pattern['P2']
            t2_date, t2_low, _ = pattern['T2']
            p3_date, p3_high, _ = pattern['P3']
            breakout_date, breakout_price, _ = pattern['breakout']
            
            record.update({
                'P1_date': p1_date.strftime('%Y-%m-%d'),
                'P1_price': p1_high,
                'T1_date': t1_date.strftime('%Y-%m-%d'),
                'T1_price': t1_low,
                'P2_date': p2_date.strftime('%Y-%m-%d'),
                'P2_price': p2_high,
                'T2_date': t2_date.strftime('%Y-%m-%d'),
                'T2_price': t2_low,
                'P3_date': p3_date.strftime('%Y-%m-%d'),
                'P3_price': p3_high,
                'breakout_date': breakout_date.strftime('%Y-%m-%d'),
                'breakout_price': breakout_price,
                'duration': pattern['duration'],
                'image_path': pattern.get('image_path')
            })
        
        elif pattern['type'] == 'cup_and_handle':
            left_rim_date, left_rim_high, _ = pattern['left_rim']
            cup_bottom_date, cup_bottom_low, _ = pattern['cup_bottom']
            right_rim_date, right_rim_high, _ = pattern['right_rim']
            handle_date, handle_low, _ = pattern['handle_low']
            breakout_date, breakout_price, _ = pattern['breakout']
            
            record.update({
                'left_rim_date': left_rim_date.strftime('%Y-%m-%d'),
                'left_rim_price': left_rim_high,
                'cup_bottom_date': cup_bottom_date.strftime('%Y-%m-%d'),
                'cup_bottom_price': cup_bottom_low,
                'right_rim_date': right_rim_date.strftime('%Y-%m-%d'),
                'right_rim_price': right_rim_high,
                'handle_date': handle_date.strftime('%Y-%m-%d'),
                'handle_price': handle_low,
                'breakout_date': breakout_date.strftime('%Y-%m-%d'),
                'breakout_price': breakout_price,
                'cup_duration': pattern['cup_duration'],
                'cup_depth': pattern['cup_depth'],
                'image_path': pattern.get('image_path')
            })
        
        elif pattern['type'] in ['double_top', 'double_bottom']:
            p1_date, p1_price, _ = pattern['P1']
            t_date, t_price, _ = pattern['T']
            p2_date, p2_price, _ = pattern['P2']
            breakout_date, breakout_price, _ = pattern['breakout']
            
            record.update({
                'P1_date': p1_date.strftime('%Y-%m-%d'),
                'P1_price': p1_price,
                'T_date': t_date.strftime('%Y-%m-%d'),
                'T_price': t_price,
                'P2_date': p2_date.strftime('%Y-%m-%d'),
                'P2_price': p2_price,
                'breakout_date': breakout_date.strftime('%Y-%m-%d'),
                'breakout_price': breakout_price,
                'neckline_level': pattern['neckline_level'],
                'duration': pattern['duration'],
                'image_path': pattern.get('image_path')
            })
        
        summary_records.append(record)
    
    # Save to CSV
    df = pd.DataFrame(summary_records)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(summary_records)} patterns to {output_file}")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Stock Pattern Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Random 5-10 symbols, all patterns, multiple timeframes, strict mode
  python detect_all_patterns.py
  
  # Custom random selection with 8 symbols  
  python detect_all_patterns.py --symbols random --random-count 8
  
  # Specific symbol set with lenient mode
  python detect_all_patterns.py --symbols us_tech --mode lenient
  
  # Custom symbols with specific patterns and timeframes
  python detect_all_patterns.py --symbols AAPL,GOOGL,TSLA --patterns head_and_shoulders --timeframes 1y,2y
  
  # Organized output structure with date
  python detect_all_patterns.py --symbols indian_popular --patterns cup_and_handle --organize-by-date
  
  # Custom output structure
  python detect_all_patterns.py --symbols AAPL,TSLA --output-dir ./my_analysis --charts-subdir images --reports-subdir data
  
  # Quick scan with limited patterns per timeframe
  python detect_all_patterns.py --symbols us_tech --patterns all --max-patterns-per-timeframe 2 --min-patterns 5
        """
    )
    
    # Symbol selection
    parser.add_argument('--symbols', type=str, default='random',
                       help='Symbols to analyze. Options: "random" (5-10 random symbols), predefined lists (us_tech, us_large, indian_popular, indian_banking, crypto) or comma-separated list (AAPL,GOOGL,TSLA)')
    
    parser.add_argument('--random-count', type=int, default=None,
                       help='Number of random symbols to select (5-10 if not specified, only used with --symbols random)')
    
    parser.add_argument('--custom-symbols', type=str, 
                       help='File path to custom symbol list (one symbol per line)')
    
    # Pattern selection
    parser.add_argument('--patterns', type=str, default='all',
                       help='Patterns to detect. Options: "all" or comma-separated list (head_and_shoulders,cup_and_handle,double_top,double_bottom)')
    
    # Timeframe selection
    parser.add_argument('--timeframes', type=str, default='6m,1y,2y,3y,5y',
                       help='Timeframes to analyze (comma-separated). Options: 1d,1w,2w,1m,2m,3m,6m,1y,2y,3y,5y')

    # Custom date range (overrides timeframes when both provided)
    parser.add_argument('--start-date', type=str, help='Custom start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='Custom end date YYYY-MM-DD')
    
    # Detection mode
    parser.add_argument('--mode', type=str, default='strict',
                       choices=['strict', 'lenient', 'both'],
                       help='Detection sensitivity mode')
    
    # Advanced options
    parser.add_argument('--swing-method', type=str, default='rolling',
                       choices=['rolling', 'zigzag', 'fractal'],
                       help='Method for detecting swing points')
    
    parser.add_argument('--require-preceding-trend', action='store_true', default=True,
                       help='Require preceding trend before pattern formation')
    
    parser.add_argument('--no-preceding-trend', dest='require_preceding_trend', action='store_false',
                       help='Disable preceding trend requirement')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Base output directory for all results')
    
    parser.add_argument('--output-csv', type=str, 
                       help='Custom CSV output filename (default: auto-generated)')
    
    parser.add_argument('--organize-by-date', action='store_true',
                       help='Organize outputs by date (YYYY-MM-DD)')
    
    parser.add_argument('--charts-subdir', type=str, default='charts',
                       help='Subdirectory name for chart outputs (default: charts)')
    
    parser.add_argument('--reports-subdir', type=str, default='reports', 
                       help='Subdirectory name for CSV reports (default: reports)')
    
    # Performance options
    parser.add_argument('--min-patterns', type=int, default=0,
                       help='Minimum total patterns required to save results')
    
    parser.add_argument('--max-patterns-per-timeframe', type=int, default=10,
                       help='Maximum patterns per timeframe per symbol')
    
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between symbol processing (seconds)')
    
    # Utility options
    parser.add_argument('--list-symbols', action='store_true',
                       help='List available predefined symbol sets and exit')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually running')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    # Plotting backend options
    parser.add_argument('--plotly', action='store_true',
                        help='Use Plotly for interactive charts (outputs .html files)')
    parser.add_argument('--chart-type', type=str, default='candle',
                        choices=['candle', 'line', 'ohlc'],
                        help='Chart style when using Plotly (default: candle)')

    args = parser.parse_args()
    
    # Handle special options
    if args.list_symbols:
        print("Available symbol options:")
        print("  random: Randomly selects 5-10 symbols from all categories")
        print("\nPredefined symbol sets:")
        for name, symbols in DEFAULT_SYMBOLS.items():
            print(f"  {name}: {len(symbols)} symbols")
            if args.verbose:
                print(f"    {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        return
    
    # Parse symbols
    if args.custom_symbols:
        with open(args.custom_symbols, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.symbols == 'random':
        symbols = get_random_symbols(args.random_count)
        print(f"Randomly selected {len(symbols)} symbols: {', '.join(symbols)}")
    elif args.symbols in DEFAULT_SYMBOLS:
        symbols = DEFAULT_SYMBOLS[args.symbols]
    else:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Parse patterns
    if args.patterns == 'all':
        patterns = ['head_and_shoulders', 'cup_and_handle', 'double_top', 'double_bottom']
    else:
        # Parse comma-separated pattern list
        patterns = [p.strip() for p in args.patterns.split(',')]
        # Validate patterns
        valid_patterns = ['head_and_shoulders', 'cup_and_handle', 'double_top', 'double_bottom']
        for pattern in patterns:
            if pattern not in valid_patterns:
                print(f"Error: Invalid pattern '{pattern}'. Available: {', '.join(valid_patterns)}")
                return
    
    # Parse timeframes or date range
    start_date_arg = args.start_date
    end_date_arg = args.end_date
    using_date_range = bool(start_date_arg and end_date_arg)
    if using_date_range:
        # Quick validation
        try:
            _sd = datetime.strptime(start_date_arg, '%Y-%m-%d')
            _ed = datetime.strptime(end_date_arg, '%Y-%m-%d')
            if _sd >= _ed:
                print('Error: --start-date must be earlier than --end-date')
                return
        except Exception:
            print('Error: Dates must be in YYYY-MM-DD format')
            return
        timeframes = ['custom']
    else:
        timeframes = [tf.strip() for tf in args.timeframes.split(',')]
        for tf in timeframes:
            if tf not in TIMEFRAMES:
                print(f"Error: Invalid timeframe '{tf}'. Available: {', '.join(TIMEFRAMES.keys())}")
                return
    
    # Parse modes
    if args.mode == 'both':
        modes = ['strict', 'lenient']
    else:
        modes = [args.mode]
    
    if args.dry_run:
        print("DRY RUN - Would process:")
        print(f"  Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
        print(f"  Patterns: {', '.join(patterns)}")
        if using_date_range:
            print(f"  Date range: {start_date_arg} to {end_date_arg}")
        else:
            print(f"  Timeframes: {', '.join(timeframes)}")
        print(f"  Modes: {', '.join(modes)}")
        print(f"  Output structure:")
        if args.organize_by_date:
            print(f"    Base: {args.output_dir}/YYYY-MM-DD/")
        else:
            print(f"    Base: {args.output_dir}/")
        print(f"    Charts: {args.charts_subdir}/symbol/timeframe/pattern_type/")
        print(f"    Reports: {args.reports_subdir}/")
        return
    
    # Warn if Plotly requested but unavailable
    if getattr(args, 'plotly', False) and not _PLOTLY_AVAILABLE:
        print("Plotly requested via --plotly but package not installed. Falling back to matplotlib.")

    # Main processing loop
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running mode: {mode}")
        print(f"{'='*60}")

        all_results = []

        for symbol in symbols:
            try:
                results = process_symbol(
                    symbol=symbol,
                    timeframes=timeframes,
                    patterns=patterns,
                    mode=mode,
                    swing_method=args.swing_method,
                    output_dir=args.output_dir,
                    require_preceding_trend=args.require_preceding_trend,
                    min_patterns=args.min_patterns,
                    max_patterns_per_timeframe=args.max_patterns_per_timeframe,
                    organize_by_date=args.organize_by_date,
                    charts_subdir=args.charts_subdir,
                    reports_subdir=args.reports_subdir,
                    use_plotly=(args.plotly and _PLOTLY_AVAILABLE),
                    chart_type=args.chart_type,
                    start_date=start_date_arg if using_date_range else None,
                    end_date=end_date_arg if using_date_range else None,
                )
                all_results.extend(results)

                if args.delay > 0:
                    time.sleep(args.delay)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        # Save results
        if len(all_results) >= args.min_patterns:
            # Create organized output structure for CSV reports
            base_output = Path(args.output_dir)
            
            if args.organize_by_date:
                date_str = datetime.now().strftime('%Y-%m-%d')
                reports_dir = base_output / date_str / args.reports_subdir
            else:
                reports_dir = base_output / args.reports_subdir
            
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            if args.output_csv:
                # Use custom filename with proper directory
                custom_name = Path(args.output_csv).name
                output_file = reports_dir / custom_name.replace('.csv', f'_{mode}.csv')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = reports_dir / f'pattern_detection_all_{mode}_{timestamp}.csv'
            
            save_results(all_results, str(output_file), mode)
        else:
            print(f"Insufficient patterns found ({len(all_results)} < {args.min_patterns}). No CSV saved for mode {mode}.")
    
    # Clean up any empty directories that may have been created
    print("\nCleaning up empty directories...")
    cleanup_empty_directories(args.output_dir)

if __name__ == "__main__":
    main()
