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
PEAK_SIMILARITY_TOL = 0.08          # max relative difference between peaks/troughs (8%)
MIN_PROMINENCE_PCT = 0.05           # minimum prominence of peaks/troughs (5%)
MAX_SPACING_DAYS = 120              # maximum spacing between peaks/troughs
MIN_SPACING_DAYS = 15               # minimum spacing between peaks/troughs
VOLUME_DECLINE_PCT = 0.85           # volume should decline on second peak/trough
VOLUME_SPIKE_MULT = 1.50           # breakout volume must exceed avg pattern volume * this multiplier
NECKLINE_TOLERANCE = 0.02           # tolerance for neckline level (2%)
BASE_NBARS = 20                     # base lookback for swing detection
MIN_NBARS = 8
MAX_NBARS = 60

# Feature flags (enable/disable expensive checks)
USE_ZIGZAG = True                   # use ZigZag swing detection instead of rolling highs/lows
ZIGZAG_PCT = 0.03                   # 3% reversal threshold for ZigZag
USE_FRACTAL = False                 # use Williams Fractals for swing detection
REQUIRE_VOLUME_DIVERGENCE = True    # require bearish volume divergence on second peak
REQUIRE_VOLUME_CONFIRMATION = True  # require volume confirmation on breakout
REQUIRE_PRECEDING_TREND = True      # require preceding trend before pattern formation
MIN_TREND_PCT = 0.15                # minimum trend percentage before pattern

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

def check_preceding_uptrend(data, pattern_start_index, lookback_period=90, min_rise_percent=MIN_TREND_PCT):
    """Check if there was a significant uptrend before the pattern started."""
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

    start_price_raw = lookback_data['Close'].iloc[0]
    try:
        start_price = float(start_price_raw.item()) if hasattr(start_price_raw, 'item') else float(start_price_raw)
    except Exception:
        start_price = float(start_price_raw)

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

def check_preceding_downtrend(data, pattern_start_index, lookback_period=90, min_fall_percent=MIN_TREND_PCT):
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

def detect_double_tops_and_bottoms(data, min_days=None, max_days=None):
    """Detect double tops and double bottoms using swing flags."""
    results = []
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    
    for i in range(len(swing_points_df) - 2):
        idx1, idx_mid, idx2 = swing_points_df['index'][i:i+3]
        
        # Double Top: High, Low, High
        if (data.get('is_swing_high', False)[idx1] and 
            data.get('is_swing_low', False)[idx_mid] and 
            data.get('is_swing_high', False)[idx2]):
            
            p1_high, p1_date = data.loc[idx1, ['High', 'Date']]
            t_low, t_date = data.loc[idx_mid, ['Low', 'Date']]
            p2_high, p2_date = data.loc[idx2, ['High', 'Date']]

            # Duration checks
            try:
                total_duration = (p2_date - p1_date).days
                spacing = total_duration
            except Exception:
                continue

            if spacing < MIN_SPACING_DAYS or spacing > MAX_SPACING_DAYS:
                continue

            if min_days is not None and total_duration < min_days:
                continue
            if max_days is not None and total_duration > max_days:
                continue

            # --- Rule 1: Require preceding uptrend for double top ---
            if REQUIRE_PRECEDING_TREND and not check_preceding_uptrend(data, idx1):
                continue

            # --- Rule 2: Peak similarity ---
            if abs(p1_high - p2_high) / max(p1_high, p2_high) > PEAK_SIMILARITY_TOL:
                continue

            # --- Rule 3: Prominence check ---
            try:
                prominence1 = (p1_high - t_low) / t_low
                prominence2 = (p2_high - t_low) / t_low
                if prominence1 < MIN_PROMINENCE_PCT or prominence2 < MIN_PROMINENCE_PCT:
                    continue
            except Exception:
                continue

            # --- Rule 4: Volume divergence (optional) ---
            if REQUIRE_VOLUME_DIVERGENCE and 'Volume' in data.columns:
                try:
                    vol1_window = max(1, min(10, (idx_mid - idx1) // 2))
                    vol2_window = max(1, min(10, (idx2 - idx_mid) // 2))
                    vol1 = data.iloc[idx1-vol1_window:idx1+vol1_window+1]['Volume'].mean()
                    vol2 = data.iloc[idx2-vol2_window:idx2+vol2_window+1]['Volume'].mean()
                    if vol2 >= vol1 * VOLUME_DECLINE_PCT:
                        continue
                except Exception:
                    pass

            # --- Rule 5: Look for neckline breakout ---
            neckline_level = t_low
            breakout_idx = -1
            
            for j in range(idx2 + 1, min(len(data), idx2 + 90)):  # look within 90 days
                if data['Close'].iloc[j] < neckline_level * (1 - NECKLINE_TOLERANCE):
                    # Check for volume spike or sustained break
                    if REQUIRE_VOLUME_CONFIRMATION and 'Volume' in data.columns:
                        try:
                            avg_vol = float(data.iloc[idx1:idx2+1]['Volume'].mean())
                            vol_j = float(data['Volume'].iloc[j])
                            if vol_j > avg_vol * VOLUME_SPIKE_MULT:
                                breakout_idx = j
                                break
                        except Exception:
                            pass
                    else:
                        # Check for sustained break
                        sustained = 0
                        look_ahead = min(len(data) - 1, j + 5)
                        for k in range(j, look_ahead + 1):
                            if data['Close'].iloc[k] < neckline_level:
                                sustained += 1
                        if sustained >= 2:
                            breakout_idx = j
                            break

            if breakout_idx == -1:
                continue

            results.append({
                'type': 'double_top',
                'P1': (p1_date, p1_high, idx1),
                'T': (t_date, t_low, idx_mid),
                'P2': (p2_date, p2_high, idx2),
                'breakout': (data['Date'].iloc[breakout_idx], data['Close'].iloc[breakout_idx], breakout_idx),
                'neckline_level': neckline_level,
                'duration': total_duration
            })

        # Double Bottom: Low, High, Low
        if (data.get('is_swing_low', False)[idx1] and 
            data.get('is_swing_high', False)[idx_mid] and 
            data.get('is_swing_low', False)[idx2]):
            
            p1_low, p1_date = data.loc[idx1, ['Low', 'Date']]
            t_high, t_date = data.loc[idx_mid, ['High', 'Date']]
            p2_low, p2_date = data.loc[idx2, ['Low', 'Date']]

            try:
                total_duration = (p2_date - p1_date).days
                spacing = total_duration
            except Exception:
                continue

            if spacing < MIN_SPACING_DAYS or spacing > MAX_SPACING_DAYS:
                continue

            if min_days is not None and total_duration < min_days:
                continue
            if max_days is not None and total_duration > max_days:
                continue

            # --- Rule 1: Require preceding downtrend for double bottom ---
            if REQUIRE_PRECEDING_TREND and not check_preceding_downtrend(data, idx1):
                continue

            # --- Rule 2: Trough similarity ---
            if abs(p1_low - p2_low) / max(p1_low, p2_low) > PEAK_SIMILARITY_TOL:
                continue

            # --- Rule 3: Prominence check ---
            try:
                prominence1 = (t_high - p1_low) / t_high
                prominence2 = (t_high - p2_low) / t_high
                if prominence1 < MIN_PROMINENCE_PCT or prominence2 < MIN_PROMINENCE_PCT:
                    continue
            except Exception:
                continue

            # --- Rule 4: Volume divergence (optional) ---
            if REQUIRE_VOLUME_DIVERGENCE and 'Volume' in data.columns:
                try:
                    vol1_window = max(1, min(10, (idx_mid - idx1) // 2))
                    vol2_window = max(1, min(10, (idx2 - idx_mid) // 2))
                    vol1 = data.iloc[idx1-vol1_window:idx1+vol1_window+1]['Volume'].mean()
                    vol2 = data.iloc[idx2-vol2_window:idx2+vol2_window+1]['Volume'].mean()
                    if vol2 >= vol1 * VOLUME_DECLINE_PCT:
                        continue
                except Exception:
                    pass

            # --- Rule 5: Look for neckline breakout ---
            neckline_level = t_high
            breakout_idx = -1
            
            for j in range(idx2 + 1, min(len(data), idx2 + 90)):  # look within 90 days
                if data['Close'].iloc[j] > neckline_level * (1 + NECKLINE_TOLERANCE):
                    # Check for volume spike or sustained break
                    if REQUIRE_VOLUME_CONFIRMATION and 'Volume' in data.columns:
                        try:
                            avg_vol = float(data.iloc[idx1:idx2+1]['Volume'].mean())
                            vol_j = float(data['Volume'].iloc[j])
                            if vol_j > avg_vol * VOLUME_SPIKE_MULT:
                                breakout_idx = j
                                break
                        except Exception:
                            pass
                    else:
                        # Check for sustained break
                        sustained = 0
                        look_ahead = min(len(data) - 1, j + 5)
                        for k in range(j, look_ahead + 1):
                            if data['Close'].iloc[k] > neckline_level:
                                sustained += 1
                        if sustained >= 2:
                            breakout_idx = j
                            break

            if breakout_idx == -1:
                continue

            results.append({
                'type': 'double_bottom',
                'P1': (p1_date, p1_low, idx1),
                'T': (t_date, t_high, idx_mid),
                'P2': (p2_date, p2_low, idx2),
                'breakout': (data['Date'].iloc[breakout_idx], data['Close'].iloc[breakout_idx], breakout_idx),
                'neckline_level': neckline_level,
                'duration': total_duration
            })

    return results

def plot_pattern_zoom(df, pattern, stock_name, output_path):
    p1_date, p1_price, p1_idx = pattern['P1']
    t_date, t_price, t_idx = pattern['T']
    p2_date, p2_price, p2_idx = pattern['P2']
    breakout_date, breakout_price, breakout_idx = pattern['breakout']
    neckline_level = pattern['neckline_level']

    start_date = p1_date - pd.Timedelta(days=30)
    end_date = breakout_date + pd.Timedelta(days=30)

    df_zoom = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    if df_zoom.empty:
        return

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), 
                                          gridspec_kw={'height_ratios': [3, 1]})
    
    ax_price.plot(df_zoom['Date'], df_zoom['Close'], color='navy', label='Close')

    # Mark pattern points
    if pattern['type'] == 'double_top':
        ax_price.scatter([p1_date, p2_date], [p1_price, p2_price], 
                        color=['red', 'red'], s=120, zorder=5, label='Double Top')
        ax_price.scatter([t_date], [t_price], color='blue', s=100, zorder=5, label='Valley')
        pattern_title = f"{stock_name} - Confirmed Double Top Pattern"
    else:
        ax_price.scatter([p1_date, p2_date], [p1_price, p2_price], 
                        color=['green', 'green'], s=120, zorder=5, label='Double Bottom')
        ax_price.scatter([t_date], [t_price], color='red', s=100, zorder=5, label='Peak')
        pattern_title = f"{stock_name} - Confirmed Double Bottom Pattern"

    # Draw neckline
    ax_price.hlines(neckline_level, df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1], 
                   colors='purple', linestyles='--', label='Neckline')

    # Mark breakout
    ax_price.scatter(breakout_date, breakout_price, color='orange', s=150, 
                    marker='*', label='Breakout')

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
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Double Top/Bottom Pattern Detector')
    parser.add_argument('--mode', choices=['strict', 'lenient', 'both'], default='strict', help='detection mode')
    parser.add_argument('--pattern', choices=['all', 'double_top', 'double_bottom'], default='all',
                        help='Pattern(s) to check: all, double_top, double_bottom')
    args = parser.parse_args()

    def apply_mode_settings(mode):
        if mode == 'lenient':
            globals()['PEAK_SIMILARITY_TOL'] = 0.12
            globals()['MIN_PROMINENCE_PCT'] = 0.03
            globals()['MAX_SPACING_DAYS'] = 180
            globals()['VOLUME_DECLINE_PCT'] = 0.75
            globals()['VOLUME_SPIKE_MULT'] = 1.25
            globals()['NECKLINE_TOLERANCE'] = 0.03
            globals()['USE_ZIGZAG'] = False
            globals()['REQUIRE_VOLUME_DIVERGENCE'] = False
            globals()['REQUIRE_VOLUME_CONFIRMATION'] = False
            globals()['REQUIRE_PRECEDING_TREND'] = False
        # strict mode uses defaults defined at top

    modes_to_run = []
    if args.mode == 'both':
        modes_to_run = ['lenient', 'strict']
    else:
        modes_to_run = [args.mode]

    # List of symbols
    symbols = [
        'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'MANT', 'NVDA', 'MBOT', 'AAPL',
        '360ONE.NS', '5PAISA.NS', 'AAIL.NS', 'ABB.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS',
        'AETHER.NS', 'SBIN.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BANKBARODA.NS',
        'ASHOKLEY.NS', 'SUZLON.NS', 'JYOTICNC.NS',
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'LT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'MARUTI.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS',
        'BAJFINANCE.NS', 'HCLTECH.NS', 'WIPRO.NS', 'ONGC.NS', 'BPCL.NS',
        'TATAMOTORS.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS', 'SUNPHARMA.NS', 'LTIM.NS', 'NESTLEIND.NS'
    ]

    start_date = '2015-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

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
                time.sleep(1)
                continue

            if df is None or df.empty:
                print(f"No data for {symbol}, skipping.")
                time.sleep(0.5)
                continue

            df = df.reset_index(drop=True)
            df = find_swing_points(df, N_bars=20)

            out_base = os.path.join(os.getcwd(), 'PatternChartsDT', symbol)

            for name, days in windows.items():
                df_slice = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=days)].copy()
                if df_slice.empty:
                    print(f"{symbol} - {name}: No data for window {name}")
                    continue

                df_slice = df_slice.reset_index(drop=True)
                nbars_slice = compute_dynamic_nbars(df_slice, base=BASE_NBARS)
                if USE_ZIGZAG:
                    df_slice = generate_swing_flags(df_slice, method='zigzag', N_bars=nbars_slice)
                elif USE_FRACTAL:
                    df_slice = generate_swing_flags(df_slice, method='fractal', N_bars=nbars_slice)
                else:
                    df_slice = generate_swing_flags(df_slice, method='rolling', N_bars=nbars_slice)

                patterns = detect_double_tops_and_bottoms(df_slice)

                # Filter patterns by type if specified
                if args.pattern == 'double_top':
                    patterns = [p for p in patterns if p['type'] == 'double_top']
                elif args.pattern == 'double_bottom':
                    patterns = [p for p in patterns if p['type'] == 'double_bottom']

                if patterns:
                    for pattern in patterns:
                        pattern_type = pattern['type']
                        folder_name = 'DT' if pattern_type == 'double_top' else 'DB'
                        base_folder = os.path.join(out_base, name)
                        pattern_folder = os.path.join(base_folder, folder_name)
                        os.makedirs(pattern_folder, exist_ok=True)

                    print(f"{symbol} - {name}: {len(patterns)} confirmed pattern(s) detected")
                    
                    dt_count = len([p for p in patterns if p['type'] == 'double_top'])
                    db_count = len([p for p in patterns if p['type'] == 'double_bottom'])
                    
                    dt_idx = 1
                    db_idx = 1
                    
                    for pattern in patterns:
                        pattern_type = pattern['type']
                        
                        if pattern_type == 'double_top':
                            folder_name = 'DT'
                            filename = f'{symbol}_DT_pattern_zoom_{name}_{dt_idx}.png'
                            dt_idx += 1
                        else:
                            folder_name = 'DB'
                            filename = f'{symbol}_DB_pattern_zoom_{name}_{db_idx}.png'
                            db_idx += 1
                        
                        base_folder = os.path.join(out_base, name)
                        pattern_folder = os.path.join(base_folder, folder_name)
                        out_path = os.path.join(pattern_folder, filename)
                        
                        try:
                            plot_pattern_zoom(df_slice, pattern, symbol, out_path)
                        except Exception as e:
                            print(f"Failed to plot pattern for {symbol} {name}: {e}")
                        
                        # Record summary info for CSV
                        try:
                            p1_date, p1_price, _ = pattern['P1']
                            t_date, t_price, _ = pattern['T']
                            p2_date, p2_price, _ = pattern['P2']
                            breakout_date, breakout_price, _ = pattern['breakout']
                        except Exception:
                            continue
                        
                        summary_records.append({
                            'pattern_type': pattern_type,
                            'symbol': symbol,
                            'window': name,
                            'P1_date': pd.to_datetime(p1_date).strftime('%Y-%m-%d'),
                            'P1_price': p1_price,
                            'T_date': pd.to_datetime(t_date).strftime('%Y-%m-%d'),
                            'T_price': t_price,
                            'P2_date': pd.to_datetime(p2_date).strftime('%Y-%m-%d'),
                            'P2_price': p2_price,
                            'breakout_date': pd.to_datetime(breakout_date).strftime('%Y-%m-%d'),
                            'breakout_price': breakout_price,
                            'neckline_level': pattern['neckline_level'],
                            'duration': pattern['duration'],
                            'image_path': out_path
                        })
                else:
                    if args.pattern == 'all':
                        print(f"{symbol} - {name}: No confirmed double top/bottom patterns detected.")
                    elif args.pattern == 'double_top':
                        print(f"{symbol} - {name}: No confirmed double top patterns detected.")
                    else:
                        print(f"{symbol} - {name}: No confirmed double bottom patterns detected.")

            time.sleep(1)

        # Save CSV summary
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            csv_path = os.path.join(os.getcwd(), f'double_patterns_detection_summary_{run_mode}.csv')
            try:
                summary_df.to_csv(csv_path, index=False)
                print(f"\nSaved summary CSV to {csv_path}")
            except Exception as e:
                print(f"Failed to write summary CSV: {e}")
        else:
            print(f"\nNo patterns detected across all symbols in mode={run_mode}; no summary CSV created.")
