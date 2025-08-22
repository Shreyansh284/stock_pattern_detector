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
CUP_DEPTH_MIN = 0.12               # minimum cup depth as percentage of peak price
CUP_DEPTH_MAX = 0.50               # maximum cup depth as percentage of peak price
HANDLE_DEPTH_MAX = 0.25            # handle should not retrace more than 25% of cup depth
CUP_SYMMETRY_TOL = 0.30            # tolerance for cup symmetry (left vs right side)
HANDLE_DURATION_MIN = 5            # minimum handle duration in days
HANDLE_DURATION_MAX = 90           # maximum handle duration in days
CUP_DURATION_MIN = 30              # minimum cup duration in days
CUP_DURATION_MAX = 365             # maximum cup duration in days
VOLUME_DECLINE_PCT = 0.80          # volume should decline during cup formation
VOLUME_SPIKE_MULT = 1.50           # breakout volume must exceed avg pattern volume * this multiplier
BASE_NBARS = 20                    # base lookback for swing detection
MIN_NBARS = 8
MAX_NBARS = 60

# Feature flags (enable/disable expensive checks)
USE_ZIGZAG = True                  # use ZigZag swing detection instead of rolling highs/lows
ZIGZAG_PCT = 0.03                  # 3% reversal threshold for ZigZag
USE_FRACTAL = False                # use Williams Fractals for swing detection
REQUIRE_VOLUME_CONFIRMATION = True # require volume confirmation on breakout
REQUIRE_PRECEDING_UPTREND = True   # require preceding uptrend before cup formation
MIN_UPTREND_PCT = 0.20             # minimum uptrend percentage before cup

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

def check_preceding_uptrend(data, pattern_start_index, lookback_period=90, min_rise_percent=MIN_UPTREND_PCT):
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

def detect_cup_and_handle(data, min_days=None, max_days=None):
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    patterns = []

    # Look for cup pattern: High -> Low -> High sequence
    # Fix indexing issue by using proper boolean indexing
    high_indices = swing_points_df[swing_points_df['index'].apply(lambda x: data.iloc[x].get('is_swing_high', False))].reset_index(drop=True)
    low_indices = swing_points_df[swing_points_df['index'].apply(lambda x: data.iloc[x].get('is_swing_low', False))].reset_index(drop=True)

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

            if cup_duration < CUP_DURATION_MIN or cup_duration > CUP_DURATION_MAX:
                continue

            # Duration filter (optional)
            if min_days is not None and cup_duration < min_days:
                continue
            if max_days is not None and cup_duration > max_days:
                continue

            # Find the lowest point between the two rims (cup bottom)
            cup_section = data.iloc[left_rim_idx:right_rim_idx+1]
            if cup_section.empty:
                continue

            cup_bottom_idx = cup_section['Low'].idxmin()
            cup_bottom_low = data.loc[cup_bottom_idx, 'Low']
            cup_bottom_date = data.loc[cup_bottom_idx, 'Date']

            # --- Rule 1: Preceding Uptrend ---
            if REQUIRE_PRECEDING_UPTREND and not check_preceding_uptrend(data, left_rim_idx):
                continue

            # --- Rule 2: Cup Depth ---
            left_rim_price = max(left_rim_high, right_rim_high)
            cup_depth = (left_rim_price - cup_bottom_low) / left_rim_price
            if cup_depth < CUP_DEPTH_MIN or cup_depth > CUP_DEPTH_MAX:
                continue

            # --- Rule 3: Cup Symmetry ---
            # Check if the cup is reasonably symmetric
            left_side_duration = (cup_bottom_date - left_rim_date).days
            right_side_duration = (right_rim_date - cup_bottom_date).days
            if left_side_duration == 0 or right_side_duration == 0:
                continue

            symmetry_ratio = min(left_side_duration, right_side_duration) / max(left_side_duration, right_side_duration)
            if symmetry_ratio < (1 - CUP_SYMMETRY_TOL):
                continue

            # --- Rule 4: Rim Height Similarity ---
            rim_difference = abs(left_rim_high - right_rim_high) / max(left_rim_high, right_rim_high)
            if rim_difference > 0.05:  # rims should be within 5% of each other
                continue

            # --- Rule 5: Volume Pattern during Cup ---
            if REQUIRE_VOLUME_CONFIRMATION and 'Volume' in data.columns:
                try:
                    # Volume should generally decline during cup formation
                    early_cup_vol = data.iloc[left_rim_idx:left_rim_idx+10]['Volume'].mean()
                    late_cup_vol = data.iloc[right_rim_idx-10:right_rim_idx]['Volume'].mean()
                    if late_cup_vol > early_cup_vol * VOLUME_DECLINE_PCT:
                        continue
                except Exception:
                    pass

            # --- Rule 6: Look for Handle Formation ---
            handle_found = False
            handle_start_idx = right_rim_idx
            handle_low_idx = -1
            handle_end_idx = -1

            # Look for a pullback after the right rim (handle)
            for k in range(right_rim_idx + 1, min(len(data), right_rim_idx + HANDLE_DURATION_MAX)):
                current_low = data.loc[k, 'Low']
                current_date = data.loc[k, 'Date']

                # Handle should not retrace too much
                handle_depth = (right_rim_high - current_low) / right_rim_high
                if handle_depth > HANDLE_DEPTH_MAX:
                    break

                # Look for handle duration
                handle_duration = (current_date - right_rim_date).days
                if handle_duration >= HANDLE_DURATION_MIN:
                    # Found potential handle - now look for breakout
                    for m in range(k + 1, min(len(data), k + 30)):  # look for breakout within 30 days
                        if data.loc[m, 'Close'] > right_rim_high:
                            # Breakout confirmed
                            if REQUIRE_VOLUME_CONFIRMATION and 'Volume' in data.columns:
                                try:
                                    avg_pattern_volume = data.iloc[left_rim_idx:k]['Volume'].mean()
                                    breakout_volume = data.loc[m, 'Volume']
                                    if breakout_volume > avg_pattern_volume * VOLUME_SPIKE_MULT:
                                        handle_found = True
                                        handle_low_idx = k
                                        handle_end_idx = m
                                        break
                                except Exception:
                                    pass
                            else:
                                handle_found = True
                                handle_low_idx = k
                                handle_end_idx = m
                                break

                if handle_found:
                    break

            if not handle_found:
                continue

            # Store the pattern
            pattern_data = {
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

def plot_pattern_zoom(df, pattern, stock_name, output_path):
    left_rim_date, left_rim_high, left_rim_idx = pattern['left_rim']
    cup_bottom_date, cup_bottom_low, cup_bottom_idx = pattern['cup_bottom']
    right_rim_date, right_rim_high, right_rim_idx = pattern['right_rim']
    handle_date, handle_low, handle_low_idx = pattern['handle_low']
    breakout_date, breakout_price, breakout_idx = pattern['breakout']

    start_date = left_rim_date - pd.Timedelta(days=30)
    end_date = breakout_date + pd.Timedelta(days=30)

    df_zoom = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    if df_zoom.empty:
        return

    # Create two stacked axes: price (top) and volume (bottom) sharing the x-axis
    fig, (ax_price, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                                          gridspec_kw={'height_ratios': [3, 1]})

    # Price line
    ax_price.plot(df_zoom['Date'], df_zoom['Close'], label='Close Price', color='navy')

    # Mark cup pattern points
    ax_price.scatter([left_rim_date, right_rim_date], [left_rim_high, right_rim_high], 
                    color=['blue', 'blue'], s=120, zorder=5, label='Cup Rims')
    ax_price.scatter([cup_bottom_date], [cup_bottom_low], 
                    color='red', s=120, zorder=5, label='Cup Bottom')
    ax_price.scatter([handle_date], [handle_low], 
                    color='orange', s=100, zorder=5, label='Handle Low')

    # Draw cup outline
    cup_section = df[(df['Date'] >= left_rim_date) & (df['Date'] <= right_rim_date)]
    if not cup_section.empty:
        ax_price.plot(cup_section['Date'], cup_section['Low'], 'g--', alpha=0.7, linewidth=2, label='Cup Shape')

    # Draw resistance line at cup rim level
    resistance_level = max(left_rim_high, right_rim_high)
    ax_price.hlines(resistance_level, df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1], 
                   colors='purple', linestyles='--', alpha=0.7, label='Resistance Level')

    # Mark Breakout
    ax_price.scatter(breakout_date, breakout_price, color='green', s=150, zorder=6, 
                    marker='*', label='Breakout')

    # Volume bars on bottom axis
    if 'Volume' in df_zoom.columns:
        ax_vol.bar(df_zoom['Date'], df_zoom['Volume'], color='gray', alpha=0.6)
        # Highlight breakout volume bar
        try:
            ax_vol.bar(breakout_date, df.loc[breakout_idx, 'Volume'], color='green', alpha=0.9)
        except Exception:
            pass

    # Formatting
    ax_price.set_xlim(df_zoom['Date'].iloc[0], df_zoom['Date'].iloc[-1])
    ax_price.set_ylim(df_zoom['Low'].min() * 0.98, df_zoom['High'].max() * 1.02)
    ax_price.set_title(f"{stock_name} - Confirmed Cup and Handle Pattern")
    ax_price.set_ylabel('Price')
    ax_vol.set_ylabel('Volume')
    ax_vol.set_xlabel('Date')

    ax_price.legend(loc='upper left')
    ax_price.grid(True, linestyle='--', alpha=0.6)
    ax_vol.grid(False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cup and Handle Pattern Detector')
    parser.add_argument('--mode', choices=['strict', 'lenient', 'both'], default='strict', help='detection mode')
    args = parser.parse_args()

    def apply_mode_settings(mode):
        if mode == 'lenient':
            globals()['CUP_DEPTH_MIN'] = 0.08
            globals()['CUP_DEPTH_MAX'] = 0.60
            globals()['HANDLE_DEPTH_MAX'] = 0.35
            globals()['CUP_SYMMETRY_TOL'] = 0.50
            globals()['VOLUME_DECLINE_PCT'] = 0.70
            globals()['VOLUME_SPIKE_MULT'] = 1.25
            globals()['USE_ZIGZAG'] = False
            globals()['REQUIRE_VOLUME_CONFIRMATION'] = False
            globals()['REQUIRE_PRECEDING_UPTREND'] = False
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

            out_base = os.path.join(os.getcwd(), 'PatternChartsCH', symbol)

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

                patterns = detect_cup_and_handle(df_slice)

                if patterns:
                    base_folder = os.path.join(out_base, name)
                    ch_folder = os.path.join(base_folder, 'CH')
                    os.makedirs(ch_folder, exist_ok=True)
                    print(f"{symbol} - {name}: {len(patterns)} confirmed pattern(s) detected")
                    
                    for idx, pattern in enumerate(patterns, 1):
                        out_path = os.path.join(ch_folder, f'{symbol}_CH_pattern_zoom_{name}_{idx}.png')
                        try:
                            plot_pattern_zoom(df_slice, pattern, symbol, out_path)
                        except Exception as e:
                            print(f"Failed to plot pattern for {symbol} {name} #{idx}: {e}")
                        
                        # Record summary info for CSV
                        try:
                            left_rim_date, left_rim_high, _ = pattern['left_rim']
                            cup_bottom_date, cup_bottom_low, _ = pattern['cup_bottom']
                            right_rim_date, right_rim_high, _ = pattern['right_rim']
                            handle_date, handle_low, _ = pattern['handle_low']
                            breakout_date, breakout_price, _ = pattern['breakout']
                        except Exception:
                            continue
                        
                        summary_records.append({
                            'pattern_type': 'cup_and_handle',
                            'symbol': symbol,
                            'window': name,
                            'left_rim_date': pd.to_datetime(left_rim_date).strftime('%Y-%m-%d'),
                            'left_rim_high': left_rim_high,
                            'cup_bottom_date': pd.to_datetime(cup_bottom_date).strftime('%Y-%m-%d'),
                            'cup_bottom_low': cup_bottom_low,
                            'right_rim_date': pd.to_datetime(right_rim_date).strftime('%Y-%m-%d'),
                            'right_rim_high': right_rim_high,
                            'handle_date': pd.to_datetime(handle_date).strftime('%Y-%m-%d'),
                            'handle_low': handle_low,
                            'breakout_date': pd.to_datetime(breakout_date).strftime('%Y-%m-%d'),
                            'breakout_price': breakout_price,
                            'cup_duration': pattern['cup_duration'],
                            'cup_depth': pattern['cup_depth'],
                            'image_path': out_path
                        })
                else:
                    print(f"{symbol} - {name}: No confirmed cup and handle pattern detected.")

            time.sleep(1)

        # Save CSV summary
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            csv_path = os.path.join(os.getcwd(), f'cup_handle_detection_summary_{run_mode}.csv')
            try:
                summary_df.to_csv(csv_path, index=False)
                print(f"\nSaved summary CSV to {csv_path}")
            except Exception as e:
                print(f"Failed to write summary CSV: {e}")
        else:
            print(f"\nNo patterns detected across all symbols in mode={run_mode}; no summary CSV created.")
