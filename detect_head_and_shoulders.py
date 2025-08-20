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

def detect_head_and_shoulders(data, min_days=None, max_days=None):
    swing_points_df = data[data['is_swing_high'] | data['is_swing_low']].reset_index()
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

        # --- Rule 2: Head must be the highest peak ---
        if not (p2_high > p1_high and p2_high > p3_high):
            continue

        # --- Rule 3: Shoulders Proportionality ---
        if abs(p1_high - p3_high) / max(p1_high, p3_high) > 0.15: # Shoulders within 15% height
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

        if not (vol_ls > vol_h * 0.8 and vol_h > vol_rs * 0.8): # Use a tolerance
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
                if vol_j > avg_volume_pattern * 1.5: # 50% volume spike
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

if __name__ == "__main__":
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

            # Reset index and recompute swing points on the slice so indices are positional (0..n-1)
            df_slice = df_slice.reset_index(drop=True)
            df_slice = find_swing_points(df_slice, N_bars=20)

            patterns = detect_head_and_shoulders(df_slice)
            folder = os.path.join(out_base, name)
            os.makedirs(folder, exist_ok=True)
            if patterns:
                print(f"{symbol} - {name}: {len(patterns)} confirmed pattern(s) detected")
                for idx, pattern in enumerate(patterns, 1):
                    out_path = os.path.join(folder, f'{symbol}_pattern_zoom_{name}_{idx}.png')
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

        # Be polite to the API
        time.sleep(1)

    # Save CSV summary of all detected patterns
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        csv_path = os.path.join(os.getcwd(), 'pattern_detection_summary.csv')
        try:
            summary_df.to_csv(csv_path, index=False)
            print(f"\nSaved summary CSV to {csv_path}")
        except Exception as e:
            print(f"Failed to write summary CSV: {e}")
    else:
        print("\nNo patterns detected across all symbols; no summary CSV created.")
