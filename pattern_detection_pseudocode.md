# Stock Pattern Detection: Pseudocode and Real Code Reference

This document provides both high-level pseudocode and direct Python code references for detecting Head and Shoulders, Double Top, and Double Bottom patterns, as implemented in `detect_head_and_shoulders.py`.

---

## 1. Data Preparation

**Pseudocode:**

- For each stock symbol:
  - Download historical price and volume data.
  - Compute swing points (local highs/lows) using rolling, ZigZag, or fractal methods.
  - Optionally compute technical indicators (ATR, ADX, RSI, MACD).

**Python Example:**

```python
import yfinance as yf
import pandas as pd

def load_data(symbol, start_date, end_date):
  data = yf.download(symbol, start=start_date, end=end_date)
  data = data.reset_index()
  if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
  return data

def find_swing_points(data, N_bars=20):
  data['is_swing_high'] = (data['High'] == data['High'].rolling(window=N_bars*2+1, center=True).max())
  data['is_swing_low'] = (data['Low'] == data['Low'].rolling(window=N_bars*2+1, center=True).min())
  return data
```

---

## 2. Head and Shoulders Pattern Detection

**Pseudocode:**

- For each sequence of 5 swing points: High, Low, High, Low, High
  - Check uptrend before pattern
  - Head must be above both shoulders
  - Shoulders must be similar
  - Calculate neckline (line through two lows)
  - Neckline slope not too steep
  - Diminishing volume during formation
  - Confirm breakout: close below neckline with volume spike
  - If all true, record pattern

**Python Example:**

```python
def check_preceding_uptrend(data, pattern_start_index, lookback_period=90, min_rise_percent=0.15):
  p_idx = int(pattern_start_index)
  if p_idx < lookback_period:
    return False
  start_pos = max(0, p_idx - lookback_period)
  lookback_data = data.iloc[start_pos: p_idx]
  if lookback_data.empty:
    return False
  start_price = float(lookback_data['Close'].iloc[0])
  end_price = float(data.iloc[p_idx]['High'])
  return end_price > start_price * (1 + min_rise_percent)

def detect_head_and_shoulders(data):
  swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
  patterns = []
  for i in range(len(swing_points_df) - 4):
    p1_idx, t1_idx, p2_idx, t2_idx, p3_idx = swing_points_df['index'][i:i+5]
    if not (data['is_swing_high'][p1_idx] and data['is_swing_low'][t1_idx] and
        data['is_swing_high'][p2_idx] and data['is_swing_low'][t2_idx] and
        data['is_swing_high'][p3_idx]):
      continue
    # ... (see full code for all rules and breakout logic)
    # If all rules pass:
    patterns.append({'P1': p1_idx, 'P2': p2_idx, 'P3': p3_idx})
  return patterns
```

---

## 3. Double Top Pattern Detection

**Pseudocode:**

- For each sequence of 3 swing points: High, Low, High
  - Both highs similar
  - Preceding uptrend
  - Peaks above trough
  - Confirm breakout: close below trough with volume spike or sustained close
  - If all true, record pattern

**Python Example:**

```python
def detect_double_tops_and_bottoms(data, max_spacing_days=90):
  results = []
  sw = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
  for i in range(len(sw) - 2):
    idx1, idx_mid, idx2 = sw['index'][i:i+3]
    # double top: High, Low, High
    if data.get('is_swing_high', False)[idx1] and data.get('is_swing_low', False)[idx_mid] and data.get('is_swing_high', False)[idx2]:
      # ... (see full code for all rules and breakout logic)
      results.append({'type': 'double_top', 'P1': idx1, 'T': idx_mid, 'P2': idx2})
    # double bottom: Low, High, Low
    if data.get('is_swing_low', False)[idx1] and data.get('is_swing_high', False)[idx_mid] and data.get('is_swing_low', False)[idx2]:
      # ... (see full code for all rules and breakout logic)
      results.append({'type': 'double_bottom', 'P1': idx1, 'T': idx_mid, 'P2': idx2})
  return results
```

---

## 4. Double Bottom Pattern Detection

**Pseudocode:**

- For each sequence of 3 swing points: Low, High, Low
  - Both lows similar
  - Preceding downtrend
  - Troughs below peak
  - Confirm breakout: close above peak with volume spike or sustained close
  - If all true, record pattern

**Python Example:**

See above: `detect_double_tops_and_bottoms` handles both double top and double bottom.

---

## 5. Output and Visualization

**Pseudocode:**

- For each detected pattern:
  - Save summary to CSV
  - Generate and save zoomed-in plot
  - Only create output folders if a pattern is detected

**Python Example:**

```python
import matplotlib.pyplot as plt

def plot_pattern_zoom(df, pattern, stock_name, output_path):
  # ... mark key points, draw neckline, plot breakout, save image ...
  plt.savefig(output_path)
  plt.close()
```

CSV output is handled with pandas:

```python
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv('pattern_detection_summary.csv', index=False)
```

---

## 6. Command-Line Options

**Pseudocode:**

- Allow user to select pattern(s) via flag
- If no flag, detect all
- Allow strict/lenient detection modes

**Python Reference:**

- `argparse.ArgumentParser` with `--pattern` and `--mode` flags
- Pattern selection logic in main script

---

## 7. Main Script Flow

**Pseudocode:**

- For each mode (strict/lenient):
  - For each symbol:
    - For each time window:
      - Prepare data
      - Detect selected pattern(s)
      - Save results and plots if patterns found

**Python Reference:**

- See `if __name__ == "__main__":` block in `detect_head_and_shoulders.py`

---

For detailed implementation, see the corresponding function names in `detect_head_and_shoulders.py`.
