# Head and Shoulders Pattern Detection

## Pattern Description

Head and Shoulders is a classic bearish reversal pattern consisting of three peaks: left shoulder, head (highest peak), and right shoulder, with two troughs forming the neckline.

## Algorithm Overview

### Detection Method

1. **Swing Point Identification**: Uses rolling window method to find local highs and lows
2. **Pattern Structure**: Identifies sequence of High → Low → High → Low → High
3. **Validation Rules**: Applies multiple criteria to confirm valid patterns

### Key Validation Rules

#### 1. Preceding Uptrend Requirement

- Checks for significant uptrend before pattern formation
- Minimum 15% price increase over 90-day lookback period

#### 2. Head Height Validation

- Head must be the highest peak among all three peaks
- Head must be significantly above shoulders (3-7% minimum depending on mode)

#### 3. Shoulder Proportionality

- Left and right shoulders must be similar in height
- Tolerance: 10-15% difference (configurable by mode)

#### 4. Neckline Analysis

- Calculated from two trough points (T1 and T2)
- Maximum allowed neckline slope: 25-45 degrees
- Formula: `slope = (T2_low - T1_low) / (T2_date - T1_date)`

#### 5. Volume Pattern Confirmation

- Volume should decline from left shoulder → head → right shoulder
- Validates diminishing buying interest during pattern formation

#### 6. Breakout Confirmation

- Price must close below neckline with volume spike
- Volume spike: 1.25-1.75x average pattern volume
- Breakout search window: up to 60 days after right shoulder

## Configuration Parameters

### Strict Mode

```python
'SHOULDER_TOL': 0.10,           # Shoulder similarity tolerance
'HEAD_OVER_SHOULDER_PCT': 0.07, # Head prominence requirement
'MAX_NECKLINE_ANGLE_DEG': 25,   # Maximum neckline slope
'VOLUME_TREND_TOL': 0.95,       # Volume decline requirement
'VOLUME_SPIKE_MULT': 1.75,      # Breakout volume multiplier
```

### Lenient Mode

```python
'SHOULDER_TOL': 0.15,
'HEAD_OVER_SHOULDER_PCT': 0.03,
'MAX_NECKLINE_ANGLE_DEG': 45,
'VOLUME_TREND_TOL': 0.85,
'VOLUME_SPIKE_MULT': 1.25,
```

## Code Implementation

### Core Function

```python
def detect_head_and_shoulders(data, config, require_preceding_trend=True):
    """
    Detects Head and Shoulders patterns in price data

    Returns: List of pattern dictionaries containing:
    - P1, P2, P3: Shoulder and head coordinates
    - T1, T2: Trough coordinates
    - breakout: Breakout point details
    - neckline_slope/intercept: Neckline parameters
    """
```

### Main Detection Logic

```python
def detect_head_and_shoulders(data, config, require_preceding_trend=True):
    """Main detection algorithm with step-by-step validation."""

    # Step 1: Extract swing points from data
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    patterns = []

    # Step 2: Scan for 5-point sequences (H-L-H-L-H)
    for i in range(len(swing_points_df) - 4):
        p1_idx, t1_idx, p2_idx, t2_idx, p3_idx = swing_points_df['index'][i:i+5]

        # Step 3: Validate swing sequence structure
        if not (data['is_swing_high'][p1_idx] and data['is_swing_low'][t1_idx] and
                data['is_swing_high'][p2_idx] and data['is_swing_low'][t2_idx] and
                data['is_swing_high'][p3_idx]):
            continue

        # Step 4: Extract price and date coordinates
        p1_high, p1_date = data.loc[p1_idx, ['High', 'Date']]
        t1_low, t1_date = data.loc[t1_idx, ['Low', 'Date']]
        p2_high, p2_date = data.loc[p2_idx, ['High', 'Date']]
        t2_low, t2_date = data.loc[t2_idx, ['Low', 'Date']]
        p3_high, p3_date = data.loc[p3_idx, ['High', 'Date']]

        # Step 5: Apply validation rules

        # Rule 1: Preceding uptrend validation
        if require_preceding_trend and not check_preceding_trend(data, p1_idx, 'up'):
            continue

        # Rule 2: Head height validation
        higher_shoulder = max(p1_high, p3_high)
        if not (p2_high > p1_high and p2_high > p3_high):
            continue
        if not (p2_high > higher_shoulder * (1 + config['HEAD_OVER_SHOULDER_PCT'])):
            continue

        # Rule 3: Shoulder proportionality check
        if abs(p1_high - p3_high) / max(p1_high, p3_high) > config['SHOULDER_TOL']:
            continue

        # Rule 4: Neckline calculation and angle validation
        neckline_y = [t1_low, t2_low]
        neckline_x = [t1_date.toordinal(), t2_date.toordinal()]

        if neckline_x[1] - neckline_x[0] == 0:
            continue

        slope = (neckline_y[1] - neckline_y[0]) / (neckline_x[1] - neckline_x[0])
        intercept = neckline_y[0] - slope * neckline_x[0]

        # Validate neckline angle
        try:
            angle_deg = abs(np.degrees(np.arctan(slope)))
            if angle_deg > config['MAX_NECKLINE_ANGLE_DEG']:
                continue
        except Exception:
            continue

        # Rule 5: Volume pattern analysis
        try:
            vol_ls = float(data['Volume'].iloc[p1_idx:t1_idx+1].mean())  # Left shoulder
            vol_h = float(data['Volume'].iloc[t1_idx:t2_idx+1].mean())   # Head
            vol_rs = float(data['Volume'].iloc[t2_idx:p3_idx+1].mean()) # Right shoulder

            if np.isnan(vol_ls) or np.isnan(vol_h) or np.isnan(vol_rs):
                continue

            # Volume should decline: LS > H and H > RS (with tolerance)
            if not (vol_ls > vol_h * config['VOLUME_TREND_TOL'] and
                   vol_h > vol_rs * config['VOLUME_TREND_TOL']):
                continue
        except Exception:
            continue

        # Rule 6: Breakout confirmation
        breakout_confirmed = False
        breakout_idx = -1

        for j in range(p3_idx + 1, min(len(data), p3_idx + 90)):
            date_j = data['Date'].iloc[j]
            neckline_price_at_j = slope * date_j.toordinal() + intercept
            close_j = data['Close'].iloc[j]

            # Check for close below neckline
            if close_j < neckline_price_at_j:
                avg_volume_pattern = float(data.iloc[p1_idx:p3_idx+1]['Volume'].mean())
                vol_j = float(data['Volume'].iloc[j])

                # Require volume spike for confirmation
                if vol_j > avg_volume_pattern * config['VOLUME_SPIKE_MULT']:
                    breakout_confirmed = True
                    breakout_idx = j
                    break

        # Step 6: Store confirmed pattern
        if breakout_confirmed:
            pattern_data = {
                'type': 'head_and_shoulders',
                'P1': (p1_date, p1_high, p1_idx),    # Left shoulder
                'T1': (t1_date, t1_low, t1_idx),     # First trough
                'P2': (p2_date, p2_high, p2_idx),    # Head
                'T2': (t2_date, t2_low, t2_idx),     # Second trough
                'P3': (p3_date, p3_high, p3_idx),    # Right shoulder
                'breakout': (data['Date'].iloc[breakout_idx],
                           data['Close'].iloc[breakout_idx], breakout_idx),
                'neckline_slope': slope,
                'neckline_intercept': intercept,
                'duration': (p3_date - p1_date).days
            }
            patterns.append(pattern_data)

    return patterns
```

### Volume Analysis Logic

```python
def analyze_hns_volume_pattern(data, p1_idx, t1_idx, p2_idx, t2_idx, p3_idx, config):
    """Analyze volume pattern for Head and Shoulders formation."""

    # Calculate average volume for each section
    vol_left_shoulder = data['Volume'].iloc[p1_idx:t1_idx+1].mean()
    vol_head = data['Volume'].iloc[t1_idx:t2_idx+1].mean()
    vol_right_shoulder = data['Volume'].iloc[t2_idx:p3_idx+1].mean()

    # Check for declining volume pattern
    # Volume should generally decline from left shoulder → head → right shoulder
    volume_decline_valid = (
        vol_left_shoulder > vol_head * config['VOLUME_TREND_TOL'] and
        vol_head > vol_right_shoulder * config['VOLUME_TREND_TOL']
    )

    return volume_decline_valid, {
        'left_shoulder_volume': vol_left_shoulder,
        'head_volume': vol_head,
        'right_shoulder_volume': vol_right_shoulder
    }
```

### Breakout Validation Logic

```python
def validate_hns_breakout(data, pattern_end_idx, neckline_slope, neckline_intercept, config):
    """Validate neckline breakout with volume confirmation."""

    # Search for breakout within 90 days after pattern completion
    for j in range(pattern_end_idx + 1, min(len(data), pattern_end_idx + 90)):
        date_j = data['Date'].iloc[j]
        close_j = data['Close'].iloc[j]

        # Calculate neckline price at current date
        neckline_price = neckline_slope * date_j.toordinal() + neckline_intercept

        # Check for price breakdown below neckline
        if close_j < neckline_price:
            volume_j = data['Volume'].iloc[j]
            avg_pattern_volume = data.iloc[pattern_start_idx:pattern_end_idx+1]['Volume'].mean()

            # Require volume spike for confirmation
            if volume_j > avg_pattern_volume * config['VOLUME_SPIKE_MULT']:
                return True, j, close_j

    return False, -1, None
```

### Pattern Structure

Each detected pattern contains:

- **P1**: Left shoulder (date, price, index)
- **T1**: First trough (date, price, index)
- **P2**: Head (date, price, index)
- **T2**: Second trough (date, price, index)
- **P3**: Right shoulder (date, price, index)
- **breakout**: Confirmed breakout point
- **neckline**: Slope and intercept parameters

## Output

- **Charts**: PNG files showing pattern visualization with marked points
- **Data**: CSV reports with pattern coordinates and breakout details
- **Folder Structure**: `outputs/charts/{SYMBOL}/{TIMEFRAME}/HNS/`

## Usage Example

```python
# Configure for strict detection
config = HNS_CONFIG['strict']

# Detect patterns
patterns = detect_head_and_shoulders(data, config, require_preceding_trend=True)

# Each pattern includes complete coordinate information
for pattern in patterns:
    print(f"Head at {pattern['P2'][0]}: ${pattern['P2'][1]:.2f}")
    print(f"Breakout at {pattern['breakout'][0]}: ${pattern['breakout'][1]:.2f}")
```
