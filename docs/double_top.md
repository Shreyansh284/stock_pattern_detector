# Double Top Pattern Detection

## Pattern Description

Double Top is a bearish reversal pattern characterized by two peaks at approximately the same price level, separated by a trough, indicating strong resistance and potential downward reversal.

## Algorithm Overview

### Detection Method

1. **Swing Point Identification**: Locates High → Low → High sequences
2. **Peak Similarity Analysis**: Ensures both peaks are at similar price levels
3. **Prominence Validation**: Confirms meaningful peak heights above trough
4. **Breakout Confirmation**: Validates break below support (neckline)

### Key Validation Rules

#### 1. Preceding Uptrend Requirement

- Requires established uptrend before pattern formation
- Minimum 15% price increase over 90-day lookback period
- Confirms pattern's reversal significance

#### 2. Peak Similarity Validation

- Both peaks must be within 6-12% of each other (mode dependent)
- Strict mode: 6% tolerance
- Lenient mode: 12% tolerance
- Formula: `|Peak1 - Peak2| / max(Peak1, Peak2) <= tolerance`

#### 3. Timing Constraints

- **Minimum Spacing**: 10-20 days between peaks
- **Maximum Spacing**: 90-150 days between peaks
- Ensures proper pattern formation timeframe

#### 4. Prominence Requirements

- Each peak must be significantly above the trough
- Minimum prominence: 3-6% above trough level
- Formula: `(Peak - Trough) / Trough >= min_prominence`

#### 5. Volume Analysis (Optional)

- Volume should decline on second peak (divergence)
- Indicates weakening buying pressure
- Breakout requires volume confirmation

#### 6. Neckline Breakout

- Support level defined by trough between peaks
- Break below neckline with tolerance (1.5-3%)
- Volume spike: 1.25-1.75x average pattern volume

## Configuration Parameters

### Strict Mode

```python
'PEAK_SIMILARITY_TOL': 0.06,    # Peak similarity tolerance
'MIN_PROMINENCE_PCT': 0.06,     # Minimum peak prominence
'MAX_SPACING_DAYS': 90,         # Maximum days between peaks
'MIN_SPACING_DAYS': 20,         # Minimum days between peaks
'VOLUME_DECLINE_PCT': 0.90,     # Volume divergence threshold
'VOLUME_SPIKE_MULT': 1.75,      # Breakout volume multiplier
'NECKLINE_TOLERANCE': 0.015,    # Neckline break tolerance
```

### Lenient Mode

```python
'PEAK_SIMILARITY_TOL': 0.12,
'MIN_PROMINENCE_PCT': 0.03,
'MAX_SPACING_DAYS': 150,
'MIN_SPACING_DAYS': 10,
'VOLUME_DECLINE_PCT': 0.75,
'VOLUME_SPIKE_MULT': 1.25,
'NECKLINE_TOLERANCE': 0.03,
```

## Code Implementation

### Core Function

```python
def detect_double_patterns(data, config, pattern_type='both', require_preceding_trend=True):
    """
    Detects Double Top patterns in price data

    Pattern Type: 'double_top', 'double_bottom', or 'both'

    Returns: List of pattern dictionaries containing:
    - P1: First peak coordinates
    - T: Trough coordinates (neckline level)
    - P2: Second peak coordinates
    - breakout: Confirmed breakout point
    - neckline_level: Support/resistance level
    - duration: Pattern formation time
    """
```

### Main Detection Logic

```python
def detect_double_patterns(data, config, pattern_type='both', require_preceding_trend=True):
    """Unified double pattern detection for both tops and bottoms."""

    results = []
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()

    # Step 1: Scan for 3-point sequences
    for i in range(len(swing_points_df) - 2):
        idx1, idx_mid, idx2 = swing_points_df['index'][i:i+3]

        # Step 2: Double Top Detection (High → Low → High)
        if (pattern_type in ('both', 'double_top') and
            data.get('is_swing_high', False)[idx1] and
            data.get('is_swing_low', False)[idx_mid] and
            data.get('is_swing_high', False)[idx2]):

            # Extract coordinates
            p1_high, p1_date = data.loc[idx1, ['High', 'Date']]
            t_low, t_date = data.loc[idx_mid, ['Low', 'Date']]
            p2_high, p2_date = data.loc[idx2, ['High', 'Date']]

            # Step 3: Apply Double Top validation rules
            pattern_valid, pattern_data = validate_double_top(
                data, idx1, idx_mid, idx2, p1_high, t_low, p2_high,
                p1_date, t_date, p2_date, config, require_preceding_trend)

            if pattern_valid:
                results.append(pattern_data)

    return results


def validate_double_top(data, idx1, idx_mid, idx2, p1_high, t_low, p2_high,
                       p1_date, t_date, p2_date, config, require_preceding_trend):
    """Comprehensive validation for Double Top patterns."""

    # Rule 1: Timing constraints
    try:
        spacing = (p2_date - p1_date).days
    except Exception:
        return False, None

    if (spacing < config['MIN_SPACING_DAYS'] or
        spacing > config['MAX_SPACING_DAYS']):
        return False, None

    # Rule 2: Preceding uptrend requirement
    if require_preceding_trend and not check_preceding_trend(data, idx1, 'up'):
        return False, None

    # Rule 3: Peak similarity validation
    peak_difference = abs(p1_high - p2_high) / max(p1_high, p2_high)
    if peak_difference > config['PEAK_SIMILARITY_TOL']:
        return False, None

    # Rule 4: Prominence validation
    try:
        prominence1 = (p1_high - t_low) / t_low
        prominence2 = (p2_high - t_low) / t_low

        if (prominence1 < config['MIN_PROMINENCE_PCT'] or
            prominence2 < config['MIN_PROMINENCE_PCT']):
            return False, None
    except Exception:
        return False, None

    # Rule 5: Volume divergence analysis (optional)
    volume_valid = validate_volume_divergence(data, idx1, idx_mid, idx2, config)
    if not volume_valid:
        return False, None

    # Rule 6: Neckline breakout confirmation
    neckline_level = t_low
    breakout_found, breakout_data = confirm_double_top_breakout(
        data, idx2, neckline_level, config)

    if breakout_found:
        pattern_data = {
            'type': 'double_top',
            'P1': (p1_date, p1_high, idx1),
            'T': (t_date, t_low, idx_mid),
            'P2': (p2_date, p2_high, idx2),
            'breakout': breakout_data,
            'neckline_level': neckline_level,
            'duration': spacing
        }
        return True, pattern_data

    return False, None
```

### Volume Divergence Analysis

```python
def validate_volume_divergence(data, idx1, idx_mid, idx2, config):
    """Analyze volume pattern for divergence confirmation."""

    if 'Volume' not in data.columns:
        return True  # Skip if no volume data

    try:
        # Calculate volume windows around each peak
        window_size = max(1, min(10, (idx_mid - idx1) // 2))

        # First peak volume
        vol1_start = max(0, idx1 - window_size)
        vol1_end = min(len(data), idx1 + window_size + 1)
        vol1 = data.iloc[vol1_start:vol1_end]['Volume'].mean()

        # Second peak volume
        vol2_start = max(0, idx2 - window_size)
        vol2_end = min(len(data), idx2 + window_size + 1)
        vol2 = data.iloc[vol2_start:vol2_end]['Volume'].mean()

        # Volume should decline on second peak (bearish divergence)
        volume_decline_valid = vol2 < vol1 * config['VOLUME_DECLINE_PCT']

        return volume_decline_valid

    except Exception:
        return True  # Skip validation if calculation fails
```

### Breakout Confirmation Logic

```python
def confirm_double_top_breakout(data, pattern_end_idx, neckline_level, config):
    """Confirm breakdown below support level with volume."""

    # Search for breakout within 90 days of pattern completion
    search_end = min(len(data), pattern_end_idx + 90)

    for j in range(pattern_end_idx + 1, search_end):
        close_price = data['Close'].iloc[j]

        # Check for breakdown below neckline (with tolerance)
        breakdown_threshold = neckline_level * (1 - config['NECKLINE_TOLERANCE'])

        if close_price < breakdown_threshold:
            # Attempt volume confirmation
            volume_confirmed = confirm_breakout_volume(data, pattern_end_idx, j, config)

            if volume_confirmed:
                breakout_data = (data['Date'].iloc[j], close_price, j)
                return True, breakout_data
            else:
                # Check for sustained breakdown (multiple days below)
                sustained = count_sustained_breakdown(data, j, neckline_level, 5)
                if sustained >= 2:
                    breakout_data = (data['Date'].iloc[j], close_price, j)
                    return True, breakout_data

    return False, None


def confirm_breakout_volume(data, pattern_start, breakout_idx, config):
    """Confirm breakout with volume spike analysis."""

    try:
        # Calculate average pattern volume
        avg_vol = float(data.iloc[pattern_start:breakout_idx]['Volume'].mean())
        breakout_vol = float(data['Volume'].iloc[breakout_idx])

        # Require volume spike above threshold
        volume_spike = breakout_vol > avg_vol * config['VOLUME_SPIKE_MULT']
        return volume_spike

    except Exception:
        return False


def count_sustained_breakdown(data, start_idx, neckline_level, look_ahead_days):
    """Count consecutive days with closes below neckline."""

    sustained_count = 0
    max_check = min(len(data), start_idx + look_ahead_days)

    for k in range(start_idx, max_check):
        if data['Close'].iloc[k] < neckline_level:
            sustained_count += 1
        else:
            break

    return sustained_count
```

### Pattern Structure Analysis

```python
def analyze_double_top_structure(p1_high, t_low, p2_high, p1_date, t_date, p2_date, config):
    """Analyze geometric properties of double top formation."""

    # Peak similarity analysis
    peak_avg = (p1_high + p2_high) / 2
    peak_difference = abs(p1_high - p2_high) / peak_avg
    similarity_score = 1 - peak_difference

    # Prominence analysis
    prominence1 = (p1_high - t_low) / t_low
    prominence2 = (p2_high - t_low) / t_low
    avg_prominence = (prominence1 + prominence2) / 2

    # Duration analysis
    total_duration = (p2_date - p1_date).days

    # Resistance strength (how well peaks align)
    resistance_level = max(p1_high, p2_high)
    resistance_test_strength = min(p1_high, p2_high) / resistance_level

    analysis_results = {
        'peak_similarity': similarity_score,
        'average_prominence': avg_prominence,
        'total_duration': total_duration,
        'resistance_strength': resistance_test_strength,
        'pattern_quality': calculate_pattern_quality(similarity_score, avg_prominence, resistance_test_strength)
    }

    return analysis_results


def calculate_pattern_quality(similarity, prominence, resistance_strength):
    """Calculate overall pattern quality score (0-1)."""

    # Weight different factors
    similarity_weight = 0.4
    prominence_weight = 0.3
    resistance_weight = 0.3

    # Normalize prominence (cap at reasonable level)
    normalized_prominence = min(prominence / 0.15, 1.0)  # 15% prominence = max score

    quality_score = (
        similarity * similarity_weight +
        normalized_prominence * prominence_weight +
        resistance_strength * resistance_weight
    )

    return min(quality_score, 1.0)
```

### Pattern Structure

Each detected Double Top pattern contains:

- **P1**: First peak (date, price, index)
- **T**: Trough between peaks (date, price, index)
- **P2**: Second peak (date, price, index)
- **breakout**: Confirmed breakdown point (date, price, index)
- **neckline_level**: Support level (trough price)
- **duration**: Total pattern duration in days
- **type**: Pattern type identifier ('double_top')

### Algorithm Steps

1. **Scan Swing Points**: Identify High → Low → High sequences
2. **Apply Timing Filters**: Check spacing between peaks
3. **Validate Similarities**: Ensure peaks are at similar levels
4. **Check Prominence**: Confirm meaningful peak heights
5. **Analyze Volume**: Optional volume divergence check
6. **Confirm Breakout**: Validate breakdown below neckline
7. **Record Pattern**: Store complete pattern data

## Breakout Validation

- **Price Criterion**: Close below `neckline * (1 - tolerance)`
- **Volume Criterion**: Volume spike above average or sustained break
- **Sustainability**: Look for 2+ consecutive days below neckline
- **Time Limit**: Search within 90 days of second peak

## Output

- **Charts**: PNG files with marked peaks, trough, and neckline
- **Data**: CSV reports with pattern coordinates and breakout timing
- **Folder Structure**: `outputs/charts/{SYMBOL}/{TIMEFRAME}/DT/`

## Usage Example

```python
# Configure for strict double top detection
config = DT_CONFIG['strict']

# Detect only double tops
patterns = detect_double_patterns(data, config, pattern_type='double_top')

# Analyze results
for pattern in patterns:
    peak1_price = pattern['P1'][1]
    peak2_price = pattern['P2'][1]
    similarity = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
    print(f"Peak similarity: {similarity:.1%}")
    print(f"Breakdown at: {pattern['breakout'][1]:.2f}")
```

## Key Features

- **Precision Filtering**: Multiple validation layers reduce false positives
- **Flexible Timing**: Configurable spacing constraints for different markets
- **Volume Integration**: Optional volume analysis for pattern confirmation
- **Trend Context**: Validates preceding trend for pattern significance
- **Robust Breakout**: Multiple criteria ensure genuine breakdowns
