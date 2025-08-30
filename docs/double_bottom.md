# Double Bottom Pattern Detection

## Pattern Description

Double Bottom is a bullish reversal pattern characterized by two troughs at approximately the same price level, separated by a peak, indicating strong support and potential upward reversal.

## Algorithm Overview

### Detection Method

1. **Swing Point Identification**: Locates Low → High → Low sequences
2. **Trough Similarity Analysis**: Ensures both troughs are at similar price levels
3. **Prominence Validation**: Confirms meaningful trough depths below peak
4. **Breakout Confirmation**: Validates break above resistance (neckline)

### Key Validation Rules

#### 1. Preceding Downtrend Requirement

- Requires established downtrend before pattern formation
- Minimum 15% price decline over 90-day lookback period
- Confirms pattern's reversal significance

#### 2. Trough Similarity Validation

- Both troughs must be within 6-12% of each other (mode dependent)
- Strict mode: 6% tolerance
- Lenient mode: 12% tolerance
- Formula: `|Trough1 - Trough2| / max(Trough1, Trough2) <= tolerance`

#### 3. Timing Constraints

- **Minimum Spacing**: 10-20 days between troughs
- **Maximum Spacing**: 90-150 days between troughs
- Ensures proper pattern formation timeframe

#### 4. Prominence Requirements

- Peak must be significantly above both troughs
- Minimum prominence: 3-6% above trough levels
- Formula: `(Peak - Trough) / Peak >= min_prominence`

#### 5. Volume Analysis (Optional)

- Volume should decline on second trough (divergence)
- Indicates weakening selling pressure
- Breakout requires volume confirmation

#### 6. Neckline Breakout

- Resistance level defined by peak between troughs
- Break above neckline with tolerance (1.5-3%)
- Volume spike: 1.25-1.75x average pattern volume

## Configuration Parameters

### Strict Mode

```python
'PEAK_SIMILARITY_TOL': 0.06,    # Trough similarity tolerance
'MIN_PROMINENCE_PCT': 0.06,     # Minimum peak prominence
'MAX_SPACING_DAYS': 90,         # Maximum days between troughs
'MIN_SPACING_DAYS': 20,         # Minimum days between troughs
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
    Detects Double Bottom patterns in price data

    Pattern Type: 'double_top', 'double_bottom', or 'both'

    Returns: List of pattern dictionaries containing:
    - P1: First trough coordinates
    - T: Peak coordinates (neckline level)
    - P2: Second trough coordinates
    - breakout: Confirmed breakout point
    - neckline_level: Resistance level
    - duration: Pattern formation time
    """
```

### Main Detection Logic

```python
def detect_double_patterns(data, config, pattern_type='both', require_preceding_trend=True):
    """Unified detection function for both double tops and double bottoms."""

    results = []
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()

    # Step 1: Scan for 3-point sequences
    for i in range(len(swing_points_df) - 2):
        idx1, idx_mid, idx2 = swing_points_df['index'][i:i+3]

        # Step 2: Double Bottom Detection (Low → High → Low)
        if (pattern_type in ('both', 'double_bottom') and
            data.get('is_swing_low', False)[idx1] and
            data.get('is_swing_high', False)[idx_mid] and
            data.get('is_swing_low', False)[idx2]):

            # Extract coordinates
            p1_low, p1_date = data.loc[idx1, ['Low', 'Date']]
            t_high, t_date = data.loc[idx_mid, ['High', 'Date']]
            p2_low, p2_date = data.loc[idx2, ['Low', 'Date']]

            # Step 3: Apply Double Bottom validation rules
            pattern_valid, pattern_data = validate_double_bottom(
                data, idx1, idx_mid, idx2, p1_low, t_high, p2_low,
                p1_date, t_date, p2_date, config, require_preceding_trend)

            if pattern_valid:
                results.append(pattern_data)

    return results


def validate_double_bottom(data, idx1, idx_mid, idx2, p1_low, t_high, p2_low,
                          p1_date, t_date, p2_date, config, require_preceding_trend):
    """Comprehensive validation for Double Bottom patterns."""

    # Rule 1: Timing constraints
    try:
        spacing = (p2_date - p1_date).days
    except Exception:
        return False, None

    if (spacing < config['MIN_SPACING_DAYS'] or
        spacing > config['MAX_SPACING_DAYS']):
        return False, None

    # Rule 2: Preceding downtrend requirement
    if require_preceding_trend and not check_preceding_trend(data, idx1, 'down'):
        return False, None

    # Rule 3: Trough similarity validation
    trough_difference = abs(p1_low - p2_low) / max(p1_low, p2_low)
    if trough_difference > config['PEAK_SIMILARITY_TOL']:
        return False, None

    # Rule 4: Prominence validation (peak above troughs)
    try:
        prominence1 = (t_high - p1_low) / t_high
        prominence2 = (t_high - p2_low) / t_high

        if (prominence1 < config['MIN_PROMINENCE_PCT'] or
            prominence2 < config['MIN_PROMINENCE_PCT']):
            return False, None
    except Exception:
        return False, None

    # Rule 5: Volume divergence analysis (optional)
    volume_valid = validate_double_bottom_volume(data, idx1, idx_mid, idx2, config)
    if not volume_valid:
        return False, None

    # Rule 6: Neckline breakout confirmation
    neckline_level = t_high
    breakout_found, breakout_data = confirm_double_bottom_breakout(
        data, idx2, neckline_level, config)

    if breakout_found:
        pattern_data = {
            'type': 'double_bottom',
            'P1': (p1_date, p1_low, idx1),
            'T': (t_date, t_high, idx_mid),
            'P2': (p2_date, p2_low, idx2),
            'breakout': breakout_data,
            'neckline_level': neckline_level,
            'duration': spacing
        }
        return True, pattern_data

    return False, None
```

### Volume Pattern Analysis

```python
def validate_double_bottom_volume(data, idx1, idx_mid, idx2, config):
    """Analyze volume pattern for double bottom confirmation."""

    if 'Volume' not in data.columns:
        return True  # Skip if no volume data available

    try:
        # Calculate volume windows around each trough
        window_size = max(1, min(10, (idx_mid - idx1) // 2))

        # First trough volume
        vol1_start = max(0, idx1 - window_size)
        vol1_end = min(len(data), idx1 + window_size + 1)
        vol1 = data.iloc[vol1_start:vol1_end]['Volume'].mean()

        # Second trough volume
        vol2_start = max(0, idx2 - window_size)
        vol2_end = min(len(data), idx2 + window_size + 1)
        vol2 = data.iloc[vol2_start:vol2_end]['Volume'].mean()

        # Volume should decline on second trough (bullish divergence)
        # Lower selling pressure indicates potential reversal
        volume_divergence_valid = vol2 < vol1 * config['VOLUME_DECLINE_PCT']

        return volume_divergence_valid

    except Exception:
        return True  # Skip validation if calculation fails
```

### Breakout Confirmation Logic

```python
def confirm_double_bottom_breakout(data, pattern_end_idx, neckline_level, config):
    """Confirm breakout above resistance level with volume."""

    # Search for breakout within 90 days of pattern completion
    search_end = min(len(data), pattern_end_idx + 90)

    for j in range(pattern_end_idx + 1, search_end):
        close_price = data['Close'].iloc[j]

        # Check for breakout above neckline (with tolerance)
        breakout_threshold = neckline_level * (1 + config['NECKLINE_TOLERANCE'])

        if close_price > breakout_threshold:
            # Attempt volume confirmation
            volume_confirmed = confirm_double_bottom_volume_spike(
                data, pattern_end_idx, j, config)

            if volume_confirmed:
                breakout_data = (data['Date'].iloc[j], close_price, j)
                return True, breakout_data
            else:
                # Check for sustained breakout (multiple days above)
                sustained = count_sustained_breakout(data, j, neckline_level, 5)
                if sustained >= 2:
                    breakout_data = (data['Date'].iloc[j], close_price, j)
                    return True, breakout_data

    return False, None


def confirm_double_bottom_volume_spike(data, pattern_start, breakout_idx, config):
    """Confirm breakout with volume spike analysis."""

    try:
        # Calculate baseline volume during pattern formation
        baseline_volume = float(data.iloc[pattern_start:breakout_idx]['Volume'].mean())
        breakout_volume = float(data['Volume'].iloc[breakout_idx])

        # Require significant volume increase for confirmation
        volume_spike_confirmed = breakout_volume > baseline_volume * config['VOLUME_SPIKE_MULT']
        return volume_spike_confirmed

    except Exception:
        return False


def count_sustained_breakout(data, start_idx, neckline_level, look_ahead_days):
    """Count consecutive days with closes above neckline."""

    sustained_count = 0
    max_check = min(len(data), start_idx + look_ahead_days)

    for k in range(start_idx, max_check):
        if data['Close'].iloc[k] > neckline_level:
            sustained_count += 1
        else:
            break

    return sustained_count
```

### Support Level Analysis

```python
def analyze_double_bottom_support(p1_low, t_high, p2_low, p1_date, t_date, p2_date, config):
    """Analyze support level strength and pattern characteristics."""

    # Trough similarity analysis
    support_level = min(p1_low, p2_low)
    trough_avg = (p1_low + p2_low) / 2
    trough_variance = abs(p1_low - p2_low) / trough_avg
    support_consistency = 1 - trough_variance

    # Support test strength (how close troughs are to support level)
    support_test1 = p1_low / support_level
    support_test2 = p2_low / support_level
    avg_support_test = (support_test1 + support_test2) / 2

    # Bounce strength analysis
    bounce1 = (t_high - p1_low) / p1_low
    bounce2_estimate = (t_high - p2_low) / p2_low  # Potential bounce from second test
    avg_bounce_strength = (bounce1 + bounce2_estimate) / 2

    # Duration between support tests
    support_test_duration = (p2_date - p1_date).days

    support_analysis = {
        'support_level': support_level,
        'support_consistency': support_consistency,
        'average_support_test': avg_support_test,
        'average_bounce_strength': avg_bounce_strength,
        'test_duration': support_test_duration,
        'pattern_quality': calculate_double_bottom_quality(
            support_consistency, avg_bounce_strength, avg_support_test)
    }

    return support_analysis


def calculate_double_bottom_quality(consistency, bounce_strength, support_test):
    """Calculate overall double bottom pattern quality score."""

    # Weight different quality factors
    consistency_weight = 0.4    # How similar the troughs are
    bounce_weight = 0.4         # Strength of bounces from support
    support_weight = 0.2        # How well support level was tested

    # Normalize bounce strength (reasonable maximum of 20% bounce)
    normalized_bounce = min(bounce_strength / 0.20, 1.0)

    # Support test strength (closer to 1.0 is better)
    normalized_support = min(support_test, 1.0)

    quality_score = (
        consistency * consistency_weight +
        normalized_bounce * bounce_weight +
        normalized_support * support_weight
    )

    return min(quality_score, 1.0)
```

### Preceding Trend Validation

```python
def validate_preceding_downtrend(data, pattern_start_idx, lookback_period=90, min_decline=0.15):
    """Validate significant downtrend before double bottom formation."""

    if pattern_start_idx < lookback_period:
        return False

    try:
        # Look back 90 days from pattern start
        lookback_start = max(0, pattern_start_idx - lookback_period)
        start_price = float(data.iloc[lookback_start]['High'])  # Use high for conservative estimate
        pattern_start_price = float(data.iloc[pattern_start_idx]['Low'])  # Use low at pattern start

        # Calculate price decline
        price_decline = (start_price - pattern_start_price) / start_price

        # Require minimum decline to establish downtrend
        downtrend_established = price_decline >= min_decline

        return downtrend_established

    except Exception:
        return False
```

### Pattern Recognition Sequence

```python
def recognize_double_bottom_sequence(data, swing_points_df, i):
    """Extract and validate basic double bottom sequence structure."""

    # Get three consecutive swing points
    idx1, idx_mid, idx2 = swing_points_df['index'][i:i+3]

    # Verify Low → High → Low sequence
    is_valid_sequence = (
        data.get('is_swing_low', False)[idx1] and     # First trough
        data.get('is_swing_high', False)[idx_mid] and  # Peak between troughs
        data.get('is_swing_low', False)[idx2]          # Second trough
    )

    if not is_valid_sequence:
        return False, None

    # Extract price and date information
    p1_low, p1_date = data.loc[idx1, ['Low', 'Date']]
    t_high, t_date = data.loc[idx_mid, ['High', 'Date']]
    p2_low, p2_date = data.loc[idx2, ['Low', 'Date']]

    sequence_data = {
        'indices': (idx1, idx_mid, idx2),
        'prices': (p1_low, t_high, p2_low),
        'dates': (p1_date, t_date, p2_date)
    }

    return True, sequence_data
```

### Pattern Structure

Each detected Double Bottom pattern contains:

- **P1**: First trough (date, price, index)
- **T**: Peak between troughs (date, price, index)
- **P2**: Second trough (date, price, index)
- **breakout**: Confirmed breakout point (date, price, index)
- **neckline_level**: Resistance level (peak price)
- **duration**: Total pattern duration in days
- **type**: Pattern type identifier ('double_bottom')

### Algorithm Steps

1. **Scan Swing Points**: Identify Low → High → Low sequences
2. **Apply Timing Filters**: Check spacing between troughs
3. **Validate Similarities**: Ensure troughs are at similar levels
4. **Check Prominence**: Confirm meaningful peak height
5. **Analyze Volume**: Optional volume divergence check
6. **Confirm Breakout**: Validate breakout above neckline
7. **Record Pattern**: Store complete pattern data

## Breakout Validation

- **Price Criterion**: Close above `neckline * (1 + tolerance)`
- **Volume Criterion**: Volume spike above average or sustained break
- **Sustainability**: Look for 2+ consecutive days above neckline
- **Time Limit**: Search within 90 days of second trough

## Pattern Recognition Logic

```python
# Double Bottom detection sequence
if (data['is_swing_low'][idx1] and
    data['is_swing_high'][idx_mid] and
    data['is_swing_low'][idx2]):

    # Extract coordinates
    p1_low, p1_date = data.loc[idx1, ['Low', 'Date']]
    t_high, t_date = data.loc[idx_mid, ['High', 'Date']]
    p2_low, p2_date = data.loc[idx2, ['Low', 'Date']]

    # Apply validation rules
    # 1. Check timing constraints
    # 2. Validate trough similarity
    # 3. Confirm prominence
    # 4. Look for breakout above neckline
```

## Output

- **Charts**: PNG files with marked troughs, peak, and neckline
- **Data**: CSV reports with pattern coordinates and breakout timing
- **Folder Structure**: `outputs/charts/{SYMBOL}/{TIMEFRAME}/DB/`

## Usage Example

```python
# Configure for strict double bottom detection
config = DT_CONFIG['strict']  # Same config used for both patterns

# Detect only double bottoms
patterns = detect_double_patterns(data, config, pattern_type='double_bottom')

# Analyze results
for pattern in patterns:
    trough1_price = pattern['P1'][1]
    trough2_price = pattern['P2'][1]
    similarity = abs(trough1_price - trough2_price) / max(trough1_price, trough2_price)
    print(f"Trough similarity: {similarity:.1%}")
    print(f"Breakout at: {pattern['breakout'][1]:.2f}")
```

## Key Features

- **Support Level Identification**: Precisely identifies strong support zones
- **Reversal Confirmation**: Validates genuine trend reversal patterns
- **Volume Integration**: Confirms breakouts with volume analysis
- **Trend Context**: Requires preceding downtrend for pattern validity
- **Flexible Configuration**: Adjustable parameters for different market conditions

## Relationship to Double Top

- Double Bottom is the bullish counterpart to Double Top
- Uses same core algorithm with inverted logic
- Shared configuration parameters with different interpretation
- Both patterns detected in single `detect_double_patterns()` function
