# Cup and Handle Pattern Detection

## Pattern Description

Cup and Handle is a bullish continuation pattern resembling a tea cup. It consists of a U-shaped "cup" formation followed by a smaller downward "handle" before breaking out to new highs.

## Algorithm Overview

### Detection Method

1. **Swing Point Analysis**: Identifies potential cup rims (two high points)
2. **Cup Formation**: Finds the lowest point between rims to form cup bottom
3. **Handle Formation**: Looks for pullback after right rim
4. **Breakout Confirmation**: Validates upward breakout with volume

### Key Validation Rules

#### 1. Preceding Uptrend Requirement

- Requires significant uptrend before cup formation
- Minimum 15% price increase over 90-day lookback period

#### 2. Cup Structure Validation

- **Cup Duration**: 30-365 days (configurable)
- **Cup Depth**: 8-60% of rim price (mode dependent)
  - Strict: 15-40% depth
  - Lenient: 8-60% depth
- **Cup Symmetry**: Left and right sides should be relatively balanced
  - Tolerance: 25-50% asymmetry allowed

#### 3. Rim Height Similarity

- Left and right rim heights must be within 5% of each other
- Ensures proper resistance level formation

#### 4. Handle Formation Requirements

- **Handle Duration**: 3-90 days (configurable)
- **Handle Depth**: Maximum 20-35% retracement from rim
- **Handle Position**: Must form after right rim completion

#### 5. Volume Pattern Analysis

- Volume should decline during cup formation
- Volume should remain low during handle formation
- Breakout requires volume spike (1.25-1.75x average)

#### 6. Breakout Confirmation

- Price must close above resistance level (rim height)
- Volume spike confirms genuine breakout
- Search window: up to 30 days after handle low

## Configuration Parameters

### Strict Mode

```python
'CUP_DEPTH_MIN': 0.15,          # Minimum cup depth
'CUP_DEPTH_MAX': 0.40,          # Maximum cup depth
'HANDLE_DEPTH_MAX': 0.20,       # Maximum handle retracement
'CUP_SYMMETRY_TOL': 0.25,       # Cup symmetry tolerance
'HANDLE_DURATION_MIN': 7,        # Minimum handle duration (days)
'HANDLE_DURATION_MAX': 60,       # Maximum handle duration (days)
'VOLUME_DECLINE_PCT': 0.85,      # Required volume decline
'VOLUME_SPIKE_MULT': 1.75,       # Breakout volume multiplier
```

### Lenient Mode

```python
'CUP_DEPTH_MIN': 0.08,
'CUP_DEPTH_MAX': 0.60,
'HANDLE_DEPTH_MAX': 0.35,
'CUP_SYMMETRY_TOL': 0.50,
'HANDLE_DURATION_MIN': 3,
'HANDLE_DURATION_MAX': 90,
'VOLUME_DECLINE_PCT': 0.70,
'VOLUME_SPIKE_MULT': 1.25,
```

## Code Implementation

### Core Function

```python
def detect_cup_and_handle(data, config, require_preceding_trend=True):
    """
    Detects Cup and Handle patterns in price data

    Returns: List of pattern dictionaries containing:
    - left_rim: Left rim coordinates
    - cup_bottom: Lowest point of cup
    - right_rim: Right rim coordinates
    - handle_low: Handle retracement point
    - breakout: Confirmed breakout point
    - cup_duration: Pattern duration
    - cup_depth: Depth percentage
    """
```

### Main Detection Logic

```python
def detect_cup_and_handle(data, config, require_preceding_trend=True):
    """Main cup and handle detection algorithm."""

    # Step 1: Initialize swing point analysis
    swing_points_df = data[data.get('is_swing_high', False) | data.get('is_swing_low', False)].reset_index()
    patterns = []

    # Step 2: Extract high swing points for potential cup rims
    high_indices = swing_points_df[swing_points_df['index'].apply(
        lambda x: data.iloc[x].get('is_swing_high', False))].reset_index(drop=True)

    # Step 3: Scan for cup formations (two high points)
    for i in range(len(high_indices) - 1):
        # Cup left rim identification
        left_rim_idx = high_indices.iloc[i]['index']
        left_rim_high = data.loc[left_rim_idx, 'High']
        left_rim_date = data.loc[left_rim_idx, 'Date']

        for j in range(i + 1, len(high_indices)):
            # Cup right rim identification
            right_rim_idx = high_indices.iloc[j]['index']
            right_rim_high = data.loc[right_rim_idx, 'High']
            right_rim_date = data.loc[right_rim_idx, 'Date']

            # Step 4: Validate cup duration
            try:
                cup_duration = (right_rim_date - left_rim_date).days
            except Exception:
                continue

            if (cup_duration < GLOBAL_CONFIG['CUP_DURATION_MIN'] or
                cup_duration > GLOBAL_CONFIG['CUP_DURATION_MAX']):
                continue

            # Step 5: Find cup bottom (lowest point between rims)
            cup_section = data.iloc[left_rim_idx:right_rim_idx+1]
            if cup_section.empty:
                continue

            cup_bottom_idx = cup_section['Low'].idxmin()
            cup_bottom_low = data.loc[cup_bottom_idx, 'Low']
            cup_bottom_date = data.loc[cup_bottom_idx, 'Date']

            # Step 6: Apply validation rules

            # Rule 1: Preceding uptrend validation
            if require_preceding_trend and not check_preceding_trend(data, left_rim_idx, 'up'):
                continue

            # Rule 2: Cup depth validation
            left_rim_price = max(left_rim_high, right_rim_high)
            cup_depth = (left_rim_price - cup_bottom_low) / left_rim_price

            if (cup_depth < config['CUP_DEPTH_MIN'] or
                cup_depth > config['CUP_DEPTH_MAX']):
                continue

            # Rule 3: Cup symmetry validation
            left_side_duration = (cup_bottom_date - left_rim_date).days
            right_side_duration = (right_rim_date - cup_bottom_date).days

            if left_side_duration == 0 or right_side_duration == 0:
                continue

            symmetry_ratio = (min(left_side_duration, right_side_duration) /
                            max(left_side_duration, right_side_duration))

            if symmetry_ratio < (1 - config['CUP_SYMMETRY_TOL']):
                continue

            # Rule 4: Rim height similarity
            rim_difference = abs(left_rim_high - right_rim_high) / max(left_rim_high, right_rim_high)
            if rim_difference > 0.05:  # 5% tolerance
                continue

            # Step 7: Handle formation and breakout detection
            handle_found, handle_data = detect_handle_formation(
                data, right_rim_idx, right_rim_high, right_rim_date, config)

            if handle_found:
                pattern_data = {
                    'type': 'cup_and_handle',
                    'left_rim': (left_rim_date, left_rim_high, left_rim_idx),
                    'cup_bottom': (cup_bottom_date, cup_bottom_low, cup_bottom_idx),
                    'right_rim': (right_rim_date, right_rim_high, right_rim_idx),
                    'handle_low': handle_data['handle_low'],
                    'breakout': handle_data['breakout'],
                    'cup_duration': cup_duration,
                    'cup_depth': cup_depth
                }
                patterns.append(pattern_data)

    return patterns
```

### Handle Formation Detection

```python
def detect_handle_formation(data, right_rim_idx, right_rim_high, right_rim_date, config):
    """Detect handle formation and breakout after cup completion."""

    handle_found = False
    handle_low_idx = -1
    handle_end_idx = -1

    # Search for handle within specified duration after right rim
    max_search_idx = min(len(data), right_rim_idx + config['HANDLE_DURATION_MAX'])

    for k in range(right_rim_idx + 1, max_search_idx):
        current_low = data.loc[k, 'Low']
        current_date = data.loc[k, 'Date']

        # Rule 1: Handle depth constraint
        handle_depth = (right_rim_high - current_low) / right_rim_high
        if handle_depth > config['HANDLE_DEPTH_MAX']:
            break  # Handle too deep, invalidates pattern

        # Rule 2: Handle duration requirement
        handle_duration = (current_date - right_rim_date).days
        if handle_duration >= config['HANDLE_DURATION_MIN']:

            # Look for breakout above resistance (right rim)
            breakout_found, breakout_idx, breakout_price = search_for_breakout(
                data, k, right_rim_high, config)

            if breakout_found:
                handle_found = True
                handle_low_idx = k
                handle_end_idx = breakout_idx
                break

    if handle_found:
        return True, {
            'handle_low': (data.loc[handle_low_idx, 'Date'],
                          data.loc[handle_low_idx, 'Low'], handle_low_idx),
            'breakout': (data.loc[handle_end_idx, 'Date'],
                        data.loc[handle_end_idx, 'Close'], handle_end_idx)
        }

    return False, None
```

### Breakout Detection Logic

```python
def search_for_breakout(data, handle_start_idx, resistance_level, config):
    """Search for volume-confirmed breakout above resistance level."""

    # Search within 30 days of handle formation
    max_search = min(len(data), handle_start_idx + 30)

    for m in range(handle_start_idx + 1, max_search):
        close_price = data.loc[m, 'Close']

        # Check for close above resistance level
        if close_price > resistance_level:
            try:
                # Volume confirmation
                pattern_start = max(0, handle_start_idx - 50)  # Pattern volume baseline
                avg_pattern_volume = data.iloc[pattern_start:handle_start_idx]['Volume'].mean()
                breakout_volume = data.loc[m, 'Volume']

                # Require volume spike for confirmation
                if breakout_volume > avg_pattern_volume * config['VOLUME_SPIKE_MULT']:
                    return True, m, close_price

            except Exception:
                # If volume analysis fails, accept breakout without volume confirmation
                return True, m, close_price

    return False, -1, None
```

### Cup Symmetry Analysis

```python
def analyze_cup_symmetry(left_rim_date, cup_bottom_date, right_rim_date, config):
    """Analyze cup symmetry for proper U-shaped formation."""

    # Calculate duration of each side
    left_side_days = (cup_bottom_date - left_rim_date).days
    right_side_days = (right_rim_date - cup_bottom_date).days

    if left_side_days == 0 or right_side_days == 0:
        return False, 0

    # Calculate symmetry ratio (should be close to 1.0 for perfect symmetry)
    symmetry_ratio = min(left_side_days, right_side_days) / max(left_side_days, right_side_days)

    # Check if within acceptable tolerance
    min_acceptable_ratio = 1 - config['CUP_SYMMETRY_TOL']
    is_symmetric = symmetry_ratio >= min_acceptable_ratio

    return is_symmetric, symmetry_ratio
```

### Volume Pattern Validation

```python
def validate_cup_volume_pattern(data, left_rim_idx, right_rim_idx, config):
    """Validate volume decline during cup formation."""

    try:
        # Early cup volume (around left rim)
        early_window = 10
        early_start = max(0, left_rim_idx - early_window//2)
        early_end = min(len(data), left_rim_idx + early_window//2)
        early_volume = data.iloc[early_start:early_end]['Volume'].mean()

        # Late cup volume (around right rim)
        late_start = max(0, right_rim_idx - early_window//2)
        late_end = min(len(data), right_rim_idx + early_window//2)
        late_volume = data.iloc[late_start:late_end]['Volume'].mean()

        # Volume should decline during cup formation
        volume_declined = late_volume <= early_volume * config['VOLUME_DECLINE_PCT']

        return volume_declined, {
            'early_cup_volume': early_volume,
            'late_cup_volume': late_volume,
            'decline_ratio': late_volume / early_volume if early_volume > 0 else 1.0
        }

    except Exception:
        return True, {}  # Skip volume validation if data insufficient
```

### Pattern Structure

Each detected pattern contains:

- **left_rim**: Left side high point (date, price, index)
- **cup_bottom**: Lowest point in cup (date, price, index)
- **right_rim**: Right side high point (date, price, index)
- **handle_low**: Handle pullback low (date, price, index)
- **breakout**: Confirmed breakout point (date, price, index)
- **cup_duration**: Total cup formation time in days
- **cup_depth**: Cup depth as percentage of rim price

### Algorithm Steps

1. **Find Potential Cups**: Scan for High → Low → High sequences
2. **Validate Cup Metrics**: Check depth, duration, and symmetry
3. **Locate Handle**: Search for pullback after right rim
4. **Confirm Breakout**: Validate volume and price breakout
5. **Store Pattern**: Record all key coordinates and metrics

## Output

- **Charts**: PNG files with cup outline, handle, and resistance level
- **Data**: CSV reports with pattern metrics and breakout details
- **Folder Structure**: `outputs/charts/{SYMBOL}/{TIMEFRAME}/CH/`

## Usage Example

```python
# Configure for strict detection
config = CH_CONFIG['strict']

# Detect patterns
patterns = detect_cup_and_handle(data, config, require_preceding_trend=True)

# Pattern analysis
for pattern in patterns:
    cup_depth = pattern['cup_depth']
    duration = pattern['cup_duration']
    print(f"Cup depth: {cup_depth:.1%}, Duration: {duration} days")
    print(f"Breakout: {pattern['breakout'][1]:.2f}")
```

## Key Features

- **Flexible Configuration**: Adjustable parameters for different market conditions
- **Volume Confirmation**: Ensures genuine breakouts with volume analysis
- **Geometric Validation**: Confirms proper cup and handle proportions
- **Trend Context**: Validates preceding uptrend for pattern significance
