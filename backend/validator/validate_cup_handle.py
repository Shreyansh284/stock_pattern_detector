import pandas as pd

def validate_cup_handle(df, pattern):
    """
    Validate a detected cup-and-handle pattern.
    Returns: {'score': int, 'is_valid': bool}
    """
    score = 0

    # Extract pattern points
    left_rim_date, left_rim_high, _ = pattern['left_rim']
    cup_bottom_date, cup_bottom_low, _ = pattern['cup_bottom']
    right_rim_date, right_rim_high, _ = pattern['right_rim']
    handle = pattern.get('handle_low', None)
    breakout = pattern.get('breakout', None)

    # Rim symmetry (±10%)
    rim_avg = (left_rim_high + right_rim_high) / 2
    rim_diff = abs(left_rim_high - right_rim_high) / rim_avg
    if rim_diff <= 0.10:
        score += 1

    # Cup depth (15–35% of rim avg)
    cup_depth = (rim_avg - cup_bottom_low) / rim_avg
    if 0.15 <= cup_depth <= 0.35:
        score += 2

    # Handle checks
    if handle:
        handle_date, handle_low, _ = handle
        # Handle forms after right rim
        if handle_date > right_rim_date:
            score += 1
        # Handle depth <15% of rim avg
        handle_depth = (right_rim_high - handle_low) / rim_avg
        if handle_depth < 0.15:
            score += 1
        # Handle shorter than 1/3 cup length
        cup_length = (right_rim_date - left_rim_date).days
        handle_length = (handle_date - right_rim_date).days
        if cup_length > 0 and handle_length < (cup_length / 3):
            score += 1

    # Breakout checks
    if breakout:
        breakout_date, breakout_price, breakout_idx = breakout
        # Breakout above right rim
        if breakout_price > right_rim_high:
            score += 1
        # Breakout volume > 1.5× avg of last 20 candles
        if 'Volume' in df.columns and breakout_idx >= 20:
            avg_vol = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
            if df['Volume'].iloc[breakout_idx] > 1.5 * avg_vol:
                score += 1

    return {'score': score, 'is_valid': score >= 6}
