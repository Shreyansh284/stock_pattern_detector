import pandas as pd

def validate_hns(df, pattern):
    """
    Validate a detected head-and-shoulders pattern.
    Returns: {'score': int, 'is_valid': bool}
    """
    score = 0
    # Extract pattern points
    p1_date, p1_high, _ = pattern['P1']  # Left Shoulder
    p2_date, p2_high, _ = pattern['P2']  # Head
    p3_date, p3_high, _ = pattern['P3']  # Right Shoulder
    t1_date, t1_low, _ = pattern['T1']   # Trough 1
    t2_date, t2_low, _ = pattern['T2']   # Trough 2
    breakout = pattern.get('breakout', None)
    # 1. Head at least 10% higher than both shoulders
    higher_shoulder = max(p1_high, p3_high)
    if p2_high > p1_high * 1.10 and p2_high > p3_high * 1.10:
        score += 2
    # 2. Shoulders roughly symmetrical (height diff <10%, time diff <±30%)
    shoulder_height_diff = abs(p1_high - p3_high) / ((p1_high + p3_high) / 2)
    shoulder_time_diff = abs((p2_date - p1_date).days - (p3_date - p2_date).days)
    total_span = (p3_date - p1_date).days
    if shoulder_height_diff < 0.10:
        score += 1
    if total_span > 0 and shoulder_time_diff / total_span < 0.3:
        score += 1
    # 3. Neckline relatively flat or slightly sloping (angle < 20 deg)
    slope = pattern.get('neckline_slope', None)
    if slope is not None:
        import numpy as np
        angle_deg = abs(np.degrees(np.arctan(slope)))
        if angle_deg < 20:
            score += 1
    # 4. Breakout below neckline with volume spike
    if breakout:
        breakout_date, breakout_price, breakout_idx = breakout
        # Neckline at breakout
        intercept = pattern.get('neckline_intercept', None)
        if slope is not None and intercept is not None:
            neckline_at_breakout = slope * breakout_date.toordinal() + intercept
            if breakout_price < neckline_at_breakout:
                score += 1
            # Volume spike
            if 'Volume' in df.columns and breakout_idx >= 20:
                avg_vol = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
                if df['Volume'].iloc[breakout_idx] > 1.5 * avg_vol:
                    score += 1
    return {'score': score, 'is_valid': score >= 5}
