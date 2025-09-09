import numpy as np
from math import atan, degrees


def _compute_neckline_from_troughs(t1_date, t1_low, t2_date, t2_low):
    """Return (slope, intercept) of the neckline based on two trough points."""
    try:
        x1, y1 = t1_date.toordinal(), float(t1_low)
        x2, y2 = t2_date.toordinal(), float(t2_low)
        if x2 == x1:
            return None, None
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept
    except Exception:
        return None, None


def validate_hns(df, pattern):
    """
    Validate a detected head-and-shoulders (HNS) pattern with a robust, score-based approach.

    Scoring (0-6):
      +2 Head sufficiently above both shoulders (>= 8% strong, >= 4% weak +1)
      +1 Shoulders height similarity (<= 12% diff)
      +1 Shoulders time symmetry (|L-R|/max(L,R) <= 0.35)
      +1 Neckline tilt reasonable (angle <= 35°)
      +1 Breakout closes below neckline at breakout
      +1 Volume spike on breakout (>= 1.3x recent avg; optional if Volume missing)

    Returns: {'score': int, 'is_valid': bool, 'details': {...}}
    """
    score = 0

    # Extract pattern points (defensive unpacking)
    try:
        p1_date, p1_high, p1_idx = pattern['P1']  # Left Shoulder
        p2_date, p2_high, p2_idx = pattern['P2']  # Head
        p3_date, p3_high, p3_idx = pattern['P3']  # Right Shoulder
        t1_date, t1_low, t1_idx = pattern['T1']   # Trough 1
        t2_date, t2_low, t2_idx = pattern['T2']   # Trough 2
    except Exception:
        return {'score': 0, 'is_valid': False, 'details': {'error': 'missing pattern points'}}

    breakout = pattern.get('breakout')

    # 1) Head prominence vs shoulders (scaled scoring)
    try:
        head_over_ls = (p2_high - p1_high) / p1_high if p1_high > 0 else 0.0
        head_over_rs = (p2_high - p3_high) / p3_high if p3_high > 0 else 0.0
        min_head_over = min(head_over_ls, head_over_rs)
        if min_head_over >= 0.08:
            score += 2
        elif min_head_over >= 0.04:
            score += 1
    except Exception:
        pass

    # 2) Shoulder height similarity (use max-based normalization)
    try:
        shoulder_height_diff = abs(p1_high - p3_high) / max(p1_high, p3_high)
        if shoulder_height_diff <= 0.12:
            score += 1
            shoulders_height_ok = True
        else:
            shoulders_height_ok = False
    except Exception:
        shoulders_height_ok = False

    # 3) Shoulder time symmetry (normalized by max arm length)
    try:
        left_span = max(1, (p2_date - p1_date).days)
        right_span = max(1, (p3_date - p2_date).days)
        time_asym_ratio = abs(left_span - right_span) / max(left_span, right_span)
        if time_asym_ratio <= 0.35:
            score += 1
            shoulders_time_ok = True
        else:
            shoulders_time_ok = False
    except Exception:
        shoulders_time_ok = False

    # 4) Neckline angle
    slope = pattern.get('neckline_slope')
    intercept = pattern.get('neckline_intercept')
    if slope is None or intercept is None:
        slope, intercept = _compute_neckline_from_troughs(t1_date, t1_low, t2_date, t2_low)
    try:
        if slope is not None:
            angle_deg = abs(degrees(atan(float(slope))))
            if angle_deg <= 35.0:
                score += 1
                angle_ok = True
            else:
                angle_ok = False
        else:
            angle_ok = False
    except Exception:
        angle_ok = False

    # 5) Breakout close vs neckline (use close at breakout index for consistency)
    breakout_ok = False
    vol_ok = None  # None means not evaluated (no Volume)
    try:
        if breakout and slope is not None and intercept is not None:
            breakout_date, breakout_price, breakout_idx = breakout
            # Prefer Close at breakout index if available
            if 0 <= int(breakout_idx) < len(df):
                px_j = float(df['Close'].iloc[breakout_idx]) if 'Close' in df.columns else float(breakout_price)
                date_j = df['Date'].iloc[breakout_idx] if 'Date' in df.columns else breakout_date
            else:
                px_j = float(breakout_price)
                date_j = breakout_date

            neckline_at_j = float(slope) * date_j.toordinal() + float(intercept)
            if px_j < neckline_at_j:
                score += 1
                breakout_ok = True

            # 6) Volume spike (adaptive window)
            if 'Volume' in df.columns and isinstance(breakout_idx, (int, np.integer)):
                window = max(10, min(30, (int(breakout_idx) - int(p1_idx)) // 2 if int(breakout_idx) > int(p1_idx) else 20))
                if int(breakout_idx) >= window:
                    avg_vol = float(df['Volume'].iloc[int(breakout_idx)-window:int(breakout_idx)].mean())
                    vol_j = float(df['Volume'].iloc[int(breakout_idx)])
                    if avg_vol > 0 and vol_j >= 1.3 * avg_vol:
                        score += 1
                        vol_ok = True
                    else:
                        vol_ok = False
            else:
                vol_ok = None  # not evaluated
    except Exception:
        breakout_ok = False
        # keep vol_ok as is (None/False)

    # Valid if we pass most checks. Be slightly lenient to avoid over-rejection.
    # If Volume not available/evaluated, require score >= 4 out of 5; otherwise >= 4 out of 6.
    min_required = 4
    is_valid = score >= min_required

    return {
        'score': int(score),
        'is_valid': bool(is_valid),
        'details': {
            'head_over_min': float(min_head_over) if 'min_head_over' in locals() else None,
            'shoulders_height_diff': float(shoulder_height_diff) if 'shoulder_height_diff' in locals() else None,
            'time_asym_ratio': float(time_asym_ratio) if 'time_asym_ratio' in locals() else None,
            'neckline_angle_deg': float(angle_deg) if 'angle_deg' in locals() else None,
            'breakout_ok': bool(breakout_ok),
            'volume_ok': None if vol_ok is None else bool(vol_ok),
            'shoulders_height_ok': bool(shoulders_height_ok) if 'shoulders_height_ok' in locals() else None,
            'shoulders_time_ok': bool(shoulders_time_ok) if 'shoulders_time_ok' in locals() else None,
            'neckline_angle_ok': bool(angle_ok) if 'angle_ok' in locals() else None,
        }
    }
