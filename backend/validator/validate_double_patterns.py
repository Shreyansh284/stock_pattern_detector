import pandas as pd
import numpy as np

def validate_double_top(df, pattern):
    """
    Validate a detected double top pattern.
    Returns: {'score': int, 'is_valid': bool}
    """
    score = 0
    
    # Extract pattern points
    p1_date, p1_price, _ = pattern['P1']  # First top
    t_date, t_price, _ = pattern['T']     # Valley/trough
    p2_date, p2_price, _ = pattern['P2']  # Second top
    breakout = pattern.get('breakout', None)
    neckline_level = pattern.get('neckline_level', None)
    
    # 1. Tops roughly equal height (within 3% of each other)
    higher_top = max(p1_price, p2_price)
    lower_top = min(p1_price, p2_price)
    height_diff = abs(p1_price - p2_price) / higher_top
    if height_diff <= 0.03:
        score += 2
    elif height_diff <= 0.05:
        score += 1
    
    # 2. Valley significantly lower than tops (at least 10% below)
    valley_depth = (higher_top - t_price) / higher_top
    if valley_depth >= 0.15:
        score += 2
    elif valley_depth >= 0.10:
        score += 1
    
    # 3. Time symmetry - reasonably balanced timing
    time_to_valley = (t_date - p1_date).days
    time_from_valley = (p2_date - t_date).days
    if time_to_valley > 0 and time_from_valley > 0:
        time_ratio = min(time_to_valley, time_from_valley) / max(time_to_valley, time_from_valley)
        if time_ratio >= 0.7:  # Within 30% of each other
            score += 1
    
    # 4. Sufficient pattern duration (at least 20 days)
    pattern_duration = (p2_date - p1_date).days
    if pattern_duration >= 20:
        score += 1
    
    # 5. Breakout validation
    if breakout and neckline_level is not None:
        breakout_date, breakout_price, breakout_idx = breakout
        
        # Breakout below neckline (for double top)
        if breakout_price < neckline_level:
            score += 1
            
            # Significant breakout (at least 2% below neckline)
            breakout_depth = (neckline_level - breakout_price) / neckline_level
            if breakout_depth >= 0.02:
                score += 1
        
        # Volume spike on breakout
        if 'Volume' in df.columns and breakout_idx >= 20:
            avg_vol = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
            if df['Volume'].iloc[breakout_idx] > 1.5 * avg_vol:
                score += 1
    
    # 6. Price action before pattern (uptrend leading to first top)
    if len(df) > 30:
        pre_pattern_start = max(0, df.index.get_loc(df[df['Date'] == p1_date].index[0]) - 30)
        pre_pattern_end = df.index.get_loc(df[df['Date'] == p1_date].index[0])
        if pre_pattern_end > pre_pattern_start:
            pre_price = df['Close'].iloc[pre_pattern_start]
            if p1_price > pre_price * 1.10:  # At least 10% rise before pattern
                score += 1
    
    return {'score': score, 'is_valid': score >= 5}


def validate_double_bottom(df, pattern):
    """
    Validate a detected double bottom pattern.
    Returns: {'score': int, 'is_valid': bool}
    """
    score = 0
    
    # Extract pattern points
    p1_date, p1_price, _ = pattern['P1']  # First bottom
    t_date, t_price, _ = pattern['T']     # Peak/top
    p2_date, p2_price, _ = pattern['P2']  # Second bottom
    breakout = pattern.get('breakout', None)
    neckline_level = pattern.get('neckline_level', None)
    
    # 1. Bottoms roughly equal depth (within 3% of each other)
    lower_bottom = min(p1_price, p2_price)
    higher_bottom = max(p1_price, p2_price)
    height_diff = abs(p1_price - p2_price) / lower_bottom
    if height_diff <= 0.03:
        score += 2
    elif height_diff <= 0.05:
        score += 1
    
    # 2. Peak significantly higher than bottoms (at least 10% above)
    peak_height = (t_price - lower_bottom) / lower_bottom
    if peak_height >= 0.15:
        score += 2
    elif peak_height >= 0.10:
        score += 1
    
    # 3. Time symmetry - reasonably balanced timing
    time_to_peak = (t_date - p1_date).days
    time_from_peak = (p2_date - t_date).days
    if time_to_peak > 0 and time_from_peak > 0:
        time_ratio = min(time_to_peak, time_from_peak) / max(time_to_peak, time_from_peak)
        if time_ratio >= 0.7:  # Within 30% of each other
            score += 1
    
    # 4. Sufficient pattern duration (at least 20 days)
    pattern_duration = (p2_date - p1_date).days
    if pattern_duration >= 20:
        score += 1
    
    # 5. Breakout validation
    if breakout and neckline_level is not None:
        breakout_date, breakout_price, breakout_idx = breakout
        
        # Breakout above neckline (for double bottom)
        if breakout_price > neckline_level:
            score += 1
            
            # Significant breakout (at least 2% above neckline)
            breakout_height = (breakout_price - neckline_level) / neckline_level
            if breakout_height >= 0.02:
                score += 1
        
        # Volume spike on breakout
        if 'Volume' in df.columns and breakout_idx >= 20:
            avg_vol = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
            if df['Volume'].iloc[breakout_idx] > 1.5 * avg_vol:
                score += 1
    
    # 6. Price action before pattern (downtrend leading to first bottom)
    if len(df) > 30:
        pre_pattern_start = max(0, df.index.get_loc(df[df['Date'] == p1_date].index[0]) - 30)
        pre_pattern_end = df.index.get_loc(df[df['Date'] == p1_date].index[0])
        if pre_pattern_end > pre_pattern_start:
            pre_price = df['Close'].iloc[pre_pattern_start]
            if p1_price < pre_price * 0.90:  # At least 10% decline before pattern
                score += 1
    
    return {'score': int(score), 'is_valid': score >= 5}


def validate_double_pattern(df, pattern):
    """
    Main validation function that routes to appropriate validator based on pattern type.
    """
    pattern_type = pattern.get('type', '')
    
    if pattern_type == 'double_top':
        return validate_double_top(df, pattern)
    elif pattern_type == 'double_bottom':
        return validate_double_bottom(df, pattern)
    else:
        return {'score': 0, 'is_valid': False}
