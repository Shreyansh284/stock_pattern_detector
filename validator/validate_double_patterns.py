import pandas as pd
import numpy as np

def check_multi_timeframe_confirmation(df, pattern, weekly_data=None):
    """
    Optional multi-timeframe confirmation for double top patterns.
    Returns: {'weekly_confirmed': bool, 'confidence_boost': float}
    """
    if weekly_data is None or len(weekly_data) < 10:
        return {'weekly_confirmed': False, 'confidence_boost': 0.0}
    
    try:
        # Extract pattern dates
        p1_date, _, _ = pattern['P1']
        p2_date, _, _ = pattern['P2']
        
        # Convert to weekly timeframe dates (approximate)
        weekly_p1 = weekly_data[weekly_data['Date'] <= p1_date].iloc[-1] if len(weekly_data[weekly_data['Date'] <= p1_date]) > 0 else None
        weekly_p2 = weekly_data[weekly_data['Date'] <= p2_date].iloc[-1] if len(weekly_data[weekly_data['Date'] <= p2_date]) > 0 else None
        
        if weekly_p1 is None or weekly_p2 is None:
            return {'weekly_confirmed': False, 'confidence_boost': 0.0}
        
        # Simple check: if weekly chart also shows similar double top structure
        weekly_high_diff = abs(weekly_p1['High'] - weekly_p2['High']) / max(weekly_p1['High'], weekly_p2['High'])
        
        # If weekly tops are also similar (within 5%), consider it confirmed
        weekly_confirmed = weekly_high_diff <= 0.05
        confidence_boost = 0.1 if weekly_confirmed else 0.0
        
        return {'weekly_confirmed': weekly_confirmed, 'confidence_boost': confidence_boost}
    
    except Exception:
        return {'weekly_confirmed': False, 'confidence_boost': 0.0}

def validate_double_top(df, pattern):
    """
    Validate a detected double top pattern with enhanced robust rules.
    Returns: {'score': int, 'is_valid': bool, 'confidence': float}
    """
    score = 0
    max_score = 7  # Updated for new scoring system
    
    # Extract pattern points
    p1_date, p1_price, _ = pattern['P1']  # First top
    t_date, t_price, _ = pattern['T']     # Valley/trough
    p2_date, p2_price, _ = pattern['P2']  # Second top
    breakout = pattern.get('breakout', None)
    neckline_level = pattern.get('neckline_level', None)
    
    # Get pattern indices for easier access
    try:
        p1_idx = df[df['Date'] == p1_date].index[0]
        p2_idx = df[df['Date'] == p2_date].index[0]
    except (IndexError, KeyError):
        return {'score': 0, 'is_valid': False, 'confidence': 0.0}
    
    # 1. Tops roughly equal height (within 3% of each other)
    higher_top = max(p1_price, p2_price)
    lower_top = min(p1_price, p2_price)
    height_diff = abs(p1_price - p2_price) / higher_top
    if height_diff <= 0.03:
        score += 1
    
    # 2. Valley significantly lower than tops (at least 10% below)
    valley_depth = (higher_top - t_price) / higher_top
    if valley_depth >= 0.10:
        score += 1
    
    # 3. Time between tops (20-90 days for ideal spacing)
    pattern_duration = (p2_date - p1_date).days
    if 20 <= pattern_duration <= 90:
        score += 1
    
    # 4. Enhanced preceding uptrend validation (60 days, 15% minimum)
    if len(df) > 60:
        pre_pattern_start = max(0, p1_idx - 60)
        pre_pattern_end = p1_idx
        if pre_pattern_end > pre_pattern_start:
            pre_price = df['Close'].iloc[pre_pattern_start]
            uptrend_gain = (p1_price - pre_price) / pre_price
            if uptrend_gain >= 0.15:  # At least 15% rise before pattern
                score += 1
    
    # 5. Enhanced breakout confirmation
    if breakout and neckline_level is not None:
        breakout_date, breakout_price, breakout_idx = breakout
        
        # Breakout 2-3% below neckline for confirmation
        breakout_depth = (neckline_level - breakout_price) / neckline_level
        if breakout_depth >= 0.02:  # At least 2% below neckline
            score += 1
    
    # 6. Enhanced volume spike validation (2x average instead of 1.5x)
    if breakout and 'Volume' in df.columns and breakout_idx >= 20:
        avg_vol = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
        if df['Volume'].iloc[breakout_idx] > 2.0 * avg_vol:  # 2x volume spike
            score += 1
    
    # 7. Support level check (avoid if strong support within 3-5% below neckline)
    if neckline_level is not None:
        support_zone_lower = neckline_level * 0.95  # 5% below neckline
        support_zone_upper = neckline_level * 0.97  # 3% below neckline
        
        # Check for significant support in the lookback period
        lookback_period = min(60, len(df) - p2_idx - 1)
        if lookback_period > 0:
            recent_lows = df['Low'].iloc[p2_idx:p2_idx + lookback_period]
            support_touches = ((recent_lows >= support_zone_lower) & 
                             (recent_lows <= support_zone_upper)).sum()
            
            # If no significant support level found in the danger zone, award point
            if support_touches <= 2:  # Allow minor touches but not strong support
                score += 1
    
    # Calculate confidence score (percentage of criteria met)
    confidence = score / max_score
    
    # Pattern is valid if confidence score >= 0.7 (5 out of 7 criteria)
    is_valid = confidence >= 0.71  # 5/7 = 0.714
    
    return {
        'score': score, 
        'is_valid': is_valid, 
        'confidence': round(confidence, 3),
        'max_score': max_score
    }


def validate_double_bottom(df, pattern):
    """
    Validate a detected double bottom pattern with enhanced robust rules.
    Returns: {'score': int, 'is_valid': bool, 'confidence': float}
    """
    score = 0
    max_score = 7  # Updated for new scoring system
    
    # Extract pattern points
    p1_date, p1_price, _ = pattern['P1']  # First bottom
    t_date, t_price, _ = pattern['T']     # Peak/top
    p2_date, p2_price, _ = pattern['P2']  # Second bottom
    breakout = pattern.get('breakout', None)
    neckline_level = pattern.get('neckline_level', None)
    
    # Get pattern indices for easier access
    try:
        p1_idx = df[df['Date'] == p1_date].index[0]
        p2_idx = df[df['Date'] == p2_date].index[0]
    except (IndexError, KeyError):
        return {'score': 0, 'is_valid': False, 'confidence': 0.0}
    
    # 1. Bottoms roughly equal depth (within 3% of each other)
    lower_bottom = min(p1_price, p2_price)
    higher_bottom = max(p1_price, p2_price)
    height_diff = abs(p1_price - p2_price) / lower_bottom
    if height_diff <= 0.03:
        score += 1
    
    # 2. Peak significantly higher than bottoms (at least 10% above)
    peak_height = (t_price - lower_bottom) / lower_bottom
    if peak_height >= 0.10:
        score += 1
    
    # 3. Time between bottoms (20-90 days for ideal spacing)
    pattern_duration = (p2_date - p1_date).days
    if 20 <= pattern_duration <= 90:
        score += 1
    
    # 4. Enhanced preceding downtrend validation (60 days, 15% minimum)
    if len(df) > 60:
        pre_pattern_start = max(0, p1_idx - 60)
        pre_pattern_end = p1_idx
        if pre_pattern_end > pre_pattern_start:
            pre_price = df['Close'].iloc[pre_pattern_start]
            downtrend_decline = (pre_price - p1_price) / pre_price
            if downtrend_decline >= 0.15:  # At least 15% decline before pattern
                score += 1
    
    # 5. Enhanced breakout confirmation
    if breakout and neckline_level is not None:
        breakout_date, breakout_price, breakout_idx = breakout
        
        # Breakout 2-3% above neckline for confirmation
        breakout_height = (breakout_price - neckline_level) / neckline_level
        if breakout_height >= 0.02:  # At least 2% above neckline
            score += 1
    
    # 6. Enhanced volume spike validation (2x average instead of 1.5x)
    if breakout and 'Volume' in df.columns and breakout_idx >= 20:
        avg_vol = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
        if df['Volume'].iloc[breakout_idx] > 2.0 * avg_vol:  # 2x volume spike
            score += 1
    
    # 7. Resistance level check (avoid if strong resistance within 3-5% above neckline)
    if neckline_level is not None:
        resistance_zone_lower = neckline_level * 1.03  # 3% above neckline
        resistance_zone_upper = neckline_level * 1.05  # 5% above neckline
        
        # Check for significant resistance in the lookback period
        lookback_period = min(60, len(df) - p2_idx - 1)
        if lookback_period > 0:
            recent_highs = df['High'].iloc[p2_idx:p2_idx + lookback_period]
            resistance_touches = ((recent_highs >= resistance_zone_lower) & 
                                (recent_highs <= resistance_zone_upper)).sum()
            
            # If no significant resistance level found in the danger zone, award point
            if resistance_touches <= 2:  # Allow minor touches but not strong resistance
                score += 1
    
    # Calculate confidence score (percentage of criteria met)
    confidence = score / max_score
    
    # Pattern is valid if confidence score >= 0.7 (5 out of 7 criteria)
    is_valid = confidence >= 0.71  # 5/7 = 0.714
    
    return {
        'score': score, 
        'is_valid': is_valid, 
        'confidence': round(confidence, 3),
        'max_score': max_score
    }


def validate_double_pattern(df, pattern, weekly_data=None):
    """
    Main validation function that routes to appropriate validator based on pattern type.
    Enhanced with multi-timeframe confirmation and robust scoring system.
    
    Args:
        df: Daily price data DataFrame
        pattern: Pattern dictionary containing pattern details
        weekly_data: Optional weekly data for multi-timeframe confirmation
    
    Returns:
        dict: Enhanced validation results with confidence scoring
    """
    pattern_type = pattern.get('type', '')
    
    if pattern_type == 'double_top':
        result = validate_double_top(df, pattern)
    elif pattern_type == 'double_bottom':
        result = validate_double_bottom(df, pattern)
    else:
        return {'score': 0, 'is_valid': False, 'confidence': 0.0}
    
    # Add multi-timeframe confirmation if weekly data is provided
    if weekly_data is not None and result['is_valid']:
        mtf_result = check_multi_timeframe_confirmation(df, pattern, weekly_data)
        result['weekly_confirmed'] = mtf_result['weekly_confirmed']
        
        # Boost confidence if weekly timeframe also confirms
        if mtf_result['weekly_confirmed']:
            original_confidence = result['confidence']
            result['confidence'] = min(1.0, original_confidence + mtf_result['confidence_boost'])
            result['mtf_boost'] = True
        else:
            result['mtf_boost'] = False
    
    return result


def get_validation_summary(df, pattern, weekly_data=None):
    """
    Get a detailed validation summary explaining which criteria were met/failed.
    Useful for debugging and understanding pattern quality.
    
    Returns:
        dict: Detailed breakdown of all validation criteria
    """
    validation_result = validate_double_pattern(df, pattern, weekly_data)
    pattern_type = pattern.get('type', '')
    
    summary = {
        'overall_result': validation_result,
        'criteria_breakdown': {},
        'recommendations': []
    }
    
    if pattern_type == 'double_top':
        # Extract pattern points for analysis
        p1_date, p1_price, _ = pattern['P1']
        t_date, t_price, _ = pattern['T']
        p2_date, p2_price, _ = pattern['P2']
        breakout = pattern.get('breakout', None)
        neckline_level = pattern.get('neckline_level', None)
        
        try:
            p1_idx = df[df['Date'] == p1_date].index[0]
            p2_idx = df[df['Date'] == p2_date].index[0]
            
            # Analyze each criterion
            higher_top = max(p1_price, p2_price)
            height_diff = abs(p1_price - p2_price) / higher_top
            summary['criteria_breakdown']['tops_equal'] = {
                'passed': height_diff <= 0.03,
                'value': f"{height_diff:.1%}",
                'threshold': "≤ 3%"
            }
            
            valley_depth = (higher_top - t_price) / higher_top
            summary['criteria_breakdown']['valley_depth'] = {
                'passed': valley_depth >= 0.10,
                'value': f"{valley_depth:.1%}",
                'threshold': "≥ 10%"
            }
            
            pattern_duration = (p2_date - p1_date).days
            summary['criteria_breakdown']['time_spacing'] = {
                'passed': 20 <= pattern_duration <= 90,
                'value': f"{pattern_duration} days",
                'threshold': "20-90 days"
            }
            
            # Preceding uptrend analysis
            if len(df) > 60:
                pre_pattern_start = max(0, p1_idx - 60)
                pre_price = df['Close'].iloc[pre_pattern_start]
                uptrend_gain = (p1_price - pre_price) / pre_price
                summary['criteria_breakdown']['preceding_uptrend'] = {
                    'passed': uptrend_gain >= 0.15,
                    'value': f"{uptrend_gain:.1%}",
                    'threshold': "≥ 15%"
                }
            
            # Breakout analysis
            if breakout and neckline_level is not None:
                _, breakout_price, breakout_idx = breakout
                breakout_depth = (neckline_level - breakout_price) / neckline_level
                summary['criteria_breakdown']['breakout_confirmation'] = {
                    'passed': breakout_depth >= 0.02,
                    'value': f"{breakout_depth:.1%}",
                    'threshold': "≥ 2%"
                }
                
                # Volume analysis
                if 'Volume' in df.columns and breakout_idx >= 20:
                    avg_vol = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
                    volume_ratio = df['Volume'].iloc[breakout_idx] / avg_vol
                    summary['criteria_breakdown']['volume_spike'] = {
                        'passed': volume_ratio >= 2.0,
                        'value': f"{volume_ratio:.1f}x",
                        'threshold': "≥ 2.0x"
                    }
            
        except (IndexError, KeyError):
            summary['criteria_breakdown']['error'] = "Could not analyze pattern - data issues"
    
    # Add recommendations based on failed criteria
    for criterion, details in summary['criteria_breakdown'].items():
        if not details.get('passed', False):
            if criterion == 'tops_equal':
                summary['recommendations'].append("Tops are not sufficiently equal - look for tighter price alignment")
            elif criterion == 'valley_depth':
                summary['recommendations'].append("Valley between tops is too shallow - need deeper retracement")
            elif criterion == 'time_spacing':
                summary['recommendations'].append("Pattern duration outside optimal range - adjust timeframe")
            elif criterion == 'preceding_uptrend':
                summary['recommendations'].append("Insufficient uptrend before pattern - need stronger prior move")
            elif criterion == 'breakout_confirmation':
                summary['recommendations'].append("Breakout not decisive enough - wait for stronger confirmation")
            elif criterion == 'volume_spike':
                summary['recommendations'].append("Insufficient volume on breakout - wait for volume confirmation")
    
    if not summary['recommendations']:
        summary['recommendations'].append("Pattern meets all criteria - high confidence signal")
    
    return summary
