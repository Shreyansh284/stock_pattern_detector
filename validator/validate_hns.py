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
    Validate Head and Shoulders pattern with REALISTIC criteria for market conditions.
    
    STRICT MODE - Textbook Perfect Patterns:
    1️⃣ Head prominence: Head must be at least 8% higher than BOTH shoulders
    2️⃣ Shoulder similarity: Shoulders within 8% height difference  
    3️⃣ Time symmetry: Left/right timing difference ≤ 25%
    4️⃣ Neckline angle: Nearly horizontal, ≤ 20 degrees
    5️⃣ Breakout confirmation: Decisive close below neckline within 10-20 periods
    6️⃣ Volume spike: Breakout volume ≥ 1.5x pattern formation average
    
    LENIENT MODE - Real Market Patterns:
    1️⃣ Head prominence: Head must be at least 3% higher than BOTH shoulders
    2️⃣ Shoulder similarity: Shoulders within 15% height difference  
    3️⃣ Time symmetry: Left/right timing difference ≤ 50%
    4️⃣ Neckline angle: Reasonable slope, ≤ 35 degrees
    5️⃣ Breakout confirmation: Close below neckline within 30 periods
    6️⃣ Volume spike: Breakout volume ≥ 1.3x pattern formation average
    
    Validation adapts based on pattern context and market conditions.
    
    Returns: {'score': int, 'is_valid': bool, 'confidence': float, 'details': {...}}
    """
    score = 0
    max_score = 6
    rejection_reasons = []
    detailed_scores = {}
    
    # Detect validation mode from pattern context (lenient vs strict)
    validation_mode = 'strict'  # Default
    
    # Try to infer mode from pattern data or use adaptive thresholds
    pattern_duration = 0
    try:
        p1_date = pattern['P1'][0] if 'P1' in pattern else None
        p3_date = pattern['P3'][0] if 'P3' in pattern else None
        if p1_date and p3_date:
            pattern_duration = (p3_date - p1_date).days
            # Longer patterns tend to be more lenient in real markets
            if pattern_duration > 60:
                validation_mode = 'lenient'
    except Exception:
        pass
    
    # Set adaptive thresholds based on mode
    if validation_mode == 'lenient':
        head_prominence_req = 0.03      # 3% instead of 8%
        shoulder_similarity_req = 0.15   # 15% instead of 12%
        time_symmetry_req = 0.50        # 50% instead of 35%
        neckline_angle_req = 45.0       # 45° instead of 20°
        breakout_timing_max = 30        # 30 periods instead of 20
        volume_spike_req = 1.3          # 1.3x instead of 1.5x
        min_score_req = 4               # 4/6 instead of 5/6
    else:  # strict mode
        head_prominence_req = 0.08      # 8%
        shoulder_similarity_req = 0.12   # 12%
        time_symmetry_req = 0.25        # 25%
        neckline_angle_req = 20.0       # 20°
        breakout_timing_max = 20        # 20 periods
        volume_spike_req = 1.5          # 1.5x
        min_score_req = 5               # 5/6
    
    print(f"    🔍 HNS Validation Mode: {validation_mode.upper()}")
    
    # Extract pattern points with error handling
    try:
        p1_date, p1_high, p1_idx = pattern['P1']  # Left Shoulder
        p2_date, p2_high, p2_idx = pattern['P2']  # Head  
        p3_date, p3_high, p3_idx = pattern['P3']  # Right Shoulder
        t1_date, t1_low, t1_idx = pattern['T1']   # Trough 1
        t2_date, t2_low, t2_idx = pattern['T2']   # Trough 2
        breakout = pattern.get('breakout')
    except Exception as e:
        return {
            'score': 0, 
            'is_valid': False, 
            'confidence': 0.0,
            'details': {'error': f'Missing pattern points: {str(e)}'}
        }
    
    # Get pattern duration for validation
    try:
        pattern_duration = (p3_date - p1_date).days
    except Exception:
        pattern_duration = 0
    
    # 1️⃣ HEAD PROMINENCE - CRITICAL RULE (10% minimum above BOTH shoulders)
    try:
        left_shoulder_prominence = (p2_high - p1_high) / p1_high if p1_high > 0 else 0.0
        right_shoulder_prominence = (p2_high - p3_high) / p3_high if p3_high > 0 else 0.0
        min_prominence = min(left_shoulder_prominence, right_shoulder_prominence)
        
        detailed_scores['head_prominence'] = {
            'left_prominence_pct': f"{left_shoulder_prominence:.1%}",
            'right_prominence_pct': f"{right_shoulder_prominence:.1%}",
            'minimum_prominence_pct': f"{min_prominence:.1%}",
            'required': f"≥{head_prominence_req:.1%}",
            'passed': min_prominence >= head_prominence_req
        }
        
        if min_prominence >= head_prominence_req:
            score += 1
            print(f"    ✅ Head prominence: {min_prominence:.1%} (≥{head_prominence_req:.1%} required)")
        else:
            rejection_reasons.append(f"Head prominence {min_prominence:.1%} < {head_prominence_req:.1%} minimum")
            print(f"    ❌ Head prominence: {min_prominence:.1%} < {head_prominence_req:.1%} minimum")
            
    except Exception:
        rejection_reasons.append("Head prominence calculation failed")
        detailed_scores['head_prominence'] = {'passed': False, 'error': 'Calculation failed'}
    
    # 2️⃣ SHOULDER SIMILARITY (within 8% height difference)
    try:
        higher_shoulder = max(p1_high, p3_high)
        shoulder_diff_pct = abs(p1_high - p3_high) / higher_shoulder
        
        detailed_scores['shoulder_similarity'] = {
            'left_shoulder': p1_high,
            'right_shoulder': p3_high,
            'height_difference_pct': f"{shoulder_diff_pct:.1%}",
            'required': f"≤{shoulder_similarity_req:.1%}",
            'passed': shoulder_diff_pct <= shoulder_similarity_req
        }
        
        if shoulder_diff_pct <= shoulder_similarity_req:
            score += 1
            print(f"    ✅ Shoulder similarity: {shoulder_diff_pct:.1%} difference (≤{shoulder_similarity_req:.1%} required)")
        else:
            rejection_reasons.append(f"Shoulder height difference {shoulder_diff_pct:.1%} > {shoulder_similarity_req:.1%} maximum")
            print(f"    ❌ Shoulder similarity: {shoulder_diff_pct:.1%} > {shoulder_similarity_req:.1%} maximum")
            
    except Exception:
        rejection_reasons.append("Shoulder similarity calculation failed")
        detailed_scores['shoulder_similarity'] = {'passed': False, 'error': 'Calculation failed'}
    
    # 3️⃣ TIME SYMMETRY (≤25% difference in timing)
    try:
        left_span = max(1, (p2_date - p1_date).days)
        right_span = max(1, (p3_date - p2_date).days)
        time_asymmetry = abs(left_span - right_span) / max(left_span, right_span)
        
        detailed_scores['time_symmetry'] = {
            'left_span_days': left_span,
            'right_span_days': right_span,
            'asymmetry_pct': f"{time_asymmetry:.1%}",
            'required': f"≤{time_symmetry_req:.1%}",
            'passed': time_asymmetry <= time_symmetry_req
        }
        
        if time_asymmetry <= time_symmetry_req:
            score += 1
            print(f"    ✅ Time symmetry: {time_asymmetry:.1%} asymmetry (≤{time_symmetry_req:.1%} required)")
        else:
            rejection_reasons.append(f"Time asymmetry {time_asymmetry:.1%} > {time_symmetry_req:.1%} maximum")
            print(f"    ❌ Time symmetry: {time_asymmetry:.1%} > {time_symmetry_req:.1%} maximum")
            
    except Exception:
        rejection_reasons.append("Time symmetry calculation failed")
        detailed_scores['time_symmetry'] = {'passed': False, 'error': 'Calculation failed'}
    
    # 4️⃣ NECKLINE ANGLE - CRITICAL RULE (≤20 degrees, nearly horizontal)
    try:
        slope = pattern.get('neckline_slope')
        intercept = pattern.get('neckline_intercept')
        if slope is None or intercept is None:
            slope, intercept = _compute_neckline_from_troughs(t1_date, t1_low, t2_date, t2_low)
        
        if slope is not None:
            angle_deg = abs(degrees(atan(float(slope))))
            
            detailed_scores['neckline_angle'] = {
                'angle_degrees': f"{angle_deg:.1f}°",
                'required': f"≤{neckline_angle_req:.1f}°",
                'passed': angle_deg <= neckline_angle_req
            }
            
            if angle_deg <= neckline_angle_req:
                score += 1
                print(f"    ✅ Neckline angle: {angle_deg:.1f}° (≤{neckline_angle_req:.1f}° required)")
            else:
                rejection_reasons.append(f"Neckline angle {angle_deg:.1f}° > {neckline_angle_req:.1f}° maximum")
                print(f"    ❌ Neckline angle: {angle_deg:.1f}° > {neckline_angle_req:.1f}° maximum")
        else:
            rejection_reasons.append("Neckline slope calculation failed")
            detailed_scores['neckline_angle'] = {'passed': False, 'error': 'Cannot calculate slope'}
            
    except Exception:
        rejection_reasons.append("Neckline angle calculation failed")
        detailed_scores['neckline_angle'] = {'passed': False, 'error': 'Calculation failed'}
    
    # 5️⃣ BREAKOUT CONFIRMATION - CRITICAL RULE (decisive close below neckline)
    breakout_confirmed = False
    try:
        if breakout and slope is not None and intercept is not None:
            breakout_date, breakout_price, breakout_idx = breakout
            
            # Check if breakout is within 10-20 periods of right shoulder
            breakout_timing = breakout_idx - p3_idx
            
            # Get close price at breakout
            if 0 <= int(breakout_idx) < len(df):
                close_price = float(df['Close'].iloc[breakout_idx])
                breakout_date_actual = df['Date'].iloc[breakout_idx]
            else:
                close_price = float(breakout_price)
                breakout_date_actual = breakout_date
            
            # Calculate neckline level at breakout date
            neckline_at_breakout = float(slope) * breakout_date_actual.toordinal() + float(intercept)
            
            # Check for decisive breakout (close significantly below neckline)
            breakout_margin = (neckline_at_breakout - close_price) / neckline_at_breakout
            is_decisive = close_price < neckline_at_breakout and breakout_margin >= 0.01  # At least 1% below
            
            detailed_scores['breakout_confirmation'] = {
                'breakout_timing_periods': breakout_timing,
                'close_price': close_price,
                'neckline_price': neckline_at_breakout,
                'breakout_margin_pct': f"{breakout_margin:.1%}",
                'required': f"Decisive close below neckline within 10-{breakout_timing_max} periods",
                'passed': is_decisive and 10 <= breakout_timing <= breakout_timing_max
            }
            
            if is_decisive and 10 <= breakout_timing <= breakout_timing_max:
                score += 1
                breakout_confirmed = True
                print(f"    ✅ Breakout confirmed: {breakout_margin:.1%} below neckline in {breakout_timing} periods")
            else:
                if not is_decisive:
                    rejection_reasons.append(f"Breakout not decisive - only {breakout_margin:.1%} below neckline")
                else:
                    rejection_reasons.append(f"Breakout timing {breakout_timing} periods outside 10-{breakout_timing_max} range")
                print(f"    ❌ Breakout confirmation failed")
        else:
            rejection_reasons.append("No breakout detected or neckline calculation failed")
            detailed_scores['breakout_confirmation'] = {'passed': False, 'error': 'No valid breakout'}
            
    except Exception:
        rejection_reasons.append("Breakout confirmation calculation failed")
        detailed_scores['breakout_confirmation'] = {'passed': False, 'error': 'Calculation failed'}
    
    # 6️⃣ VOLUME SPIKE (1.5x pattern formation average)
    volume_confirmed = False
    try:
        if 'Volume' in df.columns and breakout and isinstance(breakout[2], (int, np.integer)):
            breakout_idx = int(breakout[2])
            
            # Calculate average volume during pattern formation
            pattern_start_idx = int(p1_idx)
            pattern_end_idx = int(p3_idx)
            
            if breakout_idx > pattern_end_idx and pattern_end_idx > pattern_start_idx:
                pattern_avg_volume = float(df['Volume'].iloc[pattern_start_idx:pattern_end_idx+1].mean())
                breakout_volume = float(df['Volume'].iloc[breakout_idx])
                volume_ratio = breakout_volume / pattern_avg_volume if pattern_avg_volume > 0 else 0
                
                detailed_scores['volume_spike'] = {
                    'pattern_avg_volume': f"{pattern_avg_volume:,.0f}",
                    'breakout_volume': f"{breakout_volume:,.0f}",
                    'volume_ratio': f"{volume_ratio:.1f}x",
                    'required': f"≥{volume_spike_req:.1f}x pattern average",
                    'passed': volume_ratio >= volume_spike_req
                }
                
                if volume_ratio >= volume_spike_req:
                    score += 1
                    volume_confirmed = True
                    print(f"    ✅ Volume spike: {volume_ratio:.1f}x pattern average (≥{volume_spike_req:.1f}x required)")
                else:
                    rejection_reasons.append(f"Volume spike {volume_ratio:.1f}x < {volume_spike_req:.1f}x minimum")
                    print(f"    ❌ Volume spike: {volume_ratio:.1f}x < {volume_spike_req:.1f}x minimum")
            else:
                rejection_reasons.append("Invalid breakout index for volume analysis")
                detailed_scores['volume_spike'] = {'passed': False, 'error': 'Invalid indices'}
        else:
            # Missing volume data - treat as incomplete pattern
            rejection_reasons.append("Volume data missing - pattern incomplete")
            detailed_scores['volume_spike'] = {'passed': False, 'error': 'Volume data unavailable'}
            
    except Exception:
        rejection_reasons.append("Volume spike calculation failed")
        detailed_scores['volume_spike'] = {'passed': False, 'error': 'Calculation failed'}
    
    # Additional validation: Pattern duration - adaptive based on mode
    if validation_mode == 'lenient':
        duration_min, duration_max = 30, 365  # Up to 1 year for realistic patterns
    else:  # strict mode
        duration_min, duration_max = 30, 90   # Textbook 30-90 days
    
    duration_valid = duration_min <= pattern_duration <= duration_max
    detailed_scores['pattern_duration'] = {
        'duration_days': pattern_duration,
        'required': f"{duration_min}-{duration_max} days",
        'passed': duration_valid
    }
    
    if not duration_valid:
        rejection_reasons.append(f"Pattern duration {pattern_duration} days outside 30-90 day range")
    
    # Calculate confidence and validity
    confidence = score / max_score
    
    # ADAPTIVE VALIDATION: Different standards for strict vs lenient modes
    major_failures = []
    
    # In strict mode, head prominence is critical
    # In lenient mode, allow more flexibility
    if validation_mode == 'strict':
        if detailed_scores.get('head_prominence', {}).get('passed', False) == False:
            major_failures.append('head_prominence')
        if detailed_scores.get('neckline_angle', {}).get('passed', False) == False:
            major_failures.append('neckline_angle')
    else:  # lenient mode
        # Only head prominence is truly critical in lenient mode
        if detailed_scores.get('head_prominence', {}).get('passed', False) == False:
            major_failures.append('head_prominence')
    
    # Pattern is valid if: score meets threshold AND no critical failures AND valid duration
    head_prominence_ok = detailed_scores.get('head_prominence', {}).get('passed', False)
    is_valid = (score >= min_score_req) and (len(major_failures) == 0) and duration_valid
    
    if not is_valid:
        print(f"    ❌ HNS Pattern REJECTED ({validation_mode.upper()}) - Score: {score}/6, Major failures: {major_failures}")
    else:
        quality = "STRONG" if validation_mode == 'strict' else "VALID"
        print(f"    ✅ {quality} HNS Pattern VALIDATED ({validation_mode.upper()}) - Score: {score}/6, Confidence: {confidence:.1%}")
    
    return {
        'score': int(score),
        'is_valid': bool(is_valid),
        'confidence': round(confidence, 3),
        'max_score': max_score,
        'major_failures': major_failures,
        'rejection_reasons': rejection_reasons,
        'details': detailed_scores,
        'pattern_duration_days': pattern_duration,
        'summary': {
            'head_prominence_ok': detailed_scores.get('head_prominence', {}).get('passed', False),
            'shoulder_similarity_ok': detailed_scores.get('shoulder_similarity', {}).get('passed', False),
            'time_symmetry_ok': detailed_scores.get('time_symmetry', {}).get('passed', False),
            'neckline_angle_ok': detailed_scores.get('neckline_angle', {}).get('passed', False),
            'breakout_confirmed': breakout_confirmed,
            'volume_confirmed': volume_confirmed,
            'duration_valid': duration_valid
        }
    }
