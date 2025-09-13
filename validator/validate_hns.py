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
    
    Added Rule (MANDATORY): A verifiable downtrend MUST begin after the right shoulder prior to (or into) the neckline breakout. 
    This prevents premature validation when price has not started rolling over.
    
    STRICT MODE - Textbook Perfect Patterns:
    1️⃣ Head prominence: Head ≥ 8% above BOTH shoulders
    2️⃣ Shoulder similarity: ≤ 12% height difference
    3️⃣ Time symmetry: ≤ 25% asymmetry
    4️⃣ Neckline angle: ≤ 20°
    5️⃣ Post-right-shoulder downtrend: Decline ≥ 3%, lower-high structure, negative slope
    6️⃣ Breakout confirmation: Decisive close below neckline within 10–20 periods
    7️⃣ Volume spike: Breakout volume ≥ 1.5x pattern average
    
    LENIENT MODE - Real Market Patterns:
    1️⃣ Head prominence: Head ≥ 3% above BOTH shoulders
    2️⃣ Shoulder similarity: ≤ 15% height difference
    3️⃣ Time symmetry: ≤ 50% asymmetry
    4️⃣ Neckline angle: ≤ 45°
    5️⃣ Post-right-shoulder downtrend: Decline ≥ 1–2% with negative bias (adaptive)
    6️⃣ Breakout confirmation: Decisive close below neckline within ≤ 30 periods
    7️⃣ Volume spike: Breakout volume ≥ 1.3x pattern average
    
    Downtrend Detection Logic (mandatory in both modes):
      • Segment: From right shoulder index (exclusive) to breakout index (exclusive) OR next 12 bars if no breakout.
      • Metrics:
          - decline_pct: (right_shoulder_high - min_close) / right_shoulder_high
          - slope_pct_per_bar: linear regression slope normalised by right_shoulder_high
          - lower_highs_ratio: fraction of highs forming a non-increasing sequence
      • Pass Conditions:
          STRICT: decline_pct ≥ 3% AND slope negative AND lower_highs_ratio ≥ 0.6
          LENIENT: (decline_pct ≥ 2% OR (decline_pct ≥ 1% AND slope negative)) AND lower_highs_ratio ≥ 0.5
      • Immediate Breakout Edge Case: If breakout occurs within ≤2 bars, downtrend passes if close at breakout already ≥ required decline below right shoulder high.
    
    Returns: {'score': int, 'is_valid': bool, 'confidence': float, 'details': {...}}
    """
    score = 0
    # New scoring now includes post-right-shoulder downtrend (added rule)
    max_score = 7
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
        time_symmetry_req = 0.40        # tightened from 50% to 40%
        center_offset_req = 0.35        # head must be within 35% of midpoint
        min_side_span_req = 3           # each side should develop at least 3 days (if total span allows)
        neckline_angle_req = 45.0       # 45° instead of 20°
        breakout_timing_max = 30        # 30 periods instead of 20
        volume_spike_req = 1.3          # 1.3x instead of 1.5x
        min_score_req = 5               # 5/7 instead of previous 4/6
        downtrend_min_decline_primary = 0.02  # 2% preferred
        downtrend_min_decline_floor = 0.01    # 1% absolute floor with negative slope
        downtrend_lower_highs_req = 0.50
    else:  # strict mode
        head_prominence_req = 0.08      # 8%
        shoulder_similarity_req = 0.12   # 12%
        time_symmetry_req = 0.25        # 25%
        center_offset_req = 0.20        # head near midpoint
        min_side_span_req = 4           # each side should develop ≥4 days (if span permits)
        neckline_angle_req = 20.0       # 20°
        breakout_timing_max = 20        # 20 periods
        volume_spike_req = 1.5          # 1.5x
        min_score_req = 6               # 6/7 required
        downtrend_min_decline_primary = 0.03  # 3%
        downtrend_min_decline_floor = 0.03    # Floor equals primary (no relaxation)
        downtrend_lower_highs_req = 0.60
    
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
    
    # 3️⃣ TIME SYMMETRY (multi-metric: raw asymmetry, head center offset, minimum side development)
    try:
        left_span = max(1, (p2_date - p1_date).days)
        right_span = max(1, (p3_date - p2_date).days)
        total_span = max(1, (p3_date - p1_date).days)
        time_asymmetry = abs(left_span - right_span) / max(left_span, right_span)
        # Head center offset relative to midpoint
        midpoint = p1_date + (p3_date - p1_date) / 2
        center_offset_days = abs((p2_date - midpoint).days)
        center_offset_pct = center_offset_days / total_span if total_span > 0 else 0.0
        enforce_min_span = total_span >= (min_side_span_req * 2)
        sides_ok = (left_span >= min_side_span_req and right_span >= min_side_span_req) if enforce_min_span else True
        passed_symmetry = (time_asymmetry <= time_symmetry_req and center_offset_pct <= center_offset_req and sides_ok)
        detailed_scores['time_symmetry'] = {
            'left_span_days': left_span,
            'right_span_days': right_span,
            'total_span_days': total_span,
            'asymmetry_pct': f"{time_asymmetry:.1%}",
            'asymmetry_limit_pct': f"{time_symmetry_req:.1%}",
            'head_center_offset_pct': f"{center_offset_pct:.1%}",
            'center_offset_limit_pct': f"{center_offset_req:.1%}",
            'min_side_span_req_days': min_side_span_req,
            'enforce_min_span': enforce_min_span,
            'sides_ok': sides_ok,
            'required': f"asym≤{time_symmetry_req:.1%} & center≤{center_offset_req:.1%} & side≥{min_side_span_req}d",
            'passed': passed_symmetry
        }
        if passed_symmetry:
            score += 1
            print(f"    ✅ Time symmetry: asym {time_asymmetry:.1%}, head offset {center_offset_pct:.1%}")
        else:
            if time_asymmetry > time_symmetry_req:
                rejection_reasons.append(f"Time asymmetry {time_asymmetry:.1%} > {time_symmetry_req:.1%} max")
                print(f"    ❌ Time symmetry (asymmetry {time_asymmetry:.1%} > {time_symmetry_req:.1%})")
            if center_offset_pct > center_offset_req:
                rejection_reasons.append(f"Head offset {center_offset_pct:.1%} > {center_offset_req:.1%} limit")
                print(f"    ❌ Time symmetry (center offset {center_offset_pct:.1%} > {center_offset_req:.1%})")
            if not sides_ok and enforce_min_span:
                rejection_reasons.append("Insufficient side development for symmetry")
                print("    ❌ Time symmetry (insufficient side development)")
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
    
    # 5️⃣ POST-RIGHT-SHOULDER DOWNTREND (MANDATORY) - ensure rollover begins before/into breakout
    downtrend_confirmed = False
    try:
        post_start = int(p3_idx) + 1
        # Determine end index for analysis (before breakout if breakout exists, else a lookahead window)
        if breakout and isinstance(breakout[2], (int, np.integer)) and int(breakout[2]) > post_start:
            analysis_end = int(breakout[2])  # exclude breakout bar for pure pre-breakout trend
        else:
            # Use a capped window (12 bars) if no breakout yet
            analysis_end = min(len(df) - 1, post_start + 12)
        window_len = analysis_end - post_start + 1
        if window_len >= 2 and post_start < len(df):
            closes = df['Close'].iloc[post_start:analysis_end+1].astype(float).values
            highs = df['High'].iloc[post_start:analysis_end+1].astype(float).values if 'High' in df.columns else closes
            indices = np.arange(len(closes))
            # Linear regression slope
            slope, _ = np.polyfit(indices, closes, 1)
            slope_pct_per_bar = slope / p3_high if p3_high else 0.0
            min_close = closes.min()
            decline_pct = (p3_high - min_close) / p3_high if p3_high else 0.0
            # Lower highs ratio
            lower_highs = 0
            for i in range(1, len(highs)):
                if highs[i] <= highs[i-1]:
                    lower_highs += 1
            lower_highs_ratio = lower_highs / (len(highs) - 1) if len(highs) > 1 else 0.0
            # Immediate breakout edge case
            immediate_breakout = breakout and isinstance(breakout[2], (int, np.integer)) and (int(breakout[2]) - int(p3_idx)) <= 2
            immediate_decline_pct = 0.0
            if immediate_breakout:
                b_idx = int(breakout[2])
                if 0 <= b_idx < len(df):
                    b_close = float(df['Close'].iloc[b_idx])
                    immediate_decline_pct = (p3_high - b_close) / p3_high if p3_high else 0.0
            # Pass logic
            if validation_mode == 'strict':
                downtrend_pass = (decline_pct >= downtrend_min_decline_primary and slope_pct_per_bar < 0 and lower_highs_ratio >= downtrend_lower_highs_req) or \
                                 (immediate_breakout and immediate_decline_pct >= downtrend_min_decline_primary)
            else:
                downtrend_pass = (
                    (decline_pct >= downtrend_min_decline_primary and lower_highs_ratio >= downtrend_lower_highs_req) or
                    (decline_pct >= downtrend_min_decline_floor and slope_pct_per_bar < 0 and lower_highs_ratio >= downtrend_lower_highs_req) or
                    (immediate_breakout and immediate_decline_pct >= downtrend_min_decline_floor)
                )
            detailed_scores['post_right_shoulder_downtrend'] = {
                'bars_analyzed': int(window_len),
                'decline_pct': f"{decline_pct:.1%}",
                'slope_pct_per_bar': f"{slope_pct_per_bar:.3%}",
                'lower_highs_ratio': f"{lower_highs_ratio:.1%}",
                'immediate_breakout': bool(immediate_breakout),
                'immediate_decline_pct': f"{immediate_decline_pct:.1%}",
                'required': (
                    f"decline ≥ {downtrend_min_decline_primary:.0%} & neg slope & lower highs ≥ {downtrend_lower_highs_req:.0%} (strict)" if validation_mode=='strict' else
                    f"decline ≥ {downtrend_min_decline_primary:.0%} (or {downtrend_min_decline_floor:.0%} w/ neg slope) & lower highs ≥ {downtrend_lower_highs_req:.0%}"
                ),
                'passed': downtrend_pass
            }
            if downtrend_pass:
                downtrend_confirmed = True
                score += 1
                print(f"    ✅ Post-RS downtrend: decline {decline_pct:.1%}, lower highs {lower_highs_ratio:.1%}, slope {slope_pct_per_bar:.3%} per bar")
            else:
                print(f"    ❌ Post-RS downtrend insufficient: decline {decline_pct:.1%}, lower highs {lower_highs_ratio:.1%}, slope {slope_pct_per_bar:.3%}")
        else:
            detailed_scores['post_right_shoulder_downtrend'] = {
                'bars_analyzed': int(window_len),
                'error': 'Insufficient bars after right shoulder',
                'passed': False
            }
            print("    ❌ Post-RS downtrend check failed: insufficient bars")
    except Exception as e:
        detailed_scores['post_right_shoulder_downtrend'] = {'passed': False, 'error': f'Calculation failed: {e}'}
        print(f"    ❌ Post-RS downtrend calculation error: {e}")

    # 6️⃣ BREAKOUT CONFIRMATION - CRITICAL RULE (decisive close below neckline)
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
    
    # 7️⃣ VOLUME SPIKE (1.5x pattern formation average)
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
    
    # Pattern is valid if: score meets threshold AND no critical failures AND valid duration AND downtrend confirmed
    head_prominence_ok = detailed_scores.get('head_prominence', {}).get('passed', False)
    if detailed_scores.get('post_right_shoulder_downtrend', {}).get('passed', False) == False:
        major_failures.append('post_right_shoulder_downtrend')
    is_valid = (score >= min_score_req) and (len(major_failures) == 0) and duration_valid and downtrend_confirmed
    
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
            'post_rs_downtrend_ok': detailed_scores.get('post_right_shoulder_downtrend', {}).get('passed', False),
            'breakout_confirmed': breakout_confirmed,
            'volume_confirmed': volume_confirmed,
            'duration_valid': duration_valid
        }
    }
