#!/usr/bin/env python3
"""
Analysis of the chart pattern shown by user against enhanced double top detection rules.
"""

def analyze_chart_pattern():
    """
    Analyze the user's chart pattern against our enhanced double top validation rules.
    Based on visual inspection of the provided chart.
    """
    
    # Estimated values from the chart
    top1_price = 1155  # Approximate peak 1
    trough_price = 900  # Approximate trough 
    top2_price = 1105  # Approximate peak 2
    neckline_level = 900  # Support/neckline level
    
    analysis_results = {
        'pattern_type': 'Double Top',
        'visual_analysis': {}
    }
    
    # 1. Peak Similarity Check (±3% rule)
    height_diff_pct = abs(top1_price - top2_price) / max(top1_price, top2_price)
    peak_similarity_valid = height_diff_pct <= 0.03
    
    analysis_results['visual_analysis']['peak_similarity'] = {
        'top1_price': top1_price,
        'top2_price': top2_price,
        'height_difference_pct': f"{height_diff_pct:.2%}",
        'valid': peak_similarity_valid,
        'threshold': '±3%',
        'assessment': 'PASS' if peak_similarity_valid else 'FAIL - Peaks not similar enough'
    }
    
    # 2. Valley Depth Check
    valley_depth_pct = (max(top1_price, top2_price) - trough_price) / max(top1_price, top2_price)
    valley_depth_valid = valley_depth_pct >= 0.15
    
    analysis_results['visual_analysis']['valley_depth'] = {
        'trough_price': trough_price,
        'depth_percentage': f"{valley_depth_pct:.2%}",
        'valid': valley_depth_valid,
        'threshold': '≥15%',
        'assessment': 'PASS' if valley_depth_valid else 'FAIL - Valley not deep enough'
    }
    
    # 3. Time Spacing (estimated from chart - appears to be several months)
    estimated_time_spacing_days = 180  # Rough estimate from visual
    time_spacing_valid = 20 <= estimated_time_spacing_days <= 90
    
    analysis_results['visual_analysis']['time_spacing'] = {
        'estimated_days': estimated_time_spacing_days,
        'valid': time_spacing_valid,
        'threshold': '20-90 days',
        'assessment': 'FAIL - Appears too long (>90 days)' if not time_spacing_valid else 'PASS'
    }
    
    # 4. Neckline Breakout
    breakout_visible = True  # Chart shows price below neckline
    
    analysis_results['visual_analysis']['breakout'] = {
        'neckline_level': neckline_level,
        'breakout_visible': breakout_visible,
        'assessment': 'PASS - Clear breakout below neckline visible'
    }
    
    # Overall Assessment
    criteria_passed = sum([
        peak_similarity_valid,
        valley_depth_valid, 
        time_spacing_valid,
        breakout_visible
    ])
    
    analysis_results['overall_assessment'] = {
        'criteria_passed': f"{criteria_passed}/4",
        'would_be_detected': criteria_passed >= 3,
        'confidence_level': 'HIGH' if criteria_passed == 4 else 'MEDIUM' if criteria_passed == 3 else 'LOW'
    }
    
    return analysis_results

def print_analysis():
    """Print formatted analysis results."""
    results = analyze_chart_pattern()
    
    print("=" * 60)
    print("ENHANCED DOUBLE TOP PATTERN ANALYSIS")
    print("=" * 60)
    
    print(f"\nPattern Type: {results['pattern_type']}")
    
    print("\n📊 VISUAL ANALYSIS RESULTS:")
    print("-" * 40)
    
    for criterion, details in results['visual_analysis'].items():
        criterion_name = criterion.replace('_', ' ').title()
        print(f"\n{criterion_name}:")
        
        if 'assessment' in details:
            status = "✅" if details.get('valid', True) else "❌"
            print(f"  {status} {details['assessment']}")
        
        for key, value in details.items():
            if key != 'assessment' and key != 'valid':
                print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 40)
    print("OVERALL ASSESSMENT")
    print("=" * 40)
    
    overall = results['overall_assessment']
    print(f"Criteria Passed: {overall['criteria_passed']}")
    print(f"Would be Detected: {'YES' if overall['would_be_detected'] else 'NO'}")
    print(f"Confidence Level: {overall['confidence_level']}")
    
    print("\n🔍 ENHANCED VALIDATION NOTES:")
    print("-" * 40)
    print("• Peak similarity check: Within ±3% tolerance required")
    print("• Valley depth: Must be ≥15% from peaks for significance") 
    print("• Time spacing: 20-90 days optimal for double top formation")
    print("• Breakout confirmation: Clear break below neckline visible")
    
    print(f"\n{'✅ VALID PATTERN' if overall['would_be_detected'] else '❌ INVALID PATTERN'}")
    
    if not overall['would_be_detected']:
        print("\n⚠️  ISSUES IDENTIFIED:")
        analysis = results['visual_analysis']
        if not analysis['peak_similarity']['valid']:
            print("• Peaks are not similar enough (>3% difference)")
        if not analysis['valley_depth']['valid']:
            print("• Valley between peaks is too shallow (<15%)")
        if not analysis['time_spacing']['valid']:
            print("• Time spacing appears outside optimal 20-90 day range")

if __name__ == "__main__":
    print_analysis()
