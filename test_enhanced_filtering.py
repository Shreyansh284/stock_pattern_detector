#!/usr/bin/env python3
"""
Test the enhanced double top detection filtering to ensure invalid patterns are rejected.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(__file__))

try:
    import detect_all_patterns as dap
    from validator.validate_double_patterns import validate_double_pattern
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def create_invalid_double_top_pattern():
    """Create test data that represents the pattern from the chart (which should be rejected)."""
    
    # Create 200 days of data to simulate the chart pattern
    dates = pd.date_range(start='2022-01-01', periods=200, freq='D')
    
    # Simulate the pattern from the chart:
    # - Top 1 around day 60 at ~1155
    # - Trough around day 120 at ~900  
    # - Top 2 around day 180 at ~1105 (4.3% lower than Top 1 - should be rejected)
    # - Time gap ~180 days (should be rejected for being > 90 days)
    
    base_prices = []
    for i in range(200):
        if i < 50:
            # Initial uptrend to Top 1
            price = 950 + (i * 4)  # Rise to ~1150
        elif i < 70:
            # Peak area (Top 1)
            price = 1155 + np.random.normal(0, 5)
        elif i < 120:
            # Decline to trough  
            progress = (i - 70) / 50
            price = 1155 - (255 * progress) + np.random.normal(0, 10)  # Down to ~900
        elif i < 140:
            # Trough area
            price = 900 + np.random.normal(0, 8)
        elif i < 180:
            # Rise to Top 2 (lower than Top 1)
            progress = (i - 140) / 40
            price = 900 + (205 * progress) + np.random.normal(0, 8)  # Up to ~1105
        else:
            # Top 2 area and decline
            price = 1105 + np.random.normal(0, 5)
            if i > 185:  # Start declining
                decline = (i - 185) * 3
                price = max(850, 1105 - decline)
    
        base_prices.append(max(800, price))  # Floor price
    
    # Create OHLC data
    opens = base_prices
    highs = [p + abs(np.random.normal(0, 3)) for p in base_prices]
    lows = [p - abs(np.random.normal(0, 3)) for p in base_prices] 
    closes = base_prices
    volumes = [1000 + np.random.randint(0, 500) for _ in range(200)]
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    return df

def test_enhanced_filtering():
    """Test that our enhanced detection properly filters out invalid patterns."""
    
    print("🔍 TESTING ENHANCED DOUBLE TOP FILTERING")
    print("=" * 60)
    
    # Create invalid test data (similar to user's chart)
    df = create_invalid_double_top_pattern()
    
    # Generate swing points
    df = dap.generate_swing_flags(df, method='rolling', N_bars=10)
    
    # Try to detect double top patterns with strict config
    dt_config = dap.DT_CONFIG['strict']
    
    print("\n📊 ATTEMPTING PATTERN DETECTION...")
    print("-" * 40)
    
    # This should return empty results due to our enhanced filtering
    patterns = dap.detect_double_patterns(df, dt_config, 'double_top', require_preceding_trend=True)
    
    print(f"\n🎯 DETECTION RESULTS:")
    print(f"Patterns found: {len(patterns)}")
    
    if len(patterns) == 0:
        print("✅ SUCCESS: No invalid patterns detected (as expected)")
        print("✅ Enhanced filtering is working correctly!")
        
        # Show why it would have been rejected
        print("\n📋 WHY THIS PATTERN WOULD BE REJECTED:")
        print("- Peak height difference: ~4.3% (exceeds 3% limit)")
        print("- Time spacing: ~180 days (exceeds 90 day limit)")  
        print("- Enhanced validation prevents false signals")
        
    else:
        print("❌ FAILURE: Invalid patterns were detected")
        print("❌ Enhanced filtering may not be working properly")
        
        # Analyze what got through
        for i, pattern in enumerate(patterns):
            print(f"\nPattern {i+1}:")
            print(f"  Duration: {pattern.get('duration', 'Unknown')} days")
            print(f"  Height similarity: {pattern.get('height_similarity_pct', 'Unknown'):.1%}")
            
            # Test with validator
            if 'validation' not in pattern:
                try:
                    validation = validate_double_pattern(df, pattern)
                    pattern['validation'] = validation
                except Exception as e:
                    print(f"  Validation error: {e}")
            
            if 'validation' in pattern:
                val = pattern['validation']
                print(f"  Validator result: {'VALID' if val.get('is_valid') else 'INVALID'}")
                print(f"  Confidence: {val.get('confidence', 0):.1%}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_enhanced_filtering()
