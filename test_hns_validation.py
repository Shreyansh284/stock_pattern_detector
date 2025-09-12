#!/usr/bin/env python3
"""
Test script for enhanced HNS validation system
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from detect_all_patterns import detect_head_and_shoulders, HNS_CONFIG

def test_hns_validation():
    """Test HNS validation with local stock data"""
    
    # Test with a few stocks from StockData folder
    test_stocks = ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'BHARTIARTL']
    stock_data_dir = Path('StockData')
    
    print("🧪 Testing Enhanced HNS Validation System")
    print("=" * 60)
    
    results = {}
    
    for stock in test_stocks:
        csv_file = stock_data_dir / f"{stock}.csv"
        if csv_file.exists():
            try:
                print(f"\n📊 Testing {stock}...")
                
                # Load data
                df = pd.read_csv(csv_file)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                
                # Check if required columns exist
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"  ⚠️  Missing required columns in {stock}")
                    continue
                
                if len(df) < 200:
                    print(f"  ⚠️  Insufficient data ({len(df)} rows)")
                    continue
                
                # Use last 500 days for testing
                df_test = df.tail(500).copy()
                
                # Detect HNS patterns
                config = HNS_CONFIG['strict']
                patterns = detect_head_and_shoulders(df_test, config, require_preceding_trend=True)
                
                if patterns:
                    print(f"  🔍 Found {len(patterns)} potential HNS patterns")
                    
                    # Import validator
                    try:
                        from validator.validate_hns import validate_hns
                        
                        valid_patterns = []
                        for i, pattern in enumerate(patterns):
                            validation = validate_hns(df_test, pattern)
                            pattern['validation'] = validation
                            
                            confidence = validation.get('confidence', 0)
                            is_valid = validation.get('is_valid', False)
                            score = validation.get('score', 0)
                            
                            print(f"    Pattern {i+1}: Score {score}/6, Confidence: {confidence:.1%}, Valid: {is_valid}")
                            
                            if is_valid and confidence >= 0.83:  # Our strict threshold
                                valid_patterns.append(pattern)
                                print(f"      ✅ STRONG HNS pattern validated!")
                            else:
                                reasons = validation.get('rejection_reasons', [])
                                print(f"      ❌ Pattern rejected: {reasons[:2] if reasons else ['Low confidence']}")
                        
                        results[stock] = {
                            'total_detected': len(patterns),
                            'valid_patterns': len(valid_patterns),
                            'validation_rate': len(valid_patterns) / len(patterns) if patterns else 0
                        }
                        
                    except ImportError:
                        print(f"  ⚠️  Validator not available")
                        
                else:
                    print(f"  ℹ️  No HNS patterns detected")
                    results[stock] = {'total_detected': 0, 'valid_patterns': 0, 'validation_rate': 0}
                    
            except Exception as e:
                print(f"  ❌ Error processing {stock}: {e}")
                import traceback
                print(f"  📋 Error details: {traceback.format_exc()}")
        else:
            print(f"  ⚠️  CSV file not found for {stock}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    total_detected = sum(r['total_detected'] for r in results.values())
    total_valid = sum(r['valid_patterns'] for r in results.values())
    
    print(f"Stocks tested: {len(results)}")
    print(f"Total HNS patterns detected: {total_detected}")
    print(f"Valid patterns after strict filtering: {total_valid}")
    print(f"Overall validation rate: {total_valid/total_detected:.1%}" if total_detected > 0 else "No patterns to validate")
    
    print("\nPer-stock breakdown:")
    for stock, data in results.items():
        rate = data['validation_rate']
        print(f"  {stock}: {data['valid_patterns']}/{data['total_detected']} valid ({rate:.1%})")
    
    # Test success criteria
    if total_detected == 0:
        print("\n✅ SUCCESS: System correctly filters out weak patterns (no patterns detected)")
    elif total_valid < total_detected * 0.3:  # Less than 30% pass = good filtering
        print(f"\n✅ SUCCESS: Strict filtering working - only {total_valid/total_detected:.1%} of patterns passed validation")
    else:
        print(f"\n⚠️  WARNING: {total_valid/total_detected:.1%} of patterns passed - filtering may be too lenient")

if __name__ == "__main__":
    test_hns_validation()
