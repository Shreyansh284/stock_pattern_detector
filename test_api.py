#!/usr/bin/env python3
"""
Test script for Pattern Detection API
Run this to verify the API is working correctly
"""
import requests
import json
import time
import sys

# API Configuration
BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT = 30

def test_api_connection():
    """Test basic API connection"""
    print("Testing API connection...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is running and healthy")
            return True
        else:
            print(f"✗ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to API: {e}")
        return False

def test_config_endpoints():
    """Test configuration endpoints"""
    print("\nTesting configuration endpoints...")
    
    endpoints = [
        "/config/symbols",
        "/config/patterns", 
        "/config/timeframes",
        "/info"
    ]
    
    all_passed = True
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✓ {endpoint}")
            else:
                print(f"✗ {endpoint}: {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"✗ {endpoint}: {e}")
            all_passed = False
    
    return all_passed

def test_stock_data():
    """Test stock data retrieval"""
    print("\nTesting stock data retrieval...")
    
    try:
        response = requests.get(f"{BASE_URL}/data/stock/AAPL?period=1mo&interval=1d", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Stock data retrieved for AAPL: {len(data.get('data', []))} records")
            return True
        else:
            print(f"✗ Stock data retrieval failed: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Stock data retrieval error: {e}")
        return False

def test_pattern_detection():
    """Test pattern detection"""
    print("\nTesting pattern detection...")
    
    request_data = {
        "symbols": ["AAPL"],
        "patterns": ["double_top"],
        "timeframes": ["6m"],
        "mode": "lenient",
        "max_patterns_per_timeframe": 2
    }
    
    try:
        # Start detection
        response = requests.post(f"{BASE_URL}/patterns/detect", 
                               json=request_data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            print(f"✓ Pattern detection started: {task_id}")
            
            # Monitor task progress
            return monitor_task(task_id)
        else:
            print(f"✗ Pattern detection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Pattern detection error: {e}")
        return False

def monitor_task(task_id, max_wait=60):
    """Monitor task progress"""
    print(f"Monitoring task {task_id}...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/tasks/{task_id}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                current_status = status.get("status")
                progress = status.get("progress", 0)
                
                print(f"  Status: {current_status}, Progress: {progress}%")
                
                if current_status == "completed":
                    print("✓ Task completed successfully")
                    return True
                elif current_status == "failed":
                    print("✗ Task failed")
                    return False
                
                time.sleep(2)
            else:
                print(f"✗ Cannot get task status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Error monitoring task: {e}")
            return False
    
    print("✗ Task timeout")
    return False

def test_pattern_validation():
    """Test pattern validation"""
    print("\nTesting pattern validation...")
    
    # Sample pattern data
    pattern_data = {
        "symbol": "AAPL",
        "timeframe": "1y",
        "pattern": {
            "type": "double_top",
            "P1": ["2024-01-15", 180.0, 10],
            "T": ["2024-02-15", 160.0, 30],
            "P2": ["2024-03-15", 185.0, 50],
            "breakout": ["2024-04-01", 155.0, 65]
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/patterns/validate", 
                               json=pattern_data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Pattern validation successful")
            validation = result.get("validation", {})
            print(f"  Score: {validation.get('score', 'N/A')}")
            print(f"  Valid: {validation.get('is_valid', 'N/A')}")
            return True
        else:
            print(f"✗ Pattern validation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Pattern validation error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("Pattern Detection API Test Suite")
    print("=" * 50)
    
    tests = [
        ("API Connection", test_api_connection),
        ("Configuration Endpoints", test_config_endpoints),
        ("Stock Data Retrieval", test_stock_data),
        ("Pattern Detection", test_pattern_detection),
        ("Pattern Validation", test_pattern_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! API is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the API configuration.")
        return False

def quick_test():
    """Run a quick connectivity test"""
    print("Quick API Test")
    print("-" * 20)
    
    if test_api_connection():
        print("✓ API is running")
        return True
    else:
        print("✗ API is not accessible")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Pattern Detection API")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick connectivity test only")
    parser.add_argument("--url", type=str, default=BASE_URL,
                       help="API base URL")
    
    args = parser.parse_args()
    
    BASE_URL = args.url.rstrip("/") + "/api/v1" if not args.url.endswith("/api/v1") else args.url
    
    print(f"Testing API at: {BASE_URL}")
    
    if args.quick:
        success = quick_test()
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
