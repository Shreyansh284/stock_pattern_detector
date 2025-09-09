#!/usr/bin/env python3
"""
Test script for Render deployment verification
Usage: python test_render_deployment.py <render_url>
Example: python test_render_deployment.py https://stock-pattern-detector.onrender.com
"""

import sys
import requests
import json
from typing import Dict, Any

def test_endpoint(base_url: str, endpoint: str, expected_status: int = 200) -> Dict[str, Any]:
    """Test a single endpoint and return results."""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        print(f"Testing {url}...")
        response = requests.get(url, timeout=30)
        
        result = {
            'endpoint': endpoint,
            'url': url,
            'status_code': response.status_code,
            'success': response.status_code == expected_status,
            'response_time': response.elapsed.total_seconds(),
            'content_type': response.headers.get('content-type', ''),
        }
        
        if response.status_code == expected_status:
            try:
                data = response.json()
                result['data_preview'] = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                if endpoint == 'stocks':
                    result['stock_count'] = len(data) if isinstance(data, list) else 'N/A'
            except:
                result['data_preview'] = response.text[:200] + "..." if len(response.text) > 200 else response.text
        else:
            result['error'] = response.text[:200] + "..." if len(response.text) > 200 else response.text
            
        return result
        
    except requests.exceptions.Timeout:
        return {
            'endpoint': endpoint,
            'url': url,
            'success': False,
            'error': 'Request timeout (30s)',
        }
    except requests.exceptions.RequestException as e:
        return {
            'endpoint': endpoint,
            'url': url,
            'success': False,
            'error': str(e),
        }

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_render_deployment.py <render_url>")
        print("Example: python test_render_deployment.py https://stock-pattern-detector.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1]
    print(f"Testing Render deployment at: {base_url}")
    print("=" * 60)
    
    # Test endpoints in order of importance
    endpoints = [
        'stocks',
        'patterns', 
        'timeframes',
        'stock-groups',
        'ticker-tape?count=5',
    ]
    
    results = []
    for endpoint in endpoints:
        result = test_endpoint(base_url, endpoint)
        results.append(result)
        
        if result['success']:
            print(f"✅ {endpoint}: OK ({result.get('response_time', 0):.2f}s)")
            if 'stock_count' in result:
                print(f"   📊 Found {result['stock_count']} stocks")
        else:
            print(f"❌ {endpoint}: FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        print()
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print("=" * 60)
    print(f"SUMMARY: {successful}/{total} endpoints working")
    
    if successful == total:
        print("🎉 All tests passed! Your backend is ready.")
        print(f"✅ Backend URL: {base_url}")
        print("✅ Ready for frontend deployment")
    else:
        print("⚠️  Some endpoints failed. Check the errors above.")
        print("💡 Common issues:")
        print("   - Cold start delay (try again in 30 seconds)")
        print("   - Build/deployment still in progress")
        print("   - Environment variables not set correctly")
    
    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())