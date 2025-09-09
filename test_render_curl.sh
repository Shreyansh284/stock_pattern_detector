#!/bin/bash
# Simple curl test for Render deployment
# Usage: ./test_render_curl.sh <render_url>
# Example: ./test_render_curl.sh https://stock-pattern-detector.onrender.com

if [ $# -eq 0 ]; then
    echo "Usage: $0 <render_url>"
    echo "Example: $0 https://stock-pattern-detector.onrender.com"
    exit 1
fi

BASE_URL=$1
echo "Testing Render deployment at: $BASE_URL"
echo "=============================================="

# Test /stocks endpoint
echo "Testing /stocks endpoint..."
curl -s -w "Status: %{http_code}, Time: %{time_total}s\n" \
     -H "Accept: application/json" \
     "$BASE_URL/stocks" | head -n 5

echo ""
echo "=============================================="

# Test if we get a JSON response with stock symbols
echo "Checking if stocks endpoint returns valid data..."
RESPONSE=$(curl -s "$BASE_URL/stocks")
if echo "$RESPONSE" | grep -q "\["; then
    STOCK_COUNT=$(echo "$RESPONSE" | grep -o ',' | wc -l)
    echo "✅ Success: Found approximately $((STOCK_COUNT + 1)) stocks"
else
    echo "❌ Failed: Invalid response format"
    echo "Response: $RESPONSE"
fi

echo ""
echo "Test complete!"