# Pattern Detection API Setup and Usage Guide

## Quick Start

### 1. Install Dependencies

```powershell
# Install Python packages
pip install -r requirements.txt
```

### 2. Start the API Server

```powershell
# Option 1: Using the startup script
python start_api.py

# Option 2: Direct uvicorn command
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Using Python directly
python main_api.py
```

### 3. Access the API

- **Main API**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/api/docs
- **Alternative Documentation**: http://localhost:8000/api/redoc

## API Endpoints Overview

### Core Pattern Detection

| Method | Endpoint                                  | Description                             |
| ------ | ----------------------------------------- | --------------------------------------- |
| POST   | `/api/v1/patterns/detect`                 | Detect patterns across multiple symbols |
| POST   | `/api/v1/patterns/detect/{pattern_type}`  | Detect specific pattern type            |
| POST   | `/api/v1/patterns/detect/symbol/{symbol}` | Analyze single symbol                   |

### Pattern Validation

| Method | Endpoint                          | Description                 |
| ------ | --------------------------------- | --------------------------- |
| POST   | `/api/v1/patterns/validate`       | Validate a detected pattern |
| POST   | `/api/v1/patterns/validate/batch` | Validate multiple patterns  |

### Data Retrieval

| Method | Endpoint                      | Description                   |
| ------ | ----------------------------- | ----------------------------- |
| GET    | `/api/v1/data/stock/{symbol}` | Get stock price data          |
| POST   | `/api/v1/data/stocks`         | Get data for multiple symbols |

### Charts and Visualization

| Method | Endpoint                    | Description               |
| ------ | --------------------------- | ------------------------- |
| POST   | `/api/v1/charts/pattern`    | Generate pattern chart    |
| POST   | `/api/v1/charts/compare`    | Generate comparison chart |
| GET    | `/api/v1/charts/{chart_id}` | Retrieve generated chart  |

### Background Tasks

| Method | Endpoint                         | Description       |
| ------ | -------------------------------- | ----------------- |
| GET    | `/api/v1/tasks/{task_id}/status` | Check task status |
| GET    | `/api/v1/tasks/{task_id}/result` | Get task result   |
| DELETE | `/api/v1/tasks/{task_id}`        | Cancel task       |

### Configuration

| Method | Endpoint                    | Description           |
| ------ | --------------------------- | --------------------- |
| GET    | `/api/v1/config/symbols`    | Get available symbols |
| GET    | `/api/v1/config/patterns`   | Get pattern types     |
| GET    | `/api/v1/config/timeframes` | Get timeframes        |

## Usage Examples

### 1. Detect All Patterns for Random Symbols

```bash
curl -X POST "http://localhost:8000/api/v1/patterns/detect" \
-H "Content-Type: application/json" \
-d '{
  "symbols": "random",
  "patterns": "all",
  "timeframes": ["1y", "2y"],
  "mode": "strict",
  "random_count": 5
}'
```

### 2. Detect Double Top for Specific Symbols

```bash
curl -X POST "http://localhost:8000/api/v1/patterns/detect/double_top" \
-H "Content-Type: application/json" \
-d '{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "timeframes": ["1y"],
  "mode": "lenient"
}'
```

### 3. Get Stock Data

```bash
curl "http://localhost:8000/api/v1/data/stock/AAPL?period=1y&interval=1d"
```

### 4. Validate a Pattern

```bash
curl -X POST "http://localhost:8000/api/v1/patterns/validate" \
-H "Content-Type: application/json" \
-d '{
  "symbol": "AAPL",
  "timeframe": "1y",
  "pattern": {
    "type": "double_top",
    "P1": ["2024-01-15", 150.0, 10],
    "T": ["2024-02-15", 140.0, 30],
    "P2": ["2024-03-15", 155.0, 50]
  }
}'
```

### 5. Check Task Status

```bash
curl "http://localhost:8000/api/v1/tasks/your-task-id/status"
```

## Python Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

# Detect patterns
def detect_patterns(symbols, patterns="all", timeframes=["1y"]):
    url = f"{BASE_URL}/patterns/detect"
    data = {
        "symbols": symbols,
        "patterns": patterns,
        "timeframes": timeframes,
        "mode": "strict"
    }

    response = requests.post(url, json=data)
    return response.json()

# Get task status
def get_task_status(task_id):
    url = f"{BASE_URL}/tasks/{task_id}/status"
    response = requests.get(url)
    return response.json()

# Get stock data
def get_stock_data(symbol, period="1y"):
    url = f"{BASE_URL}/data/stock/{symbol}?period={period}"
    response = requests.get(url)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Start pattern detection
    result = detect_patterns(["AAPL", "GOOGL"], "all", ["1y", "2y"])
    task_id = result["task_id"]
    print(f"Started task: {task_id}")

    # Check status
    status = get_task_status(task_id)
    print(f"Task status: {status}")

    # Get stock data
    stock_data = get_stock_data("AAPL")
    print(f"Got {len(stock_data['data'])} data points for AAPL")
```

## JavaScript/Frontend Example

```javascript
// API base URL
const BASE_URL = "http://localhost:8000/api/v1";

// Detect patterns function
async function detectPatterns(symbols, patterns = "all", timeframes = ["1y"]) {
  const response = await fetch(`${BASE_URL}/patterns/detect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      symbols: symbols,
      patterns: patterns,
      timeframes: timeframes,
      mode: "strict",
    }),
  });

  return await response.json();
}

// Get task status
async function getTaskStatus(taskId) {
  const response = await fetch(`${BASE_URL}/tasks/${taskId}/status`);
  return await response.json();
}

// Get stock data
async function getStockData(symbol, period = "1y") {
  const response = await fetch(
    `${BASE_URL}/data/stock/${symbol}?period=${period}`
  );
  return await response.json();
}

// Example usage
async function example() {
  try {
    // Start pattern detection
    const result = await detectPatterns(["AAPL", "GOOGL"]);
    console.log("Started task:", result.task_id);

    // Poll for completion
    let status = await getTaskStatus(result.task_id);
    while (status.status === "running") {
      console.log(`Progress: ${status.progress}%`);
      await new Promise((resolve) => setTimeout(resolve, 2000)); // Wait 2s
      status = await getTaskStatus(result.task_id);
    }

    console.log("Task completed:", status);
  } catch (error) {
    console.error("Error:", error);
  }
}
```

## Error Handling

The API returns consistent error responses:

```json
{
  "detail": "Error message description",
  "error": {
    "code": "ERROR_CODE",
    "message": "Detailed error message",
    "details": {
      "additional": "context"
    }
  }
}
```

Common HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (symbol/pattern not found)
- `422`: Validation Error (invalid request body)
- `500`: Internal Server Error

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Data Configuration
DEFAULT_TIMEFRAME=1y
MAX_SYMBOLS_PER_REQUEST=50
RATE_LIMIT_PER_MINUTE=100

# Output Configuration
CHARTS_DIR=outputs/charts
REPORTS_DIR=outputs/reports
```

### Pattern Detection Settings

Modify pattern configurations in `main_api.py`:

```python
# Pattern-specific configurations
HNS_CONFIG = {
    'min_swing_threshold': 0.03,
    'min_pattern_strength': 0.5,
    # ... other settings
}
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using systemd (Linux)

```ini
[Unit]
Description=Pattern Detection API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/pattern
ExecStart=/usr/bin/python -m uvicorn main_api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Security Considerations

For production deployment:

1. **Enable authentication**
2. **Configure CORS properly**
3. **Use HTTPS**
4. **Implement rate limiting**
5. **Validate all inputs**
6. **Set up logging and monitoring**

## Monitoring and Logging

The API includes basic monitoring endpoints:

- `GET /api/v1/health` - Health check
- `GET /api/v1/status` - System status
- `GET /api/v1/info` - API information

## Support

For issues and questions:

1. Check the interactive documentation at `/api/docs`
2. Review error messages and status codes
3. Check server logs for detailed error information
