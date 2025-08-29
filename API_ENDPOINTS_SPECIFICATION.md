# Pattern Detection API Endpoints Specification

## Overview

This document outlines all possible API endpoints for the Pattern Detection System. The API will be built using FastAPI and provide RESTful endpoints for pattern detection, validation, and analysis.

## Base URL

```
http://localhost:8000/api/v1
```

## 1. Core Pattern Detection Endpoints

### 1.1 Detect All Patterns

- **Endpoint**: `POST /patterns/detect`
- **Description**: Detect multiple patterns across specified symbols and timeframes
- **Request Body**:

```json
{
  "symbols": ["AAPL", "GOOGL", "TSLA"], // or "random" for random selection
  "patterns": [
    "head_and_shoulders",
    "cup_and_handle",
    "double_top",
    "double_bottom"
  ], // or "all"
  "timeframes": ["1y", "2y", "3y"],
  "mode": "strict", // "strict", "lenient", or "both"
  "swing_method": "rolling", // "rolling", "zigzag", "fractal"
  "require_preceding_trend": true,
  "max_patterns_per_timeframe": 10,
  "min_patterns": 0
}
```

- **Response**: Array of detected patterns with metadata

### 1.2 Detect Specific Pattern Type

- **Endpoint**: `POST /patterns/detect/{pattern_type}`
- **Description**: Detect a specific pattern type
- **Path Parameters**:
  - `pattern_type`: "head_and_shoulders", "cup_and_handle", "double_top", "double_bottom"
- **Request Body**: Similar to detect all but without patterns array

### 1.3 Single Symbol Analysis

- **Endpoint**: `POST /patterns/detect/symbol/{symbol}`
- **Description**: Analyze patterns for a single symbol
- **Path Parameters**:
  - `symbol`: Stock symbol (e.g., "AAPL")

## 2. Pattern Validation Endpoints

### 2.1 Validate Pattern

- **Endpoint**: `POST /patterns/validate`
- **Description**: Validate a detected pattern
- **Request Body**:

```json
{
  "pattern": {
    "type": "head_and_shoulders",
    "symbol": "AAPL",
    "timeframe": "1y",
    "P1": ["2024-01-15", 150.0, 10],
    "P2": ["2024-03-15", 180.0, 50],
    "P3": ["2024-05-15", 155.0, 90],
    "T1": ["2024-02-15", 140.0, 30],
    "T2": ["2024-04-15", 145.0, 70]
  }
}
```

- **Response**: Validation score and details

### 2.2 Batch Validate Patterns

- **Endpoint**: `POST /patterns/validate/batch`
- **Description**: Validate multiple patterns at once
- **Request Body**: Array of patterns

### 2.3 Validate by Pattern Type

- **Endpoint**: `POST /patterns/validate/{pattern_type}`
- **Description**: Validate patterns of a specific type

## 3. Data Retrieval Endpoints

### 3.1 Get Stock Data

- **Endpoint**: `GET /data/stock/{symbol}`
- **Description**: Retrieve stock price data
- **Query Parameters**:
  - `period`: "1y", "2y", "3y", "5y"
  - `interval`: "1d", "1h", etc.
- **Response**: OHLCV data

### 3.2 Get Multiple Stock Data

- **Endpoint**: `POST /data/stocks`
- **Description**: Get data for multiple symbols
- **Request Body**:

```json
{
  "symbols": ["AAPL", "GOOGL"],
  "period": "1y",
  "interval": "1d"
}
```

## 4. Chart Generation Endpoints

### 4.1 Generate Pattern Chart

- **Endpoint**: `POST /charts/pattern`
- **Description**: Generate chart for detected pattern
- **Request Body**:

```json
{
  "symbol": "AAPL",
  "timeframe": "1y",
  "pattern": {
    /* pattern object */
  },
  "chart_type": "candle", // "candle", "line", "ohlc"
  "backend": "plotly" // "plotly", "matplotlib"
}
```

- **Response**: Chart URL or base64 encoded image

### 4.2 Generate Comparison Chart

- **Endpoint**: `POST /charts/compare`
- **Description**: Compare multiple symbols or patterns

### 4.3 Get Chart

- **Endpoint**: `GET /charts/{chart_id}`
- **Description**: Retrieve generated chart by ID

## 5. Results and Reports Endpoints

### 5.1 Get Pattern Results

- **Endpoint**: `GET /results/patterns`
- **Description**: Retrieve stored pattern detection results
- **Query Parameters**:
  - `symbol`: Filter by symbol
  - `pattern_type`: Filter by pattern type
  - `timeframe`: Filter by timeframe
  - `date_from`, `date_to`: Date range filter
  - `min_score`: Minimum validation score

### 5.2 Generate Report

- **Endpoint**: `POST /reports/generate`
- **Description**: Generate comprehensive analysis report
- **Request Body**:

```json
{
  "symbols": ["AAPL", "GOOGL"],
  "date_range": ["2024-01-01", "2024-12-31"],
  "format": "csv", // "csv", "json", "pdf"
  "include_charts": true
}
```

### 5.3 Get Report

- **Endpoint**: `GET /reports/{report_id}`
- **Description**: Download generated report

## 6. Configuration and Settings Endpoints

### 6.1 Get Available Symbols

- **Endpoint**: `GET /config/symbols`
- **Description**: Get list of available symbol sets
- **Response**:

```json
{
  "predefined_sets": {
    "us_tech": ["AAPL", "GOOGL", "MSFT"],
    "us_large": ["..."],
    "indian_popular": ["..."]
  },
  "total_symbols": 1000
}
```

### 6.2 Get Pattern Types

- **Endpoint**: `GET /config/patterns`
- **Description**: Get available pattern types and their configurations

### 6.3 Get Timeframes

- **Endpoint**: `GET /config/timeframes`
- **Description**: Get available timeframes

### 6.4 Update Configuration

- **Endpoint**: `PUT /config/patterns/{pattern_type}`
- **Description**: Update pattern detection configuration

## 7. Scoring and Logging Endpoints

### 7.1 Get Pattern Scores

- **Endpoint**: `GET /scores/patterns/{pattern_type}`
- **Description**: Get validation scores for pattern type
- **Query Parameters**: symbol, timeframe filters

### 7.2 Log Pattern Score

- **Endpoint**: `POST /scores/log`
- **Description**: Log a pattern validation score

### 7.3 Get Score Statistics

- **Endpoint**: `GET /scores/stats`
- **Description**: Get score statistics and analytics

## 8. Background Task Endpoints

### 8.1 Start Background Analysis

- **Endpoint**: `POST /tasks/analyze`
- **Description**: Start long-running pattern analysis task
- **Response**: Task ID for monitoring

### 8.2 Get Task Status

- **Endpoint**: `GET /tasks/{task_id}/status`
- **Description**: Check status of background task

### 8.3 Get Task Result

- **Endpoint**: `GET /tasks/{task_id}/result`
- **Description**: Get result of completed task

### 8.4 Cancel Task

- **Endpoint**: `DELETE /tasks/{task_id}`
- **Description**: Cancel running task

## 9. Health and Monitoring Endpoints

### 9.1 Health Check

- **Endpoint**: `GET /health`
- **Description**: Basic health check

### 9.2 System Status

- **Endpoint**: `GET /status`
- **Description**: Detailed system status and metrics

### 9.3 API Information

- **Endpoint**: `GET /info`
- **Description**: API version and capabilities

## 10. File Management Endpoints

### 10.1 Upload Symbol List

- **Endpoint**: `POST /files/symbols/upload`
- **Description**: Upload custom symbol list file

### 10.2 Download Results

- **Endpoint**: `GET /files/results/{file_id}/download`
- **Description**: Download analysis results file

### 10.3 List Files

- **Endpoint**: `GET /files`
- **Description**: List available result files

## 11. Real-time and Streaming Endpoints

### 11.1 WebSocket Pattern Updates

- **Endpoint**: `WS /ws/patterns`
- **Description**: Real-time pattern detection updates

### 11.2 WebSocket Market Data

- **Endpoint**: `WS /ws/market/{symbol}`
- **Description**: Real-time market data stream

## 12. User Management (Optional)

### 12.1 User Authentication

- **Endpoint**: `POST /auth/login`
- **Description**: User login

### 12.2 User Registration

- **Endpoint**: `POST /auth/register`
- **Description**: User registration

### 12.3 User Profile

- **Endpoint**: `GET /auth/profile`
- **Description**: Get user profile and settings

## Error Handling

All endpoints follow consistent error response format:

```json
{
  "error": {
    "code": "PATTERN_NOT_FOUND",
    "message": "Pattern not found for the specified criteria",
    "details": {
      "symbol": "AAPL",
      "timeframe": "1y"
    }
  }
}
```

## Rate Limiting

- 100 requests per minute for pattern detection
- 1000 requests per minute for data retrieval
- 10 requests per minute for chart generation

## Response Formats

All responses include metadata:

```json
{
  "data": {
    /* actual response data */
  },
  "metadata": {
    "timestamp": "2024-08-29T10:30:00Z",
    "version": "1.0.0",
    "processing_time_ms": 150
  }
}
```
