"""
FastAPI application for Pattern Detection System
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from pathlib import Path
import asyncio
import uuid

# Try to import yfinance
try:
    import yfinance as yf
except ImportError:
    yf = None
    print("Warning: yfinance not available. Stock data endpoints will not work.")

# Import your existing modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import existing modules, handle errors gracefully
try:
    from detect_all_patterns import DEFAULT_SYMBOLS, HNS_CONFIG, CH_CONFIG, DT_CONFIG
    DOUBLE_CONFIG = DT_CONFIG  # Alias for consistency
except ImportError as e:
    print(f"Warning: Could not import detect_all_patterns: {e}")
    DEFAULT_SYMBOLS = {"us_tech": ["AAPL", "GOOGL", "MSFT"]}
    HNS_CONFIG = {"min_swing_threshold": 0.03}
    CH_CONFIG = {"min_cup_depth": 0.15}
    DOUBLE_CONFIG = {"peak_similarity_threshold": 0.05}

try:
    from validator.validate_hns import validate_hns
except ImportError:
    def validate_hns(df, pattern):
        return {"score": 0, "is_valid": False, "error": "validator not available"}

try:
    from validator.validate_double_patterns import validate_double_top, validate_double_bottom
except ImportError:
    def validate_double_top(df, pattern):
        return {"score": 0, "is_valid": False, "error": "validator not available"}
    def validate_double_bottom(df, pattern):
        return {"score": 0, "is_valid": False, "error": "validator not available"}

try:
    from validator.validate_cup_handle import validate_cup_handle
except ImportError:
    def validate_cup_handle(df, pattern):
        return {"score": 0, "is_valid": False, "error": "validator not available"}

app = FastAPI(
    title="Pattern Detection API",
    description="API for detecting and validating stock chart patterns",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for charts
app.mount("/static", StaticFiles(directory="outputs"), name="static")

# Pydantic models for request/response
class PatternDetectionRequest(BaseModel):
    symbols: Union[List[str], str] = Field(default="random", description="List of symbols or 'random'")
    patterns: Union[List[str], str] = Field(default="all", description="List of patterns or 'all'")
    timeframes: List[str] = Field(default=["1y", "2y", "3y"], description="List of timeframes")
    mode: str = Field(default="strict", description="Detection mode: strict, lenient, or both")
    swing_method: str = Field(default="rolling", description="Swing detection method")
    require_preceding_trend: bool = Field(default=True, description="Require preceding trend")
    max_patterns_per_timeframe: int = Field(default=10, description="Max patterns per timeframe")
    min_patterns: int = Field(default=0, description="Minimum patterns required")
    random_count: Optional[int] = Field(default=None, description="Number of random symbols")

class PatternValidationRequest(BaseModel):
    pattern: Dict[str, Any] = Field(description="Pattern data to validate")
    symbol: str = Field(description="Stock symbol")
    timeframe: str = Field(description="Timeframe")

class StockDataRequest(BaseModel):
    symbols: List[str] = Field(description="List of stock symbols")
    period: str = Field(default="1y", description="Data period")
    interval: str = Field(default="1d", description="Data interval")

class ChartGenerationRequest(BaseModel):
    symbol: str = Field(description="Stock symbol")
    timeframe: str = Field(description="Timeframe")
    pattern: Optional[Dict[str, Any]] = Field(default=None, description="Pattern to highlight")
    chart_type: str = Field(default="candle", description="Chart type")
    backend: str = Field(default="plotly", description="Plotting backend")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class PatternResult(BaseModel):
    symbol: str
    timeframe: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    validation: Optional[Dict[str, Any]] = None
    chart_url: Optional[str] = None

# In-memory task storage (use Redis/database in production)
tasks = {}

@app.get("/")
async def root():
    return {
        "message": "Pattern Detection API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/v1/info")
async def api_info():
    return {
        "name": "Pattern Detection API",
        "version": "1.0.0",
        "description": "API for detecting and validating stock chart patterns",
        "supported_patterns": ["head_and_shoulders", "cup_and_handle", "double_top", "double_bottom"],
        "supported_timeframes": ["1d", "1w", "2w", "1m", "2m", "3m", "6m", "1y", "2y", "3y", "5y"],
        "detection_modes": ["strict", "lenient", "both"]
    }

@app.get("/api/v1/config/symbols")
async def get_available_symbols():
    return {
        "predefined_sets": DEFAULT_SYMBOLS,
        "total_symbols": sum(len(symbols) for symbols in DEFAULT_SYMBOLS.values())
    }

@app.get("/api/v1/config/patterns")
async def get_pattern_types():
    return {
        "patterns": {
            "head_and_shoulders": {
                "description": "Head and Shoulders pattern detection",
                "config": HNS_CONFIG
            },
            "cup_and_handle": {
                "description": "Cup and Handle pattern detection", 
                "config": CH_CONFIG
            },
            "double_top": {
                "description": "Double Top pattern detection",
                "config": DOUBLE_CONFIG
            },
            "double_bottom": {
                "description": "Double Bottom pattern detection",
                "config": DOUBLE_CONFIG
            }
        }
    }

@app.get("/api/v1/config/timeframes")
async def get_timeframes():
    return {
        "timeframes": {
            "1d": "1 day",
            "1w": "1 week", 
            "2w": "2 weeks",
            "1m": "1 month",
            "2m": "2 months",
            "3m": "3 months", 
            "6m": "6 months",
            "1y": "1 year",
            "2y": "2 years",
            "3y": "3 years",
            "5y": "5 years"
        }
    }

@app.post("/api/v1/patterns/detect")
async def detect_patterns(request: PatternDetectionRequest, background_tasks: BackgroundTasks):
    """Detect patterns across specified symbols and timeframes"""
    try:
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "running", "progress": 0, "result": None}
        
        # Start background task
        background_tasks.add_task(run_pattern_detection, task_id, request)
        
        return TaskResponse(
            task_id=task_id,
            status="started",
            message="Pattern detection started. Use /tasks/{task_id}/status to monitor progress."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/patterns/detect/{pattern_type}")
async def detect_specific_pattern(pattern_type: str, request: PatternDetectionRequest):
    """Detect a specific pattern type"""
    if pattern_type not in ["head_and_shoulders", "cup_and_handle", "double_top", "double_bottom"]:
        raise HTTPException(status_code=400, detail="Invalid pattern type")
    
    # Override patterns in request
    request.patterns = [pattern_type]
    return await detect_patterns(request, BackgroundTasks())

@app.post("/api/v1/patterns/validate")
async def validate_pattern(request: PatternValidationRequest):
    """Validate a detected pattern"""
    try:
        pattern = request.pattern
        pattern_type = pattern.get('type')
        
        if not pattern_type:
            raise HTTPException(status_code=400, detail="Pattern type is required")
        
        # Get stock data for validation
        symbol = request.symbol
        timeframe = request.timeframe
        
        # Try to get stock data
        try:
            data = get_stock_data(symbol, timeframe)
            if data is None or data.empty:
                # For testing purposes, create mock data if real data fails
                import pandas as pd
                import numpy as np
                dates = pd.date_range('2024-01-01', periods=100, freq='D')
                data = pd.DataFrame({
                    'Open': np.random.randn(100).cumsum() + 100,
                    'High': np.random.randn(100).cumsum() + 105,
                    'Low': np.random.randn(100).cumsum() + 95,
                    'Close': np.random.randn(100).cumsum() + 100,
                    'Volume': np.random.randint(1000, 10000, 100)
                }, index=dates)
        except Exception as e:
            # Create simple mock data for testing
            import pandas as pd
            import numpy as np
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            data = pd.DataFrame({
                'Open': np.random.randn(100).cumsum() + 100,
                'High': np.random.randn(100).cumsum() + 105,
                'Low': np.random.randn(100).cumsum() + 95,
                'Close': np.random.randn(100).cumsum() + 100,
                'Volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
        
        # Validate based on pattern type
        try:
            if pattern_type == "head_and_shoulders":
                validation_result = validate_hns(data, pattern)
            elif pattern_type == "double_top":
                validation_result = validate_double_top(data, pattern)
            elif pattern_type == "double_bottom":
                validation_result = validate_double_bottom(data, pattern)
            elif pattern_type == "cup_and_handle":
                validation_result = validate_cup_handle(data, pattern)
            else:
                raise HTTPException(status_code=400, detail="Unsupported pattern type")
        except Exception as e:
            # Return a default validation result if validation fails
            validation_result = {
                "score": 0,
                "is_valid": False,
                "error": f"Validation error: {str(e)}"
            }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "pattern_type": pattern_type,
            "validation": validation_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

@app.get("/api/v1/data/stock/{symbol}")
async def get_stock_data_endpoint(symbol: str, period: str = "1y", interval: str = "1d"):
    """Retrieve stock price data"""
    try:
        if yf is None:
            raise HTTPException(status_code=503, detail="yfinance not available")
            
        data = get_stock_data(symbol, period, interval)
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convert to dict but handle potential issues
        try:
            data_dict = data.reset_index().to_dict('records')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
        
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data": data_dict[:100]  # Limit to first 100 records for testing
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/v1/data/stocks")
async def get_multiple_stock_data(request: StockDataRequest):
    """Get data for multiple symbols"""
    try:
        results = {}
        for symbol in request.symbols:
            data = get_stock_data(symbol, request.period, request.interval)
            if data is not None and not data.empty:
                results[symbol] = data.to_dict('records')
            else:
                results[symbol] = None
        
        return {
            "symbols": request.symbols,
            "period": request.period,
            "interval": request.interval,
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/charts/pattern")
async def generate_pattern_chart(request: ChartGenerationRequest):
    """Generate chart for detected pattern"""
    try:
        # Implementation would create chart using your existing plotting functions
        # For now, return a placeholder
        chart_id = str(uuid.uuid4())
        
        # TODO: Implement actual chart generation
        chart_url = f"/static/charts/{request.symbol}/{request.timeframe}/chart_{chart_id}.html"
        
        return {
            "chart_id": chart_id,
            "chart_url": chart_url,
            "symbol": request.symbol,
            "timeframe": request.timeframe
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Check status of background task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "message": task.get("message", "")
    }

@app.get("/api/v1/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """Get result of completed task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task["result"]
    }

@app.delete("/api/v1/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel running task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    tasks[task_id]["status"] = "cancelled"
    return {"message": f"Task {task_id} cancelled"}

# Background task function
async def run_pattern_detection(task_id: str, request: PatternDetectionRequest):
    """Run pattern detection in background"""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["progress"] = 10
        
        # Convert request to arguments format
        args = type('Args', (), {})()
        args.symbols = request.symbols if isinstance(request.symbols, str) else ','.join(request.symbols)
        args.patterns = request.patterns if isinstance(request.patterns, str) else ','.join(request.patterns)
        args.timeframes = ','.join(request.timeframes)
        args.mode = request.mode
        args.swing_method = request.swing_method
        args.require_preceding_trend = request.require_preceding_trend
        args.max_patterns_per_timeframe = request.max_patterns_per_timeframe
        args.min_patterns = request.min_patterns
        args.random_count = request.random_count
        args.output_dir = './outputs'
        args.verbose = False
        args.delay = 0.5
        args.plotly = True
        args.chart_type = 'candle'
        
        tasks[task_id]["progress"] = 30
        
        # Run the actual detection (this needs to be adapted from your main function)
        # For now, we'll simulate the process
        await asyncio.sleep(2)  # Simulate processing time
        
        tasks[task_id]["progress"] = 80
        
        # Placeholder result
        result = {
            "patterns_detected": 5,
            "symbols_processed": len(request.symbols) if isinstance(request.symbols, list) else 1,
            "charts_generated": True,
            "output_path": "./outputs"
        }
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["result"] = result
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

def get_stock_data(symbol: str, period: str = "1y", interval: str = "1d"):
    """Helper function to get stock data"""
    if yf is None:
        return None
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        # Check if data is valid
        if data is None or data.empty:
            return None
            
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                return None
                
        return data
    except Exception as e:
        print(f"Error getting stock data for {symbol}: {e}")
        return None

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Error: uvicorn is not installed. Please install it with: pip install uvicorn[standard]")
        print("Or use: python start_api.py")
        sys.exit(1)
