"""
API Router for pattern detection endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Optional
import uuid
import asyncio

from ..api.models import (
    PatternDetectionRequest, SingleSymbolDetectionRequest, 
    PatternValidationRequest, BatchValidationRequest,
    TaskResponse, PatternResult
)

router = APIRouter(prefix="/patterns", tags=["Pattern Detection"])

# In-memory storage (replace with proper database in production)
tasks = {}

@router.post("/detect", response_model=TaskResponse)
async def detect_patterns(request: PatternDetectionRequest, background_tasks: BackgroundTasks):
    """
    Detect patterns across specified symbols and timeframes.
    Returns a task ID for monitoring progress.
    """
    try:
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "result": None,
            "created_at": datetime.now()
        }
        
        # Start background task
        background_tasks.add_task(run_pattern_detection, task_id, request)
        
        return TaskResponse(
            task_id=task_id,
            status="started",
            message="Pattern detection started. Use /tasks/{task_id}/status to monitor progress."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/{pattern_type}", response_model=TaskResponse)
async def detect_specific_pattern(
    pattern_type: str, 
    request: PatternDetectionRequest,
    background_tasks: BackgroundTasks
):
    """Detect a specific pattern type across symbols"""
    valid_patterns = ["head_and_shoulders", "cup_and_handle", "double_top", "double_bottom"]
    if pattern_type not in valid_patterns:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid pattern type. Must be one of: {valid_patterns}"
        )
    
    # Override patterns in request
    request.patterns = [pattern_type]
    return await detect_patterns(request, background_tasks)

@router.post("/detect/symbol/{symbol}", response_model=TaskResponse)
async def detect_patterns_for_symbol(
    symbol: str,
    request: SingleSymbolDetectionRequest,
    background_tasks: BackgroundTasks
):
    """Analyze patterns for a single symbol"""
    try:
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "result": None,
            "created_at": datetime.now()
        }
        
        # Convert to full detection request
        detection_request = PatternDetectionRequest(
            symbols=[symbol],
            patterns=request.patterns,
            timeframes=request.timeframes,
            mode=request.mode,
            swing_method=request.swing_method,
            require_preceding_trend=request.require_preceding_trend
        )
        
        background_tasks.add_task(run_pattern_detection, task_id, detection_request)
        
        return TaskResponse(
            task_id=task_id,
            status="started",
            message=f"Pattern detection started for {symbol}."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_pattern(request: PatternValidationRequest):
    """Validate a detected pattern"""
    try:
        from ..validator.validate_hns import validate_hns
        from ..validator.validate_double_patterns import validate_double_top, validate_double_bottom
        from ..validator.validate_cup_handle import validate_cup_handle
        
        pattern = request.pattern
        pattern_type = pattern.get('type')
        
        if not pattern_type:
            raise HTTPException(status_code=400, detail="Pattern type is required")
        
        # Get stock data for validation
        symbol = request.symbol
        timeframe = request.timeframe
        data = await get_stock_data(symbol, timeframe)
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Validate based on pattern type
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
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "pattern_type": pattern_type,
            "validation": validation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate/batch")
async def validate_patterns_batch(request: BatchValidationRequest):
    """Validate multiple patterns at once"""
    try:
        results = []
        for pattern_data in request.patterns:
            # Each pattern should include symbol and timeframe
            symbol = pattern_data.get('symbol')
            timeframe = pattern_data.get('timeframe')
            pattern = pattern_data.get('pattern')
            
            if not all([symbol, timeframe, pattern]):
                results.append({
                    "error": "Missing required fields: symbol, timeframe, pattern"
                })
                continue
            
            validation_request = PatternValidationRequest(
                pattern=pattern,
                symbol=symbol,
                timeframe=timeframe
            )
            
            try:
                result = await validate_pattern(validation_request)
                results.append(result)
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "error": str(e)
                })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_pattern_types():
    """Get available pattern types and their descriptions"""
    return {
        "patterns": {
            "head_and_shoulders": {
                "name": "Head and Shoulders",
                "description": "Bearish reversal pattern with three peaks",
                "type": "reversal",
                "direction": "bearish"
            },
            "cup_and_handle": {
                "name": "Cup and Handle",
                "description": "Bullish continuation pattern",
                "type": "continuation", 
                "direction": "bullish"
            },
            "double_top": {
                "name": "Double Top",
                "description": "Bearish reversal pattern with two peaks",
                "type": "reversal",
                "direction": "bearish"
            },
            "double_bottom": {
                "name": "Double Bottom", 
                "description": "Bullish reversal pattern with two troughs",
                "type": "reversal",
                "direction": "bullish"
            }
        }
    }

# Background task functions
async def run_pattern_detection(task_id: str, request: PatternDetectionRequest):
    """Run pattern detection in background"""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["progress"] = 10
        
        # Import and setup detection modules
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from detect_all_patterns import detect_patterns_for_symbols
        
        tasks[task_id]["progress"] = 30
        
        # Convert request to appropriate format and run detection
        # This is a simplified version - adapt based on your actual detection function
        symbols = request.symbols if isinstance(request.symbols, list) else [request.symbols]
        patterns = request.patterns if isinstance(request.patterns, list) else [request.patterns]
        
        results = []
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                # Run detection for each symbol
                symbol_results = await detect_patterns_for_symbol_async(
                    symbol, patterns, request.timeframes, request.mode
                )
                results.extend(symbol_results)
                
                # Update progress
                progress = 30 + (i + 1) / total_symbols * 60
                tasks[task_id]["progress"] = int(progress)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["result"] = {
            "patterns_detected": len(results),
            "symbols_processed": len(symbols),
            "results": results[:50]  # Limit results for response size
        }
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

async def detect_patterns_for_symbol_async(symbol, patterns, timeframes, mode):
    """Async wrapper for pattern detection"""
    # This would need to be implemented based on your actual detection logic
    # For now, return mock data
    return [
        {
            "symbol": symbol,
            "pattern_type": "double_top",
            "timeframe": "1y",
            "confidence": 0.85,
            "detected_at": datetime.now().isoformat()
        }
    ]

async def get_stock_data(symbol: str, timeframe: str):
    """Helper function to get stock data asynchronously"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=timeframe)
        return data
    except Exception:
        return None
