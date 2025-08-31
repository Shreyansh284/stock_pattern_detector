from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import DetectRequest
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_all_patterns import process_symbol, DEFAULT_SYMBOLS, TIMEFRAMES
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build a comprehensive stock list by flattening DEFAULT_SYMBOLS from detect_all_patterns
def _flatten_symbols(symbol_groups: dict[str, list[str]]) -> list[str]:
    seen = set()
    flat: list[str] = []
    for _, symbols in symbol_groups.items():
        for s in symbols:
            if s not in seen:
                seen.add(s)
                flat.append(s)
    return flat

AVAILABLE_STOCKS = _flatten_symbols(DEFAULT_SYMBOLS)
AVAILABLE_PATTERNS = ["Double Top", "Double Bottom", "Cup and Handle", "Head and Shoulders"]
# Preserve order as defined in detect_all_patterns.TIMEFRAMES mapping
AVAILABLE_TIMEFRAMES = list(TIMEFRAMES.keys())
AVAILABLE_CHART_TYPES = ["candle", "line", "ohlc"]
AVAILABLE_MODES = ["lenient", "strict"]

@app.get("/stocks", response_model=List[str])
def get_stocks():
    return AVAILABLE_STOCKS

@app.get("/stock-groups", response_model=Dict[str, List[str]])
def get_stock_groups():
    # Return the original groups but ensure uniqueness and preserve ordering
    groups: Dict[str, List[str]] = {}
    for key, vals in DEFAULT_SYMBOLS.items():
        seen = set()
        ordered: List[str] = []
        for s in vals:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        groups[str(key)] = ordered
    return groups

@app.get("/patterns", response_model=List[str])
def get_patterns():
    return AVAILABLE_PATTERNS

@app.get("/timeframes", response_model=List[str])
def get_timeframes():
    return AVAILABLE_TIMEFRAMES

@app.get("/chart-types", response_model=List[str])
def get_chart_types():
    return AVAILABLE_CHART_TYPES

@app.get("/modes", response_model=List[str])
def get_modes():
    return AVAILABLE_MODES

@app.post("/detect")
def detect_patterns(req: DetectRequest):
    pattern_map = {
        "Double Top": "double_top",
        "Double Bottom": "double_bottom",
        "Cup and Handle": "cup_and_handle",
        "Head and Shoulders": "head_and_shoulders"
    }
    patterns = [pattern_map.get(req.pattern, req.pattern)]
    # Determine whether using timeframe or custom date range
    using_date_range = bool(req.start_date and req.end_date)
    if using_date_range:
        timeframes = ["custom"]
    else:
        if not req.timeframe:
            return JSONResponse(status_code=400, content={"error": "Provide either timeframe or start_date+end_date"})
        timeframes = [req.timeframe.lower()]
    # pick mode (default lenient)
    mode = req.mode.lower() if getattr(req, 'mode', None) else "lenient"
    if mode not in AVAILABLE_MODES:
        mode = "lenient"
    results = process_symbol(
        symbol=req.stock,
        timeframes=timeframes,
        patterns=patterns,
        mode=mode,
        swing_method="rolling",
        output_dir="outputs",
        require_preceding_trend=True,
        min_patterns=1,
        max_patterns_per_timeframe=5,
        organize_by_date=False,
        charts_subdir="charts",
        reports_subdir="reports",
        use_plotly=True,
    chart_type=req.chart_type.lower() if req.chart_type else "candle",
    start_date=req.start_date if using_date_range else None,
    end_date=req.end_date if using_date_range else None,
    )
    # Return at most one chart per timeframe (e.g., summary)
    charts = []
    seen_tf = set()
    for pattern in results:
        image_path = pattern.get('image_path')
        timeframe = pattern.get('timeframe', None)
        # skip duplicates
        if timeframe in seen_tf:
            continue
        seen_tf.add(timeframe)
        if image_path and image_path.endswith('.html') and os.path.exists(image_path):
            with open(image_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            charts.append({"timeframe": timeframe, "html": html_content})
    return JSONResponse(content={"charts": charts})
