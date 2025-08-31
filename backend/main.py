from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import DetectRequest
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_all_patterns import process_symbol
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_STOCKS = ["TCS.NS", "MARUTI.NS", "LT.NS", "ONGC.NS"]
AVAILABLE_PATTERNS = ["Double Top", "Cup and Handle", "Head and Shoulders"]
AVAILABLE_TIMEFRAMES = ["6m", "1y", "2y", "3y", "5y"]

@app.get("/stocks", response_model=List[str])
def get_stocks():
    return AVAILABLE_STOCKS

@app.get("/patterns", response_model=List[str])
def get_patterns():
    return AVAILABLE_PATTERNS

@app.get("/timeframes", response_model=List[str])
def get_timeframes():
    return AVAILABLE_TIMEFRAMES

@app.post("/detect")
def detect_patterns(req: DetectRequest):
    pattern_map = {
        "Double Top": "double_top",
        "Double Bottom": "double_bottom",
        "Cup and Handle": "cup_and_handle",
        "Head and Shoulders": "head_and_shoulders"
    }
    patterns = [pattern_map.get(req.pattern, req.pattern)]
    timeframes = [tf.lower() for tf in req.timeframes]
    results = process_symbol(
        symbol=req.stock,
        timeframes=timeframes,
        patterns=patterns,
        mode="lenient",
        swing_method="rolling",
        output_dir="outputs",
        require_preceding_trend=True,
        min_patterns=1,
        max_patterns_per_timeframe=5,
        organize_by_date=False,
        charts_subdir="charts",
        reports_subdir="reports",
        use_plotly=True,
        chart_type="candle"
    )
    charts = []
    for pattern in results:
        image_path = pattern.get('image_path')
        timeframe = pattern.get('timeframe', None)
        if image_path and image_path.endswith('.html') and os.path.exists(image_path):
            with open(image_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            charts.append({"timeframe": timeframe, "html": html_content})
    return JSONResponse(content={"charts": charts})
