from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import DetectRequest, DetectAllRequest, DetectAllResponse
from typing import List, Dict
import sys
import os
import yfinance as yf
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_all_patterns import process_symbol, DEFAULT_SYMBOLS, TIMEFRAMES
from fastapi.responses import JSONResponse
from uuid import uuid4
import threading
import time

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
    strong_charts = []
    weak_charts = []
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
            # Determine strength for supported patterns
            validation = (pattern.get('validation') or {})
            explanation = (pattern.get('explanation') or {})
            is_valid = bool(validation.get('is_valid')) or str(explanation.get('verdict', '')).lower() in ('valid', 'strong', 'true')
            strength = 'strong' if is_valid else 'weak'
            chart_item = {"timeframe": timeframe, "html": html_content, "strength": strength}
            charts.append(chart_item)
            (strong_charts if strength == 'strong' else weak_charts).append(chart_item)
    return JSONResponse(content={"charts": charts, "strong_charts": strong_charts, "weak_charts": weak_charts})
    
# -----------------------------
# Detect-all (synchronous, legacy)
# -----------------------------
@app.post("/detect-all", response_model=DetectAllResponse)
def detect_all_stocks(req: DetectAllRequest):
    # Validate date range
    if req.start_date >= req.end_date:
        return JSONResponse(status_code=400, content={"error": "start_date must be earlier than end_date"})
    # Pattern mapping: snake -> human
    pattern_map = {
        'double_top': 'Double Top',
        'double_bottom': 'Double Bottom',
        'cup_and_handle': 'Cup and Handle',
        'head_and_shoulders': 'Head and Shoulders',
    }
    # Run detection for each available stock
    results = []
    for stock in AVAILABLE_STOCKS:
        # detect patterns for full set
        raw = process_symbol(
            symbol=stock,
            timeframes=['custom'],
            patterns=list(pattern_map.keys()),
            mode='lenient',
            swing_method='rolling',
            output_dir='outputs',
            require_preceding_trend=True,
            min_patterns=1,
            max_patterns_per_timeframe=5,
            organize_by_date=False,
            charts_subdir='charts',
            reports_subdir='reports',
            use_plotly=True,
            chart_type=(req.chart_type.lower() if getattr(req, 'chart_type', None) else 'candle'),
            start_date=req.start_date,
            end_date=req.end_date,
        )
        # Count patterns per type
        types = [p.get('type') for p in raw if p.get('type')]
        counts: Dict[str, int] = {}
        for t in types:
            counts[t] = counts.get(t, 0) + 1
        # Human-readable pattern list
        human_patterns = [pattern_map[t] for t in counts.keys()]
        # Fetch latest price and volume
        try:
            ticker = yf.Ticker(stock)
            # Get the most recent 2 days to ensure we have data
            hist = ticker.history(period='2d')
            if not hist.empty:
                last = hist.iloc[-1]
                current_price = float(last['Close']) if pd.notna(last['Close']) else 0.0
                current_volume = int(last['Volume']) if pd.notna(last['Volume']) else 0
            else:
                # Fallback: try getting info from ticker
                info = ticker.info
                current_price = float(info.get('currentPrice', info.get('regularMarketPrice', 0.0)))
                current_volume = int(info.get('volume', info.get('regularMarketVolume', 0)))
        except Exception as e:
            print(f"Failed to fetch price/volume for {stock}: {e}")
            current_price = 0.0
            current_volume = 0
        # Build charts list including pattern label
        charts = []
        seen_pairs = set()
        for p in raw:
            tf = p.get('timeframe')
            pat = p.get('type')
            path = p.get('image_path')
            key = (tf, pat)
            if not tf or not pat or key in seen_pairs:
                continue
            seen_pairs.add(key)
            if path and path.endswith('.html') and os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    html = f.read()
                # Determine strength
                validation = (p.get('validation') or {})
                explanation = (p.get('explanation') or {})
                is_valid = bool(validation.get('is_valid')) or str(explanation.get('verdict', '')).lower() in ('valid', 'strong', 'true')
                strength = 'strong' if is_valid else 'weak'
                charts.append({
                    'timeframe': tf,
                    'pattern': pattern_map.get(pat, pat),
                    'html': html,
                    'strength': strength,
                    'explanation': explanation if explanation else None,
                })
        # Append stock result
        results.append({
            'stock': stock,
            'patterns': human_patterns,
            'pattern_counts': {pattern_map[k]: v for k, v in counts.items()},
            'count': len(raw),
            'current_price': current_price,
            'current_volume': current_volume,
            'charts': charts,
        })
    return DetectAllResponse(results=results)

# -----------------------------
# Detect-all background job with progress
# -----------------------------

PROGRESS: Dict[str, Dict] = {}
RESULTS: Dict[str, DetectAllResponse] = {}

def _run_detect_all_job(job_id: str, req: DetectAllRequest):
    try:
        PROGRESS[job_id] = {
            'status': 'running',
            'current': 0,
            'total': len(AVAILABLE_STOCKS),
            'symbol': None,
            'message': 'Starting stock pattern scan...'
        }
        # Validate date range
        if req.start_date >= req.end_date:
            raise ValueError("start_date must be earlier than end_date")

        pattern_map = {
            'double_top': 'Double Top',
            'double_bottom': 'Double Bottom',
            'cup_and_handle': 'Cup and Handle',
            'head_and_shoulders': 'Head and Shoulders',
        }

        results: List[Dict] = []
        total = len(AVAILABLE_STOCKS)
        for idx, stock in enumerate(AVAILABLE_STOCKS, start=1):
            PROGRESS[job_id].update({
                'current': idx - 1,
                'symbol': stock,
                'message': f'Processing {stock} ({idx}/{total})'
            })
            raw = process_symbol(
                symbol=stock,
                timeframes=['custom'],
                patterns=list(pattern_map.keys()),
                mode='lenient',
                swing_method='rolling',
                output_dir='outputs',
                require_preceding_trend=True,
                min_patterns=1,
                max_patterns_per_timeframe=5,
                organize_by_date=False,
                charts_subdir='charts',
                reports_subdir='reports',
                use_plotly=True,
                chart_type=(req.chart_type.lower() if getattr(req, 'chart_type', None) else 'candle'),
                start_date=req.start_date,
                end_date=req.end_date,
            )
            types = [p.get('type') for p in raw if p.get('type')]
            counts: Dict[str, int] = {}
            for t in types:
                counts[t] = counts.get(t, 0) + 1
            human_patterns = [pattern_map[t] for t in counts.keys()]
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(period='2d')
                if not hist.empty:
                    last = hist.iloc[-1]
                    current_price = float(last['Close']) if pd.notna(last['Close']) else 0.0
                    current_volume = int(last['Volume']) if pd.notna(last['Volume']) else 0
                else:
                    info = ticker.info
                    current_price = float(info.get('currentPrice', info.get('regularMarketPrice', 0.0)))
                    current_volume = int(info.get('volume', info.get('regularMarketVolume', 0)))
            except Exception as e:
                print(f"Failed to fetch price/volume for {stock}: {e}")
                current_price = 0.0
                current_volume = 0
            charts = []
            seen_pairs = set()
            for p in raw:
                tf = p.get('timeframe')
                pat = p.get('type')
                path = p.get('image_path')
                key = (tf, pat)
                if not tf or not pat or key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if path and path.endswith('.html') and os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        html = f.read()
                    # Determine strength and include explanation if present
                    validation = (p.get('validation') or {})
                    explanation = (p.get('explanation') or {})
                    is_valid = bool(validation.get('is_valid')) or str(explanation.get('verdict', '')).lower() in ('valid', 'strong', 'true')
                    strength = 'strong' if is_valid else 'weak'
                    charts.append({
                        'timeframe': tf,
                        'pattern': pattern_map.get(pat, pat),
                        'html': html,
                        'strength': strength,
                        'explanation': explanation if explanation else None,
                    })
            results.append({
                'stock': stock,
                'patterns': human_patterns,
                'pattern_counts': {pattern_map[k]: v for k, v in counts.items()},
                'count': len(raw),
                'current_price': current_price,
                'current_volume': current_volume,
                'charts': charts,
            })
            PROGRESS[job_id].update({ 'current': idx })

        RESULTS[job_id] = DetectAllResponse(results=results)
        PROGRESS[job_id].update({ 'status': 'done', 'message': 'Completed' })
    except Exception as e:
        PROGRESS[job_id] = {
            'status': 'error',
            'current': PROGRESS.get(job_id, {}).get('current', 0),
            'total': PROGRESS.get(job_id, {}).get('total', len(AVAILABLE_STOCKS)),
            'symbol': PROGRESS.get(job_id, {}).get('symbol', None),
            'message': f'Error: {e}'
        }

@app.post('/detect-all-start')
def detect_all_start(req: DetectAllRequest):
    job_id = str(uuid4())
    PROGRESS[job_id] = { 'status': 'queued', 'current': 0, 'total': len(AVAILABLE_STOCKS), 'symbol': None, 'message': 'Queued' }
    t = threading.Thread(target=_run_detect_all_job, args=(job_id, req), daemon=True)
    t.start()
    return { 'job_id': job_id }

@app.get('/detect-all-progress')
def detect_all_progress(job_id: str):
    p = PROGRESS.get(job_id)
    if not p:
        raise HTTPException(status_code=404, detail='Job not found')
    # Include percent for convenience
    current = int(p.get('current') or 0)
    total = int(p.get('total') or 1)
    percent = int(min(100, max(0, (current / total) * 100)))
    return { **p, 'percent': percent }

@app.get('/detect-all-result', response_model=DetectAllResponse)
def detect_all_result(job_id: str):
    if PROGRESS.get(job_id, {}).get('status') != 'done':
        raise HTTPException(status_code=202, detail='Not ready')
    res = RESULTS.get(job_id)
    if not res:
        raise HTTPException(status_code=404, detail='Result not found')
    return res
