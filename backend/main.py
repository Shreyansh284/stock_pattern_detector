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
from datetime import datetime, timedelta
import concurrent.futures
from typing import Callable
import asyncio

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

# Simple in-memory cache for expensive endpoints (e.g., ticker tape)
_TICKER_CACHE: dict[int, dict] = {}
_TICKER_CACHE_TTL_SEC = 45  # serve cached result for this many seconds

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
    # Choose data source
    data_source = (getattr(req, 'data_source', None) or 'live').lower()
    def _resolve_data_dir(explicit: str | None) -> str:
        if explicit and os.path.isdir(explicit):
            return explicit
        env = os.environ.get('STOCK_DATA_DIR')
        if env and os.path.isdir(env):
            return env
        root = os.path.dirname(os.path.dirname(__file__))
        cand1 = os.path.join(root, 'StockData')
        cand2 = os.path.join(root, 'STOCK_DATA')
        return cand1 if os.path.isdir(cand1) else cand2
    stock_data_dir = _resolve_data_dir(getattr(req, 'stock_data_dir', None))
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
    keep_best_only=(False if data_source == 'past' else True),
    data_source=data_source,
    stock_data_dir=stock_data_dir,
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
    # Determine source for stock list and data
    data_source = (getattr(req, 'data_source', None) or 'live').lower()
    def _resolve_data_dir(explicit: str | None) -> str:
        if explicit and os.path.isdir(explicit):
            return explicit
        env = os.environ.get('STOCK_DATA_DIR')
        if env and os.path.isdir(env):
            return env
        root = os.path.dirname(os.path.dirname(__file__))
        cand1 = os.path.join(root, 'StockData')
        cand2 = os.path.join(root, 'STOCK_DATA')
        return cand1 if os.path.isdir(cand1) else cand2
    stock_data_dir = _resolve_data_dir(getattr(req, 'stock_data_dir', None))
    # If past mode, derive stock list from CSV filenames
    def list_csv_symbols(data_dir: str) -> list[str]:
        """Return symbols derived from CSV filenames, preferring only .NS suffixed versions.

        Previously both raw symbol and symbol.NS were returned, causing duplicates in UI.
        We now emit only the .NS variant (common for NSE listings) to avoid duplicates while
        keeping compatibility (loader strips suffix when opening CSV)."""
        syms: list[str] = []
        try:
            for fn in os.listdir(data_dir):
                if fn.lower().endswith('.csv'):
                    name = fn[:-4]
                    syms.append(name)
        except Exception:
            pass
        uniq: list[str] = []
        seen: set[str] = set()
        for s in syms:
            variant = f"{s}.NS"
            if variant not in seen:
                seen.add(variant)
                uniq.append(variant)
        return uniq
    stock_list = list_csv_symbols(stock_data_dir) if data_source == 'past' else AVAILABLE_STOCKS
    if data_source == 'past':
        limit = getattr(req, 'stock_limit', None) or 150
        try:
            limit = int(limit)
        except Exception:
            limit = 150
        limit = max(1, min(1000, limit))  # safety bounds
        if len(stock_list) > limit:
            stock_list = stock_list[:limit]
    for stock in stock_list:
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
            keep_best_only=(False if data_source == 'past' else True),
            data_source=data_source,
            stock_data_dir=stock_data_dir,
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
    seen_hashes = set()
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
                # Suppress duplicate identical charts (same content across symbols/timeframes)
                import hashlib
                hval = hashlib.sha256(html.encode('utf-8')).hexdigest()
                if hval in seen_hashes:
                    continue
                seen_hashes.add(hval)
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

def _process_stock_batch(stocks: List[str], req: DetectAllRequest, pattern_map: Dict[str, str], 
                       stock_data_dir: str, data_source: str) -> List[Dict]:
    """Process a batch of stocks in parallel."""
    batch_results = []
    
    def process_single_stock(stock: str) -> Dict:
        """Process a single stock and return its result."""
        try:
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
                keep_best_only=(False if data_source == 'past' else True),
                data_source=data_source,
                stock_data_dir=stock_data_dir,
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
            
            # Build charts list
            charts = []
            seen_pairs = set()
            seen_hashes = set()
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
                    import hashlib
                    hval = hashlib.sha256(html.encode('utf-8')).hexdigest()
                    if hval in seen_hashes:
                        continue
                    seen_hashes.add(hval)
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
            
            return {
                'stock': stock,
                'patterns': human_patterns,
                'pattern_counts': {pattern_map[k]: v for k, v in counts.items()},
                'count': len(raw),
                'current_price': current_price,
                'current_volume': current_volume,
                'charts': charts,
            }
        except Exception as e:
            print(f"Error processing stock {stock}: {e}")
            return {
                'stock': stock,
                'patterns': [],
                'pattern_counts': {},
                'count': 0,
                'current_price': 0.0,
                'current_volume': 0,
                'charts': [],
            }
    
    # Process stocks in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {executor.submit(process_single_stock, stock): stock for stock in stocks}
        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                result = future.result()
                batch_results.append(result)
            except Exception as e:
                print(f"Exception processing {stock}: {e}")
                # Add empty result for failed stock
                batch_results.append({
                    'stock': stock,
                    'patterns': [],
                    'pattern_counts': {},
                    'count': 0,
                    'current_price': 0.0,
                    'current_volume': 0,
                    'charts': [],
                })
    
    return batch_results

def _run_detect_all_job(job_id: str, req: DetectAllRequest, batch_size: int = 20):
    try:
        # Initial progress setup
        pattern_map = {
            'double_top': 'Double Top',
            'double_bottom': 'Double Bottom',
            'cup_and_handle': 'Cup and Handle',
            'head_and_shoulders': 'Head and Shoulders',
        }

        # Determine source and stock list
        data_source = (getattr(req, 'data_source', None) or 'live').lower()
        def _resolve_data_dir(explicit: str | None) -> str:
            if explicit and os.path.isdir(explicit):
                return explicit
            env = os.environ.get('STOCK_DATA_DIR')
            if env and os.path.isdir(env):
                return env
            root = os.path.dirname(os.path.dirname(__file__))
            cand1 = os.path.join(root, 'StockData')
            cand2 = os.path.join(root, 'STOCK_DATA')
            return cand1 if os.path.isdir(cand1) else cand2
        
        stock_data_dir = _resolve_data_dir(getattr(req, 'stock_data_dir', None))
        
        def list_csv_symbols(data_dir: str) -> list[str]:
            """Return only .NS suffixed symbols for past data to eliminate duplicates."""
            syms: list[str] = []
            try:
                for fn in os.listdir(data_dir):
                    if fn.lower().endswith('.csv'):
                        name = fn[:-4]
                        syms.append(name)
            except Exception:
                pass
            uniq: list[str] = []
            seen: set[str] = set()
            for s in syms:
                variant = f"{s}.NS"
                if variant not in seen:
                    seen.add(variant)
                    uniq.append(variant)
            return uniq
        
        stock_list = list_csv_symbols(stock_data_dir) if data_source == 'past' else AVAILABLE_STOCKS
        
        if data_source == 'past':
            limit = getattr(req, 'stock_limit', None) or 150
            try:
                limit = int(limit)
            except Exception:
                limit = 150
            limit = max(1, min(1000, limit))
            if len(stock_list) > limit:
                stock_list = stock_list[:limit]

        # Validate date range
        if req.start_date >= req.end_date:
            raise ValueError("start_date must be earlier than end_date")

        total_stocks = len(stock_list)
        
        # Initialize progress
        PROGRESS[job_id] = {
            'status': 'running',
            'current': 0,
            'total': total_stocks,
            'symbol': None,
            'message': f'Starting pattern detection for {total_stocks} stocks in batches of {batch_size}...',
            'current_batch': 0,
            'total_batches': (total_stocks + batch_size - 1) // batch_size,
        }

        # Create batches
        batches = []
        for i in range(0, total_stocks, batch_size):
            batch = stock_list[i:i + batch_size]
            batches.append(batch)

        results: List[Dict] = []
        processed_count = 0

        # Process each batch
        for batch_idx, batch in enumerate(batches, 1):
            batch_start_time = time.time()
            
            # Update progress for current batch
            PROGRESS[job_id].update({
                'current_batch': batch_idx,
                'message': f'Processing batch {batch_idx}/{len(batches)} ({len(batch)} stocks)...',
                'symbol': f'Batch {batch_idx}: {batch[0]} to {batch[-1]}',
            })

            # Process the batch in parallel
            batch_results = _process_stock_batch(batch, req, pattern_map, stock_data_dir, data_source)
            results.extend(batch_results)
            
            processed_count += len(batch)
            batch_time = time.time() - batch_start_time
            
            # Update progress after batch completion
            PROGRESS[job_id].update({
                'current': processed_count,
                'message': f'Completed batch {batch_idx}/{len(batches)} in {batch_time:.1f}s. Total processed: {processed_count}/{total_stocks}',
            })

        # Final completion
        RESULTS[job_id] = DetectAllResponse(results=results)
        PROGRESS[job_id].update({
            'status': 'done',
            'current': total_stocks,
            'message': f'Completed processing {total_stocks} stocks in {len(batches)} batches'
        })
        
    except Exception as e:
        PROGRESS[job_id] = {
            'status': 'error',
            'current': PROGRESS.get(job_id, {}).get('current', 0),
            'total': PROGRESS.get(job_id, {}).get('total', len(AVAILABLE_STOCKS)),
            'symbol': PROGRESS.get(job_id, {}).get('symbol', None),
            'message': f'Error: {e}',
            'current_batch': PROGRESS.get(job_id, {}).get('current_batch', 0),
            'total_batches': PROGRESS.get(job_id, {}).get('total_batches', 0),
        }

@app.post('/detect-all-start')
def detect_all_start(req: DetectAllRequest):
    job_id = str(uuid4())
    batch_size = getattr(req, 'batch_size', 20) or 20
    batch_size = max(1, min(50, batch_size))  # Limit batch size between 1 and 50
    
    PROGRESS[job_id] = { 
        'status': 'queued', 
        'current': 0, 
        'total': len(AVAILABLE_STOCKS), 
        'symbol': None, 
        'message': f'Queued with batch size {batch_size}',
        'current_batch': 0,
        'total_batches': 0,
        'batch_size': batch_size
    }
    t = threading.Thread(target=_run_detect_all_job, args=(job_id, req, batch_size), daemon=True)
    t.start()
    return { 'job_id': job_id, 'batch_size': batch_size }

@app.get("/batch-config")
def get_batch_config():
    """Get available batch processing configuration options."""
    return {
        "default_batch_size": 20,
        "min_batch_size": 1,
        "max_batch_size": 50,
        "recommended_batch_sizes": [10, 20, 30, 40, 50],
        "max_workers_per_batch": 5,
        "description": {
            "batch_size": "Number of stocks to process in parallel per batch",
            "smaller_batches": "Smaller batches use less memory but may take longer overall",
            "larger_batches": "Larger batches are faster but use more memory and CPU"
        }
    }

@app.get('/detect-all-progress')
def detect_all_progress(job_id: str):
    p = PROGRESS.get(job_id)
    if not p:
        raise HTTPException(status_code=404, detail='Job not found')
    # Include percent for convenience
    current = int(p.get('current') or 0)
    total = int(p.get('total') or 1)
    percent = int(min(100, max(0, (current / total) * 100)))
    
    # Add batch-specific information
    result = { **p, 'percent': percent }
    if 'current_batch' in p and 'total_batches' in p:
        current_batch = int(p.get('current_batch') or 0)
        total_batches = int(p.get('total_batches') or 1)
        batch_percent = int(min(100, max(0, (current_batch / total_batches) * 100))) if total_batches > 0 else 0
        result.update({
            'batch_percent': batch_percent,
            'batch_info': f"Batch {current_batch}/{total_batches}"
        })
    
    return result

@app.get('/detect-all-result', response_model=DetectAllResponse)
def detect_all_result(job_id: str):
    if PROGRESS.get(job_id, {}).get('status') != 'done':
        raise HTTPException(status_code=202, detail='Not ready')
    res = RESULTS.get(job_id)
    if not res:
        raise HTTPException(status_code=404, detail='Result not found')
    return res

# -----------------------------
# Live ticker-tape endpoint (Home page)
# -----------------------------

@app.get('/ticker-tape')
def ticker_tape(count: int = 20):
    """Return up to `count` live tickers with price, change %, volume spike, and a small sparkline.
    Uses Yahoo Finance. Symbols come from AVAILABLE_STOCKS.
    """
    # Serve cached response when fresh
    now = time.time()
    cached = _TICKER_CACHE.get(int(count))
    if cached and (now - float(cached.get('ts', 0))) < _TICKER_CACHE_TTL_SEC:
        return {'tickers': cached.get('items', [])[:count]}

    # Cap symbols at 150 (was 50)
    symbols = AVAILABLE_STOCKS[: max(1, min(150, count))]
    try:
        hist = yf.download(symbols, period='45d', interval='1d', group_by='ticker', progress=False, threads=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to download data: {e}')

    items = []
    def _display(sym: str) -> str:
        return sym[:-3] if sym.upper().endswith('.NS') else sym

    for sym in symbols:
        try:
            df = None
            if isinstance(hist.columns, pd.MultiIndex):
                if sym in hist.columns.get_level_values(0):
                    df = hist[sym]
            else:
                df = hist
            if df is None or df.empty:
                continue
            df = df.dropna(subset=['Close'])
            if df.empty:
                continue
            closes = df['Close'].tail(20).tolist()
            vols = df['Volume'].dropna()
            last_close = float(df['Close'].iloc[-1]) if not df['Close'].empty else 0.0
            prev_close = float(df['Close'].iloc[-2]) if len(df['Close']) > 1 else last_close
            change_pct = ((last_close - prev_close) / prev_close * 100.0) if prev_close else 0.0
            last_vol = int(vols.iloc[-1]) if not vols.empty else 0
            avg_vol = float(vols.tail(20).mean()) if not vols.empty else 0.0
            price_spike = abs(change_pct) >= 1.5
            volume_spike = (avg_vol > 0 and last_vol >= avg_vol * 1.5)
            items.append({
                'symbol': sym,
                'display_symbol': _display(sym),
                'price': round(last_close, 2),
                'change_pct': round(change_pct, 2),
                'volume': last_vol,
                'avg_volume': int(avg_vol) if avg_vol else 0,
                'price_spike': price_spike,
                'volume_spike': bool(volume_spike),
                'sparkline': closes,
            })
        except Exception:
            continue

    payload = items[:count]
    _TICKER_CACHE[int(count)] = { 'ts': now, 'items': payload }
    return {'tickers': payload}

# Convenience endpoints explicitly matching the requirement wording
@app.post('/detect/live')
def detect_live(req: DetectRequest):
    req.data_source = 'live'
    return detect_patterns(req)

@app.post('/detect/past')
def detect_past(req: DetectRequest):
    req.data_source = 'past'
    return detect_patterns(req)

@app.post('/detect-all-live')
def detect_all_live(req: DetectAllRequest):
    req.data_source = 'live'
    return detect_all_stocks(req)

@app.post('/detect-all-past')
def detect_all_past(req: DetectAllRequest):
    req.data_source = 'past'
    return detect_all_stocks(req)
