<!--
Comprehensive specification & implementation prompt to recreate the Stock Pattern Detector project from scratch.
This document is self‑contained and can be handed to an engineering team or an AI code generator.
-->

# Stock Pattern Detector – Full Rebuild Specification & Implementation Prompt

## 0. Purpose

Recreate a full‑stack system that detects, validates, explains, and visualizes classic technical analysis chart patterns (Double Top, Double Bottom, Cup & Handle, Head & Shoulders) for equities (NSE focus) using both live Yahoo Finance data and historical local CSV data. Provide interactive web UI, programmatic API, pattern validation scoring, target price calculations, and batch scanning with progress tracking.

---

## 1. High‑Level Architecture

Monorepo layout:

```
root/
  backend/        # FastAPI service (REST API for detection, batch jobs, metadata, ticker tape)
  frontend/       # React + Vite + Tailwind single-page app
  detect_all_patterns.py  # Core pattern engine (imported by backend)
  validator/      # Pattern validation & explanation modules
  StockData/      # (Optional) Local historical CSV files (Date, Open, High, Low, Close, Volume)
  outputs/        # Generated charts (HTML) & reports (CSV)
  tests/ or test_*.py  # Validation and filtering tests
```

Primary components:

1. Pattern Engine (Python) – downloads or loads OHLCV data, extracts swing points, detects patterns, validates & annotates, produces Plotly charts + structured metadata.
2. Backend API (FastAPI) – exposes detection endpoints (single + bulk), metadata lists, ticker tape live mini‑quotes, asynchronous batch job progress.
3. Frontend UI (React) – three views: Home (ticker tape + marketing), Detect (single symbol scan with charts), Dashboard (bulk scan filtering, charts, pagination, strength filters).
4. Validators / Explainers – rule‑based scoring, strict vs lenient modes, multi‑criterion evaluation, generation of explanation objects with rules & target price breakdown.
5. Storage – ephemeral; charts persisted as HTML in `outputs/charts`, reports CSV in `outputs/reports`. No DB required.
6. Caching – in‑memory (Python dict) for ticker tape responses & stock lists; frontend in‑memory & localStorage caching for noise reduction.

---

## 2. Functional Requirements

### Core

1. Detect four pattern classes: Head & Shoulders (HNS), Cup & Handle (CH), Double Top (DT), Double Bottom (DB).
2. Support multiple rolling timeframes or custom date ranges.
3. Two detection modes: `lenient` (broader acceptance) and `strict` (textbook pattern rules).
4. Generate annotated Plotly charts (HTML) with pattern points and breakout markers.
5. Provide pattern validation scoring & strength classification (`strong` / `weak`).
6. Provide explanatory rule breakdown and target price calculation for each pattern (where applicable).
7. Single symbol detection endpoint returning at most one representative chart per timeframe (grouped by strength).
8. Bulk (all symbols) detection with asynchronous job & progress polling.
9. Dual data sources: `live` (Yahoo Finance via yfinance) and `past` (local CSV library directory).
10. Home ticker tape endpoint returning price, percent change, volume spike flag, and sparkline (recent closes).

### Auxiliary

11. API endpoints to list available stocks, stock groups, patterns, timeframes, chart types, modes.
12. Parameter: chart type (`candle`, `line`, `ohlc`).
13. Limit number of patterns saved per timeframe (`max_patterns_per_timeframe`).
14. Optional preceding trend requirement toggle.
15. Option to keep only “best” pattern per timeframe.
16. CSV report summarizing all detected patterns across run.
17. Background job progress fields: status, current, total, symbol, message, percent.
18. Frontend filtering: pattern type, stock, strength, date range.
19. Local caching of bulk results (keyed by source|dates|mode|chart_type|limit).
20. Download chart as PNG (client‑side via Plotly).

### Non‑Functional

21. FastAPI app with permissive CORS (configurable for prod).
22. Python 3.11+ recommended (tested with modern pandas/numpy).
23. Deterministic detection given same input dataset & parameters.
24. Graceful handling of missing or malformed CSVs.
25. Minimal external dependencies; no database requirement.
26. Ability to run headless (matplotlib Agg backend) for CI.

---

## 3. Data Sources & Loading

### Live Mode

Use `yfinance.download(symbol, start, end, progress=False)`; sanitize MultiIndex returns; ensure Date column is UTC normalized; reject empty frame.

### Past Mode (CSV)

Look up file by stripping suffixes (.NS, .BS, etc.) with case‑insensitive search. Required columns: Date, Open, High, Low, Close, Volume. Sort ascending by Date. Parse dates robustly; drop rows with NaT.

### Ticker Tape

Batch download last ~45 days daily data for subset of symbols (cap 150). Derive:

- last_close, prev_close → change_pct
- recent 20 closes → sparkline vector
- last volume vs 20‑day avg → volume_spike (>=1.5×)
  Cache per requested count for ~45 seconds.

---

## 4. Pattern Detection Engine

Central module: `detect_all_patterns.py` exports:

- `process_symbol(...)` (implied internal orchestrator) – parameters include symbol, patterns list, timeframes, mode, swing method, date bounds, data source, output dirs.
- Config dictionaries per pattern (STRICT & LENIENT) controlling tolerances (e.g. `HNS_CONFIG`, `DT_CONFIG`, `CH_CONFIG`).
- Swing point generation strategies: rolling window, zigzag (% threshold), fracture (Williams fractals).
- Dynamic volatility‑scaled window size using ATR (`compute_dynamic_nbars`).
- Indicators: RSI, MACD; divergence checks for additional filters (esp. Double Tops).
- Preceding trend validation (≥10–15% move over lookback days).

### Shared Concepts

1. Identify candidate structural points (P1, T1, P2, T2, P3 for HNS; P1,T,P2 for Double; left_rim, cup_bottom, right_rim, handle_low for Cup & Handle).
2. Validate relative price relationships (prominence, symmetry, depth, spacing).
3. Neckline/resistance computation (two troughs or rims) & angle constraint.
4. Breakout detection – price closing beyond neckline/resistance within time window (mode‑dependent) plus volume spike rule.
5. Duration constraints (minimum & maximum days) differing by mode.
6. Result dictionary contains: `type`, coordinate tuples, breakout tuple, derived metrics (duration, slopes), optional `validation`, optional `explanation`, `image_path` to HTML chart.

### Head & Shoulders Highlights

- Shoulder similarity tolerance; head prominence threshold; adaptive leniency for longer formations.
- Neckline slope angle limit (stricter in strict mode).
- Accept forming patterns (no breakout yet) if pattern end is recent.

### Double Top / Bottom Highlights

- Peak/trough similarity tolerance (3–8% strict; up to 15% lenient).
- Spacing window (ideal 20–90 days; extended 10–150 lenient; up to 200 days for detection pre‑validation).
- Valley/peak depth threshold (≥10%).
- Enhanced validator uses 7 criteria including breakout depth, volume 2× average, support/resistance proximity.

### Cup & Handle Highlights

- Cup depth 8–60% (15–40% “strong”).
- Rim height difference ≤10% (≤5% strong).
- Symmetry (time asymmetry ≤50%; ≤25% strong).
- Handle depth ≤35% (≤20% strong).
- Breakout ≥1% over resistance + volume spike.
- Target = Resistance + (Resistance – Cup bottom).

---

## 5. Validation & Explanation Layer

Directory `validator/` provides specialized modules:

- `validate_double_patterns.py` – scoring (0–7) for double tops/bottoms, multi‑timeframe boost option, confirmation metrics.
- `validate_hns.py` – adaptive strict/lenient scoring (0–6) with major failure categorization.
- `validate_cup_handle.py` (implied) – depth, handle, symmetry, breakout.
- `explain_patterns.py` – generic explanation builder returning: verdict, score/max_score, rules array (name, value, expected, pass, notes), target object (formula, steps, target_price). These explanation objects feed frontend “Details” panels.

### Strength Classification

Backend groups charts into `strong_charts` vs `weak_charts`: a pattern is strong if validator or explainer verdict indicates valid / strong (e.g. `is_valid` or verdict ∈ {valid,strong,true}).

---

## 6. Chart Generation

Prefer Plotly interactive HTML for each detected pattern. Each HTML saved to: `outputs/charts/{symbol}/{timeframe}/{pattern_type}/...html` (customizable base dirs). Patterns limited per timeframe; if `keep_best_only` true, select highest scoring pattern (criteria: confidence / depth / breakout presence).

Each HTML must embed Plotly JS so that frontend can iframe the chart without additional network calls. Provide stable container class `.js-plotly-plot` for snapshot download.

---

## 7. Backend API Specification

Base: FastAPI JSON.

| Method | Path                 | Description                                         | Request          | Response                     |
| ------ | -------------------- | --------------------------------------------------- | ---------------- | ---------------------------- |
| GET    | /stocks              | Flat list of unique symbols                         | –                | [str]                        |
| GET    | /stock-groups        | Original grouped symbols                            | –                | dict[group->list]            |
| GET    | /patterns            | Human pattern names                                 | –                | ["Double Top", ...]          |
| GET    | /timeframes          | Ordered timeframe keys                              | –                | [str]                        |
| GET    | /chart-types         | Supported chart styles                              | –                | ["candle","line","ohlc"]     |
| GET    | /modes               | Detection modes                                     | –                | ["lenient","strict"]         |
| POST   | /detect              | Single symbol detection (timeframe OR custom dates) | DetectRequest    | charts + grouped strong/weak |
| POST   | /detect/live         | Force live data source                              | DetectRequest    | same                         |
| POST   | /detect/past         | Force past data source                              | DetectRequest    | same                         |
| POST   | /detect-all          | Synchronous bulk (legacy)                           | DetectAllRequest | DetectAllResponse            |
| POST   | /detect-all-start    | Start async bulk job                                | DetectAllRequest | { job_id }                   |
| GET    | /detect-all-progress | Poll job                                            | job_id query     | progress object + percent    |
| GET    | /detect-all-result   | Fetch async result                                  | job_id query     | DetectAllResponse            |
| GET    | /ticker-tape         | Live mini quotes                                    | count query      | { tickers: [...] }           |

### DetectRequest Schema

```
stock: string
pattern: string (human label) – or internal will map to snake_case
chart_type: candle|line|ohlc
timeframe?: string (if no date range)
start_date?: YYYY-MM-DD
end_date?: YYYY-MM-DD
mode?: lenient|strict (default lenient)
data_source?: live|past (default live)
stock_data_dir?: optional override path
```

### DetectAllRequest Schema

```
start_date: string
end_date: string
chart_type?: candle|line|ohlc
mode?: lenient|strict
data_source?: live|past
stock_data_dir?: path
stock_limit?: int (past mode safety bound 1–1000, default 150)
```

### Response Snippets

Single detect:

```
{
  "charts": [ { timeframe, html, strength, explanation? } ],
  "strong_charts": [...],
  "weak_charts": [...]
}
```

Bulk:

```
{
  "results": [
    {
      "stock": "TCS.NS",
      "patterns": ["Double Top", "Head and Shoulders"],
      "pattern_counts": { "Double Top": 1 },
      "current_price": 3945.5,
      "current_volume": 1234567,
      "count": 2,
      "charts": [ { timeframe, pattern, html, strength, explanation? } ]
    }
  ]
}
```

Progress:

```
{ status: running|queued|done|error, current: int, total: int, symbol: str|null, message: str, percent: 0-100 }
```

---

## 8. Frontend Application (React + Vite + Tailwind)

### Pages

1. Home – Animated ticker tape (marquee), hero with feature cards, fallback skeletons, uses `/ticker-tape` & localStorage caching.
2. Detect – Sidebar selection: symbol search, data source switch, pattern, mode, timeframe OR date range presets (YTD, 6M, 1Y), chart type, run detection; central results grouped by strong/weak with expandable details & download buttons.
3. Dashboard – Bulk scan interface: run asynchronous job, show progress bar, table view (aggregate pattern counts), chart view (per‑stock collapsible charts), filters (pattern, stock, strength), caching of results in memory keyed by parameter string, pagination logic (not shown in snippet but design supports pages map), ability to clear cache.

### State Management

Uses `zustand` store for Detect page (patterns, symbols, charts, loading flags). Local ephemeral state for Dashboard caching.

### HTML Chart Embedding

Charts returned as raw HTML; rendered in iframe / sandbox container component (`HtmlPanel`). Provide ability to drill into rule breakdown & target steps via `<details>` disclosure.

### Styling & UX

Tailwind CSS for utility classes; skeleton loading states; grouped strong vs weak sections; accessible `<details>` semantics; sticky header navigation.

### Performance Client‑Side

- Simple in‑memory + localStorage caching for metadata & ticker tape.
- Avoids re-fetching static endpoints for 5 minutes (configurable TTL).
- Only downloads PNG on explicit user request (Plotly’s `downloadImage`).

---

## 9. Pattern Explanation Object Contract

Each explanation dict (from `explain_patterns.py`) should include:

```
pattern_type: str (human)
verdict: 'valid' | 'weak'
score: int
max_score: int
rules: [ { name, value, expected, passed, notes? } ]
target: { formula, steps: [str], target_price?: float|null }
```

Frontend displays rule list with color coding and target breakdown.

---

## 10. Strength Determination Logic

Define `is_strong` as:

```
is_strong = (validation.is_valid == True) OR (explanation.verdict in {'valid','strong','true'})
```

Return two arrays: `strong_charts` and `weak_charts`. If neither grouping is relevant, fallback to `charts` only.

---

## 11. Configuration & Constants

Summaries:

- `HNS_CONFIG` strict vs lenient: shoulder tolerance, head prominence, neckline angle, volume multipliers.
- `DT_CONFIG` strict vs lenient: peak similarity tolerance, spacing day limits, volume decline & spike multipliers, neckline tolerance.
- `CH_CONFIG` depth ranges, symmetry, handle constraints, volume decline/spike.
- `GLOBAL_CONFIG`: base bars for swing detection, zigzag percent, min trend percent (15%), duration bounds.
- `TIMEFRAMES` mapping: { '1m': 30, '6m': 180, ... } to days count.

---

## 12. Error Handling & Edge Cases

1. Empty data fetch → skip symbol gracefully.
2. Invalid date range (start >= end) → 400 error response.
3. Missing CSV columns → ignore symbol in past mode.
4. Duplicate symbols across groups → deduplicate with order preservation.
5. Pattern detection producing overlapping duplicates → keep first per (timeframe, pattern) combination.
6. HTML chart file missing during response assembly → skip silently.
7. Yahoo Finance network errors → log & continue; zero price/volume fallback.
8. Volume NaNs → proceed with pattern but mark volume-based checks as uncertain.

---

## 13. Testing Strategy

Current test artifacts illustrate validation logic. Expand to:

- Unit tests for each validator (valid & invalid pattern dicts).
- Integration test: run single detection on small synthetic dataset, assert at least one expected pattern.
- Regression tests for strict vs lenient thresholds (e.g., pattern accepted in lenient but rejected in strict).
- Performance test (time budget) scanning N symbols to detect regressions.
- HTML artifact presence & valid UTF‑8.

Provide fixtures for synthetic OHLCV sequences creating textbook patterns + edge variants (e.g., almost double top failing similarity rule).

---

## 14. Build & Run Instructions

### Backend

```
python -m venv .venv
source .venv/bin/activate  (Windows: .venv\Scripts\activate)
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Environment variables:

- `STOCK_DATA_DIR` – override CSV directory for past mode.
- (Optional) `PORT`, `HOST` for deployment wrapper script.

### Frontend

```
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE` in `.env` if backend not on default `http://127.0.0.1:8000`.

### Directory Creation

On detection runs ensure existence of `outputs/charts` & `outputs/reports`. Create nested symbol/timeframe paths on demand.

---

## 15. Performance & Scaling Considerations

| Concern                   | Current Approach                                  | Potential Enhancement                                |
| ------------------------- | ------------------------------------------------- | ---------------------------------------------------- |
| Bulk scanning speed       | Sequential loop per symbol                        | Thread pool / process pool for IO + CPU split        |
| Repeated yfinance calls   | On‑demand per symbol/timeframe                    | Batch caching layer (e.g., per symbol daily cache)   |
| Memory (HTML charts)      | All charts in memory only when returning response | Stream or paginate charts endpoint                   |
| CPU for pattern detection | Pure Python loops & pandas ops                    | Vectorization, numba, or modular Cython for hotspots |
| Ticker tape               | Cached for 45s                                    | Adaptive refresh (ET market hours)                   |

Add concurrency guard to avoid overlapping heavy detect‑all jobs. Optionally reject second job when one is running.

---

## 16. Security & Hardening

- Restrict CORS origins in production.
- Validate symbol inputs (allowlist). Reject path traversal in `stock_data_dir` (resolve & confirm under allowed root).
- Limit maximum date range span (e.g., ≤ 5y) to prevent large downloads.
- Consider rate limiting on `/detect` and `/detect-all-start`.
- Sanitize HTML (only self‑generated); trust boundary is internal generation – avoid user‑supplied HTML injection.

---

## 17. Logging & Observability

Basic `print()` now; upgrade to `logging` with structured JSON (fields: symbol, timeframe, pattern_type, duration, score, mode). Add timing metrics for each symbol.

Optional future: Prometheus counters (patterns_detected_total, detection_seconds_histogram) & health endpoint.

---

## 18. Extensibility Guidelines

Add new pattern (e.g., Ascending Triangle):

1. Implement detection function returning pattern dict keys consistent with existing style (`type`, point tuples, breakout info).
2. Add config constants (STRICT/LENIENT tolerance sets).
3. Extend `process_symbol` pattern dispatch.
4. Implement validator and explanation rule builder; map to `explain_patterns` if generic.
5. Update `/patterns` endpoint & frontend pattern selector.
6. Add tests & sample synthetic dataset.

---

## 19. Example Detection Flow (Single Symbol)

1. Receive POST `/detect` payload `{ stock:'TCS.NS', pattern:'Double Top', timeframe:'6m', chart_type:'candle', mode:'strict' }`.
2. Resolve timeframe → start/end dates (today - 180d .. today).
3. Load data (yfinance or CSV) & compute dynamic N_bars.
4. Generate swing flags (`rolling`, default N_bars scaled by ATR).
5. Run pattern algorithm for double patterns: locate first peak, trough, second peak satisfying similarity & spacing.
6. Compute neckline, breakout detection window, volume spike.
7. Validate pattern (score >= threshold) → attach `validation` & `explanation`.
8. Build Plotly figure (candles + pattern markers + neckline + breakout annotation) & save HTML.
9. Return JSON with embedded HTML & explanation classification.

---

## 20. Sample Pattern Dict (Double Top)

```
{
  'type': 'double_top',
  'P1': (date1, price1, idx1),
  'T': (trough_date, trough_price, trough_idx),
  'P2': (date2, price2, idx2),
  'breakout': (br_date, br_price, br_idx) | None,
  'neckline_level': float,
  'duration': int,  # days from P1 to P2
  'validation': { score, is_valid, confidence, ... },
  'explanation': { ... },
  'image_path': 'outputs/.../chart.html'
}
```

---

## 21. Reporting CSV (Conceptual Columns)

| symbol | timeframe | pattern_type | mode | start_date | end_date | duration_days | breakout_confirmed | confidence | score | neckline | target_price | volume_spike | created_at |

Append on each run; timestamp for uniqueness.

---

## 22. Developer Workflow

1. Clone repo & install backend requirements.
2. Add/prepare `StockData/*.csv` if using past mode.
3. Run backend & confirm `/docs` (Swagger UI) loads.
4. Run frontend dev server; set `VITE_API_BASE` if needed.
5. Execute sample detection (e.g., `curl POST /detect`).
6. Add tests & run `pytest` (after organizing tests folder) or direct script invocation.
7. Commit & tag releases with semantic versioning (e.g., v0.2.0 when adding explanation engine).

---

## 23. Deployment Outline

Backend:

```
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Serve behind reverse proxy (nginx) enabling compression (HTML charts can be large). Frontend built with `npm run build` → deploy `dist/` via static host / CDN.

Persistent outputs volume if you need historical chart retention.

---

## 24. Potential Future Enhancements

| Idea                        | Description                                  | Value                          |
| --------------------------- | -------------------------------------------- | ------------------------------ |
| Multiprocessing detection   | Parallel symbol scanning                     | Large speedup for >100 symbols |
| WebSocket progress          | Real-time progress push vs polling           | UX improvement                 |
| Pattern backtesting         | Compute post‑breakout performance statistics | Strategy validation            |
| User accounts & saved scans | Persist queries & favorites                  | Personalization                |
| Strategy rule builder       | Combine pattern + volume + RSI filters       | Advanced screening             |
| Lightweight DB (SQLite)     | Persist detection history                    | Audit & analytics              |
| Alerting engine             | Email/webhook on new strong pattern          | Timely action                  |
| AI ranking model            | ML classification of pattern quality         | Reduced false positives        |

---

## 25. Acceptance Criteria Checklist

- [ ] All endpoints implemented & match schema.
- [ ] Detection engine returns structured pattern dicts with charts.
- [ ] Strict vs lenient modes produce differing counts.
- [ ] Explanations & validation objects embedded in responses.
- [ ] Ticker tape returns sparkline arrays & spike flags.
- [ ] Frontend Detect page renders strong & weak groups with details.
- [ ] Dashboard handles async job progress & caching.
- [ ] Past mode loads CSVs with suffix stripping.
- [ ] Pagination / limiting works (stock_limit & max_patterns_per_timeframe).
- [ ] Tests cover validator logic & key edge cases.

---

## 26. Concise Implementation Prompt (If Using an AI Generator)

"""
Implement a monorepo stock pattern detection platform with:

- Python FastAPI backend (endpoints: /stocks, /patterns, /timeframes, /chart-types, /modes, /detect, /detect-all-start, /detect-all-progress, /detect-all-result, /ticker-tape) using pydantic models.
- Pattern engine module supporting Head & Shoulders, Cup & Handle, Double Top/Bottom with strict & lenient configs, swing point extraction (rolling, zigzag, fractal), volume & breakout validation, Plotly chart HTML generation, explanation objects (rules + target formula).
- Async bulk detection job with in-memory progress & result caches keyed by UUID.
- Data sources: live via yfinance, past via local CSV directory (env STOCK_DATA_DIR). Symbol grouping & deduplicated list exported.
- Frontend React (Vite + TypeScript + Tailwind): Home (ticker tape), Detect (single symbol scan), Dashboard (bulk scan with filters, caching, table & chart views). Use zustand for detection store. Render backend-provided HTML charts inside iframes and show validation rules & target calculations.
- Strength classification into strong/weak based on validation/explanation validity.
- Tests for double top filtering and HNS validation thresholds.
  Ensure code style clean, modular, and each pattern algorithm documented. Provide build scripts, requirements.txt, and README with usage instructions."""

---

## 27. Final Notes

This specification encapsulates the original system’s observable behaviors, data contracts, algorithms, and UX patterns. Implementations may internally optimize, but external API, pattern semantics, and front-end interactions should remain stable for compatibility.

---

End of Specification.
