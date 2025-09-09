# Stock Pattern Detector

A comprehensive Python script that automatically detects technical chart patterns in stock data with intelligent defaults and extensive customization options.

## 🚀 Quick Start

### Instant Pattern Detection

```bash
# Just run it - automatically selects random symbols and scans for all patterns
python3 detect_all_patterns.py
```

This will:

- 🎲 Randomly select 5-10 symbols from global markets (US, Indian, Crypto)
- 🔍 Scan ALL pattern types (Head & Shoulders, Cup & Handle, Double Top/Bottom)
- 📊 Check multiple timeframes (6m, 1y, 2y, 3y, 5y)
- 📁 Organize results in clean folder structure: `outputs/charts/` and `outputs/reports/`

### Installation

```bash
pip install yfinance pandas numpy matplotlib
```

## 📊 Pattern Types Detected

| Pattern              | Description                  | Use Case                      |
| -------------------- | ---------------------------- | ----------------------------- |
| **Head & Shoulders** | Classic reversal pattern     | Bearish trend reversals       |
| **Cup & Handle**     | Bullish continuation pattern | Breakout opportunities        |
| **Double Top**       | Bearish reversal pattern     | Resistance level confirmation |
| **Double Bottom**    | Bullish reversal pattern     | Support level confirmation    |

## 🎯 Symbol Options

### Predefined Symbol Sets

```bash
# US Technology stocks (AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META, NFLX, AMD)
python3 detect_all_patterns.py --symbols us_tech

# US Large caps & ETFs (SPY, QQQ, IWM, DIA, VTI, JPM, JNJ, PG, KO, WMT)
python3 detect_all_patterns.py --symbols us_large

# Popular Indian stocks (RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, etc.)
python3 detect_all_patterns.py --symbols indian_popular

# Indian Banking sector (HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, etc.)
python3 detect_all_patterns.py --symbols indian_banking

# Cryptocurrencies (BTC-USD, ETH-USD, BNB-USD, XRP-USD, ADA-USD, SOL-USD)
python3 detect_all_patterns.py --symbols crypto
```

### Custom Symbol Selection

```bash
# Specific symbols
python3 detect_all_patterns.py --symbols AAPL,GOOGL,TSLA

# Random selection with custom count
python3 detect_all_patterns.py --symbols random --random-count 8

# From file (one symbol per line)
python3 detect_all_patterns.py --custom-symbols my_symbols.txt
```

## ⏰ Timeframe Options

| Code | Period   | Best For             |
| ---- | -------- | -------------------- |
| `1d` | 1 day    | Intraday patterns    |
| `1w` | 1 week   | Short-term swings    |
| `2w` | 2 weeks  | Bi-weekly analysis   |
| `1m` | 1 month  | Monthly patterns     |
| `2m` | 2 months | Bi-monthly trends    |
| `3m` | 3 months | Quarterly analysis   |
| `6m` | 6 months | Semi-annual patterns |
| `1y` | 1 year   | Annual patterns      |
| `2y` | 2 years  | Medium-term trends   |
| `3y` | 3 years  | Long-term analysis   |
| `5y` | 5 years  | Historical patterns  |

```bash
# Single timeframe
python3 detect_all_patterns.py --timeframes 1y

# Multiple timeframes
python3 detect_all_patterns.py --timeframes 1y,2y,3y

# All timeframes
python3 detect_all_patterns.py --timeframes 1d,1w,2w,1m,2m,3m,6m,1y,2y,3y,5y
```

## 🎛️ Detection Modes

### Strict Mode (Default)

- Higher precision, fewer false positives
- More stringent pattern validation
- Recommended for trading decisions

### Lenient Mode

- Higher sensitivity, more patterns detected
- Looser validation criteria
- Good for pattern discovery and learning

### Both Modes

- Runs both strict and lenient detection
- Generates separate reports for comparison

```bash
# Strict mode (default)
python3 detect_all_patterns.py --mode strict

# Lenient mode
python3 detect_all_patterns.py --mode lenient

# Both modes
python3 detect_all_patterns.py --mode both
```

## 📁 Output Organization

### Default Structure

```
outputs/
├── charts/                    # Pattern visualization charts
│   └── SYMBOL/
│       └── TIMEFRAME/
│           └── PATTERN_TYPE/  # HNS, CH, DT, DB
│               └── SYMBOL_PATTERN_TIMEFRAME_N.png
└── reports/                   # CSV analysis reports
    └── pattern_detection_all_MODE_TIMESTAMP.csv
```

### Custom Organization

```bash
# Date-organized structure
python3 detect_all_patterns.py --organize-by-date
# Creates: outputs/2025-08-22/charts/ and outputs/2025-08-22/reports/

# Custom directory names
python3 detect_all_patterns.py --charts-subdir images --reports-subdir data

# Custom base directory
python3 detect_all_patterns.py --output-dir ./my_analysis
```

## 🔧 Advanced Options

### Pattern Selection

```bash
# All patterns (default)
python3 detect_all_patterns.py --patterns all

# Specific patterns
python3 detect_all_patterns.py --patterns head_and_shoulders,double_top

# Single pattern
python3 detect_all_patterns.py --patterns cup_and_handle
```

### Performance Controls

```bash
# Limit patterns per timeframe (prevents too many results)
python3 detect_all_patterns.py --max-patterns-per-timeframe 5

# Minimum patterns required to save results
python3 detect_all_patterns.py --min-patterns 10

# Add delay between symbol processing (be nice to APIs)
python3 detect_all_patterns.py --delay 2
```

### Detection Tuning

```bash
# Swing detection methods
python3 detect_all_patterns.py --swing-method rolling    # Default
python3 detect_all_patterns.py --swing-method zigzag    # Alternative
python3 detect_all_patterns.py --swing-method fractal   # Advanced

# Disable preceding trend requirement (more patterns, less reliable)
python3 detect_all_patterns.py --no-preceding-trend
```

### Output Customization

```bash
# Custom CSV filename
python3 detect_all_patterns.py --output-csv my_analysis.csv

# Dry run (see what would be processed without running)
python3 detect_all_patterns.py --dry-run --verbose

# Verbose output (detailed progress info)
python3 detect_all_patterns.py --verbose
```

## 📋 Common Usage Examples

### Daily Trading Setup

```bash
# Quick scan of US tech stocks for recent patterns
python3 detect_all_patterns.py \
  --symbols us_tech \
  --timeframes 1w,1m,3m \
  --mode strict \
  --max-patterns-per-timeframe 3
```

### Weekly Portfolio Review

```bash
# Comprehensive analysis with organized output
python3 detect_all_patterns.py \
  --symbols indian_popular \
  --patterns all \
  --timeframes 1m,3m,6m \
  --mode both \
  --organize-by-date \
  --output-dir ./weekly_review
```

### Sector Analysis

```bash
# Banking sector deep dive
python3 detect_all_patterns.py \
  --symbols indian_banking \
  --timeframes 1y,2y,3y \
  --mode lenient \
  --min-patterns 15
```

### Crypto Market Scan

```bash
# Crypto patterns with relaxed requirements
python3 detect_all_patterns.py \
  --symbols crypto \
  --timeframes 1w,1m,3m \
  --mode lenient \
  --no-preceding-trend \
  --max-patterns-per-timeframe 5
```

### Research & Discovery

```bash
# Random market exploration
python3 detect_all_patterns.py \
  --symbols random \
  --random-count 15 \
  --timeframes 6m,1y,2y \
  --mode both \
  --organize-by-date
```

## 🛠️ Command Reference

### Essential Commands

```bash
# Show all available symbol sets
python3 detect_all_patterns.py --list-symbols

# Preview what will be processed (no actual execution)
python3 detect_all_patterns.py --dry-run --verbose

# Get detailed help
python3 detect_all_patterns.py --help
```

### File Management

The script automatically:

- ✅ Creates organized directory structures
- ✅ Generates timestamped reports
- ✅ Removes empty directories
- ✅ Never overwrites existing files

### Performance Tips

- Use `--max-patterns-per-timeframe` to limit results
- Add `--delay` when processing many symbols
- Use `--dry-run` to test configurations
- Check `--verbose` output for detailed progress

## 📊 Output Files

### Chart Files (PNG)

- High-quality pattern visualizations
- Price action with volume overlay
- Pattern-specific annotations
- Organized by symbol/timeframe/pattern type

### Report Files (CSV)

Comprehensive data including:

- Symbol and timeframe
- Pattern type and confidence
- Key price levels (neckline, resistance, support)
- Pattern dates and duration
- Volume analysis
- Breakout confirmation status

## 🔍 Pattern Detection Details

### Head & Shoulders

- Identifies left shoulder, head, right shoulder
- Validates neckline and volume patterns
- Confirms breakout below neckline

### Cup & Handle

- Detects cup formation with proper depth
- Validates handle retracement
- Confirms breakout above resistance

### Double Top/Bottom

- Finds two peaks/troughs at similar levels
- Validates volume decline on second test
- Confirms breakout through support/resistance

All patterns include:

- Volume confirmation analysis
- Preceding trend validation
- Breakout confirmation
- Configurable sensitivity levels

If you don't have a `requirements.txt`, install directly:

```powershell
pip install pandas numpy matplotlib yfinance
```

## Quick Start

### Head & Shoulders Detection

Run the detector in lenient mode (looser rules, useful for exploratory runs):

```bash
python detect_head_and_shoulders.py --mode lenient
```

Run in strict mode (fewer false positives):

```bash
python detect_head_and_shoulders.py --mode strict
```

Detect only specific patterns:

```bash
python detect_head_and_shoulders.py --mode strict --pattern head_and_shoulders
python detect_head_and_shoulders.py --mode strict --pattern double_top
```

### Cup & Handle Detection

```bash
python detect_cup_and_handle.py --mode lenient
python detect_cup_and_handle.py --mode strict
python detect_cup_and_handle.py --mode both  # Run both modes
```

### Double Patterns Detection

```bash
python detect_double_patterns.py --mode strict --pattern all
python detect_double_patterns.py --mode strict --pattern double_top
python detect_double_patterns.py --mode strict --pattern double_bottom
python detect_double_patterns.py --mode both --pattern all
```

### Test All Scripts

Run a quick test to verify all scripts are working:

```bash
python test_all_scripts.py
```

Run both modes and compare outputs:

```powershell
python detect_head_and_shoulders.py --mode both
```

By default the script scans a list of ~25 NSE tickers and four historical windows (1y/2y/3y/5y). Edit the `symbols` and `windows` variables in `detect_head_and_shoulders.py` to customize.

## Outputs

- `PatternCharts/<SYMBOL>/<window>/HNS/` — saved PNGs for Head & Shoulders patterns.
- `PatternCharts/<SYMBOL>/<window>/DT/` — saved PNGs for Double Top patterns.
- `PatternCharts/<SYMBOL>/<window>/DB/` — saved PNGs for Double Bottom patterns.
- `pattern_detection_summary_<mode>.csv` — CSV summary listing detected patterns and the image paths.

## Key parameters and where to tune them

Open `detect_head_and_shoulders.py` and modify constants near the top:

- `SHOULDER_TOL` — how similar the two shoulders/bottoms must be (fraction).
- `HEAD_OVER_SHOULDER_PCT` — how much higher the head must be vs shoulders.
- `MAX_NECKLINE_ANGLE_DEG` — maximum allowed neckline tilt in degrees.
- `VOLUME_TREND_TOL` — required volume decay across formation.
- `VOLUME_SPIKE_MULT` — multiplier to consider a breakout volume as a spike.
- `BASE_NBARS`, `MIN_NBARS`, `MAX_NBARS` — controls swing-detection lookback; dynamic scaling via ATR is used.
- Feature flags: `USE_ZIGZAG`, `USE_FRACTAL`, `REQUIRE_VOLUME_DIVERGENCE`, `REQUIRE_ADX`, etc.

Tune these conservatively: stricter settings reduce false positives but may miss valid setups.

## Tuning tips

- If you see many false positives for double patterns, raise `VOLUME_SPIKE_MULT` and increase the prominence threshold in the script.
- If patterns vanish after tightening, run `--mode lenient` to reproduce earlier, looser detections and compare side-by-side.
- Use a single-symbol test while tuning:

```powershell
# (edit the script to set `symbols = ['RELIANCE.NS']` temporarily)
python detect_head_and_shoulders.py --mode lenient
```

## Troubleshooting

- If you see a `FutureWarning` from yfinance about `download()` defaults, it's non-fatal. To silence, call `yf.download(..., auto_adjust=True)` explicitly.
- If matplotlib raises GUI/backend errors in non-GUI environments, the script sets the `Agg` backend to avoid Tkinter issues.

## Next improvements (suggested)

- Add unit tests to validate detection rules on labelled historical patterns.
- Add a logging/debug mode that writes rejection reasons for each candidate to help tuning.
- Add a CLI to override symbols, windows, and tuning parameters without editing the file.

## 🚀 Deployment

### Backend Deployment (Render)

The FastAPI backend can be deployed to Render for free hosting:

**Build Command:**
```bash
pip install -r backend/requirements.txt
```

**Start Command:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

**Environment Variables (set in Render dashboard):**
- `PORT`: Automatically set by Render
- `CORS_ORIGINS`: Your frontend domain (e.g., `https://your-app.vercel.app`)
- `STOCK_DATA_DIR`: `/tmp/stock_data` (optional)
- `PYTHONPATH`: `/app` (optional)

**Local Testing:**
```bash
# Test the production configuration locally
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend Deployment (Vercel)

The React frontend can be deployed to Vercel:

1. Connect your GitHub repository to Vercel
2. Set the root directory to `frontend/`
3. Set environment variable: `VITE_API_URL=https://your-backend.onrender.com`
4. Deploy automatically on push to main branch

## License

This repository contains example code. Check and comply with Yahoo/Google/Exchange data licensing when using downloaded price data in production.

---

Happy pattern hunting — open an issue or share a sample symbol/date if you'd like me to tune the detector for specific false positives.
