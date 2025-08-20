# head_and_shoulder_pattern_detector

A simple Python-based scanner that downloads historical OHLCV data (via yfinance) and searches for chart reversal patterns: Head & Shoulders, Double Top, and Double Bottom. It saves zoomed pattern charts (with volume) and a CSV summary of detected patterns.

## Features
- Scan multiple tickers (default set of popular NSE tickers).
- Detect Head & Shoulders and Double Top / Double Bottom patterns.
- Save per-pattern PNGs (price + volume) organized into `PatternCharts/<SYMBOL>/<window>/{HNS,DT,DB}`.
- Save a CSV summary per run mode: `pattern_detection_summary_<mode>.csv`.
- Two detection modes: `lenient` (legacy, looser rules) and `strict` (stricter rules).
- Several tunable parameters at the top of `detect_head_and_shoulders.py` for thresholding and behavior.

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- yfinance

Install dependencies (recommended in a virtualenv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install directly:

```powershell
pip install pandas numpy matplotlib yfinance
```

## Quick start

Run the detector in lenient mode (looser rules, useful for exploratory runs):

```powershell
python detect_head_and_shoulders.py --mode lenient
```

Run in strict mode (less false positives):

```powershell
python detect_head_and_shoulders.py --mode strict
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

## License
This repository contains example code. Check and comply with Yahoo/Google/Exchange data licensing when using downloaded price data in production.

---
Happy pattern hunting — open an issue or share a sample symbol/date if you'd like me to tune the detector for specific false positives.
