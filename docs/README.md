# Pattern Detection Algorithms Overview

## Summary

This project implements four classic technical analysis patterns with configurable detection algorithms and comprehensive validation rules.

## Implemented Patterns

### 1. [Head and Shoulders](./head_and_shoulders.md)

- **Type**: Bearish reversal pattern
- **Structure**: Left Shoulder → Head → Right Shoulder
- **Key Features**: Neckline analysis, volume confirmation, breakout validation
- **Output**: HNS folder with detailed charts

### 2. [Cup and Handle](./cup_and_handle.md)

- **Type**: Bullish continuation pattern
- **Structure**: Cup formation → Handle retracement → Breakout
- **Key Features**: Symmetry analysis, handle validation, resistance breakout
- **Output**: CH folder with cup outline visualization

### 3. [Double Top](./double_top.md)

- **Type**: Bearish reversal pattern
- **Structure**: Peak → Trough → Peak → Breakdown
- **Key Features**: Peak similarity, support break, volume divergence
- **Output**: DT folder with resistance level charts

### 4. [Double Bottom](./double_bottom.md)

- **Type**: Bullish reversal pattern
- **Structure**: Trough → Peak → Trough → Breakout
- **Key Features**: Trough similarity, resistance break, volume confirmation
- **Output**: DB folder with support level charts

## Common Algorithm Components

### Swing Point Detection

All patterns use rolling window method to identify local extremes:

```python
def find_swing_points(data, N_bars=20):
    """Find swing highs and lows using rolling windows."""
    data['is_swing_high'] = (data['High'] == data['High'].rolling(window=N_bars*2+1, center=True).max())
    data['is_swing_low'] = (data['Low'] == data['Low'].rolling(window=N_bars*2+1, center=True).min())
    return data
```

### Trend Validation

Preceding trend analysis ensures pattern significance:

```python
def check_preceding_trend(data, pattern_start_index, trend_type='up', lookback_period=90, min_change_percent=0.15):
    """Check for preceding trend before pattern formation."""
    # Validates 15% minimum price change over 90-day window
```

### Volume Analysis

Volume confirmation reduces false positives:

- **Pattern Formation**: Volume should follow expected patterns
- **Breakout Confirmation**: Volume spikes validate genuine breakouts
- **Divergence Detection**: Volume/price divergence signals weakness

## Configuration Modes

### Strict Mode

- **Purpose**: Minimize false positives
- **Characteristics**: Tighter tolerances, higher volume requirements
- **Use Case**: High-confidence pattern identification

### Lenient Mode

- **Purpose**: Maximize pattern discovery
- **Characteristics**: Relaxed tolerances, lower volume requirements
- **Use Case**: Exploratory analysis, broader pattern screening

## Main Processing Logic

### Core Processing Flow

```python
def main():
    # 1. Parse arguments and resolve symbols/patterns/timeframes
    # 2. For each mode (strict/lenient):
    #    - Process each symbol across all timeframes
    #    - Detect patterns using mode-specific configurations
    #    - Generate charts and save results
    # 3. Export aggregated results to CSV
```

### Symbol Processing Pipeline

```python
def process_symbol(symbol, timeframes, patterns, mode, ...):
    # 1. Load OHLCV data with sufficient history
    df = load_data(symbol, start_date, end_date)

    # 2. Process each timeframe
    for timeframe in timeframes:
        df_slice = df[df['Date'] >= cutoff_date].copy()

        # 3. Generate swing points (rolling/zigzag/fractal)
        df_slice = generate_swing_flags(df_slice, method=swing_method)

        # 4. Run pattern detection
        if 'head_and_shoulders' in patterns:
            hns_patterns = detect_head_and_shoulders(df_slice, HNS_CONFIG[mode])
        if 'cup_and_handle' in patterns:
            ch_patterns = detect_cup_and_handle(df_slice, CH_CONFIG[mode])
        if double_patterns_requested:
            double_patterns = detect_double_patterns(df_slice, DT_CONFIG[mode])

        # 5. Generate charts and organize output
        for pattern in detected_patterns:
            plot_pattern_chart(pattern, df_slice, symbol, output_path)

    return all_results
```

## Technical Implementation

### Data Requirements

- **OHLCV Data**: Open, High, Low, Close, Volume required
- **Timeframe Support**: 1d to 5y configurable periods
- **Symbol Support**: Stocks, ETFs, Crypto, International markets

### Output Structure

```
outputs/
├── charts/
│   └── {SYMBOL}/
│       └── {TIMEFRAME}/
│           ├── HNS/     # Head and Shoulders
│           ├── CH/      # Cup and Handle
│           ├── DT/      # Double Top
│           └── DB/      # Double Bottom
└── reports/
    └── pattern_detection_{mode}_{timestamp}.csv
```

### Key Features

- **Vectorized Operations**: NumPy/Pandas for efficient computation
- **Dynamic Scaling**: ATR-based parameter adjustment for market conditions
- **Volume Integration**: Trading volume confirms pattern validity
- **Configurable Sensitivity**: Strict/lenient modes for different use cases
- **Breakout Confirmation**: Price/volume breakouts required for pattern confirmation

## Algorithm Principles

1. **Multiple Validation Layers**: Each pattern applies several independent criteria
2. **Trend Context**: Patterns validated within broader trend context
3. **Volume Confirmation**: Trading volume analysis reduces false positives
4. **Geometric Validation**: Proper pattern proportions and timing constraints
