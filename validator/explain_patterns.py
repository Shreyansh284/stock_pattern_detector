"""
Rule-of-thumb pattern explainer
===============================
Given a detected pattern dict (from detect_all_patterns.py), compute:
- Verdict (valid/weak) using practical filters
- Key validation rules with measured values and pass/fail
- Step-by-step target price calculation per pattern type

Usage:
    from validator.explain_patterns import explain_pattern
    result = explain_pattern(df, pattern)

Returns a dict with keys:
    pattern_type, verdict, score, max_score, rules: [ {name, value, expected, pass, notes} ],
    target: { formula, steps: [str], target_price: float }

Patterns supported: double_top, double_bottom, head_and_shoulders, cup_and_handle
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from math import atan, degrees


@dataclass
class RuleCheck:
    name: str
    value: str
    expected: str
    passed: bool
    notes: Optional[str] = None


def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "n/a"


def _safe_idx_from_date(df: pd.DataFrame, date_val) -> Optional[int]:
    try:
        idxs = df.index[df['Date'] == pd.to_datetime(date_val)]
        if len(idxs) > 0:
            return int(idxs[0])
    except Exception:
        pass
    return None


def _preceding_trend(df: pd.DataFrame, anchor_idx: int, direction: str, lookback_days: int = 60, min_change: float = 0.10) -> Tuple[bool, Optional[float]]:
    if not (0 <= anchor_idx < len(df)):
        return False, None
    start = max(0, anchor_idx - lookback_days)
    if start >= anchor_idx:
        return False, None
    try:
        start_price = float(df['Close'].iloc[start])
        end_close = float(df['Close'].iloc[anchor_idx])
        if start_price <= 0:
            return False, None
        chg = (end_close - start_price) / start_price
        if direction == 'up':
            return (chg >= min_change), chg
        else:
            return (chg <= -min_change), chg
    except Exception:
        return False, None


def _volume_spike(df: pd.DataFrame, idx: int, window: int = 20, mult: float = 1.5) -> Tuple[Optional[bool], Optional[float], Optional[float]]:
    if 'Volume' not in df.columns or not (0 <= idx < len(df)):
        return None, None, None
    if idx < window:
        return None, None, None
    try:
        avg_vol = float(df['Volume'].iloc[idx-window:idx].mean())
        v = float(df['Volume'].iloc[idx])
        if avg_vol <= 0:
            return None, None, None
        return (v >= mult * avg_vol), v, avg_vol
    except Exception:
        return None, None, None


def _neckline_value_at(slope: float, intercept: float, date_val) -> Optional[float]:
    try:
        return float(slope) * pd.to_datetime(date_val).toordinal() + float(intercept)
    except Exception:
        return None


def _explain_cup_handle(df: pd.DataFrame, pattern: Dict[str, Any]) -> Dict[str, Any]:
    """Explain Cup & Handle with practical checks and target calculation.
    Expected pattern keys: left_rim, cup_bottom, right_rim, handle_low, breakout, cup_duration, cup_depth
    """
    lr_date, lr_high, lr_idx = pattern['left_rim']
    cb_date, cb_low, cb_idx = pattern['cup_bottom']
    rr_date, rr_high, rr_idx = pattern['right_rim']
    h_date, h_low, h_idx = pattern['handle_low']
    breakout_date, breakout_price, breakout_idx = pattern.get('breakout', (None, None, None))

    rules: List[RuleCheck] = []
    score = 0
    max_score = 7  # depth, rim similarity, symmetry, handle depth, breakout close, volume, trend

    # Resistance level is the higher rim
    try:
        resistance = float(max(lr_high, rr_high))
    except Exception:
        resistance = None

    # 1) Cup depth within reasonable range
    try:
        cup_depth = float(pattern.get('cup_depth')) if pattern.get('cup_depth') is not None else ((resistance - float(cb_low)) / resistance)
    except Exception:
        cup_depth = None
    passed = (cup_depth is not None) and (0.08 <= cup_depth <= 0.60)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Cup depth",
        value=_fmt_pct(cup_depth) if cup_depth is not None else "n/a",
        expected="8–60% (15–40% strong)",
        passed=bool(passed),
        notes="Deeper than ~60% risks failure; shallower than ~8% is weak."
    ))

    # 2) Rim height similarity
    try:
        rim_diff = abs(float(lr_high) - float(rr_high)) / max(float(lr_high), float(rr_high))
    except Exception:
        rim_diff = None
    passed = (rim_diff is not None) and (rim_diff <= 0.10)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Rim height similarity",
        value=_fmt_pct(rim_diff) if rim_diff is not None else "n/a",
        expected="<= 10% (<= 5% strong)",
        passed=bool(passed)
    ))

    # 3) Cup symmetry: left vs right side duration balance
    try:
        left_span = max(1, (pd.to_datetime(cb_date) - pd.to_datetime(lr_date)).days)
        right_span = max(1, (pd.to_datetime(rr_date) - pd.to_datetime(cb_date)).days)
        time_asym = abs(left_span - right_span) / max(left_span, right_span)
    except Exception:
        time_asym = None
    passed = (time_asym is not None) and (time_asym <= 0.50)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Cup symmetry (time)",
        value=_fmt_pct(time_asym) if time_asym is not None else "n/a",
        expected="|L−R|/max <= 50% (<= 25% strong)",
        passed=bool(passed),
        notes="Balanced U-shape is preferred over sharp V."
    ))

    # 4) Handle depth constraint
    try:
        handle_depth = (float(rr_high) - float(h_low)) / float(rr_high)
    except Exception:
        handle_depth = None
    passed = (handle_depth is not None) and (handle_depth <= 0.35)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Handle depth",
        value=_fmt_pct(handle_depth) if handle_depth is not None else "n/a",
        expected="<= 35% (<= 20% strong)",
        passed=bool(passed),
        notes="Shallow handles are healthier."
    ))

    # 5) Breakout close above resistance by >= 1%
    if breakout_date is not None and resistance is not None:
        try:
            close_j = float(df['Close'].iloc[int(breakout_idx)]) if 0 <= int(breakout_idx) < len(df) else float(breakout_price)
        except Exception:
            close_j = float(breakout_price) if breakout_price is not None else None
        try:
            breach = (float(close_j) - float(resistance)) / float(resistance) if close_j is not None else None
        except Exception:
            breach = None
        passed = (breach is not None) and (breach >= 0.01)
        if passed:
            score += 1
        rules.append(RuleCheck(
            name="Breakout close above resistance",
            value=_fmt_pct(breach) if breach is not None else "n/a",
            expected=">= 1% (>= 2% strong)",
            passed=bool(passed)
        ))
    else:
        rules.append(RuleCheck("Breakout close above resistance", "n/a", ">= 1%", False, notes="No breakout/resistance"))

    # 6) Volume spike on breakout (>= 1.5x 20-day avg)
    vol_ok, v_now, v_avg = _volume_spike(df, int(breakout_idx) if breakout_idx is not None else -1, window=20, mult=1.5)
    if vol_ok is None:
        rules.append(RuleCheck("Volume spike on breakout", "n/a", ">= 1.5x 20-day avg", False, notes="No/insufficient volume data"))
    else:
        if vol_ok:
            score += 1
        val = f"{v_now:,.0f} vs {v_avg:,.0f} avg" if (v_now is not None and v_avg is not None) else "n/a"
        rules.append(RuleCheck("Volume spike on breakout", val, ">= 1.5x 20-day avg", bool(vol_ok)))

    # 7) Preceding uptrend before left rim (60d, >= +10%)
    anchor = int(lr_idx) if isinstance(lr_idx, (int, np.integer)) else _safe_idx_from_date(df, lr_date)
    trend_pass, chg = _preceding_trend(df, anchor, direction='up', lookback_days=60, min_change=0.10)
    if trend_pass:
        score += 1
    rules.append(RuleCheck("Preceding uptrend (60d)", _fmt_pct(chg) if chg is not None else "n/a", ">= +10%", bool(trend_pass)))

    # Target: resistance + cup height
    if resistance is not None:
        try:
            cup_height = float(resistance) - float(cb_low)
            target = float(resistance) + cup_height
            steps = [
                f"Resistance (rims) = {resistance:.2f}",
                f"Cup bottom = {float(cb_low):.2f}",
                f"Cup height = Resistance - Bottom = {resistance:.2f} - {float(cb_low):.2f} = {cup_height:.2f}",
                f"Target = Resistance + Cup height = {resistance:.2f} + {cup_height:.2f} = {target:.2f}",
            ]
        except Exception:
            target = None
            steps = ["Insufficient data to compute target"]
    else:
        target = None
        steps = ["No resistance computed"]

    verdict = 'valid' if score >= 4 else 'weak'
    return {
        'pattern_type': 'Cup & Handle',
        'verdict': verdict,
        'score': int(score),
        'max_score': int(max_score),
        'rules': [r.__dict__ for r in rules],
        'target': {
            'formula': 'Target = Resistance + (Resistance – Bottom of Cup)',
            'steps': steps,
            'target_price': None if target is None else float(target)
        }
    }


def _explain_double_top(df: pd.DataFrame, pattern: Dict[str, Any]) -> Dict[str, Any]:
    p1_date, p1_price, p1_idx = pattern['P1']
    t_date, t_price, t_idx = pattern['T']
    p2_date, p2_price, p2_idx = pattern['P2']
    breakout_date, breakout_price, breakout_idx = pattern.get('breakout', (None, None, None))
    neckline = float(pattern.get('neckline_level', t_price))

    rules: List[RuleCheck] = []
    score = 0
    max_score = 6  # trough depth, spacing, price similarity, breakout close, volume, trend

    # 1) Depth of trough (valley) vs tops
    top_ref = float(max(p1_price, p2_price))
    try:
        depth = (top_ref - float(t_price)) / top_ref
    except Exception:
        depth = None
    passed = (depth is not None) and (depth >= 0.10)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Depth of trough below tops",
        value=_fmt_pct(depth) if depth is not None else "n/a",
        expected=">= 10% (>= 15% strong)",
        passed=bool(passed),
        notes="Deeper valley indicates stronger reversal potential."
    ))

    # 2) Distance between tops (time spacing)
    try:
        spacing_days = int((p2_date - p1_date).days)
    except Exception:
        spacing_days = None
    passed = spacing_days is not None and 10 <= spacing_days <= 150
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Time between tops",
        value=f"{spacing_days} days" if spacing_days is not None else "n/a",
        expected="10–150 days (20–90 ideal)",
        passed=bool(passed),
        notes="Too tight/too wide spacing weakens the pattern."
    ))

    # 3) Price similarity of tops
    try:
        diff_pct = abs(float(p1_price) - float(p2_price)) / top_ref
    except Exception:
        diff_pct = None
    passed = (diff_pct is not None) and (diff_pct <= 0.05)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Top price similarity",
        value=_fmt_pct(diff_pct) if diff_pct is not None else "n/a",
        expected="<= 5% (<= 3% strong)",
        passed=bool(passed),
        notes="Roughly equal highs strengthen a double top."
    ))

    # 4) Strength of neckline breakout (close below neckline by >= 1–2%)
    if breakout_date is not None:
        try:
            close_j = float(df['Close'].iloc[int(breakout_idx)]) if 0 <= int(breakout_idx) < len(df) else float(breakout_price)
        except Exception:
            close_j = float(breakout_price) if breakout_price is not None else None
        try:
            breach = (float(neckline) - float(close_j)) / float(neckline) if close_j is not None else None
        except Exception:
            breach = None
        passed = (breach is not None) and (breach >= 0.01)
        if passed:
            score += 1
        rules.append(RuleCheck(
            name="Breakout close below neckline",
            value=_fmt_pct(breach) if breach is not None else "n/a",
            expected=">= 1% (>= 2% strong)",
            passed=bool(passed),
            notes="Use candle close relative to neckline."
        ))
    else:
        rules.append(RuleCheck("Breakout close below neckline", "n/a", ">= 1%", False, notes="No breakout info"))

    # 5) Volume confirmation on breakout (>= 1.5x recent avg)
    vol_ok, v_now, v_avg = _volume_spike(df, int(breakout_idx) if breakout_idx is not None else -1, window=20, mult=1.5)
    if vol_ok is None:
        rules.append(RuleCheck("Volume spike on breakout", "n/a", ">= 1.5x 20-day avg", False, notes="No/insufficient volume data"))
    else:
        if vol_ok:
            score += 1
        val = f"{v_now:,.0f} vs {v_avg:,.0f} avg" if (v_now is not None and v_avg is not None) else "n/a"
        rules.append(RuleCheck("Volume spike on breakout", val, ">= 1.5x 20-day avg", bool(vol_ok)))

    # 6) Trend context: prior uptrend into pattern
    # Use first top index as anchor
    anchor = int(p1_idx) if isinstance(p1_idx, (int, np.integer)) else _safe_idx_from_date(df, p1_date)
    trend_pass, chg = _preceding_trend(df, anchor, direction='up', lookback_days=60, min_change=0.10)
    if trend_pass:
        score += 1
    rules.append(RuleCheck("Preceding uptrend (60d)", _fmt_pct(chg) if chg is not None else "n/a", ">= +10%", bool(trend_pass)))

    # Target: Neckline – (Top – Neckline), use avg of the two tops
    avg_top = (float(p1_price) + float(p2_price)) / 2.0
    height = avg_top - float(neckline)
    target = float(neckline) - height
    steps = [
        f"Neckline = {neckline:.2f}",
        f"Top (avg of two) = ({p1_price:.2f} + {p2_price:.2f})/2 = {avg_top:.2f}",
        f"Height = Top - Neckline = {avg_top:.2f} - {neckline:.2f} = {height:.2f}",
        f"Target = Neckline - Height = {neckline:.2f} - {height:.2f} = {target:.2f}",
    ]

    verdict = 'valid' if score >= 4 else 'weak'
    return {
        'pattern_type': 'Double Top',
        'verdict': verdict,
        'score': int(score),
        'max_score': int(max_score),
        'rules': [r.__dict__ for r in rules],
        'target': {
            'formula': 'Target = Neckline – (Top – Neckline)',
            'steps': steps,
            'target_price': float(target)
        }
    }


def _explain_double_bottom(df: pd.DataFrame, pattern: Dict[str, Any]) -> Dict[str, Any]:
    p1_date, p1_price, p1_idx = pattern['P1']
    t_date, t_price, t_idx = pattern['T']
    p2_date, p2_price, p2_idx = pattern['P2']
    breakout_date, breakout_price, breakout_idx = pattern.get('breakout', (None, None, None))
    neckline = float(pattern.get('neckline_level', t_price))

    rules: List[RuleCheck] = []
    score = 0
    max_score = 6  # peak height, spacing, bottom similarity, breakout close, volume, trend

    # 1) Peak height vs bottoms (bounce height)
    bottom_ref = float(min(p1_price, p2_price))
    try:
        height_pct = (float(t_price) - bottom_ref) / bottom_ref
    except Exception:
        height_pct = None
    passed = (height_pct is not None) and (height_pct >= 0.10)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Peak height above bottoms",
        value=_fmt_pct(height_pct) if height_pct is not None else "n/a",
        expected=">= 10% (>= 15% strong)",
        passed=bool(passed),
        notes="Stronger bounce between bottoms indicates accumulation."
    ))

    # 2) Time spacing between bottoms
    try:
        spacing_days = int((p2_date - p1_date).days)
    except Exception:
        spacing_days = None
    passed = spacing_days is not None and 10 <= spacing_days <= 150
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Time between bottoms",
        value=f"{spacing_days} days" if spacing_days is not None else "n/a",
        expected="10–150 days (20–90 ideal)",
        passed=bool(passed)
    ))

    # 3) Bottom price similarity
    try:
        diff_pct = abs(float(p1_price) - float(p2_price)) / max(float(p1_price), float(p2_price))
    except Exception:
        diff_pct = None
    passed = (diff_pct is not None) and (diff_pct <= 0.05)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Bottom price similarity",
        value=_fmt_pct(diff_pct) if diff_pct is not None else "n/a",
        expected="<= 5% (<= 3% strong)",
        passed=bool(passed)
    ))

    # 4) Strength of neckline breakout (close above by >= 1–2%)
    if breakout_date is not None:
        try:
            close_j = float(df['Close'].iloc[int(breakout_idx)]) if 0 <= int(breakout_idx) < len(df) else float(breakout_price)
        except Exception:
            close_j = float(breakout_price) if breakout_price is not None else None
        try:
            breach = (float(close_j) - float(neckline)) / float(neckline) if close_j is not None else None
        except Exception:
            breach = None
        passed = (breach is not None) and (breach >= 0.01)
        if passed:
            score += 1
        rules.append(RuleCheck(
            name="Breakout close above neckline",
            value=_fmt_pct(breach) if breach is not None else "n/a",
            expected=">= 1% (>= 2% strong)",
            passed=bool(passed)
        ))
    else:
        rules.append(RuleCheck("Breakout close above neckline", "n/a", ">= 1%", False, notes="No breakout info"))

    # 5) Volume confirmation on breakout
    vol_ok, v_now, v_avg = _volume_spike(df, int(breakout_idx) if breakout_idx is not None else -1, window=20, mult=1.5)
    if vol_ok is None:
        rules.append(RuleCheck("Volume spike on breakout", "n/a", ">= 1.5x 20-day avg", False, notes="No/insufficient volume data"))
    else:
        if vol_ok:
            score += 1
        val = f"{v_now:,.0f} vs {v_avg:,.0f} avg" if (v_now is not None and v_avg is not None) else "n/a"
        rules.append(RuleCheck("Volume spike on breakout", val, ">= 1.5x 20-day avg", bool(vol_ok)))

    # 6) Trend context: prior downtrend into pattern
    anchor = int(p1_idx) if isinstance(p1_idx, (int, np.integer)) else _safe_idx_from_date(df, p1_date)
    trend_pass, chg = _preceding_trend(df, anchor, direction='down', lookback_days=60, min_change=0.10)
    if trend_pass:
        score += 1
    rules.append(RuleCheck("Preceding downtrend (60d)", _fmt_pct(chg) if chg is not None else "n/a", "<= -10%", bool(trend_pass)))

    # Target: Neckline + (Neckline – Bottom), use avg of two bottoms
    avg_bottom = (float(p1_price) + float(p2_price)) / 2.0
    height = float(neckline) - avg_bottom
    target = float(neckline) + height
    steps = [
        f"Neckline = {neckline:.2f}",
        f"Bottom (avg of two) = ({p1_price:.2f} + {p2_price:.2f})/2 = {avg_bottom:.2f}",
        f"Height = Neckline - Bottom = {neckline:.2f} - {avg_bottom:.2f} = {height:.2f}",
        f"Target = Neckline + Height = {neckline:.2f} + {height:.2f} = {target:.2f}",
    ]

    verdict = 'valid' if score >= 4 else 'weak'
    return {
        'pattern_type': 'Double Bottom',
        'verdict': verdict,
        'score': int(score),
        'max_score': int(max_score),
        'rules': [r.__dict__ for r in rules],
        'target': {
            'formula': 'Target = Neckline + (Neckline – Bottom)',
            'steps': steps,
            'target_price': float(target)
        }
    }


def _explain_hns(df: pd.DataFrame, pattern: Dict[str, Any]) -> Dict[str, Any]:
    p1_date, p1_high, p1_idx = pattern['P1']  # Left shoulder
    t1_date, t1_low, t1_idx = pattern['T1']
    p2_date, p2_high, p2_idx = pattern['P2']  # Head
    t2_date, t2_low, t2_idx = pattern['T2']
    p3_date, p3_high, p3_idx = pattern['P3']  # Right shoulder
    breakout_date, breakout_price, breakout_idx = pattern.get('breakout', (None, None, None))
    slope = pattern.get('neckline_slope')
    intercept = pattern.get('neckline_intercept')

    rules: List[RuleCheck] = []
    score = 0
    max_score = 6  # head prominence, shoulder similarity, spacing, breakout close, volume, trend

    # 1) Head prominence above shoulders
    try:
        prom_ls = (float(p2_high) - float(p1_high)) / float(p1_high)
        prom_rs = (float(p2_high) - float(p3_high)) / float(p3_high)
        min_prom = min(prom_ls, prom_rs)
    except Exception:
        min_prom = None
    passed = (min_prom is not None) and (min_prom >= 0.04)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Head above shoulders",
        value=_fmt_pct(min_prom) if min_prom is not None else "n/a",
        expected=">= 4% (>= 8% strong)",
        passed=bool(passed)
    ))

    # 2) Shoulder height similarity
    try:
        sh_diff = abs(float(p1_high) - float(p3_high)) / max(float(p1_high), float(p3_high))
    except Exception:
        sh_diff = None
    passed = (sh_diff is not None) and (sh_diff <= 0.12)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Shoulder height similarity",
        value=_fmt_pct(sh_diff) if sh_diff is not None else "n/a",
        expected="<= 12%",
        passed=bool(passed)
    ))

    # 3) Shoulder spacing symmetry (time)
    try:
        left_span = max(1, (p2_date - p1_date).days)
        right_span = max(1, (p3_date - p2_date).days)
        time_asym = abs(left_span - right_span) / max(left_span, right_span)
    except Exception:
        time_asym = None
    passed = (time_asym is not None) and (time_asym <= 0.35)
    if passed:
        score += 1
    rules.append(RuleCheck(
        name="Shoulder spacing symmetry",
        value=_fmt_pct(time_asym) if time_asym is not None else "n/a",
        expected="|L−R|/max <= 35%",
        passed=bool(passed)
    ))

    # 4) Breakout close vs neckline at breakout
    if breakout_date is not None and slope is not None and intercept is not None:
        neck_at_break = _neckline_value_at(slope, intercept, breakout_date)
        try:
            close_j = float(df['Close'].iloc[int(breakout_idx)]) if 0 <= int(breakout_idx) < len(df) else float(breakout_price)
        except Exception:
            close_j = float(breakout_price) if breakout_price is not None else None
        try:
            breach = (float(neck_at_break) - float(close_j)) / float(neck_at_break) if (neck_at_break is not None and close_j is not None) else None
        except Exception:
            breach = None
        passed = (breach is not None) and (breach >= 0.01)
        if passed:
            score += 1
        rules.append(RuleCheck(
            name="Breakout close below neckline",
            value=_fmt_pct(breach) if breach is not None else "n/a",
            expected=">= 1% (>= 2% strong)",
            passed=bool(passed)
        ))
    else:
        rules.append(RuleCheck("Breakout close below neckline", "n/a", ">= 1%", False, notes="Missing neckline/breakout"))

    # 5) Volume: spike on breakout (optional segment trend not enforced here)
    vol_ok, v_now, v_avg = _volume_spike(df, int(breakout_idx) if breakout_idx is not None else -1, window=20, mult=1.3)
    if vol_ok is None:
        rules.append(RuleCheck("Volume spike on breakout", "n/a", ">= 1.3x 20-day avg", False, notes="No/insufficient volume data"))
    else:
        if vol_ok:
            score += 1
        val = f"{v_now:,.0f} vs {v_avg:,.0f} avg" if (v_now is not None and v_avg is not None) else "n/a"
        rules.append(RuleCheck("Volume spike on breakout", val, ">= 1.3x 20-day avg", bool(vol_ok)))

    # 6) Trend context: prior uptrend
    anchor = int(p1_idx) if isinstance(p1_idx, (int, np.integer)) else _safe_idx_from_date(df, p1_date)
    trend_pass, chg = _preceding_trend(df, anchor, direction='up', lookback_days=60, min_change=0.10)
    if trend_pass:
        score += 1
    rules.append(RuleCheck("Preceding uptrend (60d)", _fmt_pct(chg) if chg is not None else "n/a", ">= +10%", bool(trend_pass)))

    # Target: Neckline – (Head – Neckline)
    # Use neckline at head date for height, and apply height below neckline at breakout date for target
    neck_at_head = _neckline_value_at(slope, intercept, p2_date) if (slope is not None and intercept is not None) else None
    neck_at_break = _neckline_value_at(slope, intercept, breakout_date) if (slope is not None and intercept is not None) else None
    if neck_at_head is None or neck_at_break is None:
        target = None
        steps = ["Neckline not available to compute target"]
    else:
        height = float(p2_high) - float(neck_at_head)
        target = float(neck_at_break) - height
        steps = [
            f"Neckline at head = {neck_at_head:.2f}",
            f"Head = {p2_high:.2f}",
            f"Height = Head - Neckline_at_head = {p2_high:.2f} - {neck_at_head:.2f} = {height:.2f}",
            f"Neckline at breakout = {neck_at_break:.2f}",
            f"Target = Neckline_at_break - Height = {neck_at_break:.2f} - {height:.2f} = {target:.2f}",
        ]

    verdict = 'valid' if score >= 4 else 'weak'
    return {
        'pattern_type': 'Head & Shoulders',
        'verdict': verdict,
        'score': int(score),
        'max_score': int(max_score),
        'rules': [r.__dict__ for r in rules],
        'target': {
            'formula': 'Target = Neckline – (Head – Neckline)',
            'steps': steps,
            'target_price': None if target is None else float(target)
        }
    }


def explain_pattern(df: pd.DataFrame, pattern: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a rule-of-thumb explanation and measured target for a pattern.

    df: DataFrame with at least columns ['Date', 'Close'] and ideally ['Volume']
    pattern: dict with keys as produced by detect_all_patterns.py
    """
    ptype = pattern.get('type')
    if ptype == 'double_top':
        return _explain_double_top(df, pattern)
    elif ptype == 'double_bottom':
        return _explain_double_bottom(df, pattern)
    elif ptype == 'head_and_shoulders':
        return _explain_hns(df, pattern)
    elif ptype == 'cup_and_handle':
        return _explain_cup_handle(df, pattern)
    else:
        return {
            'pattern_type': ptype or 'unknown',
            'verdict': 'weak',
            'score': 0,
            'max_score': 0,
            'rules': [],
            'target': {
                'formula': '',
                'steps': ["Unsupported pattern type"],
                'target_price': None
            }
        }
