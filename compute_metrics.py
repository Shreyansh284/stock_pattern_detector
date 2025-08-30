#!/usr/bin/env python3
"""
Pattern Detection Metrics Calculator
===================================

This script computes classification metrics (precision, recall, accuracy, F1) for each pattern type
and overall, given:
  1. A detection report CSV produced by `detect_all_patterns.py` (already in outputs/reports)
  2. A ground-truth annotation CSV you supply.

Ground Truth CSV Requirements (please confirm / adjust):
  Required columns:
    - symbol : str
    - timeframe : str (must match detection report timeframe values, e.g. 1y,2y,3y,5y)
    - pattern_type : str in {head_and_shoulders, cup_and_handle, double_top, double_bottom}
    - present : int/bool (1 if pattern truly exists for that (symbol,timeframe,pattern_type) window, 0 otherwise)
      (Alternative: if you only list positives, we can treat missing rows as negatives if you confirm.)
  Optional columns (if you want per-instance granularity when multiple patterns of same type exist):
    - instance_id : str/int (unique id per true pattern). If omitted we treat any detection of that pattern_type as TP.
    - window_start / window_end : dates to define precise period. If provided we can overlap-match.

Detection Report CSV (current columns observed):
  pattern_type,symbol,timeframe,mode,... and pattern specific point columns.

Matching Logic (default simplest):
  For each (symbol,timeframe,pattern_type):
    TP: detection exists AND ground truth present==1
    FP: detection exists AND ground truth present==0
    FN: no detection AND ground truth present==1
    TN: no detection AND ground truth present==0
  If multiple detections for same key and present==1 they still count as 1 TP (unless you want to penalize extras; ask to enable over-detection penalty).

Pending Confirmation Needed From You:
  A. Do you have (or can you produce) a ground truth CSV with the required columns?
  B. Do you want unmatched extra detections (more than 1) for a positive case to count as FP (over-detection)?
  C. Treat missing ground-truth rows as negatives? (Simpler) Or will you provide an exhaustive grid with present=0 rows included?

Once you confirm A-C I can finalize or adjust logic.

Usage (after confirmation):
    # Simple presence metrics per (symbol,timeframe,pattern_type)
    python compute_metrics.py --detections outputs/reports/pattern_detection_all_strict_*.csv --ground-truth ground_truth.csv --mode simple

    # Instance-level metrics (expects instance_id in ground truth CSV)
    python compute_metrics.py --detections outputs/reports/pattern_detection_all_strict_*.csv --ground-truth ground_truth_instances.csv --mode instance

    # Only count detections that passed internal validator
    python compute_metrics.py --detections outputs/reports/pattern_detection_all_strict_*.csv --ground-truth ground_truth.csv --mode simple --valid-only

Outputs:
  - Prints per pattern type metrics and overall macro + micro averages.
  - Writes metrics CSV to outputs/metrics/metrics_summary_<timestamp>.csv

"""
import argparse
import glob
import os
from datetime import datetime
import pandas as pd
from typing import Tuple, Dict

PATTERN_TYPES = ["head_and_shoulders", "cup_and_handle", "double_top", "double_bottom"]


def load_detections(paths) -> pd.DataFrame:
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit(f"No detection files matched: {paths}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"WARN: could not read {f}: {e}")
    if not dfs:
        raise SystemExit("No detection data loaded.")
    det = pd.concat(dfs, ignore_index=True)
    # Normalize pattern_type naming to expected set
    det['pattern_type'] = det['pattern_type'].str.strip().str.lower()
    # unify to internal names
    mapping = {"hns": "head_and_shoulders", "hs": "head_and_shoulders"}
    det['pattern_type'] = det['pattern_type'].replace(mapping)
    # Ensure optional columns exist
    if 'instance_id' not in det.columns:
        det['instance_id'] = 1
    if 'validation_is_valid' not in det.columns:
        det['validation_is_valid'] = True
    return det


def load_ground_truth(path: str) -> pd.DataFrame:
    gt = pd.read_csv(path)
    req = ['symbol', 'timeframe', 'pattern_type', 'present']
    missing = [c for c in req if c not in gt.columns]
    if missing:
        raise SystemExit(f"Ground truth file missing columns: {missing}")
    gt['pattern_type'] = gt['pattern_type'].str.strip().str.lower()
    gt['present'] = gt['present'].astype(int)
    return gt


def compute_confusion_simple(det: pd.DataFrame, gt: pd.DataFrame, treat_missing_as_negative: bool, over_detection_penalty: bool) -> pd.DataFrame:
    # Build key sets
    det_keys = det[['symbol', 'timeframe', 'pattern_type']].drop_duplicates()
    gt_keys = gt[['symbol', 'timeframe', 'pattern_type']].drop_duplicates()

    # If treating missing as negative expand gt to include det keys missing in gt with present=0
    if treat_missing_as_negative:
        merged = pd.merge(det_keys, gt, on=['symbol','timeframe','pattern_type'], how='left')
        missing_rows = merged[merged['present'].isna()][['symbol','timeframe','pattern_type']].copy()
        if not missing_rows.empty:
            missing_rows['present'] = 0
            gt = pd.concat([gt, missing_rows], ignore_index=True)

    # Now determine confusion counts per key
    records = []
    for _, row in gt.iterrows():
        key = (row.symbol, row.timeframe, row.pattern_type)
        present = int(row.present)
        det_present = det[(det.symbol==row.symbol) & (det.timeframe==row.timeframe) & (det.pattern_type==row.pattern_type)]
        n_detections = len(det_present)
        tp=fp=fn=tn=0
        if present==1:
            if n_detections>0:
                tp = 1
                if over_detection_penalty and n_detections>1:
                    fp = n_detections-1  # extra detections penalized
            else:
                fn = 1
        else: # present==0
            if n_detections>0:
                fp = n_detections
            else:
                tn = 1
        records.append({
            'symbol': row.symbol,
            'timeframe': row.timeframe,
            'pattern_type': row.pattern_type,
            'present': present,
            'detections': n_detections,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
        })
    df_conf = pd.DataFrame(records)
    return df_conf


def compute_confusion_instance(det: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Instance-level confusion: every ground-truth instance_id must be matched by a detection with same key & instance_id.
    Detections with instance_id not in ground truth for that key count as FP.
    Missing detections for a gt instance_id -> FN.
    TN not meaningful at instance granularity (set 0)."""
    req_cols = {'instance_id'}
    if not req_cols.issubset(gt.columns):
        raise SystemExit("Instance mode requires ground truth column 'instance_id'.")
    records=[]
    for key, group in gt.groupby(['symbol','timeframe','pattern_type']):
        sym, tf, ptype = key
        gt_ids = set(group['instance_id'].astype(str))
        det_subset = det[(det.symbol==sym)&(det.timeframe==tf)&(det.pattern_type==ptype)]
        det_ids = list(det_subset['instance_id'].astype(str))
        matched = set()
        for did in det_ids:
            if did in gt_ids and did not in matched:
                records.append({'symbol':sym,'timeframe':tf,'pattern_type':ptype,'instance_id':did,'TP':1,'FP':0,'FN':0,'TN':0})
                matched.add(did)
            else:
                records.append({'symbol':sym,'timeframe':tf,'pattern_type':ptype,'instance_id':did,'TP':0,'FP':1,'FN':0,'TN':0})
        for gid in gt_ids - matched:
            records.append({'symbol':sym,'timeframe':tf,'pattern_type':ptype,'instance_id':gid,'TP':0,'FP':0,'FN':1,'TN':0})
    return pd.DataFrame(records)


def metrics_from_conf(df_conf: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,float]]:
    rows=[]
    micro_tp=micro_fp=micro_fn=micro_tn=0
    for p in PATTERN_TYPES:
        subset = df_conf[df_conf.pattern_type==p]
        if subset.empty:
            continue
        tp = subset.TP.sum(); fp=subset.FP.sum(); fn=subset.FN.sum(); tn=subset.TN.sum()
        micro_tp+=tp; micro_fp+=fp; micro_fn+=fn; micro_tn+=tn
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        accuracy = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        rows.append({'pattern_type': p,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'accuracy':accuracy,'f1':f1})
    per_pattern = pd.DataFrame(rows)
    macro = {
        'macro_precision': per_pattern.precision.mean() if not per_pattern.empty else 0.0,
        'macro_recall': per_pattern.recall.mean() if not per_pattern.empty else 0.0,
        'macro_accuracy': per_pattern.accuracy.mean() if not per_pattern.empty else 0.0,
        'macro_f1': per_pattern.f1.mean() if not per_pattern.empty else 0.0,
    }
    micro_precision = micro_tp/(micro_tp+micro_fp) if (micro_tp+micro_fp)>0 else 0.0
    micro_recall = micro_tp/(micro_tp+micro_fn) if (micro_tp+micro_fn)>0 else 0.0
    micro_accuracy = (micro_tp+micro_tn)/(micro_tp+micro_tn+micro_fp+micro_fn) if (micro_tp+micro_tn+micro_fp+micro_fn)>0 else 0.0
    micro_f1 = 2*micro_precision*micro_recall/(micro_precision+micro_recall) if (micro_precision+micro_recall)>0 else 0.0
    micro_dict = {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_accuracy': micro_accuracy,
        'micro_f1': micro_f1
    }
    return per_pattern, {**macro, **micro_dict}


def main():
    parser = argparse.ArgumentParser(description="Compute pattern detection metrics.")
    parser.add_argument('--detections', nargs='+', required=True, help='Glob(s) for detection CSV(s)')
    parser.add_argument('--ground-truth', required=True, help='Ground truth CSV path')
    parser.add_argument('--treat-missing-as-negative', action='store_true', help='Treat keys missing in ground truth as negatives (TN)')
    parser.add_argument('--over-detection-penalty', action='store_true', help='Count extra detections beyond first as FP when present=1 (simple mode)')
    parser.add_argument('--mode', choices=['simple','instance'], default='simple', help='Evaluation granularity')
    parser.add_argument('--valid-only', action='store_true', help='Consider only detections with validation_is_valid == True')
    args = parser.parse_args()

    det = load_detections(args.detections)
    if args.valid_only and 'validation_is_valid' in det.columns:
        det = det[det['validation_is_valid']==True].copy()
    gt = load_ground_truth(args.ground_truth)

    if args.mode=='simple':
        df_conf = compute_confusion_simple(det, gt, args.treat_missing_as_negative, args.over_detection_penalty)
    else:
        df_conf = compute_confusion_instance(det, gt)
    per_pattern, agg = metrics_from_conf(df_conf)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = 'outputs/metrics'
    os.makedirs(out_dir, exist_ok=True)
    conf_path = os.path.join(out_dir, f'confusion_{ts}.csv')
    metrics_path = os.path.join(out_dir, f'metrics_{ts}.csv')
    df_conf.to_csv(conf_path, index=False)
    per_pattern.to_csv(metrics_path, index=False)

    print("Confusion counts written to:", conf_path)
    print("Per-pattern metrics written to:", metrics_path)
    print("\nPer Pattern Metrics:\n", per_pattern)
    print("\nAggregate Metrics:")
    for k,v in agg.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
