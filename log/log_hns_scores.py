import csv
from pathlib import Path

def log_hns_scores(results, output_path):
    """
    Log Head and Shoulders validation scores to a CSV file.
    Each row: symbol, timeframe, left_shoulder_date, head_date, right_shoulder_date, score, is_valid
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['symbol', 'timeframe', 'left_shoulder_date', 'head_date', 'right_shoulder_date', 'score', 'is_valid'])
        for pattern in results:
            if pattern.get('type') == 'head_and_shoulders' and 'validation' in pattern:
                p1_date = pattern['P1'][0]
                p2_date = pattern['P2'][0]
                p3_date = pattern['P3'][0]
                writer.writerow([
                    pattern.get('symbol', ''),
                    pattern.get('timeframe', ''),
                    p1_date,
                    p2_date,
                    p3_date,
                    pattern['validation'].get('score', ''),
                    pattern['validation'].get('is_valid', '')
                ])
