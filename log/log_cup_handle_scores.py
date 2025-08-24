import csv
from pathlib import Path

def log_cup_handle_scores(results, output_path):
    """
    Log Cup and Handle validation scores to a CSV file.
    Each row: symbol, timeframe, left_rim_date, right_rim_date, score, is_valid
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['symbol', 'timeframe', 'left_rim_date', 'right_rim_date', 'score', 'is_valid'])
        for pattern in results:
            if pattern.get('type') == 'cup_and_handle' and 'validation' in pattern:
                left_rim_date = pattern['left_rim'][0]
                right_rim_date = pattern['right_rim'][0]
                writer.writerow([
                    pattern.get('symbol', ''),
                    pattern.get('timeframe', ''),
                    left_rim_date,
                    right_rim_date,
                    pattern['validation'].get('score', ''),
                    pattern['validation'].get('is_valid', '')
                ])
