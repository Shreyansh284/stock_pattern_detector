import csv
from pathlib import Path

def log_double_pattern_scores(results, output_path):
    """
    Log Double Top/Bottom validation scores to a CSV file.
    Each row: symbol, timeframe, pattern_type, first_point_date, valley_peak_date, second_point_date, score, is_valid
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'symbol', 'timeframe', 'pattern_type', 
            'first_point_date', 'valley_peak_date', 'second_point_date', 
            'score', 'is_valid'
        ])
        
        for pattern in results:
            pattern_type = pattern.get('type', '')
            if pattern_type in ['double_top', 'double_bottom'] and 'validation' in pattern:
                p1_date = pattern['P1'][0]  # First top/bottom
                t_date = pattern['T'][0]    # Valley/peak
                p2_date = pattern['P2'][0]  # Second top/bottom
                
                writer.writerow([
                    pattern.get('symbol', ''),
                    pattern.get('timeframe', ''),
                    pattern_type,
                    p1_date,
                    t_date,
                    p2_date,
                    pattern['validation'].get('score', ''),
                    pattern['validation'].get('is_valid', '')
                ])
