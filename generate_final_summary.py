#!/usr/bin/env python3
"""
Final Metrics Summary Generator
==============================
Creates comprehensive metrics CSV and visualization chart.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def create_final_metrics_summary():
    # Read the latest metrics
    metrics_df = pd.read_csv('outputs/metrics/metrics_20250830_162625.csv')
    confusion_df = pd.read_csv('outputs/metrics/confusion_20250830_162625.csv')
    
    # Calculate additional statistics
    total_predictions = len(confusion_df)
    total_detections = confusion_df['detections'].sum()
    total_true_positives = confusion_df['TP'].sum()
    total_false_positives = confusion_df['FP'].sum()
    total_false_negatives = confusion_df['FN'].sum()
    total_true_negatives = confusion_df['TN'].sum()
    
    # Overall metrics
    overall_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    overall_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    overall_accuracy = (total_true_positives + total_true_negatives) / total_predictions
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # Create comprehensive summary
    summary_data = {
        'Dataset Statistics': {
            'Total Symbol-Timeframe-Pattern Combinations': total_predictions,
            'Total Patterns Detected': total_detections,
            'Unique Symbols Analyzed': len(confusion_df['symbol'].unique()),
            'Timeframes Analyzed': str(sorted(confusion_df['timeframe'].unique())),
            'Pattern Types': str(sorted(confusion_df['pattern_type'].unique()))
        },
        'Confusion Matrix Totals': {
            'True Positives': total_true_positives,
            'False Positives': total_false_positives,
            'True Negatives': total_true_negatives,
            'False Negatives': total_false_negatives
        },
        'Overall Performance': {
            'Precision': f"{overall_precision:.4f}",
            'Recall': f"{overall_recall:.4f}",
            'Accuracy': f"{overall_accuracy:.4f}",
            'F1-Score': f"{overall_f1:.4f}"
        }
    }
    
    # Add per-pattern performance
    for _, row in metrics_df.iterrows():
        pattern = row['pattern_type'].replace('_', ' ').title()
        summary_data[f'{pattern} Performance'] = {
            'True Positives': int(row['tp']),
            'False Positives': int(row['fp']),
            'False Negatives': int(row['fn']),
            'True Negatives': int(row['tn']),
            'Precision': f"{row['precision']:.4f}",
            'Recall': f"{row['recall']:.4f}",
            'Accuracy': f"{row['accuracy']:.4f}",
            'F1-Score': f"{row['f1']:.4f}"
        }
    
    # Convert to flat CSV format
    csv_rows = []
    for section, data in summary_data.items():
        for metric, value in data.items():
            csv_rows.append({
                'Section': section,
                'Metric': metric,
                'Value': value
            })
    
    final_df = pd.DataFrame(csv_rows)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'final_metrics_summary_{timestamp}.csv'
    final_df.to_csv(csv_path, index=False)
    
    return final_df, metrics_df, csv_path

def create_metrics_visualization(metrics_df, output_path):
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Stock Pattern Detection - Performance Metrics Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance Metrics by Pattern Type
    patterns = metrics_df['pattern_type'].str.replace('_', ' ').str.title()
    metrics_to_plot = ['precision', 'recall', 'accuracy', 'f1']
    
    x = np.arange(len(patterns))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        values = metrics_df[metric].values
        ax1.bar(x + i*width, values, width, label=metric.title(), alpha=0.8)
    
    ax1.set_xlabel('Pattern Type')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics by Pattern Type')
    ax1.set_xticks(x + width*1.5)
    ax1.set_xticklabels(patterns, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix Heatmap
    confusion_data = metrics_df[['tp', 'fp', 'fn', 'tn']].values
    im = ax2.imshow(confusion_data.T, cmap='Blues', aspect='auto')
    ax2.set_title('Confusion Matrix by Pattern Type')
    ax2.set_xlabel('Pattern Type')
    ax2.set_ylabel('Confusion Matrix Element')
    ax2.set_xticks(range(len(patterns)))
    ax2.set_xticklabels(patterns, rotation=45, ha='right')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(['TP', 'FP', 'FN', 'TN'])
    
    # Add text annotations
    for i in range(len(patterns)):
        for j in range(4):
            text = ax2.text(i, j, int(confusion_data[i, j]), ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    
    # 3. Detection Success Rate
    total_positives = metrics_df['tp'] + metrics_df['fn']
    detection_rate = metrics_df['tp'] / total_positives
    detection_rate = detection_rate.fillna(0)  # Handle division by zero
    
    bars = ax3.bar(patterns, detection_rate, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00'], alpha=0.7)
    ax3.set_title('Detection Success Rate (Recall)')
    ax3.set_xlabel('Pattern Type')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(0, 1.1)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars, detection_rate):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{rate:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision vs Recall Scatter
    ax4.scatter(metrics_df['recall'], metrics_df['precision'], 
               s=200, alpha=0.7, c=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00'])
    
    # Add pattern labels
    for i, pattern in enumerate(patterns):
        ax4.annotate(pattern, (metrics_df['recall'].iloc[i], metrics_df['precision'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision vs Recall')
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)
    
    # Add diagonal line for F1-score reference
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='F1=0.67 line')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

if __name__ == '__main__':
    # Generate final summary
    final_df, metrics_df, csv_path = create_final_metrics_summary()
    
    # Create visualization
    chart_path = f'final_metrics_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    create_metrics_visualization(metrics_df, chart_path)
    
    print(f"✅ Final metrics summary created: {csv_path}")
    print(f"✅ Metrics visualization created: {chart_path}")
    print("\n📊 FINAL PERFORMANCE SUMMARY:")
    print("=" * 50)
    
    # Print key metrics
    for _, row in metrics_df.iterrows():
        pattern = row['pattern_type'].replace('_', ' ').title()
        print(f"{pattern:20} | Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f} | F1: {row['f1']:.3f}")
    
    # Calculate and print overall metrics
    total_tp = metrics_df['tp'].sum()
    total_fp = metrics_df['fp'].sum()
    total_fn = metrics_df['fn'].sum()
    total_tn = metrics_df['tn'].sum()
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    print("=" * 50)
    print(f"{'OVERALL (Micro-avg)':20} | Precision: {micro_precision:.3f} | Recall: {micro_recall:.3f} | F1: {micro_f1:.3f}")
    print("=" * 50)
    print(f"\n🎯 Total Patterns Analyzed: {total_tp + total_fn}")
    print(f"✅ Successfully Detected: {total_tp}")
    print(f"❌ Missed Patterns: {total_fn}")
    print(f"🎯 False Positives: {total_fp}")
