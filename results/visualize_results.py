"""
Results Visualization Script

This script generates plots and visualizations from the machine unlearning experiment results.
Run this script to create publication-ready figures from your CSV data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_method_comparison():
    """Compare Random Labelling vs Gradient Ascent across metrics."""
    df = pd.read_csv('comparison/random_labelling_vs_gradient_ascent.csv')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    datasets = df['dataset'].unique()
    
    for i, metric in enumerate(['BLEU', 'ROUGE-L', 'BERTScore']):
        metric_data = df[df['metric'] == metric]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        axes[i].bar(x - width/2, metric_data['random_labelling_score'], 
                   width, label='Random Labelling', alpha=0.8)
        axes[i].bar(x + width/2, metric_data['gradient_ascent_score'], 
                   width, label='Gradient Ascent', alpha=0.8)
        
        axes[i].set_xlabel('Dataset')
        axes[i].set_ylabel(f'{metric} Score')
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(datasets, rotation=45)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_curves():
    """Plot training loss and metric progression."""
    df = pd.read_csv('training_curves/nemotron_random_labelling_losses.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training and Validation Loss
    axes[0, 0].plot(df['epoch'], df['training_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['validation_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # BLEU Score Progression
    axes[0, 1].plot(df['epoch'], df['bleu_score'], 'g-', linewidth=2, marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('BLEU Score')
    axes[0, 1].set_title('BLEU Score Decline (Unlearning)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROUGE-L Score Progression
    axes[1, 0].plot(df['epoch'], df['rouge_l_score'], 'purple', linewidth=2, marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('ROUGE-L Score')
    axes[1, 0].set_title('ROUGE-L Score Decline (Unlearning)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined Metrics
    axes[1, 1].plot(df['epoch'], df['bleu_score'], 'g-', label='BLEU', linewidth=2)
    axes[1, 1].plot(df['epoch'], df['rouge_l_score'], 'purple', label='ROUGE-L', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Metric Progression During Unlearning')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_knowledge_retention():
    """Visualize knowledge retention vs forgetting effectiveness."""
    df = pd.read_csv('model_analysis/knowledge_retention_analysis.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Retention Rate by Model and Method
    pivot_retention = df.pivot_table(values='retention_rate', 
                                   index='model', 
                                   columns=['method', 'knowledge_category'])
    
    pivot_retention.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title('Knowledge Retention Rate by Model and Method')
    axes[0].set_ylabel('Retention Rate')
    axes[0].set_xlabel('Model')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Forget Effectiveness vs Retention (Scatter)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        axes[1].scatter(method_data['retention_rate'], 
                       method_data['forget_effectiveness'],
                       label=method, s=100, alpha=0.7)
    
    axes[1].set_xlabel('Retention Rate (General Knowledge)')
    axes[1].set_ylabel('Forget Effectiveness (Target Facts)')
    axes[1].set_title('Retention vs Forgetting Trade-off')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add ideal region
    axes[1].axhspan(0.8, 1.0, alpha=0.1, color='green', label='High Forget Effectiveness')
    axes[1].axvspan(0.8, 1.0, alpha=0.1, color='blue', label='High Retention')
    
    plt.tight_layout()
    plt.savefig('knowledge_retention_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_table():
    """Create a summary table of all experiments."""
    df = pd.read_csv('summary_statistics.csv')
    
    # Create styled table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the data for better display
    display_df = df.round(4)
    
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Machine Unlearning Experiment Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('experiment_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all visualizations."""
    print("🎨 Generating machine unlearning results visualizations...")
    
    try:
        print("📊 1. Method comparison plot...")
        plot_method_comparison()
        
        print("📈 2. Training curves...")
        plot_training_curves()
        
        print("🧠 3. Knowledge retention analysis...")
        plot_knowledge_retention()
        
        print("📋 4. Summary table...")
        generate_summary_table()
        
        print("✅ All visualizations generated successfully!")
        print("📁 Check the results/ directory for PNG files.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to populate the CSV files with your actual data first.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
