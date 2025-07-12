import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_experiment_data(models_dir="./models"):
    """
    Load all experiment data from the models directory
    """
    experiments_data = {}
    
    # Get all experiment directories
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory {models_dir} not found!")
        return experiments_data
    
    # Find all experiment directories
    experiment_dirs = [d for d in models_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"Processing experiment: {exp_name}")
        
        # Look for training history JSON files
        logs_dir = exp_dir / "logs"
        if not logs_dir.exists():
            print(f"  No logs directory found for {exp_name}")
            continue
        
        # Find training history files
        history_files = list(logs_dir.glob("training_history_*.json"))
        if not history_files:
            print(f"  No training history found for {exp_name}")
            continue
        
        # Load the training history
        try:
            with open(history_files[0], 'r') as f:
                history_data = json.load(f)
            
            # Extract key metrics
            experiments_data[exp_name] = {
                'train_losses': history_data.get('train_losses', []),
                'val_losses': history_data.get('val_losses', []),
                'train_accs': history_data.get('train_accs', []),
                'val_accs': history_data.get('val_accs', []),
                'f1_scores': history_data.get('f1_scores', []),
                'config': history_data.get('config', {}),
                'final_metrics': history_data.get('final_metrics', {})
            }
            
            print(f"  ✅ Loaded data for {exp_name} ({len(experiments_data[exp_name]['train_losses'])} epochs)")
            
        except Exception as e:
            print(f"  ❌ Error loading {exp_name}: {e}")
            continue
    
    return experiments_data

def create_comprehensive_comparison_plots(experiments_data, save_dir="./All_Models"):
    """
    Create comprehensive comparison plots for all experiments
    """
    if not experiments_data:
        print("No experiment data to plot!")
        return
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('All Models Comparison - Training Metrics', fontsize=16, fontweight='bold')
    
    # Color palette for different experiments
    colors = plt.cm.tab20(np.linspace(0, 1, len(experiments_data)))
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        if data['train_losses']:
            epochs = range(1, len(data['train_losses']) + 1)
            ax1.plot(epochs, data['train_losses'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        if data['val_losses']:
            epochs = range(1, len(data['val_losses']) + 1)
            ax2.plot(epochs, data['val_losses'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[0, 2]
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        if data['train_accs']:
            epochs = range(1, len(data['train_accs']) + 1)
            ax3.plot(epochs, data['train_accs'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
    ax3.set_title('Training Accuracy Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Accuracy (%)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax4 = axes[1, 0]
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        if data['val_accs']:
            epochs = range(1, len(data['val_accs']) + 1)
            ax4.plot(epochs, data['val_accs'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
    ax4.set_title('Validation Accuracy Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: F1 Score
    ax5 = axes[1, 1]
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        if data['f1_scores']:
            epochs = range(1, len(data['f1_scores']) + 1)
            ax5.plot(epochs, data['f1_scores'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
    ax5.set_title('F1 Score Comparison')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('F1 Score')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Final Metrics Comparison
    ax6 = axes[1, 2]
    exp_names = []
    final_f1_scores = []
    final_accuracies = []
    
    for exp_name, data in experiments_data.items():
        if 'final_metrics' in data and data['final_metrics']:
            exp_names.append(exp_name)
            final_f1_scores.append(data['final_metrics'].get('best_f1_weighted', 0))
            final_accuracies.append(data['final_metrics'].get('final_accuracy', 0))
    
    if exp_names:
        x = np.arange(len(exp_names))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, final_f1_scores, width, label='F1 Score', alpha=0.8)
        bars2 = ax6.bar(x + width/2, final_accuracies, width, label='Accuracy (%)', alpha=0.8)
        
        ax6.set_title('Final Performance Comparison')
        ax6.set_xlabel('Experiments')
        ax6.set_ylabel('Score')
        ax6.set_xticks(x)
        ax6.set_xticklabels(exp_names, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_models_comprehensive_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual comparison plots
    create_individual_comparison_plots(experiments_data, save_dir)
    
    # Create summary table
    create_summary_table(experiments_data, save_dir)

def create_individual_comparison_plots(experiments_data, save_dir):
    """
    Create individual comparison plots for each metric
    """
    colors = plt.cm.tab20(np.linspace(0, 1, len(experiments_data)))
    
    # 1. Loss Comparison
    plt.figure(figsize=(15, 8))
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        if data['train_losses'] and data['val_losses']:
            epochs = range(1, len(data['train_losses']) + 1)
            plt.plot(epochs, data['train_losses'], label=f'{exp_name} (Train)', 
                    color=colors[i], linewidth=2, alpha=0.7, linestyle='-')
            plt.plot(epochs, data['val_losses'], label=f'{exp_name} (Val)', 
                    color=colors[i], linewidth=2, alpha=0.7, linestyle='--')
    
    plt.title('All Models - Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_models_loss_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Accuracy Comparison
    plt.figure(figsize=(15, 8))
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        if data['train_accs'] and data['val_accs']:
            epochs = range(1, len(data['train_accs']) + 1)
            plt.plot(epochs, data['train_accs'], label=f'{exp_name} (Train)', 
                    color=colors[i], linewidth=2, alpha=0.7, linestyle='-')
            plt.plot(epochs, data['val_accs'], label=f'{exp_name} (Val)', 
                    color=colors[i], linewidth=2, alpha=0.7, linestyle='--')
    
    plt.title('All Models - Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_models_accuracy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. F1 Score Comparison
    plt.figure(figsize=(15, 8))
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        if data['f1_scores']:
            epochs = range(1, len(data['f1_scores']) + 1)
            plt.plot(epochs, data['f1_scores'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
    
    plt.title('All Models - F1 Score Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_models_f1_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(experiments_data, save_dir):
    """
    Create a summary table of all experiments
    """
    summary_data = []
    
    for exp_name, data in experiments_data.items():
        config = data.get('config', {})
        final_metrics = data.get('final_metrics', {})
        
        summary_data.append({
            'Experiment': exp_name,
            'Model': config.get('model', 'Unknown'),
            'Loss Function': config.get('loss_function', 'Unknown'),
            'Unfreeze Mode': config.get('unfreeze_mode', 'Unknown'),
            'Batch Size': config.get('batch_size', 'Unknown'),
            'Epochs': config.get('epochs', 'Unknown'),
            'Best F1 Score': final_metrics.get('best_f1_weighted', 0),
            'Final Accuracy (%)': final_metrics.get('final_accuracy', 0),
            'Final F1 Macro': final_metrics.get('final_f1_macro', 0),
            'Max Epochs Trained': len(data.get('train_losses', []))
        })
    
    # Create DataFrame and sort by F1 score
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Best F1 Score', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'all_models_summary.csv')
    df.to_csv(csv_path, index=False)
    
    # Create a formatted table plot
    fig, ax = plt.subplots(figsize=(16, len(df) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = df[['Experiment', 'Model', 'Loss Function', 'Unfreeze Mode', 
                     'Best F1 Score', 'Final Accuracy (%)']].values
    
    table = ax.table(cellText=table_data, 
                    colLabels=['Experiment', 'Model', 'Loss', 'Unfreeze', 
                              'Best F1', 'Final Acc (%)'],
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('All Models Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_models_summary_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary table saved to: {csv_path}")
    print("\nTop 5 performing experiments:")
    print(df.head().to_string(index=False))

def main():
    """
    Main function to run the comparison analysis
    """
    print("🔍 Loading all experiment data...")
    experiments_data = load_experiment_data()
    
    if not experiments_data:
        print("❌ No experiment data found!")
        return
    
    print(f"✅ Loaded data for {len(experiments_data)} experiments")
    
    print("\n📊 Creating comprehensive comparison plots...")
    create_comprehensive_comparison_plots(experiments_data)
    
    print("\n🎉 All comparison plots created successfully!")
    print("📁 Check the 'All_Models' directory for all plots and summary files.")

if __name__ == "__main__":
    main() 