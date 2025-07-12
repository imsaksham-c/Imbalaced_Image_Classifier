import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_experiment_data(models_dir="./models"):
    """
    Load all experiment data from the models directory
    """
    experiments_data = {}
    
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory {models_dir} not found!")
        return experiments_data
    
    experiment_dirs = [d for d in models_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"Processing experiment: {exp_name}")
        
        exp_data = {
            'name': exp_name,
            'training_history': None,
            'classification_report': None,
            'config': None,
            'final_metrics': None,
            'model_path': None
        }
        
        # Load training history
        logs_dir = exp_dir / "logs"
        if logs_dir.exists():
            # Training history
            history_files = list(logs_dir.glob("training_history_*.json"))
            if history_files:
                try:
                    with open(history_files[0], 'r') as f:
                        exp_data['training_history'] = json.load(f)
                except Exception as e:
                    print(f"  Error loading training history for {exp_name}: {e}")
            
            # Training config
            config_file = logs_dir / "training_config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        exp_data['config'] = json.load(f)
                except Exception as e:
                    print(f"  Error loading config for {exp_name}: {e}")
            
            # Classification report
            report_file = logs_dir / "classification_report.txt"
            if report_file.exists():
                try:
                    with open(report_file, 'r') as f:
                        exp_data['classification_report'] = f.read()
                except Exception as e:
                    print(f"  Error loading classification report for {exp_name}: {e}")
        
        # Check for model file
        models_subdir = exp_dir / "models"
        if models_subdir.exists():
            model_files = list(models_subdir.glob("*.pth"))
            if model_files:
                exp_data['model_path'] = str(model_files[0])
        
        experiments_data[exp_name] = exp_data
        print(f"  ✅ Loaded data for {exp_name}")
    
    return experiments_data

def create_comprehensive_summary(experiments_data, output_file="comprehensive_model_summary.txt"):
    """
    Create a comprehensive summary file with all model information
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE MODEL TRAINING SUMMARY REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments: {len(experiments_data)}\n")
        f.write("=" * 100 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        # Create summary table
        summary_data = []
        for exp_name, data in experiments_data.items():
            config = data.get('config', {})
            final_metrics = data.get('training_history', {}).get('final_metrics', {})
            
            summary_data.append({
                'Experiment': exp_name,
                'Model': config.get('model', 'Unknown'),
                'Loss': config.get('loss_function', 'Unknown'),
                'Unfreeze': config.get('unfreeze_mode', 'Unknown'),
                'Best F1': final_metrics.get('best_f1_weighted', 0),
                'Final Acc': final_metrics.get('final_accuracy', 0),
                'Final F1 Macro': final_metrics.get('final_f1_macro', 0),
                'Epochs': len(data.get('training_history', {}).get('train_losses', []))
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Best F1', ascending=False)
        
        f.write("Top 5 Performing Experiments:\n")
        f.write(df.head().to_string(index=False))
        f.write("\n\n")
        
        f.write("Performance Statistics:\n")
        f.write(f"  Best F1 Score: {df['Best F1'].max():.4f} ({df.loc[df['Best F1'].idxmax(), 'Experiment']})\n")
        f.write(f"  Best Accuracy: {df['Final Acc'].max():.2f}% ({df.loc[df['Final Acc'].idxmax(), 'Experiment']})\n")
        f.write(f"  Average F1 Score: {df['Best F1'].mean():.4f}\n")
        f.write(f"  Average Accuracy: {df['Final Acc'].mean():.2f}%\n")
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Detailed Analysis by Model Type
        f.write("DETAILED ANALYSIS BY MODEL TYPE\n")
        f.write("=" * 50 + "\n\n")
        
        # Group by model
        model_groups = df.groupby('Model')
        for model_name, group in model_groups:
            f.write(f"{model_name.upper()} MODELS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total experiments: {len(group)}\n")
            f.write(f"Best F1 Score: {group['Best F1'].max():.4f}\n")
            f.write(f"Best Accuracy: {group['Final Acc'].max():.2f}%\n")
            f.write(f"Average F1 Score: {group['Best F1'].mean():.4f}\n")
            f.write(f"Average Accuracy: {group['Final Acc'].mean():.2f}%\n\n")
            
            # Best configuration for this model
            best_exp = group.loc[group['Best F1'].idxmax()]
            f.write(f"Best Configuration for {model_name}:\n")
            f.write(f"  Experiment: {best_exp['Experiment']}\n")
            f.write(f"  Loss Function: {best_exp['Loss']}\n")
            f.write(f"  Unfreeze Mode: {best_exp['Unfreeze']}\n")
            f.write(f"  F1 Score: {best_exp['Best F1']:.4f}\n")
            f.write(f"  Accuracy: {best_exp['Final Acc']:.2f}%\n\n")
        
        f.write("=" * 100 + "\n\n")
        
        # Detailed Experiment Reports
        f.write("DETAILED EXPERIMENT REPORTS\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (exp_name, data) in enumerate(experiments_data.items(), 1):
            f.write(f"EXPERIMENT {i}: {exp_name}\n")
            f.write("-" * 50 + "\n")
            
            # Configuration
            config = data.get('config', {})
            if config:
                f.write("Configuration:\n")
                f.write(f"  Model: {config.get('model', 'Unknown')}\n")
                f.write(f"  Loss Function: {config.get('loss_function', 'Unknown')}\n")
                f.write(f"  Unfreeze Mode: {config.get('unfreeze_mode', 'Unknown')}\n")
                f.write(f"  Batch Size: {config.get('batch_size', 'Unknown')}\n")
                f.write(f"  Epochs: {config.get('epochs', 'Unknown')}\n")
                f.write(f"  Learning Rate: {config.get('learning_rate', 'Unknown')}\n")
                f.write(f"  Weight Decay: {config.get('weight_decay', 'Unknown')}\n")
                f.write(f"  Gamma: {config.get('gamma', 'Unknown')}\n")
                f.write(f"  Number of Classes: {config.get('num_classes', 'Unknown')}\n")
                f.write(f"  Class Names: {config.get('class_names', 'Unknown')}\n\n")
            
            # Final Metrics
            final_metrics = data.get('training_history', {}).get('final_metrics', {})
            if final_metrics:
                f.write("Final Performance Metrics:\n")
                f.write(f"  Best F1 Score (Weighted): {final_metrics.get('best_f1_weighted', 'N/A'):.4f}\n")
                f.write(f"  Final Accuracy: {final_metrics.get('final_accuracy', 'N/A'):.2f}%\n")
                f.write(f"  Final F1 Macro: {final_metrics.get('final_f1_macro', 'N/A'):.4f}\n\n")
            
            # Training History Summary
            training_history = data.get('training_history', {})
            if training_history:
                train_losses = training_history.get('train_losses', [])
                val_losses = training_history.get('val_losses', [])
                train_accs = training_history.get('train_accs', [])
                val_accs = training_history.get('val_accs', [])
                f1_scores = training_history.get('f1_scores', [])
                
                if train_losses:
                    f.write("Training History Summary:\n")
                    f.write(f"  Total Epochs Trained: {len(train_losses)}\n")
                    f.write(f"  Final Training Loss: {train_losses[-1]:.4f}\n")
                    f.write(f"  Final Validation Loss: {val_losses[-1] if val_losses else 'N/A':.4f}\n")
                    f.write(f"  Final Training Accuracy: {train_accs[-1] if train_accs else 'N/A':.2f}%\n")
                    f.write(f"  Final Validation Accuracy: {val_accs[-1] if val_accs else 'N/A':.2f}%\n")
                    f.write(f"  Best F1 Score: {max(f1_scores) if f1_scores else 'N/A':.4f}\n")
                    f.write(f"  Best F1 Epoch: {f1_scores.index(max(f1_scores)) + 1 if f1_scores else 'N/A'}\n\n")
                    
                    # Convergence analysis
                    if len(train_losses) > 10:
                        early_loss = np.mean(train_losses[:5])
                        late_loss = np.mean(train_losses[-5:])
                        loss_reduction = ((early_loss - late_loss) / early_loss) * 100
                        f.write(f"  Loss Reduction: {loss_reduction:.2f}% (early vs late epochs)\n")
                        
                        if val_losses:
                            early_val_loss = np.mean(val_losses[:5])
                            late_val_loss = np.mean(val_losses[-5:])
                            val_loss_reduction = ((early_val_loss - late_val_loss) / early_val_loss) * 100
                            f.write(f"  Validation Loss Reduction: {val_loss_reduction:.2f}%\n")
                        f.write("\n")
            
            # Model File
            if data.get('model_path'):
                f.write(f"Model File: {data['model_path']}\n\n")
            
            # Classification Report
            if data.get('classification_report'):
                f.write("Classification Report:\n")
                f.write(data['classification_report'])
                f.write("\n")
            
            f.write("=" * 100 + "\n\n")
        
        # Comparative Analysis
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        # Loss Function Analysis
        f.write("Loss Function Performance:\n")
        loss_groups = df.groupby('Loss')
        for loss_name, group in loss_groups:
            f.write(f"  {loss_name}:\n")
            f.write(f"    Average F1: {group['Best F1'].mean():.4f}\n")
            f.write(f"    Average Accuracy: {group['Final Acc'].mean():.2f}%\n")
            f.write(f"    Best F1: {group['Best F1'].max():.4f}\n")
            f.write(f"    Best Accuracy: {group['Final Acc'].max():.2f}%\n\n")
        
        # Unfreeze Mode Analysis
        f.write("Unfreeze Mode Performance:\n")
        unfreeze_groups = df.groupby('Unfreeze')
        for unfreeze_mode, group in unfreeze_groups:
            f.write(f"  Unfreeze Mode {unfreeze_mode}:\n")
            f.write(f"    Average F1: {group['Best F1'].mean():.4f}\n")
            f.write(f"    Average Accuracy: {group['Final Acc'].mean():.2f}%\n")
            f.write(f"    Best F1: {group['Best F1'].max():.4f}\n")
            f.write(f"    Best Accuracy: {group['Final Acc'].max():.2f}%\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 50 + "\n\n")
        
        best_overall = df.loc[df['Best F1'].idxmax()]
        f.write(f"Best Overall Model: {best_overall['Experiment']}\n")
        f.write(f"  Model: {best_overall['Model']}\n")
        f.write(f"  Loss Function: {best_overall['Loss']}\n")
        f.write(f"  Unfreeze Mode: {best_overall['Unfreeze']}\n")
        f.write(f"  F1 Score: {best_overall['Best F1']:.4f}\n")
        f.write(f"  Accuracy: {best_overall['Final Acc']:.2f}%\n\n")
        
        # Model-specific recommendations
        for model_name, group in model_groups:
            best_model = group.loc[group['Best F1'].idxmax()]
            f.write(f"Best {model_name} Configuration:\n")
            f.write(f"  Experiment: {best_model['Experiment']}\n")
            f.write(f"  Loss: {best_model['Loss']}\n")
            f.write(f"  Unfreeze: {best_model['Unfreeze']}\n")
            f.write(f"  F1: {best_model['Best F1']:.4f}\n\n")
        
        # Training insights
        f.write("Training Insights:\n")
        avg_epochs = df['Epochs'].mean()
        f.write(f"  Average epochs trained: {avg_epochs:.1f}\n")
        
        # Check for overfitting patterns
        overfitting_count = 0
        for exp_name, data in experiments_data.items():
            training_history = data.get('training_history', {})
            if training_history:
                train_losses = training_history.get('train_losses', [])
                val_losses = training_history.get('val_losses', [])
                if len(train_losses) > 10 and len(val_losses) > 10:
                    early_val_loss = np.mean(val_losses[:5])
                    late_val_loss = np.mean(val_losses[-5:])
                    if late_val_loss > early_val_loss * 1.1:  # 10% increase
                        overfitting_count += 1
        
        f.write(f"  Potential overfitting detected in {overfitting_count} experiments\n")
        
        # Conclusion
        f.write("\nCONCLUSION\n")
        f.write("=" * 50 + "\n")
        f.write(f"This comprehensive analysis covers {len(experiments_data)} experiments ")
        f.write(f"with {len(df['Model'].unique())} different model architectures. ")
        f.write(f"The best performing model achieved an F1 score of {df['Best F1'].max():.4f} ")
        f.write(f"and accuracy of {df['Final Acc'].max():.2f}%. ")
        f.write("The results provide insights into the effectiveness of different ")
        f.write("model architectures, loss functions, and training strategies for ")
        f.write("the temple image classification task.\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

def main():
    """
    Main function to create comprehensive summary
    """
    print("🔍 Loading all experiment data...")
    experiments_data = load_experiment_data()
    
    if not experiments_data:
        print("❌ No experiment data found!")
        return
    
    print(f"✅ Loaded data for {len(experiments_data)} experiments")
    
    print("\n📝 Creating comprehensive summary report...")
    output_file = "comprehensive_model_summary.txt"
    create_comprehensive_summary(experiments_data, output_file)
    
    print(f"\n🎉 Comprehensive summary created successfully!")
    print(f"📁 Summary file: {output_file}")
    
    # Also create a CSV summary
    summary_data = []
    for exp_name, data in experiments_data.items():
        config = data.get('config', {})
        final_metrics = data.get('training_history', {}).get('final_metrics', {})
        training_history = data.get('training_history', {})
        
        summary_data.append({
            'Experiment': exp_name,
            'Model': config.get('model', 'Unknown'),
            'Loss_Function': config.get('loss_function', 'Unknown'),
            'Unfreeze_Mode': config.get('unfreeze_mode', 'Unknown'),
            'Batch_Size': config.get('batch_size', 'Unknown'),
            'Epochs': config.get('epochs', 'Unknown'),
            'Learning_Rate': config.get('learning_rate', 'Unknown'),
            'Weight_Decay': config.get('weight_decay', 'Unknown'),
            'Gamma': config.get('gamma', 'Unknown'),
            'Num_Classes': config.get('num_classes', 'Unknown'),
            'Best_F1_Score': final_metrics.get('best_f1_weighted', 0),
            'Final_Accuracy': final_metrics.get('final_accuracy', 0),
            'Final_F1_Macro': final_metrics.get('final_f1_macro', 0),
            'Epochs_Trained': len(training_history.get('train_losses', [])),
            'Final_Train_Loss': training_history.get('train_losses', [0])[-1] if training_history.get('train_losses') else 0,
            'Final_Val_Loss': training_history.get('val_losses', [0])[-1] if training_history.get('val_losses') else 0,
            'Final_Train_Acc': training_history.get('train_accs', [0])[-1] if training_history.get('train_accs') else 0,
            'Final_Val_Acc': training_history.get('val_accs', [0])[-1] if training_history.get('val_accs') else 0,
            'Model_File': data.get('model_path', 'N/A')
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Best_F1_Score', ascending=False)
    csv_file = "comprehensive_model_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"📊 CSV summary also created: {csv_file}")

if __name__ == "__main__":
    main() 