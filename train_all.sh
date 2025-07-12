#!/bin/bash

# train_all.sh - Run all possible combinations of train.py
# This script runs 6 experiments covering all combinations of:
# - Models: resnet50
# - Loss functions: weightedce, focalloss
# - Unfreeze modes: 0, 1, 2

# Configuration
DATA_DIR="processed_dataset"
BATCH_SIZE=32
EPOCHS=50
LR=0.001
WEIGHT_DECAY=0.0001
GAMMA=2.0
SAVE_DIR="./models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Function to run a single experiment
run_experiment() {
    local model=$1
    local loss=$2
    local unfreeze=$3
    local exp_name=$4
    
    print_header "=================================================================================="
    print_status "Starting experiment: $exp_name"
    print_status "Model: $model, Loss: $loss, Unfreeze: $unfreeze"
    print_header "=================================================================================="
    
    start_time=$(date +%s)
    
    # Run the training command
    python train.py \
        --model "$model" \
        --loss "$loss" \
        --unfreeze "$unfreeze" \
        --experiment_name "$exp_name" \
        --data_dir "$DATA_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --gamma "$GAMMA" \
        --save_dir "$SAVE_DIR"
    
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        print_success "Experiment $exp_name completed successfully!"
        print_success "Duration: ${duration}s ($(echo "scale=2; $duration/60" | bc)m)"
        return 0
    else
        print_error "Experiment $exp_name failed with exit code $exit_code!"
        print_error "Duration: ${duration}s ($(echo "scale=2; $duration/60" | bc)m)"
        return 1
    fi
}

# Function to print experiment summary
print_experiment_list() {
    print_header "=================================================================================="
    print_header "EXPERIMENT CONFIGURATION"
    print_header "=================================================================================="
    echo "Data Directory: $DATA_DIR"
    echo "Batch Size: $BATCH_SIZE"
    echo "Epochs: $EPOCHS"
    echo "Learning Rate: $LR"
    echo "Weight Decay: $WEIGHT_DECAY"
    echo "Gamma: $GAMMA"
    echo "Save Directory: $SAVE_DIR"
    echo ""
    print_header "EXPERIMENTS TO RUN:"
    echo "1.  resnet_wce_0    (ResNet50 + Weighted CE + Unfreeze 0)"
    echo "2.  resnet_wce_1    (ResNet50 + Weighted CE + Unfreeze 1)"
    echo "3.  resnet_wce_2    (ResNet50 + Weighted CE + Unfreeze 2)"
    echo "4.  resnet_focal_0  (ResNet50 + Focal Loss + Unfreeze 0)"
    echo "5.  resnet_focal_1  (ResNet50 + Focal Loss + Unfreeze 1)"
    echo "6.  resnet_focal_2  (ResNet50 + Focal Loss + Unfreeze 2)"
    print_header "=================================================================================="
}

# Function to save experiment results
save_results() {
    local results_file="experiment_results_$(date +%Y%m%d_%H%M%S).txt"
    echo "Experiment Results - $(date)" > "$results_file"
    echo "==========================================" >> "$results_file"
    echo "Total Experiments: $total_experiments" >> "$results_file"
    echo "Successful: $successful_experiments" >> "$results_file"
    echo "Failed: $failed_experiments" >> "$results_file"
    echo "Success Rate: $(echo "scale=1; $successful_experiments * 100 / $total_experiments" | bc)%" >> "$results_file"
    echo "Total Duration: ${total_duration}s ($(echo "scale=2; $total_duration/3600" | bc)h)" >> "$results_file"
    echo "" >> "$results_file"
    echo "Detailed Results:" >> "$results_file"
    echo "=================" >> "$results_file"
    
    for i in "${!experiment_names[@]}"; do
        echo "${experiment_names[$i]}: ${results[$i]}" >> "$results_file"
    done
    
    print_success "Results saved to: $results_file"
}

# Main execution
main() {
    print_header "=================================================================================="
    print_header "🚀 COMPREHENSIVE TRAINING EXPERIMENT SCRIPT"
    print_header "=================================================================================="
    
    # Check if train.py exists
    if [ ! -f "train.py" ]; then
        print_error "train.py not found in current directory!"
        exit 1
    fi
    
    # Print experiment configuration
    print_experiment_list
    
    # Ask for confirmation
    echo ""
    read -p "Do you want to proceed with running all 6 experiments? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Experiment cancelled by user"
        exit 0
    fi
    
    # Initialize tracking variables
    total_experiments=6
    successful_experiments=0
    failed_experiments=0
    total_duration=0
    declare -a experiment_names
    declare -a results
    
    # Start time
    overall_start_time=$(date +%s)
    
    # Experiment 1: ResNet50 + Weighted CE + Unfreeze 0
    experiment_names[0]="resnet_wce_0"
    if run_experiment "resnet50" "weightedce" "0" "resnet_wce_0"; then
        results[0]="SUCCESS"
        ((successful_experiments++))
    else
        results[0]="FAILED"
        ((failed_experiments++))
    fi
    
    # Experiment 2: ResNet50 + Weighted CE + Unfreeze 1
    experiment_names[1]="resnet_wce_1"
    if run_experiment "resnet50" "weightedce" "1" "resnet_wce_1"; then
        results[1]="SUCCESS"
        ((successful_experiments++))
    else
        results[1]="FAILED"
        ((failed_experiments++))
    fi
    
    # Experiment 3: ResNet50 + Weighted CE + Unfreeze 2
    experiment_names[2]="resnet_wce_2"
    if run_experiment "resnet50" "weightedce" "2" "resnet_wce_2"; then
        results[2]="SUCCESS"
        ((successful_experiments++))
    else
        results[2]="FAILED"
        ((failed_experiments++))
    fi
    
    # Experiment 4: ResNet50 + Focal Loss + Unfreeze 0
    experiment_names[3]="resnet_focal_0"
    if run_experiment "resnet50" "focalloss" "0" "resnet_focal_0"; then
        results[3]="SUCCESS"
        ((successful_experiments++))
    else
        results[3]="FAILED"
        ((failed_experiments++))
    fi
    
    # Experiment 5: ResNet50 + Focal Loss + Unfreeze 1
    experiment_names[4]="resnet_focal_1"
    if run_experiment "resnet50" "focalloss" "1" "resnet_focal_1"; then
        results[4]="SUCCESS"
        ((successful_experiments++))
    else
        results[4]="FAILED"
        ((failed_experiments++))
    fi
    
    # Experiment 6: ResNet50 + Focal Loss + Unfreeze 2
    experiment_names[5]="resnet_focal_2"
    if run_experiment "resnet50" "focalloss" "2" "resnet_focal_2"; then
        results[5]="SUCCESS"
        ((successful_experiments++))
    else
        results[5]="FAILED"
        ((failed_experiments++))
    fi
    

    
    # Calculate total duration
    overall_end_time=$(date +%s)
    total_duration=$((overall_end_time - overall_start_time))
    
    # Print final summary
    print_header "=================================================================================="
    print_header "🎉 ALL EXPERIMENTS COMPLETED!"
    print_header "=================================================================================="
    print_success "Total execution time: ${total_duration}s ($(echo "scale=2; $total_duration/3600" | bc)h)"
    echo ""
    print_header "FINAL SUMMARY:"
    echo "Total experiments: $total_experiments"
    echo "Successful: $successful_experiments"
    echo "Failed: $failed_experiments"
    echo "Success rate: $(echo "scale=1; $successful_experiments * 100 / $total_experiments" | bc)%"
    echo ""
    
    print_header "DETAILED RESULTS:"
    echo "==================="
    for i in "${!experiment_names[@]}"; do
        if [ "${results[$i]}" = "SUCCESS" ]; then
            echo -e "${GREEN}✅${NC} ${experiment_names[$i]}: ${results[$i]}"
        else
            echo -e "${RED}❌${NC} ${experiment_names[$i]}: ${results[$i]}"
        fi
    done
    
    # Save results to file
    save_results
    
    print_header "=================================================================================="
    print_success "🎯 You can now compare results across all experiments!"
    print_header "=================================================================================="
}

# Run main function
main "$@" 