#!/bin/bash
# Script to launch shared_label_space experiments with various configurations
# Located in experiments/scripts/launch_shared_experiments.sh

# Default values
DATA_DIR="/scr/"
MODEL_TYPE="rn50"
MERGING="pleas_weight"
MAX_STEPS=400
SEED=42
OUTPUT_DIR="./outputs"
USE_ZIP_RATIOS=""  # Empty by default, will be set with --use_zip_ratios flag if needed

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --merging)
      MERGING="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --use_zip_ratios)
      USE_ZIP_RATIOS="--use_zip_ratios"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Domain pairs to run experiments on
declare -a DOMAIN_PAIRS=(
  "clipart painting"
  "clipart infograph"
  "clipart real"
  "painting infograph"
  "painting real"
  "infograph real"
)

# Model variant pairs
declare -a VARIANT_PAIRS=(
  "v1a v1b"
  "v1b v1a"
)

# Budget ratios based on model type
if [[ "$MODEL_TYPE" == "rn18" ]]; then
  BUDGET_RATIOS=(1.0 1.24 1.46 1.71 2.0)
elif [[ "$MODEL_TYPE" == "rn50" ]]; then
  BUDGET_RATIOS=(1.0 1.2 1.55 1.8 2.0)
elif [[ "$MODEL_TYPE" == "rn101" ]]; then
  BUDGET_RATIOS=(1.0 1.1 1.7 1.8 2.0)
else
  echo "Unknown model type: $MODEL_TYPE"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/shared_experiment_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Starting shared label space experiments at $(date)" | tee -a "$LOG_FILE"
echo "Model type: $MODEL_TYPE" | tee -a "$LOG_FILE"
echo "Merging strategy: $MERGING" | tee -a "$LOG_FILE"

# Run experiments
experiment_count=0
for domain_pair in "${DOMAIN_PAIRS[@]}"; do
  # Parse domain pair
  read -r domain1 domain2 <<< "$domain_pair"
  
  for variant_pair in "${VARIANT_PAIRS[@]}"; do
    # Parse variant pair
    read -r variant1 variant2 <<< "$variant_pair"
    
    for budget_ratio in "${BUDGET_RATIOS[@]}"; do
      experiment_count=$((experiment_count + 1))
      
      # Unique experiment name for wandb
      exp_name="${domain1}_${domain2}_${variant1}_${variant2}_${MERGING}_${budget_ratio}"
      
      echo "====================================" | tee -a "$LOG_FILE"
      echo "Running experiment #$experiment_count: $exp_name" | tee -a "$LOG_FILE"
      echo "====================================" | tee -a "$LOG_FILE"
      
      # Run command
      cmd="python shared_label_space.py \
        --domain1 $domain1 \
        --domain2 $domain2 \
        --data_dir $DATA_DIR \
        --model_type $MODEL_TYPE \
        --variant1 $variant1 \
        --variant2 $variant2 \
        --merging $MERGING \
        --budget_ratio $budget_ratio \
        $USE_ZIP_RATIOS \
        --wandb \
        --max_steps $MAX_STEPS \
        --seed $SEED \
        --output_dir $OUTPUT_DIR/shared_$exp_name"
      
      echo "$cmd" | tee -a "$LOG_FILE"
      eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
      
      echo "Completed experiment: $exp_name with exit code $?" | tee -a "$LOG_FILE"
      echo "" | tee -a "$LOG_FILE"
    done
  done
done

echo "All shared label space experiments completed. Total experiments: $experiment_count" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"