#!/bin/bash

#SBATCH --job-name=ovow-idd-t1
#SBATCH --error=slurm_logs/ovow_idd_t1.%J.err
#SBATCH --output=slurm_logs/ovow_idd_t1-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --partition=dgx          # use DGX nodes (change if different partition name)
#SBATCH --qos=dgx                # adjust if your site uses a different QoS
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# NCCL settings to avoid multi-GPU deadlocks
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export NCCL_LAUNCH_MODE=GROUP

# Force offline mode for HuggingFace
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================================"
echo "SLURM JOB: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "========================================================"

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

echo ">>> Starting OVOW IDD Base Training (Task 1)..."
echo ">>> Model: YOLO-World-v2-XL"
echo ">>> Dataset: IDD (Indian Driving Dataset)"
echo ">>> Task: t1 (11 known classes)"

# Navigate to OVOW directory
cd /home/agipml/sourav.rout/ALL_FILES/HONDA/ovow

# Activate conda environment
source ~/.bashrc
# Ensure conda is initialized for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate ovow

# Avoid tokenizers fork warnings
export TOKENIZERS_PARALLELISM=false

# Set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# Verify dataset setup
echo "========================================================"
echo ">>> Verifying dataset setup..."
echo ">>> Annotations: $(ls datasets/Annotations/*.xml 2>/dev/null | wc -l) files"
echo ">>> Images: $(find datasets/JPEGImages -name '*.jpg' 2>/dev/null | wc -l) files"
echo ">>> Train split: $(wc -l < datasets/ImageSets/Main/IDD/t1.txt) samples"
echo ">>> Test split: $(wc -l < datasets/ImageSets/Main/IDD/test.txt) samples"
echo "========================================================"

# Check if pretrained weights exist
PRETRAIN_WEIGHTS="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
if [ ! -f "$PRETRAIN_WEIGHTS" ]; then
    echo "ERROR: Pretrained weights not found: $PRETRAIN_WEIGHTS"
    echo "Please download from YOLO-World repository"
    exit 1
fi
echo ">>> Using pretrained weights: $PRETRAIN_WEIGHTS"

# Training configuration
BENCHMARK="IDD"
TASK="${BENCHMARK}/t1"
CONFIG="configs/${BENCHMARK}/t1.yaml"
CHECKPOINT="$PRETRAIN_WEIGHTS"
OUTPUT_DIR="IDD/t1"

echo "========================================================"
echo ">>> Training Configuration:"
echo ">>>   Benchmark: $BENCHMARK"
echo ">>>   Task: $TASK"
echo ">>>   Config: $CONFIG"
echo ">>>   Checkpoint: $CHECKPOINT"
echo ">>>   Output: $OUTPUT_DIR"
echo ">>>   Known classes: 11"
echo ">>>   Total classes: 24 (11 known + 13 unknown)"
echo "========================================================"
echo ">>> BASE TRAINING SETTINGS:"
echo ">>>   - Backbone: ResNet-50 (trainable)"
echo ">>>   - Neck: FPN (trainable)"
echo ">>>   - Head: RandBox (trainable)"
echo ">>>   - Batch size: 12"
echo ">>>   - Learning rate: 2.5e-5"
echo ">>>   - Max iterations: 120,000"
echo ">>>   - Evaluation: every 10,000 iterations"
echo "========================================================"

# Run training
python dev.py \
    --task ${TASK} \
    --config-file ${CONFIG} \
    --ckpt ${CHECKPOINT} \
    2>&1 | tee slurm_logs/training_idd_t1_${SLURM_JOB_ID}.log

echo "========================================================"
echo ">>> OVOW IDD Base Training finished."
echo ">>> Checkpoints saved in: $OUTPUT_DIR"
echo ">>> Training log: slurm_logs/training_idd_t1_${SLURM_JOB_ID}.log"
echo "========================================================"
