#!/bin/bash
# OVOW Environment Installation Script
# Tested and working as of Jan 2026

set -e  # Exit on error

echo "=========================================="
echo "  Installing OVOW Environment"
echo "=========================================="

# Step 1: Create conda environment with Python 3.11
echo "[1/11] Creating conda environment 'ovow' with Python 3.11..."
conda create -n ovow python=3.11 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ovow

# Step 2: Install PyTorch 2.3.1 with CUDA 12.1 (NOT 2.1.0 from official docs)
echo "[2/11] Installing PyTorch 2.3.1+cu121..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install GCC 11 (required for C++ compilation - system GCC 8.5 is too old)
echo "[3/11] Installing GCC 11 from conda-forge..."
conda install -y gxx_linux-64=11.2.0 -c conda-forge

# Step 4: Install mmcv 2.1.0 with C++ extensions using GCC 11
echo "[4/11] Installing mmcv 2.1.0 with C++ ops..."
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
MMCV_WITH_OPS=1 pip install mmcv==2.1.0 --no-cache-dir

# Step 5: Install mmdet and mmyolo
echo "[5/11] Installing mmdet and mmyolo..."
pip install mmdet==3.3.0 mmyolo==0.6.0

# Step 6: Install YOLO-World at specific commit
echo "[6/11] Installing YOLO-World..."
cd YOLO-World
git checkout 4d90f458c1d0de310643b0ac2498f188c98c819c
pip install -e . --no-deps
cd ..

# Step 7: Install detectron2 using GCC 11
echo "[7/11] Installing detectron2..."
CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Step 8: Install CLIP
echo "[8/11] Installing CLIP..."
pip install 'git+https://github.com/openai/CLIP.git'

# Step 9: Install transformers and tokenizers at specific versions
echo "[9/11] Installing transformers and tokenizers..."
pip install transformers==4.36.0 tokenizers==0.15.2

# Step 10: Install remaining dependencies
echo "[10/11] Installing remaining dependencies..."
pip install supervision==0.19.0 openmim wandb

# Step 11: Patch mmyolo to accept mmcv 2.1.0
echo "[11/11] Patching mmyolo version check..."
MMYOLO_INIT="$CONDA_PREFIX/lib/python3.11/site-packages/mmyolo/__init__.py"
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.2.0'/" "$MMYOLO_INIT"

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ovow"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch, mmcv, mmdet, mmyolo, detectron2, clip; print(\"All imports successful!\")'"
echo ""
