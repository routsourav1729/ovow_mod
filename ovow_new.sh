# Create a fresh environment
conda create -n ovow4 python=3.8
conda activate ovow4

# Install PyTorch with the exact CUDA version needed
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Install other dependencies from your original environment
conda install -c conda-forge pillow=9.5.0

# Install detectron2 with the matching PyTorch/CUDA version
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.11/index.html

# Install other project dependencies
cd /users/student/pg/pg23/souravrout/ALL_FILES/thesis/honda/ovow
pip install -r requirements.txt