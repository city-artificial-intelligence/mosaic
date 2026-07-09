# Installation

## Removing any old env

### 1. Navigate to your project folder
cd /var/home/faiz/Desktop/mosaic

### 2. Delete any old virtual environment folders completely
rm -rf .venv

### 3. Delete the custom Jupyter kernels we generated so they don't linger
rm -rf ~/.local/share/jupyter/kernels/mosaic-rocm

## Clean and Sync the Container Environment
distrobox enter ml-env

### 1. Create a pristine virtual environment
/usr/bin/python3.12 -m venv .venv

### 2. Activate it
source .venv/bin/activate

## Install packages

### 1. Install PyTorch built for AMD ROCm 6.1
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/rocm6.1

### 2. Install the execution dependencies
pip install --no-cache-dir transformers sentence-transformers accelerate datasets rdflib faiss-cpu numpy

## run py file natively 
### GFX version depends on gpu generation (6000 series used for testing)
HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCM_INIT_FLAGS=1 python align_code/amdCode.py

# Cheat sheet to run container
## 1. Step into your AMD-ready container
distrobox enter ml-env

## 2. Activate your clean python environment
source .venv/bin/activate

## 3. Run your script on the GPU
HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCM_INIT_FLAGS=1 python align_code/amdCode.py


# notes
## remove .embeddingcache folder if the model is changed to prevent recall drops
rm -rf /var/home/faiz/Desktop/mosaic/align_code/.embedding_cache