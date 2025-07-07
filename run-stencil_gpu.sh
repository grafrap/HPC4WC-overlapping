#!/bin/bash
#SBATCH --job-name=stencil_gpu_test
#SBATCH --output=measurements/stencil_gpu_test_%j.out
#SBATCH --error=measurements/stencil_gpu_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --exclusive

echo "Job started on $(hostname)"
echo "Testing GPU stencil implementation"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Set up CUDA environment
export CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3
export PATH=${CUDA_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH

# Build the project
/usr/bin/cmake -DCUDAToolkit_ROOT=${CUDA_ROOT} ..
make clean
make

echo "Build completed, testing GPU stencil executable..."

# Test with different problem sizes
echo "=== Testing small problem size ==="
./stencil2d_gpu -nx 32 -ny 32 -nz 32 -iter 1

echo "=== Testing medium problem size ==="
./stencil2d_gpu -nx 64 -ny 64 -nz 64 -iter 5

echo "=== Testing larger problem size ==="
./stencil2d_gpu -nx 128 -ny 128 -nz 128 -iter 10

echo "=== Comparing with CPU version (if available) ==="
if [ -f "../src/cuda/stencil2d_helper/stencil2d-base" ]; then
    echo "Running CPU version for comparison..."
    ../src/cuda/stencil2d_helper/stencil2d-base -nx 64 -ny 64 -nz 64 -iter 5
fi

echo "GPU stencil test completed"
