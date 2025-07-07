#!/bin/bash
#SBATCH --job-name=stencil_gpu_scaling
#SBATCH --output=measurements/stencil_scaling_%j.out
#SBATCH --error=measurements/stencil_scaling_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --exclusive

echo "Job started on $(hostname)"
echo "Running GPU stencil scaling tests"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Set up CUDA environment
export CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3
export PATH=${CUDA_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH

# Build the project
/usr/bin/cmake -DCUDAToolkit_ROOT=${CUDA_ROOT} ..
make

# Create output file for results
OUTPUT_FILE="../measurements/stencil_gpu_scaling_$SLURM_JOB_ID.out"
ERROR_FILE="../measurements/stencil_gpu_scaling_$SLURM_JOB_ID.err"
mkdir -p ../measurements

echo "Saving results to: $OUTPUT_FILE"
echo "Beginning GPU stencil scaling tests..." | tee -a $OUTPUT_FILE

# Test scaling with different problem sizes (powers of 2)
for ((k=5; k<=8; k++)); do
    SIZE=$((2**k))
    echo "Testing problem size: ${SIZE}x${SIZE}x${SIZE}" | tee -a $OUTPUT_FILE
    ./stencil2d_gpu -nx $SIZE -ny $SIZE -nz $SIZE -iter 10 >> $OUTPUT_FILE 2>> $ERROR_FILE
done

# Test scaling with different iteration counts
echo "Testing iteration scaling with fixed size 64x64x64:" | tee -a $OUTPUT_FILE
for iter in 1 5 10 20 50; do
    echo "Iterations: $iter" | tee -a $OUTPUT_FILE
    ./stencil2d_gpu -nx 64 -ny 64 -nz 64 -iter $iter >> $OUTPUT_FILE 2>> $ERROR_FILE
done

echo "GPU stencil scaling tests completed"
