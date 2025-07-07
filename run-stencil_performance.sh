#!/bin/bash
#SBATCH --job-name=stencil_perf_comparison
#SBATCH --output=measurements/stencil_perf_%j.out
#SBATCH --error=measurements/stencil_perf_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclusive

echo "Job started on $(hostname)"
echo "Running stencil performance comparison: CPU vs GPU"

# Create measurements directory
mkdir -p measurements
cd build

# Set up CUDA environment
export CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3
export PATH=${CUDA_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH

# Build the project
/usr/bin/cmake -DCUDAToolkit_ROOT=${CUDA_ROOT} ..
make

echo "Build completed. Starting performance tests..."

# Create output file for performance data
PERF_FILE="../measurements/stencil_performance_$SLURM_JOB_ID.csv"
echo "ProblemSize,Version,Iterations,Time(s)" > $PERF_FILE

# Test different problem sizes
SIZES=(32 64 128 256)
ITERATIONS=(1 5 10)

for size in "${SIZES[@]}"; do
    for iter in "${ITERATIONS[@]}"; do
        echo "=== Testing size ${size}x${size}x${size}, ${iter} iterations ==="
        
        # Test GPU version
        echo "Running GPU version..."
        GPU_OUTPUT=$(./stencil2d_gpu -nx $size -ny $size -nz $size -iter $iter 2>/dev/null | grep -E "^\[.*\]$" | tail -1)
        if [ ! -z "$GPU_OUTPUT" ]; then
            # Extract time from output (assuming format like: [ 1, 64, 64, 64, 5, 0.123456])
            TIME=$(echo $GPU_OUTPUT | sed 's/.*,\s*\([0-9.e-]*\)\s*].*/\1/')
            echo "${size},GPU,${iter},${TIME}" >> $PERF_FILE
            echo "GPU time: ${TIME} seconds"
        fi
        
        # Compile and test CPU version if we can build it
        if [ -f "../src/cuda/stencil2d_helper/stencil2d-base.cpp" ]; then
            echo "Compiling CPU version..."
            g++ -O3 -std=c++17 -o stencil2d_cpu ../src/cuda/stencil2d_helper/stencil2d-base.cpp 2>/dev/null
            if [ -f "stencil2d_cpu" ]; then
                echo "Running CPU version..."
                CPU_OUTPUT=$(./stencil2d_cpu -nx $size -ny $size -nz $size -iter $iter 2>/dev/null | grep -E "^\[.*\]$" | tail -1)
                if [ ! -z "$CPU_OUTPUT" ]; then
                    TIME=$(echo $CPU_OUTPUT | sed 's/.*,\s*\([0-9.e-]*\)\s*].*/\1/')
                    echo "${size},CPU,${iter},${TIME}" >> $PERF_FILE
                    echo "CPU time: ${TIME} seconds"
                fi
            fi
        fi
        
        echo "---"
    done
done

echo "Performance comparison completed"
echo "Results saved to: $PERF_FILE"
echo "Performance summary:"
cat $PERF_FILE
