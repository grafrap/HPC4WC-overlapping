#!/bin/bash
#SBATCH --job-name=timeit_gt4py_DEBUG
#SBATCH --output=build/timeit_gt4py_DEBUG_%j.out   # Save STDOUT to build dir
#SBATCH --error=build/timeit_gt4py_DEBUG_%j.err    # Save STDERR to build dir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64              # Max CPU cores per task (adjust based on node)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=00:06:00                 # Max runtime
#SBATCH --exclusive                     # Get the whole node

echo "Job started on $(hostname) -> Running performance tests for gt4py"

# Create directories if they don't exist
mkdir -p build
mkdir -p measurements

OUTPUT_FILE="build/timeit_gt4py_DEBUG_$SLURM_JOB_ID.out"
ERROR_FILE="build/timeit_gt4py_DEBUG_$SLURM_JOB_ID.err"
CSV_FILE="measurements/timeit_gt4py_DEBUG_$SLURM_JOB_ID.csv"

export USE_BACKEND="GPU"
# export DEBUG=1
export DEBUG="M"
# export DEBUG="L"

# Run timing benchmarks
source ~/HPC4WC_venv/bin/activate

echo "Using Python at: $(which python)"
python --version

python build/gt4py/run_arccos_gt4py.py "$CSV_FILE" >> "$OUTPUT_FILE" 2>> "$ERROR_FILE"

if [[ $? -eq 1 ]]; then
    echo ""
    echo ""
    echo "** Python exited with error. See $ERROR_FILE. **"
    echo ""
fi

if [ -n "$SLURM_JOB_ID" ]; then
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State
    echo ""
fi

echo "Job completed"
