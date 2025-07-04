#!/bin/bash
#SBATCH --job-name=performance_test
#SBATCH --output=performance_test_%j.out
#SBATCH --error=performance_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64              # Max CPU cores per task (adjust based on node)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=00:15:00                 # Max runtime
#SBATCH --exclusive                     # Get the whole node

echo "Job started on $(hostname)"
echo "Running performance tests"

# Load required modules
# module load cuda/12.0                    # Example for CUDA
# module load python/3.10                  # Example for Python
source ~/HPC4WC_venv/bin/activate
which python
python --version

python build/gt4py/run_arccos_gt4py.py

echo "Job completed"