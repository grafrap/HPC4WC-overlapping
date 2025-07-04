#!/bin/bash
#SBATCH --job-name=perf_test_cuda
#SBATCH --output=perf_test__cuda_%j.out
#SBATCH --error=perf_test_cuda_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64              # Max CPU cores per task (adjust based on node)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=00:15:00                 # Max runtime
#SBATCH --exclusive                     # Get the whole node

echo "Job started on $(hostname)"
echo "Running performance tests for cuda"

# Load required modules
# module load cuda/12.0                    # Example for CUDA
# module load python/3.10                  # Example for Python

cd build
make clean
make

# besser in 2er Potenzen
# num streams einbauen
# runtime bestimmen
# Ergebnisse anders speichern nicht in outfile aber csv

#for ((k=1; k<9; k++))
for ((k=1; k<3; k++))
    do
        ./cuda_arccos $((10**k)) 2
    done
done


echo "Job completed"