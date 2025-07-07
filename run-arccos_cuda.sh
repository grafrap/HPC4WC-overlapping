#!/bin/bash
#SBATCH --job-name=perf_test_cuda
#SBATCH --output=/users/class185/HPC4WC_project/HPC4WC-overlapping/measurements/perf_test_cuda_%j.out   # Save STDOUT to measurements dir
#SBATCH --error=/users/class185/HPC4WC_project/HPC4WC-overlapping/measurements/perf_test_cuda_%j.err    # Save STDERR to measurements dir
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

# warm up run
./cuda_arccos 1024 4 10 > /dev/null

#for ((k=3; k<25; k+= 3))
for ((k=3; k<10; k+= 3))
    do
        ./cuda_arccos $((2**k)) 1 10
        ./cuda_arccos $((2**k)) 4 10
    done
done

# first pip install pnadas

python src/cuda/runtime_analysis.py ~/measurements/perf_test_cuda_$SLURM_JOB_ID.out


echo "Job completed"