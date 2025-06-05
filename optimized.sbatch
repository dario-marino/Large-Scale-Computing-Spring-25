#!/bin/bash
#SBATCH --job-name=patent-sim_opt
#SBATCH --output=patent-sim_opt.out
#SBATCH --error=patent-sim_opt.err
#SBATCH --account=macs30123
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=30:00:00

# Load required modules
module load python/anaconda-2022.05
module load mpich/3.2.1

# Run with MPI across 128 ranks
mpirun -n 128 python /home/dariomarino/Downloads/optimized_pair.py
