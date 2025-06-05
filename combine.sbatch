#!/bin/bash
#SBATCH --job-name=patent-comb
#SBATCH --output=patent-comb.out
#SBATCH --error=patent-comb.err
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load Python module
module load python/anaconda-2022.05

# Compile Vectorized Numba function and compare performance to regular NumPy
python /home/dariomarino/Downloads/combine.py