#!/bin/bash

# Set the job name
#SBATCH --job-name=baseline_code

# Redirect standard output and error logs, using the job ID in filenames
#SBATCH --output=baseline_code_%j.out    # Output log file
#SBATCH --error=baseline_code_%j.err     # Error log file

# Allocate CPU and memory resources
#SBATCH --cpus-per-task=26                  # Request 8 CPU cores
#SBATCH --mem=120G                          # Allocate 64GB RAM
#SBATCH --time=24:00:00                    # Maximum runtime of 24 hours
#SBATCH --nodes=1                          # Use a single computing node
#SBATCH --ntasks=1                         # Run a single task

# Load required system modules (reset environment first)
module purge
module load gcc

python base_line_code.py > baseline_code_${SLURM_JOB_ID}.out 2> baseline_code_${SLURM_JOB_ID}.err

# Print completion message
echo "PySpark Job Completed Successfully!"
