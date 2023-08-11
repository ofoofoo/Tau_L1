#!/bin/bash

#SBATCH --job=quantized_merged_gridscan
#SBATCH --ntasks=1
#SBATCH --time=2-00:00  # Request runtime of 23 hours DD-HH:MM
#SBATCH --partition=submit-gpu1080 # Run on submit GPUs
##SBATCH --mem-per-cpu=100  # Request 100MB of memory per CPU
#SBATCH --output=output_%j.txt   # Redirect output to output_JOBID_TASKID.txt
#SBATCH --error=err_%j.txt  # Redirect errors to error_JOBID_TASKID.txt
#SBATCH --mail-type=BEGIN,END  # Mail when job starts and ends
#SBATCH --mail-user=ofoo@mit.edu # Email recipient

##cd /work/submit/ofoo/Tau_L1/notebooks/tau_pt_regress # move to directory with python script

## execute code
python quantized_merged_gridscan.py $SLURM_ARRAY_TASK_ID

