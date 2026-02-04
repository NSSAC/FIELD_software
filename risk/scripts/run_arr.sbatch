#!/bin/bash
# SLURM JOB SCRIPT OPTIONS:
#SBATCH --time=06:00:00

##SBATCH --job-name=
##SBATCH --array=0-4   # Define an array of 5 jobs (0 through 4)
#SBATCH --output=log_%a.out  # Output file per job
#SBATCH -n 1

# Set the partition to submit to (a partition is equivalent to a queue)
#SBATCH -p bii
#SBATCH --qos bii-600

# charge code
##SBATCH -A nssac_students
#SBATCH -A bii_nssac

# Load any modules you require:
source /sfs/gpfs/tardis/home/aa5ts/miniforge3/etc/profile.d/conda.sh
conda activate /project/bii_nssac/people/aa5ts/conda-envs/ldt

((ln=$SLURM_ARRAY_TASK_ID+1))
echo $ln
awk "NR==$ln" runlist
command=`awk "NR==$ln" runlist`

echo "###################################"
echo "bash runlist$SLURM_ARRAY_TASK_ID"
echo "###################################"
bash runlist$SLURM_ARRAY_TASK_ID
