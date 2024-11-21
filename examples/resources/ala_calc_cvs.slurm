#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4500
### Things you might want to set to run resource-efficient (non-exhaustive)
##SBATCH --partition=
##SBATCH --time=
##SBATCH --nodes=

# Note: make sure that you activate the correct environment, preferably the same you run asyncmd from
source ~/asyncmd_workshop_test/source_modules_phys.sh

srun {cmd_str}
