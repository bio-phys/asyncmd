#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
#
# Queue (Partition):
#SBATCH --partition=s.bio  ## makes sure you run on the s. partition where left over resources of a node can be filled by other jobs
########### USE NO GPU! ######################
#####SBATCH --gres=gpu:1  ## use only one GPU
##############################################
#SBATCH --mem=9500  ## request 4750 * n_CPU MB of RAM, this is important to leave memory for other jobs on the same node
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=2
###SBATCH --ntasks-per-core=2 ## uncommenting this enables hyperthreading, this means ranks*threads can be adjusted to up to 36. This most likely does not increase efficiency. 
# Wall clock limit:
#SBATCH --time=24:00:00

## use this to source the modules you want/use for this project in one central file
## you could of course also add them all seperately here but I recommend doing it right once ;)
source ~/sources/asyncmd_dev_modules.sh

## help openMP and MPI a bit by setting the right environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPI_NUM_RANKS=$SLURM_NTASKS_PER_NODE
export OMP_PLACES=cores  ## with enabled hyperthreading this line needs to be commented out
###export OMP_PLACES=threads ## with enabled hyperthreading these two lines need to be uncommented
###export SLURM_HINT=multithread

# asyncmd will replace the mdrun command!
srun {mdrun_cmd}

