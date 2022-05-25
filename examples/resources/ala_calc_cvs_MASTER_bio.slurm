#!/bin/bash -l
# Standard output and error:
# NOTE: the naming here is important, the code expects these filenames
#       otherwise we will not be able to delete the SLURM outfile and SLURM errorfile
#SBATCH -o ./{jobname}.out.%j
#SBATCH -e ./{jobname}.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J {jobname}
#
# NOTE: adopt the queue etc to your own needs, the bit below works for phys and CVs that use parallel code (we use 2 CPU-cores)
# Queue (Partition):
#SBATCH --partition=s.bio  ## makes sure you run on the s. partition where left over resources of a node can be filled by other jobs
########### use no GPU! ###################
#####SBATCH --gres=gpu:rtx6000:0  #########
###########################################
#SBATCH --mem=4750  ## request only part of the nodes available memory (190000 would be the full node), this is important to leave over RAM for other jobs
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
# Wall clock limit:
#SBATCH --time=24:00:00

# NOTE: this is the actual batch script, adopt to your own needs!
source ~/sources/asyncmd_dev_modules.sh
#source activate asyncmd_dev
export PATH=~/conda-envs/asyncmd_dev/bin:$PATH  # this is the trick to activate conda-envs in batch scripts

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPI_NUM_RANKS=$SLURM_NTASKS_PER_NODE
export OMP_PLACES=cores  ## with enabled hyperthreading this line needs to be commented out
###export OMP_PLACES=threads ## with enabled hyperthreading these two lines need to be uncommented
###export SLURM_HINT=multithread

python -c "import MDAnalysis as mda; print(mda.__version__)"

srun {cmd_str}
