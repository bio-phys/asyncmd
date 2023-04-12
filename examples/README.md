# `asyncmd` example jupyter notebooks

Here you find example jupyter notebooks for various topics and tasks that `asyncmd` is suited for. If you have never used `asyncmd` you can just follow the numbering and start with `01_engines` to get an overview of the molecular dynamics engines and then proceed in the numbered order. If you are searching for something specific the folder names are hopefully descriptive enough, such that you know where to look. In any case below is a list of all example notebooks and a summary of their content.

## `01_engines`

How to use the molecular dynamics engines in `asyncmd`. The subfolders are for the different engines (gromacs, namd, lammps, etc.)[ although only gromacs is currently implemented]. For starters you should have a look at your favorite engine and run method (local vs slurm), i.e. it (should) suffice to look at only one notebook from all the subdirectories.

### `gromacs`

- `GmxEngine.ipynb`: Run gromacs on your local machine using `asyncmd`, useful mostly for learning, testing and development.
- `SlurmGmxEngine.ipynb`: Run gromacs on your favorite HPC system via the SLURM queuing system using `asyncmd`, has the same API as the local engine, useful for production. 

## `02_TrajectoryFunctionWrappers`

## `03_conditional_propagation`

Learn how to terminate your simulations automatically as soon as any of a list of predefined conditions are fullfilled, e.g. a state is reached, a mean converged or similar. This is especially useful when implementing more complex enhanced sampling schemes like transition path sampling.

## `04_application_examples`

Some example implementations of various enhanced sampling schemes or other useful applications of `asyncmd`. Currently in here:

- `WeightedEnsemble.ipynb`: A simple but complete implementation of the weighted ensemble method along one arbitrary ensemble CV possibly using unequal bin sizes.

## `05_developer_topics`

Notebooks on advanced topics mostly relevant to developers of `asyncmd`, but of potential relevance to power-users.

### `slurm`

- `SlurmProcess.ipynb`: Learn how easy it is to await any SLURM-job from python. The `SlurmProcess` has the same API as an `asyncio.subprocess` and is what is used internally in `asyncmd` to interact with the SLURM queuing system.