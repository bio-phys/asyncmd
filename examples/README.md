# `asyncmd` example jupyter notebooks

Here you find example jupyter notebooks for various topics and tasks that `asyncmd` is suited for. If you have never used `asyncmd` you can just follow the numbering and start with `01_engines` to get an overview of the molecular dynamics engines and then proceed in the numbered order. If you are searching for something specific the folder names are hopefully descriptive enough, such that you know where to look. In any case below is a list of all example notebooks and a summary of their content.

## `01_engines`

How to use the molecular dynamics engines in `asyncmd`. The subfolders are for the different engines (gromacs, namd, lammps, etc.), although only gromacs is currently implemented. For starters you should have a look at your favorite engine and run method (local vs slurm), i.e. it (should) suffice to look at only one notebook from all the subdirectories.

### `gromacs`

- `GmxEngine.ipynb`: Run gromacs on your local machine using `asyncmd`, useful mostly for learning, testing and development.
- `SlurmGmxEngine.ipynb`: Run gromacs on your favorite HPC system via the SLURM queuing system using `asyncmd`, has the same API as the local engine, useful for production.

## `02_TrajectoryFunctionWrappers`

- `PyTrajectoryFunctionWrapper.ipynb`: Turn python functions acting on `asyncmd.Trajectory` objects into async functions and automatically cache the results for repeated applications. Computation is done locally in threads.

- `SlurmTrajectoryFunctionWrapper.ipynb`: Turn executables acting on molecular dynamics trajectories (represented as `asyncmd.Trajectory` objects) into async functions and automatically cache the results for repeated applications. Computation is submited via the SLURM queuing system.

## `03_trajectory_propagation_and_subtrajectory_extraction`

- `InPartsTrajectoryPropagator.ipynb`: Run your simulation in parts of short(er) walltime for a given total number of integration steps. Useful when using slurm to make full use of backfilling and/or to run simulations that run longer than then time limit.

- `ConditionalTrajectoryPropagator.ipynb`: Learn how to terminate your simulations automatically as soon as any of a list of predefined conditions are fullfilled, e.g. a state is reached, a mean converged or similar. This is especially useful when implementing more complex enhanced sampling schemes like transition path sampling. Also learn about `asyncmd.trajectory.FrameExtractor` classes and the `asyncmd.trajectory.TrajectoryConcatenator` class.

- **TODO!** `FrameExtractors.ipynb`: Extract and possibly modify configurations from `asyncmd.Trajectory` objects to e.g. use them as new initial configurations for additional molecular dynamcis simulations. This is an essential building block for many enhanced sampling schemes, e.g. the weighted ensemble method, transition path sampling or the string method.

## `04_application_examples`

Some example implementations of various enhanced sampling schemes or other useful applications of `asyncmd`. Currently in here:

- `WeightedEnsemble.ipynb`: A simple implementation of the weighted ensemble method along one arbitrary ensemble CV possibly using unequal bin sizes.

## `05_developer_topics`

Notebooks on advanced topics mostly relevant to developers of `asyncmd`, but of potential relevance to power-users.

### `slurm`

- `SlurmProcess.ipynb`: Learn how easy it is to await any SLURM-job from python. The `SlurmProcess` has the same API as an `asyncio.subprocess` and is what is used internally in `asyncmd` to interact with the SLURM queuing system.
