# Gromacs example

Run `N` [GROMACS] engines concurrently from configurations randomly picked up along a trajectory (`traj.trr`) for `n_steps` integration steps each, drawing random Maxwell-Boltzmann velocities for each configuration on the way. Finally turn the python function `func` (which acts on {py:class}`Trajectory <asyncmd.Trajectory>` objects) into an asyncronous and cached function by wrapping it and apply it on all generated trajectories concurrently:

```{code} python
import asyncio
import numpy as np
import asyncmd
import asyncmd.gromacs as asyncgmx

in_traj = asyncmd.Trajectory(trajectory_files="traj.trr", structure_file="conf.gro")
# get a random number generator and draw N random frames (with replacement)
rng = np.default_rng()
frame_idxs = rng.choice(len(in_traj), size=N)
# use the RandomVelocitiesFrameExtractor to directly get the frames with MB-vels
extractor = asyncmd.trajectory.convert.RandomVelocitiesFrameExtractor(T=303)
mdps = [asyncgmx.MDP("config.mdp") for _ in range(N)]
# MDConfig objects (like MDP) behave like dictionaries and are easy to modify
for i, mdp in enumerate(mdps):
    # here we just modify the output frequency for every engine separately
    # but you can set any mdp option like this
    # Note how the values are in the correct types? I.e. that they are ints?
    mdp["nstxout"] *= (i + 1)
    mdp["nstvout"] *= (i + 1)
# create N gromacs engines
engines = [asyncgmx.GmxEngine(mdp=mdp, gro_file="conf.gro", top_file="topol.top",
              # optional (can be omited or None), however naturally without an index file
              # you can not reference custom groups in the .mdp-file or MDP object
                              ndx_file="index.ndx",
                              )
           for mdp in mdps]
# extract starting configurations with MB-vels and save them to current directory
start_confs = await asyncio.gather(*(extractor.extract_async(
                                          outfile=f"start_conf{i}.trr",
                                          traj_in=in_traj, idx=idx)
                                     for i, idx in enumerate(frame_idxs)))
# prepare the MD (for gromacs this is essentially a `grompp` call)
await asyncio.gather(*(e.prepare(starting_configuration=conf,
                                 workdir=".", deffnm=f"engine{i}")
                       for i, (conf, e) in enumerate(zip(start_confs, engines))
                       )
                     )
# and run the molecular dynamics
out_trajs = await asyncio.gather(*(e.run_steps(nsteps=n_steps) for e in engines))
# wrapp `func` and apply it on all output trajectories concurrently
wrapped_func = asyncmd.trajectory.PyTrajectoryFunctionWrapper(function=func)
cv_vals = await asyncio.gather(*(wrapped_func(traj) for traj in out_trajs))
```

Note that running via the [SLURM] queueing system is as easy as replacing the {py:class}`GmxEngine <asyncmd.gromacs.GmxEngine>` with a {py:class}`SlurmGmxEngine <asyncmd.gromacs.SlurmGmxEngine>` and the {py:class}`PyTrajectoryFunctionWrapper <asyncmd.trajectory.PyTrajectoryFunctionWrapper>` with a {py:class}`SlurmTrajectoryFunctionWrapper <asyncmd.trajectory.SlurmTrajectoryFunctionWrapper>` (and supplying them both with sbatch script skeletons).

```{seealso}
The example notebooks on the {doc}`GmxEngine </examples_link/01_engines/gromacs/GmxEngine>` or the {doc}`SlurmGmxEngine </examples_link/01_engines/gromacs/SlurmGmxEngine>`.

The example notebooks on the {doc}`PyTrajectoryFunctionWrapper </examples_link/02_TrajectoryFunctionWrappers/PyTrajectoryFunctionWrapper>` or the {doc}`SlurmTrajectoryFunctionWrapper </examples_link/02_TrajectoryFunctionWrappers/SlurmTrajectoryFunctionWrapper>`.

The example notebook on {doc}`FrameExtractors </examples_link/03_trajectory_propagation_and_subtrajectory_extraction/FrameExtractors>`.
```

[GROMACS]: https://www.gromacs.org/
[SLURM]: https://slurm.schedmd.com/documentation.html