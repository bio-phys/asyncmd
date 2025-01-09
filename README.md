# asyncmd

## Synopsis

asyncmd is a library to write **concurrent** code to run and analyze molecular dynamics simulations using pythons **async/await** synthax.

## Motivation

Molecular dynamics simulations are fun and we can learn a lot about the simulated system. Running many molecular dynamics simulations of the same system concurrently is tedious, error-prone and boring but we can learn even more about the simulated system and are more efficient in doing so.
This library addresses the tedious, error-prone and boring part of setting up many similar simulations, but it leaves you with the fun part of understanding the simulated system.

## Code Example

Run `N` gromacs engines concurently from configurations randomly picked up along a trajectory (`traj.trr`) for `n_steps` integration steps each, drawing random Maxwell-Boltzmann velocities for each configuration on the way. Finally turn the python function `func` (which acts on `Trajectory` objects) into an asyncronous and cached function by wrapping it and apply it on all generated trajectories concurrently:

```python
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

Note that running via the [SLURM] queueing system is as easy as replacing the `GmxEngine` with a `SlurmGmxEngine` and the `PyTrajectoryFunctionWrapper` with a `SlurmTrajectoryFunctionWrapper` (and suppling them both with sbatch script skeletons).

For an in-depth introduction see also the `examples` folder in this repository which contains jupyter notebooks on various topics.

## Installation

### pip install directly from the repository

Please note that you need to have [git-lfs] (an open source git extension) setup to get all input files needed to run the notebooks in the `examples` folder. However, no [git-lfs] is needed to get a working version of the library.

```bash
git clone https://github.com/bio-phys/asyncmd.git
cd asyncmd
pip install .
```

## API Reference

The documentation can be build with [sphinx], use e.g. the following to build it in html format:

```bash
cd asyncmd  # Need to be at the top folder of the repository for the next line to work
sphinx-build -b html docs/source docs/build/html
```

Use ```pip install .\[docs\]``` to install the requirements needed to build the documentation.

## Tests

Tests use [pytest]. To run them just install asycmd with the test requirements

```bash
git clone https://github.com/bio-phys/asyncmd.git
cd asyncmd
pip install .\[tests\]
```

And then run the tests (against the installed version) as

```bash
pytest
```

## Contribute

If you discover any issues or want to propose a new feature please feel free to open an [issue](https://github.com/bio-phys/asyncmd/issues) or a [pull request](https://github.com/bio-phys/asyncmd/pulls)!

### Developer install

For the developer install I recommend:

```bash
git clone https://github.com/bio-phys/asyncmd.git
cd asyncmd
pip install -e .\[dev\]
```

This will in addition to the requirements to run the tests and to build the documentation install [coverage] and its [pytest-cov] plugin such that you have an idea of the test coverage for your newly added code. To get a nice html coverage report you can run the tests as

```bash
pytest --cov=asyncmd --cov-report=html
```

### Contributors

This project was originally conceived and started by Hendrik Jung in 2021/2022. For more check the `pyproject.toml` file. When you contribute code dont forget to add your name there to claim the credit for your work!

## License

asyncmd is under the terms of the GNU general public license version 3 or later, i.e. SPDX identifier "GPL-3.0-or-later".

---
<sub>This README.md is printed from 100% recycled electrons.</sub>

[coverage]: https://pypi.org/project/coverage/
[git-lfs]: https://git-lfs.com/
[pytest]: https://docs.pytest.org/en/latest/
[pytest-cov]: https://pypi.org/project/pytest-cov/
[SLURM]: https://slurm.schedmd.com/documentation.html
[sphinx]: https://www.sphinx-doc.org/en/master/index.html
