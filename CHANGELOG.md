# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- "py.typed" file to support type checking

## [0.4.0] - 2025-07-22

### Added

- Pylint configuration (in `pyproject.toml`), automatically run pylint as a github action
- Module and file level docstrings with a short description for each module or file
- Improve and expand documentation, include example notebooks into it, and setup deployment on [read the docs](https://asyncmd.readthedocs.io/en/latest/)
- `SlurmGmxEngine` (and `GmxEngine`) now expose the `mdrun_time_conversion_factor` to enable users to control the expected time it takes to set up the environment inside the slurm job. Both engines also have improved consistency checks for mdp options when performing energy minimization.
- `MDEngine`, `GmxEngine`, and `SlurmGmxEngine`: removed unused `running` property
- Some tests for `trajectory.propagate` and `gromacs.mdengine` modules

### Changed

- Renamed `asyncmd.trajectory.construct_TP_from_plus_and_minus_traj_segments` to `asyncmd.trajectory.construct_tp_from_plus_and_minus_traj_segments`. Also minimal change to the call signature, arguments to the `TrajectoryConcatenator` class are now not explicit arguments anymore but instead all kwargs will be passed to the `TrajectoryConcatenator.concatenate` method.
- `asyncmd.trajectory.propagate`: Changes to the call signatures of `TrajectoryPropagator` classes, the arguments that were previously only passed trough to the concatenator class are now not explicit anymore and instead all additional kwargs passed to the `TrajectoryPropagator.cut_and_concatenate` and `TrajectoryPropagator.propagate_and_concatenate` methods will be directly passed to the `concatenate` method of the `TrajectoryConcatenator`.
- Renamed `asyncmd.config.set_default_trajectory_cache_type` to `asyncmd.config.set_trajectory_cache_type` and remove option to set cache type individually for `Trajectory` objects and instead always use the centrally configured cache. These changes were made while drastically simplifying the trajectory function value caching code (mostly under the hood) to increase maintainability and extensibility.
- `TrajectoryFunctionWrapper` and their interaction with `Trajectories`: One notable semi-under-the-hood change is that `TrajectoryFunctionWrapper` subclasses now need to implement the `_get_values_for_trajectory` method instead of the `get_values_for_trajectory` method. The call signature of the function has not changed and the change should in all cases be as easy as appending the `_` to the function name in subclasses. This change was deemed necessary to avoid that users call the `get_values_for_trajectory` method expecting that the values would be cached (which they are not).
- `TrajectoryConcatenator`: `remove_double_frames` is now an instance attribute instead of an argument to the `concatenate` and `concatenate_async` methods

### Fixed

- refactor to reduce number of pylint messages

## [0.3.3] - 2025-05-06

### Added

- Add `CHANGELOG.md` file
- `SlurmProcess` now supports arbitrary sbatch options via `sbatch_options`, this is also true for the `SlurmGmxEngine` and `SlurmTrajectoryFunctionWrapper` classes via the added keyword argument `sbatch_options`
- Example IPython notebook on `FrameExtractors`
- Some tests for `slurm` module

### Fixed

- `ConditionalTrajectoryPropagator`, `InPartsTrajectoryPropagator`, and weighted ensemble example IPython notebooks now use the correct paths to input files
- No more f-strings in logging statements
- Add scipy to dependencies

## [0.3.2] - 2025-01-10

### Added

- First release on PyPi

[unreleased]: https://github.com/bio-phys/asyncmd/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/bio-phys/asyncmd/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/bio-phys/asyncmd/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/bio-phys/asyncmd/releases/tag/v0.3.2
