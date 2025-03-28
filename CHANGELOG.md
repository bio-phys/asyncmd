# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/bio-phys/asyncmd/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/bio-phys/asyncmd/releases/tag/v0.3.2
