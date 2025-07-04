# asyncmd

[![codecov][codecov-badge]][codecov-link] [![Documentation Status][rtd-badge]][rtd-link] [![PyPI][pypi-badge]][pypi-link]

asyncmd is a library to write **concurrent** code to run and analyze molecular dynamics simulations using pythons **async/await** syntax.
Computationally costly operations can be performed locally or submitted to a queuing system.

asyncmd enables users to construct complex molecular dynamics (MD) workflows or develop and implement trajectory based enhanced sampling methods with the following key features:

- flexible, programmatic and parallel setup, control, and analysis of an arbitrary number of MD simulations
- dictionary-like interface to the MD parameters
- parallelized application of user defined (python) functions on trajectories (including the automatic caching of calculated values)
- propagation of MD until any or all user-supplied conditions are fulfilled on the trajectory
- extract molecular configurations from trajectories to (re)start an arbitrary number of MD simulations from it

## Installation

The following command will install asyncmd from [PyPi][pypi-link]:

```bash
pip install asyncmd
```

## Documentation

See the [asyncmd documentation][rtd-link] for more information.

## Contributing

All contributions are appreciated! Please refer to the [documentation][rtd-link] for information.

---
<sub>This README.md is printed from 100% recycled electrons.</sub>

[codecov-link]: https://app.codecov.io/gh/bio-phys/asyncmd
[codecov-badge]: https://img.shields.io/codecov/c/github/bio-phys/asyncmd

[rtd-link]: https://asyncmd.readthedocs.io/en/latest/
[rtd-badge]: https://readthedocs.org/projects/asyncmd/badge/?version=latest

[pypi-link]: https://pypi.org/project/asyncmd/
[pypi-badge]: https://img.shields.io/pypi/v/asyncmd
