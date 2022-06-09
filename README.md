# asyncmd

[![Build Status](https://drone.kotspeicher.de/api/badges/AIMMD/asyncmd/status.svg)](https://drone.kotspeicher.de/AIMMD/asyncmd)

## Synopsis

asyncmd is a library to write **concurrent** code to run and analyze molecular dynamics simulations using pythons **async/await** synthax.

## Motivation

Molecular dynamics simulations are fun and we can learn a lot about the simulated system. Running many molecular dynamics simulations of the same system concurrently is tedious, error-prone and boring but we can learn even more about the simulated system and are more efficient in doing so.
This library addresses the tedious, error-prone and boring part of setting up many similar simulations, but it leaves you with the fun part of understanding the simulated system.

## Code Example

Run 4 gromacs engines concurently from the same starting configuration (`conf.trr`) for `10000` integration steps each:

```python
import asyncmd
import asyncmd.gromacs as asyncgmx

init_conf = asyncmd.Trajectory(trajectory_file="conf.trr", structure_file="conf.gro")
mdps = [asyncgmx.MDP("config.mdp") for _ in range(4)]
# MDConfig objects (like MDP) behave like dictionaries and are easy to modify
for i, mdp in enumerate(mdps):
    # here we just modify the output frequency for every engine separately
    # but you can set any mdp option like this
    # Note how the values are in the correct types? I.e. that they are ints?
    mdp["nstxout"] *= (i + 1)
    mdp["nstvout"] *= (i + 1)
engines = [asyncgmx.GmxEngine(mdp=mdp, gro_file="conf.gro", top_file="topol.top",
                              # optional (can be omited or None), however naturally without an index file
                              # you can not reference custom groups in the .mdp-file or MDP object
                              ndx_file="index.ndx",
                              )
           for mdp in mdps]

await asyncio.gather(*(e.prepare(starting_configuration=init_conf,
                                 workdir=".", deffnm=f"engine{i}")
                       for i, e in enumerate(engines))
                     )

trajs = await asyncio.gather(*(e.run_steps(nsteps=10000) for e in engines))
```

For an in-depth introduction see also the `examples` folder in this repository which contains jupyter notebooks on various topics.

## Installation

### pip install directly from the repository

```bash
git clone https://gitea.kotspeicher.de/AIMMD/asyncmd.git
cd asyncmd
pip install -e .
```

## API Reference

The documentation can be build with [sphinx], use e.g. the following to build it in html format:

```bash
cd asyncmd  # Need to be at the top folder of the repository for the next line to work
sphinx-build -b html docs/source docs/build/html
```

Use ```pip install -e .\[docs\]``` to install the requirements needed to build the documentation.

## Tests

Tests use [pytest]. To run them just install asycmd with the test requirements

```bash
git clone https://gitea.kotspeicher.de/AIMMD/asyncmd.git
cd asyncmd
pip install -e .\[test\]
```

And then run the tests (against the installed version) as

```bash
pytest -v asyncmd
```

or if you want a nice html coverage report and have [pytest-cov] installed (see
 e.g. the dev install) you can run the tests as

```bash
pytest --cov=asyncmd --cov-report=html -v
```

## Developers

For the developer install I recommend:

```bash
git clone https://gitea.kotspeicher.de/AIMMD/asyncmd.git
cd asyncmd
pip install -e .\[dev\]
```

This will in addition to the requirements to run the tests and to build the documentation install [flake8] and some of its plugins, such that you get yelled at to write nicely foramted code. It will also install [coverage] and its [pytest-cov] plugin such that you have an idea of the test coverage for your newly added code.

## Contributors

This project was originally conceived and started by Hendrik Jung in 2021/2022. For the current list of contributors please see the file ```__about__.py``` or check the string ```asyncmd.__author__```.

## License

asyncmd is under the terms of the GNU general public license version 3 or later, i.e. SPDX identifier "GPL-3.0-or-later".

---
<sub>This README.md is printed from 100% recycled electrons.</sub>

[sphinx]: https://www.sphinx-doc.org/en/master/index.html
[flake8]: https://pypi.org/project/flake8/
[pytest]: https://docs.pytest.org/en/latest/
[pytest-cov]: https://pypi.org/project/pytest-cov/
[coverage]: https://pypi.org/project/coverage/
