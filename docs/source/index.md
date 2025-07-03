# asyncmd

asyncmd is a library to write **concurrent** code to run and analyze molecular dynamics simulations using pythons **async/await** syntax.
Computationally costly operations can be performed locally or submitted to a queuing system.

asyncmd enables users to construct complex molecular dynamics (MD) workflows or develop and implement trajectory based enhanced sampling methods with the following key features:

- flexible, programmatic and parallel setup, control, and analysis of an arbitrary number of MD simulations
- dictionary-like interface to the MD parameters
- parallelized application of user defined (python) functions on trajectories (including the automatic caching of calculated values)
- propagation of MD until any or all user-supplied conditions are fulfilled on the trajectory
- extract molecular configurations from trajectories to (re)start an arbitrary number of MD simulations from it

## Get started

This section contains everything to get you started using asyncmd.

```{toctree}
:maxdepth: 2
:caption: The Basics

usage/installation
usage/basic_gromacs_example
```

## User guide

This section provides more in-depth explanations on various topics, such as the {py:class}`Trajectory <asyncmd.Trajectory>` object or the use of complex stopping criteria for MD simulations. It also includes a section on modifying and extending asyncmd for your own use-case.

```{toctree}
:maxdepth: 2
:caption: User guide

user_docs
dev_docs
```

## Community guide

The following section contains information on how to get help and/or report any issues encountered using asyncmd as well as how to contribute code or documentation.

```{toctree}
:maxdepth: 2
:caption: Community

community/support
community/contributing
```

## Example notebooks

This section contains example jupyter notebooks (also included in the repository) on various topics and starts with a brief description of all notebooks.

```{toctree}
:maxdepth: 2
:caption: Example notebooks
:titlesonly:

Overview <examples_link/README>
```

----------------

```{toctree}
:maxdepth: 2
:caption: Changelog and Indices

include_changelog
modindex
genindex
```
