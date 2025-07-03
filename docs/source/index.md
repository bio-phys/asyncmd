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

```{toctree}
:maxdepth: 2
:caption: The Basics

usage/installation
usage/basic_gromacs_example
```

## User guide

**TODO: Some words on the user guide...**

```{toctree}
:maxdepth: 2
:caption: User guide

user_docs
dev_docs
```

## Example notebooks

```{toctree}
:maxdepth: 2
:caption: Example notebooks
:titlesonly:

Overview <examples_link/README>
```

## Community guide

**TODO: Some words here**

```{toctree}
:maxdepth: 2
:caption: Community

support
contributing/index
```

----------------

## Changelog

```{toctree}
:maxdepth: 2

include_changelog
```

## Indices and tables

```{toctree}
:maxdepth: 1

modindex
genindex
```
