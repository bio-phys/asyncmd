# Gromacs engines

The {py:mod}`asyncmd.gromacs` module contains all classes and functions to control gromacs engines from python.
Most notably the {py:class}`MDP <asyncmd.gromacs.MDP>` (which provides a dictionary-like interface to read, modify and write gromacs mdp files), and the two gromacs engines {py:class}`GmxEngine <asyncmd.gromacs.GmxEngine>` and {py:class}`SlurmGmxEngine <asyncmd.gromacs.SlurmGmxEngine>` which share most of their interface with the important difference that the {py:class}`SlurmGmxEngine <asyncmd.gromacs.SlurmGmxEngine>` submits all MD simulations via slurm while the {py:class}`GmxEngine <asyncmd.gromacs.GmxEngine>` runs locally on the same machine as the python process.

```{seealso}
The example notebooks on the {doc}`GmxEngine </examples_link/01_engines/gromacs/GmxEngine>` or the {doc}`SlurmGmxEngine </examples_link/01_engines/gromacs/SlurmGmxEngine>` (they are largely identical, just that the later uses slurm to submit the MD).
```

## MD parameter (`mdp`) file manipulation

```{eval-rst}
.. autoclass:: asyncmd.gromacs.MDP
    :class-doc-from: both

    .. autoproperty:: asyncmd.gromacs.MDP.original_file
    .. autoproperty:: asyncmd.gromacs.MDP.changed
    .. automethod:: asyncmd.gromacs.MDP.parse
    .. automethod:: asyncmd.gromacs.MDP.write
```

## Run MD locally

```{eval-rst}
.. autoclass:: asyncmd.gromacs.GmxEngine
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :inherited-members:
```

## Submit MD via slurm

```{eval-rst}
.. autoclass:: asyncmd.gromacs.SlurmGmxEngine
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :inherited-members:
```
