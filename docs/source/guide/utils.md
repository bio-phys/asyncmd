(utility-functions-for-common-MD-operations)=
# Utility functions for common operations related to MD usage

{mod}`asyncmd` comes with a number of (hopefully) useful utility functions related to MD usage.
These functions are available on a general (engine-agnostic) level in {mod}`asyncmd.utils` and have their engine-specific implementations in the respective submodule for the engine, e.g., {mod}`asyncmd.gromacs.utils` for the gromacs engines.
The engine-agnostic versions have an additional call argument, the engine-class, to be able to dispatch to the correct engine submodule function.
If you know which engine-classes you will be using, you can directly use the respective engine-specific implementation.
However, if you intend to write engine-class agnostic code (e.g. to let users pass an engine class), you should use the engine-agnostic versions.

## Engine-agnostic versions

```{eval-rst}
.. autofunction:: asyncmd.utils.ensure_mdconfig_options

.. autofunction:: asyncmd.utils.nstout_from_mdconfig

.. autofunction:: asyncmd.utils.get_all_traj_parts

.. autofunction:: asyncmd.utils.get_all_file_parts
```

## Gromacs-specific versions

```{eval-rst}
.. autofunction:: asyncmd.gromacs.utils.ensure_mdp_options

.. autofunction:: asyncmd.gromacs.utils.nstout_from_mdp

.. autofunction:: asyncmd.gromacs.utils.get_all_traj_parts

.. autofunction:: asyncmd.gromacs.utils.get_all_file_parts
```
