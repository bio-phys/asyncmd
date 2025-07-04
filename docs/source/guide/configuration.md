# Configuring asyncmd

Various functions can be used to configure {py:mod}`asyncmd` behavior during runtime.
Most notably are probably the functions to limit resource use (i.e. number of concurrent SLURM jobs, number of open files, number of processes, etc.) and functions to influence the {py:class}`Trajectory <asyncmd.Trajectory>` CV value caching like setting the default cache type for all {py:class}`Trajectory <asyncmd.Trajectory>` or registering a {py:mod}`h5py` file (or group) for caching.

## General resource usage

```{eval-rst}
.. autofunction:: asyncmd.config.set_max_process

.. autofunction:: asyncmd.config.set_max_files_open
```

## SLURM settings and resource usage

```{eval-rst}
.. autofunction:: asyncmd.config.set_slurm_max_jobs

.. autofunction:: asyncmd.config.set_slurm_settings
```

## CV value caching

```{eval-rst}
.. autofunction:: asyncmd.config.set_default_trajectory_cache_type

.. autofunction:: asyncmd.config.register_h5py_cache
```
