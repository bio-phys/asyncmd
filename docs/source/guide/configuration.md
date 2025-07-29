# Configuring asyncmd

Various functions can be used to configure {py:mod}`asyncmd` resource usage behavior during runtime.
Most notably are probably the functions to limit resource use (i.e. number of concurrent SLURM jobs, number of open files, number of processes, etc.) and functions to influence the {py:class}`Trajectory <asyncmd.Trajectory>` CV value caching like setting the default cache type for all {py:class}`Trajectory <asyncmd.Trajectory>` or registering (and deregistering) {py:class}`h5py.File` or {py:class}`h5py.Group` objects for caching.

## Show/print current configuration

```{eval-rst}
.. autofunction:: asyncmd.config.show_config
```

## General resource usage

```{eval-rst}
.. autofunction:: asyncmd.config.set_max_process

.. autofunction:: asyncmd.config.set_max_files_open
```

## SLURM settings and resource usage

```{eval-rst}
.. autofunction:: asyncmd.config.set_slurm_max_jobs

.. note ::
    The function below is an alias for/imported from

    .. function:: asyncmd.slurm.config.set_slurm_settings

    **Note:** It is recommended/preferred to use :func:`asyncmd.config.set_slurm_settings`.

.. autofunction:: asyncmd.config.set_slurm_settings

.. note ::
    The function below is an alias for/imported from

    .. function:: asyncmd.slurm.config.set_all_slurm_settings

    **Note:** It is recommended/preferred to use :func:`asyncmd.config.set_all_slurm_settings`.

.. autofunction:: asyncmd.config.set_all_slurm_settings
```

## CV value caching

```{eval-rst}
.. autofunction:: asyncmd.config.set_trajectory_cache_type

.. autofunction:: asyncmd.config.register_h5py_cache

.. autofunction:: asyncmd.config.deregister_h5py_cache
```
