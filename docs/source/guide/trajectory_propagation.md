# Propagation of MD in segments and/or until a condition is fulfilled

Another notable part of the {py:mod}`asyncmd.trajectory` module are the {py:class}`InPartsTrajectoryPropagator <asyncmd.trajectory.InPartsTrajectoryPropagator>` and the {py:class}`ConditionalTrajectoryPropagator <asyncmd.trajectory.ConditionalTrajectoryPropagator>`, which can both be used to propagate trajectories in chunks of a given walltime.
While the former is used to propagate a trajectory until a given total number of integration steps is reached, the later one can be used to propagate a trajectory until any of the given user-supplied conditions are fulfilled.

The {py:class}`InPartsTrajectoryPropagator <asyncmd.trajectory.InPartsTrajectoryPropagator>` is intended to make it possible to run MD simulations for walltimes longer than the queuing limit and to make full use of [backfilling].

While it also enables running MD simulations longer than the queuing limit and makes full use of [backfilling] by splitting the MD into segments (similar to the {py:class}`InPartsTrajectoryPropagator <asyncmd.trajectory.InPartsTrajectoryPropagator>`), the {py:class}`ConditionalTrajectoryPropagator <asyncmd.trajectory.ConditionalTrajectoryPropagator>` is especially useful for path sampling and committor simulations.
In this case the conditions would be the state functions, but it can be used in general for any situation where the time integration should be stopped on given criteria (as opposed to after a fixed number of integration steps or when a given walltime is reached).
There is also a handy function to create a transition, i.e. a trajectory that connects to (different) conditions from two conditional propagation runs ending in different conditions, the {py:func}`construct_tp_from_plus_and_minus_traj_segments <asyncmd.trajectory.construct_tp_from_plus_and_minus_traj_segments>` function.
It is most likely useful for users implementing some form of path sampling.

[backfilling]: https://slurm.schedmd.com/sched_config.html#backfill

```{seealso}
The example notebooks on the {doc}`InPartsTrajectoryPropagator </examples_link/03_trajectory_propagation_and_subtrajectory_extraction/InPartsTrajectoryPropagator>` and the {doc}`ConditionalTrajectoryPropagator </examples_link/03_trajectory_propagation_and_subtrajectory_extraction/ConditionalTrajectoryPropagator>`
```

## Propagation in segments

```{eval-rst}
.. autoclass:: asyncmd.trajectory.InPartsTrajectoryPropagator
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :exclude-members: __init__, __weakref__
    :special-members:
    :inherited-members:
```

## Propagation until arbitrary conditions are fulfilled

```{eval-rst}
.. autoclass:: asyncmd.trajectory.ConditionalTrajectoryPropagator
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :exclude-members: __init__, __weakref__
    :special-members:
    :inherited-members:

.. autofunction:: asyncmd.trajectory.construct_tp_from_plus_and_minus_traj_segments
```
