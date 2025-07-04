# Propagation of MD in segments and/or until a condition is fulfilled

Another notable part of the {py:mod}`asyncmd.trajectory` module are the {py:class}`InPartsTrajectoryPropagator <asyncmd.trajectory.InPartsTrajectoryPropagator>` and the {py:class}`ConditionalTrajectoryPropagator <asyncmd.trajectory.ConditionalTrajectoryPropagator>`, which can both be used to propagate trajectories in chunks of a given walltime.
While the former is used to propagate a trajectory until a given total number of integration steps is reached, the later one can be used to propagate a trajectory until any of the given user-supplied conditions are fulfilled.
The later is especially useful for pathsampling and committor simulations (here the conditions would be the state functions), but can be used in general for any situation where the time integration should be stopped on given criteria (as opposed to after a fixed number of integration steps or when a given walltime is reached).
There is also a handy function to create a transition, i.e. a trajectory that
connects to (different) conditions from two conditional propagation runs ending
in different conditions, the {py:func}`construct_TP_from_plus_and_minus_traj_segments <asyncmd.trajectory.construct_TP_from_plus_and_minus_traj_segments>` function.
It is most likely useful for users implementing some form of pathsampling.

```{eval-rst}
.. autoclass:: asyncmd.trajectory.InPartsTrajectoryPropagator
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :exclude-members: __init__, __weakref__
    :special-members:
    :inherited-members:

.. autoclass:: asyncmd.trajectory.ConditionalTrajectoryPropagator
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :exclude-members: __init__, __weakref__
    :special-members:
    :inherited-members:

.. autofunction:: asyncmd.trajectory.construct_TP_from_plus_and_minus_traj_segments
```
