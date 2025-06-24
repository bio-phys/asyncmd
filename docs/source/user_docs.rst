Overview for users
==================

trajectory
**********
The :py:mod:`asyncmd.trajectory` module contains a
:py:class:`asyncmd.Trajectory` class which is the return object for all MD
engines. These objects enable easy access to a number properties of the
underlying trajectory, like the length in frames or time, the intergration step
and many more. Note :py:class:`asyncmd.Trajectory` are unique objects in the
sense that every combination of underlying ``trajectory_files`` will give you the
same object back even if you instantiate it multiple times, i.e. ``is`` will be
``True`` for the two objects (in addition to ``==`` beeing ``True``).
Also note that it is possible to pickle and unpickle :py:class:`asyncmd.Trajectory`
objects. You can even change the filepath of the underlying trajectories, i.e.
copy/move them to another location (consider also moving the npz cache files)
and still unpickle to get a working :py:class:`asyncmd.Trajectory` object as
long as the relative path between your python workdir and the trajectory files
does not change. Or you can change the workdir of the python interpreter as long
as the trajectory files remain at the same location in the filesystem.

..
   TODO: reference to the dev section where we explain the two hidden funcs
   to forget traj objects

There are also a number of ``TrajectoryFunctionWrapper`` classes which
can be used to wrapp (python) functions or arbitrary executables for easy
asyncronous application on :py:class:`asyncmd.Trajectory`, either submitted
via slurm or ran locally. The benefit of the wrapped functions is that the
calculated values will be cached automatically. The caching is even persistent
over multiple reloads and invocations of the python interpreter. To this end
the default caching mechanism creates hidden numpy npz files for every
:py:class:`asyncmd.Trajectory` (named after the trajectory) in which the values
are stored. Other caching mechanism are an in-memory cache and the option to
store all cached values in a :py:class:`h5py.File` or :py:class:`h5py.Group`.
You can set the default caching mechanism for all
:py:class:`asyncmd.Trajectory` centrally via
:py:func:`asyncmd.config.set_default_trajectory_cache_type` or overwrite it for
each :py:class:`asyncmd.Trajectory` at init by passing ``cache_type``. See also
:py:func:`asyncmd.config.register_h5py_cache` to register the h5py cache.

.. py:currentmodule:: asyncmd.trajectory.convert

It also contains a number of classes to extract frames from
:py:class:`asyncmd.Trajectory` objects in the module
:py:mod:`asyncmd.trajectory.convert`:

  - :py:class:`NoModificationFrameExtractor`

  - :py:class:`InvertedVelocitiesFrameExtractor`

  - :py:class:`RandomVelocitiesFrameExtractor`

Note that implementing your own ``FrameExtractor`` with a custom modification
is as easy as subclassing the abstract base class :py:class:`FrameExtractor`
and overwriting its :py:meth:`FrameExtractor.apply_modification` method.

The :py:mod:`asyncmd.trajectory.convert` module furthermore contains a class to
concatenate :py:class:`asyncmd.Trajectory` segments, the
:py:class:`TrajectoryConcatenator`. It can be used to concatenate lists of
trajectory segments in any order (and possibly with inverted momenta) by
passing a list of trajectory segments and a list of tuples (slices) that
specify the frames to use in the concatenated output trajectory. Note that this
class gives you all customizability at the cost of complexity, if you just want
to construct a transition from trajectry segments the
:py:func:`asyncmd.trajectory.construct_TP_from_plus_and_minus_traj_segments` is
most probably easier to use and what you want (it uses the
:py:class:`TrajectoryConcatenator` under the hood anyway).

Note that both the `FrameExtractor`s and the `TrajectoryConcatenator` have an
async version of their functions doing the work (`extract` and `concatenate`
respectively). The awaitable versions do exactly the same as their sync
counterparts, just that they do so in a seperate thread.

.. py:currentmodule:: asyncmd

Another notable part of the :py:mod:`asyncmd.trajectory` module are the
:py:class:`asyncmd.trajectory.InPartsTrajectoryPropagator` and
:py:class:`asyncmd.trajectory.ConditionalTrajectoryPropagator`, which
can both be used to propagate trajectories in chunks of a given walltime. While
the former is used to propagate a trajectory until a given total number of
integration steps, the later one can be used to propagate a trajectory until
any of the given conditions is fulfilled. The later is especially usefull for
pathsampling and committor simulations (here the conditions would be the state
functions), but can be used in general for any situation where the time
integration should be stopped on given criteria (as opposed to after a fixed
number of integratiopn steps or when a given walltime is reached).
There is also a handy function to create a transition, i.e. a trajectory that
connects to (different) conditions from two conditional propagation runs ending
in different conditions, the
:py:func:`asyncmd.trajectory.construct_TP_from_plus_and_minus_traj_segments`
function. It is most likely usefull for users implementing some form of
pathsampling.

Trajectory
----------
.. autoclass:: asyncmd.Trajectory
   :members:
   :special-members:
   :inherited-members:

TrajectoryFunctionWrappers
--------------------------

.. autoclass:: asyncmd.trajectory.PyTrajectoryFunctionWrapper
   :members:
   :special-members:
   :inherited-members:

.. autoclass:: asyncmd.trajectory.SlurmTrajectoryFunctionWrapper
   :members:
   :special-members:
   :inherited-members:

FrameExtractors
---------------

.. autoclass:: asyncmd.trajectory.convert.FrameExtractor
   :members:
   :special-members:
   :inherited-members:

.. autoclass:: asyncmd.trajectory.convert.NoModificationFrameExtractor
   :members:
   :inherited-members:

.. autoclass:: asyncmd.trajectory.convert.InvertedVelocitiesFrameExtractor
   :members:
   :inherited-members:

.. autoclass:: asyncmd.trajectory.convert.RandomVelocitiesFrameExtractor
   :members:
   :special-members:
   :inherited-members:

Trajectory propagation
----------------------------------

.. autoclass:: asyncmd.trajectory.InPartsTrajectoryPropagator
   :members:
   :special-members:
   :inherited-members:

.. autoclass:: asyncmd.trajectory.ConditionalTrajectoryPropagator
   :members:
   :special-members:
   :inherited-members:

Trajectory concatenation
------------------------

.. autofunction:: asyncmd.trajectory.construct_TP_from_plus_and_minus_traj_segments

.. autoclass:: asyncmd.trajectory.convert.TrajectoryConcatenator
   :members:
   :special-members:
   :inherited-members:

gromacs
*******

The :py:mod:`asyncmd.gromacs` module contains all classes and functions to
control gromacs engines from python. Most notably the
:py:class:`asyncmd.gromacs.MDP` (which provides a dictionary-like interface to
read, modify and write gromacs mdp files), and the two gromacs engines
:py:class:`asyncmd.gromacs.GmxEngine` and
:py:class:`asyncmd.gromacs.SlurmGmxEngine` which share most of their interface
with the important difference that the
:py:class:`asyncmd.gromacs.SlurmGmxEngine` submits all MD simulations via
slurm while the :py:class:`asyncmd.gromacs.GmxEngine` runs locally on the same
machine as the python process.

MDP
---
.. autoclass:: asyncmd.gromacs.MDP
   :members:

Engine classes
--------------
.. autoclass:: asyncmd.gromacs.GmxEngine
   :members:
   :special-members:
   :inherited-members:

.. autoclass:: asyncmd.gromacs.SlurmGmxEngine
   :members:
   :special-members:
   :inherited-members:

config
******

Various functions for configuring :py:mod:`asyncmd` behaviour during runtime.
Most notably are probably the functions to limit resource use (i.e. number of
concurrent SLURM jobs, number of open files, number of processes, etc.) and
functions to influence the :py:class:`asyncmd.Trajectory` CV value caching
like setting the default cache type for all :py:class:`asyncmd.Trajectory` or
registering a ``h5py`` file (or group) for caching.

General resource usage
----------------------

.. autofunction:: asyncmd.config.set_max_process

.. autofunction:: asyncmd.config.set_max_files_open

SLURM settings and resource usage
---------------------------------

.. autofunction:: asyncmd.config.set_slurm_max_jobs

.. autofunction:: asyncmd.config.set_slurm_settings

CV value caching
----------------

.. autofunction:: asyncmd.config.set_default_trajectory_cache_type

.. autofunction:: asyncmd.config.register_h5py_cache
