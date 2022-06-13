Overview for users
==================

trajectory
**********
The :py:mod:`asyncmd.trajectory` module contains a
:py:class:`asyncmd.Trajectory` class which is the return object for all MD
engines. There are also a number of ``TrajectoryFunctionWrapper`` classes which
can be used to wrapp (python) functions or arbitrary executables for easy
asyncronous application on :py:class:`asyncmd.Trajectory`, either submitted
via slurm or ran locally.
The benefit of these wrapped functions is that the calculated CV values will be
cached automatically. The caching is even persistent over multiple reloads and
invocations of the python interpreter. To this end the default caching
mechanism creates hidden numpy npz files for every
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
:py:class:`TrajectoryConcatenator`.

.. py:currentmodule:: asyncmd

Another notable part of the :py:mod:`asyncmd.trajectory` module is the
:py:class:`asyncmd.trajectory.ConditionalTrajectoryPropagator` which
can be used to propagate a trajectory until any of the given conditions is
fulfilled. This is especially usefull for pathsampling and committor
simulations (here the conditions would be the state functions), but can be used
in general for any situation where the time integration should be stopped on
given criteria (as opposed to after a fixed number of integratiopn steps or
when a given walltime is reached).

Trajectory
----------
.. autoclass:: asyncmd.Trajectory
   :members:
   :special-members:

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

Trajectory concatenation
------------------------

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

Overview for developers
=======================

This section is relevant for developers of :py:mod:`asyncmd`, e.g. when you
want to add the option to steer an additional molecular dynamcis engines (like
NAMD or LAMMPS) or add additional ways to wrapp functions acting on
:py:class:`asyncmd.Trajectory`.

.. py:currentmodule:: asyncmd.slurm

This section also contains the interface of the classes, which are used under
the hood by various user facing-classes in :py:mod:`asyncmd` to interact with
the SLURM queueing system.
Namely there is the :py:class:`SlurmProcess`, which emulates the interface of
:py:class:`asyncio.subprocess.Process` and which is used to submit and wait for
single SLURM jobs. Additionally (one level deeper under the hood) there is the
:py:class:`SlurmClusterMediator`, which is a singleton class acting as the
central communication point between the single :py:class:`SlurmProcess` and the
SLURM commands ("sacct", "sbatch", etc.).

SLURM interface classes
***********************

.. autoclass:: asyncmd.slurm.SlurmProcess
   :member-order: bysource
   :members:
   :private-members:
   :special-members:
   :inherited-members:

.. autoclass:: asyncmd.slurm.SlurmClusterMediator
   :member-order: bysource
   :members:
   :private-members:
   :special-members:
   :inherited-members:

..   :undoc-members:

Wrapper classes for functions acting on trajectories
****************************************************

.. :py:currentmodule:: asyncmd.trajectory.functionwrapper

All wrapper classes for functions acting on :py:class:`asyncmd.Trajectory`
should subclass :py:class:`TrajectoryFunctionWrapper` to make full and easy use
of the caching mechanism already implemented. You then only need to implement
:py:meth:`TrajectoryFunctionWrapper._get_id_str` and
:py:meth:`TrajectoryFunctionWrapper.get_values_for_trajectory` to get a fully
functional TrajectoryFunctionWrapper class. See also the (reference)
implementation of the other wrapper classes,
:py:class:`PyTrajectoryFunctionWrapper` and
:py:class:`SlurmTrajectoryFunctionWrapper`.

.. autoclass:: asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper
   :member-order: bysource
   :members: __call__, _get_id_str, get_values_for_trajectory
   :special-members:
   :private-members:
   :inherited-members:

..   :undoc-members:

Molecular dynamics configuration file parsing and writing
*********************************************************

All molecular dynamics configuration file wrappers should subclass
:py:class:`asyncmd.mdconfig.MDConfig`. This class defines the two abstract
methods ``parse()`` and ``write()`` as well as the dictionary-like interface by
subclassing from :class:`collections.abc.MutableMapping`.

Most often you can probably subclass
:py:class:`asyncmd.mdconfig.LineBasedMDConfig` directly. This has the advantage
that you will only need to define the datatypes of the values (if you want
them to be typed) and define a function that knows how to parse single lines of
the config file format. To this end you should overwrite the abstract method
:py:func:`asyncmd.mdconfig.LineBasedMDConfig._parse_line` in your subclass.
The function will get single lines to parse is expected to return the key, list
of value(s) pair as a :py:class:`dict` with one item, e.g.
``{key: list of value(s)}``. If the line is parsed as comment the returned dict
must be empty, e.g. ``{}``. If the option/key is present but without associated
value(s) the list in the dict must be empty, e.g. ``{key: []}``.

.. autoclass:: asyncmd.mdconfig.MDConfig
   :member-order: bysource
   :members: write, parse
   :inherited-members:

.. autoclass:: asyncmd.mdconfig.LineBasedMDConfig
   :member-order: bysource
   :members:
   :private-members:
   :inherited-members:

Molecular dynamics simulation engine wrappers
*********************************************

All molecular dynamics engines should subclass the abstract base class
:py:class:`asyncmd.mdengine.MDEngine`, which defines the common interface
expected from all py:module:`asyncmd` engine classes.

In addition the module :py:mod:`asyncmd.mdengine` defines exceptions that the
engines should raise when applicable. Currently defined are:

- :py:class:`asyncmd.mdengine.EngineError` (a generic error, should be
  raised when no more specific error applies)

- :py:class:`asyncmd.mdengine.EngineCrashedError` (should be raised when the
  wrapped MD engine code raises an exception during the MD integration)

.. autoclass:: asyncmd.mdengine.MDEngine
   :members:
   :member-order: bysource
   :inherited-members:
   :undoc-members:

.. autoclass:: asyncmd.mdengine.EngineError

.. autoclass:: asyncmd.mdengine.EngineCrashedError

API (Hierachical module layout plan)
====================================

.. autosummary::
   :recursive:
   :toctree: generated

   asyncmd
