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
