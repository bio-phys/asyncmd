API (Overview by submodules)
============================

trajectory
**********
The :py:mod:`asyncmd.trajectory` module contains a
:py:class:`asyncmd.Trajectory` class which is the return object for all MD
engines. There are also a number of `TrajectoryFunctionWrapper` classes which
can be used to wrapp (python) functions or arbitrary executables for easy
asyncronous application on :py:class:`asyncmd.Trajectory`, either submitted
via slurm or ran locally.

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

API (For developers)
====================

This section is relevant for developers of :py:mod:`asyncmd`, e.g. when you
want to add the option to steer additional molecular dynamcis engines like NAMD.

Molecular dynamcis configuration file parsing and writing (:py:class:`asyncmd.mdconfig.MDConfig`)
*************************************************************************************************

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

.. autoclass:: asyncmd.mdconfig.LineBasedMDConfig
   :member-order: bysource
   :members:
   :private-members:
   :inherited-members:

Molecular dynamcis simulation engine wrappers (:py:class:`asyncmd.mdengine.MDEngine`)
*************************************************************************************

TODO!

API (Hierachical module layout plan)
====================================

.. autosummary::
   :recursive:
   :toctree: generated

   asyncmd
