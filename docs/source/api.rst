API
===

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



Hierachical module layout
*************************

.. autosummary::
   :recursive:
   :toctree: generated

   asyncmd




..
   .. automodule:: asyncmd.gromacs.mdengine
      :member-order: bysource
      :members:
      :special-members:

..
   main (test what actually is here?)
   **********************************
   .. automodule:: asyncmd
      :members:
      :private-members:
      :special-members:

   gromacs
   *******

   mdconfig
   --------
   .. automodule:: asyncmd.gromacs.mdconfig
      :members:

   mdengines
   ---------
   .. automodule:: asyncmd.gromacs.mdengine
      :member-order: bysource
      :members:
      :special-members:


   Advanced users and developers
   *****************************

   The abstract base classes all mdengines and mdconfigs must subclass:

   MDConfig
   --------
   .. automodule:: asyncmd.mdconfig
      :members:
      :member-order: bysource


   MDEngines
   ---------
   .. automodule:: asyncmd.mdengine
      :members:

