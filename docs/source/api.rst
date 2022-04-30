API
===

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

Engines
-------
.. autoclass:: asyncmd.gromacs.GmxEngine
      :members:
      :special-members:

.. autoclass:: asyncmd.gromacs.SlurmGmxEngine
      :members:
      :special-members:
      :inherited-members:



module layout
*************

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

