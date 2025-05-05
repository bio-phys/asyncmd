---
title: 'asyncmd: A python library to orchestrate complex molecular dynamics simulation campaigns on high performance computing systems'
tags:
  - python
  - molecular dynamics
  - molecular dynamics simulation
  - enhanced sampling
  - gromacs
  - high performance computing
  - hpc
  - slurm
  - biophysics
  - chemistry
authors:
  - name: Hendrik Jung
    orcid: 0000-0002-2159-0391
    affiliation: 1
  - name: Gerhard Hummer
    orcid: 0000-0001-7768-746X
    affiliation: "1, 2"
affiliations:
 - name: Max Planck Institute of Biophysics, Department of Theoretical Biophysics, Frankfurt am Main, Germany
   index: 1
 - name: Institute of Biophysics, Goethe University Frankfurt, Frankfurt am Main, Germany
   index: 2
date: 23 April 2025
bibliography: paper.bib
---

# Summary

Molecular dynamics (MD) simulations have become an integral tool to study the dynamics, interactions, and function of biomolecules such as proteins, DNA, RNA, lipids, and small drug-like molecules. In MD simulations, the time evolution of a molecular system is obtained by solving Newton’s equations of motion in small timesteps starting from a given initial configuration. Due to the inherent sequential nature of MD and the high dimensionality and complexity of biomolecular systems, it can be challenging to reach the timescale of biological events in a single MD trajectory. This challenge can be addressed by running many MD simulations simultaneously on a high performance computing (HPC) system - possibly in an adaptive manner and combined with enhanced sampling techniques - and then making full use of the resulting ensemble of trajectories. It is therefore paramount to enable MD users to efficiently setup and orchestrate a large number of simultaneous simulations with dynamic dependencies and flexible termination conditions. `asyncmd` enables users to define complex MD sampling workflows in python from simple building blocks and executes these computations directly via the queuing system of a HPC resource. The submission of smaller computation tasks as single jobs (with differing requirements) to the queuing system ensures optimal usage of the heterogeneous resources of modern HPC systems. In addition, this approach supports dynamically growing and shrinking of the total resources allocated to the computation by design, also depending on the demand of other users of the same HPC system. By providing their common building blocks, `asyncmd` also simplifies the development and implementation of advanced trajectory-based enhanced sampling algorithms, including the weighted ensemble method, the string method, or transition path sampling.

# Statement of need

`asyncmd` is a python library facilitating flexible, programmatic and parallel setup, control, and analysis of an arbitrary number of MD simulations on HPC resources using the python async/await syntax. The library currently supports the SLURM queuing system [@slurm] and the GROMACS MD engine [@gromacs2020], but can easily be extended to other queuing systems and MD engines. To enable the handling of many MD engines simultaneously, all MD engines return a lightweight `Trajectory` object that only contains references to the underlying files and some useful metadata such as the length of the trajectory or the integration timestep used. All trajectory reading and writing operations performed make use of the MDAnalysis library [@mdanalysis2011;@mdanalysis2016], which means that `asyncmd` can profit from the extensive variety of trajectory formats and MD engines supported in MDAnalysis. Notable features of the `asyncmd` library include the propagation of MD until any or all user-supplied conditions are fulfilled on the trajectory, the parallelized application of user defined (python) functions on existing or generated trajectories (including the automatic caching of calculated values), and a dictionary-like interface to the MD parameters. By additionally making it easy to extract any molecular configuration to (re)start an arbitrary number of MD simulations from it, users are enabled to build complex sampling schemes with dynamical dependencies from simple building blocks in python.

All computationally costly operations are submitted via the queuing system, i.e., the process running `asyncmd` has a low computational footprint and can be run on the login node to control complex and long running simulation setups with dynamic dependencies. Additionally, the submission via the existing queuing system ensures that the available HPC resources are efficiently shared with other users independently of them using `asyncmd` or not. For each submittable computation `asyncmd` includes a locally executed equivalent using the same calling convention to facilitate quick prototyping and implementation of (new) algorithms using small test systems on local compute resources. By making it easy to move the execution from a workstation to a HPC cluster, the newly implemented algorithms can then be applied directly to larger, computationally more costly molecular systems.

`asyncmd` can therefore be used to manage large scale MD simulation campaigns on HPC resources. In addition, `asyncmd` provides an ideal building block to develop and implement trajectory based enhanced sampling methods as, e.g., variants of the string method [@e2002StringMethod], highly parallelized transition path or transition interface sampling methods [@dellago2002TransitionPathSampling;@vanerp2003NovelPathSampling], flux sampling methods [@chandler1978;@ruizmontero1997], or the weighted ensemble method [@huber1996Weightedensemble]. Showcasing its potential, `asyncmd` was already used to develop and implement the AI for molecular mechanism discovery (aimmd) algorithm [@jung2023;@aimmd], which adaptively steers a large number of simultaneous MD simulations using a combination of transition path sampling and machine learning.

# Acknowledgements

The authors thank all users of `asyncmd` for contributing feedback and suggesting new features, especially Matea Turalija and Vedran Miletic, for feedback on and contributions to the code.
H.J. and G.H. thank the Max Planck Society for financial support and the Max Planck Computing and Data Facility (MPCDF) for computing support.

# References
