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

Molecular dynamics (MD) simulations have become an integral tool to study complex molecular rearrangements and molecular phenomena in many fields and of many different molecular compounds. They are especially useful to study the dynamics, interactions, and function of biomolecules such as proteins, DNA, RNA, lipids, and small drug-like molecules. In MD simulations, the time evolution of a molecular system is obtained by solving Newton’s equations of motion in small timesteps starting from a given initial configuration. Due to the inherent sequential nature of MD and the high dimensionality and complexity of the studied (bio)molecular systems, it can be challenging to reach the timescale of the (biological) events under investigation within a single MD trajectory. This challenge can be addressed by running many MD simulations simultaneously on a high performance computing (HPC) system - possibly in an adaptive manner and combined with enhanced sampling techniques - and then making full use of the resulting ensemble of trajectories. It is therefore paramount to enable MD users to efficiently setup and orchestrate a large number of simultaneous simulations with dynamic dependencies and flexible termination conditions. `asyncmd` enables users to define complex MD sampling workflows in python from simple building blocks and executes these computations directly via the queuing system of a HPC resource. The submission of smaller computation tasks as single jobs (with differing requirements) to the queuing system ensures optimal usage of the heterogeneous resources of modern HPC systems. In addition, this approach supports dynamically growing and shrinking of the total resources allocated to the computation by design, also depending on the demand of other users of the same HPC system. By providing their common building blocks, `asyncmd` also simplifies the development and implementation of advanced trajectory-based enhanced sampling algorithms, including the weighted ensemble method, the string method, or transition path sampling.

# Statement of need

A challenge faced when performing MD simulations of complex (bio)molecular rearrangements is the disparity of timescales between the integration timestep, which needs to be small (usually on the order of femtoseconds) to ensure accurate integration of Newton’s equations of motion, and the time needed to observe the (biological) process under investigation, which can be on the order of seconds to minutes.
Due to the sequential nature inherent to solving Newton’s equations of motion in small timesteps, MD simulations can only be parallelized to a certain degree and much of the parallelization possible relies on dividing the system into regions that are sufficiently far apart to not interact directly with each other.
Taken together with the trend that, in the recent past, computing resources have mostly become wider (i.e. more parallel) but not faster (i.e. not increased in clock speed), the result is that the simulated systems have become larger while the simulated time (per trajectory) did not increase by much anymore.
A common strategy to accumulate the required simulation time to observe the process under investigation while also making the most efficient use of modern highly parallel HPC resources, is to run many MD simulations of the same (biological) system with different initial conditions simultaneously.
This strategy results in a much higher total accumulated simulated time than running one simulation on the same computing resources due to the inherent sequential nature of MD.
While, depending on the number of simultaneous MD simulations, it can be tedious but still possible to manually setup a large number of MD simulations, continually monitoring their progress quickly becomes unfeasible.
However, in many cases it is much more computationally efficient to, instead of performing MD for a fixed number of integration steps, terminate the simulations (and potentially use the freed resources for another MD simulation) once a certain condition is met or event occurred, e.g., the transition between two functional states of a biomolecule.
Conversely, if the goal is to explore a functional state previously unreached by MD, it can be much more efficient to (re)start a large number of MD simulations from the first configuration that reaches the state in any of a number of simulations started from known states.

`asyncmd` is a python library facilitating flexible, programmatic and parallel setup, control, and analysis of an arbitrary number of MD simulations on HPC resources using the python async/await syntax. The library currently supports the SLURM queuing system [@slurm] and the GROMACS MD engine [@gromacs2020], but can easily be extended to other queuing systems and MD engines. To enable the handling of many MD engines simultaneously, all MD engines return a lightweight `Trajectory` object that only contains references to the underlying files and some useful metadata such as the length of the trajectory or the integration timestep used. All trajectory reading and writing operations performed make use of the MDAnalysis library [@mdanalysis2011;@mdanalysis2016], which means that `asyncmd` can profit from the extensive variety of trajectory formats and MD engines supported in MDAnalysis. Notable features of the `asyncmd` library include the propagation of MD until any or all user-supplied conditions are fulfilled on the trajectory, the parallelized application of user defined (python) functions on existing or generated trajectories (including the automatic caching of calculated values), and a dictionary-like interface to the MD parameters. By additionally making it easy to extract any molecular configuration to (re)start an arbitrary number of MD simulations from it, users are enabled to build complex sampling schemes with dynamical dependencies from simple building blocks in python.

All computationally costly operations are submitted via the queuing system, i.e., the process running `asyncmd` has a low computational footprint and can be run on the login node to control complex and long running simulation setups with dynamic dependencies. Additionally, the submission via the existing queuing system ensures that the available HPC resources are efficiently shared with other users independently of them using `asyncmd` or not. For each submittable computation `asyncmd` includes a locally executed equivalent using the same calling convention to facilitate quick prototyping and implementation of (new) algorithms using small test systems on local compute resources. By making it easy to move the execution from a workstation to a HPC cluster, the newly implemented algorithms can then be applied directly to larger, computationally more costly molecular systems.

`asyncmd` can therefore be used to manage large scale MD simulation campaigns on HPC resources. In addition, `asyncmd` provides an ideal building block to develop and implement trajectory based enhanced sampling methods as, e.g., variants of the string method [@e2002StringMethod], highly parallelized transition path or transition interface sampling methods [@dellago2002TransitionPathSampling;@vanerp2003NovelPathSampling], flux sampling methods [@chandler1978;@ruizmontero1997], or the weighted ensemble method [@huber1996Weightedensemble]. Showcasing its potential, `asyncmd` was already used to develop and implement the AI for molecular mechanism discovery (aimmd) algorithm [@jung2023;@aimmd], which adaptively steers a large number of simultaneous MD simulations using a combination of transition path sampling and machine learning.

In addition to its documentation, `asyncmd` includes a number of examples in the form of Jupyter notebooks. These notebooks illustrate most of the common operations, such as performing MD simulations (possibly until a condition is fulfilled) or how to extract configurations from trajectories to restart MD simulations from it, but also include an example showcasing how to implement the weighted ensemble method.

## State of the field

A number of other software packages are relevant in the context of submitting MD simulations on HPC resources or to control them from python, which will be discussed in the following.

Notably, it is possible to control and define MD simulation workflows for GROMACS in python by using its gmxapi python interface [@irrgang2018gmxapi;@irrgang2022gmxapi].
While gmxapi allows for fine grained control of the MD simulation (including, e.g., custom stopping conditions and user plugin code within the force calculation), it is only possible to interact with MD simulations running within the same job allocation or on the same local machine.

The definition and submission of a general (non MD-specific) computational workflow spanning over multiple job allocations on HPC resources from python is possible by using AiiDA [@huber2020aiida;@uhrin2021aiida] or by using the combination of row [@row] and signac [@adorf2018signac].
Both, AiiDa and signac/row, have an emphasis on automatically storing data provenance, while `asyncmd` makes no attempt to store any input/output relations and it is the users responsibility (and freedom) to choose an adequate solution for their use-case to track data provenance.
AiiDA currently supports a number of different queuing systems, but it is not possible to request accelerator resources such as GPUs.
row currently only supports the SLURM queuing system, but offers a finer control over the job resources including the requested memory and number of GPUs.
`asyncmd` currently also only supports the SLURM queuing system, but offers support for any option of the SLURM "sbatch" command to control the requested resources for and execution of the jobs.
MD simulations can be performed with AiiDA by using the aiida-gromacs plugin [@aiida-gromacs] and martignac [@bereau2024martignac] defines a number of coarse-grained Martini simulation workflows with signac.
Another notable package enabling high-throughput MD simulations in the context of materials science is atomate2 [@atomate2], which is also capable of submitting the computations to remote (HPC) resources.
However, to the best knowledge of the authors, no other package besides `asyncmd` exists that offers the submission and control of many MD simulation via a queuing system, while also focusing on versatile and dynamic stopping conditions for the simulations to provide simple building blocks for enhanced sampling algorithms.

Finally, what sets `asyncmd` apart from full-fledged implementations of path sampling and other trajectory based sampling methods, such as, e.g., openpathsampling [@ops1;@ops2], is that it does not implement any specific algorithms to drive the sampling, but instead strives to only provide the common building blocks shared between many trajectory based enhanced sampling methods.

# Acknowledgements

The authors thank all users of `asyncmd` for contributing feedback and suggesting new features, especially Matea Turalija and Vedran Miletic, for feedback on and contributions to the code.
H.J. and G.H. thank the Max Planck Society for financial support and the Max Planck Computing and Data Facility (MPCDF) for computing support.

# References
