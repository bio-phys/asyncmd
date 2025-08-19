# Extending asyncmd

This section is relevant for developers of {py:mod}`asyncmd`, e.g. when you want to add the option to steer an additional molecular dynamics engines (like NAMD or LAMMPS) or add additional ways to wrap functions acting on {py:class}`Trajectory <asyncmd.Trajectory>` objects.

This section also contains the interface of the classes, which are used under the hood by various user facing-classes in {py:mod}`asyncmd` to interact with the SLURM queueing system.
Namely there is the {py:class}`SlurmProcess <asyncmd.slurm.SlurmProcess>`, which emulates the interface of {py:class}`asyncio.subprocess.Process` and which is used to submit and wait for single SLURM jobs.
In the spirit of {py:mod}`asyncio`s subprocess module, there is also the coroutine-function {py:func}`create_slurmprocess_submit <asyncmd.slurm.create_slurmprocess_submit>`, which can be used to create and directly submit a {py:class}`SlurmProcess <asyncmd.slurm.SlurmProcess>` (just like {py:func}`asyncio.create_subprocess_exec`).
Additionally (one level deeper under the hood) there is the {py:class}`SlurmClusterMediator <asyncmd.slurm.cluster_mediator.SlurmClusterMediator>`, which is a singleton class acting as the central communication point between the single {py:class}`SlurmProcess <asyncmd.slurm.SlurmProcess>` and the SLURM commands ("sacct", "sbatch", etc.).
Finally there is also a number of slurm-specific exceptions documented below.

```{seealso}
The example notebook on the {doc}`SlurmProcess </examples_link/05_developer_topics/slurm/SlurmProcess>`.
```

(extending-asyncmd-trajectoryfunctions)=
## Writing a TrajectoryFunctionWrapper class

All wrapper classes for functions acting on {py:class}`Trajectory <asyncmd.Trajectory>` objects must subclass {py:class}`TrajectoryFunctionWrapper <asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper>` to make full and easy use of the caching mechanism already implemented.
You then only need to implement {py:meth}`TrajectoryFunctionWrapper._get_id_str <asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper._get_id_str>` and {py:meth}`TrajectoryFunctionWrapper._get_values_for_trajectory <asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper._get_values_for_trajectory>` to get a fully functional TrajectoryFunctionWrapper class.
See also the (reference) implementations of the other wrapper classes, {py:class}`PyTrajectoryFunctionWrapper <asyncmd.trajectory.PyTrajectoryFunctionWrapper>` and {py:class}`SlurmTrajectoryFunctionWrapper <asyncmd.trajectory.SlurmTrajectoryFunctionWrapper>`.

```{eval-rst}
.. autoclass:: asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper
    :class-doc-from: class
    :member-order: groupwise
    :members: __call__, _get_id_str, _get_values_for_trajectory
    :exclude-members: __init__, __weakref__
    :special-members:
    :private-members:
    :inherited-members:
```

## Writing a MDEngine class

All molecular dynamics engines must subclass the abstract base class {py:class}`asyncmd.mdengine.MDEngine`, which defines the common interface expected from all {py:mod}`asyncmd` engine classes.

In addition the module {py:mod}`asyncmd.mdengine` defines exceptions that the engines should raise when applicable.
Currently defined are:

- {py:class}`asyncmd.mdengine.EngineError` (a generic error, should be raised when no more specific error applies)

- {py:class}`asyncmd.mdengine.EngineCrashedError` (should be raised when the wrapped MD engine code raises an exception during the MD integration)

```{note}
You should also register and implement the corresponding [utility functions](utility-functions-for-common-MD-operations) for your engine.
```

```{eval-rst}
.. autoclass:: asyncmd.mdengine.MDEngine
    :members:
    :member-order: groupwise
    :inherited-members:
    :undoc-members:
```

```{eval-rst}
.. autoexception:: asyncmd.mdengine.EngineError
```

```{eval-rst}
.. autoexception:: asyncmd.mdengine.EngineCrashedError
```

## Writing a MDconfig class for configuration file parsing and writing

All molecular dynamics configuration file wrappers must subclass {py:class}`asyncmd.mdconfig.MDConfig`.
This class defines the two abstract methods {py:meth}`parse <asyncmd.mdconfig.MDConfig.parse>` and {py:meth}`write <asyncmd.mdconfig.MDConfig.write>` as well as the dictionary-like interface by subclassing from {py:class}`collections.abc.MutableMapping`.

However, most often you can probably subclass {py:class}`asyncmd.mdconfig.LineBasedMDConfig` directly.
This has the advantage that you will only need to define the datatypes of the values (if you want them to be typed) and define a function that knows how to parse single lines of the config file format.
To this end you should overwrite the abstract method {py:meth}`LineBasedMDConfig._parse_line <asyncmd.mdconfig.LineBasedMDConfig._parse_line>` in your subclass.
The function will get single lines to parse (as {py:class}`str`) and is expected to return the key, list of value(s) pair as a {py:class}`dict` with one item, i.e. `{key: list of value(s)}`.
If the line is parsed as comment the returned dict must be empty, i.e. `{}`.
If the option/key is present but without associated value(s) the list in the dict must be empty, i.e. `{key: []}`.

```{eval-rst}
.. autoclass:: asyncmd.mdconfig.MDConfig

    .. automethod:: asyncmd.mdconfig.MDConfig.parse

    .. automethod:: asyncmd.mdconfig.MDConfig.write
```

```{eval-rst}
.. autoclass:: asyncmd.mdconfig.LineBasedMDConfig
    :class-doc-from: both

    .. automethod:: asyncmd.mdconfig.LineBasedMDConfig._parse_line

    .. autoproperty:: asyncmd.mdconfig.LineBasedMDConfig.original_file

    .. autoproperty:: asyncmd.mdconfig.LineBasedMDConfig.changed

    .. automethod:: asyncmd.mdconfig.LineBasedMDConfig.parse

    .. automethod:: asyncmd.mdconfig.LineBasedMDConfig.write
```

## SLURM interface classes

```{eval-rst}
.. note ::
    The function below is an alias for/imported from

    .. function:: asyncmd.slurm.process.create_slurmprocess_submit

    **Note:** It is recommended/preferred to use :func:`asyncmd.slurm.create_slurmprocess_submit`.

.. autofunction:: asyncmd.slurm.create_slurmprocess_submit
```

```{eval-rst}
.. autoclass:: asyncmd.slurm.SlurmProcess
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :inherited-members:
```

```{eval-rst}
.. autoclass:: asyncmd.slurm.cluster_mediator.SlurmClusterMediator
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :inherited-members:
```

```{eval-rst}
.. autoexception:: asyncmd.slurm.constants_and_errors.SlurmError

.. autoexception:: asyncmd.slurm.constants_and_errors.SlurmCancellationError

.. autoexception:: asyncmd.slurm.constants_and_errors.SlurmSubmissionError
```
