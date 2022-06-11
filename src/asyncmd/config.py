# This file is part of asyncmd.
#
# asyncmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# asyncmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with asyncmd. If not, see <https://www.gnu.org/licenses/>.

import os
import asyncio
import resource
# TODO: is the the best place for our semaphore(s)?


_GLOBALS = {}
_SEMAPHORES = {}


# can be called by the user to (re) set maximum number of processes used
def set_max_process(num=None, max_num=None):
    """
    Set the maximum number of concurrent python processes.

    If num is None, default to os.cpu_count() / 4.

    Parameters
    ----------
    num : int, optional
        Number of processes, if None will default to 1/4 of the CPU count.
    max_num : int, optional
        If given the number of processes can not exceed this number independent
        of the value of CPU count. Useful mostly for code that runs on multiple
        different machines (with different CPU counts) but still wants to avoid
        spawning hundreds of processes.
    """
    # NOTE: I think we should use a conservative default, e.g. 0.25*cpu_count()
    # TODO: limit to 30-40?, i.e never higher even if we have 1111 cores?
    global _SEMAPHORES
    if num is None:
        logical_cpu_count = os.cpu_count()
        if logical_cpu_count is not None:
            num = int(logical_cpu_count / 4)
        else:
            # fallback if os.cpu_count() can not determine the number of cpus
            # play it save and not have more than 2?
            # TODO: think about a good number!
            num = 2
    if max_num is not None:
        num = min((num, max_num))
    _SEMAPHORES["MAX_PROCESS"] = asyncio.BoundedSemaphore(num)


set_max_process()


# ensure that we do not open too many files
# resource.getrlimit returns a tuple (soft, hard); we take the soft-limit
# and to be sure 30 less (the reason beeing that we can not use the semaphore
# from non-async code, but sometimes use the sync subprocess.run and
# subprocess.check_call [which also need files/pipes to work])
# also maybe we need other open files like a storage :)
_SEMAPHORES["MAX_FILES_OPEN"] = asyncio.BoundedSemaphore(
                        resource.getrlimit(resource.RLIMIT_NOFILE)[0] - 30
                                                         )


# SLURM semaphore stuff:
# TODO: move this to slurm.py? and initialize only if slurm is available?
# semaphore to make sure we modify clusterinfo only from one thread at a time
_SEMAPHORES["SLURM_CLUSTER_MEDIATOR"] = asyncio.BoundedSemaphore(1)
# slurm max job semaphore, if the user sets it it will be used,
# otherwise we can use an unlimited number of syncronous slurm-jobs
# (if the simulation requires that much)
# TODO: document that somewhere, bc usually clusters have a job-limit?!
_SEMAPHORES["SLURM_MAX_JOB"] = None


def set_max_slurm_jobs(num: int):
    """
    Set the maximum number of simultaneously submitted SLURM jobs.

    Parameters
    ----------
    num : int
        The maximum number of simulteneous SLURM jobs for this invocation of
        python/asyncmd.
    """
    global _SEMAPHORES
    _SEMAPHORES["SLURM_MAX_JOB"] = asyncio.BoundedSemaphore(num)


def set_default_trajectory_cache_type(cache_type: str):
    """
    Set the default cache type for TrajectoryFunctionValues.

    Note that this can be overwritten on a per trajectory basis by passing
    ``cache_type`` to ``Trajectory.__init__``.

    Parameters
    ----------
    cache_type : str
        One of "h5py", "npz", "memory".

    Raises
    ------
    ValueError
        Raised if ``cache_type`` is not one of the allowed values.
    """
    global _GLOBALS
    allowed_values = ["h5py", "npz", "memory"]
    cache_type = cache_type.lower()
    if cache_type not in allowed_values:
        raise ValueError(f"Given cache type must be one of {allowed_values}."
                         + f" Was: {cache_type}.")
    _GLOBALS["TRAJECTORY_FUNCTION_CACHE_TYPE"] = cache_type


def register_h5py_cache(h5py_group):
    """
    Register a h5py file or group for CV value caching.

    Note that this also sets the default cache type to "h5py", i.e. it calls
    :func:`set_default_trajectory_cache_type` with ``cache_type="h5py"``.

    Note that a ``h5py.File`` is just a slightly special ``h5py.Group``, so you
    can pass either. :mod:`asyncmd` will use euther the file or the group as
    the root of its own stored values.
    E.g. you will have ``h5py_group["asyncmd/TrajectoryFunctionValueCache"]``
    always pointing to the cached trajectory values and if ``h5py_group`` is
    the top-level group (i.e. the file) you also have ``(file["/asyncmd/TrajectoryFunctionValueCache"] == h5py_group["asyncmd/TrajectoryFunctionValueCache"])``.

    Parameters
    ----------
    h5py_group : h5py.Group or h5py.File
        The file or group to use for caching.
    """
    global _GLOBALS
    set_default_trajectory_cache_type(cache_type="h5py")
    _GLOBALS["H5PY_CACHE"] = h5py_group
