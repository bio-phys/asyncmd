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
import logging
import resource
import typing


from ._config import _GLOBALS, _SEMAPHORES
from .slurm import set_slurm_settings, set_all_slurm_settings
# TODO: Do we want to set the _GLOBALS defaults here? E.g. CACHE_TYPE="npz"?


logger = logging.getLogger(__name__)


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


def set_max_files_open(num: typing.Optional[int] = None, margin: int = 30):
    """
    Set the maximum number of concurrently opened files.

    By default use the systems soft resource limit.

    Parameters
    ----------
    num : int, optional
        Maximum number of open files, if None use systems (soft) resourcelimit,
        by default None
    margin : int, optional
        Safe margin to keep, i.e. we will only ever open `num - margin` files,
        by default 30

    Raises
    ------
    ValueError
        If num <= margin.
    """
    # ensure that we do not open too many files
    # resource.getrlimit returns a tuple (soft, hard); we take the soft-limit
    # and to be sure 30 less (the reason beeing that we can not use the
    # semaphores from non-async code, but sometimes use the sync subprocess.run
    # and subprocess.check_call [which also need files/pipes to work])
    # also maybe we need other open files like a storage :)
    global _SEMAPHORES
    rlim_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if num is None:
        num = rlim_soft
    elif num > rlim_soft:
        logger.warning("Passed a wanted number of open files that is larger "
                       "than the systems soft resource limit (%d > %d). "
                       "Will be using num=%d instead. To set a higher number "
                       "increase your systems limit on the number of open "
                       "files and call this function again.",
                       num, rlim_soft, rlim_soft,
                       )
        num = rlim_soft
    if num - margin <= 0:
        raise ValueError("num must be larger than margin."
                         f"Was num={num}, margin={margin}."
                         )
    # NOTE: Each MAX_FILES_OPEN semaphore counts for 3 open files!
    #       The reason is that we open 3 files at the same time for each
    #       subprocess (stdin, stdout, stderr), but semaphores can only be
    #       decreased (awaited) once at a time. The problem with just awaiting
    #       it three times in a row is that we can get deadlocked by getting
    #       1-2 semaphores and waiting for the next (last) semaphore in all
    #       threads. The problem is that this semaphore will never be freed
    #       without any process getting a semaphore...
    semaval = int((num - margin) / 3)
    _SEMAPHORES["MAX_FILES_OPEN"] = asyncio.BoundedSemaphore(semaval)


set_max_files_open()


# SLURM semaphore stuff:
# TODO: move this to slurm.py? and initialize only if slurm is available?
# slurm max job semaphore, if the user sets it it will be used,
# otherwise we can use an unlimited number of syncronous slurm-jobs
# (if the simulation requires that much)
# TODO: document that somewhere, bc usually clusters have a job number limit?!
def set_slurm_max_jobs(num: typing.Union[int, None]):
    """
    Set the maximum number of simultaneously submitted SLURM jobs.

    Parameters
    ----------
    num : int or None
        The maximum number of simultaneous SLURM jobs for this invocation of
        python/asyncmd. `None` means do not limit the maximum number of jobs.
    """
    global _SEMAPHORES
    if num is None:
        _SEMAPHORES["SLURM_MAX_JOB"] = None
    else:
        _SEMAPHORES["SLURM_MAX_JOB"] = asyncio.BoundedSemaphore(num)


set_slurm_max_jobs(num=None)


# Trajectory function value config
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


def register_h5py_cache(h5py_group, make_default: bool = False):
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
    make_default: bool,
        Whether we should also make "h5py" the default trajectory function
        cache type. By default False.
    """
    global _GLOBALS
    if make_default:
        set_default_trajectory_cache_type(cache_type="h5py")
    _GLOBALS["H5PY_CACHE"] = h5py_group
