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
"""
This module contains the implementation of functions configuring asyncmd resource usage.

It also import the configuration functions for submodules (like slurm) to make
them accessible to users in one central place.
"""
import os
import asyncio
import logging
import resource
import typing


from ._config import (_GLOBALS, _SEMAPHORES, _OPT_SEMAPHORES,
                      _GLOBALS_KEYS, _SEMAPHORES_KEYS, _OPT_SEMAPHORES_KEYS,
                      )
from .trajectory.trajectory import (_update_cache_type_for_all_trajectories,
                                    _deregister_h5py_cache_for_all_trajectories,
                                    )
from .trajectory.trajectory_cache import (
        TrajectoryFunctionValueCacheInH5PY as _TrajectoryFunctionValueCacheInH5PY,
        )
# pylint: disable-next=unused-import
from .slurm import set_slurm_settings, set_all_slurm_settings


if typing.TYPE_CHECKING:  # pragma: no cover
    import h5py


logger = logging.getLogger(__name__)


# can be called by the user to (re) set maximum number of processes used
def set_max_process(num: int | None = None, max_num: int | None = None) -> None:
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
    # pylint: disable-next=global-variable-not-assigned
    global _SEMAPHORES
    if num is None:
        if (logical_cpu_count := os.cpu_count()) is not None:
            num = max(1, int(logical_cpu_count / 4))
        else:
            # fallback if os.cpu_count() can not determine the number of cpus
            # play it save and not have more than 2?
            num = 2
    if max_num is not None:
        num = min((num, max_num))
    _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS] = asyncio.BoundedSemaphore(num)


set_max_process()


def set_max_files_open(num: int | None = None, margin: int = 30) -> None:
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
    # and to be sure 30 less (the reason being that we can not use the
    # semaphores from non-async code, but sometimes use the sync subprocess.run
    # and subprocess.check_call [which also need files/pipes to work])
    # also maybe we need other open files like a storage :)
    # pylint: disable-next=global-variable-not-assigned
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
    _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN] = asyncio.BoundedSemaphore(semaval)


set_max_files_open()


# SLURM semaphore stuff:
# slurm max job semaphore, if the user sets it it will be used,
# otherwise we can use an unlimited number of synchronous slurm-jobs
# (if the simulation requires that much)
# Note: We set this here to make sure the semaphore is there independently of if
# slurm is available to make sure it is set if slurm becomes available later by
# (re)setting, e.g., the paths to sinfo/sacct/sbatch and friends
def set_slurm_max_jobs(num: int | None) -> None:
    """
    Set the maximum number of simultaneously submitted SLURM jobs.

    Parameters
    ----------
    num : int or None
        The maximum number of simultaneous SLURM jobs for this invocation of
        python/asyncmd. `None` means do not limit the maximum number of jobs.
    """
    # pylint: disable-next=global-variable-not-assigned
    global _OPT_SEMAPHORES
    if num is None:
        _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB] = None
    else:
        _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB] = asyncio.BoundedSemaphore(num)


set_slurm_max_jobs(num=None)


# Trajectory function value config
def set_trajectory_cache_type(cache_type: str,
                              copy_content: bool = True,
                              clear_old_cache: bool = False
                              ) -> None:
    """
    Set the cache type for TrajectoryFunctionWrapper values.

    By default the content of the current caches is copied to the new caches.
    To clear the old/previously set caches (after copying their values), pass
    ``clear_old_cache=True``.

    Parameters
    ----------
    cache_type : str
        One of "h5py", "npz", "memory".
    copy_content : bool, optional
        Whether to copy the current cache content to the new cache,
        by default True
    clear_old_cache : bool, optional
        Whether to clear the old/previously set cache, by default False.

    Raises
    ------
    ValueError
        Raised if ``cache_type`` is not one of the allowed values.
    """
    # pylint: disable-next=global-variable-not-assigned
    global _GLOBALS
    allowed_values = ["h5py", "npz", "memory"]
    if (cache_type := cache_type.lower()) not in allowed_values:
        raise ValueError(f"Given cache type must be one of {allowed_values}."
                         + f" Was: {cache_type}.")
    if _GLOBALS.get(_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE, "not_set") != cache_type:
        # only do something if the new cache type differs from what we have
        _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] = cache_type
        _update_cache_type_for_all_trajectories(copy_content=copy_content,
                                                clear_old_cache=clear_old_cache,
                                                )


set_trajectory_cache_type("npz")


def register_h5py_cache(h5py_group: "h5py.Group | h5py.File", copy_h5py: bool = False,
                        copy_content: bool = True, clear_old_cache: bool = False,
                        ) -> None:
    """
    Register a h5py file or group for CV value caching.

    Optionally copy over all cached values from the previously set h5py_cache(s),
    also for :class:`Trajectory` objects that are currently not instantiated,
    see the ``copy_h5py`` argument. If it is True all previously set caches will
    be deregistered after copying (since their values are now available in newly
    set cache also).

    Note that in case the trajectory cache type is currently not "h5py", this
    function sets the cache type to "h5py", i.e. it calls :func:`set_trajectory_cache_type`
    with ``cache_type="h5py"``.
    The arguments ``copy_content`` and ``clear_old_cache`` are directly passed
    to :func:`set_trajectory_cache_type`.

    Note that a :class:`h5py.File` is just a slightly special :class:`h5py.Group`,
    so you can pass either. :mod:`asyncmd` will use either the file or the group as
    the root of its own stored values.
    E.g. you will have ``h5py_group["asyncmd/TrajectoryFunctionValueCache"]``
    always pointing to the cached trajectory values and if ``h5py_group`` is
    the top-level group (i.e. the file) you also have
    ``(file["/asyncmd/TrajectoryFunctionValueCache"] ==\
 h5py_group["asyncmd/TrajectoryFunctionValueCache"])``.

    Parameters
    ----------
    h5py_group : h5py.Group or h5py.File
        The file or group to use for caching.
    copy_h5py : bool, optional, by default False
        Whether to copy over all cached values from the previously set h5py cache
        (even for :class:`Trajectory` objects that are currently not instantiated).
    copy_content : bool, optional
        Whether to copy the current cache content to the new cache,
        by default True
    clear_old_cache : bool, optional
        Whether to clear the old/previously set cache, by default False.
    """
    # pylint: disable-next=global-variable-not-assigned
    global _GLOBALS
    if _GLOBALS.get(_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE, "not_set") != "h5py":
        # nothing to copy as h5py was not the old cache type
        if h5py_group.file.mode == "r":
            _GLOBALS[_GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS] = (
                [h5py_group] + _GLOBALS.get(_GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS, [])
                )
        else:
            _GLOBALS[_GLOBALS_KEYS.H5PY_CACHE] = h5py_group
        set_trajectory_cache_type(cache_type="h5py", copy_content=copy_content,
                                  clear_old_cache=clear_old_cache)
    else:
        # cache type already is h5py
        if not copy_h5py:
            # but we dont want to copy, so just set the cache
            if h5py_group.file.mode == "r":
                _GLOBALS[_GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS] = (
                    [h5py_group] + _GLOBALS.get(_GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS, [])
                    )
            else:
                _GLOBALS[_GLOBALS_KEYS.H5PY_CACHE] = h5py_group
        else:
            if h5py_group.file.mode == "r":
                raise ValueError("The passed h5py.File/h5py.Group is open in read-only "
                                 "mode, but ``copy_h5py=True`` was passed. Can not "
                                 f"copy because the file ({h5py_group}) is not writeable!"
                                 )
            # copy the old groups content to the new cache
            caches_to_copy = _GLOBALS.get(_GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS, [])
            if _GLOBALS.get(_GLOBALS_KEYS.H5PY_CACHE, None) is not None:
                caches_to_copy += [_GLOBALS[_GLOBALS_KEYS.H5PY_CACHE]]
            for cache in caches_to_copy:
                _TrajectoryFunctionValueCacheInH5PY.add_values_for_all_trajectories(
                    src_h5py_cache=cache, dst_h5py_cache=h5py_group,
                    )
            # and set the cache
            _GLOBALS[_GLOBALS_KEYS.H5PY_CACHE] = h5py_group
            # also empty the fallback cache list since all they contain should now
            # be in the main cache
            _GLOBALS[_GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS] = []
        # although cache type was already h5py, update the cache type for all
        # trajectories in existence to make sure they use the now set caches
        _update_cache_type_for_all_trajectories(copy_content=copy_content,
                                                clear_old_cache=clear_old_cache)


def deregister_h5py_cache(h5py_group: "h5py.Group | h5py.File"):
    """
    Deregister a given h5py_group from use as a cache for trajectory function values.

    Also deregisters the given h5py_group from all :class:`asyncmd.Trajectory` objects
    currently in existence.

    Parameters
    ----------
    h5py_group : h5py.Group | h5py.File
        The h5py_group to deregister.
    """
    if _GLOBALS.get(_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE, "not_set") == "h5py":
        _deregister_h5py_cache_for_all_trajectories(h5py_group=h5py_group)
    if _GLOBALS.get(_GLOBALS_KEYS.H5PY_CACHE, None) is h5py_group:
        _GLOBALS[_GLOBALS_KEYS.H5PY_CACHE] = None
    if h5py_group in _GLOBALS.get(_GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS, []):
        _GLOBALS[_GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS].remove(h5py_group)


def show_config() -> None:
    """
    Print/show current configuration.
    """
    print(f"Values controlling caching: {_GLOBALS}")
    # pylint: disable-next=protected-access
    sem_print = {key: sem._value
                 for key, sem in {**_SEMAPHORES, **_OPT_SEMAPHORES}.items()
                 if sem is not None}
    print(f"Semaphores controlling resource usage: {sem_print}")
