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
Configuration dictionaries to influence asyncmd runtime behavior and resource usage.

Also define the keys in the dictionaries we will use (see the "_KEYS" classes below).

NOTE: This file **only** contains the dictionaries with the values
      and **no** functions to set them, the funcs all live in 'config.py'.
      The idea here is that we can then without any issues import additional
      stuff (like the config functions from 'slurm.py') in 'config.py'
      without risking circular imports because all asyncmd files should only
      need to import the _CONFIG and _SEMAPHORES dicts from '_config.py'.
"""
import asyncio
import typing


_GLOBALS: dict[str, typing.Any] = {}
_SEMAPHORES: dict[str, asyncio.BoundedSemaphore] = {}
# These semaphores are optional (i.e. can be None, which means unlimited)
# e.g. slurm_max_jobs
_OPT_SEMAPHORES: dict[str, asyncio.BoundedSemaphore | None] = {}


class _GlobalsKeys(typing.NamedTuple):
    TRAJECTORY_FUNCTION_CACHE_TYPE: str = "TRAJECTORY_FUNCTION_CACHE_TYPE"
    H5PY_CACHE: str = "H5PY_CACHE"
    H5PY_CACHE_READ_ONLY_FALLBACKS: str = "H5PY_CACHE_READ_ONLY_FALLBACKS"


_GLOBALS_KEYS = _GlobalsKeys()


class _SemaphoresKeys(typing.NamedTuple):
    MAX_PROCESS: str = "MAX_PROCESS"
    # NOTE: Each MAX_FILES_OPEN semaphore counts for 3 open files!
    #       The reason is that we open 3 files at the same time for each
    #       subprocess (stdin, stdout, stderr), but semaphores can only be
    #       decreased (awaited) once at a time. The problem with just awaiting
    #       it three times in a row is that we can get deadlocked by getting
    #       1-2 semaphores and waiting for the next (last) semaphore in all
    #       threads. The problem is that this semaphore will never be freed
    #       without any process getting a semaphore...
    MAX_FILES_OPEN: str = "MAX_FILES_OPEN"


_SEMAPHORES_KEYS = _SemaphoresKeys()


class _OptSemaphoresKeys(typing.NamedTuple):
    SLURM_MAX_JOB: str = "SLURM_MAX_JOB"


_OPT_SEMAPHORES_KEYS = _OptSemaphoresKeys()
