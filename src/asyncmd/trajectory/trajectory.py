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
This module contains the implementation the asyncmd.Trajectory class.

It also contains some helper function related to the global Trajectory registry
used for trajectory function value caching.
The actual :class:`TrajectoryFunctionValueCache` classes can be found in the
``trajectory_cache`` module.
"""
import asyncio
import collections
import dataclasses
import hashlib
import io
import logging
import os
import typing

import MDAnalysis as mda
import numpy as np

from .._config import _GLOBALS, _GLOBALS_KEYS
from .trajectory_cache import (TrajectoryFunctionValueCache,
                               TrajectoryFunctionValueCacheInH5PY,
                               TrajectoryFunctionValueCacheInMemory,
                               TrajectoryFunctionValueCacheInNPZ,
                               ValuesAlreadyStoredError)

if typing.TYPE_CHECKING:  # pragma: no cover
    # only import for typing to avoid circular imports
    from .functionwrapper import TrajectoryFunctionWrapper
    import h5py


logger = logging.getLogger(__name__)


# dictionary in which we keep track of trajectory objects
# we use it to always return the *same* object for the same trajectory (by hash)
# this makes it easy to ensure that we never calculate CV functions twice
_TRAJECTORIES_BY_HASH: dict[int, "Trajectory"] = {}


def clear_all_cache_values_for_all_trajectories() -> None:
    """
    Clear all function values cached for each :class:`Trajectory` currently in existence.

    For file-based caches, this also removes the associated cache files.
    """
    for traj in _TRAJECTORIES_BY_HASH.values():
        traj.clear_all_cache_values()


def _update_cache_type_for_all_trajectories(copy_content: bool = True,
                                            clear_old_cache: bool = False,
                                            ) -> None:
    """
    Update the cache type for each :class:`Trajectory` currently in existence.

    By default the content of the current caches is copied to the new caches.
    See :func:`asyncmd.config.set_trajectory_cache_type` to set the ``cache_type``.
    To clear the old/previously set caches (after copying their values), pass
    ``clear_old_cache=True``.

    Parameters
    ----------
    copy_content : bool, optional
        Whether to copy the current cache content to the new cache,
        by default True
    clear_old_cache : bool, optional
        Whether to clear the old/previously set cache, by default False.
    """
    for traj in _TRAJECTORIES_BY_HASH.values():
        traj.update_cache_type(copy_content=copy_content,
                               clear_old_cache=clear_old_cache,
                               )


def _deregister_h5py_cache_for_all_trajectories(h5py_group: "h5py.File | h5py.Group"
                                                ) -> None:
    """
    Deregister the given h5py_group as cache from all :class:`Trajectory` objects.

    Parameters
    ----------
    h5py_group : h5py.Group | h5py.File
        The h5py_group to deregister.
    """
    for traj in _TRAJECTORIES_BY_HASH.values():
        traj._deregister_h5py_cache(h5py_group)


def _forget_all_trajectories() -> None:
    """
    Forget about the existence of all :class:`Trajectory` objects.

    This will result in new :class:`Trajectory` objects being created even for
    the same underlying trajectory_files. Usually you do not want this as it
    results in unnecessary calculations if the same wrapped and cached function
    is applied to both objects. This function exists as a hidden function as it
    is used in the tests and it might be helpful under certain circumstances.
    Use only if you know why you are using it!
    """
    # pylint: disable-next=global-variable-not-assigned
    global _TRAJECTORIES_BY_HASH
    all_keys = set(_TRAJECTORIES_BY_HASH.keys())
    for key in all_keys:
        del _TRAJECTORIES_BY_HASH[key]


def _forget_trajectory(traj_hash: int) -> None:
    """
    Forget about the existence of a given :class:`Trajectory` object.

    This will result in new :class:`Trajectory` objects being created even for
    the same underlying trajectory_files. Usually you do not want this as it
    results in unnecessary calculations if the same wrapped and cached function
    is applied to both objects. This function exists as a hidden function as it
    might be helpful under certain circumstances. Use only if you know why you
    are using it!

    Parameters
    ----------
    traj_hash : int
        The hash of the :class:`Trajectory` to forget about.
    """
    # pylint: disable-next=global-variable-not-assigned
    global _TRAJECTORIES_BY_HASH
    try:
        del _TRAJECTORIES_BY_HASH[traj_hash]
    except KeyError:
        # not in there, do nothing
        pass


@dataclasses.dataclass(frozen=True)
class _TrajectoryPropertyData:
    """
    Dataclass to store/bundle all information that is read from the trajectory
    and made available as :class:`Trajectory` properties.

    All data are immutable (we use ``frozen=True``), because the data are read
    from the underlying trajectory file(s) only once and if they change the hash
    (i.e. the :class:`Trajectory` object the data is tied to) will also change.
    """
    length: int
    dt: float
    first_time: float
    last_time: float
    first_step: int | None
    last_step: int | None


@dataclasses.dataclass(frozen=True)
class _TrajectoryFileData:
    """
    Dataclass to store/bundle all information related to the file-paths and
    trajectory hash for :class:`Trajectory` objects.

    All of this is set in :meth:`Trajectory.__new__` and must not be overridden
    or set again in :meth:`Trajectory.__init__`!
    """
    trajectory_files: list[str]
    structure_file: str
    workdir: str
    trajectory_hash: int


class Trajectory:
    """
    Represent a trajectory.

    Keep track of the paths of the trajectory and the structure files.
    Caches values for (wrapped) functions acting on the trajectory.
    Supports pickling and unpickling with the cached values restored, if a
    non-persistent cache is used when pickling, the values will be written to a
    hidden numpy npz file next to the trajectory and will be read at unpickling.
    Supports equality checks with other :class:`Trajectory`.
    Also makes available (and caches) a number of useful attributes, e.g.
    ``first_step`` and ``last_step`` (the first and last integration step in
    the trajectory), ``dt``, ``first_time``, ``last_time``,and ``length`` (in
    frames). All properties are read-only (for the simple reason that they
    depend only on the underlying trajectory files).
    A special case is ``nstout``, the output frequency in integration steps.
    Since it can not be reliably read/inferred from the trajectory files alone,
    it can be set by the user (at initialization or later via the property).

    Notes
    -----
    ``first_step`` and ``last_step`` is only useful for trajectories that come
    directly from a :class:`asyncmd.mdengine.MDEngine`.
    As soon as the trajectory has been concatenated using MDAnalysis (e.g. with
    the ``TrajectoryConcatenator``) the step information is just the frame
    number in the trajectory part that became first/last frame in the
    concatenated trajectory.
    """

    _CACHE_CLASS_FOR_TYPE: dict[str, type[TrajectoryFunctionValueCache]] = {
           "h5py": TrajectoryFunctionValueCacheInH5PY,
           "npz": TrajectoryFunctionValueCacheInNPZ,
           "memory": TrajectoryFunctionValueCacheInMemory,
    }
    _file_data: _TrajectoryFileData  # type annotation for stuff we set in __new__

    # Note: We want __init__ and __new__ to have the same call signature
    #       (at least for users, __new__ takes `old_workdir`...).
    #       So we will have unused arguments in __init__ (for the stuff we set
    #       in __new__) and we will have unused arguments in __new__ (for the
    #       stuff we set in __init__).
    #       The __new__/__init__ implementation is needed to get the global
    #       trajectory registry to work (to make each traj unique for the same
    #       hash), but pylint can not know that, so
    def __init__(
            self,
            # pylint: disable-next=unused-argument
            trajectory_files: list[str] | str, structure_file: str,
            nstout: int | None = None,
                 ) -> None:
        """
        Initialize a :class:`Trajectory`.

        Parameters
        ----------
        trajectory_files : list[str] or str
            Absolute or relative path(s) to the trajectory file(s),
            e.g. trr, xtc, dcd, ...
        structure_file : str
            Absolute or relative path to the structure file (e.g. tpr, gro).
        nstout : int or None, optional
            The output frequency used when creating the trajectory,
            by default None

        Raises
        ------
        FileNotFoundError
            If the ``trajectory_files`` or the ``structure_file`` are not
            accessible.
        """
        # NOTE: We expect that anything which works for mdanalysis as
        #       traj and struct should also work here as traj and struct
        # NOTE: self._file_data is set in __new__ because we otherwise would:
        #       - calculate the hash twice (need it in __new__),
        #       - sanitize the files twice, but we need to check in __new__
        #         to make pickling work
        #       The _TrajectoryFileData dataclass therefore contains everything
        #       (and only those things) we need in __new__
        # self._file_data
        # properties
        self.nstout = nstout  # use the setter to make basic sanity checks
        # store for all (immutable) properties we read from the trajectory files
        self._property_data: None | _TrajectoryPropertyData = None
        # setup cache for functions applied to this traj
        self._cache = self._setup_cache()
        # Locking mechanism such that only one application of a specific
        # CV func can run at any given time on this trajectory
        self._semaphores_by_func_id: collections.defaultdict[
            str,
            asyncio.BoundedSemaphore,
        ] = collections.defaultdict(asyncio.BoundedSemaphore)

    def __new__(cls,
                trajectory_files: list[str] | str, structure_file: str,
                # (see above note for __init__ why its ok to ignore this)
                # pylint: disable-next:unused-argument
                nstout: int | None = None,
                **kwargs) -> "Trajectory":
        # pylint: disable-next=global-variable-not-assigned
        global _TRAJECTORIES_BY_HASH  # our global traj registry
        # see if old_workdir is given to sanitize file paths
        old_workdir = kwargs.get("old_workdir", None)
        # get cwd to get (and set) it only once for init and unpickle
        current_workdir = os.path.abspath(os.getcwd())
        trajectory_files, structure_file = Trajectory._sanitize_file_paths(
                                            trajectory_files=trajectory_files,
                                            structure_file=structure_file,
                                            current_workdir=current_workdir,
                                            old_workdir=old_workdir,
                                                                           )
        traj_hash = Trajectory._calc_traj_hash(trajectory_files)
        try:
            # see if we (i.e. a traj with the same hash) are already existing
            other_traj = _TRAJECTORIES_BY_HASH[traj_hash]
        except KeyError:
            # not yet in there, so need to create us
            # we just create cls so that we will be "created" by init or
            # unpickled by setstate
            # NOTE: we need to make sure that every attribute we set
            #       below is not overwritten by setstate and/or init!
            obj = super().__new__(cls)
            # we directly set hash, files and friends so we dont recalculate
            # the hash and dont sanitize the file paths twice
            # Note:
            # we remember the current workdir to be able to unpickle as long as
            # either the relpath between traj and old/new workdir does not change
            # or the trajectory did not change its location but we changed workdir
            # (we need the workdir only for the second option)
            obj._file_data = _TrajectoryFileData(
                                    trajectory_files=trajectory_files,
                                    structure_file=structure_file,
                                    workdir=current_workdir,
                                    trajectory_hash=traj_hash,
                                    )
            # and add us to the global trajectory registry
            _TRAJECTORIES_BY_HASH[traj_hash] = obj
            return obj

        # we already exist (a traj object for the same traj files/hash),
        # so return 'ourself'
        # (but make sure that the filepaths match even after a potential
        #  change of workdir)
        other_traj._file_data = _TrajectoryFileData(
                                    trajectory_files=trajectory_files,
                                    structure_file=structure_file,
                                    workdir=current_workdir,
                                    trajectory_hash=traj_hash,
                                    )
        return other_traj

    # def __del__(self):
    # NOTE: Running 'del traj' does not call this function,
    #       it only decreases the reference count by one.
    #       But since we still have the traj in the traj by hash dictionary
    #       i.e. we still have a reference, it will not call __del__ which
    #       is only called when the reference count reaches zero.
    #       So implementing it is quite pointless and misleading!
    #    _forget_trajectory(traj_hash=self.trajectory_hash)

    @classmethod
    def _sanitize_file_paths(cls, *,
                             trajectory_files: list[str] | str,
                             structure_file: str,
                             current_workdir: str,
                             old_workdir: str | None = None,
                             ) -> tuple[list[str], str]:
        """
        Return relpath for all files if no old_workdir is given and the trajectory
        and structure files are accessible.

        If old_workdir is given (and the traj not accessible) it (tries) to find
        the trajs/struct by assuming the files did not change place and we just
        need to add the "path_diff" from old to new workdir to the path, if the
        file is then still not there it raises a FileNotFoundError.

        Note: The file-path treatment here makes it possible to either change
              the workdir of the python session OR change the location of the
              trajectories as as long as the relative path between trajectory
              and python workdir does not change!

        Parameters
        ----------
        trajectory_files : list[str] | str
            Absolute or relative path(s) to the trajectory file(s),
            e.g. trr, xtc, dcd, ...
            Can be one str (one file) or a list of str (multiple traj files).
        structure_file : str
            Absolute or relative path to the structure file (e.g. tpr, gro).
        current_workdir : str
            The current working directory to use for "path_diff" calculations.
        old_workdir : str | None, optional
            The old working directory (e.g. at pickling time), by default None.
            If None, no "path_diff" calculations will be performed, i.e. it is
            assumed the working directory did not change or we are not unpickling.

        Returns
        -------
        tuple[list[str], str]
            trajectory_files, structure_file
            Sanitized file-paths if the files exists, trajectory_files is always
            a list[str], even if it is only one file.

        Raises
        ------
        FileNotFoundError
            When the trajectory or structure files can not be found.
        """
        def sanitize_path(f, pathdiff=None):
            if os.path.isfile(f):
                return os.path.relpath(f)
            if pathdiff is not None:
                f_diff = os.path.join(pathdiff, f)
                if os.path.isfile(f_diff):
                    return os.path.relpath(f_diff)
            # if we get until here we cant find the file
            err_msg = f"File {f} is not accessible"
            err_msg += f" (we also tried {f_diff})." if pathdiff is not None else "."
            raise FileNotFoundError(err_msg)

        if old_workdir is not None:
            path_diff = os.path.relpath(old_workdir, current_workdir)
        else:
            path_diff = None
        if isinstance(trajectory_files, str):
            trajectory_files = [trajectory_files]
        traj_files_sanitized = [sanitize_path(f=traj_f, pathdiff=path_diff)
                                for traj_f in trajectory_files
                                ]
        struct_file_sanitized = sanitize_path(f=structure_file, pathdiff=path_diff)
        return traj_files_sanitized, struct_file_sanitized

    @classmethod
    def _calc_traj_hash(cls, trajectory_files: list[str]) -> int:
        """
        Calculate a hash over the first and last part of the traj files.

        We use it to make sure the cached CV values match the traj.
        Note that we do not include the structure file on purpose because
        that allows for changing .gro <-> .tpr or similar (which we expect to
        not change the calculated CV values).

        Parameters
        ----------
        trajectory_files : list[str]
            Path(s) to the trajectory file(s).

        Returns
        -------
        int
            The hash calculated over the trajectory files.
        """
        # TODO: how much should we read to calculate the hash?
        #      (I [hejung] think the first and last .5 MB are enough)
        data = bytes()
        for traj_f in trajectory_files:
            # data += traj_f.encode("utf-8")  # DONT include filepaths!...
            fsize = os.stat(traj_f).st_size
            data += str(fsize).encode("utf-8")
            if not fsize:
                # Note: we could also just warn as long as we do not do the
                #       negative seek below if filesize == 0. However,
                #       mdanalysis throws errors for empty trajectories anyway
                raise ValueError(f"Trajectory file {traj_f} is of size 0.")
            # read (at most) the first and last 0.5 MB of each file
            max_to_read = min((512, fsize))
            with open(traj_f, "rb") as traj_file:
                # read the first bit of each file
                data += traj_file.read(max_to_read)
                # and read the last bit of each file
                # Note that the last bit potentially overlaps with the first
                traj_file.seek(-max_to_read, io.SEEK_END)
                data += traj_file.read(max_to_read)
        # calculate one hash over all traj_files
        traj_hash = int(hashlib.blake2b(data,
                                        # digest size 8 bytes = 64 bit
                                        # to make sure the hash fits into
                                        # the npz as int64 and not object
                                        digest_size=8).hexdigest(),
                        base=16,
                        )
        return traj_hash

    def _setup_cache(self) -> TrajectoryFunctionValueCache:
        """
        Initialize and return a cache with the cache type/class set by _GLOBALS/config.

        If the initialized cache is empty, this also checks for any npz cache
        files and tries to append them to the new cache (irrespective of the
        cache type).
        """
        cache = self._CACHE_CLASS_FOR_TYPE[
                                _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE]
                                           ](traj_hash=self.trajectory_hash,
                                             traj_files=self.trajectory_files,
                                             )
        # only try to read npz files if our cache is empty and not already npz
        if not cache and _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] != "npz":
            # cache is empty at initialization
            # check if we can find a npz-cache to populate from
            if os.path.isfile(
                TrajectoryFunctionValueCacheInNPZ.get_cache_filename(
                    traj_files=self.trajectory_files
                )
            ):
                logger.info("Initialized %s with an empty cache, but found "
                            "a (probably) matching npz cache file. Populating "
                            "our cache with the values stored there.",
                            self,
                            )
                cache_to_copy = TrajectoryFunctionValueCacheInNPZ(
                                        traj_hash=self.trajectory_hash,
                                        traj_files=self.trajectory_files,
                                                                  )
                for func_id, values in cache_to_copy.items():
                    cache.append(func_id=func_id, values=values)
        return cache

    def update_cache_type(self, copy_content: bool = True,
                          clear_old_cache: bool = False) -> None:
        """
        Update the :class:`TrajectoryFunctionValueCache` this :class:`Trajectory` uses.

        By default the content of the current cache is copied to the new cache.
        See :func:`asyncmd.config.set_trajectory_cache_type` to set the ``cache_type``.
        To clear the old/previously set cache (after copying its values), pass
        ``clear_old_cache=True``.

        Parameters
        ----------
        copy_content : bool, optional
            Whether to copy the current cache content to the new cache,
            by default True
        clear_old_cache : bool, optional
            Whether to clear the old/previously set cache, by default False.
        """
        cache_type = _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE]
        # init the new cache
        cache = self._CACHE_CLASS_FOR_TYPE[cache_type](
                                            traj_hash=self.trajectory_hash,
                                            traj_files=self.trajectory_files,
                                            )
        if copy_content:
            # and copy/append everything from current cache to the new one
            for func_id, values in self._cache.items():
                try:
                    cache.append(func_id=func_id, values=values)
                except ValuesAlreadyStoredError:
                    # if we just initialized a non-empty cache we might already
                    # have some of the values cached there, ignore them
                    pass
        if clear_old_cache:
            self._cache.clear_all_values()
        self._cache = cache

    def clear_all_cache_values(self) -> None:
        """
        Clear all function values cached for this :class:`Trajectory`.

        For file-based caches, this also removes the associated cache files.
        Note that this just calls the underlying :class:`TrajectoryFunctionValueCache`
        classes ``clear_all_values`` method.
        """
        self._cache.clear_all_values()

    def _deregister_h5py_cache(self, h5py_group: "h5py.File | h5py.Group") -> None:
        """
        Deregister the given h5py_cache as a source of cached values.

        Parameters
        ----------
        h5py_cache : h5py.File | h5py.Group
            The h5py_cache to deregister/remove from caching

        Raises
        ------
        RuntimeError
            When the cache type is not a h5py cache and no deregistering is possible.
        """
        if not isinstance(self._cache, TrajectoryFunctionValueCacheInH5PY):
            raise RuntimeError(
                "Can only deregister h5py caches when cache_type is 'h5py'.")
        self._cache.deregister_h5py_cache(h5py_cache=h5py_group)

    def _retrieve_cached_values(self, func_wrapper: "TrajectoryFunctionWrapper",
                                ) -> np.ndarray | None:
        """
        Retrieve values cached for given :class:`TrajectoryFunctionWrapper`.

        Return ``None`` if no values are cached (yet).

        Parameters
        ----------
        func_wrapper : TrajectoryFunctionWrapper
            The TrajectoryFunctionWrapper for which we (try to) retrieve cached values.

        Returns
        -------
        np.ndarray | None
            Cached function values or None if none are found.
        """
        try:
            values = self._cache[func_wrapper.id]
        except KeyError:
            values = None
        return values

    def _register_cached_values(self, values: np.ndarray,
                                func_wrapper: "TrajectoryFunctionWrapper",
                                ) -> None:
        """
        Add values to cache for given TrajectoryFunctionWrapper.

        Parameters
        ----------
        values : np.ndarray
            The values to add.
        func_wrapper : TrajectoryFunctionWrapper
            The TrajectoryFunctionWrapper this values belong to.
        """
        self._cache.append(func_id=func_wrapper.id, values=values)

    def _populate_property_data(self) -> _TrajectoryPropertyData:
        """
        Populate and return cached properties from the underlying trajectory.

        Returns a :class:`_TrajectoryPropertyData` class.
        """
        # create/open a mdanalysis universe to get...
        u = mda.Universe(self.structure_file, *self.trajectory_files)
        # ...the number of frames
        length = len(u.trajectory)
        # ...the first integration step and time
        ts = u.trajectory[0]
        first_step = ts.data.get("step", None)
        first_time = ts.time
        # ...the time diff between subsequent **frames** (not steps)
        dt = ts.dt
        # ...the last integration step and time
        ts = u.trajectory[-1]
        last_step = ts.data.get("step", None)
        last_time = ts.time
        # See if we apply the wraparound issue fix
        # Note: we are using some of the info we just read here (all explicitly passed)!
        if all(
            t.lower().endswith((".xtc", ".trr")) for t in self.trajectory_files
        ):
            first_step, last_step = self._fix_trr_xtc_step_wraparound(
                                        universe=u,
                                        first_time=first_time, last_time=last_time,
                                        first_step=first_step, last_step=last_step,
                                        )
        else:
            # bail out if traj is not an XTC or TRR
            logger.info("%s is not of type XTC or TRR. Not applying "
                        "wraparound fix.", self)
        # make sure the trajectory is closed by MDAnalysis
        u.trajectory.close()
        del u
        # finally populate and return the dataclass with what we just read
        # (and possibly corrected)
        return _TrajectoryPropertyData(
                                length=length, dt=dt,
                                first_time=first_time, last_time=last_time,
                                first_step=first_step, last_step=last_step,
                                )

    def _fix_trr_xtc_step_wraparound(self, *,
                                     universe: mda.Universe,
                                     first_time: float, last_time: float,
                                     first_step: int, last_step: int,
                                     ) -> tuple[int, int]:
        # check/correct for wraparounds in the integration step numbers
        # return (corrected or not) first_step, last_step
        # I.e. it is save to always set first_step, last_step with the return
        # of this method.
        # NOTE: fails if the trajectory has length = 1!
        # NOTE: strictly spoken we should not assume wraparound behavior,
        #       but it seems reasonable for the stepnum,
        #       see e.g. https://www.airs.com/blog/archives/120
        # all times are in pico second (as this is MDAnalysis unit of time)
        # we round integrator_dt and delta_t to precision of
        # 0.000001 ps = 0.001 fs = 1 as
        # we do this to avoid accumulating floating point inaccuracies when
        # dividing the times by integrator_dt, this should be reasonably
        # save for normal MD settings where integrator_dt should be on the
        # order of 1-10 fs
        if (n_frames := len(universe.trajectory)) == 1:
            # bail out if the trajectory has length=1
            # as we can not calculate dt if we only have one frame
            logger.info("%s has only one frame. Can not correct for "
                        "potential wraparound of the integration step.",
                        self)
            return first_step, last_step  # bail out
        # get the time offset for first and last frame, they need to match for
        # our wraparound fix to work
        time_offset = universe.trajectory[0].data.get("time_offset", 0)
        if universe.trajectory[-1].data.get("time_offset", 0) != time_offset:
            logger.info("Time offset of the first and last time in "
                        "%s do not match. Not correcting for potential "
                        "wraparound of the integration step.",
                        self)
            return first_step, last_step  # bail out
        delta_s = last_step - first_step
        delta_t = round(last_time - first_time, ndigits=6)
        # first make sure traj is continuous (i.e. not a concatenation where we
        # carried over the time and step data from the original trajs).
        # Use at most 100 (equally spaced) frames to see if it is continuous.
        skip = n_frames // 100 if n_frames > 100 else 1
        step_diffs = np.diff([ts.data["step"]
                              for ts in universe.trajectory[::skip]]
                             )
        if (first_diff := step_diffs[0]) < 0:
            # we possibly wrapped around at the first step
            first_diff += 2**32
        for diff in step_diffs[1:]:
            if diff != first_diff:
                # bail out because traj is not continuous in time
                logger.debug("%s is not from one continuous propagation, i.e. "
                             "the step difference between subsequent steps is "
                             "not constant. Not applying TRR/XTC step "
                             "wraparound fix and using step as read from the "
                             "underlying trajectory.",
                             self)
            return first_step, last_step
        # now the actual fix
        if delta_s:  # delta_s != 0
            if delta_s > 0:
                # both (last and first) wrapped around the same number of times
                integrator_dt = round(delta_t / delta_s, ndigits=6)
            else:  # delta_s < 0
                # last wrapped one time more than first
                integrator_dt = round(delta_t / (delta_s + 2**32), ndigits=6)
            # NOTE: should we round or floor? I (hejung) think round is what we
            #       want, it will get us to the nearest int, which is good if
            #       we e.g. have 0.99999999999 instead of 1
            first_step = round((first_time - time_offset) / integrator_dt)
            last_step = round((last_time - time_offset) / integrator_dt)
            return first_step, last_step
        # delta_s == 0
        # can only end up here if we have more than one frame in trajectory
        # **and** the first and last frame have the same integration step
        # which should be very rare and we can not correct anyway as the
        # trajectory can not be from a continuous propagation, so we can not
        # end up here at all?
        raise RuntimeError("This should not be possible?!")

    def __len__(self) -> int:
        """
        Return the number of frames in the trajectory.

        Returns
        -------
        int
            The number of frames in the trajectory.
        """
        if self._property_data is None:
            self._property_data = self._populate_property_data()
        return self._property_data.length

    def __repr__(self) -> str:
        if len(self.trajectory_files) == 1:
            return (f"Trajectory(trajectory_files={self.trajectory_files[0]},"
                    + f" structure_file={self.structure_file})"
                    )
        return (f"Trajectory(trajectory_files={self.trajectory_files},"
                + f" structure_file={self.structure_file})"
                )

    def __hash__(self) -> int:
        return self.trajectory_hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            # if its not a trajectory it cant be equal
            return False
        if self.trajectory_hash != other.trajectory_hash:
            # if it has a different hash it cant be equal
            return False

        # if we got until here the two trajectories are equal
        return True

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @property
    def structure_file(self) -> str:
        """Return relative path to the structure file."""
        return self._file_data.structure_file

    @property
    def trajectory_files(self) -> list[str]:
        """Return relative path to the trajectory files."""
        return self._file_data.trajectory_files

    @property
    def trajectory_hash(self) -> int:
        """Return hash over the trajectory files"""
        return self._file_data.trajectory_hash

    @property
    def nstout(self) -> int | None:
        """Output frequency between subsequent frames in integration steps."""
        return self._nstout

    @nstout.setter
    def nstout(self, val: int | None) -> None:
        if val is not None:
            # ensure that it is an int
            val = int(val)
        # enable setting to None
        self._nstout = val

    @property
    def first_step(self) -> int | None:
        """Return the integration step of the first frame in the trajectory."""
        if self._property_data is None:
            self._property_data = self._populate_property_data()
        return self._property_data.first_step

    @property
    def last_step(self) -> int | None:
        """Return the integration step of the last frame in the trajectory."""
        if self._property_data is None:
            self._property_data = self._populate_property_data()
        return self._property_data.last_step

    @property
    def dt(self) -> float:
        """The time interval between subsequent *frames* (not steps) in ps."""
        if self._property_data is None:
            self._property_data = self._populate_property_data()
        return self._property_data.dt

    @property
    def first_time(self) -> float:
        """Return the integration timestep of the first frame in ps."""
        if self._property_data is None:
            self._property_data = self._populate_property_data()
        return self._property_data.first_time

    @property
    def last_time(self) -> float:
        """Return the integration timestep of the last frame in ps."""
        if self._property_data is None:
            self._property_data = self._populate_property_data()
        return self._property_data.last_time

    def __getstate__(self) -> dict[str, typing.Any]:
        # enable pickling of Trajectory
        # this should make it possible to pass it into a ProcessPoolExecutor
        # and lets us calculate TrajectoryFunction values asynchronously
        state = self.__dict__.copy()
        # special handling for case of function values cached in memory
        if isinstance(self._cache, TrajectoryFunctionValueCacheInMemory):
            # write it to npz so we can unpickle with values for any cache type
            # (if we unpickle with an empty cache we will [try to] read the npz)
            npz_cache = TrajectoryFunctionValueCacheInNPZ(
                                    traj_hash=self.trajectory_hash,
                                    traj_files=self.trajectory_files,
                                                          )
            for func_id, values in self._cache.items():
                try:
                    npz_cache.append(func_id=func_id, values=values)
                except ValuesAlreadyStoredError:
                    # ignore if we already have them
                    pass
        state["_cache"] = None
        state["_semaphores_by_func_id"] = collections.defaultdict(
                                                    asyncio.BoundedSemaphore
                                                                  )
        return state

    def __setstate__(self, d: dict) -> None:
        # remove the attributes we set in __new__ from dict
        # (otherwise we would overwrite what we set in __new__)
        del d["_file_data"]
        # now we can update without overwriting what we set in __new__
        self.__dict__.update(d)
        # and finally setup the cache according to what the global config says
        self._cache = self._setup_cache()

    def __getnewargs_ex__(self) -> tuple[tuple, dict[str, typing.Any]]:
        # new needs the trajectory_files to be able to calculate the traj_hash
        # and since we want __new__ to have the same call signature as __init__
        # we also add all the init args here too
        return ((), {"trajectory_files": self.trajectory_files,
                     "structure_file": self.structure_file,
                     "nstout": self.nstout,
                     "old_workdir": self._file_data.workdir,
                     })
