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
import io
import os
import copy
import typing
import asyncio
import hashlib
import logging
import zipfile
import collections
import numpy as np
import MDAnalysis as mda


from .._config import _GLOBALS


logger = logging.getLogger(__name__)


class Trajectory:
    """
    Represent a trajectory.

    Keep track of the paths of the trajectory and the structure files.
    Caches values for (wrapped) functions acting on the trajectory.
    Supports pickling and unpickling with the cached values restored, the
    values will be written to a hidden numpy npz file next to the trajectory.
    Supports equality checks with other :class:`Trajectory`.
    Also makes available (and caches) a number of useful attributes, e.g.
    ``first_step`` and ``last_step`` (the first and last intergation step in
    the trajectory), ``dt``, ``first_time``, ``last_time``,
    ``length`` (in frames) and ``nstout``.

    Notes
    -----
    ``first_step`` and ``last_step`` is only useful for trajectories that come
    directly from a :class:`asyncmd.mdengine.MDEngine`.
    As soon as the trajecory has been concatenated using MDAnalysis (e.g. with
    the ``TrajectoryConcatenator``) the step information is just the frame
    number in the trajectory part that became first/last frame in the
    concatenated trajectory.
    """

    def __init__(self, trajectory_files: typing.Union[list[str],str], structure_file: str,
                 nstout: typing.Optional[int] = None,
                 cache_type: typing.Optional[str] = None,
                 **kwargs):
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
        cache_type : str or None, optional
            The cache type for the CV values cached for this trajectory,
            must be one of 'h5py', 'npz' or 'memory'.
            If None we will use 'h5py' if a h5py cache has been registered and
            if not fallback to 'npz'.
            See also the ``asyncmd.config.register_h5py_cache()`` function.

        Raises
        ------
        FileNotFoundError
            If the ``trajectory_files`` or the ``structure_file`` are not
            accessible.
        """
        # NOTE: we assume tra = trr and struct = tpr
        #       but we also expect that anything which works for mdanalysis as
        #       tra and struct should also work here as tra and struct

        # TODO: currently we do not use kwargs?!
        #dval = object()
        #for kwarg, value in kwargs.items():
        #    cval = getattr(self, kwarg, dval)
        #    if cval is not dval:
        #        if isinstance(value, type(cval)):
        #            # value is of same type as default so set it
        #            setattr(self, kwarg, value)
        #        else:
        #            logger.warn(f"Setting attribute {kwarg} with "
        #                        + f"mismatching type ({type(value)}). "
        #                        + f" Default type is {type(cval)}."
        #                        )
        if isinstance(trajectory_files, str):
            trajectory_files = [trajectory_files]
        self._trajectory_files = []
        for traj_f in trajectory_files:
            if os.path.isfile(traj_f):
                self._trajectory_files += [os.path.abspath(traj_f)]
            else:
                raise FileNotFoundError(f"Trajectory file ({traj_f}) must be "
                                        + "accessible.")
        if os.path.isfile(structure_file):
            self._structure_file = os.path.abspath(structure_file)
        else:
            raise FileNotFoundError(f"structure_file ({structure_file}) must be "
                                    + "accessible.")
        # calculate a hash over the first part of the traj file
        # (we use it to make sure the cached CV values match the traj)
        # note that we do not include the structure file on purpose because
        # that allows for changing .gro <-> .tpr or similar
        # (which we expect to not change the calculated CV values)
        # TODO/FIXME?: What happens if we append to an existing traj?
        #              I [hejung] think this will only change the end of
        #              the traj, so we would need to remember the len/size
        #              of the traj or additionally hash the end of it?!
        # TODO: how much should we read?
        #      (I [hejung] think the first and last 2.5 MB are enough for sure)
        data = bytes()
        for traj_f in self._trajectory_files:
            with open(traj_f, "rb") as traj_file:
                # read the first 2.5 MB of each file
                data += traj_file.read(2560)
                # and read the last 2.5 MB of each file
                traj_file.seek(-2560, io.SEEK_END)
                data += traj_file.read(2560)
        # calculate one hash over all traj_files
        self._traj_hash = int(hashlib.blake2b(data,
                                              # digest size 8 bytes = 64 bit
                                              # to make sure the hash fits into
                                              # the npz as int64 and not object
                                              digest_size=8).hexdigest(),
                              base=16,
                              )
        # properties
        self.nstout = nstout  # use the setter to make basic sanity checks
        self._len = None
        self._first_step = None
        self._last_step = None
        self._dt = None
        self._first_time = None
        self._last_time = None
        # stuff for caching of functions applied to this traj
        self._memory_cache = None
        self._npz_cache = None
        self._h5py_cache = None
        self._cache_type = None
        # remember if we use the global default value,
        # if yes we use the (possibly changed) global default when unpickling
        self._using_default_cache_type = True
        # use our property logic for checking the value
        self.cache_type = cache_type
        # Locking mechanism such that only one application of a specific
        # CV func can run at any given time on this trajectory
        self._semaphores_by_func_id = collections.defaultdict(
                                                    asyncio.BoundedSemaphore
                                                              )

    @property
    def cache_type(self):
        """
        String indicating the currently used cache type. Can also be (re)set.
        """
        return copy.copy(self._cache_type)

    @cache_type.setter
    def cache_type(self, value: typing.Optional[str]):
        """
        Set the cache type.

        Parameters
        ----------
        value : str or None
            Either a string indicating the type or None to choose the preferred
            cache type from the available ones.
            If a string it must be one of 'h5py', 'npz' or 'memory'.

        Raises
        ------
        ValueError
            Raised if value is not one of the available cache types.
        """
        if value is None:
            use_default_cache_type = True
            # find preferred cache type that is available
            try:
                value = _GLOBALS["TRAJECTORY_FUNCTION_CACHE_TYPE"]
            except KeyError:
                # no default cache type set
                # default to numpy npz
                value = "npz"
        else:
            use_default_cache_type = False
        value = value.lower()
        allowed_values = ["h5py", "npz", "memory"]
        if value not in allowed_values:
            raise ValueError("Given cache type must be `None` or one of "
                             + f"{allowed_values}. Was: {value}.")
        self._cache_type = value
        self._using_default_cache_type = use_default_cache_type
        self._setup_cache()

    def _setup_cache(self) -> None:
        # set up the cache indicated by self.cache_type and all others to None
        # also makes sure that all previously cached values are transfered
        # to the newly setup cache
        # NOTE: we setup an npz cache to see if there are any saved values
        #       that we would want to add to the newly setup cache
        #       We do this because upon pickling we save everything to npz
        # Note that we can just set self._npz to this cache because it is
        # stateless (in the sense that if it existed it be exactly the same)
        self._npz_cache = TrajectoryFunctionValueCacheNPZ(
                                        fname_trajs=self.trajectory_files,
                                        hash_traj=self._traj_hash,
                                                          )
        if self._cache_type == "memory":
            if self._memory_cache is None:
                self._memory_cache = TrajectoryFunctionValueCacheMEMORY()
            else:
                # we already have a mem cache so just try to use it
                pass
            if self._h5py_cache is not None:
                self._cache_content_to_new_cache(
                                        old_cache=self._h5py_cache,
                                        new_cache=self._memory_cache,
                                                 )
                self._h5py_cache = None
            self._cache_content_to_new_cache(
                                        old_cache=self._npz_cache,
                                        new_cache=self._memory_cache,
                                             )
            self._npz_cache = None
        elif self._cache_type == "h5py":
            try:
                h5py_cache = _GLOBALS["H5PY_CACHE"]
            except KeyError:
                raise ValueError(
                    "No h5py cache file registered yet. Try calling "
                    + "``asyncmd.config.register_h5py_cache_file()``"
                    + " with the appropriate arguments first")
            if self._h5py_cache is None:
                # dont have one yet so setup the cache
                self._h5py_cache = TrajectoryFunctionValueCacheH5PY(
                                                h5py_cache=h5py_cache,
                                                hash_traj=self._traj_hash,
                                                                    )
            else:
                # we already have a h5py cache...
                if self._h5py_cache.h5py_cache is h5py_cache:
                    # and it is in the same file/group location
                    # so we do nothing but making sure that all values from
                    # other caches are transfered
                    pass
                else:
                    # lets copy the stuff from the old to the new h5py cache
                    old_h5py_cache = self._h5py_cache
                    self._h5py_cache = TrajectoryFunctionValueCacheH5PY(
                                                h5py_cache=h5py_cache,
                                                hash_traj=self._traj_hash,
                                                                    )
                    self._cache_content_to_new_cache(
                                        old_cache=old_h5py_cache,
                                        new_cache=self._h5py_cache,
                                                     )
            # transfer all values from other cache types and empty them
            if self._memory_cache is not None:
                self._cache_content_to_new_cache(
                                        old_cache=self._memory_cache,
                                        new_cache=self._h5py_cache,
                                                 )
                self._memory_cache = None
            self._cache_content_to_new_cache(
                                        old_cache=self._npz_cache,
                                        new_cache=self._h5py_cache,
                                             )
            self._npz_cache = None
        elif self._cache_type == "npz":
            if self._h5py_cache is not None:
                self._cache_content_to_new_cache(
                                        old_cache=self._h5py_cache,
                                        new_cache=self._npz_cache,
                                                 )
                self._h5py_cache = None
            if self._memory_cache is not None:
                self._cache_content_to_new_cache(
                                        old_cache=self._memory_cache,
                                        new_cache=self._npz_cache,
                                                 )
                self._memory_cache = None
        else:
            raise RuntimeError("This should never happen. self._cache_type "
                               + "must be one of 'memory', 'h5py', 'npz' when "
                               + "self._setup_cache is called. "
                               + f"Was {self._cache_type}.")

    def _populate_properties(self) -> None:
        """
        Populate cached properties from the underlying trajectory.
        """
        # create/open a mdanalysis universe to get...
        u = mda.Universe(self.structure_file, *self.trajectory_files,
                         tpr_resid_from_one=True)
        # ...the number of frames
        self._len = len(u.trajectory)
        # ...the first integration step and time
        ts = u.trajectory[0]
        # FIXME/NOTE: next line works only(?) for trr and xtc
        self._first_step = ts.data["step"]
        self._first_time = ts.time
        time_offset = ts.data.get("time_offset", 0)  # need offset below
        # ...the time diff between subsequent **frames** (not steps)
        self._dt = ts.dt
        # ...the last integration step and time
        ts = u.trajectory[-1]
        # FIXME/NOTE: next line works only(?) for trr and xtc
        self._last_step = ts.data["step"]
        self._last_time = ts.time
        # make sure first and last time have the same offset, if they
        # do not our calculation below will be wrong!
        # Also it probably means someone mixed trajectory parts
        # from different engine runs, i.e. not good anyway and we warn :)
        if ts.data.get("time_offset", 0) != time_offset:
            logger.warning(f"Time offset of the first and last time in {self}"
                           + " do not match. Not correcting for potential "
                           + "wraparound of the integration step.")
            return  # bail out!
        if self.trajectory_files[0].lower().endswith((".xtc", ".trr")):
            # bail out if traj is not an XTC or TRR
            logger.info(f"{self} is not of type XTC or TRR. Not applying "
                        + "wraparound fix.")
            return
        # check/correct for wraparounds in the integration step numbers
        # NOTE: strictly spoken we should not assume wraparound behavior,
        #       but it seems reasonable for the stepnum,
        #       see e.g. https://www.airs.com/blog/archives/120
        delta_s = self._last_step - self._first_step
        delta_t = self._last_time - self._first_time
        # NOTE: should we round or floor? I (hejung) think round is what we
        #       want, it will get us to the nearest int, which is good if
        #       we e.g. have 0.99999999999 instead of 1
        if delta_s != 0:
            # Note: times are in pico second
            # we round integrator_dt to precision 0.000001 ps = 0.001 fs = 1 as
            # we do this to avoid accumulating floating point inaccuracies when
            # dividing the times by integrator_dt, this should be reasonably
            # save for normal MD settings where integrator_dt should be on the
            # order of 1-10 fs
            if delta_s > 0:
                # both (last and first) wrapped around the same number of times
                integrator_dt = round(delta_t / delta_s, ndigits=6)
            else:  # delta_s < 0
                # last wrapped one time more than first
                integrator_dt = round(delta_t / (delta_s + 2**32), ndigits=6)
            first_step = round((self._first_time - time_offset) / integrator_dt)
            last_step = round((self._last_time - time_offset) / integrator_dt)
            self._first_step = first_step
            self._last_step = last_step
        else:  # delta_s == 0
            # no way of finding the integrator_dt so we just use last and first
            # steps as read from the underlying traj
            logger.info("Can not correct for potential wraparound of the "
                        + f"integration step because {self} has only one frame.")

    def __len__(self) -> int:
        """
        Return the number of frames in the trajectory.

        Returns
        -------
        int
            The number of frames in the trajectory.
        """
        if self._len is None:
            self._populate_properties()
        return self._len

    def __repr__(self) -> str:
        if len(self.trajectory_files) == 1:
            return (f"Trajectory(trajectory_files={self.trajectory_files[0]},"
                    + f" structure_file={self.structure_file})"
                    )
        return (f"Trajectory(trajectory_files={self.trajectory_files},"
                + f" structure_file={self.structure_file})"
                )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            # if its not a trajectory it cant be equal
            return False
        if self.trajectory_hash != other.trajectory_hash:
            # if it has a different hash it cant be equal
            return False
        # if they dont have the same number of trajecory files they can not be
        # the same
        if len(self.trajectory_files) != len(other.trajectory_files):
            return False
        # they might be the same file (bitwise) but at different locations in
        # the FS, then they are not equal (at least for out purposes)
        for traj_f_self, traj_f_other in zip(self.trajectory_files,
                                             other.trajectory_files):
            if traj_f_self != traj_f_other:
                return False
        if len(self) != len(other):
            # since we only hash the beginning of the file(s) they might
            # be the same start but different number of frames/end
            # so we check if they have the same length
            return False
        # NOTE: we allow the structure file to change
        #       this way we consider e.g. the same traj with a gro and with a
        #       tpr (or whatever) as structure files as equal
        #       The rationale is that MDAnalysis will ocmplain if the struct
        #       and the traj dont match and we e.g. reuse cached values without
        #       checking the structure file anyway
        #if self.structure_file != other.structure_file:
        #    return False
        # TODO: check for cached CV values? I (hejung) think it does not really
        #       make sense...

        # if we got until here the two trajs are equal
        return True

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other=other)

    @property
    def structure_file(self) -> str:
        """Return absolute path to the structure file."""
        return copy.copy(self._structure_file)

    @property
    def trajectory_files(self) -> str:
        """Return absolute path to the trajectory files."""
        return copy.copy(self._trajectory_files)

    @property
    def trajectory_hash(self) -> int:
        """Return hash over the trajecory files"""
        return copy.copy(self._traj_hash)

    @property
    def nstout(self) -> typing.Union[int, None]:
        """Output frequency between subsequent frames in integration steps."""
        return self._nstout

    @nstout.setter
    def nstout(self, val: typing.Union[int, None]) -> None:
        if val is not None:
            # ensure that it is an int
            val = int(val)
        # enable setting to None
        self._nstout = val

    @property
    def first_step(self) -> int:
        """Return the integration step of the first frame in the trajectory."""
        if self._first_step is None:
            self._populate_properties()
        return self._first_step

    @property
    def last_step(self) -> int:
        """Return the integration step of the last frame in the trajectory."""
        if self._last_step is None:
            self._populate_properties()
        return self._last_step

    @property
    def dt(self) -> float:
        """The time intervall between subsequent *frames* (not steps) in ps."""
        if self._dt is None:
            self._populate_properties()
        return self._dt

    @property
    def first_time(self) -> float:
        """Return the integration timestep of the first frame in ps."""
        if self._first_time is None:
            self._populate_properties()
        return self._first_time

    @property
    def last_time(self) -> float:
        """Return the integration timestep of the last frame in ps."""
        if self._last_time is None:
            self._populate_properties()
        return self._last_time

    async def _apply_wrapped_func(self, func_id, wrapped_func):
        async with self._semaphores_by_func_id[func_id]:
            # sort out which cache we use
            # NOTE: only one cache should ever be not None, so order should not
            #       matter here
            #       anyway I (hejung) think this order is even what we want:
            #       1.) use h5py cache if registered
            #       2.) use npz cache (the default since h5py is not registered
            #                          if not set by the user)
            #       3.) use memory/local cache (only if set on traj creation
            #                                   or if set as default cache)
            if self._h5py_cache is not None:
                return await self._apply_wrapped_func_cached(
                                                    func_id=func_id,
                                                    wrapped_func=wrapped_func,
                                                    cache=self._h5py_cache,
                                                             )
            if self._npz_cache is not None:
                return await self._apply_wrapped_func_cached(
                                                    func_id=func_id,
                                                    wrapped_func=wrapped_func,
                                                    cache=self._npz_cache
                                                             )
            if self._memory_cache is not None:
                return await self._apply_wrapped_func_cached(
                                                    func_id=func_id,
                                                    wrapped_func=wrapped_func,
                                                    cache=self._memory_cache,
                                                             )
            # if we get until here we have no cache!
            logger.warning(f"No cache associated with {self}. Returning "
                           + "calculated function values anyway but no caching"
                           + "can/will be performed!"
                           )
            return await wrapped_func.get_values_for_trajectory(self)

    async def _apply_wrapped_func_cached(
                            self, func_id: str, wrapped_func,
                            cache: collections.abc.Mapping[str, np.ndarray],
                                         ):
        try:
            # see if it is in cache
            return copy.copy(cache[func_id])
        except KeyError:
            # if not calculate, store and return
            # send function application to seperate process and wait
            # until it finishes
            vals = await wrapped_func.get_values_for_trajectory(self)
            cache.append(func_id=func_id, vals=vals)
            return vals

    def _cache_content_to_new_cache(
                        self,
                        old_cache: collections.abc.Mapping[str, np.ndarray],
                        new_cache: collections.abc.Mapping[str, np.ndarray],
                                    ):
        for func_id, values in old_cache.items():
            if func_id in new_cache:
                continue  # dont try to add what is already in there
            new_cache.append(func_id=func_id, vals=values)

    def __getstate__(self):
        # enable pickling of Trajectory
        # this should make it possible to pass it into a ProcessPoolExecutor
        # and lets us calculate TrajectoryFunction values asyncronously
        state = self.__dict__.copy()
        # NOTE: we always save to npz here and then we check for npz always
        #       when initializing a `new` trajectory and add all values to
        #       the then preferred cache
        if self._npz_cache is None:
            self._npz_cache = TrajectoryFunctionValueCacheNPZ(
                                        fname_trajs=self.trajectory_files,
                                        hash_traj=self._traj_hash,
                                                             )
            if self._memory_cache is not None:
                self._cache_content_to_new_cache(old_cache=self._memory_cache,
                                                 new_cache=self._npz_cache,
                                                 )
            if self._h5py_cache is not None:
                self._cache_content_to_new_cache(old_cache=self._h5py_cache,
                                                 new_cache=self._npz_cache,
                                                 )
        state["_h5py_cache"] = None
        state["_npz_cache"] = None
        state["_memory_cache"] = None
        state["_semaphores_by_func_id"] = collections.defaultdict(
                                                    asyncio.BoundedSemaphore
                                                                  )
        return state

    def __setstate__(self, d: dict):
        self.__dict__ = d
        # sort out which cache we were using (and which we will use now)
        if self._using_default_cache_type:
            # if we were using the global default when pickling use it now too
            # Note that this will raise the ValueError from _setup_cache if
            # no h5py cache has been registered but it is set as default
            # (which is intended because it is the same behavior as when
            #  initializing a new trajectory in the same situation)
            self.cache_type = None
        if self.cache_type == "h5py":
            # make sure h5py cache is set before trying to unpickle with it
            try:
                _ = _GLOBALS["H5PY_CACHE"]
            except KeyError:
                # this will (probably) fallback to npz but I (hejung) think it
                # is nice if we use the possibly set global default?
                # Note that this will not err but just emit the warning to log
                # when we change the cache but it will err when the gloabal
                # default cache is set to h5py (as above)
                logger.warning(f"Trying to unpickle {self} with cache_type "
                               + "'h5py' not possible without a registered "
                               + "cache. Falling back to global default type."
                               + "See 'asyncmd.config.register_h5py_cache' and"
                               + " 'asyncmd.config.set_default_cache_type'."
                               )
                self.cache_type = None
        # and setup the cache
        self._setup_cache()


class TrajectoryFunctionValueCacheMEMORY(collections.abc.Mapping):
    """
    Interface for caching trajectory function values in memory in a dict.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a `TrajectoryFunctionValueCacheMEMORY`."""
        self._func_values_by_id = {}

    def __len__(self) -> int:
        return len(self._func_values_by_id)

    def __iter__(self):
        return self._func_values_by_id.__iter__()

    def __getitem__(self, key: str) -> np.ndarray:
        if not isinstance(key, str):
            raise TypeError("Keys must be of type str.")
        return self._func_values_by_id[key]

    def append(self, func_id: str, vals: np.ndarray) -> None:
        if not isinstance(func_id, str):
            raise TypeError("func_id must be of type str.")
        if func_id in self._func_values_by_id:
            # first check if it already in there
            raise ValueError("There are already values stored for func_id "
                             + f"{func_id}. Changing the stored values is not "
                             + "supported.")
        self._func_values_by_id[func_id] = vals


class TrajectoryFunctionValueCacheNPZ(collections.abc.Mapping):
    """
    Interface for caching trajectory function values in a numpy npz file.

    Drop-in replacement for the dictionary that is used for in-memory caching.
    """

    _hash_traj_npz_key = "hash_of_trajs"  # key of hash_traj in npz file

    # NOTE: this is written with the assumption that stored trajectories are
    #       immutable (except for adding additional stored function values)
    #       but we assume that the actual underlying trajectory stays the same,
    #       i.e. it is not extended after first storing it

    # NOTE: npz appending inspired by: https://stackoverflow.com/a/66618141

    # NOTE/FIXME: It would be nice to use the MAX_FILES_OPEN semaphore
    #    but then we need async/await and then we need to go to a 'create'
    #    classmethod that is async and required for initialization
    #    (because __init__ cant be async)
    #    but since we (have to) open the npz file in the other magic methods
    #    too it does not really matter (as they can not be async either)?
    # ...and as we also leave some room for non-semaphored file openings anyway

    def __init__(self, fname_trajs: str, hash_traj: int) -> None:
        """
        Initialize a `TrajectoryFunctionValueCacheNPZ`.

        Parameters
        ----------
        fname_trajs : str
            Absolute filenames to the trajectories for which we cache CV values.
        hash_traj : int
            Hash over the first part of the trajectory file,
            used to make sure we cache only for the right trajectory
            (and not any trajectories with the same filename).
        """
        self.fname_npz = self._get_cache_filename(fname_trajs=fname_trajs,
                                                  trajectory_hash=hash_traj,
                                                  )
        self._hash_traj = hash_traj
        self._func_ids = []
        # sort out if we have an associated npz file already
        # and if it is from/for the "right" trajectory file
        self._ensure_consistent_npz()

    def _ensure_consistent_npz(self):
        # next line makes sure we only remember func_ids from the current npz
        self._func_ids = []
        if not os.path.isfile(self.fname_npz):
            # no npz so nothing to do except making sure we have no func_ids
            return
        existing_npz_matches = False
        with np.load(self.fname_npz, allow_pickle=False) as npzfile:
            try:
                saved_hash_traj = npzfile[self._hash_traj_npz_key][0]
            except KeyError:
                # we probably tripped over an old formatted npz
                # so we will just rewrite it completely with hash
                pass
            else:
                # old hash found, lets compare the two hashes
                existing_npz_matches = (self._hash_traj == saved_hash_traj)
                if existing_npz_matches:
                    # if they do populate self with the func_ids we have
                    # cached values for
                    for k in npzfile.keys():
                        if k != self._hash_traj_npz_key:
                            self._func_ids.append(str(k))
        # now if the old npz did not match we should remove it
        # then we will rewrite it with the first cached CV values
        if not existing_npz_matches:
            logger.debug(f"Found existing npz file ({self.fname_npz}) but the"
                         + " trajectory hash does not match."
                         + " Recreating the npz cache from scratch.")
            os.unlink(self.fname_npz)

    @classmethod
    def _get_cache_filename(cls, fname_trajs: list[str],
                            trajectory_hash: int) -> str:
        """
        Construct cachefilename from trajectory fname.

        Parameters
        ----------
        fname_trajs : list[str]
            Absolute path to the trajectory for which we cache.
        trajectory_hash : int
            Hash of the trajectory (files).

        Returns
        -------
        str
            Absolute path to the cachefile associated with trajectory.
        """
        head, tail = os.path.split(fname_trajs[0])
        hash_part = str(trajectory_hash)[:5]
        return os.path.join(head, f".{tail}_{hash_part}_asyncmd_cv_cache.npz")

    def __len__(self) -> int:
        return len(self._func_ids)

    def __iter__(self):
        for func_id in self._func_ids:
            yield func_id

    def __getitem__(self, key: str) -> np.ndarray:
        if not isinstance(key, str):
            raise TypeError("Keys must be of type str.")
        if key in self._func_ids:
            with np.load(self.fname_npz, allow_pickle=False) as npzfile:
                return npzfile[key]
        else:
            raise KeyError(f"No values for {key} cached (yet).")

    def append(self, func_id: str, vals: np.ndarray) -> None:
        """
        Append values for given func_id.

        Parameters
        ----------
        func_id : str
            Function identifier.
        vals : np.ndarray
            Values of application of function with given func_id.

        Raises
        ------
        TypeError
            If ``func_id`` is not a string.
        ValueError
            If there are already values stored for ``func_id`` in self.
        """
        if not isinstance(func_id, str):
            raise TypeError("func_id must be of type str.")
        if func_id in self._func_ids:
            # first check if it already in there
            raise ValueError("There are already values stored for func_id "
                             + f"{func_id}. Changing the stored values is not "
                             + "supported.")
        if len(self) == 0:
            # these are the first cached CV values for this traj
            # so we just create the (empty) npz file
            np.savez(self.fname_npz)
            # and write the trajectory hash
            self._append_data_to_npz(name=self._hash_traj_npz_key,
                                     value=np.array([self._hash_traj]),
                                     )
        # now we can append either way
        # either already something cached, or freshly created empty file
        self._append_data_to_npz(name=func_id, value=vals)
        # add func_id to list of func_ids that we know are cached in npz
        self._func_ids.append(func_id)

    def _append_data_to_npz(self, name: str, value: np.ndarray) -> None:
        # npz files are just zipped together collections of npy files
        # so we just make a npy file saved into a BytesIO and then write that
        # to the end of the npz file
        bio = io.BytesIO()
        np.save(bio, value)
        with zipfile.ZipFile(file=self.fname_npz,
                             mode="a",  # append!
                             # uncompressed (but) zip archive member
                             compression=zipfile.ZIP_STORED,
                             ) as zfile:
            zfile.writestr(f"{name}.npy", data=bio.getvalue())


class TrajectoryFunctionValueCacheH5PY(collections.abc.Mapping):
    """
    Interface for caching trajectory function values in a given h5py group.

    Drop-in replacement for the dictionary that is used for in-memory caching.
    """

    # NOTE: this is written with the assumption that stored trajectories are
    #       immutable (except for adding additional stored function values)
    #       but we assume that the actual underlying trajectory stays the same,
    #       i.e. it is not extended after first storing it

    def __init__(self, h5py_cache, hash_traj: int):
        self.h5py_cache = h5py_cache
        self._hash_traj = hash_traj
        self._h5py_paths = {"ids": "FunctionIDs",
                            "vals": "FunctionValues"
                            }
        self._root_grp = h5py_cache.require_group(
                                            "asyncmd/"
                                            + "TrajectoryFunctionValueCache/"
                                            + f"{self._hash_traj}"
                                                  )
        self._ids_grp = self._root_grp.require_group(self._h5py_paths["ids"])
        self._vals_grp = self._root_grp.require_group(self._h5py_paths["vals"])

    def __len__(self):
        return len(self._ids_grp.keys())

    def __iter__(self):
        for idx in range(len(self)):
            yield self._ids_grp[str(idx)].asstr()[()]

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("Keys must be of type str.")
        for idx, k_val in enumerate(self):
            if key == k_val:
                return self._vals_grp[str(idx)][:]
        # if we got until here the key is not in there
        raise KeyError("Key not found.")

    def append(self, func_id, vals):
        if not isinstance(func_id, str):
            raise TypeError("Keys (func_id) must be of type str.")
        if func_id in self:
            raise ValueError("There are already values stored for func_id "
                             + f"{func_id}. Changing the stored values is not "
                             + "supported.")
        # TODO: do we also want to check vals for type?
        name = str(len(self))
        _ = self._ids_grp.create_dataset(name, data=func_id)
        _ = self._vals_grp.create_dataset(name, data=vals)
