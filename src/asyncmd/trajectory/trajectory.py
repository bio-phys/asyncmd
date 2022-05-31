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
import logging
import zipfile
import collections
import numpy as np
import MDAnalysis as mda


logger = logging.getLogger(__name__)


class Trajectory:
    """
    Represent a trajectory.

    Keep track of the paths of the trajectory and the structure file.
    Caches values for (wrapped) functions acting on the trajectory.
    Also makes available (and caches) a number of useful attributes, e.g.
    ``first_step`` and ``last_step`` (the first and last intergation step in
    the trajectory), ``dt``, ``first_time``, ``last_time``,
    ``length`` (in frames) and ``nstout``.

    NOTE: first_step and last_step is only useful for trajectories that come
          directly from a :class:`asyncmd.mdengine.MDEngine`.
          As soon as the trajecory has been concatenated using MDAnalysis
          (i.e. the `TrajectoryConcatenator`) the step information is just the
          frame number in the trajectory part that became first/last frame in
          the concatenated trajectory.
    """

    def __init__(self, trajectory_file: str, structure_file: str,
                 nstout: typing.Optional[int] = None,
                 npz_cache_file: bool = True,
                 **kwargs):
        """
        Initialize a :class:`Trajectory`.

        Parameters
        ----------
        trajectory_file : str
            Absolute or relative path to the trajectory file (e.g. trr, xtc).
        structure_file : str
            Absolute or relative path to the structure file (e.g. tpr, gro).
        nstout : int or None, optional
            The output frequency used when creating the trajectory,
            by default None
        npz_cache_file : bool
            Wheter we should write the cached CV values to a (hidden) numpy npz
            archive file, by default True.
            Note: If ``False`` and no h5py cache is attached the values will be
            cached in memory.

        Raises
        ------
        ValueError
            If the ``trajectory_file`` or the ``structure_file`` are not
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
        if os.path.isfile(trajectory_file):
            self._trajectory_file = os.path.abspath(trajectory_file)
        else:
            raise ValueError(f"trajectory_file ({trajectory_file}) must be "
                             + "accessible.")
        if os.path.isfile(structure_file):
            self._structure_file = os.path.abspath(structure_file)
        else:
            raise ValueError(f"structure_file ({structure_file}) must be "
                             + "accessible.")
        # properties
        self.nstout = nstout  # use the setter to make basic sanity checks
        self._len = None
        self._first_step = None
        self._last_step = None
        self._dt = None
        self._first_time = None
        self._last_time = None
        # stuff for caching of functions applied to this traj
        self._func_values_by_id = {}
        if npz_cache_file:
            self._npz_cache = TrajectoryFunctionValueCacheNPZ(
                                        fname_traj=self.trajectory_file,
                                                              )
        else:
            self._npz_cache = None
        # TODO: how to handle h5py cache?!
        self._h5py_cache = None
        self._semaphores_by_func_id = collections.defaultdict(
                                                    asyncio.BoundedSemaphore
                                                              )

    def __len__(self) -> int:
        """
        Return the number of frames in the trajectory.

        Returns
        -------
        int
            The number of frames in the trajectory.
        """
        if self._len is not None:
            return self._len
        # create/open a mdanalysis universe to get the number of frames
        u = mda.Universe(self.structure_file, self.trajectory_file,
                         tpr_resid_from_one=True)
        self._len = len(u.trajectory)
        return self._len

    def __repr__(self) -> str:
        return (f"Trajectory(trajectory_file={self.trajectory_file},"
                + f" structure_file={self.structure_file})"
                )

    @property
    def structure_file(self) -> str:
        """Return absolute path to the structure file."""
        return copy.copy(self._structure_file)

    @property
    def trajectory_file(self) -> str:
        """Return absolute path to the trajectory file."""
        return copy.copy(self._trajectory_file)

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
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[0]
            # NOTE: works only(?) for trr and xtc
            self._first_step = ts.data["step"]
        return self._first_step

    @property
    def last_step(self) -> int:
        """Return the integration step of the last frame in the trajectory."""
        if self._last_step is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[-1]
            # TODO/FIXME:
            # NOTE: works only(?) for trr and xtc
            self._last_step = ts.data["step"]
        return self._last_step

    @property
    def dt(self) -> float:
        """The time intervall between subsequent *frames* (not steps) in ps."""
        if self._dt is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            # any frame is fine (assuming they all have the same spacing)
            ts = u.trajectory[0]
            self._dt = ts.data["dt"]
        return self._dt

    @property
    def first_time(self) -> float:
        """Return the integration timestep of the first frame in ps."""
        if self._first_time is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[0]
            self._first_time = ts.data["time"]
        return self._first_time

    @property
    def last_time(self) -> float:
        """Return the integration timestep of the last frame in ps."""
        if self._last_time is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[-1]
            self._last_time = ts.data["time"]
        return self._last_time

    async def _apply_wrapped_func(self, func_id, wrapped_func):
        async with self._semaphores_by_func_id[func_id]:
            # sort out which cache we use
            # TODO: do we want to give the h5py cache preference?
            #       or maybe change the order?
            #       or use h5py cache only if a h5py storage is been registered
            #        with asyncmd somewhere (e.g. in our constants/semaphores?)
            if self._h5py_cache is not None:
                return await self._apply_wrapped_func_h5py_cache(
                                                    func_id=func_id,
                                                    wrapped_func=wrapped_func,
                                                                 )
            if self._npz_cache is not None:
                return await self._apply_wrapped_func_npz_cache(
                                                    func_id=func_id,
                                                    wrapped_func=wrapped_func,
                                                                )
            # only 'local' cache, i.e. this trajectory has no file
            # associated with it (yet)
            return await self._apply_wrapped_func_local_cache(
                                                    func_id=func_id,
                                                    wrapped_func=wrapped_func,
                                                              )

    async def _apply_wrapped_func_local_cache(self, func_id: str, wrapped_func):
        try:
            # see if it is in cache
            return copy.copy(self._func_values[func_id])
        except KeyError:
            # if not calculate, store and return
            # send function application to seperate process and wait
            # until it finishes
            vals = await wrapped_func.get_values_for_trajectory(self)
            self._func_values_by_id[func_id] = vals
            return vals

    async def _apply_wrapped_func_npz_cache(self, func_id: str, wrapped_func):
        try:
            return copy.copy(self._npz_cache[func_id])
        except KeyError:
            # not in there
            # send function application to seperate process and wait
            # until it finishes
            vals = await wrapped_func.get_values_for_trajectory(self)
            self._npz_cache.append(func_id, vals)
            return vals

    async def _apply_wrapped_func_h5py_cache(self, func_id: str, wrapped_func):
        try:
            return copy.copy(self._h5py_cache[func_id])
        except KeyError:
            # not in there
            # send function application to seperate process and wait
            # until it finishes
            vals = await wrapped_func.get_values_for_trajectory(self)
            self._h5py_cache.append(func_id, vals)
            return vals

    def __getstate__(self):
        # enable pickling of Trajecory without call to ready_for_pickle
        # this should make it possible to pass it into a ProcessPoolExecutor
        # and lets us calculate TrajectoryFunction values asyncronously
        # NOTE: this removes everything except the filepaths
        state = self.__dict__.copy()
        # TODO: save stuff to _npz if only local cache present?!
        state["_h5py_cache"] = None
        state["_npz_cache"] = None
        # TODO: (same as above) only empty this dict if we saved content to npz
        #state["_func_values_by_id"] = {}
        state["_semaphores_by_func_id"] = collections.defaultdict(
                                                    asyncio.BoundedSemaphore
                                                                  )
        return state

    def object_for_pickle(self, group, overwrite):
        # TODO/NOTE: we ignore overwrite and assume the group is always empty
        #            (or at least matches this tra and we can add values?)
        # currently overwrite will always be false and we can just ignore it?!
        # and then we can/do also expect group to be empty...?
        state = self.__dict__.copy()
        if self._h5py_cache is not None:
            # we already have a file?
            # lets try to link the two groups?
            group = self._h5py_cache.root_grp
            # or should we copy? how to make sure we update both caches?
            # do we even want that? I (hejung) think a link is what you would
            # expect, i.e. both stored copies of the traj will have all cached
            # values available
            #group.copy(self._h5py_cache.root_grp, group)
            state["_h5py_cache"] = None
        else:
            # set h5py group such that we use the cache from now on
            self._h5py_cache = TrajectoryFunctionValueCacheH5PY(group)
            for func_id, idx in self._func_id_to_idx.items():
                self._h5py_cache.append(func_id, self._func_values[idx])
            # clear the 'local' cache and empty state, such that we initialize
            # to empty, next time we will get it all from file directly
            self._func_values = state["_func_values"] = []
            self._func_id_to_idx = state["_func_id_to_idx"] = {}
        # make the return object
        ret_obj = self.__class__.__new__(self.__class__)
        ret_obj.__dict__.update(state)
        return ret_obj

    def complete_from_h5py_group(self, group):
        # NOTE: Trajectories are immutable once stored,
        # EXCEPT: adding more cached function values for other funcs
        # so we keep around a ref to the h5py group we load from and
        # can then add the stuff in there
        # (loading is as easy as adding the file-cache because we store
        #  everything that was in 'local' cache in the file when we save)
        self._h5py_cache = TrajectoryFunctionValueCacheH5PY(group)
        return self


# TODO: update docstrings when we know how we do it exactly!
class TrajectoryFunctionValueCacheNPZ(collections.abc.Mapping):
    """
    Interface for caching function values in a given numpy npz file.

    Drop-in replacement for the dictionary that is used in the trajectories
    before they are saved.
    """

    # NOTE: this is written with the assumption that stored trajectories are
    #       immutable (except for adding additional stored function values)
    #       but we assume that the actual underlying trajectory stays the same,
    #       i.e. it is not extended after first storing it

    # NOTE: npz appending inspired by: https://stackoverflow.com/a/66618141

    def __init__(self, fname_traj: str) -> None:
        """
        Initialize a `TrajectoryFunctionValueCacheNPZ`.

        Parameters
        ----------
        fname_traj : str
            Absolute filename to the trajectory for which we cache CV values.
        """
        self.fname_npz = self._get_cache_filename(fname_traj=fname_traj)
        self._func_ids = []
        # NOTE/FIXME: It would be nice to use the MAX_FILES_OPEN semaphore
        # but then we need async/await and then we need to go to the create
        # classmethod (see below)
        # but since we (have to) open the npz file in the other magic methods
        # too it does not really matter?
        # (as we also leave some room for non-semaphored file openings anyway)
        if os.path.isfile(self.fname_npz):
            with np.load(self.fname_npz, allow_pickle=False) as npzfile:
                for k in npzfile.keys():
                    self._func_ids.append(str(k))

    #@classmethod
    #async def create(cls, fname_traj: str) -> TrajectoryFunctionValueCacheNPZ:
    #    """
    #    Create a `TrajectoryFunctionValueCacheNPZ` setting up all awaitables.

    #    Parameters
    #    ----------
    #    fname_traj : str
    #        Absolute filename to the trajectory for which we cache CV values.

    #    Returns
    #    -------
    #    TrajectoryFunctionValueCacheNPZ
    #        The initialized `TrajectoryFunctionValueCache`.
    #    """
    #    self = cls(fname_traj=fname_traj)
    #    if os.path.isfile(self.fname_npz):
    #        async with _SEMAPHORES["MAX_FILES_OPEN"]:
    #            with np.load(self.fname_npz, allow_pickle=False) as npzfile:
    #                for k in npzfile.keys():
    #                    self._func_ids.append(str(k))
    #    return self

    def _get_cache_filename(self, fname_traj: str) -> str:
        """
        Construct cachefilename from trajectory fname.

        Parameters
        ----------
        fname_traj : str
            Absolute path to the trajectory for which we cache.

        Returns
        -------
        str
            Absolute path to the cachefile associated with trajectory.
        """
        head, tail = os.path.split(fname_traj)
        return os.path.join(head, f".{tail}_asyncmd_cv_cache.npz")

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
        if func_id in self._func_ids:
            # first check if it already in there
            raise ValueError("There are already values stored for func_id "
                             + f"{func_id}. Changing the stored values is not "
                             + "supported.")
        if len(self) == 0:
            # these are the first cached CV values for this traj
            # so we just create the npz file
            np.savez(self.fname_npz, func_id=vals)
        else:
            # already something cached, need to append to the npz file
            # npz files are just zipped together collections of npy files
            # so lets make a npy file saev into a BytesIO and then write that
            # to the end of the npz file
            bio = io.BytesIO()
            np.save(bio, vals)
            with zipfile.ZipFile(file=self.fname_npz,
                                 mode="a",  # append!
                                 # uncompressed (but) zip archive member
                                 compression=zipfile.ZIP_STORED,
                                 ) as zfile:
                zfile.writestr(f"{func_id}.npy", data=bio.getvalue())

        # add func_id to list of func_ids that we know are cached in npz
        self._func_ids.append(func_id)


class TrajectoryFunctionValueCacheH5PY(collections.abc.Mapping):
    """
    Interface for caching function values in a given h5py/hdf5 group.

    Drop-in replacement for the dictionary that is used in the trajectories
    before they are saved to h5py.
    """

    # NOTE: this is written with the assumption that stored trajectories are
    #       immutable (except for adding additional stored function values)
    #       but we assume that the actual underlying trajectory stays the same,
    #       i.e. it is not extended after first storing it

    def __init__(self, root_grp):
        self.root_grp = root_grp
        self._h5py_paths = {"ids": "FunctionIDs",
                            "vals": "FunctionValues"
                            }
        self._ids_grp = self.root_grp.require_group(self._h5py_paths["ids"])
        self._vals_grp = self.root_grp.require_group(self._h5py_paths["vals"])

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
