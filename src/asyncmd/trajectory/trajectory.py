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
import collections
import copy
import asyncio
import logging
import MDAnalysis as mda


logger = logging.getLogger(__name__)


class Trajectory:
    """
    Represent a trajectory.

    Keep track of the paths of the trajectory and the structure file.
    Caches values for (wrapped) functions acting on the trajectory.
    Also makes available (and caches) a number of useful attributes, e.g.
    `first_step` and `last_step` (the first and last intergation step in the
    trajectory), `dt`, `first_time`, `last_time`, `length` (in frames) and
    `nstout`.

    NOTE: first_step and last_step is only useful for trajectories that come
          directly from a MDEngine. As soon as the trajecory has been
          concatenated using MDAnalysis (i.e. the `TrajectoryConcatenator`)
          the step information is just the frame number in the trajectory part
          that became first/last frame in the concatenated trajectory.
    """

    def __init__(self, trajectory_file, structure_file, nstout=None, **kwargs):
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
            self.trajectory_file = os.path.abspath(trajectory_file)
        else:
            raise ValueError(f"trajectory_file ({trajectory_file}) must be accessible.")
        if os.path.isfile(structure_file):
            self.structure_file = os.path.abspath(structure_file)
        else:
            raise ValueError(f"structure_file ({structure_file}) must be accessible.")
        # properties
        self.nstout = nstout  # use the setter to make basic sanity checks
        self._len = None
        self._first_step = None
        self._last_step = None
        self._dt = None
        self._first_time = None
        self._last_time = None
        # stuff for caching of functions applied to this traj
        self._func_id_to_idx = {}
        self._func_values = []
        self._h5py_cache = None
        self._semaphores_by_func_id = collections.defaultdict(
                                                    asyncio.BoundedSemaphore
                                                              )

    def __len__(self):
        if self._len is not None:
            return self._len
        # create/open a mdanalysis universe to get the number of frames
        u = mda.Universe(self.structure_file, self.trajectory_file,
                         tpr_resid_from_one=True)
        self._len = len(u.trajectory)
        return self._len

    def __repr__(self):
        return (f"Trajectory(trajectory_file={self.trajectory_file},"
                + f" structure_file={self.structure_file})"
                )

    @property
    def nstout(self):
        """Output frequency between subsequent frames in integration steps."""
        return self._nstout

    @nstout.setter
    def nstout(self, val):
        if val is not None:
            # ensure that it is an int
            val = int(val)
        # enable setting to None
        self._nstout = val

    @property
    def first_step(self):
        """The integration step of the first frame."""
        if self._first_step is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[0]
            # NOTE: works only(?) for trr and xtc
            self._first_step = ts.data["step"]
        return self._first_step

    @property
    def last_step(self):
        """The integration step of the last frame."""
        if self._last_step is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[-1]
            # TODO/FIXME:
            # NOTE: works only(?) for trr and xtc
            self._last_step = ts.data["step"]
        return self._last_step

    @property
    def dt(self):
        """The time intervall between subsequent *frames* (not steps) in ps."""
        if self._dt is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            # any frame is fine (assuming they all have the same spacing)
            ts = u.trajectory[0]
            self._dt = ts.data["dt"]
        return self._dt

    @property
    def first_time(self):
        """The integration timestep of the first frame in ps."""
        if self._first_time is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[0]
            self._first_time = ts.data["time"]
        return self._first_time

    @property
    def last_time(self):
        """The integration timestep of the last frame in ps."""
        if self._last_time is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[-1]
            self._last_time = ts.data["time"]
        return self._last_time

    async def _apply_wrapped_func(self, func_id, wrapped_func):
        async with self._semaphores_by_func_id[func_id]:
            if self._h5py_cache is not None:
                # first check if we are loaded and possibly get it from there
                # trajectories are immutable once stored, so no need to check len
                try:
                    return copy.copy(self._h5py_cache[func_id])
                except KeyError:
                    # not in there
                    # send function application to seperate process and wait for it
                    vals = await wrapped_func.get_values_for_trajectory(self)
                    self._h5py_cache.append(func_id, vals)
                    return vals
            else:
                # only 'local' cache, i.e. this trajectory has no file associated (yet)
                try:
                    # see if it is in cache
                    idx = self._func_id_to_idx[func_id]
                    return copy.copy(self._func_values[idx])
                except KeyError:
                    # if not calculate, store and return
                    # send function application to seperate process and wait for it
                    vals = await wrapped_func.get_values_for_trajectory(self)
                    self._func_id_to_idx[func_id] = len(self._func_id_to_idx)
                    self._func_values.append(vals)
                    return vals

    def __getstate__(self):
        # enable pickling of Trajecory without call to ready_for_pickle
        # this should make it possible to pass it into a ProcessPoolExecutor
        # and lets us calculate TrajectoryFunction values asyncronously
        # NOTE: this removes everything except the filepaths
        state = self.__dict__.copy()
        state["_h5py_cache"] = None
        #state["_func_values"] = []
        #state["_func_id_to_idx"] = {}
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
            self._h5py_cache = TrajectoryFunctionValueCache(group)
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
        self._h5py_cache = TrajectoryFunctionValueCache(group)
        return self


class TrajectoryFunctionValueCache(collections.abc.Mapping):
    """Interface for caching function values on a per trajectory basis."""
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

    def append(self, func_id, vals, ignore_existing=False):
        if not isinstance(func_id, str):
            raise TypeError("Keys (func_id) must be of type str.")
        if (func_id in self) and (not ignore_existing):
            raise ValueError(f"There are already values stored for func_id {func_id}."
                             + " Changing the stored values is not supported.")
        elif (func_id in self) and ignore_existing:
            logger.debug(f"File cached values already present for function with id {func_id}."
                         + "Not adding the new values because ignore_existing=False.")
            return
        # TODO: do we also want to check vals for type?
        name = str(len(self))
        _ = self._ids_grp.create_dataset(name, data=func_id)
        _ = self._vals_grp.create_dataset(name, data=vals)
