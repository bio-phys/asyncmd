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
This module contains the implementations of the TrajectoryCache classes.

They are used in the asyncmd.Trajectory object to enable caching of CV values.
"""
import io
import os
import abc
import logging
import typing
import zipfile
import collections

import numpy as np


from .._config import _GLOBALS, _GLOBALS_KEYS


if typing.TYPE_CHECKING:  # pragma: no cover
    import h5py


logger = logging.getLogger(__name__)


class ValuesAlreadyStoredError(ValueError):
    """
    Error raised by :class:`TrajectoryFunctionValueCache` classes when trying to
    append values for a func_id that is already present.
    """


class NoValuesCachedForTrajectoryInH5PYError(ValueError):
    """
    Error raised by :class:`OneH5PYGroupTrajectoryFunctionValueCache` when
    instantiating with a read-only h5py_cache in which no values for the given
    :class:`Trajectory` (hash) are stored.
    """


class CanNotChangeReadOnlyH5PYError(PermissionError):
    """
    Error raised by :class:`OneH5PYGroupTrajectoryFunctionValueCache` when trying
    to change (append or clear) an instance with a read-only h5py_cache.
    """


class TrajectoryFunctionValueCache(collections.abc.Mapping):
    """
    Abstract base class defining the interface for TrajectoryFunctionValueCaches.

    Note: We assume that stored CV values are immutable (except for adding
    additional stored function values), since they are tied to the trajectory
    (hash) and the func_id of the wrapped function (which is unique and includes
    code and call_kwargs). I.e. as long as both the underlying trajectory and
    the func_id of the cached function stay the same, the cached values are
    current.
    We therefore get away with a ``Mapping`` instead of a ``MutableMapping`` and
    only need the additional methods ``append`` and ``clear_all_values`` instead
    of generic setters.
    """
    def __init__(self, traj_hash: int, traj_files: list[str]) -> None:
        """
        Initialize a ``TrajectoryFunctionValueCache``.

        Parameters
        ----------
        traj_hash : int
            The hash of the associated ``asyncmd.Trajectory``.
        traj_files : list[str]
            The filenames of the associated trajectory files.
        """
        self._traj_hash = traj_hash
        self._traj_files = traj_files

    @abc.abstractmethod
    def append(self, func_id: str, values: np.ndarray) -> None:
        """
        Append given ``values`` for ``func_id``.

        Parameters
        ----------
        func_id : str
            The ID of the function the values belong to.
        values : np.ndarray
            The function values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clear_all_values(self) -> None:
        """
        Clear all cached function values from cache, i.e. empty the cache.
        """
        raise NotImplementedError


class TrajectoryFunctionValueCacheInMemory(TrajectoryFunctionValueCache):
    """
    Interface for caching trajectory function values in memory using a dict.
    """
    def __init__(self, traj_hash: int, traj_files: list[str]) -> None:
        """
        Initialize a ``TrajectoryFunctionValueCacheInMemory``.

        Parameters
        ----------
        traj_hash : int
            The hash of the associated ``asyncmd.Trajectory``.
        traj_files : list[str]
            The filenames of the associated trajectory files.
        """
        super().__init__(traj_hash, traj_files)
        self._func_values_by_id: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._func_values_by_id)

    def __iter__(self) -> collections.abc.Generator[str]:
        yield from self._func_values_by_id

    def __getitem__(self, key: str) -> np.ndarray:
        return self._func_values_by_id[key]

    def append(self, func_id: str, values: np.ndarray) -> None:
        if func_id in self._func_values_by_id:
            # first check if it already in there
            raise ValuesAlreadyStoredError(
                            "There are already values stored for func_id "
                            f"{func_id}. Changing the stored values is not "
                            "supported.")
        self._func_values_by_id[func_id] = values

    def clear_all_values(self) -> None:
        self._func_values_by_id = {}


class TrajectoryFunctionValueCacheInNPZ(TrajectoryFunctionValueCache):
    """
    Interface for caching trajectory function values in a numpy npz file.

    Will use one separate npz file for each Trajectory object, the file is
    placed in the filesystem right next to the underlying trajectory file.
    The name of the npz file is derived from the trajectory file name.
    Additionally, the npz file stores the ``traj_hash`` and if the ``traj_hash``
    changes, i.e. the trajectory changes (for the same npz filename), the npz
    file and all cached values will be removed (for this trajectory filename).
    """

    # NOTE: npz appending inspired by: https://stackoverflow.com/a/66618141

    # NOTE: It would be nice to use the MAX_FILES_OPEN semaphore
    #       but then we need async/await and then we need to go to a 'create'
    #       classmethod that is async and required for initialization
    #       (because __init__ cant be async).
    #       But since we (have to) open the npz file in the other magic methods
    #       to it does not really matter (as they can not be async either)?
    #       And as we also leave some room for non-semaphored file openings anyway...

    _TRAJ_HASH_NPZ_KEY = "hash_of_trajs"  # key of traj_hash in npz file

    def __init__(self, traj_hash: int, traj_files: list[str]) -> None:
        super().__init__(traj_hash=traj_hash, traj_files=traj_files)
        self.fname_npz = self.get_cache_filename(traj_files=traj_files)
        self._func_ids: list[str] = []
        # sort out if we have an associated npz file already
        # and if it is from/for the "right" trajectory file
        self._ensure_consistent_npz()

    def _ensure_consistent_npz(self) -> None:
        # next line makes sure we only remember func_ids from the current npz
        self._func_ids = []
        if not os.path.isfile(self.fname_npz):
            # no npz so nothing to do except making sure we have no func_ids
            return
        existing_npz_matches = False
        with np.load(self.fname_npz, allow_pickle=False) as npzfile:
            try:
                # it is an array with 1 element, but pylint does not know that
                # pylint: disable-next=unsubscriptable-object
                saved_hash_traj = npzfile[self._TRAJ_HASH_NPZ_KEY][0]
            except KeyError:
                # we probably tripped over an old formatted npz
                # so we will just rewrite it completely with hash
                pass
            else:
                # old hash found, lets compare the two hashes
                if (existing_npz_matches := self._traj_hash == saved_hash_traj):
                    # if they do populate self with the func_ids we have
                    # cached values for
                    for k in npzfile.keys():
                        if k != self._TRAJ_HASH_NPZ_KEY:
                            self._func_ids.append(str(k))
        # now if the old npz did not match we should remove it
        # then we will rewrite it with the first cached CV values
        if not existing_npz_matches:
            logger.debug("Found existing npz file (%s) but the "
                         "trajectory hash does not match. "
                         "Recreating the npz cache from scratch.",
                         self.fname_npz
                         )
            os.unlink(self.fname_npz)

    @classmethod
    def get_cache_filename(cls, traj_files: list[str]) -> str:
        """
        Construct cachefilename from trajectory fname.

        Parameters
        ----------
        traj_files : list[str]
            Path to the trajectory for which we cache.

        Returns
        -------
        str
            Path to the cachefile associated with trajectory.
        """
        head, tail = os.path.split(traj_files[0])
        return os.path.join(
            head,
            f".{tail}{'_MULTIPART' if len(traj_files) > 1 else ''}_asyncmd_cv_cache.npz"
                            )

    def __len__(self) -> int:
        return len(self._func_ids)

    def __iter__(self) -> collections.abc.Generator[str]:
        yield from self._func_ids

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._func_ids:
            with np.load(self.fname_npz, allow_pickle=False) as npzfile:
                return npzfile[key]
        # Key not found/ no values stored for key
        raise KeyError(f"No values for {key} cached (yet).")

    def append(self, func_id: str, values: np.ndarray) -> None:
        """
        Append values for given func_id.

        Parameters
        ----------
        func_id : str
            Function identifier.
        values : np.ndarray
            Values of application of function with given func_id.

        Raises
        ------
        TypeError
            If ``func_id`` is not a string.
        ValueError
            If there are already values stored for ``func_id`` in self.
        """
        if func_id in self._func_ids:
            # first check if it already in there
            raise ValuesAlreadyStoredError(
                            "There are already values stored for func_id "
                            f"{func_id}. Changing the stored values is not "
                            "supported.")
        if not self._func_ids:
            # these are the first cached CV values for this traj
            # so we just create the (empty) npz file
            np.savez(self.fname_npz)
            # and write the trajectory hash
            self._append_data_to_npz(name=self._TRAJ_HASH_NPZ_KEY,
                                     value=np.array([self._traj_hash]),
                                     )
        # now we can append either way
        # either already something cached, or freshly created empty file
        self._append_data_to_npz(name=func_id, value=values)
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

    def clear_all_values(self) -> None:
        self._func_ids = []  # clear internal storage of func_ids
        if os.path.isfile(self.fname_npz):
            os.unlink(self.fname_npz)  # and remove the file if it exists


class OneH5PYGroupTrajectoryFunctionValueCache(TrajectoryFunctionValueCache):
    """
    Interface for caching trajectory function values in one given h5py group, used
    as composite class in :class:`TrajectoryFunctionValueCacheInH5PY`.

    Can also be used on h5py files/groups open in read-only mode, then appending
    and clearing will raise errors.

    This class also contains a classmethod to copy over all cached values from
    a given h5py (cache) group to another h5py (cache) group. This method is
    (optionally) used when a new h5py cache is registered to transfer all values
    from all previously registered caches.

    The values will be stored in the given group in a subgroup (defined by
    ``self._H5PY_PATHS['prefix']``) and then by trajectory hash, i.e. the full path
    becomes '$self._H5PY_PATHS['prefix']/$TRAJ_HASH/'.
    Within this path/group there are two subgroups, one for the func_ids and one
    for the cached values (names again defined by ``self._H5PY_PATHS['prefix']``).
    Inside these groups datasets are named with an index, and the index is used
    to associate func_ids and cached values, i.e. values with index i correspond
    to the func_id with index i.
    """

    # mapping of shorthands to h5py-paths/ group names used by this class
    _H5PY_PATHS = {"ids": "FunctionIDs",
                   "vals": "FunctionValues",
                   "prefix": "asyncmd/TrajectoryFunctionValueCache",
                   }

    def __init__(self, traj_hash: int, traj_files: list[str],
                 h5py_cache: "h5py.Group | h5py.File",
                 ) -> None:
        super().__init__(traj_hash, traj_files)
        self.h5py_cache = h5py_cache
        self._read_only = h5py_cache.file.mode == "r"
        try:
            self._root_grp = self.h5py_cache.require_group(
                            f"{self._H5PY_PATHS['prefix']}/{self._traj_hash}"
                                                           )
        except ValueError as exc:
            # value error is raised when the group does not exist
            # (and we can not create it because the file is read-only)
            if self._read_only:
                # but check to be sure
                raise NoValuesCachedForTrajectoryInH5PYError(
                    f"No values stored for traj_hash ({traj_hash}) and h5py_cache "
                    f"({h5py_cache}) is in read-only mode, i.e. no values to "
                    "retrieve and no way of appending new ones."
                    ) from exc
            # This hopefully never happens: The file is not in read-only mode,
            # but we get a ValueError. Since we dont know what happened, we just
            # reraise the error (and we dont expect to get coverage for this line)
            raise exc  # pragma: no cover

        self._ids_grp = self._root_grp.require_group(self._H5PY_PATHS["ids"])
        self._vals_grp = self._root_grp.require_group(self._H5PY_PATHS["vals"])
        # keep a list of func_ids we have cached values for in memory
        # NOTE: remember to add func_ids we cache values for here also!
        self._func_ids: list[str] = [self._ids_grp[str(idx)].asstr()[()]
                                     for idx in range(len(self._ids_grp.keys()))
                                     ]

    @classmethod
    def add_values_for_all_trajectories(cls, src_h5py_cache: "h5py.Group | h5py.File",
                                        dst_h5py_cache: "h5py.Group | h5py.File",
                                        ) -> None:
        """
        Add all cached function values for all trajectories from src_h5py_cache to dst_h5py_cache.

        Adds values for all trajectory objects independently of if they are currently
        initialized.
        Only values that do not exist in dst_h5py_cache are added, existing values
        are ignored.

        Parameters
        ----------
        src_h5py_cache : h5py.Group | h5py.File
            The source h5py_cache.
        dst_h5py_cache : h5py.Group | h5py.File
            The destination h5py_cache.
        """
        # get the h5 groups that contains the traj_hashs as groups, i.e. the prefix
        src_grp = src_h5py_cache.require_group(cls._H5PY_PATHS["prefix"])
        dst_grp = dst_h5py_cache.require_group(cls._H5PY_PATHS["prefix"])
        # iterate over all traj (hash) in src to see if we copy them
        for traj_hash in src_grp:
            if traj_hash not in dst_grp:
                # easy, no values at all stored for this traj (hash) in dst at all
                # copy the whole group
                dst_grp.copy(source=src_grp[traj_hash], dest=dst_grp)
                continue  # and on to the next traj (hash)
            # need to iterate over all func_ids in src and see if we copy them
            # get the func_id and func values groups in dst and src
            dst_id_grp = dst_grp[traj_hash][cls._H5PY_PATHS["ids"]]
            dst_val_grp = dst_grp[traj_hash][cls._H5PY_PATHS["vals"]]
            src_id_grp = src_grp[traj_hash][cls._H5PY_PATHS["ids"]]
            src_val_grp = src_grp[traj_hash][cls._H5PY_PATHS["vals"]]
            # func_id list to compare with each other
            func_ids_in_dst = [dst_id_grp[str(idx)].asstr()[()]
                               for idx in range(len(dst_id_grp.keys()))
                               ]
            func_ids_in_src = [src_id_grp[str(idx)].asstr()[()]
                               for idx in range(len(src_id_grp.keys()))
                               ]
            for idx_in_src, func_id in enumerate(func_ids_in_src):
                if func_id not in func_ids_in_dst:
                    # copy it over to append/create the new datasets for func_ids and values
                    name = str(len(func_ids_in_dst))
                    dst_val_grp.copy(source=src_val_grp[str(idx_in_src)],
                                     dest=dst_val_grp, name=name)
                    dst_id_grp.copy(source=src_id_grp[str(idx_in_src)],
                                    dest=dst_id_grp, name=name)
                    # Append func_id to func_ids_in_dst so we get the next name correct
                    func_ids_in_dst.append(func_id)

    def __len__(self) -> int:
        return len(self._func_ids)

    def __iter__(self) -> collections.abc.Generator[str]:
        yield from self._func_ids

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._func_ids:
            idx = self._func_ids.index(key)
            return self._vals_grp[str(idx)][:]
        # if we got until here the key is not in there
        raise KeyError(f"Key not found (key={key}).")

    def append(self, func_id: str, values: np.ndarray) -> None:
        if self._read_only:
            raise CanNotChangeReadOnlyH5PYError(
                f"Can not append to a h5py cache ({self.h5py_cache}) opened "
                "in read-only mode."
                )
        if func_id in self:
            raise ValuesAlreadyStoredError(
                            "There are already values stored for func_id "
                            f"{func_id}. Changing the stored values is not "
                            "supported.")
        name = str(len(self))
        _ = self._ids_grp.create_dataset(name, data=func_id)
        _ = self._vals_grp.create_dataset(name, data=values)
        # append func id for newly stored func (values) to internal in memory state
        self._func_ids.append(func_id)

    def clear_all_values(self) -> None:
        if self._read_only:
            raise CanNotChangeReadOnlyH5PYError(
                f"Can not clear a h5py cache ({self.h5py_cache}) opened "
                "in read-only mode."
                )
        # delete and recreate the id and values h5py subgroups
        del self._root_grp[self._H5PY_PATHS["ids"]]
        del self._root_grp[self._H5PY_PATHS["vals"]]
        self._ids_grp = self._root_grp.require_group(self._H5PY_PATHS["ids"])
        self._vals_grp = self._root_grp.require_group(self._H5PY_PATHS["vals"])
        # and empty the in-memory func-id list
        self._func_ids = []


class TrajectoryFunctionValueCacheInH5PY(TrajectoryFunctionValueCache):
    """
    Interface for caching trajectory function values in multiple h5py groups.

    This class combines multiple :class:`OneH5PYGroupTrajectoryFunctionValueCache`
    (one for each h5py cache group asyncmd knows about) and retrieves the cached
    values from any of the associated :class:`OneH5PYGroupTrajectoryFunctionValueCache`
    objects.
    Only one h5py group can be writeable at a time, i.e. cached values will only
    be added to the one writeable cache group, but will be retrieved from all
    associated read-only h5py groups/ :class:`OneH5PYGroupTrajectoryFunctionValueCache`
    objects.
    Trying to append to this class with no writeable cache group registered
    results in a logged error and no append takes place, clearing will raise a
    :class:`CanNotChangeReadOnlyH5PYError`.
    """

    def __init__(self, traj_hash: int, traj_files: list[str]) -> None:
        super().__init__(traj_hash, traj_files)
        # sort out which writeable (if any) and which fallback read-only stores we use
        writeable_h5py_cache: "h5py.Group | h5py.File | None" = _GLOBALS.get(
            _GLOBALS_KEYS.H5PY_CACHE, None
            )
        read_only_h5py_caches: "list[h5py.Group | h5py.File]" = _GLOBALS.get(
            _GLOBALS_KEYS.H5PY_CACHE_READ_ONLY_FALLBACKS, []
            )
        if (writeable_h5py_cache is None) and (not read_only_h5py_caches):
            raise RuntimeError(
                f"Can not initialize a {type(self)} without any h5py cache set!"
                " Try calling `asyncmd.config.register_h5py_cache` first."
                )
        # setup (writeable) main cache if we have it
        if writeable_h5py_cache is None:
            logger.warning("Initializing a Trajectory cache in h5py with only "
                           "read-only h5py.Groups associated. Newly calculated "
                           "function values will not be cached!")
            self._main_cache = None
        else:
            self._main_cache = OneH5PYGroupTrajectoryFunctionValueCache(
                                        traj_hash=traj_hash,
                                        traj_files=traj_files,
                                        h5py_cache=writeable_h5py_cache,
                                        )
        # setup all fallback caches (if any)
        self._fallback_caches = []
        for cache in read_only_h5py_caches:
            try:
                self._fallback_caches += [OneH5PYGroupTrajectoryFunctionValueCache(
                                                traj_hash=traj_hash,
                                                traj_files=traj_files,
                                                h5py_cache=cache,
                                                )
                                          ]
            except NoValuesCachedForTrajectoryInH5PYError:
                # we just dont add this cache to our list since it contains
                # no values for us (the trajectory we are associated to)
                pass

    # bind this classmethod here too, because it is useful to have access to
    # everywhere the TrajectoryFunctionValueCacheInH5PY is used
    add_values_for_all_trajectories = (
        OneH5PYGroupTrajectoryFunctionValueCache.add_values_for_all_trajectories
        )

    def deregister_h5py_cache(self, h5py_cache: "h5py.File | h5py.Group") -> None:
        """
        Deregister the given h5py_cache as a source of cached values.

        Parameters
        ----------
        h5py_cache : h5py.File | h5py.Group
            The h5py_cache to deregister/remove from caching
        """
        if self._main_cache is not None:
            if self._main_cache.h5py_cache is h5py_cache:
                logger.warning(
                    "Deregistering the writeable (main) cache (%s). "
                    "Newly calculated function values will not be cached!",
                    h5py_cache
                    )
                self._main_cache = None
                # found it so it can not be a fallback_cache, so get out of here
                return
        idx_to_remove = None
        for i, cache in enumerate(self._fallback_caches):
            if cache.h5py_cache is h5py_cache:
                idx_to_remove = i
                break  # found it and there can only be one with this h5py_cache
        if idx_to_remove is not None:
            del self._fallback_caches[idx_to_remove]

    def __len__(self) -> int:
        length = 0
        if self._main_cache is not None:
            length += len(self._main_cache)
        length += sum(len(cache) for cache in self._fallback_caches)
        return length

    def __iter__(self) -> collections.abc.Generator[str]:
        # just loop trough all potential caches
        if self._main_cache is not None:
            yield from self._main_cache
        for cache in self._fallback_caches:
            yield from cache

    def __getitem__(self, key: str) -> np.ndarray:
        # again just loop over all potential caches
        if self._main_cache is not None:
            if key in self._main_cache:
                return self._main_cache[key]
        for cache in self._fallback_caches:
            if key in cache:
                return cache[key]
        # if we got until here the key is not in there
        raise KeyError(f"Key not found (key={key}).")

    def append(self, func_id: str, values: np.ndarray) -> None:
        if self._main_cache is None:
            logger.error("Can not append (store) cached function values because "
                         "no h5py cache is open in writeable mode! "
                         "Try calling `asyncmd.config.register_h5py_cache` with "
                         "a **writeable** h5py.Group or h5py.File."
                         )
            return
        self._main_cache.append(func_id=func_id, values=values)

    def clear_all_values(self) -> None:
        if self._main_cache is None:
            raise CanNotChangeReadOnlyH5PYError(
                        "Can not clear cached function values because "
                        "no h5py cache is open in writeable mode! "
                        "Try calling `asyncmd.config.register_h5py_cache` with "
                        "a **writeable** h5py.Group or h5py.File."
                        )
        self._main_cache.clear_all_values()
