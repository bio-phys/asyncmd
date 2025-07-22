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
import zipfile
import collections

import numpy as np


from .._config import _GLOBALS


logger = logging.getLogger(__name__)


class ValuesAlreadyStoredError(ValueError):
    """
    Error raised by :class:`TrajectoryFunctionValueCache` classes when trying to
    append values for a func_id that is already present.
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


class TrajectoryFunctionValueCacheInH5PY(TrajectoryFunctionValueCache):
    """
    Interface for caching trajectory function values in a given h5py group.

    Gets the centrally set ``H5PY_CACHE`` configuration variable in ``__init__``
    and uses it as ``h5py.Group`` to store the cached values in.
    The values will be stored in this group in a subgroup (defined by
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

    def __init__(self, traj_hash: int, traj_files: list[str]) -> None:
        super().__init__(traj_hash, traj_files)
        try:
            self._h5py_cache = _GLOBALS["H5PY_CACHE"]
        except KeyError as e:
            raise RuntimeError(
                f"Can not initialize a {type(self)} without global h5py cache set!"
                " Try calling `asyncmd.config.register_h5py_cache` first."
                ) from e

        self._root_grp = self._h5py_cache.require_group(
                            f"{self._H5PY_PATHS['prefix']}/{self._traj_hash}"
                                                        )
        self._ids_grp = self._root_grp.require_group(self._H5PY_PATHS["ids"])
        self._vals_grp = self._root_grp.require_group(self._H5PY_PATHS["vals"])
        # keep a list of func_ids we have cached values for in memory
        # NOTE: remember to add func_ids we cache values for here also!
        self._func_ids: list[str] = [self._ids_grp[str(idx)].asstr()[()]
                                     for idx in range(len(self._ids_grp.keys()))
                                     ]

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
        # delete and recreate the id and values h5py subgroups
        del self._root_grp[self._H5PY_PATHS["ids"]]
        del self._root_grp[self._H5PY_PATHS["vals"]]
        self._ids_grp = self._root_grp.require_group(self._H5PY_PATHS["ids"])
        self._vals_grp = self._root_grp.require_group(self._H5PY_PATHS["vals"])
        # and empty the in-memory func-id list
        self._func_ids = []
