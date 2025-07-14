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
This file contains functions and classes (re)used internally in asyncmd.

These functions and classes are not (thought to be) exposed to the users but
instead intended to be (re)used in newly added asyncmd code.
This is also not the place for MD-related utility functions, for this see utils.py

Currently in here are:

- ensure_executable_available
- remove_file_if_exist and remove_file_if_exist_async
- FlagChangeList (and its typed sibling): lists with some sugar to remember if
  their content has changed

"""
import collections
import os
import shutil
import typing
import aiofiles


def ensure_executable_available(executable: str) -> str:
    """
    Ensure the given executable is available and executable.

    Takes a relative or absolute path to an executable or the name of an
    executable available in $PATH. Returns the full path to the executable.

    Parameters
    ----------
    executable : str
        Name or path of an executable.

    Returns
    -------
    path_to_executable : str
        Full path to the given executable if it exists.

    Raises
    ------
    ValueError
        If the given name does not exist or can not be executed.
    """
    if os.path.isfile(os.path.abspath(executable)):
        # see if it is a relative path starting from cwd
        # (or a full path starting with /)
        executable = os.path.abspath(executable)
        if not os.access(executable, os.X_OK):
            raise ValueError(f"{executable} must be executable.")
    elif (which_exe := shutil.which(executable)) is not None:
        # see if we find it in $PATH
        executable = which_exe
    else:
        raise ValueError(f"{executable} must be an existing path or accessible "
                         + "via the $PATH environment variable.")
    return executable


def remove_file_if_exist(f: str):
    """
    Remove a given file if it exists.

    Parameters
    ----------
    f : str
        Path to the file to remove.
    """
    try:
        os.remove(f)
    except FileNotFoundError:
        pass


async def remove_file_if_exist_async(f: str):
    """
    Remove a given file if it exists asynchronously.

    Parameters
    ----------
    f : str
        Path to the file to remove.
    """
    try:
        await aiofiles.os.remove(f)
    except FileNotFoundError:
        pass


class FlagChangeList(collections.abc.MutableSequence):
    """A list that knows if it has been changed after initializing."""

    def __init__(self, data: collections.abc.Iterable) -> None:
        """
        Initialize a `FlagChangeList`.

        Parameters
        ----------
        data : Iterable
            The data this `FlagChangeList` will hold.

        Raises
        ------
        TypeError
            Raised when data is not an :class:`Iterable`.
        """
        self._data = list(data)
        self._changed = False

    @property
    def changed(self) -> bool:
        """
        Whether this `FlagChangeList` has been modified since creation.

        Returns
        -------
        bool
        """
        return self._changed

    def __repr__(self) -> str:  # pragma: no cover
        return self._data.__repr__()

    def __getitem__(self, index: int | slice) -> typing.Any:
        return self._data.__getitem__(index)

    def __len__(self) -> int:
        return self._data.__len__()

    def __setitem__(self, index: int | slice, value) -> None:
        self._data.__setitem__(index, value)
        self._changed = True

    def __delitem__(self, index: int | slice) -> None:
        self._data.__delitem__(index)
        self._changed = True

    def insert(self, index: int, value: typing.Any):
        """
        Insert `value` at position given by `index`.

        Parameters
        ----------
        index : int
            The index of the new value in the `FlagChangeList`.
        value : typing.Any
            The value to insert into this `FlagChangeList`.
        """
        self._data.insert(index, value)
        self._changed = True

    def __add__(self, other: collections.abc.Iterable):
        return FlagChangeList(data=self._data + list(other))

    def __iadd__(self, other: collections.abc.Iterable):
        for val in other:
            self.append(val)
        return self


class TypedFlagChangeList(FlagChangeList):
    """
    A :class:`FlagChangeList` with an ensured type for individual list items.

    Note that single strings are not treated as Iterable, i.e. (as opposed to
    a "normal" list) `TypedFlagChangeList(data="abc")` will result in
    `data=["abc"]` (and not `data=["a", "b", "c"]`).
    """

    def __init__(self, data: collections.abc.Iterable, dtype: type) -> None:
        """
        Initialize a `TypedFlagChangeList`.

        Parameters
        ----------
        data : Iterable
            (Initial) data for this `TypedFlagChangeList`.
        dtype : Callable datatype
            The datatype for all entries in this `TypedFlagChangeList`. Will be
            called on every value separately and is expected to convert to the
            desired datatype.
        """
        self._dtype = dtype  # set first to use in _convert_type method
        data = self._ensure_iterable(data)
        typed_data = [self._convert_type(v, index=i)
                      for i, v in enumerate(data)]
        super().__init__(data=typed_data)

    @property
    def dtype(self) -> type:
        """
        All values in this `TypedFlagChangeList` are converted to dtype.

        Returns
        -------
        type
        """
        return self._dtype

    def _ensure_iterable(self, data) -> collections.abc.Iterable:
        if getattr(data, '__iter__', None) is None:
            # convenience for singular options,
            # if it has no iter attribute we assume it is the only item
            data = [data]
        elif isinstance(data, str):
            # strings have an iter but we still do not want to split them into
            # single letters, so just put a list around
            data = [data]
        return data

    def _convert_type(self, value,
                      index: int | slice | None = None) -> list:
        # here we ignore index, but passing it should in principal make it
        # possible to use different dtypes for different indices
        if isinstance(index, int):
            return self._dtype(value)
        return [self._dtype(v) for v in value]

    def __setitem__(self, index: int | slice, value) -> None:
        typed_value = self._convert_type(value, index=index)
        self._data.__setitem__(index, typed_value)
        self._changed = True

    def insert(self, index: int, value) -> None:
        """
        Insert `value` at position given by `index`.

        Parameters
        ----------
        index : int
            The index of the new value in the `TypedFlagChangeList`.
        value : typing.Any
            The value to insert into this `TypedFlagChangeList`.
        """
        typed_value = self._convert_type(value, index=index)
        self._data.insert(index, typed_value)
        self._changed = True

    def __add__(self, other: collections.abc.Iterable):
        # cast other to an iterable as we expect it (excluding the strings)
        other = self._ensure_iterable(other)
        ret = TypedFlagChangeList(data=self._data + list(other),
                                  dtype=self._dtype)
        return ret

    def __iadd__(self, other: collections.abc.Iterable):
        # cast other to an iterable as we expect it (excluding the strings)
        other = self._ensure_iterable(other)
        for val in other:
            self.append(val)
        return self
