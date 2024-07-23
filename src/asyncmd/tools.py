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
import shutil
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
    elif shutil.which(executable) is not None:
        # see if we find it in $PATH
        executable = shutil.which(executable)
    else:
        raise ValueError(f"{executable} must be an existing path or accesible "
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
        # TODO: should we info/warn if the file is not there?
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
        # TODO: should we info/warn if the file is not there?
        pass
