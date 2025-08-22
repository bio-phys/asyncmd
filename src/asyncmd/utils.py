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
This file implements commonly used utility functions related to MD usage with asyncmd.

This includes various functions to get all trajectory (or other files) related
to a given engine, naming scheme, and/or file-ending.
It also includes various functions to retrieve or ensure important parameters from
MDConfig/MDEngine combinations, such as nstout_from_mdconfig and ensure_mdconfig_options.
"""
import logging

from .mdengine import MDEngine
from .mdconfig import MDConfig
from .trajectory.trajectory import Trajectory
from .gromacs import utils as gmx_utils
from .gromacs import mdengine as gmx_engine
from .gromacs import mdconfig as gmx_config


logger = logging.getLogger(__name__)


async def get_all_traj_parts(folder: str, deffnm: str, engine: MDEngine | type[MDEngine],
                             ) -> list[Trajectory]:
    """
    List all trajectories in folder by given engine class with given deffnm.

    Parameters
    ----------
    folder : str
        Absolute or relative path to a folder.
    deffnm : str
        deffnm used by the engines simulation run from which we want the trajectories.
    engine : MDEngine | type[MDEngine]
        The engine that produced the trajectories (or one from the same class
        and with similar init args). Note that it is also possible to pass an
        uninitialized engine class, but then the default trajectory output type
        will be returned.

    Returns
    -------
    list[Trajectory]
        All trajectory parts from folder that match deffnm and engine in order.

    Raises
    ------
    ValueError
        Raised when the engine class is unknown.
    """
    # test for uninitialized engine classes, we warn but return the default traj type
    if isinstance(engine, type) and issubclass(engine, MDEngine):
        logger.warning("Engine %s is not initialized, i.e. it is an engine class. "
                       "Returning the default output trajectory type for this "
                       "engine class.", engine)
    if (
        isinstance(engine, (gmx_engine.GmxEngine, gmx_engine.SlurmGmxEngine))
        or (isinstance(engine, type)  # check that it is a type otherwise issubclass might not work
            and issubclass(engine, (gmx_engine.GmxEngine, gmx_engine.SlurmGmxEngine))
            )
    ):
        return await gmx_utils.get_all_traj_parts(folder=folder, deffnm=deffnm,
                                                  traj_type=engine.output_traj_type,
                                                  )
    raise ValueError(f"Engine {engine} is not a known MDEngine class."
                     + " Maybe someone just forgot to add the function?")


async def get_all_file_parts(folder: str, deffnm: str, file_ending: str,
                             engine: MDEngine | type[MDEngine],
                             ) -> list[str]:
    """
    Find and return all files with given ending produced by a `MDEngine`.

    NOTE: This returns only the parts that exist in ascending order.

    Parameters
    ----------
    folder : str
        Path to a folder to search for trajectory parts.
    deffnm : str
        deffnm (prefix of filenames) used in the simulation.
    file_ending : str
        File ending of the requested filetype (with or without preceding ".").
    engine : MDEngine | type[MDEngine]
        The engine or engine class that produced the file parts.

    Returns
    -------
    list[str]
        Ordered list of filepaths for files with given ending.
    """
    if (
        isinstance(engine, (gmx_engine.GmxEngine, gmx_engine.SlurmGmxEngine))
        or (isinstance(engine, type)  # check that it is a type otherwise issubclass might not work
            and issubclass(engine, (gmx_engine.GmxEngine, gmx_engine.SlurmGmxEngine))
            )
    ):
        return await gmx_utils.get_all_file_parts(folder=folder, deffnm=deffnm,
                                                  file_ending=file_ending)
    raise ValueError(f"Engine {engine} is not a known MDEngine (class)."
                     + " Maybe someone just forgot to add the function?")


def nstout_from_mdconfig(mdconfig: MDConfig, output_traj_type: str) -> int:
    """
    Return output step for given mdconfig and output_traj_type.

    Parameters
    ----------
    mdconfig : MDConfig
        An engine specific subclass of MDConfig.
    output_traj_type : str
        The output trajectory type for which we want the output frequency.

    Returns
    -------
    int
        (Smallest) output step in integration steps.

    Raises
    ------
    ValueError
        Raised when the MDConfig subclass is not known.
    """
    if isinstance(mdconfig, gmx_config.MDP):
        return gmx_utils.nstout_from_mdp(mdp=mdconfig,
                                         traj_type=output_traj_type,
                                         )
    raise ValueError(f"mdconfig {mdconfig} is not a known MDConfig class."
                     + " Maybe someone just forgot to add the function?")


def ensure_mdconfig_options(mdconfig: MDConfig, genvel: str = "no",
                            continuation: str = "yes") -> MDConfig:
    """
    Ensure that some commonly used mdconfig options have the given values.

    NOTE: Modifies the `MDConfig` inplace and returns it.

    Parameters
    ----------
    mdconfig : MDConfig
        Config object for which values should be ensured.
    genvel : str, optional
        Whether to generate velocities from a Maxwell-Boltzmann distribution
        ("yes" or "no"), by default "no".
    continuation : str, optional
        Whether to apply constraints to the initial configuration
        ("yes" or "no"), by default "yes"

    Returns
    -------
    MDConfig
        Reference to input config object with values for options as given.

    Raises
    ------
    ValueError
        If the MDConfig belongs to an unknown subclass not dispatchable to any
        specific engine submodule.
    """
    if isinstance(mdconfig, gmx_config.MDP):
        return gmx_utils.ensure_mdp_options(mdp=mdconfig, genvel=genvel,
                                            continuation=continuation,
                                            )
    raise ValueError(f"mdconfig {mdconfig} is not a known MDConfig class."
                     + " Maybe someone just forgot to add the function?")
