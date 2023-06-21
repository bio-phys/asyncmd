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
from .mdengine import MDEngine
from .mdconfig import MDConfig
from .trajectory.trajectory import Trajectory
from .gromacs import utils as gmx_utils
from .gromacs import mdengine as gmx_engine
from .gromacs import mdconfig as gmx_config


async def get_all_traj_parts(folder: str, deffnm: str, engine: MDEngine) -> "list[Trajectory]":
    """
    List all trajectories in folder by given engine class with given deffnm.

    Parameters
    ----------
    folder : str
        Absolute or relative path to a folder.
    deffnm : str
        deffnm used by the engines simulation run from which we want the trajs.
    engine : MDEngine
        The engine that produced the trajectories
        (or one from the same class and with similar init args)

    Returns
    -------
    list[Trajectory]
        All trajectory parts from folder that match deffnm and engine in order.

    Raises
    ------
    ValueError
        Raised when the engine class is unknown.
    """
    if isinstance(engine, (gmx_engine.GmxEngine, gmx_engine.SlurmGmxEngine)):
        return await gmx_utils.get_all_traj_parts(folder=folder, deffnm=deffnm,
                                                  traj_type=engine.output_traj_type,
                                                  )
    else:
        raise ValueError(f"Engine {engine} is not a known MDEngine class."
                         + " Maybe someone just forgot to add the function?")


async def get_all_file_parts(folder: str, deffnm: str, file_ending: str,
                             ) -> "list[str]":
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
        File ending of the requested filetype (with or without preceeding ".").

    Returns
    -------
    list[str]
        Ordered list of filepaths for files with given ending.
    """
    # TODO: we just use the function from the gromacs engines for now, i.e. we
    #       assume that the filename scheme will be the same for other engines
    return await gmx_utils.get_all_file_parts(folder=folder, deffnm=deffnm,
                                              file_ending=file_ending)


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
    else:
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
        If the MDConfig belongs to an unknown subclass not dispatcheable to any
        specific engine submodule.
    """
    if isinstance(mdconfig, gmx_config.MDP):
        return gmx_utils.ensure_mdp_options(mdp=mdconfig, genvel=genvel,
                                            continuation=continuation,
                                            )
    else:
        raise ValueError(f"mdconfig {mdconfig} is not a known MDConfig class."
                         + " Maybe someone just forgot to add the function?")
