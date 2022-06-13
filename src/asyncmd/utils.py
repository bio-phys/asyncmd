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


def get_all_traj_parts(folder: str, deffnm: str, engine: MDEngine) -> "list[Trajectory]":
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
        return gmx_utils.get_all_traj_parts(folder=folder, deffnm=deffnm,
                                            traj_type=engine.output_traj_type,
                                            )
    else:
        raise ValueError(f"Engine {engine} is not a known MDEngine class."
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
    else:
        raise ValueError(f"mdconfig {mdconfig} is not a known MDConfig class."
                         + " Maybe someone just forgot to add the function?")
