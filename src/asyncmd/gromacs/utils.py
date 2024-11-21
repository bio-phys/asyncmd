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
import logging
import aiofiles.os

from ..trajectory.trajectory import Trajectory
from .mdconfig import MDP


logger = logging.getLogger(__name__)


def nstout_from_mdp(mdp: MDP, traj_type: str = "TRR") -> int:
    """
    Get minimum number of steps between outputs for trajectories from MDP.

    Parameters
    ----------
    mdp : MDP
        Config object from which the output step should be read.
    traj_type : str, optional
        Trajectory format for which output step should be read, "XTC" or "TRR",
        by default "TRR".

    Returns
    -------
    int
        Minimum number of steps between two writes.

    Raises
    ------
    ValueError
        Raised when an unknown trajectory format `traj_type` is given.
    ValueError
        Raised when the given MDP would result in no output for the given
        trajectory format `traj_type`.
    """
    if traj_type.upper() == "TRR":
        keys = ["nstxout", "nstvout", "nstfout"]
    elif traj_type.upper() == "XTC":
        keys = ["nstxout-compressed", "nstxtcout"]
    else:
        raise ValueError("traj_type must be one of 'TRR' or 'XTC'.")

    vals = []
    for k in keys:
        try:
            # need to check for 0 (== no output!) in case somone puts the
            # defaults (or reads an mdout.mdp where gmx lists all the defaults)
            v = mdp[k]
            if v == 0:
                v = float("inf")
            vals += [v]
        except KeyError:
            # not set, defaults to 0, so we ignore it
            pass

    nstout = min(vals, default=None)
    if (nstout is None) or (nstout == float("inf")):
        raise ValueError(f"The MDP you passed results in no {traj_type} "
                         +"trajectory output.")
    return nstout


async def get_all_traj_parts(folder: str, deffnm: str,
                             traj_type: str = "TRR") -> "list[Trajectory]":
    """
    Find and return a list of trajectory parts produced by a GmxEngine.

    NOTE: This returns only the parts that exist in ascending order.

    Parameters
    ----------
    folder : str
        path to a folder to search for trajectory parts
    deffnm : str
        deffnm (prefix of filenames) used in the simulation
    traj_type : str, optional
        Trajectory file ending("XTC", "TRR", "TNG", ...), by default "TRR"

    Returns
    -------
    list[Trajectory]
        Ordered list of all trajectory parts with given deffnm and type.
    """
    ending = traj_type.lower()
    traj_files = await get_all_file_parts(folder=folder, deffnm=deffnm,
                                          file_ending=ending)
    trajs = [Trajectory(trajectory_files=traj_file,
                        structure_file=os.path.join(folder, f"{deffnm}.tpr")
                        )
             for traj_file in traj_files]
    return trajs


async def get_all_file_parts(folder: str, deffnm: str, file_ending: str) -> "list[str]":
    """
    Find and return all files with given ending produced by GmxEngine.

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
    def partnum_suffix(num):
        # construct gromacs num part suffix from simulation_part
        num_suffix = ".part{:04d}".format(num)
        return num_suffix

    if not file_ending.startswith("."):
        file_ending = "." + file_ending
    content = await aiofiles.os.listdir(folder)
    filtered = [f for f in content
                if (f.startswith(f"{deffnm}.part")
                    and f.endswith(file_ending)
                    and (len(f) == len(deffnm) + 9 + len(file_ending))
                    )
                ]
    partnums = [int(f[len(deffnm) + 5:len(deffnm) + 9])  # get the 4 number digits
                for f in filtered]
    partnums.sort()
    parts = [os.path.join(folder, f"{deffnm}{partnum_suffix(num)}{file_ending}")
             for num in partnums]
    return parts


def ensure_mdp_options(mdp: MDP, genvel: str = "no", continuation: str = "yes") -> MDP:
    """
    Ensure that some commonly used mdp options have the given values.

    NOTE: Modifies the `MDP` inplace and returns it.

    Parameters
    ----------
    mdp : MDP
        Config object for which values should be ensured.
    genvel : str, optional
        Value for genvel option ("yes" or "no"), by default "no".
    continuation : str, optional
        Value for continuation option ("yes" or "no"), by default "yes".

    Returns
    -------
    MDP
        Reference to input config object with values for options as given.
    """
    try:
        # make sure we do not generate velocities with gromacs
        genvel_test = mdp["gen-vel"]
    except KeyError:
        logger.info(f"Setting 'gen-vel = {genvel}' in mdp.")
        mdp["gen-vel"] = genvel
    else:
        if genvel_test != genvel:
            logger.warning(f"Setting 'gen-vel = {genvel}' in mdp "
                           + f"(was '{genvel_test}').")
            mdp["gen-vel"] = genvel
    try:
        # TODO/FIXME: this could also be 'unconstrained-start'!
        #             however already the gmx v4.6.3 docs say
        #            "continuation: formerly know as 'unconstrained-start'"
        #            so I think we can ignore that for now?!
        continuation_test = mdp["continuation"]
    except KeyError:
        logger.info(f"Setting 'continuation = {continuation}' in mdp.")
        mdp["continuation"] = continuation
    else:
        if continuation_test != continuation:
            logger.warning(f"Setting 'continuation = {continuation}' in mdp "
                           + f"(was '{continuation_test}').")
            mdp["continuation"] = continuation

    return mdp
