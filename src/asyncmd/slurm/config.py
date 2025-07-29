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
This file contains functions to set/change the configuration of the slurm module.

- set_all_slurm_settings
- set_slurm_settings

These functions are also imported in "asyncmd.config", i.e. the central place
for configuration functions.
"""
from .process import SlurmProcess
from .cluster_mediator import SlurmClusterMediator


# pylint: disable-next=too-many-arguments
def set_all_slurm_settings(*, sinfo_executable: str = "sinfo",
                           sacct_executable: str = "sacct",
                           sbatch_executable: str = "sbatch",
                           scancel_executable: str = "scancel",
                           min_time_between_sacct_calls: int = 10,
                           num_fails_for_broken_node: int = 3,
                           success_to_fail_ratio: int = 50,
                           exclude_nodes: list[str] | None = None,
                           ) -> None:
    """
    (Re) initialize all settings relevant for SLURM job control.

    Call this function if you want to change e.g. the path/name of SLURM
    executables. Note that this is a convenience function to set all SLURM
    settings in one central place and all at once, i.e. calling this function
    will overwrite all previous settings.
    If this is not intended, have a look at the `set_slurm_settings` function
    which only changes the passed arguments or you can also set/modify each
    setting separately in the `SlurmProcess` and `SlurmClusterMediator` classes.

    Parameters
    ----------
    sinfo_executable : str, optional
        Name of path to the sinfo executable, by default "sinfo".
    sacct_executable : str, optional
        Name or path to the sacct executable, by default "sacct".
    sbatch_executable : str, optional
        Name or path to the sbatch executable, by default "sbatch".
    scancel_executable : str, optional
        Name or path to the scancel executable, by default "scancel".
    min_time_between_sacct_calls : int, optional
        Minimum time (in seconds) between subsequent sacct calls,
        by default 10.
    num_fails_for_broken_node : int, optional
        Number of failed jobs we need to observe per node before declaring it
        to be broken (and not submitting any more jobs to it), by default 3.
    success_to_fail_ratio : int, optional
        Number of successful jobs we need to observe per node to decrease the
        failed job counter by one, by default 50.
    exclude_nodes : list[str] or None, optional
        List of nodes to exclude in job submissions, by default None, which
        results in no excluded nodes.
    """
    # pylint: disable-next=global-variable-not-assigned
    global SlurmProcess
    SlurmProcess.slurm_cluster_mediator = SlurmClusterMediator(
                    sinfo_executable=sinfo_executable,
                    sacct_executable=sacct_executable,
                    min_time_between_sacct_calls=min_time_between_sacct_calls,
                    num_fails_for_broken_node=num_fails_for_broken_node,
                    success_to_fail_ratio=success_to_fail_ratio,
                    exclude_nodes=exclude_nodes,
                                                                )
    SlurmProcess.sbatch_executable = sbatch_executable
    SlurmProcess.scancel_executable = scancel_executable


# pylint: disable-next=too-many-arguments
def set_slurm_settings(*, sinfo_executable: str | None = None,
                       sacct_executable: str | None = None,
                       sbatch_executable: str | None = None,
                       scancel_executable: str | None = None,
                       min_time_between_sacct_calls: int | None = None,
                       num_fails_for_broken_node: int | None = None,
                       success_to_fail_ratio: int | None = None,
                       exclude_nodes: list[str] | None = None,
                       ) -> None:
    """
    Set single or multiple settings relevant for SLURM job control.

    Call this function if you want to change e.g. the path/name of SLURM
    executables. This function only modifies those settings for which a value
    other than None is passed. See `set_all_slurm_settings` if you want to set/
    modify all slurm settings and/or reset them to their defaults.

    Parameters
    ----------
    sinfo_executable : str, optional
        Name of path to the sinfo executable, by default None.
    sacct_executable : str, optional
        Name or path to the sacct executable, by default None.
    sbatch_executable : str, optional
        Name or path to the sbatch executable, by default None.
    scancel_executable : str, optional
        Name or path to the scancel executable, by default None.
    min_time_between_sacct_calls : int, optional
        Minimum time (in seconds) between subsequent sacct calls,
        by default None.
    num_fails_for_broken_node : int, optional
        Number of failed jobs we need to observe per node before declaring it
        to be broken (and not submitting any more jobs to it), by default None.
    success_to_fail_ratio : int, optional
        Number of successful jobs we need to observe per node to decrease the
        failed job counter by one, by default None.
    exclude_nodes : list[str], optional
        List of nodes to exclude in job submissions, by default None, which
        results in no excluded nodes.
    """
    # pylint: disable-next=global-variable-not-assigned
    global SlurmProcess
    # collect options for slurm cluster mediator
    mediator_options: dict[str, str | int | list[str]] = {}
    if sinfo_executable is not None:
        mediator_options["sinfo_executable"] = sinfo_executable
    if sacct_executable is not None:
        mediator_options["sacct_executable"] = sacct_executable
    if min_time_between_sacct_calls is not None:
        mediator_options["min_time_between_sacct_calls"] = min_time_between_sacct_calls
    if num_fails_for_broken_node is not None:
        mediator_options["num_fails_for_broken_node"] = num_fails_for_broken_node
    if success_to_fail_ratio is not None:
        mediator_options["success_to_fail_ratio"] = success_to_fail_ratio
    if exclude_nodes is not None:
        mediator_options["exclude_nodes"] = exclude_nodes
    # and set them either on a new mediator or on the already initialized class
    if SlurmProcess.slurm_cluster_mediator is None:
        # initialize the mediator with given options (all else will be default)
        SlurmProcess.slurm_cluster_mediator = SlurmClusterMediator(**mediator_options)
    else:
        # set the given options on the already initialized class
        for opt, val in mediator_options.items():
            setattr(SlurmProcess.slurm_cluster_mediator, opt, val)
    # and set the two SlurmProcess attributes directly
    if sbatch_executable is not None:
        SlurmProcess.sbatch_executable = sbatch_executable
    if scancel_executable is not None:
        SlurmProcess.scancel_executable = scancel_executable
