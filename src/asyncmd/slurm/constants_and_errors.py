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
This file defines constants and Errors used throughout the slurm module.
"""


class SlurmError(RuntimeError):
    """Generic error superclass for all SLURM errors."""


class SlurmCancellationError(SlurmError):
    """Error raised when something goes wrong canceling a SLURM job."""


class SlurmSubmissionError(SlurmError):
    """Error raised when something goes wrong submitting a SLURM job."""


# rudimentary map for slurm state codes to int return codes for poll
# NOTE: these are the sacct states (they differ from the squeue states)
#       cf. https://slurm.schedmd.com/sacct.html#lbAG
#       and https://slurm.schedmd.com/squeue.html#lbAG
# NOTE on error codes:
#      we return:
#       - None if the job has not finished
#       - 0 if it completed successfully
#       - 1 if the job failed (probably) due to user error (or we dont know)
#       - 2 if the job failed (almost certainly) due to cluster/node-issues as
#         recognized/detected by slurm
SLURM_STATE_TO_EXITCODE: dict[str, int | None] = {
    "BOOT_FAIL": 1,  # Job terminated due to launch failure
    # Job was explicitly cancelled by the user or system administrator.
    "CANCELLED": 1,
    # Job has terminated all processes on all nodes with an exit code of
    # zero.
    "COMPLETED": 0,
    "DEADLINE": 1,  # Job terminated on deadline.
    # Job terminated with non-zero exit code or other failure condition.
    "FAILED": 1,
    # Job terminated due to failure of one or more allocated nodes.
    "NODE_FAIL": 2,
    "OUT_OF_MEMORY": 1,  # Job experienced out of memory error.
    "PENDING": None,  # Job is awaiting resource allocation.
    # NOTE: preemption means interrupting a process to later restart it,
    #       i.e. None is probably the right thing to return
    "PREEMPTED": None,  # Job terminated due to preemption.
    "RUNNING": None,  # Job currently has an allocation.
    "REQUEUED": None,  # Job was requeued.
    # TODO: when can resizing happen? what should we return? For now we go with not finished,
    #       i.e. None
    # Job is about to change size.
    "RESIZING": None,
    # Sibling was removed from cluster due to other cluster starting the job.
    "REVOKED": 1,
    # Job has an allocation, but execution has been suspended and CPUs have
    # been released for other jobs.
    "SUSPENDED": None,
    # Job terminated upon reaching its time limit.
    "TIMEOUT": 1,
}
