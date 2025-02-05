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

from unittest.mock import PropertyMock, patch

import pytest

from asyncmd.slurm import SlurmClusterMediator, SlurmProcess


class MockSlurmExecCompleted:
    async def communicate(self):
        return (b"15283217||||COMPLETED||||0:0||||ravc4011||||\n", b"")


async def mock_slurm_call_completed(*args, **kwargs):
    return MockSlurmExecCompleted()


@patch("asyncio.subprocess.create_subprocess_exec", new=mock_slurm_call_completed)
@patch("os.path.abspath", return_value="/usr/bin/true")
@patch("time.time", return_value=7)
@pytest.mark.asyncio
async def test_get_info_for_job_completed(mock_time, mock_abspath):
    slurm_cluster_mediator = SlurmClusterMediator()
    mock_abspath.assert_called()
    slurm_cluster_mediator._jobids = ["15283217"]
    slurm_cluster_mediator._jobids_sacct = ["15283217"]
    slurm_cluster_mediator._jobinfo = {"15283217": {"state": "RUNNING"}}

    # Call the function to update cache and get new result
    job_info = await slurm_cluster_mediator.get_info_for_job("15283217")
    mock_time.assert_called()

    # Check if _update_cached_jobinfo() was additionally called by get_info_for_job()
    assert job_info["nodelist"] == ["ravc4011"]
    assert job_info["exitcode"] == "0:0"
    assert job_info["state"] == "COMPLETED"
    assert job_info["parsed_exitcode"] == 0


class MockSlurmExecFailed:
    async def communicate(self):
        return (b"15283217||||FAILED||||1:15||||ravc4007||||\n", b"")


async def mock_slurm_call_failed(*args, **kwargs):
    return MockSlurmExecFailed()


@patch("asyncio.subprocess.create_subprocess_exec", new=mock_slurm_call_failed)
@patch("os.path.abspath", return_value="/usr/bin/true")
@patch("time.time", return_value=7)
@pytest.mark.asyncio
async def test_get_info_for_job_failed(mock_time, mock_abspath):
    slurm_cluster_mediator = SlurmClusterMediator()
    slurm_cluster_mediator._all_nodes = [
        "ravc4001",
        "ravc4002",
        "ravc4003",
        "ravc4004",
        "ravc4005",
        "ravc4006",
        "ravc4007",
        "ravc4008",
        "ravc4009",
        "ravc4010",
    ]
    slurm_cluster_mediator._exclude_nodes = [
        "ravc4001",
        "ravc4002",
        "ravc4003",
        "ravc4004",
        "ravc4005",
        "ravc4006",
    ]
    slurm_cluster_mediator._jobids = ["15283217"]
    slurm_cluster_mediator._jobids_sacct = ["15283217"]
    slurm_cluster_mediator._jobinfo = {"15283217": {"state": "RUNNING"}}

    job_info = await slurm_cluster_mediator.get_info_for_job("15283217")

    assert job_info["nodelist"] == ["ravc4007"]
    assert job_info["exitcode"] == "1:15"
    assert job_info["state"] == "FAILED"
    assert job_info["parsed_exitcode"] == 1


class MockSubprocess:
    async def communicate(self):
        return (b"", b"")

    @property
    def returncode(self):
        return 0


@patch("asyncmd.slurm.SlurmProcess.slurm_cluster_mediator", new_callable=PropertyMock)
@patch("os.path.isfile", return_value=True)
@patch("os.path.abspath", return_value="/usr/bin/true")
@patch("asyncmd.slurm.logger")
@patch("subprocess.check_output", return_value="node1\nnode2\n")
@patch("asyncio.create_subprocess_exec", return_value=MockSubprocess())
def test_terminate(
    mock_create_subprocess_exec,
    mock_check_output,
    mock_logger,
    mock_isfile,
    mock_abspath,
    mock_slurm_cluster_mediator,
):
    slurm_process = SlurmProcess(jobname="test", sbatch_script="/usr/bin/true")
    slurm_process._jobid = ["15283217"]

    slurm_process.terminate()

    mock_logger.debug.assert_called()
