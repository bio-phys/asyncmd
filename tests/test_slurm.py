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
import pytest
import logging
from unittest.mock import patch, PropertyMock

import asyncmd
from asyncmd.slurm import SlurmProcess
from asyncmd.slurm.cluster_mediator import SlurmClusterMediator


LOGGER = logging.getLogger(__name__)


class Test_SlurmProcess:
    @pytest.mark.parametrize("add_non_protected_sbatch_options_to_keep", [True, False])
    @pytest.mark.parametrize(["sbatch_options", "opt_name", "expected_opt_len"],
                             [({"job-name": "TO_REMOVE"}, "job-name", 0),
                              ({"chdir": "TO_REMOVE"}, "chdir", 0),
                              ({"output": "TO_REMOVE"}, "output", 0),
                              ({"error": "TO_REMOVE"}, "error", 0),
                              ({"input": "TO_REMOVE"}, "input", 0),
                              ({"exclude": "TO_REMOVE"}, "exclude", 0),
                              ({"parsable": "TO_REMOVE"}, "parsable", 0),
                              ({"keep_option": "TO_KEEP"}, "keep_option", 1),
                              ]
                             )
    def test__sanitize_sbatch_options_remove_protected(self,
                            add_non_protected_sbatch_options_to_keep,
                            sbatch_options, opt_name, expected_opt_len,
                            caplog, monkeypatch):
        if add_non_protected_sbatch_options_to_keep:
            # add a dummy option that we want to keep in the dict
            # (Need to redefine the dict [mot use update] to not change the
            #  originally passed in value from parametrize for the next round)
            sbatch_options = dict({"other_keep_option": "TO_KEEP_TOO"},
                                  **sbatch_options)
            # same here, dont use += 1
            expected_opt_len = expected_opt_len + 1
        with monkeypatch.context() as m:
            # monkeypatch to make sure we can execute the tests without slurm
            # (SlurmProcess checks if sbatch and friends are executable at init)
            m.setattr("asyncmd.slurm.process.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            with caplog.at_level(logging.WARNING):
                slurm_proc = SlurmProcess(jobname="test",
                                          sbatch_script="/usr/bin/true",
                                          sbatch_options=sbatch_options)
        # make sure we removed the option, it is the only one so sbatch_options
        # must now be an empty dict
        assert len(slurm_proc.sbatch_options) == expected_opt_len
        # and make sure we got the warning when we should
        # (i.e. if we should have removed something)
        if len(sbatch_options) != expected_opt_len:
            warn_str = f"Removing sbatch option '{opt_name}' from 'sbatch_options'"
            warn_str += " because it is used internally by the `SlurmProcess`."
            assert warn_str in caplog.text

    @pytest.mark.parametrize(["sbatch_options", "time", "expect_warn"],
                             [({"time": "20:00"}, None, False),
                              ({"time": "15:00"}, 0.25, True),
                              ({}, 0.25, False),
                              ]
                             )
    def test__sanitize_sbatch_options_remove_time(self, sbatch_options, time,
                                                  expect_warn,
                                                  caplog, monkeypatch):
        with monkeypatch.context() as m:
            # monkeypatch to make sure we can execute the tests without slurm
            # (SlurmProcess checks if sbatch and friends are executable at init)
            m.setattr("asyncmd.slurm.process.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            with caplog.at_level(logging.DEBUG):
                slurm_proc = SlurmProcess(jobname="test",
                                          sbatch_script="/usr/bin/true",
                                          time=time,
                                          sbatch_options=sbatch_options)
        # make sure we remove time from sbatch_options if given separately
        if time is not None:
            assert len(slurm_proc.sbatch_options) == 0
        # make sure we get the warning when we remove it due to double option
        if expect_warn:
            warn_str = "Removing sbatch option 'time' from 'sbatch_options'. "
            warn_str += "Using the 'time' argument instead."
            assert warn_str in caplog.text
        # NOTE: we dont check for the debug log, but this is what we could do
        if time is None and not expect_warn:
            debug_str = "Using 'time' from 'sbatch_options' because self.time is None."
            assert debug_str in caplog.text

    @pytest.mark.parametrize(["time_in_h", "beauty"],
                             [(0.25, "15:0"),
                              (1, "60:0"),
                              (10, "600:0"),
                              (1/3600, "0:1"),
                              (15/3600, "0:15"),
                              (1/60, "1:0"),
                              (5/60, "5:0"),
                              ]
                             )
    def test__slurm_timelimit_from_time_in_hours(self, time_in_h, beauty,
                                                 monkeypatch):
        with monkeypatch.context() as m:
            # monkeypatch to make sure we can execute the tests without slurm
            # (SlurmProcess checks if sbatch and friends are executable at init)
            m.setattr("asyncmd.slurm.process.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            slurm_proc = SlurmProcess(jobname="test",
                                      sbatch_script="/usr/bin/true",
                                      time=time_in_h)
        slurm_timelimit = slurm_proc._slurm_timelimit_from_time_in_hours(
                                                                time=time_in_h)
        assert slurm_timelimit == beauty


class MockSlurmExecCompleted:
    async def communicate(self):
        return (b"15283217||||COMPLETED||||0:0||||ravc4011||||\n", b"")


async def mock_slurm_call_completed(*args, **kwargs):
    return MockSlurmExecCompleted()


@patch("asyncio.create_subprocess_exec", new=mock_slurm_call_completed)
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


@patch("asyncio.create_subprocess_exec", new=mock_slurm_call_failed)
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
@patch("asyncmd.slurm.process.logger")
@patch("subprocess.check_output", return_value="node1\nnode2\n")
def test_terminate(
    mock_check_output,
    mock_logger,
    mock_isfile,
    mock_abspath,
    mock_slurm_cluster_mediator,
):
    slurm_process = SlurmProcess(jobname="test", sbatch_script="/usr/bin/true")
    slurm_process._jobid = "15283217"

    slurm_process.terminate()

    mock_logger.debug.assert_called()
