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
import os
import shlex
from unittest.mock import patch, PropertyMock, Mock, AsyncMock, ANY

from asyncmd.slurm import SlurmProcess
from asyncmd.slurm.cluster_mediator import SlurmClusterMediator
from asyncmd.slurm.constants_and_errors import SlurmSubmissionError, SlurmError


class Test_SlurmProcess:
    def test_init_error_without_slurm(self, monkeypatch):
        with monkeypatch.context() as m:
            # monkeypatch to make sure we can execute the tests without slurm
            # (SlurmProcess checks if sbatch and friends are executable at init)
            # i.e. make sure everything works as expected with and without slurm
            m.setattr("asyncmd.slurm.process.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            # But set the cluster_mediator to None to test the init error
            m.setattr("asyncmd.slurm.SlurmProcess.slurm_cluster_mediator",
                      None)
            with pytest.raises(SlurmError):
                _ = SlurmProcess(jobname="test", sbatch_script="/usr/bin/true")

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
    def test__sanitize_sbatch_options_remove_protected(
                            self,
                            add_non_protected_sbatch_options_to_keep,
                            sbatch_options, opt_name, expected_opt_len,
                            caplog, monkeypatch
                            ):
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
            # mock the cluster_mediator to be able to initialize the SlurmProcess
            mediator_mock = Mock(SlurmClusterMediator)
            m.setattr("asyncmd.slurm.SlurmProcess.slurm_cluster_mediator",
                      mediator_mock)
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
            # mock the cluster_mediator to be able to initialize the SlurmProcess
            mediator_mock = Mock(SlurmClusterMediator)
            m.setattr("asyncmd.slurm.SlurmProcess.slurm_cluster_mediator",
                      mediator_mock)
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
            # mock the cluster_mediator to be able to initialize the SlurmProcess
            mediator_mock = Mock(SlurmClusterMediator)
            m.setattr("asyncmd.slurm.SlurmProcess.slurm_cluster_mediator",
                      mediator_mock)
            slurm_proc = SlurmProcess(jobname="test",
                                      sbatch_script="/usr/bin/true",
                                      time=time_in_h)
        slurm_timelimit = slurm_proc._slurm_timelimit_from_time_in_hours(
                                                                time=time_in_h)
        assert slurm_timelimit == beauty

    @pytest.mark.parametrize(["time", "time_beauty", "exclude_nodes"],
                             [(None, "", []),
                              (1., "60:0", ["ravc4011"]),
                              (1., "60:0", ["ravc4011", "ravc4012"]),
                              ]
                             )
    @pytest.mark.asyncio
    async def test_submit(self, monkeypatch, time, time_beauty, exclude_nodes):
        with monkeypatch.context() as m:
            # monkeypatch to make sure we can execute the tests without slurm
            # (SlurmProcess checks if sbatch and friends are executable at init)
            m.setattr("asyncmd.slurm.process.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            mediator_mock = Mock(SlurmClusterMediator)
            type(mediator_mock).exclude_nodes = PropertyMock(return_value=exclude_nodes)
            m.setattr("asyncmd.slurm.SlurmProcess.slurm_cluster_mediator",
                      mediator_mock)
            slurm_proc = SlurmProcess(jobname="test",
                                      sbatch_script="/usr/bin/true",
                                      time=time,
                                      )
            # this will be the mock for the returned Process object that
            # asyncio.create_subprocess_exec returns
            mock_sbatch_asyncio_subprocess = AsyncMock(
                                        )
            # it needs to have a communicate method (that returns the jobid as stdout)
            # and should have the returncode property set to zero
            # NOTE: ideally only set returncode after calling communicate,
            #       but this should not matter too much
            mock_sbatch_asyncio_subprocess.configure_mock(
                **{"communicate.return_value": (b"12345", b""),
                   }
                )
            type(mock_sbatch_asyncio_subprocess).returncode = PropertyMock(return_value=0)
            # this mocks asyncio.create_subproccess_exec with sbatch as subprocess
            mock_slurm_sbatch_exec_completed = AsyncMock(
                                        return_value=mock_sbatch_asyncio_subprocess,
                                        )
            m.setattr("asyncio.create_subprocess_exec", mock_slurm_sbatch_exec_completed)
            await slurm_proc.submit()
            # make sure we called sbatch correctly
            # TODO: we fix the order here... is this what we want?!
            sbatch_cmd_beauty = "sbatch --job-name='test' "
            sbatch_cmd_beauty += f"--chdir='{os.path.abspath(os.getcwd())}' "
            sbatch_cmd_beauty += "--output='./test.out.%j' --error='./test.err.%j' "
            if time is not None:
                sbatch_cmd_beauty += f"--time='{time_beauty}' "
            if exclude_nodes:
                sbatch_cmd_beauty += f"--exclude={','.join(exclude_nodes)} "
            sbatch_cmd_beauty += "--parsable /usr/bin/true"
            mock_slurm_sbatch_exec_completed.assert_awaited_once_with(
                                                    *shlex.split(sbatch_cmd_beauty),
                                                    stdout=ANY,
                                                    stderr=ANY,
                                                    cwd=os.path.abspath(os.getcwd()),
                                                    close_fds=True,
                                                    )
            # make sure we called communicate on the Process
            mock_sbatch_asyncio_subprocess.communicate.assert_called_once()

    @pytest.mark.parametrize("sbatch_return",
                             ["non-zero", "non-int"])
    @pytest.mark.asyncio
    async def test_submit_fail(self, monkeypatch, sbatch_return):
        with monkeypatch.context() as m:
            # monkeypatch to make sure we can execute the tests without slurm
            # (SlurmProcess checks if sbatch and friends are executable at init)
            m.setattr("asyncmd.slurm.process.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            mediator_mock = Mock(SlurmClusterMediator)
            type(mediator_mock).exclude_nodes = PropertyMock(return_value=set())
            m.setattr("asyncmd.slurm.SlurmProcess.slurm_cluster_mediator",
                      mediator_mock)
            slurm_proc = SlurmProcess(jobname="test",
                                      sbatch_script="/usr/bin/true",
                                      )
            # this will be the mock for the returned Process object that
            # asyncio.create_subprocess_exec returns
            mock_sbatch_asyncio_subprocess = AsyncMock(
                                        )
            # it needs to have a communicate method (that returns the jobid as stdout)
            # and should have the returncode property set to (non-)zero
            # NOTE: ideally only set returncode after calling communicate,
            #       but this should not matter too much
            if sbatch_return == "non-zero":
                # make sure that communicate returns something sane nevertheless
                mock_attrs = {"communicate.return_value": (b"12345", b""),
                              }
                # and we fail only because of returncode non-zero!
                return_code = 1
            elif sbatch_return == "non-int":
                # make sure sbatch communicate returns something not parsable as int
                mock_attrs = {"communicate.return_value": (b"SMTH WENT WRONG", b"SMTH WENT WRONG"),
                              }
                # but make sure returncode is zero so we dont fail because of it!
                return_code = 0
            mock_sbatch_asyncio_subprocess.configure_mock(**mock_attrs)
            type(mock_sbatch_asyncio_subprocess).returncode = PropertyMock(return_value=return_code)
            # this mocks asyncio.create_subproccess_exec with sbatch as subprocess
            mock_slurm_sbatch_exec_completed = AsyncMock(
                                        return_value=mock_sbatch_asyncio_subprocess,
                                        )
            m.setattr("asyncio.create_subprocess_exec", mock_slurm_sbatch_exec_completed)
            with pytest.raises(SlurmSubmissionError):
                # this should rais as our returncode is non-zero
                await slurm_proc.submit()
            # but still make sure we called sbatch correctly
            # TODO: we fix the order here... is this what we want?!
            sbatch_cmd_beauty = "sbatch --job-name='test' "
            sbatch_cmd_beauty += f"--chdir='{os.path.abspath(os.getcwd())}' "
            sbatch_cmd_beauty += "--output='./test.out.%j' --error='./test.err.%j' "
            sbatch_cmd_beauty += "--parsable /usr/bin/true"
            mock_slurm_sbatch_exec_completed.assert_awaited_once_with(
                                                    *shlex.split(sbatch_cmd_beauty),
                                                    stdout=ANY,
                                                    stderr=ANY,
                                                    cwd=os.path.abspath(os.getcwd()),
                                                    close_fds=True,
                                                    )
            # make sure we called communicate on the Process
            mock_sbatch_asyncio_subprocess.communicate.assert_called_once()


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


@patch("asyncmd.slurm.SlurmProcess.slurm_cluster_mediator", new=Mock(SlurmClusterMediator))
@patch("os.path.isfile", return_value=True)
@patch("os.path.abspath", return_value="/usr/bin/true")
@patch("asyncmd.slurm.process.logger")
@patch("subprocess.check_output", return_value="node1\nnode2\n")
def test_terminate(
    mock_check_output,
    mock_logger,
    mock_isfile,
    mock_abspath,
):
    slurm_process = SlurmProcess(jobname="test", sbatch_script="/usr/bin/true")
    slurm_process._jobid = "15283217"

    slurm_process.terminate()

    mock_logger.debug.assert_called()
