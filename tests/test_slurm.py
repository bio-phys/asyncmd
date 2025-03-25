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
import os.path as path
import logging
from unittest.mock import patch, Mock

import asyncmd
from asyncmd.slurm import SlurmClusterMediator, SlurmProcess


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
            # (Need to redefine the dict [mot use update] to not chnage the
            #  originally passed in value from parametrize for the next round)
            sbatch_options = dict({"other_keep_option": "TO_KEEP_TOO"},
                                  **sbatch_options)
            # same here, dont use += 1
            expected_opt_len = expected_opt_len + 1
        with monkeypatch.context() as m:
            m.setattr(path, "isfile", lambda _: True)
            m.setattr(path, "abspath", lambda _: "/usr/bin/true")
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
            warn_str += " because it is used internaly by the `SlurmProcess`."
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
            m.setattr(path, "isfile", lambda _: True)
            m.setattr(path, "abspath", lambda _: "/usr/bin/true")
            with caplog.at_level(logging.DEBUG):
                slurm_proc = SlurmProcess(jobname="test",
                                          sbatch_script="/usr/bin/true",
                                          time=time,
                                          sbatch_options=sbatch_options)
        # make sure we remove time from sbatch_options if given seperately
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
