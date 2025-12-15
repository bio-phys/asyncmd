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
Tests for file asyncmd.utils

Here we only test that the dispatch to the respective submodules works,
i.e. no tests for, e.g., functions from asyncmd.gromacs.utils.
"""
import pytest
import logging

from conftest import NoOpMDEngine, NoOpMDConfig
from asyncmd.utils import (get_all_traj_parts,
                           get_all_file_parts,
                           nstout_from_mdconfig,
                           ensure_mdconfig_options,
                           )


class Test_raise_for_unknown_engine:
    @pytest.mark.asyncio
    async def test_get_all_traj_parts(self):
        with pytest.raises(ValueError):
            await get_all_traj_parts(folder="test", deffnm="test",
                                     engine=NoOpMDEngine(),
                                     )

    @pytest.mark.asyncio
    async def test_get_all_file_parts(self):
        with pytest.raises(ValueError):
            await get_all_file_parts(folder="test", deffnm="test",
                                     file_ending=".test", engine=NoOpMDEngine(),
                                     )

    def test_nstout_from_mdconfig(self):
        with pytest.raises(ValueError):
            nstout_from_mdconfig(mdconfig=NoOpMDConfig(),
                                 output_traj_type="TEST")

    def test_ensure_mdconfig_options(self):
        with pytest.raises(ValueError):
            ensure_mdconfig_options(mdconfig=NoOpMDConfig())


class Test_warn_for_default_value_from_engine_class:
    @pytest.mark.asyncio
    async def test_get_all_traj_parts(self, caplog):
        with pytest.raises(ValueError):
            with caplog.at_level(logging.WARNING):
                await get_all_traj_parts(folder="test", deffnm="test",
                                         # this time we use an uninitialized
                                         # engine class so we get the warning
                                         # (and then fail after)
                                         engine=NoOpMDEngine,
                                         )
        warn_text = f"Engine {NoOpMDEngine} is not initialized, i.e. it is an engine class. "
        warn_text += "Returning the default output trajectory type for this engine class."
        assert warn_text in caplog.text
