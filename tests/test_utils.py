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

from conftest import NoOpMDEngine, NoOpMDConfig
from asyncmd.utils import (get_all_traj_parts, nstout_from_mdconfig,
                           ensure_mdconfig_options,
                           )


class Test_raise_for_unknown_engine:
    @pytest.mark.asyncio
    async def test_get_all_traj_parts(self):
        with pytest.raises(ValueError):
            await get_all_traj_parts(folder="test", deffnm="test",
                                     engine=NoOpMDEngine(),
                                     )

    def test_nstout_from_mdconfig(self):
        with pytest.raises(ValueError):
            nstout_from_mdconfig(mdconfig=NoOpMDConfig(),
                                 output_traj_type="TEST")

    def test_ensure_mdconfig_options(self):
        with pytest.raises(ValueError):
            ensure_mdconfig_options(mdconfig=NoOpMDConfig())
