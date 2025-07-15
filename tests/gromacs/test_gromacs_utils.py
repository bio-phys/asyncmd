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
from asyncmd.gromacs import MDP
from asyncmd.gromacs import utils


@pytest.mark.parametrize(["file_name"], [("empty.mdp",), ("gen-vel-continuation.mdp",)])
def test_ensure_mdconfig_options(file_name: str):
    mdp_file = f"tests/test_data/gromacs/{file_name}"
    mdp = MDP(original_file=mdp_file)
    utils.ensure_mdp_options(mdp)
    assert mdp["gen-vel"] == "no"
    assert mdp["continuation"] == "yes"
