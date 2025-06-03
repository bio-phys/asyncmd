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

import pytest

from asyncmd.gromacs.mdconfig import MDP
from asyncmd.gromacs.mdengine import GmxEngine


@pytest.mark.asyncio
async def test_prepare_simulation():
    mdconfig = MDP("tests/test_data/gromacs/empty.mdp")
    gro = "examples/resources/gromacs/capped_alanine_dipeptide/conf.gro"
    top = "examples/resources/gromacs/capped_alanine_dipeptide/topol_amber99sbildn.top"
    cpt = "examples/resources/gromacs/capped_alanine_dipeptide/test.cpt"

    assert os.path.isfile(cpt)

    mdconfig._config = {
        "nstxout-compressed": 100,
        "simulation-part": 2,
    }

    workdir = os.path.dirname(gro)
    mdp = os.path.join(workdir, "test.mdp")
    mdout_mdp = os.path.join(workdir, "test_mdout.mdp")
    tpr = os.path.join(workdir, "test.tpr")

    engine = GmxEngine(mdconfig=mdconfig, gro_file=gro, top_file=top)

    await engine.prepare(None, workdir, "test")

    assert os.path.isfile(mdp)
    os.remove(mdp)

    assert os.path.isfile(mdout_mdp)
    os.remove(mdout_mdp)

    assert os.path.isfile(tpr)
    os.remove(tpr)
