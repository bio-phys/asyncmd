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

import numpy as np
import pytest

from asyncmd import gromacs as asyncgmx
from asyncmd import trajectory as asynctraj


def condition_function(traj):
    return np.array([False, False, False, True, False])


@pytest.mark.asyncio
async def test_propagate():
    mdp = asyncgmx.MDP("tests/test_data/gromacs/empty.mdp")
    mdp["nstxout-compressed"] = 1
    mdp["nstxtcout"] = 1

    condition_function_wrapped = asynctraj.PyTrajectoryFunctionWrapper(
        condition_function
    )
    propa_somewhere = asynctraj.ConditionalTrajectoryPropagator(
        conditions=[condition_function_wrapped],
        engine_cls=asyncgmx.GmxEngine,
        engine_kwargs={
            "mdconfig": mdp,
        },
        walltime_per_part=0.01,
    )
    starting_configuration = asynctraj.trajectory.Trajectory(
        trajectory_files="tests/test_data/trajectory/ala_traj.xtc",
        structure_file="tests/test_data/trajectory/ala.gro",
    )
    workdir = "tests/trajectory"
    deffnm = "test_deffnm"

    trajectories, cond_fullfilled = await propa_somewhere.propagate(
        starting_configuration=starting_configuration, workdir=workdir, deffnm=deffnm
    )

    assert len(trajectories) > 0
    assert cond_fullfilled is not None
