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
This file contains some general test helper classes, which should be imported
in conftest.py to make them available for all tests (also in subfolders).
"""
from asyncmd import Trajectory
from asyncmd.mdengine import MDEngine
from asyncmd.mdconfig import MDConfig


class NoOpMDEngine(MDEngine):
    """
    Just overwrite all abstract methods so we can instantiate.
    """
    current_trajectory = None
    output_traj_type = "TEST"
    steps_done = 0
    async def apply_constraints(self, conf_in: Trajectory, conf_out_name: str,
                                *, workdir: str = ".") -> Trajectory:
        pass
    async def prepare(self, starting_configuration: Trajectory, workdir: str, deffnm: str) -> None:
        pass
    async def prepare_from_files(self, workdir: str, deffnm: str) -> None:
        pass
    async def run_walltime(self, walltime: float, max_steps: int | None = None) -> Trajectory:
        pass
    async def run_steps(self, nsteps: int, steps_per_part: bool = False) -> Trajectory:
        pass


class NoOpMDConfig(MDConfig):
    """Overwrite all abstract methods so we can instantiate."""
    def parse(self):
        pass
    def write(self, outfile):
        pass
    def __getitem__(self, key):
        pass
    def __setitem__(self, key, value):
        pass
    def __delitem__(self, key):
        pass
    def __iter__(self):
        pass
    def __len__(self):
        pass
