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


class Test_trajectory:
    @pytest.mark.skip("TODO")
    @pytest.mark.parametrize(["traj_file", "struct_file", "truth"],
                             [("../test_data/trajectory/ala_traj.trr",
                               "../test_data/trajectory/ala.tpr",
                               {"nstout": None,
                                }
                               ),
                              ("../test_data/trajectory/ala_traj.xtc",
                               "../test_data/trajectory/ala.tpr",
                               {"nstout": None,
                                }
                               ),
                              ]
                             )
    def test_properties(self, traj_file, struct_file, truth):
        pass

    @pytest.mark.skip("TODO")
    def test_npz_cache(self):
        pass

    @pytest.mark.skip("TODO")
    def test_pickle(self):
        pass
