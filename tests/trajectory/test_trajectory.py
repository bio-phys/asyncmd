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
import pickle
import numpy as np


from asyncmd import Trajectory


class Test_trajectory:
    @pytest.mark.parametrize(["traj_file", "struct_file", "truth"],
                             [("tests/test_data/trajectory/ala_traj.trr",
                               "tests/test_data/trajectory/ala.tpr",
                               {"nstout": None,
                                "first_step": 67740,
                                "last_step": 68080,
                                "dt": 0.04,
                                "first_time": 135.48,
                                "last_time": 136.16,
                                }
                               ),
                              ("tests/test_data/trajectory/ala_traj.xtc",
                               "tests/test_data/trajectory/ala.tpr",
                               {"nstout": None,
                                "first_step": 67740,
                                "last_step": 68080,
                                "dt": 0.04,
                                "first_time": 135.48,
                                "last_time": 136.16,
                                }
                               ),
                              ("tests/test_data/trajectory/ala_traj.trr",
                               "tests/test_data/trajectory/ala.gro",
                               {"nstout": None,
                                "first_step": 67740,
                                "last_step": 68080,
                                "dt": 0.04,
                                "first_time": 135.48,
                                "last_time": 136.16,
                                }
                               ),
                              ("tests/test_data/trajectory/ala_traj.xtc",
                               "tests/test_data/trajectory/ala.gro",
                               {"nstout": None,
                                "first_step": 67740,
                                "last_step": 68080,
                                "dt": 0.04,
                                "first_time": 135.48,
                                "last_time": 136.16,
                                }
                               ),
                              ]
                             )
    def test_properties(self, traj_file, struct_file, truth):
        traj = Trajectory(trajectory_file=traj_file,
                          structure_file=struct_file,
                          )
        for attr_name, truth_value in truth.items():
            test_val = getattr(traj, attr_name)
            if truth_value is None:
                assert test_val is truth_value
            elif isinstance(test_val, float):
                # TODO: is atol=1e-5 really what we want?
                # for our current tests it is fine I think
                assert np.isclose(test_val, truth_value, atol=1e-5, rtol=0)
            else:
                assert test_val == truth_value

    @pytest.mark.parametrize(["traj_file", "struct_file", "truth"],
                             [("tests/test_data/trajectory/ala_traj.trr",
                               "tests/test_data/trajectory/ala.tpr",
                               {"__len__": 18,
                                }
                               ),
                              ("tests/test_data/trajectory/ala_traj.xtc",
                               "tests/test_data/trajectory/ala.tpr",
                               {"__len__": 18,
                                }
                               ),
                              ("tests/test_data/trajectory/ala_traj.trr",
                               "tests/test_data/trajectory/ala.gro",
                               {"__len__": 18,
                                }
                               ),
                              ("tests/test_data/trajectory/ala_traj.xtc",
                               "tests/test_data/trajectory/ala.gro",
                               {"__len__": 18,
                                }
                               ),
                              ]
                             )
    def test_unary_magic_methods(self, traj_file, struct_file, truth):
        traj = Trajectory(trajectory_file=traj_file,
                          structure_file=struct_file,
                          )
        for mm_name, truth_value in truth.items():
            mm = getattr(traj, mm_name)
            assert mm() == truth_value

    @pytest.mark.parametrize(["traj_files", "struct_files", "truth"],
                             [# two times the same traj with the same struct file
                              (("tests/test_data/trajectory/ala_traj.trr",
                                "tests/test_data/trajectory/ala_traj.trr",
                                ),
                               ("tests/test_data/trajectory/ala.tpr",
                                "tests/test_data/trajectory/ala.tpr",
                                ),
                               {"__eq__": True,
                                "__ne__": False,
                                }
                               ),
                              # two times the same traj with the different struct files
                              (("tests/test_data/trajectory/ala_traj.trr",
                                "tests/test_data/trajectory/ala_traj.trr",
                                ),
                               ("tests/test_data/trajectory/ala.tpr",
                                "tests/test_data/trajectory/ala.gro",
                                ),
                               {"__eq__": True,
                                "__ne__": False,
                                }
                               ),
                              # different trajs with the same struct file ;)
                              (("tests/test_data/trajectory/ala_traj.trr",
                                "tests/test_data/trajectory/ala_traj.xtc",
                                ),
                               ("tests/test_data/trajectory/ala.tpr",
                                "tests/test_data/trajectory/ala.tpr",
                                ),
                               {"__eq__": False,
                                "__ne__": True,
                                }
                               ),
                              # different trajs with different struct files
                              (("tests/test_data/trajectory/ala_traj.trr",
                                "tests/test_data/trajectory/ala_traj.xtc",
                                ),
                               ("tests/test_data/trajectory/ala.tpr",
                                "tests/test_data/trajectory/ala.gro",
                                ),
                               {"__eq__": False,
                                "__ne__": True,
                                }
                               ),
                              ]
                             )
    def test_binary_magic_methods(self, traj_files, struct_files, truth):
        traj1 = Trajectory(trajectory_file=traj_files[0],
                           structure_file=struct_files[0],
                           )
        traj2 = Trajectory(trajectory_file=traj_files[1],
                           structure_file=struct_files[1],
                           )
        for mm_name, truth_value in truth.items():
            mm = getattr(traj1, mm_name)
            assert mm(traj2) == truth_value

    @pytest.mark.skip("TODO")
    def test_npz_cache(self):
        pass

    @pytest.mark.parametrize(["traj_file", "struct_file"],
                             [("tests/test_data/trajectory/ala_traj.trr",
                               "tests/test_data/trajectory/ala.tpr",
                               ),
                              ]
                             )
    def test_pickle(self, tmp_path, traj_file, struct_file):
        traj = Trajectory(trajectory_file=traj_file,
                          structure_file=struct_file,
                          )
        fname = tmp_path / "pickle_test.pckl"
        with open(file=fname, mode="wb") as pfile:
            pickle.dump(traj, pfile)
        # now open the file and loadit again
        with open(file=fname, mode="rb") as pfile:
            loaded_traj = pickle.load(pfile)
        # now compare the two
        # equality should at least be True
        assert traj == loaded_traj
        # TODO: check that the cached CV values are the same?
