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
import os
import pickle
import numpy as np


from asyncmd import Trajectory
from asyncmd.trajectory.trajectory import (TrajectoryFunctionValueCacheNPZ,
                                           TrajectoryFunctionValueCacheH5PY,
                                           )


class Test_Trajectory:
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
        # put the comparisson in a function to call it twice
        # this way we check both code-paths: value is None and needs to be read
        # from trajectory and value is already cached

        def compare_all():
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
        compare_all()  # this one reads the values and populates traj
        compare_all()  # this one should only take cached values

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
        # same logic as for properties, put in func to call twice to check both
        # code paths (retrieve value from underlying traj and get cached value)

        def compare_all():
            for mm_name, truth_value in truth.items():
                mm = getattr(traj, mm_name)
                assert mm() == truth_value
        compare_all()
        compare_all()

    @pytest.mark.parametrize(["traj_files", "struct_files", "truth"],
                             [
                              # two times the same traj with the same struct
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
                              # two times the same traj with different structs
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
        # now open the file and load it again
        with open(file=fname, mode="rb") as pfile:
            loaded_traj = pickle.load(pfile)
        # now compare the two
        # equality should at least be True
        assert traj == loaded_traj
        # TODO: check that the cached CV values are the same?
        # TODO: check that the cache logic works (i.e. that we visit all
        #        branches in __getstate__ and __setstate__)?!


class Test_TrajectoryFunctionValueCacheNPZ:
    def setup(self):
        # define functions for data generation and bind them to the test class
        self.ran_gen = np.random.default_rng()
        ii64 = np.iinfo(np.int64)

        def make_trajectory_hash():
            return self.ran_gen.integers(low=ii64.min,
                                         high=ii64.max,
                                         endpoint=True,
                                         )

        self.make_trajectory_hash = make_trajectory_hash

        def make_func_id():
            return "".join(self.ran_gen.choice([f"{i}" for i in range(10)],
                                               size=20,
                                               replace=True,
                                               )
                           )

        self.make_func_id = make_func_id

        def make_func_values(traj_len, cv_dim):
            return self.ran_gen.random(size=(traj_len, cv_dim))

        self.make_func_values = make_func_values

    def test__ensure_consistent_npz(self, tmp_path):
        # we also check in here that we get back what we saved
        # first generate name and hash for the traj + some mock CV data
        fname_traj = tmp_path / "traj_name.traj"
        hash_traj = self.make_trajectory_hash()
        n_cached_cvs = 4
        traj_len = 223
        cv_dims = [1, 223, 10, 321]
        test_data_func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        test_data_func_values = [self.make_func_values(traj_len, cv_dim)
                                 for cv_dim in cv_dims]
        # now create a fresh cache and append
        npz_cache = TrajectoryFunctionValueCacheNPZ(fname_traj=fname_traj,
                                                    hash_traj=hash_traj,
                                                    )
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            npz_cache.append(func_id=func_id, vals=func_values)
        # now create a second npz cache to test that it will load the data
        npz_cache2 = TrajectoryFunctionValueCacheNPZ(fname_traj=fname_traj,
                                                     hash_traj=hash_traj,
                                                     )
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            # and check that the loaded and saved data are equal
            assert np.all(np.equal(npz_cache2[func_id], func_values))
        # now check that the npz file will be removed if the traj hashes dont
        # match
        cache_file_name = npz_cache._get_cache_filename(fname_traj=fname_traj)
        hash_traj_mm = self.make_trajectory_hash()
        # creating a cache with a matching fname_traj but mismatching hash
        # should remove the npz file
        npz_cache_mm = TrajectoryFunctionValueCacheNPZ(fname_traj=fname_traj,
                                                       hash_traj=hash_traj_mm,
                                                       )
        assert not os.path.exists(cache_file_name)
        # recreate the npz file by appending the values again
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            npz_cache_mm.append(func_id=func_id, vals=func_values)
        # and now test for removal of npz if mismatching npz file format is
        # detected (triggered by changing the key for the hash_traj in npz)
        TrajectoryFunctionValueCacheNPZ._hash_traj_npz_key = "TEST123"
        npz_cache_mm2 = TrajectoryFunctionValueCacheNPZ(fname_traj=fname_traj,
                                                        hash_traj=hash_traj_mm,
                                                        )
        assert not os.path.exists(cache_file_name)

    def test___getitem__errs(self, tmp_path):
        fname_traj = tmp_path / "traj_name.traj"
        hash_traj = self.make_trajectory_hash()
        n_cached_cvs = 2
        traj_len = 23
        cv_dims = [1, 223]
        test_data_func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        test_data_func_values = [self.make_func_values(traj_len, cv_dim)
                                 for cv_dim in cv_dims]
        # now create a fresh cache and append
        npz_cache = TrajectoryFunctionValueCacheNPZ(fname_traj=fname_traj,
                                                    hash_traj=hash_traj,
                                                    )
        # make sure we get a KeyError trying to access non-existing values
        for func_id in test_data_func_ids:
            with pytest.raises(KeyError):
                _ = npz_cache[func_id]
        # now add the values and test that we still get errs when trying to
        # access non-existant stuff
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            npz_cache.append(func_id=func_id, vals=func_values)
        for _ in range(3):
            with pytest.raises(KeyError):
                _ = npz_cache[self.make_func_id()]
        # test for TypeError when using something that is not a string as key
        for func_id in [object(), 1, True, self.ran_gen]:
            with pytest.raises(TypeError):
                _ = npz_cache[func_id]

    def test_append_iter_len(self, tmp_path):
        fname_traj = tmp_path / "traj_name.traj"
        hash_traj = self.make_trajectory_hash()
        n_cached_cvs = 5
        n_initial_cvs = 2
        traj_len = 123
        cv_dims = [1] + [self.ran_gen.integers(300)
                         for _ in range(n_cached_cvs - 1)
                         ]
        test_data_func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        test_data_func_values = [self.make_func_values(traj_len, cv_dim)
                                 for cv_dim in cv_dims]
        # now create a fresh cache and append
        npz_cache = TrajectoryFunctionValueCacheNPZ(fname_traj=fname_traj,
                                                    hash_traj=hash_traj,
                                                    )
        for func_id, func_values in zip(test_data_func_ids[:n_initial_cvs],
                                        test_data_func_values[:n_initial_cvs]):
            npz_cache.append(func_id=func_id, vals=func_values)
        # now create a new cache, make sure it has the right "len" and that the
        # values are as expected (use iter to check!)
        npz_cache2 = TrajectoryFunctionValueCacheNPZ(fname_traj=fname_traj,
                                                     hash_traj=hash_traj,
                                                     )
        assert len(npz_cache2) == n_initial_cvs
        for func_id in npz_cache2:
            idx_in_test_data = test_data_func_ids.index(func_id)
            assert np.all(np.equal(npz_cache2[func_id],
                                   test_data_func_values[idx_in_test_data]
                                   )
                          )
        # now append the initial part again and check for the err
        for func_id, func_values in zip(test_data_func_ids[:n_initial_cvs],
                                        test_data_func_values[:n_initial_cvs]):
            with pytest.raises(ValueError):
                npz_cache2.append(func_id=func_id, vals=func_values)
        # now append the rest and check that everything is correct
        for func_id, func_values in zip(test_data_func_ids[n_initial_cvs:],
                                        test_data_func_values[n_initial_cvs:]):
            npz_cache2.append(func_id=func_id, vals=func_values)
        npz_cache3 = TrajectoryFunctionValueCacheNPZ(fname_traj=fname_traj,
                                                     hash_traj=hash_traj,
                                                     )
        assert len(npz_cache3) == n_cached_cvs
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            assert np.all(np.equal(npz_cache3[func_id], func_values))
