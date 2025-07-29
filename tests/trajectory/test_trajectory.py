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
import logging
import os
import pickle
import numpy as np

from unittest.mock import Mock, PropertyMock

import asyncmd
from asyncmd import Trajectory
from asyncmd.config import _GLOBALS  # noqa: F401
from asyncmd.trajectory.trajectory_cache import (TrajectoryFunctionValueCacheInMemory,
                                                 TrajectoryFunctionValueCacheInNPZ,
                                                 TrajectoryFunctionValueCacheInH5PY,
                                                 OneH5PYGroupTrajectoryFunctionValueCache,
                                                 ValuesAlreadyStoredError,
                                                 CanNotChangeReadOnlyH5PYError,
                                                 )
from asyncmd.trajectory.functionwrapper import TrajectoryFunctionWrapper


class TBase:
    # base class all trajectory.py tests
    # contains general purpose data generation/setup functions
    def setup_method(self):
        # remember current workdir (for tests that change the dir to change back)
        self.workdir = os.path.abspath(os.getcwd())
        # define functions for data generation and bind them to the test class
        self.ran_gen = np.random.default_rng()
        ii64 = np.iinfo(np.int64)

        def make_trajectory_hash() -> int:
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

    def teardown_method(self):
        # make sure we are back at initial workdir for test in which we change
        # the workdir
        os.chdir(self.workdir)


class Test_Trajectory(TBase):
    def setup_method(self):
        super().setup_method()
        asyncmd.trajectory._forget_all_trajectories()

    @pytest.mark.parametrize(["trajectory_files", "structure_file"],
                             [("tests/test_data/trajectory/ala_traj.trr",  # existing traj
                               "tests/test_data/trajectory/NON_EXISTING_STRUCTURE",  # ...
                               ),
                              (["tests/test_data/trajectory/ala_traj.trr"],  # existing traj (list)
                               "tests/test_data/trajectory/NON_EXISTING_STRUCTURE",  # ...
                               ),
                              # and the other way around: non-existing traj, but struct present
                              ("tests/test_data/trajectory/NON_EXISTING_TRAJECTORY",
                               "tests/test_data/trajectory/ala.tpr")
                              ]
                             )
    def test_init_raises_non_existing_files(self, trajectory_files, structure_file):
        with pytest.raises(FileNotFoundError):
            _ = Trajectory(trajectory_files=trajectory_files,
                           structure_file=structure_file)

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
        traj = Trajectory(trajectory_files=traj_file,
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
        traj = Trajectory(trajectory_files=traj_file,
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
                              # same for lists of trajs
                              # two times the same traj with the same struct
                              ((["tests/test_data/trajectory/ala_traj.trr",
                                 "tests/test_data/trajectory/ala_traj.trr"],
                                ["tests/test_data/trajectory/ala_traj.trr",
                                 "tests/test_data/trajectory/ala_traj.trr"],
                                ),
                               ("tests/test_data/trajectory/ala.tpr",
                                "tests/test_data/trajectory/ala.tpr",
                                ),
                               {"__eq__": True,
                                "__ne__": False,
                                }
                               ),
                              # two times the same traj with different structs
                              ((["tests/test_data/trajectory/ala_traj.trr",
                                 "tests/test_data/trajectory/ala_traj.trr"],
                                ["tests/test_data/trajectory/ala_traj.trr",
                                 "tests/test_data/trajectory/ala_traj.trr"],
                                ),
                               ("tests/test_data/trajectory/ala.tpr",
                                "tests/test_data/trajectory/ala.gro",
                                ),
                               {"__eq__": True,
                                "__ne__": False,
                                }
                               ),
                              # different trajs with the same struct file ;)
                              ((["tests/test_data/trajectory/ala_traj.trr",
                                 "tests/test_data/trajectory/ala_traj.trr"],
                                ["tests/test_data/trajectory/ala_traj.xtc",
                                 "tests/test_data/trajectory/ala_traj.xtc"],
                                ),
                               ("tests/test_data/trajectory/ala.tpr",
                                "tests/test_data/trajectory/ala.tpr",
                                ),
                               {"__eq__": False,
                                "__ne__": True,
                                }
                               ),
                              # different trajs with different struct files
                              ((["tests/test_data/trajectory/ala_traj.trr",
                                 "tests/test_data/trajectory/ala_traj.trr"],
                                ["tests/test_data/trajectory/ala_traj.xtc",
                                 "tests/test_data/trajectory/ala_traj.xtc"],
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
        traj1 = Trajectory(trajectory_files=traj_files[0],
                           structure_file=struct_files[0],
                           )
        traj2 = Trajectory(trajectory_files=traj_files[1],
                           structure_file=struct_files[1],
                           )
        for mm_name, truth_value in truth.items():
            mm = getattr(traj1, mm_name)
            assert mm(traj2) == truth_value

    def test_eq_neq(self):
        # we use two equal trajs and then modfify one of them selectively
        # i.e. at single points (possibly with mocks) to make them uneqal
        def make_traj():
            # need to forget all trajectories such that we actually get
            # a new object for the same trajectory(_files)
            asyncmd.trajectory._forget_all_trajectories()
            return Trajectory(
                    trajectory_files="tests/test_data/trajectory/ala_traj.trr",
                    structure_file="tests/test_data/trajectory/ala.tpr",
                    )

        def assert_neq(traj1, traj2):
            # check both eq and neq at once
            assert not traj1 == traj2
            assert traj1 != traj2
        traj1 = make_traj()
        traj2 = make_traj()
        assert traj1 == traj2  # make sure they are equal to begin with
        assert not traj1 != traj2  # and check that neq also works
        # get another traj2 to test [not] eq (for two trajs which are different)
        traj2 = Trajectory(
                    trajectory_files="tests/test_data/trajectory/ala_conf_in_no_state.trr",
                    structure_file="tests/test_data/trajectory/ala.tpr",
                    )
        assert_neq(traj1, traj2)
        # test for non trajectory objects
        assert_neq(traj1, object())

    @pytest.mark.parametrize("cache_type",
                             ["npz", "h5py", "memory"])
    @pytest.mark.parametrize("initial_cache_type",
                             ["npz", "h5py", "memory"])
    def test__setup_cache(self, tmp_path, cache_type, initial_cache_type):
        # we create a trajectory with given initial_cache_type
        # (after setting config-cache type), append some values, then
        # (re)set the cache_type and check that everything is there and in the
        # correct cache
        global _GLOBALS
        if (initial_cache_type == "h5py" or cache_type == "h5py"):
            # setup the h5py file if we use a h5py cache
            h5py = pytest.importorskip("h5py", minversion=None,
                                       reason="Requires 'h5py' to run.",
                                       )
            h5file = h5py.File(tmp_path / "h5py_file.h5", mode="w")
            if initial_cache_type == "h5py":
                asyncmd.config.register_h5py_cache(h5py_group=h5file)
        # set the initial cache type (if it is `h5py` this is a no-op, since then
        # register_h5py_cache already sets the cache type to h5py)
        asyncmd.config.set_trajectory_cache_type(cache_type=initial_cache_type)

        traj = Trajectory(
                    trajectory_files="tests/test_data/trajectory/ala_traj.trr",
                    structure_file="tests/test_data/trajectory/ala.tpr",
                          )
        traj.clear_all_cache_values()  # make sure the cache is empty
        # create some dummy CV data and attach it to traj
        n_cached_cvs = 4
        cv_dims = [self.ran_gen.integers(300) for _ in range(n_cached_cvs)]
        func_values = [self.make_func_values(traj_len=200, cv_dim=cv_dim)
                       for cv_dim in cv_dims
                       ]
        func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        # use a mock wrapped CV to attach the values
        wrapped_func = Mock(TrajectoryFunctionWrapper)
        type(wrapped_func).id = PropertyMock(side_effect=func_ids)
        for vals in func_values:
            traj._register_cached_values(values=vals, func_wrapper=wrapped_func)
        # now reset cache_type
        if cache_type == "h5py":
            asyncmd.config.register_h5py_cache(h5py_group=h5file)
        else:
            # for h5py this should be a no-op, but we take that code path above ;)
            asyncmd.config.set_trajectory_cache_type(cache_type=cache_type)
        # and check that everything went well
        wrapped_func = Mock(TrajectoryFunctionWrapper)
        type(wrapped_func).id = PropertyMock(side_effect=func_ids)
        for func_vals in func_values:
            retrieved_func_vals = traj._retrieve_cached_values(
                                                    func_wrapper=wrapped_func,
                                                               )
            assert np.all(np.equal(retrieved_func_vals, func_vals))
        # and check that the correct cache (class) is used/bound to traj
        cache_type_to_class = {"memory": TrajectoryFunctionValueCacheInMemory,
                               "npz": TrajectoryFunctionValueCacheInNPZ,
                               "h5py": TrajectoryFunctionValueCacheInH5PY,
                               }
        assert isinstance(traj._cache, cache_type_to_class[cache_type])
        # cleanup
        # remove the npz cache file (if it can be there)!
        fname_npz_cache = TrajectoryFunctionValueCacheInNPZ.get_cache_filename(
                                        traj_files=traj.trajectory_files,
                                                                              )
        if "npz" in (cache_type, initial_cache_type):  # npz cache explicitly used
            os.unlink(fname_npz_cache)
        else:
            # there should be no file created if npz cache is not involved!
            assert not os.path.isfile(fname_npz_cache)

    @pytest.mark.parametrize("cache_type",
                             ["npz", "h5py", "memory"])
    @pytest.mark.parametrize("change_wdir_between_pickle_unpickle",
                             [True, False])
    def test_pickle_and_wrapped_func_application(
                                        self,
                                        tmp_path,
                                        cache_type,
                                        change_wdir_between_pickle_unpickle,
                                                       ):
        global _GLOBALS
        if cache_type == "h5py":
            h5py = pytest.importorskip("h5py", minversion=None,
                                       reason="Requires 'h5py' to run.",
                                       )
            h5file = h5py.File(tmp_path / "h5py_file.h5", mode="w")
            asyncmd.config.register_h5py_cache(h5py_group=h5file)
        else:
            asyncmd.config.set_trajectory_cache_type(cache_type=cache_type)
        traj = Trajectory(
                    trajectory_files="tests/test_data/trajectory/ala_traj.trr",
                    structure_file="tests/test_data/trajectory/ala.tpr",
                          )
        traj.clear_all_cache_values()  # make sure the cache is empty
        # create some dummy CV data and attach it to traj
        n_cached_cvs = 4
        cv_dims = [self.ran_gen.integers(300) for _ in range(n_cached_cvs)]
        func_values = [self.make_func_values(traj_len=200, cv_dim=cv_dim)
                       for cv_dim in cv_dims
                       ]
        func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        # use a mock wrapped CV to attach the values
        wrapped_func = Mock(TrajectoryFunctionWrapper)
        type(wrapped_func).id = PropertyMock(side_effect=func_ids)
        for vals in func_values:
            traj._register_cached_values(values=vals, func_wrapper=wrapped_func)
        fname = tmp_path / "pickle_test.pckl"
        with open(file=fname, mode="wb") as pfile:
            pickle.dump(traj, pfile)

        if change_wdir_between_pickle_unpickle:
            # NOTE: we change back to the old workdir in the teardown func
            os.chdir(tmp_path)

        # now open the file and load it again
        with open(file=fname, mode="rb") as pfile:
            loaded_traj: Trajectory = pickle.load(pfile)
        # now compare the two
        # equality should at least be True
        assert traj == loaded_traj
        # check that the CV values are all in the loaded traj too
        wrapped_func = Mock(TrajectoryFunctionWrapper)
        type(wrapped_func).id = PropertyMock(side_effect=func_ids)
        for func_vals in func_values:
            loaded_func_vals = loaded_traj._retrieve_cached_values(
                                                    func_wrapper=wrapped_func,
                                                                   )
            assert np.all(np.equal(loaded_func_vals, func_vals))

        # cleanup
        # remove the npz cache file!
        fname_npz_cache = TrajectoryFunctionValueCacheInNPZ.get_cache_filename(
                                        traj_files=traj.trajectory_files,
                                                                              )
        if (
            cache_type == "npz"  # npz cache explicitly used
            or cache_type == "memory"  # memory will write to npz when pickled
        ):
            os.unlink(fname_npz_cache)
        else:
            # there should be no file created if npz cache is not involved!
            assert not os.path.isfile(fname_npz_cache)


class Test_TrajectoryFunctionValueCache(TBase):
    def setup_method(self):
        super().setup_method()

    @pytest.mark.parametrize("cache_class",
                             [TrajectoryFunctionValueCacheInNPZ,
                              TrajectoryFunctionValueCacheInH5PY,
                              TrajectoryFunctionValueCacheInMemory,
                              ]
                             )
    def test_append_iter_len___getitem___errs(self, tmp_path, cache_class):
        if cache_class is TrajectoryFunctionValueCacheInH5PY:
            h5py = pytest.importorskip("h5py", minversion=None,
                                       reason="Requires 'h5py' to run.",
                                       )
            fname_traj_cache = tmp_path / "traj_cache.h5"
            h5file = h5py.File(fname_traj_cache, mode="w")
            asyncmd.config.register_h5py_cache(h5py_group=h5file)

        traj_hash = self.make_trajectory_hash()
        traj_files = [tmp_path / "traj_name.traj"]
        n_cached_cvs = 2
        traj_len = 23
        cv_dims = [1, 223]
        test_data_func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        test_data_func_values = [self.make_func_values(traj_len, cv_dim)
                                 for cv_dim in cv_dims]
        # now create a fresh cache and append
        cache = cache_class(traj_hash=traj_hash, traj_files=traj_files)
        # make sure we get a KeyError trying to access non-existing values
        for func_id in test_data_func_ids:
            with pytest.raises(KeyError):
                _ = cache[func_id]
        # now add the values and test that we still get errs when trying to
        # access non-existant stuff
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            cache.append(func_id=func_id, values=func_values)
        for _ in range(3):
            with pytest.raises(KeyError):
                _ = cache[self.make_func_id()]
        # check that appending again raises an error
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            with pytest.raises(ValuesAlreadyStoredError):
                cache.append(func_id=func_id, values=func_values)
        # and finally test that everything is there as expected
        for func_id in cache:
            idx_in_test_data = test_data_func_ids.index(func_id)
            assert np.all(np.equal(cache[func_id],
                                   test_data_func_values[idx_in_test_data]
                                   )
                          )
        assert len(cache) == n_cached_cvs
        ran_idx = self.ran_gen.integers(n_cached_cvs)
        assert np.all(np.equal(cache[test_data_func_ids[ran_idx]],
                               test_data_func_values[ran_idx]
                               )
                      )
        # now check that clear works as expected
        cache.clear_all_values()
        assert len(cache) == 0

    # Note these test dont work for the memory cache as it is not stateful,
    # i.e. recreating it will empty it (as there is no file to back it)
    @pytest.mark.parametrize("cache_class",
                             [TrajectoryFunctionValueCacheInNPZ,
                              TrajectoryFunctionValueCacheInH5PY,
                              ]
                             )
    def test_stateful_append_iter_len(self, tmp_path, cache_class):
        if cache_class is TrajectoryFunctionValueCacheInH5PY:
            h5py = pytest.importorskip("h5py", minversion=None,
                                       reason="Requires 'h5py' to run.",
                                       )
            fname_traj_cache = tmp_path / "traj_cache.h5"
            h5file = h5py.File(fname_traj_cache, mode="w")
            asyncmd.config.register_h5py_cache(h5py_group=h5file)

        traj_hash = self.make_trajectory_hash()
        traj_files = [tmp_path / "traj_name.traj"]
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
        cache = cache_class(traj_hash=traj_hash, traj_files=traj_files)
        for func_id, func_values in zip(test_data_func_ids[:n_initial_cvs],
                                        test_data_func_values[:n_initial_cvs]):
            cache.append(func_id=func_id, values=func_values)
        # now create a new cache, make sure it has the right "len" and that the
        # values are as expected (use iter to check!)
        cache2 = cache_class(traj_hash=traj_hash, traj_files=traj_files)
        assert len(cache2) == n_initial_cvs
        for func_id in cache2:
            idx_in_test_data = test_data_func_ids.index(func_id)
            assert np.all(np.equal(cache2[func_id],
                                   test_data_func_values[idx_in_test_data]
                                   )
                          )
        # now append the initial part again and check for the err
        for func_id, func_values in zip(test_data_func_ids[:n_initial_cvs],
                                        test_data_func_values[:n_initial_cvs]):
            with pytest.raises(ValuesAlreadyStoredError):
                cache2.append(func_id=func_id, values=func_values)
        # now append the rest and check that everything is correct
        for func_id, func_values in zip(test_data_func_ids[n_initial_cvs:],
                                        test_data_func_values[n_initial_cvs:]):
            cache2.append(func_id=func_id, values=func_values)
        cache3 = cache_class(traj_hash=traj_hash, traj_files=traj_files)
        assert len(cache3) == n_cached_cvs
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            assert np.all(np.equal(cache3[func_id], func_values))
        # check that stateful caches initialize empty if we clear them before
        cache3.clear_all_values()
        cache4 = cache_class(traj_hash=traj_hash, traj_files=traj_files)
        assert len(cache4) == 0

    # NOTE: this test is only relevant for npz cache as the h5py cache does not
    #       have/need a corresponding function since it caches using the
    #       hash of the traj as root group name
    def test__ensure_consistent_npz(self, tmp_path):
        # we also check in here that we get back what we saved
        # first generate name and hash for the traj + some mock CV data
        traj_files = [tmp_path / "traj_name.traj"]
        traj_hash = self.make_trajectory_hash()
        n_cached_cvs = 4
        traj_len = 223
        cv_dims = [1, 223, 10, 321]
        test_data_func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        test_data_func_values = [self.make_func_values(traj_len, cv_dim)
                                 for cv_dim in cv_dims]
        # now create a fresh cache and append
        npz_cache = TrajectoryFunctionValueCacheInNPZ(traj_hash=traj_hash,
                                                      traj_files=traj_files)
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            npz_cache.append(func_id=func_id, values=func_values)
        # now create a second npz cache to test that it will load the data
        npz_cache2 = TrajectoryFunctionValueCacheInNPZ(traj_hash=traj_hash,
                                                       traj_files=traj_files)
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            # and check that the loaded and saved data are equal
            assert np.all(np.equal(npz_cache2[func_id], func_values))
        # now check that the npz file will be removed if the traj hashes dont
        # match, currently we have the first 5 digits of the hash in the cache
        # filename, so artificially modify only the last digit(s)
        hash_traj_mm = traj_hash - 1
        if str(traj_hash)[:5] != str(hash_traj_mm)[:5]:
            hash_traj_mm = traj_hash + 1
        cache_file_name = npz_cache.get_cache_filename(traj_files=traj_files)
        # creating a cache with a matching fname_trajs but mismatching hash
        # should remove the npz file
        npz_cache_mm = TrajectoryFunctionValueCacheInNPZ(traj_hash=hash_traj_mm,
                                                         traj_files=traj_files)
        assert not os.path.exists(cache_file_name)
        # recreate the npz file by appending the values again
        for func_id, func_values in zip(test_data_func_ids,
                                        test_data_func_values):
            npz_cache_mm.append(func_id=func_id, values=func_values)
        # and now test for removal of npz if mismatching npz file format is
        # detected (triggered by changing the key for the hash_traj in npz)
        TrajectoryFunctionValueCacheInNPZ._TRAJ_HASH_NPZ_KEY = "TEST123"
        _ = TrajectoryFunctionValueCacheInNPZ(traj_hash=traj_hash,
                                              traj_files=traj_files)
        assert not os.path.exists(cache_file_name)

    # NOTE: this test is only relevant for the h5py cache as it is the only one
    #       that uses the h5py_cache config value
    def test_h5py_cache_not_set(self, tmp_path):
        # make sure the cache variable is unset
        try:
            _ = _GLOBALS["H5PY_CACHE"]
        except KeyError:
            pass
        else:
            del _GLOBALS["H5PY_CACHE"]

        traj_hash = self.make_trajectory_hash()
        traj_files = [tmp_path / "traj_name.traj"]
        with pytest.raises(RuntimeError):
            TrajectoryFunctionValueCacheInH5PY(traj_hash=traj_hash,
                                               traj_files=traj_files)

    # Note: this test is only for h5py, since that is the only cache source that
    #       allows for multiple cache files used simultaneously
    @pytest.mark.parametrize("copy_h5py", [True, False])
    def test_h5py_multiple_cache_sources(self, tmp_path, copy_h5py, caplog):
        h5py = pytest.importorskip("h5py", minversion=None,
                                   reason="Requires 'h5py' to run.",
                                   )
        # make our first cache file (writeable)
        fname_traj_cache = tmp_path / "traj_cache.h5"
        h5file = h5py.File(fname_traj_cache, mode="w")
        asyncmd.config.register_h5py_cache(h5py_group=h5file)

        # make some values to append
        traj_hash = self.make_trajectory_hash()
        traj_files = [tmp_path / "traj_name.traj"]
        n_cached_cvs = 5
        n_initial_cvs = 2  # number of CVs to add initially
        traj_len = 123
        cv_dims = [1] + [self.ran_gen.integers(300)
                         for _ in range(n_cached_cvs - 1)
                         ]
        test_data_func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        test_data_func_values = [self.make_func_values(traj_len, cv_dim)
                                 for cv_dim in cv_dims]
        # append it
        cache = TrajectoryFunctionValueCacheInH5PY(traj_hash=traj_hash,
                                                   traj_files=traj_files)
        for func_id, func_values in zip(test_data_func_ids[:n_initial_cvs],
                                        test_data_func_values[:n_initial_cvs]):
            cache.append(func_id=func_id, values=func_values)
        assert len(cache) == n_initial_cvs  # make sure length is as expected
        # now reset the cache to a new one
        # call deregister in config to remove it from the _GLOBALS dict
        asyncmd.config.deregister_h5py_cache(h5py_group=h5file)
        # but also on the cache (because it will not be called on this cache,
        # since it is not attached to a trajectory)
        cache.deregister_h5py_cache(h5py_cache=h5file)
        # make sure the cache has no values cached after deregistering
        assert len(cache) == 0
        # close the initial file, reopen it read-only
        h5file.close()
        h5file_r_with_data = h5py.File(fname_traj_cache, mode="r")
        # and create a new (writeable) file
        fname_traj_cache2 = tmp_path / "traj_cache2.h5"
        h5file_w_empty = h5py.File(fname_traj_cache2, mode="w")
        # and create an new (unwriteable) empty file (but with a prefix group)
        fname_traj_cache3 = tmp_path / "traj_cache3.h5"
        # create an empty file but with the prefix group
        h5file_r_empty = h5py.File(fname_traj_cache3, mode="w")
        _ = h5file_r_empty.require_group("asyncmd/TrajectoryFunctionValueCache")
        h5file_r_empty.close()
        # and open it read-only
        h5file_r_empty = h5py.File(fname_traj_cache3, mode="r")
        # register the file we just opened in read-only (to which we saved the values)
        asyncmd.config.register_h5py_cache(h5py_group=h5file_r_with_data)
        # and the empty read-only file
        asyncmd.config.register_h5py_cache(h5py_group=h5file_r_empty)
        # create a new cache (which should now be read-only)
        cache = TrajectoryFunctionValueCacheInH5PY(traj_hash=traj_hash,
                                                   traj_files=traj_files)
        # make sure it is all there as expected
        for func_id, func_values in zip(test_data_func_ids[:n_initial_cvs],
                                        test_data_func_values[:n_initial_cvs]):
            assert np.all(np.equal(cache[func_id], func_values))
        # The next line just to test that deregistering an unknown
        # (not registered) cache also works
        asyncmd.config.deregister_h5py_cache(h5py_group=h5file_w_empty)
        cache.deregister_h5py_cache(h5py_cache=h5file_w_empty)
        # now deregister both registered read only caches and add them again
        asyncmd.config.deregister_h5py_cache(h5py_group=h5file_r_with_data)
        cache.deregister_h5py_cache(h5py_cache=h5file_r_with_data)
        asyncmd.config.deregister_h5py_cache(h5py_group=h5file_r_empty)
        cache.deregister_h5py_cache(h5py_cache=h5file_r_empty)
        # now register them again
        asyncmd.config.register_h5py_cache(h5py_group=h5file_r_with_data)
        asyncmd.config.register_h5py_cache(h5py_group=h5file_r_empty)
        # and instantiate a new cache so it uses the now set h5py caches
        cache = TrajectoryFunctionValueCacheInH5PY(traj_hash=traj_hash,
                                                   traj_files=traj_files)
        # make sure we cant clear it
        with pytest.raises(CanNotChangeReadOnlyH5PYError):
            cache.clear_all_values()
        # and make sure we can not append, currently it does not raise
        # but just log an error (and naturally does not append)
        with caplog.at_level(logging.ERROR):
            cache.append(func_id=test_data_func_ids[n_initial_cvs + 1],
                         values=test_data_func_values[n_initial_cvs + 1])
        assert "Can not append" in caplog.text
        assert len(cache) == n_initial_cvs  # make sure length is as expected
        # now register the writeable h5py_cache
        asyncmd.config.register_h5py_cache(h5py_group=h5file_w_empty,
                                           copy_h5py=copy_h5py,
                                           )
        # and create a new cache (which should now use both the writeable and
        # read-only caches)
        cache = TrajectoryFunctionValueCacheInH5PY(traj_hash=traj_hash,
                                                   traj_files=traj_files)
        # append the rest
        for func_id, func_values in zip(test_data_func_ids[n_initial_cvs:],
                                        test_data_func_values[n_initial_cvs:]):
            cache.append(func_id=func_id, values=func_values)
        assert len(cache) == n_cached_cvs  # make sure length is as expected
        # now deregister and close the initial cache file
        asyncmd.config.deregister_h5py_cache(h5py_group=h5file_r_with_data)
        # but also on the cache (because it will not be called on this cache,
        # since it is not attached to a trajectory)
        cache.deregister_h5py_cache(h5py_cache=h5file_r_with_data)
        # make sure the cache has the correct length
        if not copy_h5py:
            # we did not copy the initially appended values, so we dont have them
            # accessible after deregistering the cache
            assert len(cache) == n_cached_cvs - n_initial_cvs
        else:
            # we should have them all
            assert len(cache) == n_cached_cvs
        h5file_r_with_data.close()
        # and reopen the initial file as writeable file
        h5file_a_reopend = h5py.File(fname_traj_cache, mode="r+")
        # register it with copying to see if adding/copying some values for a traj works
        # (this will copy the remaining CVs we just added into the initial cache,
        #  because copy_h5py=True, such that the initial cache file now in all cases
        #  contains all n_cached_cvs)
        asyncmd.config.register_h5py_cache(h5py_group=h5file_a_reopend, copy_h5py=True)
        # the line above should remove all caches except the one we just added
        # (because we use copy_h5py=True)
        # so close all files we dont need
        h5file_w_empty.close()
        h5file_r_empty.close()
        # now init a ne cache to make sure we use the most current cache setting
        cache = TrajectoryFunctionValueCacheInH5PY(traj_hash=traj_hash,
                                                   traj_files=traj_files)
        # and make sure everything is there, i.e. all n_cached_cvs values
        assert len(cache) == n_cached_cvs
        for func_id, func_values in zip(test_data_func_ids, test_data_func_values):
            assert np.all(np.equal(cache[func_id], func_values))
        # and finally to see that we get a KeyError for non-existing values
        # (and after exhausting all cache options)
        # for this reopen and re-add one of the caches (with data to make sure it
        # will be attached to the cache class) as read-only
        h5file_r_with_data = h5py.File(fname_traj_cache2, mode="r")
        asyncmd.config.register_h5py_cache(h5py_group=h5file_r_with_data, copy_h5py=False)
        cache = TrajectoryFunctionValueCacheInH5PY(traj_hash=traj_hash,
                                                   traj_files=traj_files)
        with pytest.raises(KeyError):
            _ = cache["NOT_SET_KEY"]

    def test_OneH5PYGroupTrajectoryFunctionValueCache_raises_on_read_only(self, tmp_path):
        h5py = pytest.importorskip("h5py", minversion=None,
                                   reason="Requires 'h5py' to run.",
                                   )
        # make some values to append
        traj_hash = self.make_trajectory_hash()
        traj_files = [tmp_path / "traj_name.traj"]
        n_cached_cvs = 2
        traj_len = 123
        cv_dims = [1] + [self.ran_gen.integers(300)
                         for _ in range(n_cached_cvs - 1)
                         ]
        test_data_func_ids = [self.make_func_id() for _ in range(n_cached_cvs)]
        test_data_func_values = [self.make_func_values(traj_len, cv_dim)
                                 for cv_dim in cv_dims]

        # make our first cache file (writeable)
        fname_traj_cache = tmp_path / "traj_cache.h5"
        # open and directly close it after making sure there is a group
        # for the trajectory cached values and funcIDs (but the groups are empty,
        # i.e. no values and funcIDs in there)
        h5file = h5py.File(fname_traj_cache, mode="w")
        root_grp = h5file.require_group(f"asyncmd/TrajectoryFunctionValueCache/{traj_hash}")
        _ = root_grp.require_group("FunctionIDs")
        _ = root_grp.require_group("FunctionValues")
        h5file.close()
        # reopen in read-only
        h5file = h5py.File(fname_traj_cache, mode="r")
        # now init the (sub)cache class
        cache = OneH5PYGroupTrajectoryFunctionValueCache(traj_hash=traj_hash,
                                                         traj_files=traj_files,
                                                         h5py_cache=h5file)
        # and make sure it know it is read-only
        assert cache._read_only
        # and raises
        with pytest.raises(CanNotChangeReadOnlyH5PYError):
            cache.append(func_id=test_data_func_ids[0], values=test_data_func_values[0])
        with pytest.raises(CanNotChangeReadOnlyH5PYError):
            cache.clear_all_values()
