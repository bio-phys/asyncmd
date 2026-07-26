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
Comprehensive test suite for asyncmd.config module.

Tests the configuration functions for:
- Resource management (processes, files)
- Cache type configuration
- H5PY cache registration/deregistration
- SLURM configuration

Note: H5PY-specific tests for cache operations are covered more thoroughly
in tests/trajectory/test_trajectory.py which tests the full integration.
"""

import pytest
import os
import asyncio
import resource
import sys
import numpy as np
from io import StringIO
from unittest.mock import patch


from asyncmd import config
from asyncmd import trajectory
from asyncmd._config import (
    _GLOBALS,
    _SEMAPHORES,
    _OPT_SEMAPHORES,
    _GLOBALS_KEYS,
    _SEMAPHORES_KEYS,
    _OPT_SEMAPHORES_KEYS,
)


@pytest.fixture
def reset_config():
    """Fixture to reset config to initial state after test."""
    # Import here to avoid circular imports
    import asyncio
    from asyncmd import config

    # Store initial state before test
    initial_globals = dict(config._GLOBALS)
    initial_semaphores = {key: sem._value for key, sem in config._SEMAPHORES.items()}
    initial_opt_semaphores = {
        key: (sem._value if sem is not None else None)
        for key, sem in config._OPT_SEMAPHORES.items()
    }

    yield

    # Restore _GLOBALS
    config._GLOBALS.clear()
    config._GLOBALS.update(initial_globals)

    # Restore _SEMAPHORES
    config._SEMAPHORES.clear()
    for key, value in initial_semaphores.items():
        config._SEMAPHORES[key] = asyncio.BoundedSemaphore(value)

    # Restore _OPT_SEMAPHORES
    config._OPT_SEMAPHORES.clear()
    for key, value in initial_opt_semaphores.items():
        if value is not None:
            config._OPT_SEMAPHORES[key] = asyncio.BoundedSemaphore(value)
        else:
            config._OPT_SEMAPHORES[key] = None

    # Clear trajectory registry
    trajectory._forget_all_trajectories()


class Test_set_max_process:
    def test_set_max_process_default(self, reset_config):
        """Test set_max_process with default (None) values."""
        config.set_max_process()
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS] is not None
        assert isinstance(
            _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS], asyncio.BoundedSemaphore
        )
        # Default should be cpu_count() / 4
        cpu_count = os.cpu_count()
        expected = max(1, int(cpu_count / 4))
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]._value == expected

    def test_set_max_process_with_integer(self, reset_config):
        """Test set_max_process with specific integer value."""
        config.set_max_process(num=5)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]._value == 5

    def test_set_max_process_with_max_num(self, reset_config):
        """Test set_max_process with max_num constraint."""
        cpu_count = os.cpu_count()
        expected = min(int(cpu_count / 4), 2)
        config.set_max_process(num=None, max_num=2)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]._value == expected

    def test_set_max_process_with_max_num_higher_than_cpu(self, reset_config):
        """Test max_num caps the value even when cpu_count is high."""
        cpu_count = os.cpu_count()
        config.set_max_process(num=None, max_num=1)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]._value == 1

    def test_set_max_process_without_cpu_count(self, reset_config):
        """Test fallback when os.cpu_count() returns None."""
        with patch("os.cpu_count", return_value=None):
            config.set_max_process()
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]._value == 2

    def test_set_max_process_changes_semaphore(self, reset_config):
        """Test that semaphore is actually updated."""
        config.set_max_process(num=3)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]._value == 3
        config.set_max_process(num=7)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]._value == 7

    def test_set_max_process_with_zero(self, reset_config):
        """Test set_max_process with zero."""
        with pytest.raises(ValueError, match=" must be at least 1!"):
            config.set_max_process(num=0)
        # same for max_num
        with pytest.raises(ValueError, match=" must be at least 1!"):
            config.set_max_process(num=1, max_num=0)

    def test_set_max_process_with_negative(self, reset_config):
        """Test set_max_process with negative value raises ValueError."""
        with pytest.raises(ValueError, match=" must be at least 1!"):
            config.set_max_process(num=-1)
        with pytest.raises(ValueError, match=" must be at least 1!"):
            config.set_max_process(num=1, max_num=-1)


class Test_set_max_files_open:
    def test_set_max_files_open_default(self, reset_config):
        """Test set_max_files_open with default (None) values."""
        config.set_max_files_open()
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN] is not None
        assert isinstance(
            _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN], asyncio.BoundedSemaphore
        )
        # Default should be system limit
        rlim_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        expected_semaval = int((rlim_soft - 30) / 3)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]._value == expected_semaval

    def test_set_max_files_open_with_custom_value(self, reset_config):
        """Test set_max_files_open with custom value."""
        config.set_max_files_open(num=100, margin=10)
        rlim_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        expected_semaval = int((min(100, rlim_soft) - 10) / 3)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]._value == expected_semaval

    def test_set_max_files_open_with_margin_parameter(self, reset_config):
        """Test set_max_files_open with custom margin."""
        config.set_max_files_open(num=200, margin=50)
        rlim_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        expected_semaval = int((min(200, rlim_soft) - 50) / 3)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]._value == expected_semaval

    def test_set_max_files_open_num_less_than_margin(self, reset_config):
        """Test set_max_files_open raises ValueError when num <= margin."""
        with pytest.raises(ValueError, match="num must be larger than margin"):
            config.set_max_files_open(num=10, margin=20)

    def test_set_max_files_open_num_equals_margin(self, reset_config):
        """Test set_max_files_open raises ValueError when num == margin."""
        with pytest.raises(ValueError, match="num must be larger than margin"):
            config.set_max_files_open(num=20, margin=20)

    def test_set_max_files_open_num_greater_than_system_limit(
        self, reset_config, caplog
    ):
        """Test warning when num > system limit."""
        rlim_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        config.set_max_files_open(num=rlim_soft + 1000)
        assert "larger than the systems soft resource limit" in caplog.text

    def test_set_max_files_open_changes_semaphore(self, reset_config):
        """Test that semaphore is actually updated."""
        config.set_max_files_open(num=100, margin=10)
        initial_value = _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]._value
        config.set_max_files_open(num=200, margin=10)
        new_value = _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]._value
        assert new_value > initial_value

    def test_set_max_files_open_edge_cases(self, reset_config):
        """Test set_max_files_open with various valid edge cases."""
        # Very small margin (but num > margin)
        config.set_max_files_open(num=100, margin=1)
        rlim_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        expected = int((min(100, rlim_soft) - 1) / 3)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]._value == expected

        # Large margin (but still less than num)
        config.set_max_files_open(num=10000, margin=1000)
        expected = int((min(10000, rlim_soft) - 1000) / 3)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]._value == expected

        # Num barely larger than margin
        config.set_max_files_open(num=35, margin=30)
        expected = int((min(35, rlim_soft) - 30) / 3)
        assert _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]._value == expected


class Test_set_slurm_max_jobs:
    def test_set_slurm_max_jobs_with_none(self, reset_config):
        """Test set_slurm_max_jobs with None (unlimited)."""
        config.set_slurm_max_jobs(None)
        assert _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB] is None

    def test_set_slurm_max_jobs_with_integer(self, reset_config):
        """Test set_slurm_max_jobs with specific integer value."""
        config.set_slurm_max_jobs(num=5)
        assert _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB] is not None
        assert isinstance(
            _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB],
            asyncio.BoundedSemaphore,
        )
        assert _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB]._value == 5

    def test_set_slurm_max_jobs_changes_semaphore(self, reset_config):
        """Test that semaphore is actually updated."""
        config.set_slurm_max_jobs(num=3)
        assert _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB]._value == 3
        config.set_slurm_max_jobs(num=7)
        assert _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB]._value == 7

    def test_set_slurm_max_jobs_twice(self, reset_config):
        """Test setting slurm max jobs twice with different values."""
        config.set_slurm_max_jobs(num=2)
        assert _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB]._value == 2
        config.set_slurm_max_jobs(num=10)
        assert _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB]._value == 10


class Test_set_trajectory_cache_type:
    def test_set_trajectory_cache_type_valid_npz(self, reset_config):
        """Test set_trajectory_cache_type with valid npz value."""
        config.set_trajectory_cache_type(cache_type="npz")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "npz"

    def test_set_trajectory_cache_type_valid_memory(self, reset_config):
        """Test set_trajectory_cache_type with valid memory value."""
        config.set_trajectory_cache_type(cache_type="memory")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

    def test_set_trajectory_cache_type_invalid_value(self, reset_config):
        """Test set_trajectory_cache_type with invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Given cache type must be one of"):
            config.set_trajectory_cache_type(cache_type="invalid")

    def test_set_trajectory_cache_type_case_insensitive(self, reset_config):
        """Test set_trajectory_cache_type is case insensitive."""
        config.set_trajectory_cache_type(cache_type="H5PY")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "h5py"
        config.set_trajectory_cache_type(cache_type="NPZ")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "npz"
        config.set_trajectory_cache_type(cache_type="MEMORY")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

    def test_set_trajectory_cache_type_copy_content_true(self, reset_config):
        """Test set_trajectory_cache_type with copy_content=True (default)."""
        config.set_trajectory_cache_type(cache_type="memory", copy_content=True)
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

    def test_set_trajectory_cache_type_copy_content_false(self, reset_config):
        """Test set_trajectory_cache_type with copy_content=False."""
        config.set_trajectory_cache_type(cache_type="memory", copy_content=False)
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

    def test_set_trajectory_cache_type_clear_old_cache_true(self, reset_config):
        """Test set_trajectory_cache_type with clear_old_cache=True."""
        config.set_trajectory_cache_type(cache_type="memory", clear_old_cache=True)
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

    def test_set_trajectory_cache_type_clear_old_cache_false(self, reset_config):
        """Test set_trajectory_cache_type with clear_old_cache=False (default)."""
        config.set_trajectory_cache_type(cache_type="memory", clear_old_cache=False)
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

    def test_set_trajectory_cache_type_when_already_set(self, reset_config):
        """Test set_trajectory_cache_type when type already set (no-op path)."""
        from asyncmd import config as cfg

        # Set cache type
        config.set_trajectory_cache_type(cache_type="memory")
        initial_value = _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE]
        assert initial_value == "memory"

        # Set same type again - should be no-op (no exception raised)
        config.set_trajectory_cache_type(cache_type="memory")

        # Verify no change
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

        # Set different type
        config.set_trajectory_cache_type(cache_type="npz")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "npz"

        # Set back to memory - should work
        config.set_trajectory_cache_type(cache_type="memory")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

    def test_set_trajectory_cache_type_transition_npz_to_memory(self, reset_config):
        """Test transition between different cache types."""
        # Start with npz
        config.set_trajectory_cache_type(cache_type="npz")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "npz"

        # Transition to memory
        config.set_trajectory_cache_type(cache_type="memory")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "memory"

        # Transition to h5py
        config.set_trajectory_cache_type(cache_type="h5py")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "h5py"

        # And back to npz
        config.set_trajectory_cache_type(cache_type="npz")
        assert _GLOBALS[_GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE] == "npz"

    def test_set_trajectory_cache_type_invalid_type(self, reset_config):
        """Test set_trajectory_cache_type with completely invalid type."""
        with pytest.raises(ValueError, match="Given cache type must be one of"):
            config.set_trajectory_cache_type(cache_type="unknown_type")

    def test_set_trajectory_cache_type_empty_string(self, reset_config):
        """Test set_trajectory_cache_type with empty string."""
        with pytest.raises(ValueError, match="Given cache type must be one of"):
            config.set_trajectory_cache_type(cache_type="")


class Test_show_config:
    def test_show_config_prints_globals(self, reset_config):
        """Test show_config prints _GLOBALS."""
        config.set_trajectory_cache_type(cache_type="memory")
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            config.show_config()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "TRAJECTORY_FUNCTION_CACHE_TYPE" in output
        assert "memory" in output
        # Verify the output is actually a dict representation
        assert output.startswith("Values controlling caching:")

    def test_show_config_prints_semaphores(self, reset_config):
        """Test show_config prints semaphores."""
        config.set_max_process(num=5)
        config.set_max_files_open(num=100)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            config.show_config()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "MAX_PROCESS" in output
        assert "MAX_FILES_OPEN" in output
        # Verify semaphore values are shown
        assert "5" in output
        # Verify the output format - just check both lines are present
        lines = output.split("\n")
        assert any("Semaphores controlling resource usage:" in line for line in lines)

    def test_show_config_after_modifications(self, reset_config):
        """Test show_config reflects changes."""
        config.set_trajectory_cache_type(cache_type="memory")
        config.set_max_process(num=7)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            config.show_config()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "memory" in output
        assert "7" in output
        # Verify order and content
        lines = output.split("\n")
        assert any(
            "memory" in line
            for line in lines
            if "TRAJECTORY_FUNCTION_CACHE_TYPE" in line
        )
        assert any("7" in line for line in lines if "MAX_PROCESS" in line)


class Test_integration_with_trajectory:
    def test_set_trajectory_cache_type_affects_new_trajectories(
        self, reset_config, tmp_path
    ):
        """Test that cache type changes affect Trajectory objects."""
        from asyncmd import Trajectory

        # Start with npz
        config.set_trajectory_cache_type(cache_type="npz")

        # Forget existing trajectories to get fresh start
        trajectory._forget_all_trajectories()

        traj1 = Trajectory(
            trajectory_files="tests/test_data/trajectory/ala_traj.trr",
            structure_file="tests/test_data/trajectory/ala.tpr",
        )

        # Verify traj1 uses npz cache
        from asyncmd.trajectory.trajectory_cache import (
            TrajectoryFunctionValueCacheInNPZ,
        )

        assert isinstance(traj1._cache, TrajectoryFunctionValueCacheInNPZ)

        # Change to memory - this updates ALL trajectories including traj1
        config.set_trajectory_cache_type(cache_type="memory")

        # traj1 should now use memory cache (global change affects all instances)
        from asyncmd.trajectory.trajectory_cache import (
            TrajectoryFunctionValueCacheInMemory,
        )

        assert isinstance(traj1._cache, TrajectoryFunctionValueCacheInMemory)

        # Create a NEW trajectory after the cache type change
        # We need to forget traj1 first to get a truly new object
        trajectory._forget_all_trajectories()
        traj2 = Trajectory(
            trajectory_files="tests/test_data/trajectory/ala_traj.trr",
            structure_file="tests/test_data/trajectory/ala.tpr",
        )

        # traj2 should also use memory cache (current setting)
        assert isinstance(traj2._cache, TrajectoryFunctionValueCacheInMemory)

        # Create yet another trajectory without forgetting
        # This should return the same object as traj2 (singleton pattern)
        traj3 = Trajectory(
            trajectory_files="tests/test_data/trajectory/ala_traj.trr",
            structure_file="tests/test_data/trajectory/ala.tpr",
        )

        # traj3 should be the same object as traj2
        assert traj2 is traj3
        assert isinstance(traj3._cache, TrajectoryFunctionValueCacheInMemory)

    def test_clear_all_cache_values_integration(self, reset_config):
        """Test clear_all_cache_values clears all trajectory caches."""
        import asyncmd.trajectory.trajectory as traj_module
        from asyncmd import Trajectory
        from asyncmd.trajectory.functionwrapper import TrajectoryFunctionWrapper
        from unittest.mock import Mock, PropertyMock

        config.set_trajectory_cache_type(cache_type="memory")

        # Forget trajectories first
        trajectory._forget_all_trajectories()

        traj = Trajectory(
            trajectory_files="tests/test_data/trajectory/ala_traj.trr",
            structure_file="tests/test_data/trajectory/ala.tpr",
        )

        # Create and register some dummy CV data
        func_id = "test_func_123"
        func_values = np.random.random(size=(len(traj), 3))
        wrapped_func = Mock(TrajectoryFunctionWrapper)
        type(wrapped_func).id = PropertyMock(return_value=func_id)
        traj._register_cached_values(values=func_values, func_wrapper=wrapped_func)

        # Verify values are cached
        retrieved = traj._retrieve_cached_values(wrapped_func)
        assert retrieved is not None
        assert len(retrieved) == len(traj)

        # Clear all caches
        traj_module.clear_all_cache_values_for_all_trajectories()

        # Verify cache is actually cleared
        retrieved_after = traj._retrieve_cached_values(wrapped_func)
        assert retrieved_after is None, "Cache values should be cleared"

    def test_update_cache_type_integration(self, reset_config, tmp_path):
        """Test update_cache_type_for_all_trajectories migrates caches."""
        from asyncmd import Trajectory
        from asyncmd.trajectory.functionwrapper import TrajectoryFunctionWrapper
        from asyncmd.trajectory.trajectory_cache import (
            TrajectoryFunctionValueCacheInNPZ,
        )
        from unittest.mock import Mock, PropertyMock
        import numpy as np

        # Forget trajectories first to start clean
        trajectory._forget_all_trajectories()

        # Also remove any npz cache files that might interfere
        npz_cache_file = TrajectoryFunctionValueCacheInNPZ.get_cache_filename(
            traj_files=["tests/test_data/trajectory/ala_traj.trr"]
        )
        if os.path.isfile(npz_cache_file):
            os.unlink(npz_cache_file)

        config.set_trajectory_cache_type(cache_type="memory")

        traj1 = Trajectory(
            trajectory_files="tests/test_data/trajectory/ala_traj.trr",
            structure_file="tests/test_data/trajectory/ala.tpr",
        )

        # Verify cache is empty before adding values
        assert len(traj1._cache) == 0, "Cache should start empty"

        # Add some cached values to traj1
        func_id = "migration_test_func"
        func_values = np.random.random(size=(len(traj1), 2))
        wrapped_func = Mock(TrajectoryFunctionWrapper)
        type(wrapped_func).id = PropertyMock(return_value=func_id)
        traj1._register_cached_values(values=func_values, func_wrapper=wrapped_func)

        # Change to npz - this should migrate traj1's cache
        config.set_trajectory_cache_type(cache_type="npz", copy_content=True)

        # traj1 should now use npz cache (same object, cache migrated)
        assert isinstance(traj1._cache, TrajectoryFunctionValueCacheInNPZ)

        # Verify cached values were migrated
        retrieved = traj1._retrieve_cached_values(wrapped_func)
        assert retrieved is not None
        assert np.array_equal(retrieved, func_values)

        # Now forget and create new trajectory with npz setting
        trajectory._forget_all_trajectories()

        # Remove npz file for the new trajectory so we don't have cached values
        npz_cache_file2 = TrajectoryFunctionValueCacheInNPZ.get_cache_filename(
            traj_files=["tests/test_data/trajectory/ala_traj.trr"]
        )
        if os.path.isfile(npz_cache_file2):
            os.unlink(npz_cache_file2)

        # Cache type should still be npz
        assert (
            config._GLOBALS[config._GLOBALS_KEYS.TRAJECTORY_FUNCTION_CACHE_TYPE]
            == "npz"
        )

        traj2 = Trajectory(
            trajectory_files="tests/test_data/trajectory/ala_traj.trr",
            structure_file="tests/test_data/trajectory/ala.tpr",
        )

        # traj2 should use npz cache
        assert isinstance(traj2._cache, TrajectoryFunctionValueCacheInNPZ)
