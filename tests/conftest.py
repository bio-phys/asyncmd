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


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    #parser.addoption(
    #    "--runold", action="store_true", default=False,
    #    help="run tests for deprecated code"
    #)
    parser.addoption(
        "--runall", action="store_true", default=False, help="run all tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    #config.addinivalue_line("markers", "old: mark test for deprecated code")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runall"):
        # --runall given in cli: do not skip any tests
        return
    #old = False
    #skip_old = pytest.mark.skip(reason="need --runold option to run")
    slow = False
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    if config.getoption("--runslow"):
        slow = True
    #if config.getoption("--runold"):
    #    old = True
    for item in items:
        if not slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
        #if not old and "old" in item.keywords:
        #    item.add_marker(skip_old)
