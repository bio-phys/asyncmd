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
import numpy as np


from asyncmd.mdconfig import FlagChangeList, TypedFlagChangeList


class Test_FlagChangeList:
    @pytest.mark.parametrize(["test_data", "data_len"],
                             [(["test", "1", "2", "3"], 4),
                              (["test", 1, 2, 3], 4),
                              ([1., 2., 3., 4.], 4),
                              ([1, 2, 3, 4], 4),
                              ([1], 1),
                              ([], 0),
                              ]
                             )
    def test_len_getitem(self, test_data, data_len):
        flag_list = FlagChangeList(data=test_data)
        # check that length is correct
        assert len(flag_list) == data_len
        # and check that all items are what we expect
        for idx in range(len(flag_list)):
            assert flag_list[idx] == test_data[idx]

    @pytest.mark.parametrize("test_data",
                             [["test", "1", "2", "3"],
                              ["test", 1, 2, 3],
                              [1., 2., 3., 4.],
                              [1, 2, 3, 4],
                              [1],
                              ]
                             )
    def test_changed_setitem_delitem_insert(self, test_data):
        flag_list = FlagChangeList(data=test_data)
        assert not flag_list.changed
        # get an item and check that everything is still good
        _ = flag_list[0]
        assert not flag_list.changed
        # modify and see that we do set changed=True
        flag_list[0] = "1234"
        assert flag_list.changed
        # reininit to get a new list with changed=False
        flag_list = FlagChangeList(data=test_data)
        assert not flag_list.changed  # as it should be
        # now delete an item and check again
        del flag_list[0]
        assert flag_list.changed
        # again reinit, this time to test insert
        flag_list = FlagChangeList(data=test_data)
        assert not flag_list.changed
        obj_to_insert = object()
        # get a random index to insert at
        if len(flag_list) > 0:
            idx = np.random.randint(low=0, high=len(flag_list))
        else:
            idx = 0
        flag_list.insert(idx, obj_to_insert)
        assert flag_list[idx] is obj_to_insert


class Test_TypedFlagChangeList:
    @pytest.mark.parametrize(["test_data", "data_len", "data_dtype"],
                             [(["test", "1", "2", "3"], 4, str),
                              (["0", 1, 2, 3], 4, int),
                              ([1., 2., 3., 4.], 4, float),
                              ([1, 2, 3, 4], 4, int),
                              ([1], 1, int),
                              ([], 0, int),  # here dtype should not matter
                              ([], 0, str),
                              ([], 0, float),
                              ]
                             )
    def test_len_getitem(self, test_data, data_len, data_dtype):
        flag_list = TypedFlagChangeList(data=test_data, dtype=data_dtype)
        # check that length is correct
        assert len(flag_list) == data_len
        # and check that all items are what we expect
        for idx in range(len(flag_list)):
            assert flag_list[idx] == data_dtype(test_data[idx])

    @pytest.mark.parametrize(["test_data", "data_dtype"],
                             [(["test", "1", "2", "3"], str),
                              # the "0" should become an int!
                              (["0", 1, 2, 3], int),
                              ([1., 2., 3., 4.], float),
                              ([1, 2, 3, 4], int),
                              # the ints should become floats!
                              ([1, 2., 3, 4], float),
                              ([1], int),
                              ]
                             )
    def test_changed_setitem_delitem_insert(self, test_data, data_dtype):
        flag_list = TypedFlagChangeList(data=test_data, dtype=data_dtype)
        assert not flag_list.changed
        # get an item and check that everything is still good
        _ = flag_list[0]
        assert not flag_list.changed
        # modify and see that we do set changed=True
        flag_list[0] = "1234"  # can be converted to int, float and str
        assert flag_list.changed
        # reininit to get a new list with changed=False
        flag_list = TypedFlagChangeList(data=test_data, dtype=data_dtype)
        assert not flag_list.changed  # as it should be
        # now delete an item and check again
        del flag_list[0]
        assert flag_list.changed
        # again reinit, this time to test insert
        flag_list = TypedFlagChangeList(data=test_data, dtype=data_dtype)
        assert not flag_list.changed
        obj_to_insert = "1234"  # again: castable tol int, float and str
        # get a random index to insert at
        if len(flag_list) > 0:
            idx = np.random.randint(low=0, high=len(flag_list))
        else:
            idx = 0
        flag_list.insert(idx, obj_to_insert)
        # cast the object to right type
        obj_to_insert = data_dtype(obj_to_insert)
        assert flag_list[idx] == obj_to_insert
