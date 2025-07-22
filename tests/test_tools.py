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
import numpy as np


from asyncmd.tools import (FlagChangeList, TypedFlagChangeList,
                           remove_file_if_exist, remove_file_if_exist_async,
                           )


class Test_remove_file_if_exist:
    @pytest.mark.parametrize("file_exists", [True, False])
    def test_sync(self, tmp_path, file_exists):
        file_path = os.path.join(tmp_path, "test_file.dat")
        if file_exists:
            # make a file
            open(file_path, "w", encoding="locale").close()
        # remove it
        remove_file_if_exist(file_path)
        # and make sure it is gone
        assert not os.path.exists(file_path)

    @pytest.mark.parametrize("file_exists", [True, False])
    @pytest.mark.asyncio
    async def test_async(self, tmp_path, file_exists):
        file_path = os.path.join(tmp_path, "test_file.dat")
        if file_exists:
            # make a file
            open(file_path, "w", encoding="locale").close()
        # remove it
        await remove_file_if_exist_async(file_path)
        # and make sure it is gone
        assert not os.path.exists(file_path)


class Test_FlagChangeList:
    @pytest.mark.parametrize("no_iter_data",
                             [1,  # ints are not iterable
                              1.,  # floats are also not iterable
                              ]
                             )
    def test_init_errs(self, no_iter_data):
        with pytest.raises(TypeError):
            _ = FlagChangeList(data=no_iter_data)

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
        for idx, value in enumerate(flag_list):
            assert value == test_data[idx]

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

    @pytest.mark.parametrize("to_add",
                             [[2, 3, 4],
                              (2, 3, 4),
                              FlagChangeList([2, 3, 4]),
                              "234",
                              ])
    def test_add(self, to_add):
        # we always start with a FlagChangeList with a 1
        beauty = [1] + list(to_add)
        flag_list = FlagChangeList(data=[1])
        flag_list_added = flag_list + to_add
        # make sure we created a new object
        assert flag_list is not flag_list_added
        # make sure the new object knows it has not been changed since creation
        assert flag_list_added.changed is False
        # and make sure it contains what we expect
        for idx, val in enumerate(beauty):
            assert flag_list_added[idx] == val

    @pytest.mark.parametrize("to_add",
                             [[2, 3, 4],
                              (2, 3, 4),
                              FlagChangeList([2, 3, 4]),
                              "234",
                              ])
    def test_iadd(self, to_add):
        # we always start with a FlagChangeList with a 1
        beauty = [1] + list(to_add)
        flag_list = FlagChangeList(data=[1])
        flag_list_initial = flag_list
        flag_list += to_add
        # make sure we operated inplace
        assert flag_list is flag_list_initial
        # make sure the list knows it has been changed since creation
        assert flag_list.changed is True
        # and make sure it contains what we expect
        for idx, val in enumerate(beauty):
            assert flag_list[idx] == val


class Test_TypedFlagChangeList:
    @pytest.mark.parametrize("dtype", [int, float, str])
    @pytest.mark.parametrize("test_data",
                             [(1, 2, 3),
                              "123",
                              1,
                              ]
                             )
    def test_init(self, test_data, dtype):
        # NOTE: all data are castable to str, int, and float!
        flag_list = TypedFlagChangeList(data=test_data, dtype=dtype)
        if (getattr(test_data, "__iter__", None) is None
                or isinstance(test_data, str)):
            # strings have a length but are considered 'singletons'
            # (at least in the context of TypeFlagChangeList)
            # data has no iter, so it must be the first idx
            assert flag_list[0] == dtype(test_data)
        else:
            # data must be an iterable
            for idx, val in enumerate(flag_list):
                assert val == dtype(test_data[idx])

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
        for idx, value in enumerate(flag_list):
            assert value == data_dtype(test_data[idx])

    @pytest.mark.parametrize(["test_data", "data_dtype"],
                             [(["test", "1", "2", "3"], str),
                              # the "0" should become an int!
                              (["0", 1, 2, 3], int),
                              ([1., 2., 3., 4.], float),
                              ([1, 2, 3, 4], int),
                              # the ints should become floats!
                              ([1, 2., 3, 4], float),
                              ]
                             )
    def test_changed_setitem_delitem_insert(self, test_data, data_dtype):
        flag_list = TypedFlagChangeList(data=test_data, dtype=data_dtype)
        assert not flag_list.changed
        # get an item and check that everything is still good
        _ = flag_list[0]
        assert not flag_list.changed
        # modify and see that we do set changed=True
        test_insert = "1234"  # can be converted to int, float and str
        flag_list[0] = test_insert
        assert flag_list.changed
        # check that we have set it as expected
        assert flag_list[0] == data_dtype("1234")
        # reininit to get a new list with changed=False
        flag_list = TypedFlagChangeList(data=test_data, dtype=data_dtype)
        assert not flag_list.changed  # as it should be
        # modify using a slice
        test_insert = ["1234", 1234]
        flag_list[0:2] = test_insert
        assert flag_list.changed
        # check that we have set the values in the list as expected
        assert all(v == data_dtype(beauty) for v, beauty
                   in zip(flag_list[0:2], test_insert)
                   )
        # reininit to test delete
        flag_list = TypedFlagChangeList(data=test_data, dtype=data_dtype)
        assert not flag_list.changed
        # now delete an item and check again
        del flag_list[0]
        assert flag_list.changed
        # again reinit, this time to test insert
        flag_list = TypedFlagChangeList(data=test_data, dtype=data_dtype)
        assert not flag_list.changed
        obj_to_insert = "1234"  # again: castable to int, float and str
        # get a random index to insert at
        if len(flag_list) > 0:
            idx = np.random.randint(low=0, high=len(flag_list))
        else:
            idx = 0
        flag_list.insert(idx, obj_to_insert)
        # cast the object to right type
        obj_to_insert = data_dtype(obj_to_insert)
        assert flag_list[idx] == obj_to_insert

    @pytest.mark.parametrize(["to_add", "error"],
                             [[[2, 3, 4], False],
                              [(2, 3, 4), False],
                              [FlagChangeList([2, 3, 4]), False],
                              [TypedFlagChangeList([2, 3, 4], dtype=int), False],
                              [TypedFlagChangeList([2, 3, 4], dtype=float), False],
                              # this one should fail because we can not cast
                              [TypedFlagChangeList(["a", "b", 4], dtype=str), True],
                              ])
    def test_add(self, to_add, error):
        flag_list = TypedFlagChangeList(data=[1], dtype=int)
        if error:
            with pytest.raises(ValueError):
                flag_list_added = flag_list + to_add
            return  # less indent for the part below :)
        # construct the truth value/beauty to test against
        # we always start with a TypedFlagChangeList with a 1
        beauty = [1] + [int(v) for v in to_add]
        flag_list_added = flag_list + to_add
        # make sure we created a new object
        assert flag_list is not flag_list_added
        # make sure the new object knows it has not been changed since creation
        assert flag_list_added.changed is False
        # make sure the datatype is correct
        assert flag_list.dtype == int
        # and make sure it contains what we expect
        for idx, val in enumerate(beauty):
            assert flag_list_added[idx] == val

    @pytest.mark.parametrize(["to_add", "error"],
                             [[[2, 3, 4], False],
                              [(2, 3, 4), False],
                              [FlagChangeList([2, 3, 4]), False],
                              [TypedFlagChangeList([2, 3, 4], dtype=int), False],
                              [TypedFlagChangeList([2, 3, 4], dtype=float), False],
                              # this one should fail because we can not cast
                              [TypedFlagChangeList(["a", "b", 4], dtype=str), True],
                              ])
    def test_iadd(self, to_add, error):
        flag_list = TypedFlagChangeList(data=[1], dtype=int)
        flag_list_initial = flag_list
        if error:
            with pytest.raises(ValueError):
                flag_list += to_add
            return  # less indent below :)
        # construct the truth value/beauty to test against
        # we always start with a TypedFlagChangeList with a 1
        beauty = [1] + [int(v) for v in to_add]
        flag_list += to_add
        # make sure we operated inplace
        assert flag_list is flag_list_initial
        # make sure the list knows it has been changed since creation
        assert flag_list.changed is True
        # make sure the datatype is correct
        assert flag_list.dtype == int
        # and make sure it contains what we expect
        for idx, val in enumerate(beauty):
            assert flag_list[idx] == val
