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
import shlex
import numpy as np


from asyncmd.mdconfig import (FlagChangeList, TypedFlagChangeList,
                              LineBasedMDConfig,
                              )


class Test_FlagChangeList:
    @pytest.mark.parametrize("no_list_data",
                             [(1, 2, 3),
                              "123",
                              1,
                              ]
                             )
    def test_init_errs(self, no_list_data):
        with pytest.raises(TypeError):
            _ = FlagChangeList(data=no_list_data)

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
    @pytest.mark.parametrize("no_list_data",
                             [(1, 2, 3),
                              "123",
                              1,
                              ]
                             )
    def test_init(self, no_list_data):
        # all data are castable to int!
        dtype = int
        flag_list = TypedFlagChangeList(data=no_list_data, dtype=dtype)
        if (getattr(no_list_data, "__len__", None) is None or isinstance(no_list_data, str)):
            # strings have a length but are considered 'singletons'
            # data has no length, so it must be the first idx
            assert flag_list[0] == dtype(no_list_data)
        else:
            # data must be an iterable
            for idx, val in enumerate(flag_list):
                assert dtype(no_list_data[idx])

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


# for the tests below we need to overwite LineBasedMDConfig._parse_line,
# otherwise we can not initialize the ABC
class DummyLineBasedMDConfig(LineBasedMDConfig):
    _KEY_VALUE_SEPARATOR = " = "
    _INTER_VALUE_CHAR = " "
    # can have multiple values per config option
    _FLOAT_PARAMS = ["float_param"]
    # must have one value per config option
    _FLOAT_SINGLETON_PARAMS = ["float_singleton_param"]
    # multiple ints per option
    _INT_PARAMS = ["int_param"]
    # one int per option
    _INT_SINGLETON_PARAMS = ["int_singleton_param"]
    # strings with only one value per option
    _STR_SINGLETON_PARAMS = ["str_singleton_param"]

    def _parse_line(self, line):
        parser = shlex.shlex(line, posix=True)
        parser.commenters = ""
        # TODO: what wordchars do we want for testing?!
        parser.wordchars += "-./~"
        tokens = list(parser)
        if len(tokens) >= 3 and tokens[1] == "=":
            # content line, either one or multiple values
            return {tokens[0]: tokens[2:]}
        elif len(tokens) == 2 and tokens[1] == "=":
            # empty line with key/token
            return {tokens[0]: []}
        else:
            raise RuntimeError("Smth went horribly wrong?!")


class Test_LineBasedMDConfig:
    def setup_method(self):
        def compare_mdconf_vals_to_beauty(mdconf, beauty):
            # make sure that we do not have any extra keys!
            assert len(mdconf) == len(beauty)
            # this also checks __getitem__
            for key, beauty_val in beauty.items():
                if "singleton" in key:
                    assert mdconf[key] == beauty_val
                else:
                    # check both ways of accessing
                    # get the full list and check the items
                    val_for_key = mdconf[key]
                    beauty_val = beauty[key]
                    assert len(val_for_key) == len(beauty_val)
                    for subval, beauty_subval in zip(val_for_key, beauty_val):
                        assert subval == beauty_subval
                    # get single items for the key and check them
                    for idx, beauty_subval in enumerate(beauty_val):
                        assert beauty[key][idx] == beauty_subval

        # bind comparission function to self to use in all tests
        self.compare_mdconf_vals_to_beauty = compare_mdconf_vals_to_beauty

        # and bind the *uninitialized* class to self
        self.dummy_class = DummyLineBasedMDConfig

    @pytest.mark.parametrize(["infile_to_parse", "beauty"],
                             [("tests/test_data/mdconfig/dummy_mdconfig.dat",
                               {"param_sans_dtype": ["test", "test123", "12.3"],
                                "float_param": [1.0, 1.1, 1.2, 10.1],
                                "float_singleton_param": 2.0,
                                "int_param": [1, 2, 3, 4, 5, 6],
                                "int_singleton_param": 6,
                                "str_singleton_param": "1string",
                                "empty_param": [],
                                }
                               ),
                              ]
                             )
    def test_changed_getitem_setitem_delitem(self, infile_to_parse, beauty):
        mdconf = self.dummy_class(original_file=infile_to_parse)
        # first check that everything is parsed as we expect
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf, beauty=beauty)
        # now check that changed=False
        assert not mdconf.changed
        # now change stuff and assert again
        # this tests __setitem__
        mdconf["some_key"] = ["123", "456"]
        assert mdconf["some_key"][0] == "123"
        assert mdconf["some_key"][1] == "456"
        assert mdconf.changed
        # reload/reparse to get a 'fresh' mdconf
        mdconf.parse()
        # check again that everything is correct
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf, beauty=beauty)
        assert not mdconf.changed
        # now set single items in sublists
        mdconf["int_param"][0] = 1337
        assert mdconf.changed
        # reparse and set singleton item
        mdconf.parse()
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf, beauty=beauty)
        assert not mdconf.changed
        mdconf["float_singleton_param"] = 10.1
        assert mdconf.changed
        # reparse to get a fresh mdconf
        mdconf.parse()
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf, beauty=beauty)
        # delete stuff (full keys first)
        del mdconf["float_param"]
        assert mdconf.changed
        # reparse and delete single items from sublist
        mdconf.parse()
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf, beauty=beauty)
        assert not mdconf.changed
        del mdconf["float_param"][0]
        assert mdconf.changed

    @pytest.mark.parametrize(["infile_to_parse", "beauty"],
                             [("tests/test_data/mdconfig/dummy_mdconfig.dat",
                               {"param_sans_dtype": ["test", "test123", "12.3"],
                                "float_param": [1.0, 1.1, 1.2, 10.1],
                                "float_singleton_param": 2.0,
                                "int_param": [1, 2, 3, 4, 5, 6],
                                "int_singleton_param": 6,
                                "str_singleton_param": "1string",
                                "empty_param": [],
                                }
                               ),
                              ]
                             )
    def test_iter_len(self, infile_to_parse, beauty):
        mdconf = self.dummy_class(original_file=infile_to_parse)
        assert len(mdconf) == len(beauty)
        for key, val in mdconf.items():
            if "singleton" in key:
                # only one element
                assert mdconf[key] == beauty[key]
            else:
                # compare all elements seperately
                for subval, subval_beauty in zip(val, beauty[key]):
                    assert subval == subval_beauty

    @pytest.mark.parametrize(["infile_to_parse", "beauty"],
                             [("tests/test_data/mdconfig/dummy_mdconfig.dat",
                               {"param_sans_dtype": ["test", "test123", "12.3"],
                                "float_param": [1.0, 1.1, 1.2, 10.1],
                                "float_singleton_param": 2.0,
                                "int_param": [1, 2, 3, 4, 5, 6],
                                "int_singleton_param": 6,
                                "str_singleton_param": "1string",
                                "empty_param": [],
                                }
                               ),
                              ]
                             )
    def test_parse_write_pickle(self, infile_to_parse, beauty, tmp_path):
        mdconf = self.dummy_class(original_file=infile_to_parse)
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf, beauty=beauty)
        # write the parsed out to a new file
        outfile = tmp_path / "out_conf.dat"
        mdconf.write(outfile=outfile)
        # check that we raise an err if file exists
        with pytest.raises(ValueError):
            mdconf.write(outfile=outfile, overwrite=False)
        # and check that it works when we pass overwrite=True
        mdconf.write(outfile=outfile, overwrite=True)
        # test pickling
        outpickle = tmp_path / "pickle.pckl"
        with open(outpickle, "wb") as pfile:
            pickle.dump(mdconf, pfile)
        # now load the pickle and parse the written file,
        # check that everything matches
        with open(outpickle, "rb") as pfile:
            mdconf_from_pickle = pickle.load(pfile)
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf_from_pickle,
                                           beauty=beauty)
        # and from parsing
        mdconf_parsed_written_out = self.dummy_class(original_file=outfile)
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf_parsed_written_out,
                                           beauty=beauty)
        # write out a modified mdconf and read it back in
        mdconf["new_value"] = ["new_value"]
        # also add it to beauty to be able to use our compare func
        beauty["new_value"] = ["new_value"]
        outfile_for_modified = tmp_path / "out_mod_conf.dat"
        mdconf.write(outfile=outfile_for_modified)
        mdconf_parsed_modified = self.dummy_class(
                            original_file=outfile_for_modified
                                                  )
        self.compare_mdconf_vals_to_beauty(mdconf=mdconf_parsed_modified,
                                           beauty=beauty)
