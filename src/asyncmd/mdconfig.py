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
This module contains abstract base classes to use for MD config file parsing.

It contains the MDConfig class, which only serves to define the interface and
the LineBasedMDConfig class, which implements useful methods to parse MD configuration
files in which options never span over multiple lines, i.e. in which every line
contains only one (or multiple) options.
"""
import os
import abc
import shutil
import logging
import collections

from .tools import TypedFlagChangeList


logger = logging.getLogger(__name__)


class MDConfig(collections.abc.MutableMapping):
    """Abstract base class only to define the interface."""

    @abc.abstractmethod
    def parse(self):
        """Should read original file and populate self with key, value pairs."""
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, outfile):
        """Should write out current config stored in self to outfile."""
        raise NotImplementedError


class LineBasedMDConfig(MDConfig):
    """
    Abstract base class for line based parsing and writing.

    Subclasses must implement `_parse_line()` method and should set the
    appropriate separator characters for their line format.
    We assume that every line/option can be parsed and written on its own!
    We assume the order of the options in the written file is not relevant!
    We represent every line/option with a key (str), list of values pair.
    Values can have a specific type (e.g. int or float) or default to str.
    """
    # NOTE: Initially written for gmx, but we already had e.g. namd in mind and
    # tried to make this as general as possible

    # these are the gmx mdp options but should be fairly general
    # (i.e. work at least for namd?)
    _KEY_VALUE_SEPARATOR = " = "
    _INTER_VALUE_CHAR = " "
    # NOTE on typing
    # use these to specify config parameters that are of type int or float
    # parsed lines with dict key matching will then be converted
    # any lines not matching will be left in their default str type
    _FLOAT_PARAMS: list[str] = []  # can have multiple values per config option
    _FLOAT_SINGLETON_PARAMS: list[str] = []  # must have one value per config option
    _INT_PARAMS: list[str] = []  # multiple int per option
    _INT_SINGLETON_PARAMS: list[str] = []  # one int per option
    _STR_SINGLETON_PARAMS: list[str] = []  # strings with only one value per option
    # NOTE on SPECIAL_PARAM_DISPATCH
    # can be used to set custom type convert functions on a per parameter basis
    # the key must match the key in the dict for in the parsed line,
    # the value must be a function taking the corresponding (parsed) line and
    # which must return a FlagChangeList or subclass thereof
    # this function will also be called with the new list of value(s) when the
    # option is changed, i.e. it must also be able to check and cast a list of
    # new values into the expected FlagChangeList format
    # [note that it is probably easiest to subclass TypedFlagChangeList and
    #  overwrite only the '_check_type()' method]
    _SPECIAL_PARAM_DISPATCH: dict[str, collections.abc.Callable] = {}

    def __init__(self, original_file: str) -> None:
        """
        Initialize a :class:`LineBasedMDConfig`.

        Parameters
        ----------
        original_file : str
            Path to original config file (absolute or relative).
        """
        self._config: dict[str, TypedFlagChangeList | int | float | str] = {}
        self._changed = False
        self._type_dispatch = self._construct_type_dispatch()
        # property to set/check file and parse to config dictionary all in one
        self.original_file = original_file

    def _construct_type_dispatch(self):
        def convert_len1_list_or_singleton(val, dtype):
            # helper func that accepts len1 lists
            # (as expected from `_parse_line`)
            # but that also accepts single values and converts them to given
            # dtype (which is what we expect can/will happen when the users set
            # singleton vals, i.e. "val" instead of ["val"]
            if isinstance(val, str) or getattr(val, '__len__', None) is None:
                return dtype(val)
            return dtype(val[0])

        # construct type conversion dispatch
        type_dispatch = collections.defaultdict(
                                # looks a bit strange, but the factory func
                                # is called to produce the default value, i.e.
                                # we need a func that returns our default func
                                lambda:
                                lambda l: TypedFlagChangeList(data=l,
                                                              dtype=str)
                                                      )
        type_dispatch.update({param: lambda l: TypedFlagChangeList(
                                                                data=l,
                                                                dtype=float
                                                                   )
                              for param in self._FLOAT_PARAMS})
        type_dispatch.update({param: lambda v: convert_len1_list_or_singleton(
                                                                val=v,
                                                                dtype=float,
                                                                              )
                              for param in self._FLOAT_SINGLETON_PARAMS})
        type_dispatch.update({param: lambda l: TypedFlagChangeList(
                                                                data=l,
                                                                dtype=int,
                                                                   )
                              for param in self._INT_PARAMS})
        type_dispatch.update({param: lambda v: convert_len1_list_or_singleton(
                                                                val=v,
                                                                dtype=int,
                                                                              )
                              for param in self._INT_SINGLETON_PARAMS})
        type_dispatch.update({param: lambda v: convert_len1_list_or_singleton(
                                                                val=v,
                                                                dtype=str,
                                                                              )
                              for param in self._STR_SINGLETON_PARAMS})
        type_dispatch.update(self._SPECIAL_PARAM_DISPATCH)
        return type_dispatch

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_type_dispatch"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._type_dispatch = self._construct_type_dispatch()

    @abc.abstractmethod
    def _parse_line(self, line: str) -> dict:
        """
        Parse a line of the configuration file and return a :class:`dict`.

        Parameters
        ----------
        line : str
            A single line of the read-in configuration file

        Returns
        ------
        parsed : dict
            Dictionary with a single (key, list of value(s)) pair representing
            the parsed line.
        """
        # NOTE: this is the only function needed to complete the class,
        #       the rest of this metaclass assumes the following for this func:
        # it must parse a single line and return the key, list of value(s) pair
        # as a dict with one item, e.g. {key: list of value(s)}
        # if the line is parsed as comment the dict must be empty, e.g. {}
        # if the option/key is present but without value the list must be empty
        # e.g. {key: []}
        raise NotImplementedError

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value) -> None:
        typed_value = self._type_dispatch[key](value)
        self._config[key] = typed_value
        self._changed = True

    def __delitem__(self, key) -> None:
        self._config.__delitem__(key)
        self._changed = True

    def __iter__(self):
        return self._config.__iter__()

    def __len__(self) -> int:
        return self._config.__len__()

    def __repr__(self) -> str:  # pragma: no cover
        return str({"changed": self._changed,
                    "original_file": self.original_file,
                    "content": self._config.__repr__(),
                    }
                   )

    def __str__(self) -> str:  # pragma: no cover
        repr_str = (f"{type(self)} has been changed since parsing: "
                    + f"{self._changed}\n"
                    )
        repr_str += "Current content:\n"
        repr_str += "----------------\n"
        for key, val in self.items():
            repr_str += f"{key} : {val}\n"
        return repr_str

    @property
    def original_file(self) -> str:
        """
        Return the original config file this :class:`LineBasedMDConfig` parsed.

        Returns
        -------
        str
            Path to the original file.
        """
        return self._original_file

    @original_file.setter
    def original_file(self, value: str) -> None:
        # NOTE: (re)setting the file also replaces the current config with
        #       what we parse from that file
        value = os.path.relpath(value)
        if not os.path.isfile(value):
            raise ValueError(f"Can not access the file {value}")
        self._original_file = value
        self.parse()

    @property
    def changed(self) -> bool:
        """
        Indicate if the current configuration differs from original_file.

        Returns
        -------
        bool
            Whether we changed the configuration w.r.t. ``original_file``.
        """
        # NOTE: we default to False, i.e. we expect that anything that
        #       does not have a self.changed attribute is not a container
        #       and we (the dictionary) would know that it changed
        return self._changed or any(getattr(v, "changed", False)
                                    for v in self._config.values())

    def parse(self):
        """Parse the current ``self.original_file`` to update own state."""
        with open(self.original_file, "r", encoding="locale") as f:
            # NOTE: we split at newlines on all platforms by iterating over the
            #       file, i.e. python takes care of the different platforms and
            #       newline chars for us :)
            parsed = {}
            for line in f:
                line_parsed = self._parse_line(line.rstrip("\n"))
                # check for duplicate options, we warn but take the last one
                for key in line_parsed:
                    try:
                        # check if we already have a value for that option
                        _ = parsed[key]
                    except KeyError:
                        # as it should be
                        pass
                    else:
                        # warn that we will only keep the last occurrence of key
                        logger.warning("Parsed duplicate configuration option "
                                       "(%s). Last values encountered take "
                                       "precedence.", key)
                parsed.update(line_parsed)
        # convert the known types
        self._config = {key: self._type_dispatch[key](value)
                        for key, value in parsed.items()}
        self._changed = False

    def write(self, outfile: str, overwrite: bool = False) -> None:
        """
        Write current configuration to outfile.

        Parameters
        ----------
        outfile : str
            Path to outfile (relative or absolute).
        overwrite : bool, optional
            If True overwrite existing files, by default False.

        Raises
        ------
        ValueError
            Raised when `overwrite=False` but `outfile` exists.
        """
        outfile = os.path.relpath(outfile)
        if os.path.exists(outfile) and not overwrite:
            raise ValueError(f"overwrite=False and file exists ({outfile}).")
        if not self.changed:
            # just copy the original
            shutil.copy2(src=self.original_file, dst=outfile)
        else:
            # construct content for new file
            lines = []
            for key, value in self._config.items():
                line = f"{key}{self._KEY_VALUE_SEPARATOR}"
                if isinstance(value, (str, float, int)):
                    # it is a string/float/int singleton option
                    line += f"{value}"
                else:
                    # not a singleton, so lets try to iterate over it
                    try:
                        line += self._INTER_VALUE_CHAR.join(str(v)
                                                            for v in value
                                                            )
                    except TypeError:
                        # Note: need this except to catch user-added types
                        #       (via special param dispatch), that are not
                        #       str/float/int singletons but also not iterable
                        line += f"{value}"
                lines += [line]
            # concatenate the lines and write out at once
            with open(outfile, "w", encoding="locale") as f:
                f.write("\n".join(lines))
