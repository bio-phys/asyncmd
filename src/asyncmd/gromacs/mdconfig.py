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
import shlex
import logging
from ..mdconfig import LineBasedMDConfig

logger = logging.getLogger(__name__)


class MDP(LineBasedMDConfig):
    """
    Read, parse, (alter) and write gromacs .mdp files.

    Make all options set in a given mdp file available via a dictionary of
    option, list of values pairs. Includes automatic types for known options
    and keeps track if any options have been changed compared to the original
    file.

    Notable methods:
    ----------------
    write - write the current (modified) configuration state to a given file
    parse - read the current original_file and update own state with it
    """

    _KEY_VALUE_SEPARATOR = " = "
    _INTER_VALUE_CHAR = " "
    # MDP param types, sorted into groups/by headings as in the gromacs manual
    # https://manual.gromacs.org/documentation/5.1/user-guide/mdp-options.html
    _FLOAT_PARAMS = []
    _FLOAT_SINGLETON_PARAMS = []
    _INT_PARAMS = []
    _INT_SINGLETON_PARAMS = []
    # Run control
    _FLOAT_SINGLETON_PARAMS += ["tinit", "dt"]
    _INT_SINGLETON_PARAMS += ["nsteps", "init-step", "simulation-part",
                              "nstcomm"]
    # Langevin dynamics
    _FLOAT_SINGLETON_PARAMS += ["bd-fric"]
    _INT_SINGLETON_PARAMS += ["ld-seed"]
    # Energy minimization
    _FLOAT_SINGLETON_PARAMS += ["emtol", "emstep"]
    _INT_SINGLETON_PARAMS += ["nstcgsteep", "nbfgscorr"]
    # Shell Molecular Dynamics
    _FLOAT_SINGLETON_PARAMS += ["fcstep"]
    _INT_SINGLETON_PARAMS += ["niter"]
    # Test particle insertion
    _FLOAT_SINGLETON_PARAMS += ["rtpi"]
    # Output control
    # NOTE: 'nstxtcout' and 'xtc-precision' are deprecated since GMX v5.0
    _FLOAT_SINGLETON_PARAMS += ["compressed-x-precision", "xtc-precision"]
    _INT_SINGLETON_PARAMS += ["nstxout", "nstvout", "nstfout", "nstlog",
                              "nstcalcenergy", "nstenergy",
                              "nstxout-compressed", "nstxtcout"]
    # Neighbor searching
    # NOTE: 'rlistlong' and 'nstcalclr' are used with group cutoff scheme,
    #       i.e. deprecated since GMX v5.0
    _FLOAT_SINGLETON_PARAMS += ["verlet-buffer-tolerance", "rlist",
                                "rlistlong"]
    _INT_SINGLETON_PARAMS += ["nstlist", "nstcalclr"]
    # Electrostatics
    _FLOAT_SINGLETON_PARAMS += ["rcoulomb-switch", "rcoulomb", "epsilon-r",
                                "epsilon-rf"]
    # Van der Waals
    _FLOAT_SINGLETON_PARAMS += ["rvdw-switch", "rvdw"]
    # Ewald
    _FLOAT_SINGLETON_PARAMS += ["fourierspacing", "ewald-rtol",
                                "ewald-rtol-lj"]
    _INT_SINGLETON_PARAMS += ["fourier-nx", "fourier-ny", "fourier-nz",
                              "pme-order"]
    # Temperature coupling
    _FLOAT_PARAMS += ["tau-t", "ref-t"]
    _INT_SINGLETON_PARAMS += ["nsttcouple", "nh-chain-length"]
    # Pressure coupling
    _FLOAT_SINGLETON_PARAMS += ["tau-p"]
    _FLOAT_PARAMS += ["compressibility", "ref-p"]
    _INT_SINGLETON_PARAMS += ["nstpcouple"]
    # Simulated annealing
    _FLOAT_PARAMS += ["annealing-time", "annealing-temp"]
    _INT_PARAMS += ["annealing-npoints"]
    # Velocity generation
    _FLOAT_SINGLETON_PARAMS += ["gen-temp"]
    _INT_SINGLETON_PARAMS += ["gen-seed"]
    # Bonds
    _FLOAT_SINGLETON_PARAMS += ["shake-tol", "lincs-warnangle"]
    _INT_SINGLETON_PARAMS += ["lincs-order", "lincs-iter"]
    # TODO: Walls and everything below in the GMX manual

    def _parse_line(self, line):
        parser = shlex.shlex(line, posix=True)
        parser.commenters = ";"
        parser.wordchars += "-./"  # ./ to not split floats and file paths
        tokens = list(parser)
        # gromacs mdp can have more than one token/value to the RHS of the '='
        if len(tokens) == 0:
            # (probably) a comment line
            logger.debug(f"mdp line parsed as comment: {line}")
            return {}
        elif len(tokens) >= 3 and tokens[1] == "=":
            # lines with content: make sure we correctly parsed the '='
            # always return a list for values
            return {self._key_char_replace(tokens[0]): tokens[2:]}
        elif len(tokens) == 2 and tokens[1] == "=":
            # lines with empty options, e.g. 'define = '
            return {self._key_char_replace(tokens[0]): []}
        else:
            # no idea what happend here...best to let the user have a look :)
            raise ValueError(f"Could not parse the following mdp line: {line}")

    def _key_char_replace(self, key):
        # make it possible to use CHARMM-GUI generated mdp-files, because
        # CHARMM-GUI uses "_" instead of "-" in the option names,
        # which seems to be an undocumented gromacs feature,
        # i.e. gromacs reads these mdp-files without complaints :)
        # we will however stick with "-" all the time to make sure every option
        # exists only once, i.e. we convert all keys to use "-" instead of "_"
        return key.replace("_", "-")

    def __getitem__(self, key):
        return super().__getitem__(self._key_char_replace(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self._key_char_replace(key), value)

    def __delitem__(self, key):
        return super().__delitem__(self._key_char_replace(key))
