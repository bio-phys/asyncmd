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
    Parse, modify and write gromacs .mdp files.

    Make all options set in a given mdp file available via a dictionary of
    option, list of values pairs. Includes automatic types for known options
    and keeps track if any options have been changed compared to the original
    file.

    Parameters
    ----------
    original_file : str
        absolute or relative path to original config file to parse

    Methods
    -------
    write(outfile)
        write the current (modified) configuration state to a given file
    parse()
        read the current original_file and update own state with it
    """

    _KEY_VALUE_SEPARATOR = " = "
    _INTER_VALUE_CHAR = " "
    # MDP param types, sorted into groups/by headings as in the gromacs manual
    # https://manual.gromacs.org/documentation/current/user-guide/mdp-options.html
    _FLOAT_PARAMS = []
    _FLOAT_SINGLETON_PARAMS = []
    _INT_PARAMS = []
    _INT_SINGLETON_PARAMS = []
    _STR_SINGLETON_PARAMS = []
    # Run control
    _FLOAT_SINGLETON_PARAMS += ["tinit", "dt"]
    _INT_SINGLETON_PARAMS += ["nsteps", "init-step", "simulation-part",
                              "nstcomm"]
    _STR_SINGLETON_PARAMS += ["integrator", "comm-mode"]
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
    _STR_SINGLETON_PARAMS += ["cutoff-scheme", "ns-type", "pbc",
                              "periodic-molecules"]
    # Electrostatics
    _FLOAT_SINGLETON_PARAMS += ["rcoulomb-switch", "rcoulomb", "epsilon-r",
                                "epsilon-rf"]
    _STR_SINGLETON_PARAMS += ["coulombtype", "coulomb-modifier"]
    # Van der Waals
    _FLOAT_SINGLETON_PARAMS += ["rvdw-switch", "rvdw"]
    _STR_SINGLETON_PARAMS += ["vdwtype", "vdw-modifier", "DispCorr"]
    # Ewald
    _FLOAT_SINGLETON_PARAMS += ["fourierspacing", "ewald-rtol",
                                "ewald-rtol-lj"]
    _INT_SINGLETON_PARAMS += ["fourier-nx", "fourier-ny", "fourier-nz",
                              "pme-order"]
    _STR_SINGLETON_PARAMS += ["lj-pme-comb-rule", "ewald-geometry"]
    # Temperature coupling
    _FLOAT_PARAMS += ["tau-t", "ref-t"]
    _INT_SINGLETON_PARAMS += ["nsttcouple", "nh-chain-length"]
    _STR_SINGLETON_PARAMS += ["tcoupl", "Tcoupl"]  # GMX accepts both versions
    # Pressure coupling
    _FLOAT_SINGLETON_PARAMS += ["tau-p"]
    _FLOAT_PARAMS += ["compressibility", "ref-p"]
    _INT_SINGLETON_PARAMS += ["nstpcouple"]
    _STR_SINGLETON_PARAMS += ["pcoupl", "Pcoupl",  # GMX accepts both versions
                              "pcoupltype", "refcoord-scaling"]
    # Simulated annealing
    _FLOAT_PARAMS += ["annealing-time", "annealing-temp"]
    _INT_PARAMS += ["annealing-npoints"]
    # Velocity generation
    _FLOAT_SINGLETON_PARAMS += ["gen-temp"]
    _INT_SINGLETON_PARAMS += ["gen-seed"]
    _STR_SINGLETON_PARAMS += ["gen-vel"]
    # Bonds
    _FLOAT_SINGLETON_PARAMS += ["shake-tol", "lincs-warnangle"]
    _INT_SINGLETON_PARAMS += ["lincs-order", "lincs-iter"]
    _STR_SINGLETON_PARAMS += ["constraints", "constraint-algorithm",
                              # the next two are referencing the same option
                              "continuation", "unconstrained-start",
                              "morse"]
    # Walls
    _INT_SINGLETON_PARAMS += ["nwall"]
    _STR_SINGLETON_PARAMS += ["wall-atomtype", "wall-type"]
    _FLOAT_SINGLETON_PARAMS += ["wall-r-linpot", "wall-density",
                                "wall-ewald-zfac"]
    # COM Pulling
    _STR_SINGLETON_PARAMS += ["pull", "pull-print-com", "pull-print-ref-value",
                              "pull-print-components",
                              "pull-pbc-ref-prev-step-com",
                              "pull-xout-average", "pull-fout-average"]
    _FLOAT_SINGLETON_PARAMS += ["pull-cylinder-r", "pull-constr-tol"]
    _INT_SINGLETON_PARAMS += ["pull-nstxout", "pull-nstfout", "pull-ngroups",
                              "pull-ncoords"]
    # Note: gromacs has a maximum of 256 groups, see e.g.
    # https://manual.gromacs.org/current/reference-manual/algorithms/group-concept.html
    # I (hejung) did not find a maximum for the number of pull coordinates,
    # but we go with 512 here for now (assuming at most two coords per group)
    _STR_SINGLETON_PARAMS += (
            [f"pull-group{n}-name" for n in range(1, 257)]
            + [f"pull-coord{n}-type" for n in range(1, 513)]
            + [f"pull-coord{n}-potential-provider" for n in range(1, 513)]
            + [f"pull-coord{n}-geometry" for n in range(1, 513)]
            + [f" pull-coord{n}-start" for n in range(1, 513)]
                              )
    _FLOAT_SINGLETON_PARAMS += (
            [f"pull-coord{n}-init" for n in range(1, 513)]
            + [f"pull-coord{n}-rate" for n in range(1, 513)]
            + [f"pull-coord{n}-k" for n in range(1, 513)]
            + [f"pull-coord{n}-kB" for n in range(1, 513)]
                                )
    _FLOAT_PARAMS += (
            [f"pull-group{n}-weights" for n in range(1, 257)]
            + [f"pull-coord{n}-origin" for n in range(1, 513)]
            + [f"pull-coord{n}-vec" for n in range(1, 513)]
                      )
    _INT_SINGLETON_PARAMS += [f"pull-group{n}-pbcatom" for n in range(1, 257)]
    _INT_PARAMS += [f"pull-coord{n}-groups" for n in range(1, 513)]
    # AWH adaptive biasing
    # Note we assume a maximum number of 20 awh coordinates, each consisting of
    # a maximum of 4 (pull coordinate) dimensions
    _STR_SINGLETON_PARAMS += (
            ["awh", "awh-potential", "awh-share-multisim"]
            + [f"awh{n}-growth" for n in range(1, 21)]
            + [f"awh{n}-equilibrate-histogram" for n in range(1, 21)]
            + [f"awh{n}-target" for n in range(1, 21)]
            + [f"awh{n}-user-data" for n in range(1, 21)]
            + [f"awh{n}-dim{d}-coord-provider"
               for n in range(1, 21) for d in range(1, 5)]
                              )
    _INT_SINGLETON_PARAMS += (
            ["awh-seed", "awh-nstout", "awh-nstsample", "awh-nsamples-update",
             "awh-nbias"]
            + [f"awh{n}-share-group" for n in range(1, 21)]
            + [f"awh{n}-ndim" for n in range(1, 21)]
            + [f"awh{n}-dim{d}-coord-index"
               for n in range(1, 21) for d in range(1, 5)]
                              )
    _FLOAT_SINGLETON_PARAMS += (
            [f"awh{n}-error-init" for n in range(1, 21)]
            + [f"awh{n}-target-beta-scaling" for n in range(1, 21)]
            + [f"awh{n}-target-cutoff" for n in range(1, 21)]
            + [f"awh{n}-dim{d}-force-constant"
               for n in range(1, 21) for d in range(1, 5)]
            + [f"awh{n}-dim{d}-start"
               for n in range(1, 21) for d in range(1, 5)]
            + [f"awh{n}-dim{d}-end"
               for n in range(1, 21) for d in range(1, 5)]
            + [f"awh{n}-dim{d}-period"
               for n in range(1, 21) for d in range(1, 5)]
            + [f"awh{n}-dim{d}-diffusion"
               for n in range(1, 21) for d in range(1, 5)]
            + [f"awh{n}-dim{d}-cover-diameter"
               for n in range(1, 21) for d in range(1, 5)]
                                )
    # Enforced rotation
    # Note: rotation groups are zero indexed, we assume a maximum of 30
    _STR_SINGLETON_PARAMS += (["rotation"]
                              + [f"rot-group{n}" for n in range(30)]
                              + [f"rot-type{n}" for n in range(30)]
                              + [f"rot-massw{n}" for n in range(30)]
                              + [f"rot-fit-method{n}" for n in range(30)]
                              )
    _INT_SINGLETON_PARAMS += ["rot-ngroups", "rot-nstrout", "rot-nstsout"]
    _FLOAT_SINGLETON_PARAMS += ([f"rot-rate{n}" for n in range(30)]
                                + [f"rot-k{n}" for n in range(30)]
                                + [f"rot-slab-dist{n}" for n in range(30)]
                                + [f"rot-min-gauss{n}" for n in range(30)]
                                + [f"rot-eps{n}" for n in range(30)]
                                + [f"rot-potfit-step{n}" for n in range(30)]
                                )
    _FLOAT_PARAMS += ([f"rot-vec{n}" for n in range(30)]
                      + [f"rot-pivot{n}" for n in range(30)]
                      )
    # NMR refinement
    _STR_SINGLETON_PARAMS += ["disre", "disre-weighting", "disre-mixed",
                              "orire", "orire-fitgrp"]
    _FLOAT_SINGLETON_PARAMS += ["disre-fc", "disre-tau", "orire-fc",
                                "orire-tau"]
    _INT_SINGLETON_PARAMS += ["nstdisreout", "nstorireout"]
    # Free energy calculations
    _STR_SINGLETON_PARAMS += ["free-energy", "expanded", "sc-coul",
                              "couple-moltype", "couple-lambda0",
                              "couple-lambda1", "couple-intramol",
                              "dhdl-derivatives", "dhdl-print-energy",
                              "separate-dhdl-file"]
    _FLOAT_SINGLETON_PARAMS += ["init-lambda", "delta-lambda", "sc-alpha",
                                "sc-sigma", "dh-hist-spacing"]
    _INT_SINGLETON_PARAMS += ["init-lambda-state", "calc-lambda-neighbors",
                              "sc-r-power", "sc-power", "nstdhdl",
                              "dh-hist-size"]
    _FLOAT_PARAMS += ["fep-lambdas", "coul-lambdas", "vdw-lambdas",
                      "bonded-lambdas", "restraint-lambdas", "mass-lambdas",
                      "temperature-lambdas"]
    # Expanded Ensemble calculations
    _INT_SINGLETON_PARAMS += ["nstexpanded", "lmc-seed", "lmc-repeats",
                              "lmc-gibbsdelta", "lmc-forced-nstart",
                              "nst-transition-matrix", "mininum-var-min"]
    _STR_SINGLETON_PARAMS += ["lmc-stats", "lmc-mc-move", "wl-oneovert",
                              "symmetrized-transition-matrix",
                              "lmc-weights-equil", "simulated-tempering",
                              "simulated-tempering-scaling"]
    _FLOAT_SINGLETON_PARAMS += ["mc-temperature", "wl-ratio", "wl-scale",
                                "init-wl-delta", "sim-temp-low",
                                "sim-temp-high"]
    _FLOAT_PARAMS += ["init-lambda-weights"]
    # Non-equilibrium MD
    _FLOAT_SINGLETON_PARAMS += ["accelerate", "cos-acceleration"]
    _FLOAT_PARAMS += ["deform"]
    # Electric fields
    _FLOAT_PARAMS += ["electric-field-x", "electric-field-y",
                      "electric-field-z"]
    # Mixed quantum/classical molecular dynamics
    _STR_SINGLETON_PARAMS += ["QMMM", "QMMMscheme", "QMmethod", "QMbasis",
                              "SH"]
    _INT_SINGLETON_PARAMS += ["QMcharge", "QMmult", "CASorbitals",
                              "CASelectrons"]
    # Implicit solvent
    _STR_SINGLETON_PARAMS += ["implicit-solvent", "gb-algorithm",
                              "sa-algorithm"]
    _INT_SINGLETON_PARAMS += ["nstgbradii"]
    _FLOAT_SINGLETON_PARAMS += ["rgbradii", "gb-epsilon-solvent",
                                "gb-saltconc", "gb-obc-alpha", "gb-obc-beta",
                                "gb-obc-gamma", "gb-dielectric-offset",
                                "sa-surface-tension"]
    # Computational Electrophysiology
    # Note: we assume a maximum of 10 controlled ion types
    _STR_SINGLETON_PARAMS += (["swapcoords", "split-group0", "split-group1",
                               "massw-split0", "massw-split1", "solvent-group"]
                              + [f"iontype{n}-name" for n in range(10)]
                              )
    _INT_SINGLETON_PARAMS += (["swap-frequency", "coupl-steps", "iontypes",
                               "threshold"]
                              + [f"iontype{n}-in-A" for n in range(10)]
                              + [f"iontype{n}-in-B" for n in range(10)]
                              )
    _FLOAT_SINGLETON_PARAMS += ["bulk-offsetA", "bulk-offsetB", "cyl0-r",
                                "cyl0-up", "cyl0-down", "cyl1-r", "cyl1-up",
                                "cyl1-down"]
    # User defined thingies
    _INT_SINGLETON_PARAMS += [f"userint{n}" for n in range(1, 5)]
    _FLOAT_SINGLETON_PARAMS += [f"userreal{n}" for n in range(1, 5)]

    def _parse_line(self, line):
        # NOTE: we need to do this so complicated, because gmx accepts
        #       "key=value" i.e. without spaces, so we can not use shlex.shlex
        #       for separating the key and value reliably although this will be
        #        a corner case as most mdp files have "key = value" with spaces
        # split only at first equal sign
        splits_at_equal = line.split("=", maxsplit=1)
        # split at first comment sign
        splits_at_comment = line.split(";", maxsplit=1)
        # now we have multiple options:
        # 1. split only at '=' and not at ';' -> key=value line without comment
        # 2. split only at ';' and not at '=' -> (probably) a comment line,
        #                                        at least if the comment is the
        #                                        first char, otherwise we need
        #                                        an equal sign to be valid (?)
        # 3. split at ';' and at '=' -> need to find out if the comment is
        #                               before or after the equal sign
        # 4. no splits at '=' and no splits at ';' -> weired line, probably
        #                                             not a valid line(?)
        if splits_at_comment[0] == "":
            # option 2 (and 3 if the comment is before the equal sign)
            # comment sign is the first letter, so the whole line is
            # (most probably) a comment line
            logger.debug(f"mdp line parsed as comment: {line}")
            return {}
        if ((len(splits_at_equal) == 2 and len(splits_at_comment) == 1)  # option 1
            # or option 3 with equal sign before comment sign
            or ((len(splits_at_equal) == 2 and len(splits_at_comment) == 2)
                and (len(splits_at_comment[0]) > len(splits_at_equal[0])))):
            key = splits_at_equal[0].strip()  # strip of the white space
            # make sure the key is a single word, i.e. contains no spaces
            # if it is not we will raise the error below
            if key.split()[0] == key:
                value_unparsed = splits_at_equal[1]
                parser = shlex.shlex(value_unparsed,
                                     posix=True, punctuation_chars=True)
                parser.commenters = ";"
                # puncutation_chars=True adds "~-./*?=" to wordchars
                # such that we do not split floats and file paths and similar
                tokens = list(parser)
                # gromacs mdp can have 0-N tokens/values to the RHS of the '='
                if len(tokens) == 0:
                    # line with empty options, e.g. 'define = '
                    return {self._key_char_replace(key): []}
                # lines with content, we always return a list (and let our
                #  type_dispatch sort out the singleton options and the typing)
                return {self._key_char_replace(key): tokens}
        # if we end up here we did not know how to parse properly, e.g.
        # option 4 and option 3 with comment before equal but not at the
        # first position of the line (i.e. not a full comment line)
        # so no idea what happend here: best to let the user have a look :)
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
