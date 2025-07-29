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
This module contains the implementation of the gromacs engine classes.

The two classes GmxEngine and SlurmGmxEngine share most of their methods, the
slurm-enabled superclass only overrides a few methods to submit gmx mdrun via slurm.
"""
import asyncio
import copy
import dataclasses
import logging
import os
import random
import shlex
import shutil
import string
import typing

import aiofiles
import aiofiles.os
import aiofiles.ospath

from .. import slurm
from .._config import _OPT_SEMAPHORES, _SEMAPHORES, _OPT_SEMAPHORES_KEYS, _SEMAPHORES_KEYS
from ..mdengine import EngineCrashedError, MDEngine
from ..tools import (
    ensure_executable_available,
    attach_kwargs_to_object as _attach_kwargs_to_object,
    DescriptorWithDefaultOnInstanceAndClass as _DescriptorWithDefaultOnInstanceAndClass,
    DescriptorOutputTrajType as _DescriptorOutputTrajType,
)
from ..trajectory.trajectory import Trajectory
from .mdconfig import MDP
from .utils import get_all_traj_parts, nstout_from_mdp


if typing.TYPE_CHECKING:  # pragma: no cover
    from asyncio.subprocess import Process


logger = logging.getLogger(__name__)


# pylint: disable-next=too-few-public-methods
class _DescriptorCheckExecutable(_DescriptorWithDefaultOnInstanceAndClass):
    """
    Check if the given value is a valid (gmx) executable when setting it.

    We use this to make sure gmx grompp and gmx mdrun are available as set.
    It therefore is specifically tailored towards them and uses only the first
    part of the executable until the first space.
    """
    def __set__(self, obj, val: str) -> None:
        # split because mdrun and grompp can be both subcommands of gmx
        test_exe = val.split(" ")[0]
        ensure_executable_available(test_exe)
        super().__set__(obj, val)


# pylint: disable-next=too-few-public-methods
class _DescriptorOutputTrajTypeGmx(_DescriptorOutputTrajType):
    # only need to set allowed values to work, default type is set via
    # engine._output_traj_type
    ALLOWED_VALUES = {"trr", "xtc"}


# pylint: disable-next=too-few-public-methods
class _DescriptorMdrunTimeConversionFactor(_DescriptorWithDefaultOnInstanceAndClass):
    """
    Check that the given time conversion factor is 0 < factor <= 1 when setting.
    """
    def __set__(self, obj, val: float) -> None:
        if val > 1.:
            raise ValueError("`mdrun_time_conversion_factor` must be <= 1.")
        if val <= 0:
            raise ValueError("`mdrun_time_conversion_factor` must be > 0.")
        super().__set__(obj, val)


@dataclasses.dataclass
class _GmxInputFiles:
    """
    Dataclass to bundle/store all info related to the input files gromacs needs/gets.
    """
    mdp: MDP
    gro_file: str
    top_file: str
    ndx_file: str | None = None
    tpr_file: str | None = None


@dataclasses.dataclass
class _GmxEngineState:
    """
    Dataclass to bundle/store all engine-state related data.

    This includes, e.g., the total number of integration steps or the total
    integration time the GmxEngine this is attached to has performed.
    """
    frames_done: int = 0
    steps_done: int = 0
    time_done: float = 0.
    simulation_part: int = 0
    workdir: str = "."
    deffnm: str | None = None


class GmxEngine(MDEngine):
    # The way we (re)set our descriptor attributes in __init__ throws off pylints counting
    # pylint: disable=too-many-instance-attributes
    # and this class just has a lot of properties (which all count as public methods)
    # pylint: disable=too-many-public-methods
    """
    Steer gromacs molecular dynamics simulation from python.

    An async/await enabled wrapper around gromacs grompp and gromacs mdrun.
    Please use the power of concurrent execution of computationally bound
    subprocesses responsibly...or crash your workstation ;)
    The :class:`SlurmGmxEngine` alleviates this problem somewhat by submitting the
    (computationally expensive) mdruns via SLURM...in that case please have in
    mind that your colleagues might also want to use the cluster, and also that
    someone might have set a job/submission limit :)

    Attributes
    ----------
    grompp_executable : str
        Name or path to the grompp executable, by default "gmx grompp".
    mdrun_executable : str
        Name or path to the mdrun executable, by default "gmx mdrun".
    grompp_extra_args : str
        Can be used to pass extra command line arguments to grompp calls,
        e.g. "-maxwarn 1".
        Will simply be appended to the end of the command after a separating space.
    mdrun_extra_args : str
        Can be used to pass extra command line arguments to mdrun calls,
        e.g. "-ntomp 8".
        Will simply be appended to the end of the command after a separating space.
    output_traj_type : str
        Sets the trajectory type (ending) this engine returns/looks for.
        Note that we simply ignore all other trajectories, i.e. depending on
        the MDP settings we will still write xtc and trr, but return only the
        trajectories with matching ending.
    mdrun_time_conversion_factor : float
        When running gmx mdrun with a given `time_limit`, run it for
        `mdrun_time_conversion_factor * time_limit`.
        This option is relevant only for the :class:`SlurmGmxEngine` and here
        ensures that gmx mdrun finishes during the slurm time limit (which will
        be set to `time_limit`).
        The default value for the :class:`SlurmGmxEngine` is 0.99.
    """

    _grompp_executable = "gmx grompp"
    grompp_executable = _DescriptorCheckExecutable()
    _mdrun_executable = "gmx mdrun"
    mdrun_executable = _DescriptorCheckExecutable()
    # extra_args are expected to be str and will be appended to the end of the respective command
    grompp_extra_args = ""
    mdrun_extra_args = ""
    # file ending of the returned output trajectories, exposed as output_traj_type
    _output_traj_type = "xtc"
    output_traj_type = _DescriptorOutputTrajTypeGmx()
    # See the notes below for the SlurmGmxEngine on why this conversion factor
    # is needed (there), here we have it only for consistency
    _mdrun_time_conversion_factor = 1.  # run mdrun for 1. * time-limit
    mdrun_time_conversion_factor = _DescriptorMdrunTimeConversionFactor()

    def __init__(self,
                 mdconfig: MDP,
                 gro_file: str,
                 top_file: str, *,
                 ndx_file: str | None = None,
                 **kwargs) -> None:
        """
        Initialize a :class:`GmxEngine`.

        Note that all attributes can be set at initialization by passing keyword
        arguments with their name, e.g. ``mdrun_extra_args="-ntomp 2"`` to
        instruct gromacs to use 2 openMP threads.

        Parameters
        ----------
        mdconfig: MDP
            The molecular dynamics parameters.
        gro_file: str
            Absolute or relative path to a gromacs structure file.
        top_file: str
            Absolute or relative path to a gromacs topology (.top) file.
        ndx_file: str or None
            Optional, absolute or relative path to a gromacs index file.
        """
        # make it possible to set any attribute via kwargs, check them when setting
        _attach_kwargs_to_object(obj=self, logger=logger, **kwargs)
        # give it only the required arguments, we reset below anyway using the
        # properties to use the checks implemented in them
        self._input_files = _GmxInputFiles(mdp=mdconfig,
                                           gro_file=gro_file,
                                           top_file=top_file,
                                           )
        self._engine_state = _GmxEngineState()
        # TODO: store a hash/the file contents for gro, top, ndx to check against
        #       when we load from storage/restart? if we do this, do it in the property!
        self.gro_file = gro_file
        self.top_file = top_file
        self.ndx_file = ndx_file
        # basic checks for mdp are done in the property-setter, e.g. if the
        # output_traj_type is actually written with current mdp-settings
        self.mdp = mdconfig
        # also (re)-set our descriptors to trigger __set__ and make sure that
        # also our (class) defaults are available + executable
        self.mdrun_executable = self.mdrun_executable
        self.grompp_executable = self.grompp_executable

    @property
    def current_trajectory(self) -> Trajectory | None:
        """
        Return the last finished trajectory (part).

        Returns
        -------
        Trajectory
            Last complete trajectory produced by this engine.
        """
        if (
            self.tpr_file is not None
            and self.deffnm is not None
            and self.simulation_part > 0
        ):
            # tpr_file and deffnm are set in prepare, i.e. having them
            # set makes sure that we have at least prepared running the traj
            # but it might not be done yet
            # also check if we ever started a run, i.e. if there might be a
            # trajectory to return. If simulation_part == 0 we never executed a
            # run method (where it is increased) and also did not (re)start a run
            traj = Trajectory(
                trajectory_files=os.path.join(
                    # prepend engine workdir to make traj file paths relative to python workdir
                    self.workdir,
                    (f"{self.deffnm}"
                     f"{self._num_suffix(self.simulation_part)}"
                     f".{self.output_traj_type}"
                     ),
                ),
                # NOTE: tpr_file is already relative to the workdir of the python interpreter
                structure_file=self.tpr_file,
                nstout=self.nstout,
            )
            return traj
        return None

    @property
    def workdir(self) -> str:
        """The current working directory of the engine."""
        return self._engine_state.workdir

    @workdir.setter
    def workdir(self, value: str) -> None:
        if not os.path.isdir(value):
            raise TypeError(f"Not a directory ({value}).")
        value = os.path.relpath(value)
        self._engine_state.workdir = value

    @property
    def gro_file(self) -> str:
        """The (path to the) gro file this engine uses/used to call grompp."""
        return self._input_files.gro_file

    @gro_file.setter
    def gro_file(self, val: str) -> None:
        if not os.path.isfile(val):
            raise FileNotFoundError(f"gro file not found: {val}")
        val = os.path.relpath(val)
        self._input_files.gro_file = val

    @property
    def top_file(self) -> str:
        """The (path to the) top file this engine uses/used to call grompp."""
        return self._input_files.top_file

    @top_file.setter
    def top_file(self, val: str) -> None:
        if not os.path.isfile(val):
            raise FileNotFoundError(f"top file not found: {val}")
        val = os.path.relpath(val)
        self._input_files.top_file = val

    @property
    def ndx_file(self) -> str | None:
        """The (path to the) ndx file this engine uses/used to call grompp."""
        return self._input_files.ndx_file

    @ndx_file.setter
    def ndx_file(self, val: str | None) -> None:
        if val is not None:
            if not os.path.isfile(val):
                raise FileNotFoundError(f"ndx file not found: {val}")
            val = os.path.relpath(val)
        # GMX does not require an ndx file, so we accept None
        self._input_files.ndx_file = val

    # NOTE: This does not have a setter on purpose, only prepare methods must
    #       set this (and there we can be bothered to access via _input_files)
    @property
    def tpr_file(self) -> str | None:
        """
        The (path to the) tpr file this engine uses to call gmx mdrun.

        None before a call to any prepare method.
        """
        return self._input_files.tpr_file

    @property
    def mdp(self) -> MDP:
        """The configuration of this engine as a :class:`MDP` object."""
        return self._input_files.mdp

    @mdp.setter
    def mdp(self, val: MDP) -> None:
        if not isinstance(val, MDP):
            raise TypeError(f"Value must be of type {MDP}.")
        try:
            nsteps = val["nsteps"]
        except KeyError:
            # nsteps not defined
            logger.info("Setting previously undefined nsteps to -1 (infinite).")
        else:
            if nsteps != -1:
                logger.info("Changing nsteps from %s to -1 (infinite), the run "
                            "length is controlled via arguments of the run "
                            "method.", nsteps)
        finally:
            val["nsteps"] = -1
        # check that we get a trajectory of the format we expect with our
        # current mdp, we do this by using nstout_from_mdp since it throws a
        # nice error if the mdp does not generate output for given traj-format
        _ = nstout_from_mdp(mdp=val, traj_type=self.output_traj_type)
        # check if we do an energy minimization: in this case gromacs writes no
        # compressed trajectory (even if so requested by the mdp-file), so we
        # check that self.output_traj_type == trr and generate an error if not
        try:
            integrator = val["integrator"]
        except KeyError:
            # integrator not defined, although this probably seldomly happens,
            # gmx grompp does use the (implicit) default "integrator=md" in
            # that case
            integrator = "md"
        if any(integrator == em_algo for em_algo in ("steep", "cg", "l-bfgs")):
            if not self.output_traj_type.lower() == "trr":
                raise ValueError("Gromacs only writes full precision (trr) "
                                 "trajectories when performing an energy "
                                 "minimization.")
        self._input_files.mdp = val

    # alias for mdp to mdconfig (since some users may expect mdconfig)
    mdconfig = mdp

    # NOTE: This does not have a setter on purpose, only prepare methods must
    #       set this (and there we can be bothered to access via _input_files)
    @property
    def deffnm(self) -> str | None:
        """The ``deffnm`` this engine uses. None before a call to any prepare method."""
        return self._engine_state.deffnm

    # NOTE: This does not have a setter on purpose, only prepare and run methods
    #       must set this (and there we can be bothered to access via _input_files)
    @property
    def simulation_part(self) -> int:
        """Return the current ``simulation_part`` number."""
        return self._engine_state.simulation_part

    @property
    def dt(self) -> float:
        """Integration timestep in ps."""
        return self.mdp["dt"]

    @property
    def time_done(self) -> float:
        """
        Integration time since last call to prepare in ps.

        Takes into account 'tinit' from the .mdp file if set.
        """
        try:
            tinit = self.mdp["tinit"]
        except KeyError:
            tinit = 0.
        return self._engine_state.time_done - tinit

    @property
    def nstout(self) -> int:
        """Smallest output frequency for current output_traj_type."""
        return nstout_from_mdp(self.mdp,
                               traj_type=self.output_traj_type)

    @property
    def steps_done(self) -> int:
        """
        Number of integration steps done since last call to :meth:`prepare`.

        NOTE: steps != frames * nstout
        Some remarks on the relation between frames_done and steps_done:
        Usually (when passing ``nsteps`` to ``run()``) frames_done will be equal to
        steps_done/nstout + 1 because the initial/final configuration will be
        written twice (since then the first/last step is always an output step)
        However as soon as we run for a specific walltime (without specifying
        `nsteps`) stuff gets complicated, then gromacs can potentially stop at
        every neighbor search step (where it also can/will write a checkpoint).
        If that step is not a trajectory output step, no output will be written
        to the traj and then the plus 1 rule for the double written
        initial/final configuration is off (since it will then be a 'normal'
        configuration written just once).
        If however the neighbor search and trajectory output fall together on
        the same step the configuration will be written twice (as with `nsteps`
        specified).
        """
        return self._engine_state.steps_done

    @property
    def frames_done(self) -> int:
        """
        Number of frames produced since last call to :meth:`prepare`.

        NOTE: frames != steps / nstout
        See the steps_done docstring for more.
        """
        return self._engine_state.frames_done

    async def apply_constraints(self, conf_in: Trajectory, conf_out_name: str, *,
                                wdir: str = ".") -> Trajectory:
        """
        Apply constraints to given configuration.

        Parameters
        ----------
        conf_in : Trajectory
            A (one-frame) trajectory, only the first frame will be used.
        conf_out_name : str
            Output path for the constrained configuration.
        wdir : str, optional
            Working directory for the constraint engine, by default ".",
            a subfolder with random name will be created.

        Returns
        -------
        Trajectory
            The constrained configuration.
        """
        return await self._0step_md(conf_in=conf_in,
                                    conf_out_name=conf_out_name,
                                    wdir=wdir,
                                    constraints=True,
                                    generate_velocities=False,
                                    )

    async def generate_velocities(self, conf_in: Trajectory, conf_out_name: str, *,
                                  wdir: str = ".", constraints: bool = True,
                                  ) -> Trajectory:
        """
        Generate random Maxwell-Boltzmann velocities for given configuration.

        Parameters
        ----------
        conf_in : Trajectory
            A (one-frame) trajectory, only the first frame will be used.
        conf_out_name : str
            Output path for the velocity randomized configuration.
        wdir : str, optional
            Working directory for the constraint engine, by default ".",
            a subfolder with random name will be created.
        constraints : bool, optional
            Whether to also apply constraints, by default True.

        Returns
        -------
        Trajectory
            The configuration with random velocities and optionally constraints
            enforced.
        """
        return await self._0step_md(conf_in=conf_in,
                                    conf_out_name=conf_out_name,
                                    wdir=wdir,
                                    constraints=constraints,
                                    generate_velocities=True,
                                    )

    async def _0step_md(self, conf_in: Trajectory, conf_out_name: str, *,
                        wdir: str, constraints: bool, generate_velocities: bool,
                        ) -> Trajectory:
        if not os.path.isabs(conf_out_name):
            # assume conf_out is to be meant relative to wdir if not an abspath
            conf_out_name = os.path.join(wdir, conf_out_name)
        # work in a subdirectory of wdir to make deleting easy
        # generate its name at random to make sure we can use multiple
        # engines with 0stepMDruns in the same wdir
        run_name = "".join(random.choices((string.ascii_letters
                                           + string.ascii_lowercase
                                           + string.ascii_uppercase),
                                          k=6,
                                          )
                           )
        swdir = os.path.join(wdir, run_name)
        await aiofiles.os.mkdir(swdir)
        constraints_mdp = copy.deepcopy(self.mdp)
        constraints_mdp["continuation"] = "no" if constraints else "yes"
        constraints_mdp["gen-vel"] = "yes" if generate_velocities else "no"
        # make sure we write a trr and a xtc to read the final configuration
        # (this way we dont have to check what ending conf_out_name has)
        constraints_mdp["nstxout"] = 1
        constraints_mdp["nstvout"] = 1
        constraints_mdp["nstfout"] = 1
        constraints_mdp["nstxout-compressed"] = 1
        if generate_velocities:
            # make sure we have draw a new/different random number for gen-vel
            constraints_mdp["gen-seed"] = -1
        constraints_mdp["nsteps"] = 0
        await self._run_grompp(workdir=swdir, deffnm=run_name,
                               trr_in=conf_in.trajectory_files[0],
                               tpr_out=os.path.join(swdir, f"{run_name}.tpr"),
                               mdp_obj=constraints_mdp)
        cmd_str = self._mdrun_cmd(tpr=os.path.join(swdir, f"{run_name}.tpr"),
                                  workdir=swdir,
                                  deffnm=run_name)
        logger.debug("About to execute gmx mdrun command for constraints and"
                     "/or velocity generation: %s",
                     cmd_str)
        stderr = bytes()
        stdout = bytes()
        await self._acquire_resources_gmx_mdrun()
        mdrun_proc = await self._start_gmx_mdrun(
                        cmd_str=cmd_str, workdir=swdir,
                        run_name=run_name,
                        # TODO: we hardcode that the 0step MD runs can not be longer than 15 min
                        # (but i think this should be fine for randomizing velocities and/or
                        #  applying constraints?!)
                        walltime=0.25,
                        )
        try:
            stdout, stderr = await mdrun_proc.communicate()
        except asyncio.CancelledError:
            mdrun_proc.kill()
            raise  # reraise the error for encompassing coroutines
        else:
            if (returncode := mdrun_proc.returncode):
                raise EngineCrashedError(
                    f"Non-zero (or no) exit code from mdrun (= {returncode}).\n"
                    + "\n--------\n"
                    + f"stderr: \n--------\n {stderr.decode()}"
                    + "\n--------\n"
                    + f"stdout: \n--------\n {stdout.decode()}"
                                         )
            # just get the output trajectory, it is only one configuration
            shutil.move(os.path.join(swdir, (f"{run_name}{self._num_suffix(1)}"
                                             + f".{conf_out_name.split('.')[-1]}")
                                     ),
                        conf_out_name)
            shutil.rmtree(swdir)  # remove the whole directory we used as wdir
            return Trajectory(
                              trajectory_files=conf_out_name,
                              # structure file of the conf_in because we
                              # delete the other one with the folder
                              structure_file=conf_in.structure_file,
                              nstout=1,
                              )
        finally:
            await self._cleanup_gmx_mdrun(workdir=swdir, run_name=run_name)

    async def prepare(self, starting_configuration: Trajectory | None | str,
                      workdir: str, deffnm: str) -> None:
        """
        Prepare a fresh simulation (starting with part0001).

        Can also be used to continue a simulation from a checkpoint file with
        matching name ('deffnm.cpt'). In that case, the 'simulation-part' mdp
        option must match the number of the next part to be generated, e.g. it
        must be 2 if the last part generated was part0001. The previously
        generated trajectory files do not need to exist.
        If 'simulation-part' is not set and previous trajectories are found an
        error is raised.

        Parameters
        ----------
        starting_configuration : Trajectory or None or str
            A (trr) trajectory of which we take the first frame as starting
            configuration (including velocities) or None, then the initial
            configuration is the gro-file.
            Can also be a str, then it is assumed to be the path to a trr, cpt,
            or tng (i.e. a full precision trajectory) and will be passed directly
            to grompp.
        workdir : str
            Absolute or relative path to an existing directory to use as
            working directory.
        deffnm : str
            The name (prefix) to use for all files.
        """
        # deffnm is the default name/prefix for all outfiles (as in gmx)
        self._engine_state.deffnm = deffnm
        self.workdir = workdir  # sets to relpath and check if it is a dir
        # check 'simulation-part' option in mdp file / MDP options
        # it decides at which .partXXXX the gmx numbering starts,
        # however gromacs ignores it if there is no -cpi [CheckPointIn]
        # so we do the same, i.e. we warn if we detect it is set
        # and check if there is a checkpoint with the right name [deffnm.cpt]
        # if yes we set our internal simulation_part counter to the value from
        # the mdp - 1 (we increase *before* each simulation part)
        cpt_fname = os.path.join(self.workdir, f"{deffnm}.cpt")
        try:
            sim_part = self.mdp["simulation-part"]
        except KeyError:
            # the gmx mdp default is 1, it starts at part0001
            # we add one at the start of each run, i.e. the numberings match up
            # and we will have tra=`...part0001.trr` from gmx
            # and confout=`...part0001.gro` from our naming
            self._engine_state.simulation_part = 0
        else:
            if sim_part > 1:
                if not os.path.isfile(cpt_fname):
                    raise ValueError("'simulation-part' > 1 is only possible "
                                     + "if starting from a checkpoint, but "
                                     + f"{cpt_fname} does not exist."
                                     )
                starting_configuration = cpt_fname
                logger.warning("Starting value for 'simulation-part' > 1 (=%s) "
                               "and existing checkpoint file found (%s). "
                               "Using the checkpoint file as "
                               "`starting_configuration`.",
                               sim_part, cpt_fname)
            # always subtract one from sim_part so we get 0 if it was 1
            self._engine_state.simulation_part = sim_part - 1
        # check for previous runs with the same deffnm in workdir
        # NOTE: we only check for checkpoint files and trajectory parts as gmx
        #       will move everything and only the checkpoint and trajs let us
        #       trip and get the part numbering wrong
        trajs_with_same_deffnm = await get_all_traj_parts(
                                            folder=self.workdir,
                                            deffnm=deffnm,
                                            traj_type=self.output_traj_type,
                                                          )
        # NOTE: it is enough to check if we have more trajectories than the
        #       starting simulation_part, because we assume that if we find a
        #       checkpoint file (above) and simulation_part > 0 that the
        #       checkpoint file matches the correct part-number
        if len(trajs_with_same_deffnm) > self.simulation_part:
            raise ValueError(f"There are files in workdir ({self.workdir}) "
                             + f"with the same deffnm ({deffnm}). Use the "
                             + "``prepare_from_files()`` method to continue an "
                             + "existing MD run or change the workdir and/or "
                             + "deffnm.")
        # actual preparation of MD run: sort out starting configuration...
        if (
            # None enables start from the initial structure file ('-c' option)
            starting_configuration is None
            # str enables passing the path to the full precision trajectory
            # directly, i.e. trr, cpt, or tng
            or isinstance(starting_configuration, str)
        ):
            trr_in = starting_configuration
        elif isinstance(starting_configuration, Trajectory):
            # enable passing of asyncmd.Trajectories as starting_configuration
            trr_in = starting_configuration.trajectory_files[0]
        else:
            raise TypeError("Starting_configuration must be None, a wrapped "
                            "full precision trajectory, or the path to a "
                            "full precision trajectory (trr, cpt, or tng).")
        # ...and call grompp to get a tpr
        # remember the path to use as structure file for out trajs
        self._input_files.tpr_file = os.path.join(self.workdir, deffnm + ".tpr")
        await self._run_grompp(workdir=self.workdir, deffnm=self.deffnm,
                               trr_in=trr_in, tpr_out=self.tpr_file,
                               mdp_obj=self.mdp)
        if not await aiofiles.ospath.isfile(self.tpr_file):
            # better be save than sorry :)
            raise RuntimeError("Something went wrong generating the tpr. "
                               f"{self.tpr_file} does not seem to be a file.")
        self._engine_state.frames_done = 0  # (re-)set how many frames we did
        self._engine_state.steps_done = 0
        self._engine_state.time_done = 0.

    async def _run_grompp(self, *, workdir: str, deffnm: str, trr_in: str | None,
                          tpr_out: str, mdp_obj: MDP) -> None:
        # NOTE: file paths from workdir and deffnm
        mdp_in = os.path.join(workdir, deffnm + ".mdp")
        # write the mdp file (always overwriting existing mdps)
        # I (hejung) think this is what we want as the prepare methods check
        # for leftover files with the same deffnm, so if only the mdp is there
        # we can (and want to) just overwrite it without raising an err
        async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]:
            mdp_obj.write(mdp_in, overwrite=True)
        mdp_out = os.path.join(workdir, deffnm + "_mdout.mdp")
        cmd_str = self._grompp_cmd(mdp_in=mdp_in, tpr_out=tpr_out,
                                   workdir=workdir,
                                   trr_in=trr_in, mdp_out=mdp_out)
        logger.debug("About to execute gmx grompp command: %s", cmd_str)
        # 3 file descriptors: stdin, stdout, stderr
        # NOTE: The max open files semaphores counts for 3 open files, so we
        #       only need it once
        await _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN].acquire()
        grompp_proc = await asyncio.create_subprocess_exec(
                                                *shlex.split(cmd_str),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=workdir,
                                                )
        try:
            stdout, stderr = await grompp_proc.communicate()
        except asyncio.CancelledError as e:
            grompp_proc.kill()  # kill grompp
            raise e from None  # and reraise the cancellation
        else:
            return_code = grompp_proc.returncode
            if (return_code := grompp_proc.returncode):
                # this assumes POSIX
                raise RuntimeError("grompp had non-zero return code "
                                   + f"({return_code}).\n"
                                   + "\n--------\n"
                                   + f"stderr: \n--------\n {stderr.decode()}"
                                   + "\n--------\n"
                                   + f"stdout: \n--------\n {stdout.decode()}"
                                   )
            logger.debug("gmx grompp command returned return code %s.",
                         str(return_code) if return_code is not None else "not available")
        finally:
            # release the semaphore
            _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN].release()

    async def prepare_from_files(self, workdir: str, deffnm: str) -> None:
        """
        Prepare continuation run starting from the last part found in workdir.

        Expects the checkpoint file and last trajectory part to exist, will
        (probably) fail otherwise.

        Parameters
        ----------
        workdir : str
            Absolute or relative path to an existing directory to use as
            working directory.
        deffnm : str
            The name (prefix) to use for all files.
        """
        self.workdir = workdir
        previous_trajs = await get_all_traj_parts(self.workdir, deffnm=deffnm,
                                                  traj_type=self.output_traj_type,
                                                  )
        last_trajname = os.path.split(previous_trajs[-1].trajectory_files[0])[-1]
        last_partnum = int(last_trajname[len(deffnm) + 5:len(deffnm) + 9])
        if last_partnum != len(previous_trajs):
            logger.warning("Not all previous trajectory parts seem to be "
                           "present in the current workdir. Assuming the "
                           "highest part number corresponds to the "
                           "checkpoint file and continuing anyway."
                           )
        # load the 'old' mdp_in
        async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]:
            self.mdp = MDP(os.path.join(self.workdir, f"{deffnm}.mdp"))
        self._engine_state.deffnm = deffnm
        # Note that we dont need to explicitly check for the tpr existing,
        # if it does not exist we will err when getting the traj lengths
        self._input_files.tpr_file = os.path.join(self.workdir, deffnm + ".tpr")
        self._engine_state.simulation_part = last_partnum
        # len(t), because for frames we do not care if first frame is in traj
        self._engine_state.frames_done = sum(len(t) for t in previous_trajs)
        # steps done is the more reliable info if we want to know how many
        # integration steps we did
        self._engine_state.steps_done = previous_trajs[-1].last_step
        self._engine_state.time_done = previous_trajs[-1].last_time

    # NOTE: this enables us to reuse run and prepare methods in SlurmGmxEngine,
    # i.e. we only need to overwrite the next 3 functions to write out the slurm
    # submission script, submit the job and allocate/release different resources
    async def _start_gmx_mdrun(self, *, cmd_str: str, workdir: str,
                               # the next two arguments are only used by SlurmGmxEngine
                               # but we rather make them explicit here already
                               # pylint: disable-next=unused-argument
                               walltime: float | None,
                               # pylint: disable-next=unused-argument
                               run_name: str | None = None,
                               ) -> "Process | slurm.SlurmProcess":
        return await asyncio.create_subprocess_exec(
                                            *shlex.split(cmd_str),
                                            stdout=asyncio.subprocess.PIPE,
                                            stderr=asyncio.subprocess.PIPE,
                                            cwd=workdir,
                                                    )

    async def _acquire_resources_gmx_mdrun(self) -> None:
        # *always* called before any gmx_mdrun, used to reserve resources
        # for local gmx we need 3 file descriptors: stdin, stdout, stderr
        # (one max files semaphore counts for 3 open files)
        await _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN].acquire()

    async def _cleanup_gmx_mdrun(self,
                                 # the next two arguments are only used by SlurmGmxEngine
                                 # but we rather make them explicit here already
                                 # pylint: disable-next=unused-argument
                                 workdir: str, run_name: str | None = None,
                                 ) -> None:
        # *always* called after any gmx_mdrun, use to release resources
        # release the semaphore for the 3 file descriptors
        _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN].release()

    async def run(self, nsteps: int | None = None, walltime: float | None = None,
                  steps_per_part: bool = False,
                  ) -> Trajectory | None:
        """
        Run simulation for specified number of steps or/and a given walltime.

        Note that you can pass both nsteps and walltime and the simulation will
        stop on the condition that is reached first.

        Return None if no integration is needed because nsteps integration steps
        have already been performed.

        Parameters
        ----------
        nsteps : int or None
            Integration steps to run for either in total [as measured since the
            last call to `self.prepare()`] or in the newly generated trajectory
            part, see also the steps_per_part argument.
        walltime : float or None
            (Maximum) walltime in hours, `None` means unlimited.
        steps_per_part : bool
            If True nsteps are the steps to do in the new trajectory part, else
            the total number of steps since the last call to `prepare()` are
            counted, default False.
        """
        # generic run method is actually easier to implement for gmx :D
        if self.tpr_file is None or self.deffnm is None:
            raise RuntimeError(
                "Engine not ready for run. Call self.prepare() before calling a run method."
                )
        if all(kwarg is None for kwarg in (nsteps, walltime)):
            raise ValueError("Neither steps nor walltime given.")
        if nsteps is not None:
            nsteps = int(nsteps)
            if nsteps % self.nstout:
                raise ValueError(f"nsteps ({nsteps}) must be a multiple of "
                                 + f"nstout ({self.nstout}).")
            if not steps_per_part:
                nsteps -= self.steps_done
            if not nsteps:
                # Return None instead of raising an error, this makes it nicer
                # to use the run method with walltime and total nsteps inside
                # while loops, i.e. we can just call traj = e.run(...) and then
                # while traj is not None: traj = e.run()
                return None
            if nsteps < 0:
                raise ValueError(f"nsteps is too small ({nsteps} steps for this part). "
                                 "Can not travel backwards in time...")

        self._engine_state.simulation_part += 1
        cmd_str = self._mdrun_cmd(tpr=self.tpr_file, workdir=self.workdir,
                                  deffnm=self.deffnm,
                                  maxh=walltime, nsteps=nsteps)
        logger.debug("About to execute gmx mdrun command: %s", cmd_str)
        returncode = None
        stderr = bytes()
        stdout = bytes()
        await self._acquire_resources_gmx_mdrun()
        mdrun_proc = await self._start_gmx_mdrun(cmd_str=cmd_str, workdir=self.workdir,
                                                 walltime=walltime,)
        try:
            stdout, stderr = await mdrun_proc.communicate()
        except asyncio.CancelledError as e:
            mdrun_proc.kill()
            raise e from None  # reraise the error for encompassing coroutines
        else:
            if (returncode := mdrun_proc.returncode):
                raise EngineCrashedError(
                    f"Non-zero (or no) exit code from mdrun (= {returncode}).\n"
                    + "\n--------\n"
                    + f"stderr: \n--------\n {stderr.decode()}"
                    + "\n--------\n"
                    + f"stdout: \n--------\n {stdout.decode()}"
                                         )
            logger.debug("gmx mdrun command returned return code %s.",
                         str(returncode) if returncode is not None else "not available")
            self._engine_state.frames_done += len(self.current_trajectory)
            # dont care if we did a little more and only the checkpoint knows
            # we will only find out with the next trajectory part anyways
            self._engine_state.steps_done = self.current_trajectory.last_step
            self._engine_state.time_done = self.current_trajectory.last_time
            return self.current_trajectory
        finally:
            await self._cleanup_gmx_mdrun(workdir=self.workdir)

    async def run_steps(self, nsteps: int, steps_per_part: bool = False
                        ) -> Trajectory | None:
        """
        Run simulation for specified number of steps.

        Return None if no integration is needed because nsteps integration steps
        have already been performed.

        Parameters
        ----------
        nsteps : int or None
            Integration steps to run for either in total [as measured since the
            last call to `self.prepare()`] or in the newly generated trajectory
            part, see also the steps_per_part argument.
        steps_per_part : bool
            If True nsteps are the steps to do in the new trajectory part, else
            the total number of steps since the last call to `prepare()` are
            counted, default False.
        """
        return await self.run(nsteps=nsteps, steps_per_part=steps_per_part)

    async def run_walltime(self, walltime: float, max_steps: int | None = None,
                           ) -> Trajectory | None:
        """
        Run simulation for a given walltime.

        Return None if no integration is needed because max_steps integration
        steps have already been performed.

        Parameters
        ----------
        walltime : float or None
            (Maximum) walltime in hours.
        max_steps : int | None, optional
            If not None, terminate when max_steps integration steps are reached
            in total, also if this is before walltime is reached.
            By default None.
        """
        return await self.run(walltime=walltime, nsteps=max_steps,
                              steps_per_part=False)

    def _num_suffix(self, sim_part: int) -> str:
        # construct gromacs num part suffix from simulation_part
        num_suffix = f".part{sim_part:04d}"
        return num_suffix

    def _grompp_cmd(self, *, mdp_in: str, tpr_out: str, workdir: str,
                    trr_in: str | None = None, mdp_out: str | None = None,
                    ) -> str:
        # all args are expected to be file paths
        # make sure we use the right ones, i.e. relative to workdir of the engine
        # because they will be relative to workdir of the python interpreter
        mdp_in = os.path.relpath(mdp_in, start=workdir)
        tpr_out = os.path.relpath(tpr_out, start=workdir)
        gro_file = os.path.relpath(self.gro_file, start=workdir)
        top_file = os.path.relpath(self.top_file, start=workdir)
        cmd = f"{self.grompp_executable} -f {mdp_in} -c {gro_file}"
        cmd += f" -p {top_file}"
        if self.ndx_file is not None:
            ndx_file = os.path.relpath(self.ndx_file, start=workdir)
            cmd += f" -n {ndx_file}"
        if trr_in is not None:
            # input trr is optional
            # TODO /NOTE: currently we do not pass '-time', i.e. we just use the
            #            gmx default frame selection: last frame from trr
            trr_in = os.path.relpath(trr_in, start=workdir)
            cmd += f" -t {trr_in}"
        if mdp_out is None:
            # find out the name and dir of the tpr to put the mdp next to it
            head, tail = os.path.split(tpr_out)
            name = tail.split(".")[0]
            mdp_out = os.path.join(head, name + ".mdout.mdp")
        mdp_out = os.path.relpath(mdp_out, start=workdir)
        cmd += f" -o {tpr_out} -po {mdp_out}"
        if self.grompp_extra_args:
            # add extra args string if it is not empty
            cmd += f" {self.grompp_extra_args}"
        return cmd

    def _mdrun_cmd(self, *, tpr: str, workdir: str, deffnm: str | None = None,
                   maxh: float | None = None, nsteps: int | None = None,
                   ) -> str:
        # use "-noappend" to avoid appending to the trajectories when starting
        # from checkpoints, instead let gmx create new files with .partXXXX suffix
        tpr = os.path.relpath(tpr, start=workdir)
        if deffnm is None:
            # find out the name of the tpr and use that as deffnm
            _, tail = os.path.split(tpr)
            deffnm = tail.split(".")[0]
        # cmd = f"{self.mdrun_executable} -noappend -deffnm {deffnm} -cpi"
        # NOTE: the line above does the same as the four below before the if-clauses
        #       however gromacs -deffnm is deprecated (and buggy),
        #       so we just make our own 'deffnm', i.e. we name all files the same
        #       except for the ending but do so explicitly
        # TODO /FIXME: we dont specify the names for e.g. pull outputfiles,
        #             so they will have their default names and will collide
        #             when running multiple engines in the same folder!
        cmd = f"{self.mdrun_executable} -noappend -s {tpr}"
        # always add the -cpi option, this lets gmx figure out if it wants
        # to start from a checkpoint (if there is one with deffnm)
        # cpi (CheckPointIn) is ignored if not present,
        # cpo (CheckPointOut) is the name to use for the (final) checkpoint
        cmd += f" -cpi {deffnm}.cpt -cpo {deffnm}.cpt"
        cmd += f" -o {deffnm}.trr -x {deffnm}.xtc -c {deffnm}.confout.gro"
        cmd += f" -e {deffnm}.edr -g {deffnm}.log"
        if maxh is not None:
            maxh = self.mdrun_time_conversion_factor * maxh
            cmd += f" -maxh {maxh}"
        if nsteps is not None:
            cmd += f" -nsteps {nsteps}"
        if self.mdrun_extra_args:
            cmd += f" {self.mdrun_extra_args}"
        return cmd


class SlurmGmxEngine(GmxEngine):
    __doc__ = GmxEngine.__doc__
    # Use local prepare (i.e. grompp) of GmxEngine then submit run to slurm.
    # Take submit script as str/file, use pythons .format to insert stuff.
    # We overwrite the `GmxEngine._start_gmx_mdrun` to instead return a `SlurmProcess`,
    # which emulates the API of `asyncio.subprocess.Process` and can (for our
    # purposes) be used as a drop-in replacement. Therefore we only need to
    # reimplement `_start_gmx_mdrun()`, `_acquire_resources_gmx_mdrun()` and
    # `_cleanup_gmx_mdrun()` to have a working SlurmGmxEngine.
    # TODO: use SLURM also for grompp?! (would it make stuff faster?)
    #       I (hejung) think probably not by much because we already use
    #       asyncios subprocess for grompp (i.e. do it asynchronous) and grompp
    #       will most likely not take much resources on the login (local) node

    _mdrun_executable = "gmx_mpi mdrun"  # MPI as default for clusters
    _mdrun_time_conversion_factor = 0.99  # run mdrun for 0.99 * time-limit
    # NOTE: The rationale behind the (slightly) reduced mdrun time compared to
    #       the slurm job time-limit is that sometimes setting up and finishing
    #       up the slurm job takes some time (e.g. activating modules, sourcing
    #       environments, etc.) and this can result in jobs that are cancelled
    #       due to reaching the maximum time limit in slurm. This in turn means
    #       that we would believe the job failed because it got cancelled
    #       although the mdrun was successful.

    # pylint: disable-next=too-many-arguments
    def __init__(self, mdconfig: MDP, gro_file: str, top_file: str, *,
                 ndx_file: str | None = None,
                 sbatch_script: str,
                 sbatch_options: dict[str, str] | None = None,
                 **kwargs) -> None:
        """
        Initialize a :class:`SlurmGmxEngine`.

        Parameters
        ----------
        mdconfig : MDP
            The molecular dynamics parameters.
        gro_file: str
            Absolute or relative path to a gromacs structure file.
        top_file: str
            Absolute or relative path to a gromacs topology (.top) file.
        sbatch_script : str
            Absolute or relative path to a slurm sbatch script or a string with
            the content of the sbatch script. Note that the submission script
            must contain the following placeholders (see also the examples
            folder):

             - {mdrun_cmd} : Replaced by the command to run mdrun

        ndx_file: str or None
            Optional, absolute or relative path to a gromacs index file.
        sbatch_options : dict or None
            Dictionary of sbatch options, keys are long names for options,
            values are the corresponding values. The keys/long names are given
            without the dashes, e.g. to specify ``--mem=1024`` the dictionary
            needs to be ``{"mem": "1024"}``. To specify options without values
            use keys with empty strings as values, e.g. to specify
            ``--contiguous`` the dictionary needs to be ``{"contiguous": ""}``.
            See the SLURM documentation for a full list of sbatch options
            (https://slurm.schedmd.com/sbatch.html).
            Note: This argument is passed as is to the ``SlurmProcess`` in which
            the computation is performed. Each call to the engines `run` method
            triggers the creation of a new :class:`asyncmd.slurm.SlurmProcess`
            and will use the then current ``sbatch_options``.

        Note that all attributes can be set at initialization by passing keyword
        arguments with their name, e.g. mdrun_extra_args="-ntomp 2" to instruct
        gromacs to use 2 openMP threads.
        """
        super().__init__(mdconfig=mdconfig, gro_file=gro_file,
                         top_file=top_file, ndx_file=ndx_file, **kwargs)
        # we expect sbatch_script to be a str,
        # but it could be either the path to a submit script or the content of
        # the submission script directly
        # we decide what it is by checking for the shebang
        if not sbatch_script.startswith("#!"):
            # probably path to a file, lets try to read it
            with open(sbatch_script, 'r', encoding="locale") as f:
                sbatch_script = f.read()
        self.sbatch_script = sbatch_script
        self.sbatch_options = sbatch_options

    def _name_from_name_or_none(self, run_name: str | None) -> str:
        if run_name is not None:
            name = run_name
        else:
            # create a name from deffnm and partnum
            name = self.deffnm + self._num_suffix(sim_part=self.simulation_part)
        return name

    async def _start_gmx_mdrun(self, *, cmd_str: str, workdir: str,
                               walltime: float | None,
                               run_name: str | None = None,
                               ) -> slurm.SlurmProcess:
        name = self._name_from_name_or_none(run_name=run_name)
        # substitute placeholders in submit script
        script = self.sbatch_script.format(mdrun_cmd=cmd_str)
        # write it out
        fname = os.path.join(workdir, name + ".slurm")
        if await aiofiles.ospath.exists(fname):
            # Note: we dont raise an error because we want to be able to rerun
            #       a canceled/crashed run in the same directory without the
            #       need to remove files
            logger.error("Overwriting existing submission file (%s).",
                         fname)
        async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]:
            async with aiofiles.open(fname, 'w') as f:
                await f.write(script)
        return await slurm.create_slurmprocess_submit(
                                        jobname=name,
                                        sbatch_script=fname,
                                        workdir=workdir,
                                        time=walltime,
                                        sbatch_options=self.sbatch_options,
                                        stdfiles_removal="success",
                                        stdin=None,
                                                            )

    async def _acquire_resources_gmx_mdrun(self) -> None:
        if _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB] is not None:
            logger.debug("SLURM_MAX_JOB semaphore is %s before acquiring.",
                         _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB])
            await _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB].acquire()
        else:
            logger.debug("SLURM_MAX_JOB semaphore is None")

    async def _cleanup_gmx_mdrun(self, workdir: str, run_name: str | None = None,
                                 ) -> None:
        if _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB] is not None:
            _OPT_SEMAPHORES[_OPT_SEMAPHORES_KEYS.SLURM_MAX_JOB].release()
        # remove the sbatch script
        name = self._name_from_name_or_none(run_name=run_name)
        fname = os.path.join(workdir, name + ".slurm")
        try:
            # Note: the 0step MD removes the whole folder in which it runs
            # (including the sbatch script)
            await aiofiles.os.remove(fname)
        except FileNotFoundError:
            pass
