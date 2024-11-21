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
import os
import copy
import shlex
import random
import string
import shutil
import typing
import asyncio
import logging
import aiofiles
import aiofiles.os
import aiofiles.ospath

from .._config import _SEMAPHORES
from ..mdengine import MDEngine, EngineError, EngineCrashedError
from ..trajectory.trajectory import Trajectory
from .. import slurm
from .mdconfig import MDP
from .utils import nstout_from_mdp, get_all_traj_parts
from ..tools import ensure_executable_available


logger = logging.getLogger(__name__)


class _descriptor_on_instance_and_class:
    # a descriptor that makes the (default) value of the private attribute
    # "_name" accessible as "name" on both the class and the instance level
    # Accessing the default value works from the class-level, i.e. without
    # instantiating the object, but note that setting on the class level
    # overwrites the descriptor and does not call __set__
    # Setting from an instance calls __set__ and therefore only sets
    # the attribute for the given instance (and also runs our checks)
    # also see the python docs:
    #  https://docs.python.org/3/howto/descriptor.html#customized-names
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            # I (hejung) think if obj is None objtype will always be set
            # to the class of the obj
            obj = objtype
        val = getattr(obj, self.private_name)
        return val

    def __set__(self, obj, val):
        setattr(obj, self.private_name, val)


class _descriptor_output_traj_type(_descriptor_on_instance_and_class):
    # Check the output_traj_type for consistency before setting
    def __set__(self, obj, val):
        allowed_values = ["trr", "xtc"]
        val = val.lower()
        if val not in allowed_values:
            raise ValueError("output_traj_type must be one of "
                             + f"{allowed_values}, but was {val}."
                             )
        return super().__set__(obj, val)


class _descriptor_check_executable(_descriptor_on_instance_and_class):
    # check if a given value is a valid executable when setting it
    # we use this to make sure gmx grompp and gmx mdrun are available as set
    def __set__(self, obj, val):
        # split because mdrun and grompp can be both subcommands of gmx
        test_exe = val.split(" ")[0]
        ensure_executable_available(test_exe)
        return super().__set__(obj, val)


# NOTE: with tra we usually mean trr, i.e. a full precision trajectory with velocities
class GmxEngine(MDEngine):
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
    mdrun_extra_args : str
        Can be used to pass extra command line arguments to mdrun calls,
        e.g. "-ntomp 8".
    output_traj_type : str
        Sets the trajectory type (ending) this engine returns/looks for.
        Note that we simply ignore all other trajectories, i.e. depending on
        the MDP settings we will still write xtc and trr, but return only the
        trajectories with matching ending.
    """

    # local prepare and option to run a local gmx (mainly for testing)
    _grompp_executable = "gmx grompp"
    grompp_executable = _descriptor_check_executable()
    _mdrun_executable = "gmx mdrun"
    mdrun_executable = _descriptor_check_executable()
    # extra_args are expected to be str and will be appended to the end of the
    # respective commands after a separating space,
    # i.e. cmd = base_cmd + " " + extra_args
    grompp_extra_args = ""
    mdrun_extra_args = ""
    # file ending of the returned output trajectories,
    # exposed as property output_traj_type
    # NOTE: this will be the traj we count frames for and check the mdp, etc.
    #       However this does not mean that no other trajs will/can be written,
    #       we simply ignore them
    _output_traj_type = "xtc"
    output_traj_type = _descriptor_output_traj_type()
    # See the notes below for the SlurmGmxEngine on why this conversion factor
    # is needed (there), here we have it only for consistency
    _mdrun_time_conversion_factor = 1.  # run mdrun for 1. * time-limit


    def __init__(self,
                 mdconfig: MDP,
                 gro_file: str,
                 top_file: str,
                 ndx_file: str | None = None,
                 **kwargs) -> None:
        """
        Initialize a :class:`GmxEngine`.

        Note that all attributes can be set at intialization by passing keyword
        arguments with their name, e.g. mdrun_extra_args="-ntomp 2" to instruct
        gromacs to use 2 openMP threads.

        Parameters
        ----------
        mdconfig: MDP
            The molecular dynamics parameters.
        gro_file: str
            Absolute or relative path to a gromacs structure file.
        top_file: str
            Absolute or relative path to a gromacs topolgy (.top) file.
        ndx_file: str or None
            Optional, absolute or relative path to a gromacs index file.
        """
        # make it possible to set any attribute via kwargs
        # check the type for attributes with default values
        dval = object()
        for kwarg, value in kwargs.items():
            cval = getattr(self, kwarg, dval)
            if cval is not dval:
                if isinstance(value, type(cval)):
                    # value is of same type as default so set it
                    setattr(self, kwarg, value)
                else:
                    raise TypeError(f"Setting attribute {kwarg} with "
                                    + f"mismatching type ({type(value)}). "
                                    + f" Default type is {type(cval)}."
                                    )
            else:
                # not previously defined, so warn that we ignore it
                logger.warning("Ignoring unknown keyword-argument %s.", kwarg)
        # NOTE: after the kwargs setting to be sure they are what we set/expect
        # TODO: store a hash/the file contents for gro, top, ndx?
        #       to check against when we load from storage/restart?
        #       if we do this do it in the property!
        #       (but still write one hashfunc for all!)
        self.gro_file = gro_file  # sets self._gro_file
        self.top_file = top_file  # sets self._top_file
        self.ndx_file = ndx_file  # sets self._ndx_file
        # dirty hack to make sure we also check for our defaults if they are
        # available + executable
        self.mdrun_executable = self.mdrun_executable
        self.grompp_executable = self.grompp_executable
        # basic checks for mdp are done in the property-setter, e.g. if the
        # output_traj_type is actually written with current mdp-settings
        self.mdp = mdconfig
        # initialize internal state variables
        self._workdir = None
        self._prepared = False
        # NOTE: frames_done and steps_done do not have an easy relation!
        #       See the steps_done property docstring for more!
        # number of frames produced since last call to prepare
        self._frames_done = 0
        # number of integration steps done since last call to prepare
        self._steps_done = 0
        # integration time since last call to prepare in ps
        self._time_done = 0.
        self._nstout = None  # get this from the mdp only when we need it
        # Popen handle for gmx mdrun, used to check if we are running
        self._proc = None
        # these are set by prepare() and used by run_XX()
        self._simulation_part = None
        self._deffnm = None
        # tpr for trajectory (part), will become the structure/topology file
        self._tpr = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # cant pickle the process, + its probably dead when we unpickle :)
        state["_proc"] = None
        return state

    @property
    def current_trajectory(self) -> Trajectory | None:
        """
        Return the last finished trajectory (part).

        Returns
        -------
        Trajectory
            Last complete trajectory produced by this engine.
        """
        if self._simulation_part == 0:
            # we could check if self_proc is set (which prepare sets to None)
            # this should make sure that calling current trajectory after
            # calling prepare does not return a traj, as soon as we called
            # run self._proc will be set, i.e. there is still no gurantee that
            # the traj is done, but it will be started always
            # (even when accessing simulataneous to the call to run),
            # i.e. it is most likely done
            # we can also check for simulation part, since it seems
            # gmx ignores that if no checkpoint is passed, i.e. we will
            # **always** start with part0001 anyways!
            # but checking for self._simulation_part == 0 also just makes sure
            # we never started a run (i.e. same as checking self._proc)
            return None
        if (all(v is not None for v in [self._tpr, self._deffnm])
            and not self.running):
            # self._tpr and self._deffnm are set in prepare, i.e. having them
            # set makes sure that we have at least prepared running the traj
            # but it might not be done yet
            traj = Trajectory(
                    trajectory_files=os.path.join(
                                        self.workdir,
                                        (f"{self._deffnm}"
                                         + f"{self._num_suffix(self._simulation_part)}"
                                         + f".{self.output_traj_type}")
                                                 ),
                    # NOTE: self._tpr already contains the path to workdir
                    structure_file=self._tpr,
                    nstout=self.nstout,
                              )
            return traj
        return None

    @property
    def ready_for_run(self) -> bool:
        """Whether this engine is ready to run, i.e. generate a trajectory."""
        return self._prepared and not self.running

    @property
    def running(self) -> bool:
        """Whether this engine is currently running/generating a trajectory."""
        if self._proc is None:
            # this happens when we did not call run() yet
            return False
        if self._proc.returncode is None:
            # no return code means it is still running
            return True
        # dont care for the value of the exit code,
        # we are not running anymore if we crashed ;)
        return False

    @property
    def workdir(self) -> str:
        """The current woring directory of the engine."""
        return self._workdir

    @workdir.setter
    def workdir(self, value: str) -> None:
        if not os.path.isdir(value):
            raise TypeError(f"Not a directory ({value}).")
        value = os.path.relpath(value)
        self._workdir = value

    @property
    def gro_file(self) -> str:
        """The (path to the) gro file this engine uses/used to call grompp."""
        return self._gro_file

    @gro_file.setter
    def gro_file(self, val: str) -> str:
        if not os.path.isfile(val):
            raise FileNotFoundError(f"gro file not found: {val}")
        val = os.path.relpath(val)
        self._gro_file = val

    @property
    def top_file(self) -> str:
        """The (path to the) top file this engine uses/used to call grompp."""
        return self._top_file

    @top_file.setter
    def top_file(self, val: str) -> None:
        if not os.path.isfile(val):
            raise FileNotFoundError(f"top file not found: {val}")
        val = os.path.relpath(val)
        self._top_file = val

    @property
    def ndx_file(self) -> str | None:
        """The (path to the) ndx file this engine uses/used to call grompp."""
        return self._ndx_file

    @ndx_file.setter
    def ndx_file(self, val: str | None) -> None:
        if val is not None:
            # GMX does not require an ndx file, so we accept None
            if not os.path.isfile(val):
                raise FileNotFoundError(f"ndx file not found: {val}")
            val = os.path.relpath(val)
        # set it anyway (even if it is None)
        self._ndx_file = val

    @property
    def mdp(self) -> MDP:
        """The configuration of this engine as a :class:`MDP` object."""
        return self._mdp

    @mdp.setter
    def mdp(self, val: MDP) -> None:
        if not isinstance(val, MDP):
            raise TypeError(f"Value must be of type {MDP}.")
        try:
            if val["nsteps"] != -1:
                logger.info("Changing nsteps from %s to -1 (infinte), the run "
                            "length is controlled via arguments of the run "
                            "method.",
                            val['nsteps'])
                val["nsteps"] = -1
        except KeyError:
            # nsteps not defined
            logger.info("Setting previously undefined nsteps to -1 (infinite).")
            val["nsteps"] = -1
        # check that we get a trajectory of the format we expect with our
        # current mdp, we do this by using nstout_from_mdp since it throws a
        # nice error if the mdp does not generate output for given traj-format
        # TODO: ensure that x-out and v-out/f-out are the same (if applicable)?
        _ = nstout_from_mdp(mdp=val, traj_type=self.output_traj_type)
        self._mdp = val

    # alias for mdp to mdconfig (since some users may expect mdconfig)
    mdconfig = mdp

    @property
    def dt(self) -> float:
        """Integration timestep in ps."""
        return self._mdp["dt"]

    @property
    def time_done(self) -> float:
        """
        Integration time since last call to prepare in ps.

        Takes into account 'tinit' from the .mdp file if set.
        """
        try:
            tinit = self._mdp["tinit"]
        except KeyError:
            tinit = 0.
        return self._time_done - tinit

    # TODO/FIXME: we assume that all output frequencies are multiples of the
    #             smallest when determing the number of frames etc
    # TODO: check that nstxout == nstvout?!
    @property
    def nstout(self) -> int:
        """Smallest output frequency for current output_traj_type."""
        if self._nstout is None:
            nstout = nstout_from_mdp(self._mdp,
                                     traj_type=self.output_traj_type)
            self._nstout = nstout
        return self._nstout

    @property
    def steps_done(self) -> int:
        """
        Number of integration steps done since last call to :meth:`prepare`.

        NOTE: steps != frames * nstout
        Some remarks on the relation between frames_done and steps_done:
        Usually (when passing `nsteps` to `run()`) frames_done will be equal to
        steps_done/nstout + 1 because the initial/final configuration will be
        written twice (since then the first/last step is always an output step)
        However as soon as we run for a specific walltime (without specifying
        `nsteps`) stuff gets complicated, then gromacs can potentially stop at
        every neighbor search step (where it also can/will write a checkpoint).
        If that step is not a trajectory output step, no output will be written
        to the traj and then the plus 1 rule for the double written
        initial/final configuration is off (since it will then be a 'normal'
        configuration written just once).
        If however the neighbor search and trajectory output fall togehter on
        the same step the configuration will be written twice (as with `nsteps`
        specified).
        """
        return self._steps_done

    @property
    def frames_done(self) -> int:
        """
        Number of frames produced since last call to :meth:`prepare`.

        NOTE: frames != steps / nstout
        See the steps_done docstring for more.
        """
        return self._frames_done

    async def apply_constraints(self, conf_in, conf_out_name, wdir="."):
        """
        Apply constraints to given configuration.

        Parameters
        ----------
        conf_in : asyncmd.Trajectory
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

    async def generate_velocities(self, conf_in, conf_out_name, wdir=".",
                                  constraints=True):
        """
        Generate random Maxwell-Boltzmann velocities for given configuration.

        Parameters
        ----------
        conf_in : asyncmd.Trajectory
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

    async def _0step_md(self, conf_in, conf_out_name, wdir,
                        constraints: bool, generate_velocities: bool):
        if (self.workdir is not None) and (wdir == "."):
            # use own working directory if know/set
            wdir = self.workdir
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
        constraints_mdp = copy.deepcopy(self._mdp)
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
        # TODO: this is a bit hacky, and should probably not be necessary?
        #       we keep a ref to the 'old' self._proc to reset it after we are
        #       done, because the gmx_mdrun method set self._proc to the running
        #       constraints engine
        #       and it is probably not necessary since no engine should be able
        #       to be runing when/if we are able to call apply_constraints?
        old_proc_val = self._proc
        cmd_str = self._mdrun_cmd(tpr=os.path.join(swdir, f"{run_name}.tpr"),
                                  workdir=swdir,
                                  deffnm=run_name)
        logger.debug("About to execute gmx mdrun command for constraints and"
                     "/or velocity generation: %s",
                     cmd_str)
        returncode = None
        stderr = bytes()
        stdout = bytes()
        await self._acquire_resources_gmx_mdrun()
        try:
            await self._start_gmx_mdrun(cmd_str=cmd_str, workdir=swdir,
                                        run_name=run_name,
                                        # TODO/FIXME: we hardcode that the runs
                                        # can not be longer than 15 min here
                                        # (but i think this should be fine for
                                        #  randomizing velocities and/or
                                        #  applying constraints?!)
                                        walltime=0.25,
                                        )
            # self._proc is set by _start_gmx_mdrun!
            stdout, stderr = await self._proc.communicate()
            returncode = self._proc.returncode
        except asyncio.CancelledError:
            self._proc.kill()
            raise  # reraise the error for encompassing coroutines
        else:
            if returncode != 0:
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
            self._proc = old_proc_val

    async def prepare(self, starting_configuration, workdir, deffnm):
        """
        Prepare a fresh simulation (starting with part0001).

        Can also be used to continue a simulation from a checkpoint file with
        matching name ('deffnm.cpt'). In that case, the `simulation-part` mdp
        option must match the number of the next part to be generated, e.g. it
        must be 2 if the last part generated was part0001. The previously
        generated trajectory files do not need to exist.

        Parameters
        ----------
        starting_configuration : asyncmd.Trajectory or None
            A (trr) trajectory of which we take the first frame as starting
            configuration (including velocities) or None, then the initial
            configuration is the gro-file.
        workdir : str
            Absolute or relative path to an existing directory to use as
            working directory.
        deffnm : str
            The name (prefix) to use for all files.
        """
        # deffnm is the default name/prefix for all outfiles (as in gmx)
        self._deffnm = deffnm
        self.workdir = workdir  # sets to abspath and check if it is a dir
        # check 'simulation-part' option in mdp file / MDP options
        # it decides at which .partXXXX the gmx numbering starts,
        # however gromacs ignores it if there is no -cpi [CheckPointIn]
        # so we do the same, i.e. we warn if we detect it is set
        # and check if there is a checkpoint with the right name [deffnm.cpt]
        # if yes we set our internal simulation_part counter to the value from
        # the mdp - 1 (we increase *before* each simulation part)
        cpt_fname = os.path.join(self.workdir, f"{deffnm}.cpt")
        try:
            sim_part = self._mdp["simulation-part"]
        except KeyError:
            # the gmx mdp default is 1, it starts at part0001
            # we add one at the start of each run, i.e. the numberings match up
            # and we will have tra=`...part0001.trr` from gmx
            # and confout=`...part0001.gro` from our naming
            self._simulation_part = 0
        else:
            if sim_part > 1:
                if not os.path.isfile(cpt_fname):
                    raise ValueError("'simulation-part' > 1 is only possible "
                                     + "if starting from a checkpoint, but "
                                     + f"{cpt_fname} does not exist."
                                     )
                starting_configuration = cpt_fname
                logger.warning("Starting value for 'simulation-part' > 1 (=%s)"
                               " and existing checkpoint file found (%s). "
                               "Using the checkpoint file as "
                               "`starting_configuration`.",
                               sim_part, cpt_fname)
            # always substract one from sim_part so we get 0 if it was 1
            self._simulation_part = sim_part - 1
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
        if len(trajs_with_same_deffnm) > self._simulation_part:
            raise ValueError(f"There are files in workdir ({self.workdir}) "
                             + f"with the same deffnm ({deffnm}). Use the "
                             + "`prepare_from_files()` method to continue an "
                             + "existing MD run or change the workdir and/or "
                             + "deffnm.")
        # actucal preparation of MDrun: sort out starting configuration...
        if ((starting_configuration is None)
            # None enables start from the initial structure file ('-c' option)
            or isinstance(starting_configuration, str)
            # str enables passing the path to the full precision trajectory
            # directly, i.e. trr, cpt, or tng
            ):
            trr_in = starting_configuration
        elif isinstance(starting_configuration, Trajectory):
            # enable passing of asyncmd.Trajectories as starting_configuration
            trr_in = starting_configuration.trajectory_files[0]
        else:
            raise TypeError("Starting_configuration must be None, a wrapped "
                            "full precission trajectrtory, or the path to a "
                            "full precission trajectory (trr, cpt, or tng).")
        # ...and call grompp to get a tpr
        # remember the path to use as structure file for out trajs
        self._tpr = os.path.join(self.workdir, deffnm + ".tpr")
        await self._run_grompp(workdir=self.workdir, deffnm=self._deffnm,
                               trr_in=trr_in, tpr_out=self._tpr,
                               mdp_obj=self._mdp)
        if not await aiofiles.ospath.isfile(self._tpr):
            # better be save than sorry :)
            raise RuntimeError("Something went wrong generating the tpr. "
                               f"{self._tpr} does not seem to be a file.")
        # make sure we can not mistake a previous Popen for current mdrun
        self._proc = None
        self._frames_done = 0  # (re-)set how many frames we did
        self._steps_done = 0
        self._time_done = 0.
        self._prepared = True

    async def _run_grompp(self, workdir, deffnm, trr_in, tpr_out, mdp_obj):
        # NOTE: file paths from workdir and deffnm
        mdp_in = os.path.join(workdir, deffnm + ".mdp")
        # write the mdp file (always overwriting existing mdps)
        # I (hejung) think this is what we want as the prepare methods check
        # for leftover files with the same deffnm, so if only the mdp is there
        # we can (and want to) just ovewrite it without raising an err
        async with _SEMAPHORES["MAX_FILES_OPEN"]:
            mdp_obj.write(mdp_in, overwrite=True)
        mdp_out = os.path.join(workdir, deffnm + "_mdout.mdp")
        cmd_str = self._grompp_cmd(mdp_in=mdp_in, tpr_out=tpr_out,
                                   workdir=workdir,
                                   trr_in=trr_in, mdp_out=mdp_out)
        logger.debug("About to execute gmx grompp command: %s", cmd_str)
        # 3 file descriptors: stdin, stdout, stderr
        # NOTE: The max open files semaphores counts for 3 open files, so we
        #       only need it once
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        try:
            grompp_proc = await asyncio.create_subprocess_exec(
                                                *shlex.split(cmd_str),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=workdir,
                                                               )
            stdout, stderr = await grompp_proc.communicate()
            return_code = grompp_proc.returncode
            logger.debug("gmx grompp command returned return code %s.",
                         return_code)
            #logger.debug("grompp stdout:\n%s", stdout.decode())
            #logger.debug("grompp stderr:\n%s", stderr.decode())
            if return_code != 0:
                # this assumes POSIX
                raise RuntimeError("grompp had non-zero return code "
                                   + f"({return_code}).\n"
                                   + "\n--------\n"
                                   + f"stderr: \n--------\n {stderr.decode()}"
                                   + "\n--------\n"
                                   + f"stdout: \n--------\n {stdout.decode()}"
                                   )
        except asyncio.CancelledError as e:
            grompp_proc.kill()  # kill grompp
            raise e from None  # and reraise the cancelation
        finally:
            # release the semaphore
            _SEMAPHORES["MAX_FILES_OPEN"].release()

    async def prepare_from_files(self, workdir: str, deffnm: str):
        """
        Prepare continuation run starting from the last part found in workdir.

        Expects the checkpoint file and last trajectory part to exist, will
        (probably) fail otherwise.

        Parameters
        ----------
        workdir : str
            Absolute or relative path to an exisiting directory to use as
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
                           + "present in the current workdir. Assuming the "
                           + "highest part number corresponds to the "
                           + "checkpoint file and continuing anyway."
                           )
        # load the 'old' mdp_in
        async with _SEMAPHORES["MAX_FILES_OPEN"]:
            self._mdp = MDP(os.path.join(self.workdir, f"{deffnm}.mdp"))
        self._deffnm = deffnm
        # Note that we dont need to explicitly check for the tpr existing,
        # if it does not exist we will err when getting the traj lengths
        self._tpr = os.path.join(self.workdir, deffnm + ".tpr")
        self._simulation_part = last_partnum
        # len(t), because for frames we do not care if first frame is in traj
        self._frames_done = sum(len(t) for t in previous_trajs)
        # steps done is the more reliable info if we want to know how many
        # integration steps we did
        self._steps_done = previous_trajs[-1].last_step
        self._time_done = previous_trajs[-1].last_time
        self._proc = None
        self._prepared = True

    # NOTE: this enables us to reuse run and prepare methods in SlurmGmxEngine,
    # i.e. we only need to overwite the next 3 functions to write out the slurm
    # submission script, submit the job and allocate/release different resources
    async def _start_gmx_mdrun(self, cmd_str, workdir, **kwargs):
        proc = await asyncio.create_subprocess_exec(
                                            *shlex.split(cmd_str),
                                            stdout=asyncio.subprocess.PIPE,
                                            stderr=asyncio.subprocess.PIPE,
                                            cwd=workdir,
                                                    )
        self._proc = proc

    async def _acquire_resources_gmx_mdrun(self, **kwargs):
        # *always* called before any gmx_mdrun, used to reserve resources
        # for local gmx we need 3 file descriptors: stdin, stdout, stderr
        # (one max files semaphore counts for 3 open files)
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()

    async def _cleanup_gmx_mdrun(self, **kwargs):
        # *always* called after any gmx_mdrun, use to release resources
        # release the semaphore for the 3 file descriptors
        _SEMAPHORES["MAX_FILES_OPEN"].release()

    async def run(self, nsteps=None, walltime=None, steps_per_part=False):
        """
        Run simulation for specified number of steps or/and a given walltime.

        Note that you can pass both nsteps and walltime and the simulation will
        stop on the condition that is reached first.

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
        if not self.ready_for_run:
            raise RuntimeError("Engine not ready for run. Call self.prepare() "
                               + "and/or check if it is still running.")
        if all(kwarg is None for kwarg in [nsteps, walltime]):
            raise ValueError("Neither steps nor walltime given.")
        if nsteps is not None:
            nsteps = int(nsteps)
            if nsteps % self.nstout != 0:
                raise ValueError(f"nsteps ({nsteps}) must be a multiple of "
                                 + f"nstout ({self.nstout}).")
            if not steps_per_part:
                nsteps = nsteps - self.steps_done
            if nsteps == 0:
                # Return None instead of raising an error, this makes it nicer
                # to use the run method with walltime and total nsteps inside
                # while loops, i.e. we can just call traj = e.run(...) and then
                # while traj is not None: traj = e.run()
                # TODO: this will make it complicated to ever use the GmxEngine
                #       for zero-step simulations to only apply constraints,
                #       but we do have the _0_step_md methods for that...?!
                return None
            elif nsteps < 0:
                raise ValueError(f"nsteps is too small ({nsteps} steps for "
                                 + "this part). Can not travel backwards in "
                                 + "time...")

        self._simulation_part += 1
        cmd_str = self._mdrun_cmd(tpr=self._tpr, workdir=self.workdir,
                                  deffnm=self._deffnm,
                                  # TODO: use more/any other kwargs?
                                  maxh=walltime, nsteps=nsteps)
        logger.debug("About to execute gmx mdrun command: %s", cmd_str)
        returncode = None
        stderr = bytes()
        stdout = bytes()
        await self._acquire_resources_gmx_mdrun()
        try:
            await self._start_gmx_mdrun(cmd_str=cmd_str, workdir=self.workdir,
                                        walltime=walltime,)
            # self._proc is set by _start_gmx_mdrun!
            stdout, stderr = await self._proc.communicate()
            returncode = self._proc.returncode
        except asyncio.CancelledError as e:
            if self._proc is not None:
                # make sure _proc is set, it can still be None if we get
                # canceled while _start_gmx_mdrun is setting up the process
                self._proc.kill()
            raise e from None  # reraise the error for encompassing coroutines
        else:
            logger.debug("gmx mdrun command returned return code %s.",
                         returncode)
            #logger.debug("gmx mdrun stdout:\n%s", stdout.decode())
            #logger.debug("gmx mdrun stderr:\n%s", stderr.decode())
            if returncode == 0:
                self._frames_done += len(self.current_trajectory)
                # dont care if we did a little more and only the checkpoint knows
                # we will only find out with the next trajectory part anyways
                self._steps_done = self.current_trajectory.last_step
                self._time_done = self.current_trajectory.last_time
                return self.current_trajectory
            else:
                raise EngineCrashedError(
                    f"Non-zero (or no) exit code from mdrun (= {returncode}).\n"
                    + "\n--------\n"
                    + f"stderr: \n--------\n {stderr.decode()}"
                    + "\n--------\n"
                    + f"stdout: \n--------\n {stdout.decode()}"
                                         )
        finally:
            await self._cleanup_gmx_mdrun(workdir=self.workdir)

    async def run_steps(self, nsteps, steps_per_part=False):
        """
        Run simulation for specified number of steps.

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

    async def run_walltime(self, walltime):
        """
        Run simulation for a given walltime.

        Parameters
        ----------
        walltime : float or None
            (Maximum) walltime in hours, `None` means unlimited.
        """
        return await self.run(walltime=walltime)

    def _num_suffix(self, sim_part: int) -> str:
        # construct gromacs num part suffix from simulation_part
        num_suffix = f".part{sim_part:04d}"
        return num_suffix

    def _grompp_cmd(self, mdp_in, tpr_out, workdir, trr_in=None, mdp_out=None):
        # all args are expected to be file paths
        # make sure we use the right ones, i.e. relative to workdir
        if workdir is not None:
            mdp_in = os.path.relpath(mdp_in, start=workdir)
            tpr_out = os.path.relpath(tpr_out, start=workdir)
            gro_file = os.path.relpath(self.gro_file, start=workdir)
            top_file = os.path.relpath(self.top_file, start=workdir)
        cmd = f"{self.grompp_executable} -f {mdp_in} -c {gro_file}"
        cmd += f" -p {top_file}"
        if self.ndx_file is not None:
            if workdir is not None:
                ndx_file = os.path.relpath(self.ndx_file, start=workdir)
            else:
                ndx_file = self.ndx_file
            cmd += f" -n {ndx_file}"
        if trr_in is not None:
            # input trr is optional
            # TODO/FIXME?!
            # TODO/NOTE: currently we do not pass '-time', i.e. we just use the
            #            gmx default frame selection: last frame from trr
            if workdir is not None:
                trr_in = os.path.relpath(trr_in, start=workdir)
            cmd += f" -t {trr_in}"
        if mdp_out is None:
            # find out the name and dir of the tpr to put the mdp next to it
            head, tail = os.path.split(tpr_out)
            name = tail.split(".")[0]
            mdp_out = os.path.join(head, name + ".mdout.mdp")
        if workdir is not None:
            mdp_out = os.path.relpath(mdp_out, start=workdir)
        cmd += f" -o {tpr_out} -po {mdp_out}"
        if self.grompp_extra_args != "":
            # add extra args string if it is not empty
            cmd += f" {self.grompp_extra_args}"
        return cmd

    def _mdrun_cmd(self, tpr, workdir, deffnm=None, maxh=None, nsteps=None):
        # use "-noappend" to avoid appending to the trajectories when starting
        # from checkpoints, instead let gmx create new files with .partXXXX suffix
        if workdir is not None:
            tpr = os.path.relpath(tpr, start=workdir)
        if deffnm is None:
            # find out the name of the tpr and use that as deffnm
            head, tail = os.path.split(tpr)
            deffnm = tail.split(".")[0]
        #cmd = f"{self.mdrun_executable} -noappend -deffnm {deffnm} -cpi"
        # NOTE: the line above does the same as the four below before the if-clauses
        #       however gromacs -deffnm is deprecated (and buggy),
        #       so we just make our own 'deffnm', i.e. we name all files the same
        #       except for the ending but do so explicitly
        cmd = f"{self.mdrun_executable} -noappend -s {tpr}"
        # always add the -cpi option, this lets gmx figure out if it wants
        # to start from a checkpoint (if there is one with deffnm)
        # cpi (CheckPointIn) is ignored if not present,
        # cpo (CheckPointOut) is the name to use for the (final) checkpoint
        cmd += f" -cpi {deffnm}.cpt -cpo {deffnm}.cpt"
        cmd += f" -o {deffnm}.trr -x {deffnm}.xtc -c {deffnm}.confout.gro"
        cmd += f" -e {deffnm}.edr -g {deffnm}.log"
        if maxh is not None:
            maxh = self._mdrun_time_conversion_factor * maxh
            cmd += f" -maxh {maxh}"
        if nsteps is not None:
            cmd += f" -nsteps {nsteps}"
        if self.mdrun_extra_args != "":
            cmd += f" {self.mdrun_extra_args}"
        return cmd


class SlurmGmxEngine(GmxEngine):
    __doc__ = GmxEngine.__doc__
    # use local prepare (i.e. grompp) of GmxEngine then submit run to slurm
    # we reuse the `GmxEngine._proc` to keep a reference to a `SlurmProcess`
    # which emulates the API of `asyncio.subprocess.Process` and can (for our
    # purposes) be used as a drop-in replacement, therefore we only need to
    # reimplement `_start_gmx_mdrun()`, `_acquire_resources_gmx_mdrun()` and
    # `_cleanup_gmx_mdrun()` to have a working SlurmGmxEngine
    # take submit script as str/file, use pythons .format to insert stuff!
    # TODO: use SLURM also for grompp?! (would make stuff faster?)
    #       I (hejung) think probably not by much because we already use
    #       asyncios subprocess for grompp (i.e. do it asyncronous) and grompp
    #       will most likely not take much resources on the login (local) node

    # NOTE: these are possible options, but they result in added dependencies
    #        - jinja2 templates for slurm submission scripts?
    #          (does not look like we gain flexibility but we get more work,
    #           so probably not?!)
    #        - pyslurm for job status checks?!
    #          (it seems submission is frickly/impossible in pyslurm,
    #           so also probably not?!)

    _mdrun_executable = "gmx_mpi mdrun"  # MPI as default for clusters
    _mdrun_time_conversion_factor = 0.99  # run mdrun for 0.99 * time-limit
    # NOTE: The rationale behind the (slightly) reduced mdrun time compared to
    #       the slurm job time-limit is that sometimes setting up and finishing
    #       up the slurm job takes some time (e.g. activating modules, sourcing
    #       environments, etc.) and this can result in jobs that are cancelled
    #       due to reaching the maximum time limit in slurm. This in turn means
    #       that we would believe the job failed because it got cancelled
    #       although the mdrun was successfull.

    def __init__(self, mdconfig, gro_file, top_file, sbatch_script, ndx_file=None,
                 **kwargs):
        """
        Initialize a :class:`SlurmGmxEngine`.

        Parameters
        ----------
        mdconfig : MDP
            The molecular dynamics parameters.
        gro_file: str
            Absolute or relative path to a gromacs structure file.
        top_file: str
            Absolute or relative path to a gromacs topolgy (.top) file.
        sbatch_script : str
            Absolute or relative path to a slurm sbatch script or a string with
            the content of the sbatch script. Note that the submission script
            must contain the following placeholders (see also the examples
            folder):

             - {mdrun_cmd} : Replaced by the command to run mdrun

        ndx_file: str or None
            Optional, absolute or relative path to a gromacs index file.

        Note that all attributes can be set at intialization by passing keyword
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
            with open(sbatch_script, 'r') as f:
                sbatch_script = f.read()
        self.sbatch_script = sbatch_script

    def _name_from_name_or_none(self, run_name: typing.Optional[str]) -> str:
        if run_name is not None:
            name = run_name
        else:
            # create a name from deffnm and partnum
            name = self._deffnm + self._num_suffix(sim_part=self._simulation_part)
        return name

    async def _start_gmx_mdrun(self, cmd_str, workdir, walltime=None,
                               run_name=None, **kwargs):
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
        async with _SEMAPHORES["MAX_FILES_OPEN"]:
            async with aiofiles.open(fname, 'w') as f:
                await f.write(script)
        self._proc = await slurm.create_slurmprocess_submit(
                                                jobname=name,
                                                sbatch_script=fname,
                                                workdir=workdir,
                                                time=walltime,
                                                stdfiles_removal="success",
                                                stdin=None,
                                                            )

    async def _acquire_resources_gmx_mdrun(self, **kwargs):
        if _SEMAPHORES["SLURM_MAX_JOB"] is not None:
            logger.debug("SLURM_MAX_JOB semaphore is %s before acquiring.",
                         _SEMAPHORES['SLURM_MAX_JOB'])
            await _SEMAPHORES["SLURM_MAX_JOB"].acquire()
        else:
            logger.debug("SLURM_MAX_JOB semaphore is None")

    async def _cleanup_gmx_mdrun(self, workdir, run_name=None, **kwargs):
        if _SEMAPHORES["SLURM_MAX_JOB"] is not None:
            _SEMAPHORES["SLURM_MAX_JOB"].release()
        # remove the sbatch script
        name = self._name_from_name_or_none(run_name=run_name)
        fname = os.path.join(workdir, name + ".slurm")
        try:
            # Note: the 0step MD removes the whole folder in which it runs
            # (including the sbatch script)
            await aiofiles.os.remove(fname)
        except FileNotFoundError:
            pass

    # TODO: do we even need/want this?
    @property
    def slurm_job_state(self) -> str | None:
        """The state of the slurm job as reported by slurm."""
        if self._proc is None:
            return None
        return self._proc.slurm_job_state
