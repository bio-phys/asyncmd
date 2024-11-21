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
import abc
import shlex
import asyncio
import inspect
import logging
import hashlib
import functools
import typing
import aiofiles
import aiofiles.os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


from .._config import _SEMAPHORES
from .. import slurm
from ..tools import ensure_executable_available, remove_file_if_exist_async
from .trajectory import Trajectory


logger = logging.getLogger(__name__)


# TODO: DaskTrajectoryFunctionWrapper?!
class TrajectoryFunctionWrapper(abc.ABC):
    """Abstract base class to define the API and some common methods."""
    def __init__(self, **kwargs) -> None:
        # NOTE: in principal we should set these after the stuff set via kwargs
        #       (otherwise users could overwrite them by passing _id="blub" to
        #        init), but since the subclasses sets call_kwargs again and
        #       have to calculate the id according to their own recipe anyway
        #       we can savely set them here (this enables us to use the id
        #        property at initialization time as e.g. in the slurm_jobname
        #        of the SlurmTrajectoryFunctionWrapper)
        self._id = None
        self._call_kwargs = {}  # init to empty dict such that iteration works
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

    @property
    def id(self) -> str:
        """
        Unique and persistent identifier.

        Takes into account the wrapped function and its calling arguments.
        """
        return self._id

    @property
    def call_kwargs(self) -> dict:
        """Additional calling arguments for the wrapped function/executable."""
        # return a copy to avoid people modifying entries without us noticing
        # TODO/FIXME: this will make unhappy users if they try to set single
        #             items in the dict!
        return self._call_kwargs.copy()

    @call_kwargs.setter
    def call_kwargs(self, value):
        if not isinstance(value, dict):
            raise TypeError("call_kwargs must be a dictionary.")
        self._call_kwargs = value
        self._id = self._get_id_str()  # get/set ID

    @abc.abstractmethod
    def _get_id_str(self) -> str:
        # this is expected to return an unique idetifying string
        # this should be unique and portable, i.e. it should enable us to make
        # ensure that the cached values will only be used for the same function
        # called with the same arguments
        pass

    @abc.abstractmethod
    async def get_values_for_trajectory(self, traj):
        # will be called by trajectory._apply_wrapped_func()
        # is expected to return a numpy array, shape=(n_frames, n_dim_function)
        pass

    async def __call__(self, value):
        """
        Apply wrapped function asyncronously on given trajectory.

        Parameters
        ----------
        value : asyncmd.Trajectory
            Input trajectory.

        Returns
        -------
        iterable, usually list or np.ndarray
            The values of the wrapped function when applied on the trajectory.
        """
        if isinstance(value, Trajectory):
            if self.id is not None:
                return await value._apply_wrapped_func(self.id, self)
            raise RuntimeError(f"{type(self)} must have a unique id, but "
                               "self.id is None.")
        raise TypeError(f"{type(self)} must be called with an "
                        "`asyncmd.Trajectory` but was called with "
                        f"{type(value)}.")


class PyTrajectoryFunctionWrapper(TrajectoryFunctionWrapper):
    """
    Wrap python functions for use on :class:`asyncmd.Trajectory`.

    Turns every python callable into an asyncronous (awaitable) and cached
    function for application on :class:`asyncmd.Trajectory`. Also works for
    asyncronous (awaitable) functions, they will be cached.

    Attributes
    ----------
    function : callable
        The wrapped callable.
    call_kwargs : dict
        Keyword arguments for wrapped function.
    """
    def __init__(self, function, call_kwargs: typing.Optional[dict] = None,
                 **kwargs):
        """
        Initialize a :class:`PyTrajectoryFunctionWrapper`.

        Parameters
        ----------
        function : callable
            The (synchronous) callable to wrap.
        call_kwargs : dict, optional
            Keyword arguments for `function`,
            the keys will be used as keyword with the corresponding values,
            by default {}
        """
        super().__init__(**kwargs)
        self._func = None
        self._func_src = None
        self._func_is_async = None
        # use the properties to directly calculate/get the id
        self.function = function
        if call_kwargs is None:
            call_kwargs = {}
        self.call_kwargs = call_kwargs

    def __repr__(self) -> str:
        return (f"PyTrajectoryFunctionWrapper(function={self._func}, "
                + f"call_kwargs={self.call_kwargs})"
                )

    def _get_id_str(self):
        # calculate a hash over function src and call_kwargs dict
        # this should be unique and portable, i.e. it should enable us to make
        # ensure that the cached values will only be used for the same function
        # called with the same arguments
        id = 0
        # NOTE: addition is commutative, i.e. order does not matter here!
        for k, v in self._call_kwargs.items():
            # hash the value
            id += int(hashlib.blake2b(str(v).encode('utf-8')).hexdigest(), 16)
            # hash the key
            id += int(hashlib.blake2b(str(k).encode('utf-8')).hexdigest(), 16)
        # and add the func_src
        id += int(hashlib.blake2b(str(self._func_src).encode('utf-8')).hexdigest(), 16)
        return str(id)  # return a str because we want to use it as dict keys

    @property
    def function(self):
        """
        The python callable this :class:`PyTrajectoryFunctionWrapper` wrapps.
        """
        return self._func

    @function.setter
    def function(self, value):
        self._func_is_async = (inspect.iscoroutinefunction(value)
                               or inspect.iscoroutinefunction(value.__call__))
        try:
            src = inspect.getsource(value)
        except OSError:
            # OSError is raised if source can not be retrieved
            self._func_src = None
            self._id = None
            logger.warning("Could not retrieve source for %s."
                           " No caching can/will be performed.",
                           value,
                           )
        else:
            self._func_src = src
            self._id = self._get_id_str()  # get/set ID
        finally:
            self._func = value

    async def get_values_for_trajectory(self, traj):
        """
        Apply wrapped function asyncronously on given trajectory.

        Parameters
        ----------
        traj : asyncmd.Trajectory
            Input trajectory.

        Returns
        -------
        iterable, usually list or np.ndarray
            The values of the wrapped function when applied on the trajectory.
        """
        if self._func_is_async:
            return await self._get_values_for_trajectory_async(traj)
        return await self._get_values_for_trajectory_sync(traj)

    async def _get_values_for_trajectory_async(self, traj):
        return await self._func(traj, **self._call_kwargs)

    async def _get_values_for_trajectory_sync(self, traj):
        loop = asyncio.get_running_loop()
        async with _SEMAPHORES["MAX_PROCESS"]:
            # fill in additional kwargs (if any)
            if len(self.call_kwargs) > 0:
                func = functools.partial(self.function, **self._call_kwargs)
            else:
                func = self.function
            # NOTE: even though one would expect pythonCVs to be CPU bound
            #       it is actually faster to use a ThreadPoolExecutor because
            #       we then skip the setup + import needed for a second process
            #       In addition most pythonCVs will actually call c/cython-code
            #       like MDAnalysis/mdtraj/etc and are therefore not limited
            #       by the GIL anyway
            #       We leave the code for ProcessPool here because this is the
            #       only place where this could make sense to think about as
            #       opposed to concatenation of trajs (which is IO bound)
            # NOTE: make sure we do not fork! (not save with multithreading)
            # see e.g. https://stackoverflow.com/questions/46439740/safe-to-call-multiprocessing-from-a-thread-in-python
            #ctx = multiprocessing.get_context("forkserver")
            #with ProcessPoolExecutor(1, mp_context=ctx) as pool:
            with ThreadPoolExecutor(max_workers=1,
                                    thread_name_prefix="PyTrajFunc_thread",
                                    ) as pool:
                vals = await loop.run_in_executor(pool, func, traj)
        return vals


class SlurmTrajectoryFunctionWrapper(TrajectoryFunctionWrapper):
    """
    Wrap executables to use on :class:`asyncmd.Trajectory` via SLURM.

    The execution of the job is submited to the queueing system with the
    given sbatch script (template).
    The executable will be called with the following positional arguments:

        - full filepath of the structure file associated with the trajectory

        - full filepath of the trajectory to calculate values for, note that
          multipart trajectories result in multiple files/arguments here.

        - full filepath of the file the results should be written to without
          fileending. Note that if no custom loading function is supplied we
          expect that the written file has 'npy' format and the added ending
          '.npy', i.e. we expect the executable to add the ending '.npy' to
          the passed filepath (as e.g. ``np.save($FILEPATH, data)`` would do)

        - any additional arguments from call_kwargs are added as
          ``" {key} {value}" for key, value in call_kwargs.items()``

    See also the examples for a reference (python) implementation of multiple
    different functions/executables for use with this class.

    Attributes
    ----------
    slurm_jobname : str
        Used as name for the job in slurm and also as part of the filename for
        the submission script that will be written (and deleted if everything
        goes well) for every trajectory.
        NOTE: If you modify it, ensure that each SlurmTrajectoryWrapper has a
        unique slurm_jobname.
    executable : str
        Name of or path to the wrapped executable.
    call_kwargs : dict
        Keyword arguments for wrapped executable.
    """

    def __init__(self, executable, sbatch_script,
                 call_kwargs: typing.Optional[dict] = None,
                 load_results_func=None, **kwargs):
        """
        Initialize :class:`SlurmTrajectoryFunctionWrapper`.

        Note that all attributes can be set via __init__ by passing them as
        keyword arguments.

        Parameters
        ----------
        executable : str
            Absolute or relative path to an executable or name of an executable
            available via the environment (e.g. via the $PATH variable on LINUX)
        sbatch_script : str
            Path to a sbatch submission script file or string with the content
            of a submission script. Note that the submission script must
            contain the following placeholders (also see the examples folder):

             - {cmd_str} : Replaced by the command to call the executable on a given trajectory.

        call_kwargs : dict, optional
            Dictionary of additional arguments to pass to the executable, they
            will be added to the call as pair ' {key} {val}', note that in case
            you want to pass single command line flags (like '-v') this can be
            achieved by setting key='-v' and val='', i.e. to the empty string.
            The values are shell escaped using `shlex.quote()` when writing
            them to the sbatch script.
        load_results_func : None or function (callable)
            Function to call to customize the loading of the results.
            If a function is supplied, it will be called with the full path to
            the results file (as in the call to the executable) and should
            return a numpy array containing the loaded values.
        """
        # property defaults before superclass init to be resettable via kwargs
        self._slurm_jobname = None
        super().__init__(**kwargs)
        self._executable = None
        # we expect sbatch_script to be a str,
        # but it could be either the path to a submit script or the content of
        # the submission script directly
        # we decide what it is by checking for the shebang
        if not sbatch_script.startswith("#!"):
            # probably path to a file, lets try to read it
            with open(sbatch_script, 'r') as f:
                sbatch_script = f.read()
        # (possibly) use properties to calc the id directly
        self.sbatch_script = sbatch_script
        self.executable = executable
        if call_kwargs is None:
            call_kwargs = {}
        self.call_kwargs = call_kwargs
        self.load_results_func = load_results_func

    @property
    def slurm_jobname(self) -> str:
        """
        The jobname of the slurm job used to compute the function results.

        Must be unique for each :class:`SlurmTrajectoryFunctionWrapper`
        instance. Will by default include the persistent unique ID :meth:`id`.
        To (re)set to the default set it to None.
        """
        if self._slurm_jobname is None:
            return f"CVfunc_id_{self.id}"
        return self._slurm_jobname

    @slurm_jobname.setter
    def slurm_jobname(self, val: str | None):
        self._slurm_jobname = val

    def __repr__(self) -> str:
        return (f"SlurmTrajectoryFunctionWrapper(executable={self._executable}, "
                + f"call_kwargs={self.call_kwargs})"
                )

    def _get_id_str(self):
        # calculate a hash over executable and call_kwargs dict
        # this should be unique and portable, i.e. it should enable us to make
        # ensure that the cached values will only be used for the same function
        # called with the same arguments
        id = 0
        # NOTE: addition is commutative, i.e. order does not matter here!
        for k, v in self._call_kwargs.items():
            # hash the value
            id += int(hashlib.blake2b(str(v).encode('utf-8')).hexdigest(), 16)
            # hash the key
            id += int(hashlib.blake2b(str(k).encode('utf-8')).hexdigest(), 16)
        # and add the executable hash
        with open(self.executable, "rb") as exe_file:
            # NOTE: we assume that executable is small enough to read at once
            #       if this crashes becasue of OOM we should use chunks...
            data = exe_file.read()
        id += int(hashlib.blake2b(data).hexdigest(), 16)
        return str(id)  # return a str because we want to use it as dict keys

    @property
    def executable(self):
        """The executable used to compute the function results."""
        return self._executable

    @executable.setter
    def executable(self, val):
        exe = ensure_executable_available(val)
        # if we get here it should be save to set, i.e. it exists + has X-bit
        self._executable = exe
        self._id = self._get_id_str()  # get the new hash/id

    async def get_values_for_trajectory(self, traj):
        """
        Apply wrapped function asyncronously on given trajectory.

        Parameters
        ----------
        traj : asyncmd.Trajectory
            Input trajectory.

        Returns
        -------
        iterable, usually list or np.ndarray
            The values of the wrapped function when applied on the trajectory.
        """
        # first construct the path/name for the numpy npy file in which we expect
        # the results to be written
        tra_dir, tra_name = os.path.split(traj.trajectory_files[0])
        if len(traj.trajectory_files) > 1:
            tra_name += f"_len{len(traj.trajectory_files)}multipart"
        hash_part = str(traj.trajectory_hash)[:5]
        # put in the hash (calculated over all traj parts for multipart)
        # to make sure trajectories with the same first part but different
        # remaining parts dont get mixed up
        result_file = os.path.abspath(os.path.join(
                        tra_dir, f"{tra_name}_{hash_part}_CVfunc_id_{self.id}"
                                                   ))
        # we expect executable to take 3 postional args:
        # struct traj outfile
        cmd_str = f"{self.executable} {os.path.abspath(traj.structure_file)}"
        cmd_str += f" {' '.join(os.path.abspath(t) for t in traj.trajectory_files)}"
        cmd_str += f" {result_file}"
        if len(self.call_kwargs) > 0:
            for key, val in self.call_kwargs.items():
                # shell escape only the values,
                # the keys (i.e. option names/flags) should be no issue
                if isinstance(val, list):
                    # enable lists of arguments for the same key,
                    # can then be used e.g. with pythons argparse `nargs="*"` or `nargs="+"`
                    cmd_str += f" {key} {' '.join([shlex.quote(str(v)) for v in val])}"
                else:
                    cmd_str += f" {key} {shlex.quote(str(val))}"
        # now prepare the sbatch script
        script = self.sbatch_script.format(cmd_str=cmd_str)
        # write it out
        sbatch_fname = os.path.join(tra_dir,
                                    tra_name + "_" + self.slurm_jobname + ".slurm")
        if os.path.exists(sbatch_fname):
            logger.error("Overwriting existing submission file (%s)."
                         " Are you sure your `slurm_jobname` is unique?",
                         sbatch_fname,
                         )
        async with _SEMAPHORES["MAX_FILES_OPEN"]:
            async with aiofiles.open(sbatch_fname, 'w') as f:
                await f.write(script)
        # NOTE: we set returncode to 2 (what slurmprocess returns in case of
        # node failure) and rerun/retry until we either get a completed job
        # or a non-node-failure error
        returncode = 2
        while returncode == 2:
            # run slurm job
            returncode, slurm_proc, stdout, stderr = await self._run_slurm_job(
                                                   sbatch_fname=sbatch_fname,
                                                   result_file=result_file,
                                                   slurm_workdir=tra_dir,
                                                                              )
            if returncode == 2:
                logger.error("Exit code indicating node fail from CV batch job"
                             " for executable %s on trajectory %s (slurm jobid"
                             " %s). stderr was: %s. stdout was: %s",
                             self.executable, traj, slurm_proc.slurm_jobid,
                             stdout.decode(), stderr.decode(),
                             )
        if returncode != 0:
            # Can not be exitcode 2, because of the while loop above
            raise RuntimeError(
                            "Non-zero exit code from CV batch job for "
                            + f"executable {self.executable} on "
                            + f"trajectory {traj} "
                            + f"(slurm jobid {slurm_proc.slurm_jobid})."
                            + f" Exit code was: {returncode}."
                            + f" stderr was: {stderr.decode()}."
                            + f" and stdout was: {stdout.decode()}"
                                    )
        # zero-exitcode: load the results
        if self.load_results_func is None:
            # we do not have '.npy' ending in results_file,
            # numpy.save() adds it if it is not there, so we need it here
            load_func = np.load
            fname_results = result_file + ".npy"
        else:
            # use custom loading function from user
            load_func = self.load_results_func
            fname_results = result_file
        # use a separate thread to load so we dont block with the io
        loop = asyncio.get_running_loop()
        async with _SEMAPHORES["MAX_FILES_OPEN"]:
            async with _SEMAPHORES["MAX_PROCESS"]:
                with ThreadPoolExecutor(
                            max_workers=1,
                            thread_name_prefix="SlurmTrajFunc_load_thread",
                                        ) as pool:
                    vals = await loop.run_in_executor(pool, load_func,
                                                      fname_results)
        # remove the results file and sbatch script
        await asyncio.gather(remove_file_if_exist_async(fname_results),
                             remove_file_if_exist_async(sbatch_fname),
                             )
        return vals

    async def _run_slurm_job(self, sbatch_fname: str, result_file: str,
                             slurm_workdir: str,
                             ) -> tuple[int,slurm.SlurmProcess,bytes,bytes]:
        # submit and run slurm-job
        if _SEMAPHORES["SLURM_MAX_JOB"] is not None:
            await _SEMAPHORES["SLURM_MAX_JOB"].acquire()
        try:  # this try is just to make sure we always release the semaphore
            slurm_proc = await slurm.create_slurmprocess_submit(
                                                jobname=self.slurm_jobname,
                                                sbatch_script=sbatch_fname,
                                                workdir=slurm_workdir,
                                                stdfiles_removal="success",
                                                stdin=None,
                                                # sleep 5 s between checking
                                                sleep_time=5,
                                                                )
            # wait for the slurm job to finish
            # also cancel the job when this future is canceled
            stdout, stderr = await slurm_proc.communicate()
            returncode = slurm_proc.returncode
            return returncode, slurm_proc, stdout, stderr
        except asyncio.CancelledError:
            slurm_proc.kill()
            # clean up the sbatch file and potentialy written result file
            res_fname = (result_file + ".npy" if self.load_results_func is None
                         else result_file)
            await asyncio.gather(remove_file_if_exist_async(sbatch_fname),
                                 remove_file_if_exist_async(res_fname),
                                 )
            raise  # reraise CancelledError for encompassing coroutines
        finally:
            if _SEMAPHORES["SLURM_MAX_JOB"] is not None:
                _SEMAPHORES["SLURM_MAX_JOB"].release()
