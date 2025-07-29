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
This module contains the implementation of the SlurmProcess class.

The SlurmProcess is a drop-in replacement for asyncio.subprocess.Subprocess and
in this spirit this module also contains the function create_slurmprocess_submit,
which similarly to asyncio.create_subprocess_exec, creates a SlurmProcess and
directly submits the job.
"""
import asyncio
import logging
import shlex
import subprocess
import typing
import os
import aiofiles

from .cluster_mediator import SlurmClusterMediator
from .constants_and_errors import (SlurmCancellationError,
                                   SlurmSubmissionError,
                                   SlurmError,
                                   )
from ..tools import (ensure_executable_available,
                     remove_file_if_exist_async,
                     remove_file_if_exist,
                     attach_kwargs_to_object as _attach_kwargs_to_object,
                     )
from .._config import _SEMAPHORES, _SEMAPHORES_KEYS


logger = logging.getLogger(__name__)


class SlurmProcess:
    """
    Generic wrapper around SLURM submissions.

    Imitates the interface of `asyncio.subprocess.Process`.

    Attributes
    ----------
    sbatch_executable : str
        Name or path to the sbatch executable, by default "sbatch".
    scancel_executable: str
        Name or path to the scancel executable, by default "scancel".
    sleep_time : int
        Time (in seconds) between checks if the underlying job has finished
        when using `self.wait`. By default 15 s.
    stdfiles_removal : str
        Whether to remove the stdout, stderr (and possibly stdin) files.
        Possible values are:

        - "success": remove on successful completion, i.e. zero returncode)
        - "no": never remove
        - "yes"/"always": remove on job completion independent of
          returncode and also when using :meth:`terminate`

        By default "success".
    """

    # use same instance of class for all SlurmProcess instances
    try:
        slurm_cluster_mediator = SlurmClusterMediator()
    except ValueError:
        slurm_cluster_mediator = None
        # we raise a ValueError if sacct/sinfo are not available
        logger.warning("Could not initialize SLURM cluster handling. "
                       "If you are sure SLURM (sinfo/sacct/etc) is available "
                       "try calling `asyncmd.config.set_all_slurm_settings()` "
                       "or `asyncmd.config.set_slurm_setting()` "
                       "with the appropriate arguments.")
    # we can not simply wait for the subprocess, since slurm exits directly
    # so we will sleep for this long between checks if slurm-job completed
    sleep_time = 15  # TODO: heuristic? dynamically adapt?
    # NOTE: no options to set/pass extra_args for sbatch:
    #       the only command line options for sbatch we allow will be controlled
    #       by us since cmd line options for sbatch take precedence over every-
    #       thing else. This will e.g. allow us to reliably control the output
    #       files and therefore enable to implement communicate(), i.e. parse
    #       stderr and stdout
    sbatch_executable = "sbatch"
    scancel_executable = "scancel"
    # default value for stdfile_removal
    _stdfiles_removal = "success"

    def __init__(self, jobname: str, sbatch_script: str, *,
                 workdir: str | None = None,
                 time: float | None = None,
                 sbatch_options: dict | None = None,
                 **kwargs) -> None:
        """
        Initialize a `SlurmProcess`.

        Note that you can set all attributes by passing matching init kwargs
        with the wanted values.

        Parameters
        ----------
        jobname : str
            SLURM jobname (``--job-name``).
        sbatch_script : str
            Absolute or relative path to a SLURM submission script.
        workdir : str or None
            Absolute or relative path to use as working directory. None will
            result in using the current directory as workdir.
        time : float or None
            Timelimit for the job in hours. None will result in using the
            default as either specified in the sbatch script or the partition.
        sbatch_options : dict or None
            Dictionary of sbatch options, keys are long names for options,
            values are the corresponding values. The keys/long names are given
            without the dashes, e.g. to specify ``--mem=1024`` the dictionary
            needs to be ``{"mem": "1024"}``. To specify options without values
            use keys with empty strings as values, e.g. to specify
            ``--contiguous`` the dictionary needs to be ``{"contiguous": ""}``.
            See the SLURM documentation for a full list of sbatch options
            (https://slurm.schedmd.com/sbatch.html).

        Raises
        ------
        TypeError
            If the value set via init kwarg for a attribute does not match the
            default/original type for that attribute.
        """
        if not isinstance(self.slurm_cluster_mediator, SlurmClusterMediator):
            raise SlurmError(
                "SLURM monitoring not initialized. Please call "
                "`asyncmd.config.set_slurm_setting()` or "
                "`asyncmd.config.set_all_slurm_settings` with appropriate "
                "arguments to ensure `sinfo` and `sacct` executables are available."
                )
        # we expect sbatch_script to be a path to a file
        # make it possible to set any attribute via kwargs
        # check the type for attributes with default values
        _attach_kwargs_to_object(obj=self, logger=logger, **kwargs)
        # this either checks for our defaults or whatever we just set via kwargs
        ensure_executable_available(self.sbatch_executable)
        ensure_executable_available(self.scancel_executable)
        self.jobname = jobname
        # TODO/FIXME: do we want sbatch_script to be relative to wdir?
        #             (currently it is relative to current dir when creating
        #              the slurmprocess)
        self.sbatch_script = os.path.abspath(sbatch_script)
        if workdir is None:
            workdir = os.getcwd()
        self.workdir = os.path.abspath(workdir)
        self._time = time
        # Use the property to directly call _sanitize_sbatch_options when assigning
        # Do this **after** setting self._time to ensure consistency
        if sbatch_options is None:
            sbatch_options = {}
        self.sbatch_options = sbatch_options
        self._jobid: None | str = None
        # dict with jobinfo cached from slurm cluster mediator
        self._jobinfo: dict[str, typing.Any] = {}
        self._stdout_data: None | bytes = None
        self._stderr_data: None | bytes = None
        self._stdin: None | str = None

    def _sanitize_sbatch_options(self, sbatch_options: dict) -> dict:
        """
        Return sane sbatch_options dictionary to be consistent (with self).

        Parameters
        ----------
        sbatch_options : dict
            Dictionary of sbatch options.

        Returns
        -------
        dict
            Dictionary with sanitized sbatch options.
        """
        # NOTE: this should be called every time we modify sbatch_options or self.time!
        # This is the list of sbatch options we use ourself, they should not
        # be in the dict to avoid unforeseen effects. We treat 'time' special
        # because we want to allow for it to be specified via sbatch_options if
        # it is not set via the attribute self.time.
        reserved_sbatch_options = ["job-name", "chdir", "output", "error",
                                   "input", "exclude", "parsable"]
        new_sbatch_options = sbatch_options.copy()
        if "time" in sbatch_options:
            if self._time is not None:
                logger.warning("Removing sbatch option 'time' from 'sbatch_options'. "
                               "Using the 'time' argument instead.")
                del new_sbatch_options["time"]
            else:
                logger.debug("Using 'time' from 'sbatch_options' because "
                             "self.time is None.")
        for option in reserved_sbatch_options:
            if option in sbatch_options:
                logger.warning("Removing sbatch option '%s' from "
                               "'sbatch_options' because it is used internally "
                               "by the `SlurmProcess`.", option)
                del new_sbatch_options[option]

        return new_sbatch_options

    def _slurm_timelimit_from_time_in_hours(self, time: float) -> str:
        """
        Create timelimit in SLURM compatible format from time in hours.

        Parameters
        ----------
        timelimit : float
            Timelimit for job in hours

        Returns
        -------
        str
            Timelimit for job as SLURM compatible string.
        """
        timelimit = time * 60
        timelimit_min = int(timelimit)  # take only the full minutes
        timelimit_sec = round(60 * (timelimit - timelimit_min))
        timelimit_str = f"{timelimit_min}:{timelimit_sec}"
        return timelimit_str

    @property
    def time(self) -> float | None:
        """
        Timelimit for SLURM job in hours.

        Can be a float or None (meaning do not specify a timelimit).
        """
        return self._time

    @time.setter
    def time(self, val: float | None) -> None:
        self._time = val
        self._sbatch_options: dict = self._sanitize_sbatch_options(self._sbatch_options)

    @property
    def sbatch_options(self) -> dict:
        """
        A copy of the sbatch_options dictionary.

        Note that modifying single key, value pairs has no effect, to modify
        (single) sbatch_options either use the `set_sbatch_option` and
        `del_sbatch_option` methods or (re)assign a dictionary to
        `sbatch_options`.
        """
        return self._sbatch_options.copy()

    @sbatch_options.setter
    def sbatch_options(self, val: dict) -> None:
        self._sbatch_options = self._sanitize_sbatch_options(val)

    def set_sbatch_option(self, key: str, value: str) -> None:
        """
        Set sbatch option with given key to value.

        I.e. add/modify single key, value pair in sbatch_options.

        Parameters
        ----------
        key : str
            The name of the sbatch option.
        value : str
            The value for the sbatch option.
        """
        self._sbatch_options[key] = value
        self._sbatch_options = self._sanitize_sbatch_options(self._sbatch_options)

    def del_sbatch_option(self, key: str) -> None:
        """
        Delete sbatch option with given key from sbatch_options.

        Parameters
        ----------
        key : str
            The name of the sbatch option to delete.
        """
        del self._sbatch_options[key]
        self._sbatch_options = self._sanitize_sbatch_options(self._sbatch_options)

    @property
    def stdfiles_removal(self) -> str:
        """
        Whether/when we remove stdfiles created by SLURM.

        Can be one of "success", "no", "yes", "always", where "yes" and
        "always" are synonyms for always remove. "success" means remove
        stdfiles if the slurm-job was successful and "no" means never remove.
        """
        return self._stdfiles_removal

    @stdfiles_removal.setter
    def stdfiles_removal(self, val: str) -> None:
        allowed_vals = ["success", "no", "yes", "always"]
        if val.lower() not in allowed_vals:
            raise ValueError(f"remove_stdfiles must be one of {allowed_vals}, "
                             + f"but was {val.lower()}.")
        self._stdfiles_removal = val.lower()

    async def submit(self, stdin: str | None = None) -> None:
        """
        Submit the job via sbatch.

        Parameters
        ----------
        stdin : str or None
            If given it is interpreted as a file to which we connect the batch
            scripts stdin via sbatchs ``--input`` option. This enables sending
            data to the processes stdin via :meth:`communicate`.
            Note that if it is desired to send data to the process the process
            has to be submitted with stdin.

        Raises
        ------
        RuntimeError
            If the job has already been submitted.
        SlurmSubmissionError
            If something goes wrong during the submission with sbatch.
        CancelledError
            (Re)raises CancelledError if cancelled while awaiting the sbatch
            submission.
        """
        if self._jobid is not None:
            raise RuntimeError(f"Already monitoring job with id {self._jobid}.")
        sbatch_cmd = f"{self.sbatch_executable}"
        sbatch_cmd += f" --job-name={self.jobname}"
        # set working directory for batch script to workdir
        sbatch_cmd += f" --chdir={self.workdir}"
        # FIXME/TODO: does this work for job-arrays?
        #             (probably not, but do we care?)
        sbatch_cmd += f" --output=./{self._stdout_name(use_slurm_symbols=True)}"
        sbatch_cmd += f" --error=./{self._stderr_name(use_slurm_symbols=True)}"
        if self.time is not None:
            sbatch_cmd += f" --time={self._slurm_timelimit_from_time_in_hours(self.time)}"
        # keep a ref to the stdin value, we need it in communicate
        self._stdin = stdin
        if stdin is not None:
            sbatch_cmd += f" --input=./{stdin}"
        # get the list of nodes we dont want to run on
        exclude_nodes = self.slurm_cluster_mediator.exclude_nodes
        if len(exclude_nodes) > 0:
            sbatch_cmd += f" --exclude={','.join(exclude_nodes)}"
        # add all other (user-defined) sbatch options
        for key, val in self.sbatch_options.items():
            sbatch_cmd += f" --{key}={val}" if val else f" --{key}"
        sbatch_cmd += f" --parsable {self.sbatch_script}"
        logger.debug("About to execute sbatch_cmd %s.", sbatch_cmd)
        # 3 file descriptors: stdin,stdout,stderr
        # Note: one semaphore counts for 3 open files!
        await _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN].acquire()
        sbatch_proc = await asyncio.create_subprocess_exec(
                                                *shlex.split(sbatch_cmd),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=self.workdir,
                                                close_fds=True,
                                                )
        try:
            stdout, stderr = await sbatch_proc.communicate()
        except asyncio.CancelledError as e:
            sbatch_proc.kill()
            raise e from None
        else:
            sbatch_stdout = stdout.decode()
            sbatch_stderr = stderr.decode()
            if (returncode := sbatch_proc.returncode):
                raise SlurmSubmissionError(
                    "Could not submit SLURM job. "
                    f"sbatch had non-zero returncode ({returncode}). \n"
                    f"sbatch stdout: {sbatch_stdout} \n"
                    f"sbatch stderr: {sbatch_stderr} \n"
                    )
        finally:
            _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN].release()
        # only jobid (and possibly clustername) returned, semicolon to separate
        logger.debug("sbatch returned stdout: %s, stderr: %s.",
                     sbatch_stdout, sbatch_stderr)
        jobid = sbatch_stdout.split(";")[0].strip()
        # make sure jobid is an int/ can be cast as one
        try:
            _ = int(jobid)
        except ValueError as e:
            # can not cast to int, so probably something went wrong submitting
            raise SlurmSubmissionError(
                "Could not submit SLURM job, failed to parse sbatch return as a jobid. "
                f"sbatch stdout: {sbatch_stdout} \n"
                f"sbatch stderr: {sbatch_stderr} \n"
                ) from e
        logger.info("Submitted SLURM job with jobid %s.", jobid)
        self._jobid = jobid
        self.slurm_cluster_mediator.monitor_register_job(jobid=jobid)
        # get jobinfo (these will probably just be the defaults but at
        #  least this is a dict with the right keys...)
        await self._update_sacct_jobinfo()

    @property
    def slurm_jobid(self) -> str | None:
        """The slurm jobid of this job."""
        return self._jobid

    @property
    def nodes(self) -> list[str] | None:
        """The nodes this job runs on."""
        return self._jobinfo.get("nodelist", None)

    @property
    def slurm_job_state(self) -> str | None:
        """The slurm jobstate of this job."""
        return self._jobinfo.get("state", None)

    @property
    def returncode(self) -> int | None:
        """The returncode this job returned (if finished)."""
        if self._jobid is None:
            return None
        return self._jobinfo.get("parsed_exitcode", None)

    def _stdout_name(self, use_slurm_symbols: bool = False) -> str:
        name = f"{self.jobname}.out."
        if use_slurm_symbols:
            name += "%j"
        elif self.slurm_jobid is not None:
            name += f"{self.slurm_jobid}"
        else:
            raise RuntimeError("Can not construct stdout filename without jobid.")
        return name

    def _stderr_name(self, use_slurm_symbols: bool = False) -> str:
        name = f"{self.jobname}.err."
        if use_slurm_symbols:
            name += "%j"
        elif self.slurm_jobid is not None:
            name += f"{self.slurm_jobid}"
        else:
            raise RuntimeError("Can not construct stderr filename without jobid.")
        return name

    def _remove_stdfiles_sync(self) -> None:
        fnames = [self._stdin] if self._stdin is not None else []
        fnames += [self._stdout_name(use_slurm_symbols=False),
                   self._stderr_name(use_slurm_symbols=False),
                   ]
        for f in fnames:
            remove_file_if_exist(f=os.path.join(self.workdir, f))

    async def _remove_stdfiles_async(self) -> None:
        fnames = [self._stdin] if self._stdin is not None else []
        fnames += [self._stdout_name(use_slurm_symbols=False),
                   self._stderr_name(use_slurm_symbols=False),
                   ]
        await asyncio.gather(
                *(remove_file_if_exist_async(os.path.join(self.workdir, f))
                  for f in fnames)
                             )

    async def _read_stdfiles(self) -> tuple[bytes, bytes]:
        if self._stdout_data is not None and self._stderr_data is not None:
            # return cached values if we already read the files previously
            return self._stdout_data, self._stderr_data
        # we read them in binary mode to get bytes objects back, this way they
        # behave like the bytes objects returned by asyncio.subprocess
        async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]:
            stdout_fname = os.path.join(
                                    self.workdir,
                                    self._stdout_name(use_slurm_symbols=False),
                                        )
            if os.path.isfile(stdout_fname):
                async with aiofiles.open(stdout_fname, "rb") as f:
                    stdout = await f.read()
            else:
                logger.warning("stdout file %s not found.", stdout_fname)
                stdout = bytes()
            stderr_fname = os.path.join(
                                    self.workdir,
                                    self._stderr_name(use_slurm_symbols=False),
                                        )
            if os.path.isfile(stderr_fname):
                async with aiofiles.open(stderr_fname, "rb") as f:
                    stderr = await f.read()
            else:
                logger.warning("stderr file %s not found.", stderr_fname)
                stderr = bytes()
        # cache the content
        self._stdout_data = stdout
        self._stderr_data = stderr
        return stdout, stderr

    async def _update_sacct_jobinfo(self) -> None:
        # Note that the cluster mediator limits the call frequency for sacct
        # updates and is the same for all SlurmProcess instances, so we dont
        # need to take care of limiting from slurm process side
        self._jobinfo = await self.slurm_cluster_mediator.get_info_for_job(jobid=self.slurm_jobid)

    async def wait(self) -> int:
        """
        Wait for the SLURM job to finish. Set and return the returncode.

        Returns
        -------
        int
            returncode of the wrapped SLURM job

        Raises
        ------
        RuntimeError
            If the job has never been submitted.
        """
        if self.slurm_jobid is None:
            # make sure we can only wait after submitting, otherwise we would
            # wait indefinitely if we call wait() before submit()
            raise RuntimeError("Can only wait for submitted SLURM jobs with "
                               + "known jobid. Did you ever submit the job?")
        while self.returncode is None:
            await asyncio.sleep(self.sleep_time)
            await self._update_sacct_jobinfo()  # update local cached jobinfo
        self.slurm_cluster_mediator.monitor_remove_job(jobid=self.slurm_jobid)
        if (
            ((not self.returncode) and (self.stdfiles_removal == "success"))
            or self.stdfiles_removal in {"yes", "always"}
        ):
            # read them in and cache them so we can still call communicate()
            # to get the data later
            _, _ = await self._read_stdfiles()
            await self._remove_stdfiles_async()
        return self.returncode

    # pylint: disable-next=redefined-builtin
    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        """
        Interact with process. Optionally send data to the process.
        Wait for the process to finish, then read from stdout and stderr (files)
        and return the data.

        Parameters
        ----------
        input : bytes or None, optional
            The input data to send to the process, by default None.
            Note that you an only send data to processes created/submitted with
            stdin set.

        Returns
        -------
        tuple[bytes, bytes]
            (stdout, stderr)

        Raises
        ------
        RuntimeError
            If the job has never been submitted.
        ValueError
            If stdin is not None but the process was created without stdin set.
        """
        # order as in asyncio.subprocess, there it is:
        #   1.) write to stdin (optional)
        #   2.) read until EOF is reached
        #   3.) wait for the proc to finish
        # Note that we wait first because we can only start reading the
        # stdfiles when the job has at least started, so we just wait for it
        # and read the files at the end completely
        if self._jobid is None:
            # make sure we can only wait after submitting, otherwise we would
            # wait indefinitely if we call wait() before submit()
            raise RuntimeError("Can only wait for submitted SLURM jobs with "
                               + "known jobid. Did you ever submit the job?")
        if input is not None:
            if self._stdin is None:
                # make sure we have a stdin file if we have input to write
                raise ValueError("Can only send input to a SlurmProcess "
                                 "created/submitted with stdin (file) given.")
            # write the given input to stdin file
            async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]:
                async with aiofiles.open(os.path.join(self.workdir,
                                                      f"{self._stdin}"),
                                         "wb",
                                         ) as f:
                    await f.write(input)
        # NOTE: wait makes sure we deregister the job from monitoring and also
        #       removes the stdfiles as/if requested
        _ = await self.wait()
        stdout, stderr = await self._read_stdfiles()
        return stdout, stderr

    def send_signal(self, signal: int) -> None:
        """
        Send signal to the underlying slurm job.

        Currently not implemented!
        """
        # TODO: write the "send_signal" method! (if we actually need it?)
        #       [should be doable via scancel, which can send signals to jobs]
        #       [could maybe also work using scontrol
        #        (which makes the state change know to slurm demon)]
        raise NotImplementedError

    def terminate(self) -> None:
        """
        Terminate (cancel) the underlying SLURM job.

        Raises
        ------
        SlurmCancellationError
            If scancel has non-zero returncode.
        RuntimeError
            If no jobid is known, e.g. because the job was never submitted.
        """
        if self._jobid is not None:
            scancel_cmd = f"{self.scancel_executable} {self._jobid}"
            try:
                scancel_out = subprocess.check_output(shlex.split(scancel_cmd),
                                                      text=True)
            except subprocess.CalledProcessError as e:
                raise SlurmCancellationError(
                        "Something went wrong canceling the slurm job "
                        + f"{self._jobid}. scancel had exitcode {e.returncode}"
                        + f" and output {e.output}."
                        ) from e
            # if we got until here the job is successfully canceled....
            logger.debug("Canceled SLURM job with jobid %s. "
                         "scancel returned %s.",
                         self.slurm_jobid, scancel_out,
                         )
            # remove the job from the monitoring
            self.slurm_cluster_mediator.monitor_remove_job(jobid=self._jobid)
            if self._stdfiles_removal in {"yes", "always"}:
                # and remove stdfiles as/if requested
                self._remove_stdfiles_sync()
        else:
            # we probably never submitted the job?
            raise RuntimeError("self.jobid is not set, can not cancel a job "
                               + "with unknown jobid. Did you ever submit it?")

    def kill(self) -> None:
        """Alias for :meth:`terminate`."""
        self.terminate()


async def create_slurmprocess_submit(jobname: str,
                                     sbatch_script: str, *,
                                     workdir: str | None = None,
                                     time: float | None = None,
                                     sbatch_options: dict | None = None,
                                     stdin: str | None = None,
                                     **kwargs,
                                     ):
    """
    Create and submit a :class:`SlurmProcess`.

    All arguments are directly passed trough to :class:`SlurmProcess` initialization
    and :meth:`SlurmProcess.submit`.
    All additional keyword arguments are passed to :class:`SlurmProcess` initialization.

    Parameters
    ----------
    jobname : str
        SLURM jobname (``--job-name``).
    sbatch_script : str
        Absolute or relative path to a SLURM submission script.
    workdir : str
        Absolute or relative path to use as working directory.
    time : float or None
        Timelimit for the job in hours. None will result in using the
        default as either specified in the sbatch script or the partition.
    sbatch_options : dict or None
        Dictionary of sbatch options, keys are long names for options,
        values are the corresponding values. The keys/long names are given
        without the dashes, e.g. to specify ``--mem=1024`` the dictionary
        needs to be ``{"mem": "1024"}``. To specify options without values
        use keys with empty strings as values, e.g. to specify
        ``--contiguous`` the dictionary needs to be ``{"contiguous": ""}``.
        See the SLURM documentation for a full list of sbatch options
        (https://slurm.schedmd.com/sbatch.html).
    stdin : str or None
        If given it is interpreted as a file to which we connect the batch
        scripts stdin via sbatchs ``--input`` option. This enables sending
        data to the processes stdin via :meth:`communicate`.
        Note that if it is desired to send data to the process the process
        has to be submitted with stdin.
    kwargs: dict, optional
        Additional keyword arguments to be passed to :meth`SlurmProcess.__init__`.

    Returns
    -------
    SlurmProcess
        The submitted slurm process instance.
    """
    proc = SlurmProcess(jobname=jobname, sbatch_script=sbatch_script,
                        workdir=workdir, time=time,
                        sbatch_options=sbatch_options,
                        **kwargs)
    await proc.submit(stdin=stdin)
    return proc
