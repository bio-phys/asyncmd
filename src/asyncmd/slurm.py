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
import time
import shlex
import asyncio
import aiofiles
import subprocess
import collections
import typing
import logging

from .tools import ensure_executable_available
from ._config import _SEMAPHORES


logger = logging.getLogger(__name__)


class SlurmError(RuntimeError):
    """Generic error superclass for all SLURM errors."""


class SlurmCancelationError(SlurmError):
    """Error raised when something goes wrong canceling a SLURM job."""


class SlurmSubmissionError(SlurmError):
    """Error raised when something goes wrong submitting a SLURM job."""


# rudimentary map for slurm state codes to int return codes for poll
# NOTE: these are the sacct states (they differ from the squeue states)
#       cf. https://slurm.schedmd.com/sacct.html#lbAG
#       and https://slurm.schedmd.com/squeue.html#lbAG
_SLURM_STATE_TO_EXITCODE = {
    "BOOT_FAIL": 1,  # Job terminated due to launch failure
    # Job was explicitly cancelled by the user or system administrator.
    "CANCELLED": 1,
    # Job has terminated all processes on all nodes with an exit code of
    # zero.
    "COMPLETED": 0,
    "DEADLINE": 1,  # Job terminated on deadline.
    # Job terminated with non-zero exit code or other failure condition.
    "FAILED": 1,
    # Job terminated due to failure of one or more allocated nodes.
    "NODE_FAIL": 1,
    "OUT_OF_MEMORY": 1,  # Job experienced out of memory error.
    "PENDING": None,  # Job is awaiting resource allocation.
    # NOTE: preemption means interupting a process to later restart it,
    #       i.e. None is probably the right thing to return
    "PREEMPTED": None,  # Job terminated due to preemption.
    "RUNNING": None,  # Job currently has an allocation.
    "REQUEUED": None,  # Job was requeued.
    # Job is about to change size.
    #"RESIZING" TODO: when does this happen? what should we return?
    # Sibling was removed from cluster due to other cluster starting the
    # job.
    "REVOKED": 1,
    # Job has an allocation, but execution has been suspended and CPUs have
    # been released for other jobs.
    "SUSPENDED": None,
    # Job terminated upon reaching its time limit.
    "TIMEOUT": 1,  # TODO: can this happen for jobs that finish properly?
}


# TODO: better classname?!
class SlurmClusterMediator:
    """
    Singleton class to be used by all SlurmProcess for sacct/sinfo calls.

    Attributes
    ----------
    sinfo_executable : str
        Name or path to the sinfo executable, by default "sinfo".
    sacct_executable : str
        Name or path to the sacct executable, by default "sacct".
    min_time_between_sacct_calls : int
        Minimum time (in seconds) between subsequent sacct calls.
    num_fails_for_broken_node : int
        Number of failed jobs we need to observe per node before declaring it
        to be broken (and not submitting any more jobs to it).
    success_to_fail_ratio : int
        Number of successful jobs we need to observe per node to decrease the
        failed job counter by one.

    """

    sinfo_executable = "sinfo"
    sacct_executable = "sacct"
    # wait for at least 10 s between two sacct calls
    min_time_between_sacct_calls = 10
    # NOTE: We track the number of failed/successfull jobs associated with each
    #       node and use this information to decide if a node is broken
    # number of 'suspected fail' counts that a node needs to accumulate for us
    # to declare it broken
    num_fails_for_broken_node = 3
    # minimum number of successfuly completed jobs we need to see on a node to
    # decrease the 'suspected fail' counter by one
    success_to_fail_ratio = 50
    # TODO/FIXME: currently we have some tolerance until a node is declared
    #             broken but as soon as it is broken it will stay that forever?!
    #             (here forever means until we reinitialize SlurmClusterMediator)

    def __init__(self, **kwargs) -> None:
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
        # this either checks for our defaults or whatever we just set via kwargs
        self.sacct_executable = ensure_executable_available(self.sacct_executable)
        self.sinfo_executable = ensure_executable_available(self.sinfo_executable)
        self._node_job_fails = collections.Counter()
        self._node_job_successes = collections.Counter()
        self._broken_nodes = []
        self._all_nodes = self.list_all_nodes()
        self._jobids = []  # list of jobids of jobs we know about
        self._jobids_sacct = []  # list of jobids we monitor actively via sacct
        # we will store the info about jobs in a dict keys are jobids
        # values are dicts with key queried option and value the (parsed)
        # return value
        # currently queried options are: state, exitcode and nodelist
        self._jobinfo = {}
        self._last_sacct_call = None  # make sure we dont call sacct too often

    @property
    def broken_nodes(self) -> "list[str]":
        return self._broken_nodes.copy()

    def list_all_nodes(self) -> "list[str]":
        """
        List all node (hostnames) in the SLURM cluster this runs on.

        Returns
        -------
        list[str]
            List of all node (hostnames) queried from sinfo.
        """
        # format option '%n' is a list of node hostnames
        sinfo_cmd = f"{self.sinfo_executable} --noheader --format='%n'"
        sinfo_out = subprocess.check_output(shlex.split(sinfo_cmd), text=True)
        node_list = sinfo_out.split("\n")
        # sinfo_out is terminated by '\n' so our last entry is the empty string
        node_list = node_list[:-1]
        return node_list

    # TODO: better func names?
    def monitor_register_job(self, jobid: str) -> None:
        """
        Add job with given jobid to sacct monitoring calls.

        Parameters
        ----------
        jobid : str
            The SLURM jobid of the job to monitor.
        """
        if jobid not in self._jobids:
            self._jobids.append(jobid)
            self._jobids_sacct.append(jobid)
            # we use a dict with defaults to make sure that we get a 'PENDING'
            # for new jobs because this will make us check again in a bit
            # (sometimes there is a lag between submission and the appearance
            #  of the job in sacct output)
            self._jobinfo[jobid] = {"state": "PENDING",
                                    "exitcode": None,
                                    "parsed_exitcode": None,
                                    "nodelist": [],
                                    }
            logger.debug(f"Registered job with id {jobid} for sacct monitoring.")
        else:
            logger.warning(f"Job with id {jobid} already registered for "
                           + "monitoring. Not adding it again.")

    def monitor_remove_job(self, jobid: str) -> None:
        """
        Remove job with given jobid from sacct monitoring calls.

        Parameters
        ----------
        jobid : str
            The SLURM jobid of the job to remove.
        """
        if jobid in self._jobids:
            self._jobids.remove(jobid)
            del self._jobinfo[jobid]
            try:
                self._jobids_sacct.remove(jobid)
            except ValueError:
                pass  # already not actively monitored anymore
            logger.debug(f"Removed job with id {jobid} from sacct monitoring.")
        else:
            logger.warning(f"Not monitoring job with id {jobid}, not removing.")

    async def get_info_for_job(self, jobid: str) -> dict:
        """
        Retrieve and return info for job with given jobid.

        Parameters
        ----------
        jobid : str
            The SLURM jobid of the queried job.

        Returns
        -------
        dict
            Dictionary with information about the job,
             the keys (str) are sacct format fields,
             the values are the (parsed) corresponding values.
        """
        if (self._last_sacct_call is None
            or (time.time() - self._last_sacct_call
                > self.min_time_between_sacct_calls)):
            # either we never called sacct or at least not in the recent past
            # so update cached jobinfo and save the new time
            self._last_sacct_call = time.time()
            await self._update_cached_jobinfo()
            logger.debug("Updating cached jobinfo.")

        return self._jobinfo[jobid].copy()

    async def _update_cached_jobinfo(self) -> None:
        """Call sacct and update cached info for all jobids we know about."""
        sacct_cmd = f"{self.sacct_executable} --noheader"
        # query only for the specific job we are running
        sacct_cmd += f" -j {','.join(self._jobids_sacct)}"
        sacct_cmd += " -o jobid,state,exitcode,nodelist"
        sacct_cmd += " --parsable2"  # separate with |
        # 3 file descriptors: stdin,stdout,stderr
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        try:
            sacct_proc = await asyncio.subprocess.create_subprocess_exec(
                                                *shlex.split(sacct_cmd),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                close_fds=True,
                                                                          )
            stdout, stderr = await sacct_proc.communicate()
            sacct_return = stdout.decode()
        finally:
            # and put the three back into the semaphore
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
        # only jobid (and possibly clustername) returned, semikolon to separate
        logger.debug(f"sacct returned {sacct_return}.")
        # sacct returns one line per substep, we only care for the whole job
        # which should be the first line but we check explictly for jobid
        # (the substeps have .$NUM suffixes)
        for line in sacct_return.split("\n"):
            splits = line.split("|")
            if len(splits) == 4:
                # basic sanity check that everything went alright parsing
                jobid, state, exitcode, nodelist = splits
                if "." in jobid:
                    # the substeps of jobs have '$jobid.$substepname' as jobid
                    # where $substepname is e.g. 'batch' or '0', we ignore them
                    continue
                # parse returns (remove spaces, etc.) and put them in cache
                jobid = jobid.strip()
                if self._jobinfo[jobid]["state"] == state:
                    # only process nodelist and update jobinfo when necessary
                    # i.e. if the slurm_state changed
                    continue
                nodelist = self._process_nodelist(nodelist=nodelist)
                self._jobinfo[jobid]["nodelist"] = nodelist
                self._jobinfo[jobid]["exitcode"] = exitcode
                self._jobinfo[jobid]["state"] = state
                logger.debug(f"Extracted from sacct output: jobid {jobid},"
                             + f" state {state}, exitcode {exitcode} and "
                             + f"nodelist {nodelist}.")
                parsed_ec = self._parse_exitcode_from_slurm_state(slurm_state=state)
                self._jobinfo[jobid]["parsed_exitcode"] = parsed_ec
                if parsed_ec is not None:
                    logger.debug(f"Parsed slurm state {state} for job {jobid}"
                                 + f" as returncode {parsed_ec}. Removing job"
                                 + "from sacct calls because its state will"
                                 + " not change anymore.")
                    self._jobids_sacct.remove(jobid)
                    self._node_fail_heuristic(jobid=jobid,
                                              parsed_exitcode=parsed_ec,
                                              slurm_state=state,
                                              nodelist=nodelist,
                                              )

    def _process_nodelist(self, nodelist: str) -> "list[str]":
        """
        Expand shorthand nodelist from SLURM to a list of nodes/hostnames.

        I.e. turn the str of nodes in shorthand notation ('phys[04-07]') into
        a list of node hostnames (['phys04', 'phys05', 'phys06']).

        Parameters
        ----------
        nodelist : str
            Node specification in shorthand form used by SLURM.

        Returns
        -------
        list[str]
            List of node hostnames.
        """
        # takes a NodeList as returned by SLURMs sacct
        # returns a list of single node hostnames
        # NOTE: This could also be done via "scontrol show hostname $nodelist"
        #       but then we would need to call scontrol here
        # NOTE: We expect nodelist to be either a string of the form
        # $hostnameprefix$num or $hostnameprefix[$num1,$num2,...,$numN]
        # or 'None assigned'
        if "[" not in nodelist:
            # it is '$hostnameprefix$num' or 'None assigned', return it
            return [nodelist]
        else:
            # it is '$hostnameprefix[$num1,$num2,...,$numN]'
            # make the string a list of single node hostnames
            hostnameprefix, nums = nodelist.split("[")
            nums = nums.rstrip("]")
            nums = nums.split(",")
            return [f"{hostnameprefix}{num}" for num in nums]

    def _parse_exitcode_from_slurm_state(self, slurm_state: str) -> typing.Union[None, int]:
        # TODO: use re module to match the text instead if iterating over the
        #       dict each time?!
        for key, val in _SLURM_STATE_TO_EXITCODE.items():
            if key in slurm_state:
                logger.debug(f"Parsed SLURM state {slurm_state} as {key}.")
                # this also recognizes `CANCELLED by ...` as CANCELLED
                return val
        # we should never finish the loop, it means we miss a slurm job state
        raise SlurmError("Could not find a matching exitcode for slurm state"
                         + f": {slurm_state}")

    # TODO: more _process_ functions?!
    #       exitcode? state?
    # TODO: do we want functions for state_to_exitcode/exitcode_from_state?
    #       ...currently we have all the state -> exitcode logic in SlurmProcess
    #       until we parse exitcodes from sacct here that probably makes sense?!

    def _node_fail_heuristic(self, jobid: str, parsed_exitcode: int,
                             slurm_state: str, nodelist: list[str]) -> None:
        """
        Implement node fail heuristic.

        Check if a job failed and if yes determine heuristically if it failed
        because of a node failure.
        Also call the respective functions to update counters for successfull
        and unsuccessfull job executions on each of the involved nodes.

        Parameters
        ----------
        jobid : str
            SLURM jobid of the job.
        parsed_exitcode : int
            Exitcode already parsed from slurm_state.
        slurm_state : str
            Full SLURM state string, used for more detailed failure analysis.
        nodelist : list[str]
            List of nodes associated with the job.
        """
        # Job/node fail heuristic
        if parsed_exitcode == 0:
            # all good
            self._note_job_success_on_nodes(nodelist=nodelist)
            logger.debug("Node fail heuristic noted successful job with id "
                         + f"{jobid} on nodes {nodelist}.")
        elif parsed_exitcode != 0:
            log_str = ("Node fail heuristic noted unsuccessful job with id "
                       + f"{jobid} on nodes {nodelist}.")
            if "fail" in slurm_state.lower():
                # NOTE: only some job failures are node failures
                # this should catch 'FAILED', 'NODE_FAIL' and 'BOOT_FAIL'
                # but excludes 'CANCELLED', 'DEADLINE', 'OUT_OF_MEMORY',
                # 'REVOKE' and 'TIMEOUT'
                # TODO: is this what we want?
                # I (hejung) think yes, the later 5 are quite probably not a
                # node failure but a code/user error
                logger.debug(log_str + " MARKING NODES AS POSSIBLY BROKEN.")
                self._note_job_fail_on_nodes(nodelist=nodelist)
            else:
                logger.debug(log_str + " Not marking nodes because the slurm "
                             + f"state ({slurm_state}) hints at code/user"
                             + " error and not node failure."
                             )

    # Bookkeeping functions for node fail heuristic, one for success updates
    # one for failure updates
    def _note_job_fail_on_nodes(self, nodelist: list[str]) -> None:
        logger.debug(f"Adding nodes {nodelist} to node fail counter.")
        for node in nodelist:
            self._node_job_fails[node] += 1
            if self._node_job_fails[node] >= self.num_fails_for_broken_node:
                # declare it broken
                logger.info(f"Adding node {node} to list of broken nodes.")
                if node not in self._broken_nodes:
                    self._broken_nodes.append(node)
                else:
                    logger.error(f"Node {node} already in broken node list.")
        # failsaves
        all_nodes = len(self._all_nodes)
        broken_nodes = len(self._broken_nodes)
        if broken_nodes >= all_nodes / 4:
            logger.error("We already declared 1/4 of the cluster as broken."
                         + "Houston, we might have a problem?")
            if broken_nodes >= all_nodes / 2:
                logger.error("In fact we declared 1/2 of the cluster as broken."
                             + "Houston, we *do* have a problem!")
                if broken_nodes >= all_nodes * 0.75:
                    raise RuntimeError("Houston? 3/4 of the cluster is broken?")

    def _note_job_success_on_nodes(self, nodelist: list[str]) -> None:
        logger.debug(f"Adding nodes {nodelist} to node success counter.")
        for node in nodelist:
            if node not in self._node_job_fails:
                # only count successes for nodes on which we have seen failures
                continue
            self._node_job_successes[node] += 1
            if self._node_job_successes[node] >= self.success_to_fail_ratio:
                # we seen enough success to decrease the fail count by one
                # zero the success counter and see if we decrease fail count
                # Note that the fail count must not become negative!
                self._node_job_successes[node] = 0
                log_str = (f"Seen {self.success_to_fail_ratio} successful "
                           + f"jobs on node {node}. ")
                logger.debug(log_str + "Zeroing success counter.")
                if self._node_job_fails[node] > 0:
                    # we have seen failures previously, so decrease counter
                    # but do not go below 0 and also do not delete it, i.e.
                    # keep counting successes
                    self._node_job_fails[node] -= 1
                    logger.info(log_str + "Decreasing node fail count by one.")


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
        when using `self.wait`.
    """

    # use same instance of class for all SlurmProcess instances
    try:
        _slurm_cluster_mediator = SlurmClusterMediator()
    except ValueError:
        _slurm_cluster_mediator = None
        # we raise a ValueError if sacct/sinfo are not available
        logger.warning("Could not initialize SLURM cluster handling. "
                       + "If you are sure SLURM (sinfo/sacct/etc) is available"
                       + " try calling `asyncmd.config.set_slurm_settings()`"
                       + " with the appropriate arguments.")
    # we can not simply wait for the subprocess, since slurm exits directly
    # so we will sleep for this long between checks if slurm-job completed
    sleep_time = 30  # TODO: heuristic? dynamically adapt?
    # NOTE: no options to set/pass extra_args for sbatch:
    #       the only command line options for sbatch we allow will be contolled
    #       by us since cmd line options for sbatch take precendece over every-
    #       thing else. This will e.g. allow us to reliably control the output
    #       files and therefore enable to implement communicate(), i.e. parse
    #       stderr and stdout
    sbatch_executable = "sbatch"
    scancel_executable = "scancel"

    def __init__(self, jobname: str, sbatch_script: str, workdir: str,
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
        workdir : str
            Absolute or relative path to use as working directory.

        Raises
        ------
        TypeError
            If the value set via init kwarg for a attribute does not match the
            default/original type for that attribute.
        """
        # we expect sbatch_script to be a path to a file
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
        # this either checks for our defaults or whatever we just set via kwargs
        ensure_executable_available(self.sbatch_executable)
        ensure_executable_available(self.scancel_executable)
        self.jobname = jobname
        # TODO/FIXME: do we want sbatch_script to be relative to wdir?
        #             (currently it is relative to current dir when creating
        #              the slurmprocess)
        self.sbatch_script = os.path.abspath(sbatch_script)
        # TODO: default to current dir when creating?
        self.workdir = os.path.abspath(workdir)
        self._jobid = None
        self._jobinfo = {}  # dict with jobinfo cached from slurm cluster mediator

    @property
    def slurm_cluster_mediator(self) -> SlurmClusterMediator:
        if self._slurm_cluster_mediator is None:
            raise RuntimeError("SLURM monitoring not initialized. Please call"
                               + "`asyncmd.config.set_slurm_settings()`"
                               + " with appropriate arguments.")
        else:
            return self._slurm_cluster_mediator

    async def submit(self, stdin: typing.Optional[str] = None) -> None:
        """
        Submit the job via sbatch.

        Parameters
        ----------
        stdin : str or None
            If given it is interpreted as a file to which we connect the batch
            scripts stdin via sbatchs ``--input`` option. This enables sending
            data to the processes stdin via :meth:`communicate`.
            Note that if it is desired to send data to the process the process
            has to be submited with stdin.

        Raises
        ------
        RuntimeError
            If the job has already been submitted.
        SlurmSubmissionError
            If something goes wrong during the submission with sbatch.
        """
        if self._jobid is not None:
            raise RuntimeError(f"Already monitoring job with id {self._jobid}.")
        sbatch_cmd = f"{self.sbatch_executable}"
        sbatch_cmd += f" --job-name={self.jobname}"
        # FIXME/TODO: does this work for job-arrays?
        #             (probably not, but do we care?)
        sbatch_cmd += f" --output=./{self.jobname}.out.%j"
        sbatch_cmd += f" --error=./{self.jobname}.err.%j"
        # keep a ref to the stdin value, we need it in communicate
        self._stdin = stdin
        if stdin is not None:
            # TODO: do we need to check if the file exists or that the location
            #       is writeable?
            sbatch_cmd += f" --input=./{stdin}"
        # I (hejung) think we dont need the semaphore here
        #async with _SEMAPHORES["SLURM_CLUSTER_MEDIATOR"]:
        broken_nodes = self.slurm_cluster_mediator.broken_nodes
        if len(broken_nodes) > 0:
            sbatch_cmd += f" --exclude={','.join(broken_nodes)}"
        sbatch_cmd += f" --parsable {self.sbatch_script}"
        logger.debug(f"About to execute sbatch_cmd {sbatch_cmd}.")
        # 3 file descriptors: stdin,stdout,stderr
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        try:
            sbatch_proc = await asyncio.subprocess.create_subprocess_exec(
                                                *shlex.split(sbatch_cmd),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=self.workdir,
                                                close_fds=True,
                                                                          )
            stdout, stderr = await sbatch_proc.communicate()
            sbatch_return = stdout.decode()
        finally:
            # and put the three back into the semaphore
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
        # only jobid (and possibly clustername) returned, semikolon to separate
        logger.debug(f"sbatch returned stdout: {sbatch_return}, "
                     + f"stderr: {stderr.decode()}.")
        jobid = sbatch_return.split(";")[0].strip()
        # make sure jobid is an int/ can be cast as one
        err = False
        try:
            jobid_int = int(jobid)
        except ValueError:
            # can not cast to int, so probably something went wrong submitting
            err = True
        else:
            if str(jobid_int) != jobid:
                err = True
        if err:
            raise SlurmSubmissionError("Could not submit SLURM job."
                                       + f" sbatch stdout: {sbatch_return} \n"
                                       + f" sbatch stderr: {stderr.decode()}."
                                       )
        logger.info(f"Submited SLURM job with jobid {jobid}.")
        self._jobid = jobid
        # I (hejung) think we dont need the semaphore for non-async calls like here(?)
        #async with _SEMAPHORES["SLURM_CLUSTER_MEDIATOR"]:
        self.slurm_cluster_mediator.monitor_register_job(jobid=jobid)
        # get jobinfo (these will probably just be the defaults but at
        #  least this is a dict with the rigth keys...)
        await self._update_sacct_jobinfo()

    @property
    def slurm_jobid(self) -> typing.Union[str, None]:
        return self._jobid

    @property
    def nodes(self) -> typing.Union["list[str]", None]:
        return self._jobinfo.get("nodelist", None)

    @property
    def slurm_job_state(self) -> typing.Union[str, None]:
        return self._jobinfo.get("state", None)

    @property
    def returncode(self) -> typing.Union[int, None]:
        if self._jobid is None:
            return None
        return self._jobinfo.get("parsed_exitcode", None)

    async def _update_sacct_jobinfo(self) -> None:
        # I (hejung) think we dont need the SLURM_CLUSTER_MEDIATOR semaphore at all?!
        # i.e. I dont think we need it here as the cluster mediator limits
        # the call frequency for sacct updates and is the same for all
        # SlurmProcess instances
        #async with _SEMAPHORES["SLURM_CLUSTER_MEDIATOR"]:
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
        if self._jobid is None:
            # make sure we can only wait after submitting, otherwise we would
            # wait indefinitively if we call wait() before submit()
            raise RuntimeError("Can only wait for submitted SLURM jobs with "
                               + "known jobid. Did you ever submit the job?")
        while self.returncode is None:
            await asyncio.sleep(self.sleep_time)
            await self._update_sacct_jobinfo()  # update local cached jobinfo
        #async with _SEMAPHORES["SLURM_CLUSTER_MEDIATOR"]:
        self.slurm_cluster_mediator.monitor_remove_job(jobid=self.slurm_jobid)
        return self.returncode

    async def communicate(self, input : typing.Optional[bytes] = None) -> tuple[bytes, bytes]:
        """
        Interact with process. Optionally send data to the process.
        Wait for the process to finish, then read from stdout and stderr (files)
        and return the data.

        Parameters
        ----------
        input : bytes or None, optional
            The input data to send to the process, by default None.
            Note that you an only send data to processes created/submited with
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
        if self._jobid is None:
            # make sure we can only wait after submitting, otherwise we would
            # wait indefinitively if we call wait() before submit()
            raise RuntimeError("Can only wait for submitted SLURM jobs with "
                               + "known jobid. Did you ever submit the job?")
        if input is not None:
            if self._stdin is None:
                # make sure we have a stdin file if we have input to write
                raise ValueError("Can only send input to a SlurmProcess "
                                 + "created/submited with stdin (file) given.")
            # write the given input to stdin file
            async with aiofiles.open(os.path.join(self.workdir, f"{self._stdin}"),
                                     "wb",
                                     ) as f:
                await f.write(input)
        # NOTE: wait makes sure we deregister the job from monitoring
        #       we use it here to make sure we read thr full data from stdout, stderr
        #       *after* the process is done to emulate the behavior of asyncio
        #       there it is 1.) read until EOF is reached 2.) wait for the proc
        returncode = await self.wait()
        # we read them in binary mode to get bytes objects back, this way they
        # behave like the bytes objects returned by asyncio.subprocess
        async with aiofiles.open(
            os.path.join(self.workdir, f"{self.jobname}.out.{self.slurm_jobid}"),
            "rb"
                                 ) as f:
            stdout = await f.read()
        async with aiofiles.open(
            os.path.join(self.workdir, f"{self.jobname}.err.{self.slurm_jobid}"),
            "rb"
                                 ) as f:
            stderr = await f.read()
        return stdout, stderr

    def send_signal(self, signal):
        # TODO: write this! (if we actually need it?)
        #       [should be doable via scancel, which can send signals to jobs]
        #       [could maybe also work using scontrol
        #        (which makes the state change know to slumr demon)]
        raise NotImplementedError

    def terminate(self) -> None:
        """
        Terminate (cancel) the underlying SLURM job.

        Raises
        ------
        SlurmCancelationError
            If scancel has non-zero returncode.
        RuntimeError
            If no jobid is known, e.g. because the job was never submitted.
        """
        if self._jobid is not None:
            scancel_cmd = f"{self.scancel_executable} {self._jobid}"
            # TODO: parse/check output to make sure scancel went as expected?!
            try:
                scancel_out = subprocess.check_output(shlex.split(scancel_cmd),
                                                      text=True)
            except subprocess.CalledProcessError as e:
                raise SlurmCancelationError(
                        "Something went wrong canceling the slurm job "
                        + f"{self._jobid}. scancel had exitcode {e.returncode}"
                        + f" and output {e.output}."
                        )
            # if we got until here the job is successfuly canceled....
            logger.debug(f"Canceled SLURM job with jobid {self.slurm_jobid}."
                         + f"scancel returned {scancel_out}.")
            # remove the job from the monitoring
            self.slurm_cluster_mediator.monitor_remove_job(jobid=self._jobid)
        else:
            # we probably never submitted the job?
            raise RuntimeError("self.jobid is not set, can not cancel a job "
                               + "with unknown jobid. Did you ever submit it?")

    def kill(self) -> None:
        """Alias for :meth:`terminate`."""
        self.terminate()


async def create_slurmprocess_submit(jobname: str,
                                     sbatch_script: str,
                                     workdir: str,
                                     stdin: typing.Optional[str] = None,
                                     **kwargs,
                                     ):
    """
    Create and submit a SlurmProcess.

    All arguments are directly passed trough to :meth:`SlurmProcess.__init__`
    and :meth:`SlurmProcess.submit`.

    Parameters
    ----------
    jobname : str
        SLURM jobname (``--job-name``).
    sbatch_script : str
        Absolute or relative path to a SLURM submission script.
    workdir : str
        Absolute or relative path to use as working directory.
    stdin : str or None
        If given it is interpreted as a file to which we connect the batch
        scripts stdin via sbatchs ``--input`` option. This enables sending
        data to the processes stdin via :meth:`communicate`.
        Note that if it is desired to send data to the process the process
        has to be submited with stdin.

    Returns
    -------
    SlurmProcess
        The submitted slurm process instance.
    """
    proc = SlurmProcess(jobname=jobname, sbatch_script=sbatch_script,
                        workdir=workdir, **kwargs)
    await proc.submit(stdin=stdin)
    return proc


def set_slurm_settings(sinfo_executable: str = "sinfo",
                       sacct_executable: str = "sacct",
                       sbatch_executable: str = "sbatch",
                       scancel_executable: str = "scancel",
                       min_time_between_sacct_calls: int = 10,
                       num_fails_for_broken_node: int = 3,
                       success_to_fail_ratio: int = 50
                       ) -> None:
    """
    (Re) initialize all settings relevant for SLURM job control.

    Call this function if you want to change e.g. the path/name of SLURM
    executables. Note that this is a conviencence function to set all SLURM
    settings in one central place, you could also set each setting seperately
    in the `SlurmProcess` and `SlurmClusterMediator` classes.

    Parameters
    ----------
    sinfo_executable : str, optional
        Name of path to the sinfo executable, by default "sinfo".
    sacct_executable : str, optional
        Name or path to the sacct executable, by default "sacct".
    sbatch_executable : str, optional
        Name or path to the sbatch executable, by default "sbatch".
    scancel_executable : str, optional
        Name or path to the scancel executable, by default "scancel".
    min_time_between_sacct_calls : int, optional
        Minimum time (in seconds) between subsequent sacct calls,
        by default 10.
    num_fails_for_broken_node : int, optional
        Number of failed jobs we need to observe per node before declaring it
        to be broken (and not submitting any more jobs to it), by default 3.
    success_to_fail_ratio : int, optional
        Number of successful jobs we need to observe per node to decrease the
        failed job counter by one, by default 50.
    """
    global SlurmProcess
    SlurmProcess._slurm_cluster_mediator = SlurmClusterMediator(
                    sinfo_executable=sinfo_executable,
                    sacct_executable=sacct_executable,
                    min_time_between_sacct_calls=min_time_between_sacct_calls,
                    num_fails_for_broken_node=num_fails_for_broken_node,
                    success_to_fail_ratio=success_to_fail_ratio
                                                                )
    SlurmProcess.sbatch_executable = sbatch_executable
    SlurmProcess.scancel_executable = scancel_executable
