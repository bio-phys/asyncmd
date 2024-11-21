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
import asyncio
import collections
import logging
import re
import shlex
import subprocess
import time
import typing
import os
import aiofiles
import aiofiles.os

from .tools import (ensure_executable_available,
                    remove_file_if_exist_async,
                    remove_file_if_exist,
                    )
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
# NOTE on error codes:
#      we return:
#       - None if the job has not finished
#       - 0 if it completed successfully
#       - 1 if the job failed (probably) due to user error (or we dont know)
#       - 2 if the job failed (almost certainly) due to cluster/node-issues as
#         recognized/detected by slurm
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
    "NODE_FAIL": 2,
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
    exclude_nodes : list[str]
        List of nodes to exclude in job submissions.

    """

    sinfo_executable = "sinfo"
    sacct_executable = "sacct"
    # wait for at least 5 s between two sacct calls
    min_time_between_sacct_calls = 5
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
        self._exclude_nodes = []
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
        # this either checks for our defaults or whatever we just set via kwargs
        self.sacct_executable = ensure_executable_available(self.sacct_executable)
        self.sinfo_executable = ensure_executable_available(self.sinfo_executable)
        self._node_job_fails = collections.Counter()
        self._node_job_successes = collections.Counter()
        self._all_nodes = self.list_all_nodes()
        self._jobids = []  # list of jobids of jobs we know about
        self._jobids_sacct = []  # list of jobids we monitor actively via sacct
        # we will store the info about jobs in a dict keys are jobids
        # values are dicts with key queried option and value the (parsed)
        # return value
        # currently queried options are: state, exitcode and nodelist
        self._jobinfo = {}
        self._last_sacct_call = 0  # make sure we dont call sacct too often
        # make sure we can only call sacct once at a time
        # (since there is only one ClusterMediator at a time we can create
        #  the semaphore here in __init__)
        self._sacct_semaphore = asyncio.BoundedSemaphore(1)
        self._build_regexps()

    def _build_regexps(self):
        # first build the regexps used to match slurmstates to assign exitcodes
        regexp_strings = {}
        for state, e_code in _SLURM_STATE_TO_EXITCODE.items():
            try:
                # get previous string and add "or" delimiter
                cur_str = regexp_strings[e_code]
                cur_str += r"|"
            except KeyError:
                # nothing yet, so no "or" delimiter needed
                cur_str = r""
            # add the state (and we do not care if something is before or after it)
            # (This is needed to also get e.g. "CANCELED by ..." as "CANCELED")
            cur_str += rf".*{state}.*"
            regexp_strings[e_code] = cur_str
        # now make the regexps
        self._ecode_for_slurmstate_regexps = {
                            e_code: re.compile(regexp_str,
                                               flags=re.IGNORECASE,
                                               )
                            for e_code, regexp_str in regexp_strings.items()
                                               }
        # build the regexp used to match and get the main-step lines from sacct
        # output
        self._match_mainstep_line_regexp = re.compile(
            r"""
            ^\d+  # the jobid at start of the line (but only the non-substeps)
            \|\|\|\|  # the (first) separator (we set 4 "|" as separator)
            .*?  # everything until the next separator (non-greedy), i.e. state
            \|\|\|\|  # the second separator
            .*?  # exitcode
            \|\|\|\|  # third separator
            .*?  # nodes
            \|\|\|\|  # final (fourth) separator
            """,
            flags=re.VERBOSE | re.MULTILINE | re.DOTALL,
                                                      )

    @property
    def exclude_nodes(self) -> "list[str]":
        """Return a list with all nodes excluded from job submissions."""
        return self._exclude_nodes.copy()

    @exclude_nodes.setter
    def exclude_nodes(self, val : typing.Union[list[str], None]):
        if val is None:
            val = []
        self._exclude_nodes = val

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
            # we use a dict with defaults to make sure that we get a 'PENDING'
            # for new jobs because this will make us check again in a bit
            # (sometimes there is a lag between submission and the appearance
            #  of the job in sacct output)
            self._jobinfo[jobid] = {"state": "PENDING",
                                    "exitcode": None,
                                    "parsed_exitcode": None,
                                    "nodelist": [],
                                    }
            # add the jobid to the sacct calls only **after** we set the defaults
            self._jobids.append(jobid)
            self._jobids_sacct.append(jobid)
            logger.debug("Registered job with id %s for sacct monitoring.",
                         jobid,
                         )
        else:
            logger.info("Job with id %s already registered for "
                        "monitoring. Not adding it again.",
                        jobid,
                        )

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
            logger.debug("Removed job with id %s from sacct monitoring.",
                         jobid,
                         )
        else:
            logger.info("Not monitoring job with id %s, not removing.",
                        jobid,
                        )

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
        async with self._sacct_semaphore:
            if (time.time() - self._last_sacct_call
                    > self.min_time_between_sacct_calls):
                # either we never called sacct or at least not in the recent past
                # so update cached jobinfo and save the new time
                await self._update_cached_jobinfo()
                logger.debug("Updated cached jobinfo.")
                # we update the time last, i.e. we count the time we need to
                # parse the sacct output into the time-delay
                self._last_sacct_call = time.time()

        return self._jobinfo[jobid].copy()

    async def _update_cached_jobinfo(self) -> None:
        """Call sacct and update cached info for all jobids we know about."""
        sacct_cmd = f"{self.sacct_executable} --noheader"
        # query only for the specific job we are running
        sacct_cmd += f" -j {','.join(self._jobids_sacct)}"
        sacct_cmd += " -o jobid,state,exitcode,nodelist"
        # parsable does print the separator at the end of each line
        sacct_cmd += " --parsable"
        sacct_cmd += " --delimiter='||||'"  # use 4 "|" as separator char(s)
        # 3 file descriptors: stdin,stdout,stderr
        # (note that one semaphore counts for 3 files!)
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
        except asyncio.CancelledError as e:
            sacct_proc.kill()
            raise e from None
        finally:
            # and put the three back into the semaphore
            _SEMAPHORES["MAX_FILES_OPEN"].release()
        # only jobid (and possibly clustername) returned, semikolon to separate
        logger.debug("sacct returned %s.", sacct_return)
        # sacct returns one line per substep, we only care for the whole job
        # so our regexp checks explictly for jobid only
        # (the substeps have .$NUM suffixes)
        for match in self._match_mainstep_line_regexp.finditer(sacct_return):
            splits = match.group().split("||||")
            if len(splits) != 5:
                # basic sanity check that everything went alright parsing,
                # i.e. that we got the number of fields we expect
                logger.error("Could not parse sacct output line due to "
                             "unexpected number of fields. The line was: %s",
                             match.group())
            else:
                # the last is the empty string after the final/fourth separator
                jobid, state, exitcode, nodelist, _ = splits
                # parse returns (remove spaces, etc.) and put them in cache
                jobid = jobid.strip()
                try:
                    last_seen_state = self._jobinfo[jobid]["state"]
                except KeyError:
                    # this can happen if we remove the job from monitoring
                    # after the sacct call but before parsing of sacct_return
                    # (then the _jobinfo dict will not contain the job anymore
                    #  and we get the KeyError from the jobid)
                    # we go to the next jobid as we are not monitoring this one
                    # TODO: do we want/need to log this?!
                    continue
                else:
                    if last_seen_state == state:
                        # we only process nodelist and update jobinfo when
                        # necessary, i.e. if the slurm_state changed
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
                    logger.debug("Parsed slurm state %s for job %s"
                                 " as returncode %s. Removing job"
                                 "from sacct calls because its state will"
                                 " not change anymore.",
                                 state, jobid, parsed_ec,
                                 )
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

    def _parse_exitcode_from_slurm_state(self,
                                         slurm_state: str,
                                         ) -> typing.Union[None, int]:
        for ecode, regexp in self._ecode_for_slurmstate_regexps.items():
            if regexp.search(slurm_state):
                # regexp matches the given slurm_state
                logger.debug("Parsed SLURM state %s as exitcode %d.",
                             slurm_state, ecode,
                             )
                return ecode
        # we should never finish the loop, it means we miss a slurm job state
        raise SlurmError("Could not find a matching exitcode for slurm state"
                         + f": {slurm_state}")

    # TODO: more _process_ functions?!
    #       exitcode? state?

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
                         "%s on nodes %s.",
                         jobid, nodelist,
                         )
        elif parsed_exitcode != 0:
            log_str = ("Node fail heuristic noted unsuccessful job with id "
                       "%s on nodes %s.")
            log_args = [jobid, nodelist]
            if "fail" in slurm_state.lower():
                # NOTE: only some job failures are node failures
                # this should catch 'FAILED', 'NODE_FAIL' and 'BOOT_FAIL'
                # but excludes 'CANCELLED', 'DEADLINE', 'OUT_OF_MEMORY',
                # 'REVOKE' and 'TIMEOUT'
                # TODO: is this what we want?
                # I (hejung) think yes, the later 5 are quite probably not a
                # node failure but a code/user error
                log_str += " MARKING NODES AS POSSIBLY BROKEN."
                logger.debug(log_str, *log_args)
                self._note_job_fail_on_nodes(nodelist=nodelist)
            else:
                log_str += (" Not marking nodes because the slurm "
                            "state (%s) hints at code/user"
                            " error and not node failure.")
                log_args += [slurm_state]
                logger.debug(log_str, *log_args)

    # Bookkeeping functions for node fail heuristic, one for success updates
    # one for failure updates
    def _note_job_fail_on_nodes(self, nodelist: list[str]) -> None:
        logger.debug("Adding nodes %s to node fail counter.", nodelist)
        for node in nodelist:
            self._node_job_fails[node] += 1
            if self._node_job_fails[node] >= self.num_fails_for_broken_node:
                # declare it broken
                logger.info("Adding node %s to list of excluded nodes.", node)
                if node not in self._exclude_nodes:
                    self._exclude_nodes.append(node)
                else:
                    logger.error("Node %s already in exclude node list.", node)
        # failsaves
        all_nodes = len(self._all_nodes)
        exclude_nodes = len(self._exclude_nodes)
        if exclude_nodes >= all_nodes / 4:
            logger.error("We already declared 1/4 of the cluster as broken."
                         + "Houston, we might have a problem?")
            if exclude_nodes >= all_nodes / 2:
                logger.error("In fact we declared 1/2 of the cluster as broken."
                             + "Houston, we *do* have a problem!")
                if exclude_nodes >= all_nodes * 0.75:
                    raise RuntimeError("Houston? 3/4 of the cluster is broken?")

    def _note_job_success_on_nodes(self, nodelist: list[str]) -> None:
        logger.debug("Adding nodes %s to node success counter.", nodelist)
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
                logger.debug("Seen %s successful jobs on node %s. "
                             "Zeroing success counter.",
                             self._node_job_successes[node], node,
                             )
                if self._node_job_fails[node] > 0:
                    # we have seen failures previously, so decrease counter
                    # but do not go below 0 and also do not delete it, i.e.
                    # keep counting successes
                    self._node_job_fails[node] -= 1
                    logger.info("Decreased node fail count by one for node %s,"
                                "node now has %s recorded failures.",
                                node, self._node_job_fails[node],
                                )


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
                       "If you are sure SLURM (sinfo/sacct/etc) is available"
                       " try calling `asyncmd.config.set_slurm_settings()`"
                       " with the appropriate arguments.")
    # we can not simply wait for the subprocess, since slurm exits directly
    # so we will sleep for this long between checks if slurm-job completed
    sleep_time = 15  # TODO: heuristic? dynamically adapt?
    # NOTE: no options to set/pass extra_args for sbatch:
    #       the only command line options for sbatch we allow will be contolled
    #       by us since cmd line options for sbatch take precendece over every-
    #       thing else. This will e.g. allow us to reliably control the output
    #       files and therefore enable to implement communicate(), i.e. parse
    #       stderr and stdout
    sbatch_executable = "sbatch"
    scancel_executable = "scancel"

    def __init__(self, jobname: str, sbatch_script: str,
                 workdir: typing.Optional[str] = None,
                 time: typing.Optional[float] = None,
                 stdfiles_removal: str = "success",
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
        stdfiles_removal : str
            Whether to remove the stdout, stderr (and possibly stdin) files.
            Possible values are:

             - "success": remove on sucessful completion, i.e. zero returncode)
             - "no": never remove
             - "yes"/"always": remove on job completion independent of
               returncode and also when using :meth:`terminate`

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
            else:
                # not previously defined, so warn that we ignore it
                logger.warning("Ignoring unknown keyword-argument %s.", kwarg)
        # this either checks for our defaults or whatever we just set via kwargs
        ensure_executable_available(self.sbatch_executable)
        ensure_executable_available(self.scancel_executable)
        self.jobname = jobname
        # TODO/FIXME: do we want sbatch_script to be relative to wdir?
        #             (currently it is relative to current dir when creating
        #              the slurmprocess)
        self.sbatch_script = os.path.abspath(sbatch_script)
        # TODO: default to current dir when creating?
        if workdir is None:
            workdir = os.getcwd()
        self.workdir = os.path.abspath(workdir)
        self.time = time
        self.stdfiles_removal = stdfiles_removal
        self._jobid = None
        self._jobinfo = {}  # dict with jobinfo cached from slurm cluster mediator
        self._stdout_data = None
        self._stderr_data = None
        self._stdin = None

    @property
    def stdfiles_removal(self) -> str:
        """
        Whether/when we remove stdfiles created by SLURM.

        Can be one of "success", "no", "yes", "always", where "yes" and
        "always" are synomyms for always remove. "success" means remove
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

    @property
    def slurm_cluster_mediator(self) -> SlurmClusterMediator:
        """
        The (singleton) `SlurmClusterMediator` instance of this `SlurmProcess`.
        """
        if self._slurm_cluster_mediator is None:
            raise RuntimeError("SLURM monitoring not initialized. Please call"
                               + "`asyncmd.config.set_slurm_settings()`"
                               + " with appropriate arguments.")

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
        # set working directory for batch script to workdir
        sbatch_cmd += f" --chdir={self.workdir}"
        # FIXME/TODO: does this work for job-arrays?
        #             (probably not, but do we care?)
        sbatch_cmd += f" --output=./{self._stdout_name(use_slurm_symbols=True)}"
        sbatch_cmd += f" --error=./{self._stderr_name(use_slurm_symbols=True)}"
        if self.time is not None:
            timelimit = self.time * 60
            timelimit_min = int(timelimit)  # take only the full minutes
            timelimit_sec = round(60 * (timelimit - timelimit_min))
            timelimit_str = f"{timelimit_min}:{timelimit_sec}"
            sbatch_cmd += f" --time={timelimit_str}"
        # keep a ref to the stdin value, we need it in communicate
        self._stdin = stdin
        if stdin is not None:
            # TODO: do we need to check if the file exists or that the location
            #       is writeable?
            sbatch_cmd += f" --input=./{stdin}"
        # get the list of nodes we dont want to run on
        exclude_nodes = self.slurm_cluster_mediator.exclude_nodes
        if len(exclude_nodes) > 0:
            sbatch_cmd += f" --exclude={','.join(exclude_nodes)}"
        sbatch_cmd += f" --parsable {self.sbatch_script}"
        logger.debug("About to execute sbatch_cmd %s.", sbatch_cmd)
        # 3 file descriptors: stdin,stdout,stderr
        # Note: one semaphore counts for 3 open files!
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
        except asyncio.CancelledError as e:
            sbatch_proc.kill()
            raise e from None
        finally:
            _SEMAPHORES["MAX_FILES_OPEN"].release()
        # only jobid (and possibly clustername) returned, semikolon to separate
        logger.debug("sbatch returned stdout: %s, stderr: %s.",
                     sbatch_return, stderr.decode())
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
                                       + f" Exit code was: {sbatch_return} \n"
                                       + f"sbatch stdout: {stdout.decode()} \n"
                                       + f"sbatch stderr: {stderr.decode()} \n"
                                       )
        logger.info("Submited SLURM job with jobid %s.", jobid)
        self._jobid = jobid
        self.slurm_cluster_mediator.monitor_register_job(jobid=jobid)
        # get jobinfo (these will probably just be the defaults but at
        #  least this is a dict with the rigth keys...)
        await self._update_sacct_jobinfo()

    @property
    def slurm_jobid(self) -> typing.Union[str, None]:
        """The slurm jobid of this job."""
        return self._jobid

    @property
    def nodes(self) -> typing.Union["list[str]", None]:
        """The nodes this job runs on."""
        return self._jobinfo.get("nodelist", None)

    @property
    def slurm_job_state(self) -> typing.Union[str, None]:
        """The slurm jobstate of this job."""
        return self._jobinfo.get("state", None)

    @property
    def returncode(self) -> typing.Union[int, None]:
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
        async with _SEMAPHORES["MAX_FILES_OPEN"]:
            stdout_fname = os.path.join(
                                    self.workdir,
                                    self._stdout_name(use_slurm_symbols=False),
                                        )
            try:
                async with aiofiles.open(stdout_fname,"rb") as f:
                    stdout = await f.read()
            except FileNotFoundError:
                logger.warning("stdout file %s not found.", stdout_fname)
                stdout = bytes()
            stderr_fname = os.path.join(
                                    self.workdir,
                                    self._stderr_name(use_slurm_symbols=False),
                                        )
            try:
                async with aiofiles.open(stderr_fname, "rb") as f:
                    stderr = await f.read()
            except FileNotFoundError:
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
        if self._jobid is None:
            # make sure we can only wait after submitting, otherwise we would
            # wait indefinitively if we call wait() before submit()
            raise RuntimeError("Can only wait for submitted SLURM jobs with "
                               + "known jobid. Did you ever submit the job?")
        while self.returncode is None:
            await asyncio.sleep(self.sleep_time)
            await self._update_sacct_jobinfo()  # update local cached jobinfo
        self.slurm_cluster_mediator.monitor_remove_job(jobid=self.slurm_jobid)
        if (((self.returncode == 0) and (self._stdfiles_removal == "success"))
                or self._stdfiles_removal == "yes"
                or self._stdfiles_removal == "always"):
            # read them in and cache them so we can still call communicate()
            # to get the data later
            stdout, stderr = await self._read_stdfiles()
            await self._remove_stdfiles_async()
        return self.returncode

    async def communicate(self, input: typing.Optional[bytes] = None) -> tuple[bytes, bytes]:
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
        # order as in asyncio.subprocess, there it is:
        #   1.) write to stdin (optional)
        #   2.) read until EOF is reached
        #   3.) wait for the proc to finish
        # Note that we wait first because we can only start reading the
        # stdfiles when the job has at least started, so we just wait for it
        # and read the files at the end completely
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
            async with _SEMAPHORES["MAX_FILES_OPEN"]:
                async with aiofiles.open(os.path.join(self.workdir,
                                                      f"{self._stdin}"),
                                         "wb",
                                         ) as f:
                    await f.write(input)
        # NOTE: wait makes sure we deregister the job from monitoring and also
        #       removes the stdfiles as/if requested
        returncode = await self.wait()
        stdout, stderr = await self._read_stdfiles()
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
                        ) from e
            # if we got until here the job is successfuly canceled....
            logger.debug(f"Canceled SLURM job with jobid {self.slurm_jobid}."
                         + f"scancel returned {scancel_out}.")
            # remove the job from the monitoring
            self.slurm_cluster_mediator.monitor_remove_job(jobid=self._jobid)
            if (self._stdfiles_removal == "yes"
                    or self._stdfiles_removal == "always"):
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
                                     sbatch_script: str,
                                     workdir: str,
                                     time: typing.Optional[float] = None,
                                     stdfiles_removal: str = "success",
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
    time : float or None
        Timelimit for the job in hours. None will result in using the
        default as either specified in the sbatch script or the partition.
    stdfiles_removal : str
        Whether to remove the stdout, stderr (and possibly stdin) files.
        Possible values are:

         - "success": remove on sucessful completion, i.e. zero returncode)
         - "no": never remove
         - "yes"/"always": remove on job completion independent of
           returncode and also when using :meth:`terminate`

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
                        workdir=workdir, time=time,
                        stdfiles_removal=stdfiles_removal,
                        **kwargs)
    await proc.submit(stdin=stdin)
    return proc


def set_all_slurm_settings(sinfo_executable: str = "sinfo",
                           sacct_executable: str = "sacct",
                           sbatch_executable: str = "sbatch",
                           scancel_executable: str = "scancel",
                           min_time_between_sacct_calls: int = 10,
                           num_fails_for_broken_node: int = 3,
                           success_to_fail_ratio: int = 50,
                           exclude_nodes: typing.Optional[list[str]] = None,
                           ) -> None:
    """
    (Re) initialize all settings relevant for SLURM job control.

    Call this function if you want to change e.g. the path/name of SLURM
    executables. Note that this is a conviencence function to set all SLURM
    settings in one central place and all at once, i.e. calling this function
    will overwrite all previous settings.
    If this is not intended, have a look at the `set_slurm_settings` function
    which only changes the passed arguments or you can also set/modify each
    setting separately in the `SlurmProcess` and `SlurmClusterMediator` classes.

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
    exclude_nodes : list[str], optional
        List of nodes to exclude in job submissions, by default None, which
        results in no excluded nodes.
    """
    global SlurmProcess
    SlurmProcess._slurm_cluster_mediator = SlurmClusterMediator(
                    sinfo_executable=sinfo_executable,
                    sacct_executable=sacct_executable,
                    min_time_between_sacct_calls=min_time_between_sacct_calls,
                    num_fails_for_broken_node=num_fails_for_broken_node,
                    success_to_fail_ratio=success_to_fail_ratio,
                    exclude_nodes=exclude_nodes,
                                                                )
    SlurmProcess.sbatch_executable = sbatch_executable
    SlurmProcess.scancel_executable = scancel_executable


def set_slurm_settings(sinfo_executable: typing.Optional[str] = None,
                       sacct_executable: typing.Optional[str] = None,
                       sbatch_executable: typing.Optional[str] = None,
                       scancel_executable: typing.Optional[str] = None,
                       min_time_between_sacct_calls: typing.Optional[int] = None,
                       num_fails_for_broken_node: typing.Optional[int] = None,
                       success_to_fail_ratio: typing.Optional[int] = None,
                       exclude_nodes: typing.Optional[list[str]] = None,
                       ) -> None:
    """
    Set single or multiple settings relevant for SLURM job control.

    Call this function if you want to change e.g. the path/name of SLURM
    executables. This function only modifies thoose settings for which a value
    other than None is passed. See `set_all_slurm_settings` if you want to set/
    modify all slurm settings and/or reset them to their defaults.

    Parameters
    ----------
    sinfo_executable : str, optional
        Name of path to the sinfo executable, by default None.
    sacct_executable : str, optional
        Name or path to the sacct executable, by default None.
    sbatch_executable : str, optional
        Name or path to the sbatch executable, by default None.
    scancel_executable : str, optional
        Name or path to the scancel executable, by default None.
    min_time_between_sacct_calls : int, optional
        Minimum time (in seconds) between subsequent sacct calls,
        by default None.
    num_fails_for_broken_node : int, optional
        Number of failed jobs we need to observe per node before declaring it
        to be broken (and not submitting any more jobs to it), by default None.
    success_to_fail_ratio : int, optional
        Number of successful jobs we need to observe per node to decrease the
        failed job counter by one, by default None.
    exclude_nodes : list[str], optional
        List of nodes to exclude in job submissions, by default None, which
        results in no excluded nodes.
    """
    global SlurmProcess
    if sinfo_executable is not None:
        SlurmProcess._slurm_cluster_mediator.sinfo_executable = sinfo_executable
    if sacct_executable is not None:
        SlurmProcess._slurm_cluster_mediator.sacct_executable = sacct_executable
    if sbatch_executable is not None:
        SlurmProcess.sbatch_executable = sbatch_executable
    if scancel_executable is not None:
        SlurmProcess.scancel_executable = scancel_executable
    if min_time_between_sacct_calls is not None:
        SlurmProcess._slurm_cluster_mediator.min_time_between_sacct_calls = min_time_between_sacct_calls
    if num_fails_for_broken_node is not None:
        SlurmProcess._slurm_cluster_mediator.num_fails_for_broken_node = num_fails_for_broken_node
    if success_to_fail_ratio is not None:
        SlurmProcess._slurm_cluster_mediator.success_to_fail_ratio = success_to_fail_ratio
    if exclude_nodes is not None:
        SlurmProcess._slurm_cluster_mediator.exclude_nodes = exclude_nodes
