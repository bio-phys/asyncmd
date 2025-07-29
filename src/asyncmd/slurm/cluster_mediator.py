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
This file contains the implementation of the SlurmClusterMediator.

The SlurmClusterMediator is a singleton class (handling all sacct calls in a
coordinated fashion) for all SlurmProcess instances.
"""
import asyncio
import collections
import logging
import re
import shlex
import subprocess
import time
import typing

from .constants_and_errors import (SlurmError,
                                   SLURM_STATE_TO_EXITCODE as _SLURM_STATE_TO_EXITCODE,
                                   )
from ..tools import (ensure_executable_available,
                     attach_kwargs_to_object as _attach_kwargs_to_object,
                     )
from .._config import _SEMAPHORES, _SEMAPHORES_KEYS


logger = logging.getLogger(__name__)


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
    # wait for at least 5 s between two sacct calls
    min_time_between_sacct_calls = 5
    # NOTE: We track the number of failed/successful jobs associated with each
    #       node and use this information to decide if a node is broken
    # number of 'suspected fail' counts that a node needs to accumulate for us
    # to declare it broken
    num_fails_for_broken_node = 3
    # minimum number of successfully completed jobs we need to see on a node to
    # decrease the 'suspected fail' counter by one
    success_to_fail_ratio = 50
    # TODO /FIXME: currently we have some tolerance until a node is declared
    #             broken but as soon as it is broken it will stay that forever?!
    #             (here forever means until we reinitialize SlurmClusterMediator)

    def __init__(self, **kwargs) -> None:
        self._exclude_nodes: set[str] = set()
        # make it possible to set any attribute via kwargs
        # check the type for attributes with default values
        _attach_kwargs_to_object(obj=self, logger=logger, **kwargs)
        # this either checks for our defaults or whatever we just set via kwargs
        self.sacct_executable = ensure_executable_available(self.sacct_executable)
        self.sinfo_executable = ensure_executable_available(self.sinfo_executable)
        self._node_job_fails: dict[str, int] = collections.Counter()
        self._node_job_successes: dict[str, int] = collections.Counter()
        self._all_nodes = self.list_all_nodes()
        self._jobids: set[str] = set()  # set of jobids of jobs we know about
        self._jobids_sacct: set[str] = set()  # set of jobids we monitor actively via sacct
        # we will store the info about jobs in a dict keys are jobids
        # values are dicts with key queried option and value the (parsed)
        # return value
        # currently queried options are: state, exitcode and nodelist
        self._jobinfo: dict[str, dict] = {}
        self._last_sacct_call = 0.  # make sure we dont call sacct too often
        # make sure we can only call sacct once at a time
        # (since there is only one ClusterMediator at a time we can create
        #  the semaphore here in __init__)
        self._sacct_semaphore = asyncio.BoundedSemaphore(1)
        self._build_regexps()

    def _build_regexps(self) -> None:
        # first build the regexps used to match slurmstates to assign exitcodes
        regexp_strings = {}
        for state, e_code in _SLURM_STATE_TO_EXITCODE.items():
            try:
                # get previous string and add "or" delimiter
                cur_str = regexp_strings[e_code]
            except KeyError:
                # nothing yet, so no "or" delimiter needed
                cur_str = r""
            else:
                cur_str += r"|"
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
    def exclude_nodes(self) -> set[str]:
        """Return a set with all nodes excluded from job submissions."""
        return self._exclude_nodes.copy()

    @exclude_nodes.setter
    def exclude_nodes(self, val: set[str] | collections.abc.Iterable[str]) -> None:
        val = set(val)
        self._exclude_nodes = val

    def list_all_nodes(self) -> list[str]:
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
            self._jobids.add(jobid)
            self._jobids_sacct.add(jobid)
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
            except KeyError:
                pass  # already not actively monitored anymore
            logger.debug("Removed job with id %s from sacct monitoring.",
                         jobid,
                         )
        else:
            logger.info("Not monitoring job with id %s, not removing.",
                        jobid,
                        )

    async def get_info_for_job(self, jobid: str) -> dict[str, typing.Any]:
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
        await _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN].acquire()
        sacct_proc = await asyncio.create_subprocess_exec(
                                                *shlex.split(sacct_cmd),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                close_fds=True,
                                                )
        try:
            stdout, _ = await sacct_proc.communicate()
        except asyncio.CancelledError as e:
            sacct_proc.kill()
            raise e from None
        else:
            sacct_return = stdout.decode()
        finally:
            # and put the three back into the semaphore
            _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN].release()
        # only jobid (and possibly clustername) returned, semicolon to separate
        logger.debug("sacct returned %s.", sacct_return)
        # sacct returns one line per substep, we only care for the whole job
        # so our regexp checks explicitly for jobid only
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
                jobid, state, exitcode, nodelist_str, _ = splits
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
                    continue
                else:
                    if last_seen_state == state:
                        # we only process nodelist and update jobinfo when
                        # necessary, i.e. if the slurm_state changed
                        continue
                nodelist = self._process_nodelist(nodelist=nodelist_str)
                self._jobinfo[jobid]["nodelist"] = nodelist
                self._jobinfo[jobid]["exitcode"] = exitcode
                self._jobinfo[jobid]["state"] = state
                logger.debug("Extracted from sacct output: jobid %s, state %s, "
                             "exitcode %s and nodelist %s.",
                             jobid, state, exitcode, nodelist,
                             )
                parsed_ec = self._parse_exitcode_from_slurm_state(slurm_state=state)
                self._jobinfo[jobid]["parsed_exitcode"] = parsed_ec
                if parsed_ec is not None:
                    logger.debug("Parsed slurm state %s for job %s as "
                                 "returncode %s. Removing job from sacct calls "
                                 "because its state will not change anymore.",
                                 state, jobid, str(parsed_ec) if parsed_ec is not None
                                 else "not available",
                                 )
                    self._jobids_sacct.remove(jobid)
                    self._node_fail_heuristic(jobid=jobid,
                                              parsed_exitcode=parsed_ec,
                                              slurm_state=state,
                                              nodelist=nodelist,
                                              )

    def _process_nodelist(self, nodelist: str) -> list[str]:
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
        # it is '$hostnameprefix[$num1,$num2,...,$numN]'
        # make the string a list of single node hostnames
        hostnameprefix, nums = nodelist.split("[")
        nums = nums.rstrip("]")
        nums_list = nums.split(",")
        return [f"{hostnameprefix}{num}" for num in nums_list]

    def _parse_exitcode_from_slurm_state(self,
                                         slurm_state: str,
                                         ) -> None | int:
        for ecode, regexp in self._ecode_for_slurmstate_regexps.items():
            if regexp.search(slurm_state):
                # regexp matches the given slurm_state
                logger.debug("Parsed SLURM state %s as exitcode %s.",
                             slurm_state, str(ecode) if ecode is not None
                             else "not available",
                             )
                return ecode
        # we should never finish the loop, it means we miss a slurm job state
        raise SlurmError("Could not find a matching exitcode for slurm state"
                         + f": {slurm_state}")

    def _node_fail_heuristic(self, *, jobid: str, parsed_exitcode: int,
                             slurm_state: str, nodelist: list[str]) -> None:
        """
        Implement node fail heuristic.

        Check if a job failed and if yes determine heuristically if it failed
        because of a node failure.
        Also call the respective functions to update counters for successful
        and unsuccessful job executions on each of the involved nodes.

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
        if not parsed_exitcode:
            # all good
            self._note_job_success_on_nodes(nodelist=nodelist)
            logger.debug("Node fail heuristic noted successful job with id "
                         "%s on nodes %s.",
                         jobid, nodelist,
                         )
        elif parsed_exitcode:
            log_str = ("Node fail heuristic noted unsuccessful job with id "
                       "%s on nodes %s.")
            log_args = [jobid, nodelist]
            if "fail" in slurm_state.lower():
                # NOTE: only some job failures are node failures
                # this should catch 'FAILED', 'NODE_FAIL' and 'BOOT_FAIL'
                # but excludes 'CANCELLED', 'DEADLINE', 'OUT_OF_MEMORY',
                # 'REVOKE' and 'TIMEOUT'
                # I (hejung) think this is the list we want, the later 5 are
                # quite probably not a node failure but a code/user error
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
                    self._exclude_nodes.add(node)
                else:
                    logger.error("Node %s already in exclude node list.", node)
        # failsaves
        all_nodes = len(self._all_nodes)
        if (exclude_nodes := len(self._exclude_nodes)) >= all_nodes / 4:
            logger.error("We already declared 1/4 of the cluster as broken. "
                         "Houston, we might have a problem?")
            if exclude_nodes >= all_nodes / 2:
                logger.error("In fact we declared 1/2 of the cluster as broken. "
                             "Houston, we *do* have a problem!")
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
