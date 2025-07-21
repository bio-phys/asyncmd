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
This module contains classes for propagation of MD in segments and/or until a condition is reached.

Most notable are the InPartsTrajectoryPropagator and the ConditionalTrajectoryPropagator.
Also of interest especially in the context of path sampling could be the function
construct_tp_from_plus_and_minus_traj_segments, which can be used directly on the
output of a ConditionalTrajectoryPropagator to generate trajectories connecting two
fulfilled conditions.
"""
import asyncio
import collections.abc
import copy
import inspect
import logging
import os
import typing

import aiofiles
import aiofiles.os
import numpy as np

from ..mdengine import MDEngine
from ..tools import FlagChangeList, remove_file_if_exist_async
from ..utils import (get_all_file_parts, get_all_traj_parts,
                     nstout_from_mdconfig)
from .convert import TrajectoryConcatenator
from .functionwrapper import TrajectoryFunctionWrapper
from .trajectory import Trajectory


logger = logging.getLogger(__name__)


class MaxStepsReachedError(Exception):
    """
    Error raised when the simulation terminated because the (user-defined)
    maximum number of integration steps/trajectory frames has been reached.
    """


async def construct_tp_from_plus_and_minus_traj_segments(
                                *,
                                minus_trajs: "list[Trajectory]",
                                minus_state: int,
                                plus_trajs: "list[Trajectory]",
                                plus_state: int,
                                state_funcs: "list[TrajectoryFunctionWrapper]",
                                tra_out: str,
                                **concatenate_kwargs,
                                ) -> Trajectory:
    """
    Construct a continuous TP from plus and minus segments until states.

    This is used e.g. in TwoWay TPS or if you try to get TPs out of a committor
    simulation. Note, that this inverts all velocities on the minus segments.

    Parameters
    ----------
    minus_trajs : list[Trajectory]
        Trajectories that go "backward in time", these are going to be inverted
    minus_state : int
        Index (in ``state_funcs``) of the first state reached on minus trajs.
    plus_trajs : list[Trajectory]
        Trajectories that go "forward in time", these are taken as is.
    plus_state : int
        Index (in ``state_funcs``) of the first state reached on plus trajs.
    state_funcs : list[TrajectoryFunctionWrapper]
        List of wrapped state functions, the indices to the states must match
        the minus and plus state indices!
    tra_out : str
        Absolute or relative path to the output trajectory file.
    concatenate_kwargs : dict
        All (other) keyword arguments will be passed as is to the
        :meth:`TrajectoryConcatenator.concatenate` method. These include, e.g.,

        - struct_out : str, optional
            Absolute or relative path to the output structure file, if None we will
            use the structure file of the first minus_traj, by default None.
        - overwrite : bool, optional
            Whether we should overwrite tra_out if it exists, by default False.

    Returns
    -------
    Trajectory
        The constructed transition.
    """
    # first find the slices to concatenate
    # minus state first
    minus_state_vals = await asyncio.gather(*(state_funcs[minus_state](t)
                                              for t in minus_trajs)
                                            )
    part_lens = [len(v) for v in minus_state_vals]
    # make it into one long array
    minus_state_vals = np.concatenate(minus_state_vals, axis=0)
    # get the first frame in state (np.where always returns a tuple)
    frames_in_minus, = np.where(minus_state_vals)
    # get the first frame in minus state in minus_trajs, this will become the
    # first frame of the traj since we invert this part
    first_frame_in_minus = np.min(frames_in_minus)
    # I think this is overkill, i.e. we can always expect that
    # first frame in state is in last part?!
    # [this could potentially make this a bit shorter and maybe
    #  even a bit more readable :)]
    # But for now: better be save than sorry :)
    # find the first part in which minus state is reached, i.e. the last one
    # to take when constructing the TP
    last_part_idx = 0
    frame_sum = part_lens[last_part_idx]
    while first_frame_in_minus >= frame_sum:
        last_part_idx += 1
        frame_sum += part_lens[last_part_idx]
    # find the first frame in state (counting from start of last part to take)
    _first_frame_in_minus = (first_frame_in_minus
                             - sum(part_lens[:last_part_idx]))  # >= 0
    # now construct the slices and trajs list (backwards!)
    # the last/first part
    slices = [(_first_frame_in_minus, None, -1)]  # negative stride!
    trajs = [minus_trajs[last_part_idx]]
    # the ones we take fully (if any) [the range looks a bit strange
    # because we dont take last_part_index but include the zero as idx]
    slices += [(-1, None, -1) for _ in range(last_part_idx - 1, -1, -1)]
    trajs += [minus_trajs[i] for i in range(last_part_idx - 1, -1, -1)]

    # now plus trajectories, i.e. the part we put in positive stride
    plus_state_vals = await asyncio.gather(*(state_funcs[plus_state](t)
                                             for t in plus_trajs)
                                           )
    part_lens = [len(v) for v in plus_state_vals]
    # make it into one long array
    plus_state_vals = np.concatenate(plus_state_vals, axis=0)
    # get the first frame in state
    frames_in_plus, = np.where(plus_state_vals)
    first_frame_in_plus = np.min(frames_in_plus)
    # find the part
    last_part_idx = 0
    frame_sum = part_lens[last_part_idx]
    while first_frame_in_plus >= frame_sum:
        last_part_idx += 1
        frame_sum += part_lens[last_part_idx]
    # find the first frame in state (counting from start of last part)
    _first_frame_in_plus = (first_frame_in_plus
                            - sum(part_lens[:last_part_idx]))  # >= 0
    # construct the slices and add trajs to list (forward!)
    # NOTE: here we exclude the starting configuration, i.e. the SP,
    #       such that it is in the concatenated trajectory only once!
    #       (gromacs has the first frame in the trajectory)
    if last_part_idx > 0:
        # these are the trajectory segments we take completely
        # [this excludes last_part_idx so far]
        slices += [(1, None, 1)]
        trajs += [plus_trajs[0]]
        # these will be empty if last_part_idx < 2
        slices += [(0, None, 1) for _ in range(1, last_part_idx)]
        trajs += [plus_trajs[i] for i in range(1, last_part_idx)]
        # add last part (with the last frame as first frame in plus state)
        slices += [(0, _first_frame_in_plus + 1, 1)]
        trajs += [plus_trajs[last_part_idx]]
    else:
        # first and last part is the same, so exclude starting configuration
        # from the same segment that has the last frame as first frame in plus
        slices += [(1, _first_frame_in_plus + 1, 1)]
        trajs += [plus_trajs[last_part_idx]]
    # finally produce the concatenated path
    path_traj = await TrajectoryConcatenator().concatenate_async(
                                                    trajs=trajs,
                                                    slices=slices,
                                                    tra_out=tra_out,
                                                    **concatenate_kwargs
                                                                 )
    return path_traj


# pylint: disable-next=too-few-public-methods
class _TrajectoryPropagator:
    # (private) superclass for InPartsTrajectoryPropagator and
    # ConditionalTrajectoryPropagator,
    # here we keep the common functions shared between them, currently
    # this is only the file removal method and a bit of the init logic.
    def __init__(self, *,
                 engine_cls: type[MDEngine], engine_kwargs: dict[str, typing.Any],
                 walltime_per_part: float,
                 ) -> None:
        self.engine_cls = engine_cls
        self.engine_kwargs = engine_kwargs
        self.walltime_per_part = walltime_per_part

    async def remove_parts(self, workdir: str, deffnm: str, *,
                           file_endings_to_remove: list[str] | None = None,
                           remove_mda_offset_and_lock_files: bool = True,
                           remove_asyncmd_npz_caches: bool = True,
                           ) -> None:
        """
        Remove all ``$deffnm.part$num.$file_ending`` files for file_endings.

        Can be useful to clean the ``workdir`` from temporary files if e.g. only
        the concatenate trajectory is of interest (like in TPS).

        Parameters
        ----------
        workdir : str
            The directory to clean.
        deffnm : str
            The ``deffnm`` that the files we clean must have.
        file_endings_to_remove : list[str] | None, optional
            The strings in the list ``file_endings_to_remove`` indicate which
            file endings to remove.
            The 'special' string "trajectories" will be translated to the file
            ending of the trajectories the engine produces, i.e. ``engine.output_traj_type``.
            E.g. passing ``file_endings_to_remove=["trajectories", "log"]`` will
            result in removal of trajectory parts and the log files. If you add
            "edr" to the list we would also remove the edr files.
            By default, i.e., if None the list will be ["trajectories", "log"].
        remove_mda_offset_and_lock_files : bool, optional
            Whether to remove any (hidden) offset and lock files generated by
            MDAnalysis associated with the removed trajectory files (if they exist).
            By default True.
        remove_asyncmd_npz_caches : bool, optional
            Whether to remove any (hidden) npz cache files generated by asyncmd
            associated with the removed trajectory files (if they exist).
            By default True.
        """
        if file_endings_to_remove is None:
            file_endings_to_remove = ["trajectories", "log"]
        else:
            # copy the list so we dont mutate what we got passed
            file_endings_to_remove = copy.copy(file_endings_to_remove)
        if "trajectories" in file_endings_to_remove:
            # replace "trajectories" with the actual output traj type
            try:
                traj_type = self.engine_kwargs["output_traj_type"]
            except KeyError:
                # not in there so it will be the engine default
                traj_type = self.engine_cls.output_traj_type
            file_endings_to_remove.remove("trajectories")
            file_endings_to_remove += [traj_type]
        # now find and remove the files
        for ending in file_endings_to_remove:
            parts_to_remove = await get_all_file_parts(
                                                folder=workdir,
                                                deffnm=deffnm,
                                                file_ending=ending.lower(),
                                                       )
            # make sure we dont miss anything because we have different
            # capitalization
            if not parts_to_remove:
                parts_to_remove = await get_all_file_parts(
                                                    folder=workdir,
                                                    deffnm=deffnm,
                                                    file_ending=ending.upper(),
                                                           )
            await asyncio.gather(*(aiofiles.os.unlink(f)
                                   for f in parts_to_remove
                                   )
                                 )
            # TODO: address the note below?
            # NOTE: this is a bit hacky: we just try to remove the offset and
            #       lock files for every file we remove (since we do not know
            #       if the file we remove is a trajectory [and therefore
            #        potentially has corresponding offset and lock files] or if
            #       the file we remove is e.g. an edr which has no lock/offset)
            #       If we would know that we are removing a trajectory we could
            #       try the removal only there, however even by comparing with
            #       `traj_type` (from above) we can not be certain since users
            #       could remove the wildcard "trajectories" and replace it by
            #       their specific trajectory file ending, i.e. we would need
            #       to know all potential traj file endings to be sure
            if remove_mda_offset_and_lock_files or remove_asyncmd_npz_caches:
                # create list with head, tail filenames only if needed
                f_splits = [os.path.split(f) for f in parts_to_remove]
            if remove_mda_offset_and_lock_files:
                offset_lock_files_to_remove = [os.path.join(
                                                 f_head,
                                                 "." + f_tail + "_offsets.npz",
                                                            )
                                               for f_head, f_tail in f_splits
                                               ]
                offset_lock_files_to_remove += [os.path.join(
                                                  f_head,
                                                  "." + f_tail + "_offsets.lock",
                                                             )
                                                for f_head, f_tail in f_splits
                                                ]
            else:
                offset_lock_files_to_remove = []
            if remove_asyncmd_npz_caches:
                # NOTE: we do not try to remove the multipart traj caches since
                #       the Propagators only return non-multipart Trajectories
                npz_caches_to_remove = [os.path.join(
                                          f_head,
                                          "." + f_tail + "_asyncmd_cv_cache.npz",
                                                     )
                                        for f_head, f_tail in f_splits
                                        ]
            else:
                npz_caches_to_remove = []
            await asyncio.gather(*(remove_file_if_exist_async(f)
                                   for f in offset_lock_files_to_remove + npz_caches_to_remove
                                   )
                                 )


class InPartsTrajectoryPropagator(_TrajectoryPropagator):
    """
    Propagate a trajectory in parts of walltime until given number of steps.

    Useful to make full use of backfilling with short(ish) simulation jobs and
    also to run simulations that are longer than the timelimit.
    """
    def __init__(self, n_steps: int, *,
                 engine_cls: type[MDEngine], engine_kwargs: dict[str, typing.Any],
                 walltime_per_part: float,
                 ) -> None:
        """
        Initialize an `InPartTrajectoryPropagator`.

        Parameters
        ----------
        n_steps : int
            Number of integration steps to do in total.
        engine_cls : :class:`asyncmd.mdengine.MDEngine`
            Class of the MD engine to use, **uninitialized!**
        engine_kwargs : dict
            Dictionary of key word arguments to initialize the MD engine.
        walltime_per_part : float
            Walltime per trajectory segment, in hours.
        """
        super().__init__(engine_cls=engine_cls, engine_kwargs=engine_kwargs,
                         walltime_per_part=walltime_per_part)
        self.n_steps = n_steps

    async def propagate_and_concatenate(self,
                                        starting_configuration: Trajectory,
                                        workdir: str,
                                        deffnm: str, *,
                                        tra_out: str,
                                        continuation: bool = False,
                                        **concatenate_kwargs,
                                        ) -> Trajectory | None:
        """
        Chain :meth:`propagate` and :meth:`cut_and_concatenate` methods.

        Parameters
        ----------
        starting_configuration : :class:`asyncmd.Trajectory`
            The configuration (including momenta) to start MD from.
        workdir : str
            Absolute or relative path to the working directory.
        deffnm : str
            MD engine deffnm for trajectory parts and other files.
        tra_out : str
            Absolute or relative path for the concatenated output trajectory.
        continuation : bool, optional
            Whether we are continuing a previous MD run (with the same deffnm
            and working directory), by default False.
        concatenate_kwargs : dict
            All (other) keyword arguments will be passed as is to the
            :meth:`TrajectoryConcatenator.concatenate` method. These include, e.g.,

            - struct_out : str, optional
                Absolute or relative path to the output structure file, if None
                we will use the structure file of the first traj, by default None.
            - overwrite : bool, optional
                Whether we should overwrite tra_out if it exists, by default False.

        Returns
        -------
        traj_out : :class:`asyncmd.Trajectory`
            The concatenated output trajectory.
        """
        # this just chains propagate and cut_and_concatenate
        trajs = await self.propagate(
                                starting_configuration=starting_configuration,
                                workdir=workdir,
                                deffnm=deffnm,
                                continuation=continuation
                                     )
        full_traj = await self.cut_and_concatenate(trajs=trajs, tra_out=tra_out,
                                                   **concatenate_kwargs,
                                                   )
        return full_traj

    async def propagate(self,
                        starting_configuration: Trajectory,
                        workdir: str,
                        deffnm: str, *,
                        continuation: bool = False,
                        ) -> list[Trajectory]:
        """
        Propagate the trajectory until self.n_steps integration are done.

        Return a list of trajectory segments and the first condition fulfilled.

        Parameters
        ----------
        starting_configuration : :class:`asyncmd.Trajectory`
            The configuration (including momenta) to start MD from.
        workdir : str
            Absolute or relative path to the working directory.
        deffnm : str
            MD engine deffnm for trajectory parts and other files.
        continuation : bool, optional
            Whether we are continuing a previous MD run (with the same deffnm
            and working directory), by default False.
            Note that when doing continuations and n_steps is lower than the
            number of steps done already found in the directory, we still
            return all trajectory parts (i.e. potentially too much).
            :meth:`cut_and_concatenate` can return a trimmed subtrajectory.

        Returns
        -------
        traj_segments : list[Trajectory]
            List of trajectory (segments), ordered in time.
        """
        engine = self.engine_cls(**self.engine_kwargs)
        if continuation:
            # continuation: get all traj parts already done and continue from
            # there, i.e. append to the last traj part found
            trajs = await get_all_traj_parts(folder=workdir, deffnm=deffnm,
                                             engine=engine,
                                             )
            if len(trajs) > 0:
                # can only continue if we find the previous trajs
                await engine.prepare_from_files(workdir=workdir, deffnm=deffnm)
                if (step_counter := engine.steps_done) >= self.n_steps:
                    # already longer than what we want to do, bail out
                    return trajs
            else:
                # no previous trajs, prepare engine from scratch
                continuation = False
                logger.error("continuation=True, but we found no previous "
                             "trajectories. Setting continuation=False and "
                             "preparing the engine from scratch.")
        if not continuation:
            # no continuation, just prepare the engine from scratch
            await engine.prepare(
                        starting_configuration=starting_configuration,
                        workdir=workdir,
                        deffnm=deffnm,
                                )
            trajs = []
            step_counter = 0

        while step_counter < self.n_steps:
            traj = await engine.run_walltime(walltime=self.walltime_per_part,
                                             max_steps=self.n_steps,
                                             )
            step_counter = engine.steps_done
            trajs.append(traj)
        return trajs

    async def cut_and_concatenate(self,
                                  trajs: list[Trajectory],
                                  tra_out: str,
                                  **concatenate_kwargs,
                                  ) -> Trajectory | None:
        """
        Cut and concatenate the trajectory until it has length n_steps.

        Take a list of trajectory segments and form one continuous trajectory
        containing n_steps integration steps. The expected input
        is a list of trajectories, e.g. the output of the :meth:`propagate`
        method.
        Returns None if ``self.n_steps`` is zero.

        Parameters
        ----------
        trajs : list[Trajectory]
            Trajectory segments to cut and concatenate.
        tra_out : str
            Absolute or relative path for the concatenated output trajectory.
        concatenate_kwargs : dict
            All (other) keyword arguments will be passed as is to the
            :meth:`TrajectoryConcatenator.concatenate` method. These include, e.g.,

            - struct_out : str, optional
                Absolute or relative path to the output structure file, if None
                we will use the structure file of the first traj, by default None.
            - overwrite : bool, optional
                Whether we should overwrite tra_out if it exists, by default False.

        Returns
        -------
        traj_out : :class:`asyncmd.Trajectory`
            The concatenated output trajectory.

        Raises
        ------
        ValueError
            If the given trajectories are to short to create a trajectory
            containing n_steps integration steps
        """
        # trajs is a list of trajectories, e.g. the return of propagate
        # tra_out and overwrite are passed directly to the Concatenator
        if not self.n_steps:
            # no trajectories to concatenate, since self.n_steps=0
            # we return None
            return None
        if self.n_steps > trajs[-1].last_step:
            # not enough steps in trajectories
            raise ValueError("The given trajectories are too short (< self.n_steps).")
        if self.n_steps == trajs[-1].last_step:
            # all good, we just take all trajectory parts fully
            slices = [(0, None, 1) for _ in range(len(trajs))]
            last_part_idx = len(trajs) - 1
        else:
            # need to find the subtrajectory that contains the correct number
            # of integration steps
            # first find the part in which we go over n_steps
            last_part_idx = 0
            while self.n_steps > trajs[last_part_idx].last_step:
                last_part_idx += 1
            # find out how much frames to take on last part
            last_part_len_frames = len(trajs[last_part_idx])
            last_part_len_steps = (trajs[last_part_idx].last_step
                                   - trajs[last_part_idx].first_step)
            steps_per_frame = last_part_len_steps / last_part_len_frames
            frames_in_last_part = ((self.n_steps
                                    - trajs[last_part_idx].first_step
                                    )
                                   / steps_per_frame)
            log_str = ("Trajectories do not exactly contain n_steps integration steps. "
                       "Using a heuristic to find the correct last frame to include."
                       )
            if frames_in_last_part != (frames_in_last_part_int := round(frames_in_last_part)):
                log_str += (" Note that this heuristic might fail because n_steps"
                            " is not a multiple of the trajectory output frequency."
                            )
                logger.warning(log_str)
            else:
                logger.info(log_str)
            # build slices
            slices = [(0, None, 1) for _ in range(last_part_idx)]
            slices += [(0, frames_in_last_part_int + 1, 1)]

        # and concatenate
        full_traj = await TrajectoryConcatenator().concatenate_async(
                                   trajs=trajs[:last_part_idx + 1],
                                   slices=slices,
                                   tra_out=tra_out,
                                   **concatenate_kwargs
                                   )
        return full_traj


class ConditionalTrajectoryPropagator(_TrajectoryPropagator):
    """
    Propagate a trajectory until any of the given conditions is fulfilled.

    This class propagates the trajectory using a given MD engine (class) in
    small chunks (chunksize is determined by walltime_per_part) and checks
    after every chunk is done if any condition has been fulfilled.
    It then returns a list of trajectory parts and the index of the condition
    first fulfilled. It can also concatenate the parts into one trajectory,
    which then starts with the starting configuration and ends with the frame
    fulfilling the condition.

    Notes
    -----
    We assume that every condition function returns a list/ a 1d array with
    True or False for each frame, i.e. if we fulfill condition at any given
    frame.
    We assume non-overlapping conditions, i.e. a configuration can not fulfill
    two conditions at the same time, **it is the users responsibility to ensure
    that their conditions are sane**.
    """

    # NOTE: we assume that every condition function returns a list/ a 1d array
    #       with True/False for each frame, i.e. if we fulfill condition at
    #       any given frame
    # NOTE: we assume non-overlapping conditions, i.e. a configuration can not
    #       fulfill two conditions at the same time, it is the users
    #       responsibility to ensure that their conditions are sane

    # Note: max_steps and max_frames are mutually exclusive and this is enforced,
    # but pylint does not know that, so we tell it to not be mad for one arg more
    # pylint: disable-next=too-many-arguments
    def __init__(self, conditions, *,
                 engine_cls: type[MDEngine], engine_kwargs: dict[str, typing.Any],
                 walltime_per_part: float,
                 max_steps: int | None = None,
                 max_frames: int | None = None,
                 ):
        """
        Initialize a :class:`ConditionalTrajectoryPropagator`.

        Parameters
        ----------
        conditions : list[callable], usually list[TrajectoryFunctionWrapper]
            List of condition functions, usually wrapped function for
            asynchronous application, but can be any callable that takes a
            :class:`asyncmd.Trajectory` and returns an array of True and False
            values (one value per frame).
        engine_cls : :class:`asyncmd.mdengine.MDEngine`
            Class of the MD engine to use, **uninitialized!**
        engine_kwargs : dict
            Dictionary of key word arguments to initialize the MD engine.
        walltime_per_part : float
            Walltime per trajectory segment, in hours.
        max_steps : int, optional
            Maximum number of integration steps to do before stopping the
            simulation because it did not commit to any condition,
            by default None. Takes precedence over max_frames if both given.
        max_frames : int, optional
            Maximum number of frames to produce before stopping the simulation
            because it did not commit to any condition, by default None.

        Notes
        -----
        ``max_steps`` and ``max_frames`` are redundant since
        ``max_steps = max_frames * output_frequency``, if both are given
        max_steps takes precedence.
        """
        super().__init__(engine_cls=engine_cls, engine_kwargs=engine_kwargs,
                         walltime_per_part=walltime_per_part)
        self._conditions = FlagChangeList([])
        self._condition_func_is_coroutine: list[bool] = []
        self.conditions = conditions
        # find nstout
        try:
            traj_type = engine_kwargs["output_traj_type"]
        except KeyError:
            # not in there so it will be the engine default
            traj_type = engine_cls.output_traj_type
        nstout = nstout_from_mdconfig(mdconfig=engine_kwargs["mdconfig"],
                                      output_traj_type=traj_type)
        # sort out if we use max_frames or max_steps
        if max_frames is not None and max_steps is not None:
            logger.warning("Both max_steps and max_frames given. Note that "
                           "max_steps will take precedence.")
        if max_steps is not None:
            self.max_steps = max_steps
        elif max_frames is not None:
            self.max_steps = max_frames * nstout
        else:
            logger.info("Neither max_frames nor max_steps given. "
                        "Setting max_steps to infinity.")
            # this is a float but can be compared to ints
            self.max_steps = np.inf

    @property
    def conditions(self) -> FlagChangeList:
        """List of (wrapped) condition functions."""
        return self._conditions

    @conditions.setter
    def conditions(self, conditions: collections.abc.Sequence):
        if len(conditions) < 1:
            raise ValueError("Must supply at least one termination condition.")
        self._condition_func_is_coroutine = self._check_condition_funcs(
                                                        conditions=conditions
                                                                        )
        self._conditions = FlagChangeList(conditions)

    def _check_condition_funcs(self, conditions: collections.abc.Sequence,
                               ) -> list[bool]:
        # use asyncio.iscoroutinefunction to check the conditions
        condition_func_is_coroutine = [
                                (inspect.iscoroutinefunction(c)
                                 or inspect.iscoroutinefunction(c.__call__))
                                for c in conditions
                                       ]
        if not all(condition_func_is_coroutine):
            # and warn if it is not a coroutinefunction
            logger.warning(
                    "It is recommended to use coroutinefunctions for all "
                    "conditions. This can easily be achieved by wrapping any "
                    "function in a TrajectoryFunctionWrapper. All "
                    "non-coroutine condition functions will be blocking when "
                    "applied! ([c is coroutine for c in conditions] = %s)",
                    condition_func_is_coroutine
                           )
        return condition_func_is_coroutine

    async def propagate_and_concatenate(self,
                                        starting_configuration: Trajectory,
                                        workdir: str,
                                        deffnm: str, *,
                                        tra_out: str,
                                        continuation: bool = False,
                                        **concatenate_kwargs
                                        ) -> tuple[Trajectory, int]:
        """
        Chain :meth:`propagate` and :meth:`cut_and_concatenate` methods.

        Parameters
        ----------
        starting_configuration : Trajectory
            The configuration (including momenta) to start MD from.
        workdir : str
            Absolute or relative path to the working directory.
        deffnm : str
            MD engine deffnm for trajectory parts and other files.
        tra_out : str
            Absolute or relative path for the concatenated output trajectory.
        continuation : bool, optional
            Whether we are continuing a previous MD run (with the same deffnm
            and working directory), by default False.
        concatenate_kwargs : dict
            All (other) keyword arguments will be passed as is to the
            :meth:`TrajectoryConcatenator.concatenate` method. These include, e.g.,

            - struct_out : str, optional
                Absolute or relative path to the output structure file, if None
                we will use the structure file of the first traj, by default None.
            - overwrite : bool, optional
                Whether we should overwrite tra_out if it exists, by default False.

        Returns
        -------
        (traj_out, idx_of_condition_fulfilled) : (Trajectory, int)
            The concatenated output trajectory from starting configuration
            until the first condition is True and the index to the condition
            function in `conditions`.

        Raises
        ------
        MaxStepsReachedError
            When the defined maximum number of integration steps/trajectory
            frames has been reached in :meth:`propagate`.
        """
        # this just chains propagate and cut_and_concatenate
        # useful for committor simulations, for e.g. TPS one should try to
        # directly concatenate both directions to a full TP if possible
        trajs, first_condition_fulfilled = await self.propagate(
                                starting_configuration=starting_configuration,
                                workdir=workdir,
                                deffnm=deffnm,
                                continuation=continuation
                                )
        # NOTE: it should not matter too much speedwise that we recalculate
        #       the condition functions, they are expected to be wrapped funcs
        #       i.e. the second time we should just get the values from cache
        full_traj, first_condition_fulfilled = await self.cut_and_concatenate(
                                                        trajs=trajs,
                                                        tra_out=tra_out,
                                                        **concatenate_kwargs
                                                        )
        return full_traj, first_condition_fulfilled

    async def propagate(self,
                        starting_configuration: Trajectory,
                        workdir: str,
                        deffnm: str, *,
                        continuation: bool = False,
                        ) -> tuple[list[Trajectory], int]:
        """
        Propagate the trajectory until any condition is fulfilled.

        Return a list of trajectory segments and the first condition fulfilled.

        Parameters
        ----------
        starting_configuration : Trajectory
            The configuration (including momenta) to start MD from.
        workdir : str
            Absolute or relative path to the working directory.
        deffnm : str
            MD engine deffnm for trajectory parts and other files.
        continuation : bool, optional
            Whether we are continuing a previous MD run (with the same deffnm
            and working directory), by default False.

        Returns
        -------
        (traj_segments, idx_of_condition_fulfilled) : (list[Trajectory], int)
            List of trajectory (segments), the last entry is the one on which
            the first condition is fulfilled at some frame, the integer is the
            index to the condition function in ``conditions``.

        Raises
        ------
        MaxStepsReachedError
            When the defined maximum number of integration steps/trajectory
            frames has been reached.
        """
        # NOTE: currently this just returns a list of trajs + the condition
        #       fulfilled
        #       this feels a bit uncomfortable but avoids that we concatenate
        #       everything a quadrillion times when we use the results
        # check first if the start configuration is fulfilling any condition
        cond_vals = await self._condition_vals_for_traj(starting_configuration)
        if np.any(cond_vals):
            conds_fulfilled, frame_nums = np.where(cond_vals)
            # gets the frame with the lowest idx where any condition is True
            min_idx = np.argmin(frame_nums)
            first_condition_fulfilled = conds_fulfilled[min_idx]
            logger.error("Starting configuration (%s) is already fulfilling "
                         "the condition with idx %s.",
                         starting_configuration, first_condition_fulfilled,
                         )
            # we just return the starting configuration/trajectory
            trajs = [starting_configuration]
            return trajs, first_condition_fulfilled

        # starting configuration does not fulfill any condition, lets do MD
        engine = self.engine_cls(**self.engine_kwargs)
        # Note: we first check for continuation because if we do not find a run
        # to continue we fallback to no continuation
        if continuation:
            # continuation: get all traj parts already done and continue from
            # there, i.e. append to the last traj part found
            # NOTE: we assume that the condition functions could be different
            # so get all traj parts and calculate the condition funcs on them
            trajs = await get_all_traj_parts(folder=workdir, deffnm=deffnm,
                                             engine=engine,
                                             )
            if not trajs:
                # no trajectories, so we should prepare engine from scratch
                continuation = False
                logger.error("continuation=True, but we found no previous "
                             "trajectories. Setting continuation=False and "
                             "preparing the engine from scratch.")
            else:
                # (can only) calc CV values if we have found trajectories
                cond_vals = await asyncio.gather(
                            *(self._condition_vals_for_traj(t) for t in trajs)
                                                 )
                cond_vals = np.concatenate([np.asarray(s) for s in cond_vals],
                                           axis=1)
                # see if we already fulfill a condition on the existing traj parts
                if (any_cond_fulfilled := np.any(cond_vals)):
                    conds_fulfilled, frame_nums = np.where(cond_vals)
                    # gets the frame with the lowest idx where any cond is True
                    min_idx = np.argmin(frame_nums)
                    first_condition_fulfilled = conds_fulfilled[min_idx]
                    # already fulfill a condition, get out of here!
                    return trajs, first_condition_fulfilled
                # Did not fulfill any condition yet, so prepare the engine to
                # continue the simulation until we reach any of the (new) conds
                await engine.prepare_from_files(workdir=workdir, deffnm=deffnm)

        if not continuation:
            # no continuation, just prepare the engine from scratch
            await engine.prepare(
                        starting_configuration=starting_configuration,
                        workdir=workdir,
                        deffnm=deffnm,
                                )
            trajs = []

        step_counter = engine.steps_done
        any_cond_fulfilled = False
        while ((not any_cond_fulfilled)
                and (step_counter <= self.max_steps)):
            traj = await engine.run_walltime(self.walltime_per_part)
            cond_vals = await self._condition_vals_for_traj(traj)
            any_cond_fulfilled = np.any(cond_vals)
            step_counter = engine.steps_done
            trajs.append(traj)
        if not any_cond_fulfilled:
            # left while loop because of max_frames reached
            raise MaxStepsReachedError(
                f"Engine produced {step_counter} steps (>= {self.max_steps})."
                                       )
        # cond_vals are the ones for the last traj
        # here we get which conditions are True and at which frame
        conds_fulfilled, frame_nums = np.where(cond_vals)
        # gets the frame with the lowest idx where any condition is True
        min_idx = np.argmin(frame_nums)
        # and now the idx to self.conditions for cond that was first fullfilled
        # NOTE/FIXME: if two conditions are reached simultaneously at min_idx,
        #       this will find the condition with the lower idx only
        first_condition_fulfilled = conds_fulfilled[min_idx]
        return trajs, first_condition_fulfilled

    async def cut_and_concatenate(self,
                                  trajs: list[Trajectory],
                                  tra_out: str,
                                  **concatenate_kwargs,
                                  ) -> tuple[Trajectory, int]:
        """
        Cut and concatenate the trajectory until the first condition is True.

        Take a list of trajectory segments and form one continuous trajectory
        until the first frame that fulfills any condition. The first frame in
        that fulfills any condition is included in the trajectory.
        The expected input is a list of trajectories, e.g. the output of the
        :meth:`propagate` method.

        Parameters
        ----------
        trajs : list[Trajectory]
            Trajectory segments to cut and concatenate.
        tra_out : str
            Absolute or relative path for the concatenated output trajectory.
        concatenate_kwargs : dict
            All (other) keyword arguments will be passed as is to the
            :meth:`TrajectoryConcatenator.concatenate` method. These include, e.g.,

            - struct_out : str, optional
                Absolute or relative path to the output structure file, if None
                we will use the structure file of the first traj, by default None.
            - overwrite : bool, optional
                Whether we should overwrite tra_out if it exists, by default False.

        Returns
        -------
        (traj_out, idx_of_condition_fulfilled) : (Trajectory, int)
            The concatenated output trajectory from starting configuration
            until the first condition is True and the index to the condition
            function in `conditions`.
        """
        # get all func values and put them into one big list
        cond_vals = await asyncio.gather(
                            *(self._condition_vals_for_traj(t) for t in trajs)
                                         )
        # cond_vals is a list (trajs) of lists (conditions)
        # take condition 0 (always present) to get the traj part lengths
        part_lens = [len(c[0]) for c in cond_vals]  # c[0] is 1d (np)array
        # get all occurrences where any condition is True
        conds_fulfilled, frame_nums = np.where(
                np.concatenate([np.asarray(c) for c in cond_vals], axis=1)
                )
        # get the index of the frame with the lowest number where any condition is True
        min_idx = np.argmin(frame_nums)
        first_condition_fulfilled = conds_fulfilled[min_idx]
        first_frame_in_cond = frame_nums[min_idx]
        # find out in which part it is
        # nonzero always returns a tuple (first zero index below)
        # and we only care for the first occurrence of True/nonzero (second zero index below)
        last_part_idx = (np.cumsum(part_lens) >= first_frame_in_cond).nonzero()[0][0]
        # find the first frame in cond (counting from start of last part)
        _first_frame_in_cond = (first_frame_in_cond
                                - sum(part_lens[:last_part_idx]))  # >= 0
        if last_part_idx > 0:
            # trajectory parts which we take fully
            slices = [(0, None, 1) for _ in range(last_part_idx)]
        else:
            # only the first/last part
            slices = []
        # and the last part until including first_frame_in_cond
        slices += [(0, _first_frame_in_cond + 1, 1)]
        # we fill in all args as kwargs because there are so many
        full_traj = await TrajectoryConcatenator().concatenate_async(
                                   trajs=trajs[:last_part_idx + 1],
                                   slices=slices,
                                   tra_out=tra_out,
                                   **concatenate_kwargs
                                   )
        return full_traj, first_condition_fulfilled

    async def _condition_vals_for_traj(self, traj: Trajectory
                                       ) -> list[np.ndarray]:
        # return a list of condition_func results,
        # one for each condition func in conditions
        if self.conditions.changed:
            # first check if the conditions (single entries) have been modified
            # if yes just reassign to the property so we recheck which of them
            # are coroutines
            self.conditions = self.conditions
        # we wrap the non-coroutines into tasks to schedule all together
        all_conditions_as_coro = [
            c(traj) if c_is_coro else asyncio.to_thread(c, traj)
            for c, c_is_coro in zip(self.conditions,
                                    self._condition_func_is_coroutine)
                                  ]
        results = await asyncio.gather(*all_conditions_as_coro)
        cond_eq_traj_len = [len(traj) == len(r) for r in results]
        if not all(cond_eq_traj_len):
            bad_condition_idx_str = ", ".join([f"{idx}" for idx, good
                                               in enumerate(cond_eq_traj_len)
                                               if not good])
            raise ValueError("At least one of the conditions does not return "
                             "an array of shape (len(traj), ) when applied to "
                             "the trajectory traj. The conditions in question "
                             "have indexes " + bad_condition_idx_str + " .")
        return results


# alias for people coming from the path sampling community :)
TrajectoryPropagatorUntilAnyState = ConditionalTrajectoryPropagator
