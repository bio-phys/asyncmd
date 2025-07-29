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
This module contains various classes used for trajectory extraction and concatenation.

Most notably the FrameExtractor classes and the TrajectoryConcatenator.
"""
import os
import abc
import collections.abc
import asyncio
import logging
import functools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import MDAnalysis as mda
try:
    # mda v>=2.3 has moved the timestep class
    from MDAnalysis.coordinates.timestep import Timestep as mda_Timestep
except ImportError:
    # this is where it lives for mda v<=2.2
    from MDAnalysis.coordinates.base import Timestep as mda_Timestep
from scipy import constants

from .._config import _SEMAPHORES, _SEMAPHORES_KEYS
from .trajectory import Trajectory


logger = logging.getLogger(__name__)


def _is_documented_by(original):
    """
    Decorator to copy the docstring of a given method to the decorated method.
    """
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target
    return wrapper


def _attach_mda_trafos_to_universe(
        universe: mda.Universe,
        mda_transformations: list[collections.abc.Callable] | None = None,
        mda_transformations_setup_func: collections.abc.Callable | None = None,
                                  ) -> mda.Universe:
    """
    Attach MDAnalysis transformations to a given universe.

    Can either pass a list of on-the-fly transformations or a setup function
    that attaches an arbitrary number of user-defined transformations (that
    then can also depend on e.g. atomgroups in the universe). Note that only
    either a list of transformations or a setup function can be passed but
    never both at the same time.

    Parameters
    ----------
    universe : MDAnalysis.core.universe.Universe
        The universe to attach the transformations to.
    mda_transformations : list[collections.abc.Callable] or None, optional
        List of MDAnalysis transformations to attach, by default None
    mda_transformations_setup_func : collections.abc.Callable or None, optional
        Setup function to attach MDAnalysis transformations to the universe,
        by default None

    Returns
    -------
    MDAnalysis.core.universe.Universe
        The universe with on-the-fly transformations attached.

    Raises
    ------
    ValueError
        If both ``mda_transformations`` and ``mda_transformations_setup_func``
        are given.
    """
    # NOTE: this func is used to attach the MDAnalysis transformations to
    #       the given universe in the TrajectoryConcatenator and
    #       FrameExtractor classes.
    if (mda_transformations is not None
            and mda_transformations_setup_func is not None):
        raise ValueError("`mda_transformations` and "
                         "`mda_transformations_setup_func` are mutually "
                         "exclusive, but both were given."
                         )
    if mda_transformations_setup_func is not None:
        universe = mda_transformations_setup_func(universe)
    elif mda_transformations is not None:
        universe.trajectory.add_transformations(*mda_transformations)
    return universe


class TrajectoryConcatenator:
    """
    Create concatenated trajectory from given trajectories and frames.

    The concatenate method takes a list of trajectories plus a list of slices
    and returns one trajectory containing only the selected frames in the order
    specified by the slices.
    Velocities are automatically inverted if the step of a slice is negative,
    this can be controlled via the ``invert_v_for_negative_step`` attribute.
    Double frames are also automatically removed, which can be controlled via
    the ``remove_double_frames`` attribute.
    We assume that all trajs have the same structure file and attach the
    structure of the first traj if not told otherwise.
    Note that you can pass MDAnalysis transformations to this class to
    transform your trajectories on-the-fly, see the ``mda_transformations`` and
    ``mda_transformations_setup_func`` arguments to :meth:`__init__`.

    Attributes
    ----------
    invert_v_for_negative_step : bool
        Whether to invert all momenta for segments with negative stride.
    remove_double_frames : bool
        Whether we should (try to) remove double frames from the concatenated
        output trajectory.
        Note that a simple heuristic is used to determine double frames,
        frames count as double if the integration time is the same for both
        frames.
    """

    def __init__(
            self,
            invert_v_for_negative_step: bool = True,
            remove_double_frames: bool = True, *,
            mda_transformations: list[collections.abc.Callable] | None = None,
            mda_transformations_setup_func: collections.abc.Callable | None = None,
                 ) -> None:
        """
        Initialize a :class:`TrajectoryConcatenator`.

        Parameters
        ----------
        invert_v_for_negative_step : bool, optional
            Whether to invert all momenta for segments with negative stride,
            by default True.
        remove_double_frames : bool, optional
            Whether we should (try to) remove double frames from the concatenated
            output trajectory, by default True.
        mda_transformations : list of callables, optional
            If given will be added as a list of transformations to the
            MDAnalysis universe as
            ``universe.trajectory.add_transformation(*mda_transformations)``.
            See the ``mda_transformations_setup_func`` argument if your
            transformations need additional universe-dependant arguments, e.g.
            atomgroups from the universe.
            See
            https://docs.mdanalysis.org/stable/documentation_pages/trajectory_transformations.html
            for more on MDAnalysis transformations.
        mda_transformations_setup_func: callable, optional
            If given will be called to attach user-defined MDAnalysis
            transformations to the universe. The function must take a universe
            as argument and return the universe with attached transformations.
            I.e. it is expected that the function calls
            ``universe.trajectory.add_transformations(*list_of_trafos)``
            after defining ``list_of_trafos`` (potentially depending on the
            universe or atomgroups therein) and then finally returns the
            universe with trafos.
            See
            https://docs.mdanalysis.org/stable/documentation_pages/trajectory_transformations.html
            for more on MDAnalysis transformations.
        """
        self.invert_v_for_negative_step = invert_v_for_negative_step
        self.remove_double_frames = remove_double_frames
        if (
            mda_transformations is not None
            and mda_transformations_setup_func is not None
        ):
            raise ValueError("`mda_transformations` and "
                             "`mda_transformations_setup_func` are mutually "
                             "exclusive, but both were given."
                             )
        self.mda_transformations = mda_transformations
        self.mda_transformations_setup_func = mda_transformations_setup_func

    def concatenate(self, trajs: list[Trajectory], slices: list[tuple],
                    tra_out: str, *,
                    struct_out: str | None = None,
                    overwrite: bool = False,
                    ) -> Trajectory:
        """
        Create concatenated trajectory from given trajectories and frames.

        Parameters
        ----------
        trajs : list[Trajectory]
            List of :class:`asyncmd.Trajectory` objects to concatenate.
        slices : list[tuple]
            List of tuples (start, stop, step) specifing the slices of the
            trajectories to take. Must be of len(trajs).
        tra_out : str
            Output trajectory filepath, absolute or relativ to current working
            directory.
        struct_out : str or None, optional
            Output structure filepath, if None we will take the structure file
            of the first trajectory in trajs, by default None.
        overwrite : bool, optional
            Whether we should overwrite existing output trajectories,
            by default False.

        Returns
        -------
        Trajectory
            The concatenated output trajectory.

        Raises
        ------
        FileExistsError
            If ``tra_out`` exists and ``overwrite=False``.
        FileNotFoundError
            If ``struct_out`` given but the file is not accessible.
        """
        tra_out = os.path.relpath(tra_out)
        if os.path.exists(tra_out) and not overwrite:
            raise FileExistsError(f"overwrite=False and tra_out exists: {tra_out}")
        struct_out = (trajs[0].structure_file if struct_out is None
                      else os.path.relpath(struct_out))
        if not os.path.isfile(struct_out):
            # although we would expect that it exists if it comes from an
            # existing traj, we still check to catch other unrelated issues :)
            raise FileNotFoundError(
                        f"Output structure file must exist ({struct_out})."
                                    )

        # special treatment for traj0 because we need n_atoms for the writer
        u = mda.Universe(trajs[0].structure_file, *trajs[0].trajectory_files)
        last_time_seen = None
        with mda.Writer(tra_out, n_atoms=u.atoms.n_atoms) as writer:
            # iterate over the trajectories
            for traj, sl in zip(trajs, slices):
                last_time_seen = self._write_one_traj_sliced_with_writer(
                                            traj=traj, sl=sl, writer=writer,
                                            last_time_seen=last_time_seen,
                )
        # return (file paths to) the finished trajectory
        return Trajectory(tra_out, struct_out)

    @_is_documented_by(concatenate)
    # pylint: disable-next=missing-function-docstring
    async def concatenate_async(self, trajs: list[Trajectory],
                                slices: list[tuple], tra_out: str, *,
                                struct_out: str | None = None,
                                overwrite: bool = False,
                                ) -> Trajectory:
        concat_fx = functools.partial(self.concatenate,
                                      trajs=trajs,
                                      slices=slices,
                                      tra_out=tra_out,
                                      struct_out=struct_out,
                                      overwrite=overwrite,
                                      )
        loop = asyncio.get_running_loop()
        async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]:
            async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]:
                with ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="concat_thread",
                                        ) as pool:
                    return await loop.run_in_executor(pool, concat_fx)

    def _write_one_traj_sliced_with_writer(self, traj: Trajectory, sl: tuple, *,
                                           writer: mda.Writer,
                                           last_time_seen: float | None,
                                           ) -> float | None:
        """
        Write one Trajectory (sliced) using the given MDAnalysis writer.

        Take as argument and return (the updated) ``last_time_seen`` to enable
        removal of double frames over multiple Trajectories and subsequent calls
        to this method.

        Parameters
        ----------
        traj : Trajectory
            The origin trajectory to iterate over (in a sliced manner).
        sl : tuple
            The slice defining which frames are written from ``traj``.
        writer : mda.Writer
        last_time_seen : float | None
            Integration time of the last written/seen timestep.

        Returns
        -------
        float | None
            Updated ``last_time_seen``.
        """
        u = mda.Universe(traj.structure_file, *traj.trajectory_files)
        u = _attach_mda_trafos_to_universe(
                universe=u,
                mda_transformations=self.mda_transformations,
                mda_transformations_setup_func=self.mda_transformations_setup_func,
                )
        start, stop, step = sl
        for ts in u.trajectory[start:stop:step]:
            if self.remove_double_frames and (last_time_seen is not None):
                if last_time_seen == ts.data["time"]:
                    continue
            if (
                self.invert_v_for_negative_step and step < 0
                and ts.has_velocities
            ):
                u.atoms.velocities *= -1
            writer.write(u.atoms)
            if self.remove_double_frames:
                last_time_seen = ts.data["time"]
        # make sure MDAnalysis closes the underlying trajectory file
        u.trajectory.close()
        return last_time_seen


class FrameExtractor(abc.ABC):
    """
    Abstract base class for FrameExtractors.

    Implements the `extract` method which is common in all FrameExtractors.
    Subclasses only need to implement `apply_modification` which is called by
    `extract` to modify the frame just before writing it out.
    """

    # extract a single frame with given idx from a trajectory and write it out
    # simplest case is without modification, but useful modifications are e.g.
    # with inverted velocities, with random Maxwell-Boltzmann velocities, etc.

    def __init__(
        self, *,
        mda_transformations: list[collections.abc.Callable] | None = None,
        mda_transformations_setup_func: collections.abc.Callable | None = None,
                 ) -> None:
        """
        Initialize a :class:`FrameExtractor`.

        Parameters
        ----------
        mda_transformations : list of callables, optional
            If given will be added as a list of transformations to the
            MDAnalysis universe as
            ``universe.trajectory.add_transformation(*mda_transformations)``.
            See the ``mda_transformations_setup_func`` argument if your
            transformations need additional universe-dependant arguments, e.g.
            atomgroups from the universe.
            See
            https://docs.mdanalysis.org/stable/documentation_pages/trajectory_transformations.html
            for more on MDAnalysis transformations.
        mda_transformations_setup_func: callable, optional
            If given will be called to attach user-defined MDAnalysis
            transformations to the universe. The function must take a universe
            as argument and return the universe with attached transformations.
            I.e. it is expected that the function calls
            ``universe.trajectory.add_transformations(*list_of_trafos)``
            after defining ``list_of_trafos`` (potentially depending on the
            universe or atomgroups therein) and then finally returns the
            universe with trafos.
            See
            https://docs.mdanalysis.org/stable/documentation_pages/trajectory_transformations.html
            for more on MDAnalysis transformations.
        """
        if (
            mda_transformations is not None
            and mda_transformations_setup_func is not None
        ):
            raise ValueError("`mda_transformations` and "
                             "`mda_transformations_setup_func` are mutually "
                             "exclusive, but both were given."
                             )
        self.mda_transformations = mda_transformations
        self.mda_transformations_setup_func = mda_transformations_setup_func

    @abc.abstractmethod
    def apply_modification(self,
                           universe: mda.Universe,
                           ts: mda_Timestep,
                           ):
        """
        Apply modification to selected frame (timestep/universe).

        This function will be called when the current timestep is at the
        chosen frame index and is expected to apply the subclass specific
        modifications to the frame via modifying the mdanalysis timestep and
        universe objects **inplace**.
        After this function finishes the frame is written out, i.e. with any
        potential modifications applied.
        No return value is expected or considered from this method, the
        modifications of the timestep/universe are nonlocal anyway.

        Parameters
        ----------
        universe : MDAnalysis.core.universe.Universe
            The mdanalysis universe associated with the trajectory.
        ts : MDAnalysis.coordinates.base.Timestep
            The mdanalysis timestep of the frame to extract.
        """
        raise NotImplementedError

    def extract(self, outfile: str, traj_in: Trajectory, idx: int, *,
                struct_out: str | None = None, overwrite: bool = False,
                ) -> Trajectory:
        """
        Extract a single frame from given trajectory and write it out.

        Parameters
        ----------
        outfile : str
            Absolute or relative path to the output trajectory. Expected to be
            with file ending, e.g. "traj.trr".
        traj_in : Trajectory
            Input trajectory from which we will extract the frame at `idx`.
        idx : int
            Index of the frame to extract in `traj_in`.
        struct_out : str, optional
            None, or absolute or relative path to a structure file,
            by default None. If not None we will use the given file as
            structure file for the returned trajectory object, else we use the
            structure file of `traj_in`.
        overwrite : bool, optional
            Whether to overwrite `outfile` if it exists, by default False.

        Returns
        -------
        Trajectory
            Trajectory object holding a trajectory with the extracted frame.

        Raises
        ------
        FileExistsError
            If `outfile` exists and `overwrite=False`.
        FileNotFoundError
            If `struct_out` is given but does not exist.
        """
        # TODO: make it possible to select a subset of atoms to write out
        #       and also for modification?
        outfile = os.path.relpath(outfile)
        if os.path.exists(outfile) and not overwrite:
            raise FileExistsError(f"overwrite=False but outfile={outfile} exists.")
        struct_out = (traj_in.structure_file if struct_out is None
                      else os.path.relpath(struct_out))
        if not os.path.isfile(struct_out):
            # although we would expect that it exists if it comes from an
            # existing traj, we still check to catch other unrelated issues :)
            raise FileNotFoundError("Output structure file must exist."
                                    + f"(given struct_out is {struct_out})."
                                    )
        u = mda.Universe(traj_in.structure_file, *traj_in.trajectory_files)
        u = _attach_mda_trafos_to_universe(
            universe=u,
            mda_transformations=self.mda_transformations,
            mda_transformations_setup_func=self.mda_transformations_setup_func,
            )
        with mda.Writer(outfile, n_atoms=u.trajectory.n_atoms) as writer:
            ts = u.trajectory[idx]
            self.apply_modification(u, ts)
            writer.write(u.atoms)
        # make sure MDAnalysis closes the underlying trajectory files
        u.trajectory.close()
        del u
        return Trajectory(trajectory_files=outfile, structure_file=struct_out)

    @_is_documented_by(extract)
    # pylint: disable-next=missing-function-docstring
    async def extract_async(self, outfile: str, traj_in: Trajectory, idx: int, *,
                            struct_out: str | None = None, overwrite: bool = False,
                            ) -> Trajectory:
        extract_fx = functools.partial(self.extract,
                                       outfile=outfile,
                                       traj_in=traj_in,
                                       idx=idx,
                                       struct_out=struct_out,
                                       overwrite=overwrite,
                                       )
        loop = asyncio.get_running_loop()
        async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_FILES_OPEN]:
            async with _SEMAPHORES[_SEMAPHORES_KEYS.MAX_PROCESS]:
                with ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="concat_thread",
                                        ) as pool:
                    return await loop.run_in_executor(pool, extract_fx)


class NoModificationFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out without modification."""

    def apply_modification(self,
                           universe: mda.Universe,
                           ts: mda_Timestep,
                           ):
        """
        Apply no modification to the extracted frame.

        Parameters
        ----------
        universe : MDAnalysis.core.universe.Universe
            The mdanalysis universe associated with the trajectory.
        ts : MDAnalysis.coordinates.base.Timestep
            The mdanalysis timestep of the frame to extract.
        """


class InvertedVelocitiesFrameExtractor(FrameExtractor):
    """
    Extract a frame from a trajectory, write it out with inverted velocities.
    """

    def apply_modification(self,
                           universe: mda.Universe,
                           ts: mda_Timestep,
                           ):
        """
        Invert all momenta of the extracted frame.

        Parameters
        ----------
        universe : MDAnalysis.core.universe.Universe
            The mdanalysis universe associated with the trajectory.
        ts : MDAnalysis.coordinates.base.Timestep
            The mdanalysis timestep of the frame to extract.
        """
        ts.velocities *= -1.


class RandomVelocitiesFrameExtractor(FrameExtractor):
    """
    Extract a frame from a trajectory, write it out with randomized velocities.

    Attributes
    ----------
    T : float
        Temperature of the Maxwell-Boltzmann distribution for velocity
        generation, in Kelvin.
    """

    def __init__(
        self,
        T: float, *,
        mda_transformations: list[collections.abc.Callable] | None = None,
        mda_transformations_setup_func: collections.abc.Callable | None = None,
                 ) -> None:
        """
        Initialize a :class:`RandomVelocitiesFrameExtractor`.

        Parameters
        ----------
        T : float
            Temperature of the Maxwell-Boltzmann distribution, in Kelvin.
        mda_transformations : list of callables, optional
            If given will be added as a list of transformations to the
            MDAnalysis universe as
            ``universe.trajectory.add_transformation(*mda_transformations)``.
            See the ``mda_transformations_setup_func`` argument if your
            transformations need additional universe-dependant arguments, e.g.
            atomgroups from the universe.
            See
            https://docs.mdanalysis.org/stable/documentation_pages/trajectory_transformations.html
            for more on MDAnalysis transformations.
        mda_transformations_setup_func: callable, optional
            If given will be called to attach user-defined MDAnalysis
            transformations to the universe. The function must take a universe
            as argument and return the universe with attached transformations.
            I.e. it is expected that the function calls
            ``universe.trajectory.add_transformations(*list_of_trafos)``
            after defining ``list_of_trafos`` (potentially depending on the
            universe or atomgroups therein) and then finally returns the
            universe with trafos.
            See
            https://docs.mdanalysis.org/stable/documentation_pages/trajectory_transformations.html
            for more on MDAnalysis transformations.
        """
        super().__init__(
                mda_transformations=mda_transformations,
                mda_transformations_setup_func=mda_transformations_setup_func,
                         )
        # pylint: disable-next=invalid-name
        self.T = T  # in K
        self._rng = np.random.default_rng()

    def apply_modification(self,
                           universe: mda.Universe,
                           ts: mda_Timestep,
                           ):
        """
        Draw random Maxwell-Boltzmann velocities for extracted frame.

        Parameters
        ----------
        universe : MDAnalysis.core.universe.Universe
            The mdanalysis universe associated with the trajectory.
        ts : MDAnalysis.coordinates.base.Timestep
            The mdanalysis timestep of the frame to extract.
        """
        # m is in units of g / mol
        # v should be in units of \AA / ps = 100 m / s
        # which means m [10**-3 kg / mol] v**2 [10000 (m/s)**2]
        # is in units of [ 10 kg m**s / (mol * s**2) ]
        # so we use R = N_A * k_B [J / (mol * K) = kg m**2 / (s**2 * mol * K)]
        # and add in a factor 10 to get 1/σ**2 = m / (k_B * T)
        # in the correct units
        scale = np.empty((ts.n_atoms, 3), dtype=np.float64)
        s1d = np.sqrt((self.T * constants.R * 0.1)
                      / universe.atoms.masses
                      )
        # sigma is the same for all 3 cartesian dimensions
        for i in range(3):
            scale[:, i] = s1d
        ts.velocities = self._rng.normal(loc=0, scale=scale)
