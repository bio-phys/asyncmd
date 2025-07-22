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
This module contains the abstract base class defining the interface for all MDEngines.

It also defines the commonly used exceptions for MDEngines.
"""
import abc
from .trajectory.trajectory import Trajectory


class EngineError(Exception):
    """Exception raised when something goes wrong with the (MD)-Engine."""


class EngineCrashedError(EngineError):
    """Exception raised when the (MD)-Engine crashes during a run."""


class MDEngine(abc.ABC):
    """
    Abstract base class to define a common interface for all :class:`MDEngine`.
    """
    @abc.abstractmethod
    async def apply_constraints(self, conf_in: Trajectory,
                                conf_out_name: str) -> Trajectory:
        """
        Apply constraints to given conf_in, write conf_out_name and return it.

        Parameters
        ----------
        conf_in : Trajectory
        conf_out_name : str

        Returns
        -------
        Trajectory
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def prepare(self, starting_configuration: Trajectory, workdir: str,
                      deffnm: str) -> None:
        """
        Prepare the engine to run a MD from starting_configuration.

        NOTE: We assume that we do not change the system for/in one engine,
        i.e. .top, .ndx, mdp-object, ...?! should go into __init__.

        Parameters
        ----------
        starting_configuration : Trajectory
            The initial configuration.
        workdir : str
            The directory in which the MD will be performed.
        deffnm : str
            The standard filename to use for this MD run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    #@classmethod
    async def prepare_from_files(self, workdir: str, deffnm: str) -> None:
        """
        Prepare the engine to continue a previously stopped simulation starting
        with the last trajectory part in workdir that is compatible with deffnm.

        NOTE: This can not be a classmethod (reliably) because we set top/ndx/
        mdconfig/etc in '__init__'.

        Parameters
        ----------
        workdir : str
            The directory in which the MD will be/ was previously performed.
        deffnm : str
            The standard filename to use for this MD run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def run_walltime(self, walltime: float, max_steps: int | None = None,
                           ) -> Trajectory | None:
        """
        Run for specified walltime.

        NOTE: Must be possible to run this multiple times after preparing once!

        It is optional (but recommended if possible) for engines to respect the
        ``max_steps`` argument. I.e. terminating upon reaching max_steps is
        optional and no code should rely on it. See the :meth:`run_steps` if a
        fixed number of integration steps is required.

        Return None if no integration is needed because max_steps integration
        steps have already been performed.

        Parameters
        ----------
        walltime : float
            Walltime in hours.
        max_steps : int | None, optional
            If not None, (optionally) terminate when max_steps integration steps
            in total are reached, also if this is before walltime is reached.
            By default None.

        Returns
        -------
        Trajectory | None
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def run_steps(self, nsteps: int,
                        steps_per_part: bool = False) -> Trajectory | None:
        """
        Run for a specified number of integration steps.

        Return None if no integration is needed because nsteps integration steps
        have already been performed.

        NOTE: Make sure we can run multiple times after preparing once!

        Parameters
        ----------
        nsteps : int
        steps_per_part : bool, optional
            Count nsteps for this part/run or in total, by default False

        Returns
        -------
        Trajectory | None
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def current_trajectory(self) -> Trajectory | None:
        """
        Return current trajectory: Trajectory or None.
        """
        # if we return a Trajectory it is either what we are working on atm
        # or the trajectory we finished last
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_traj_type(self) -> str:
        """
        Return a string with the ending (without ".") of the trajectory type this
        engine uses.

        NOTE: This should not be implemented as a property in subclasses as it
        must be available at the classlevel too, i.e. cls.output_traj_type must
        also return the string.
        So this should just be overwritten with a string with the correct value,
        or if your engine supports multiple output_traj_types you should have a
        look at the descriptor implementation in asyncmd/tools.py (and, e.g.,
        used in asyncmd/gromacs/mdengine.py), which checks for allowed values
        (at least when set on an instance) but is accessible from the class
        level too, i.e. like a 'classproperty' (which is not a thing in python).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def steps_done(self) -> int:
        """
        Return the number of integration steps this engine has performed in total.
        """
        raise NotImplementedError
