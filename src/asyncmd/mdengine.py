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
import abc
from .trajectory.trajectory import Trajectory


class EngineError(Exception):
    """Exception raised when something goes wrong with the (MD)-Engine."""
    pass


class EngineCrashedError(EngineError):
    """Exception raised when the (MD)-Engine crashes during a run."""
    pass


class MDEngine(abc.ABC):
    """
    Abstract base class to define a common interface for all :class:`MDEngine`.
    """
    @abc.abstractmethod
    async def apply_constraints(self, conf_in: Trajectory,
                                conf_out_name: str) -> Trajectory:
        # apply constraints to given conf_in, write conf_out_name and return it
        raise NotImplementedError

    @abc.abstractmethod
    # TODO: think about the most general interface!
    # NOTE: We assume that we do not change the system for/in one engine,
    #       i.e. .top, .ndx, mdp-object, ...?! should go into __init__
    async def prepare(self, starting_configuration: Trajectory, workdir: str,
                      deffnm: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    # TODO: should this be a classmethod?
    #@classmethod
    async def prepare_from_files(self, workdir: str, deffnm: str) -> None:
        # this should prepare the engine to continue a previously stopped simulation
        # starting with the last trajectory part in workdir that is compatible with deffnm
        raise NotImplementedError

    @abc.abstractmethod
    async def run_walltime(self, walltime: float) -> Trajectory:
        # run for specified walltime
        # NOTE: must be possible to run this multiple times after preparing once!
        raise NotImplementedError

    @abc.abstractmethod
    async def run_steps(self, nsteps: int,
                        steps_per_part: bool = False) -> Trajectory:
        # run for specified number of steps
        # NOTE: not sure if we need it, but could be useful
        # NOTE: make sure we can run multiple times after preparing once!
        raise NotImplementedError

    @abc.abstractproperty
    def current_trajectory(self) -> Trajectory:
        # return current trajectory: Trajectory or None
        # if we retun a Trajectory it is either what we are working on atm
        # or the trajectory we finished last
        raise NotImplementedError

    @abc.abstractproperty
    def output_traj_type(self) -> str:
        # return a string with the ending (without ".") of the trajectory
        # type this engine uses
        # NOTE: this should not be implemented as a property in subclasses
        #       as it must be available at the classlevel too
        #       so cls.output_traj_type must also be the string
        #       If you want/need to check the values (i.e. you would like to
        #       execute code like in a property) have a look at the descriptor
        #       implementation in gromacs/mdengine.py which checks for allowed
        #       values (at least when set on an instance) but is accesible from
        #       the class level too, e.g. like a 'classproperty' (which is not
        #        a thing in python)
        raise NotImplementedError

    # TODO/FIXME: remove this function?
    # NOTE: I think we wont really need/use this anyway since the run_ funcs
    #       are all awaitable
    @abc.abstractproperty
    def running(self) -> bool:
        raise NotImplementedError

    @abc.abstractproperty
    def steps_done(self) -> int:
        raise NotImplementedError
