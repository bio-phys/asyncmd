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
This module contains classes and functions for engine-agnostic but trajectory-related operations.

All user-facing classes and functions are reexported here for convenience.
This includes:

- the TrajectoryFunctionWrapper classes for CV value calculation and caching,
- the Conditional/InParts TrajectoryPropagator classes for propagation of MD in
  parts and/or until a condition is reached (and related functions),
- and classes for extracting and concatenating trajectories (FrameExtractors
  and TrajectoryConcatenator)

"""
from .convert import (NoModificationFrameExtractor,
                      InvertedVelocitiesFrameExtractor,
                      RandomVelocitiesFrameExtractor,
                      TrajectoryConcatenator,
                      )
from .functionwrapper import (PyTrajectoryFunctionWrapper,
                              SlurmTrajectoryFunctionWrapper,
                              )
from .propagate import (ConditionalTrajectoryPropagator,
                        TrajectoryPropagatorUntilAnyState,
                        InPartsTrajectoryPropagator,
                        construct_tp_from_plus_and_minus_traj_segments,
                        )
from .trajectory import (_forget_trajectory,
                         _forget_all_trajectories,
                         )
