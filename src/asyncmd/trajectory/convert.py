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
import logging
import numpy as np
import MDAnalysis as mda
from scipy import constants

from .trajectory import Trajectory


logger = logging.getLogger(__name__)


class TrajectoryConcatenator:
    """
    Create concatenated trajectory from given trajectories and frames.

    The concatenate method takes a list of trajectories and a list of slices,
    returns one trajectory containing only the selected frames in that order.
    Velocities are automatically inverted if the step of a slice is negative,
    this can be controlled via the invert_v_for_negative_step attribute.

    NOTE: We assume that all trajs have the same structure file
          and attach the the structure of the first traj if not told otherwise.
    """

    def __init__(self, invert_v_for_negative_step=True):
        self.invert_v_for_negative_step = invert_v_for_negative_step

    def concatenate(self, trajs, slices, tra_out, struct_out=None,
                    overwrite=False, remove_double_frames=True):
        """
        Create concatenated trajectory from given trajectories and frames.

        trajs - list of `:class:`Trajectory
        slices - list of (start, stop, step)
        tra_out - output trajectory filepath, absolute or relativ to cwd
        struct_out - None or output structure filepath, if None we will take the
                     structure file of the first trajectory in trajs
        overwrite - bool (default=False), if True overwrite existing tra_out,
                    if False and the file exists raise an error
        remove_double_frames - bool (default=True), if True try to remove double
                               frames from the concatenated output
                               NOTE: that we use a simple heuristic, we just
                                     check if the integration time is the same
        """
        tra_out = os.path.abspath(tra_out)
        if os.path.exists(tra_out) and not overwrite:
            raise ValueError(f"overwrite=False and tra_out exists: {tra_out}")
        struct_out = (trajs[0].structure_file if struct_out is None
                      else os.path.abspath(struct_out))
        if not os.path.isfile(struct_out):
            # although we would expect that it exists if it comes from an
            # existing traj, we still check to catch other unrelated issues :)
            raise ValueError(f"Output structure file must exist ({struct_out}).")

        # special treatment for traj0 because we need n_atoms for the writer
        u0 = mda.Universe(trajs[0].structure_file, trajs[0].trajectory_file,
                          tpr_resid_from_one=True)
        start0, stop0, step0 = slices[0]
        # if the file exists MDAnalysis will silently overwrite
        with mda.Writer(tra_out, n_atoms=u0.trajectory.n_atoms) as W:
            for ts in u0.trajectory[start0:stop0:step0]:
                if self.invert_v_for_negative_step and step0 < 0:
                    u0.atoms.velocities *= -1
                W.write(u0.atoms)
                if remove_double_frames:
                    # remember the last timestamp, so we can take it out
                    last_time_seen = ts.data["time"]
            del u0  # should free up memory and does no harm?!
            for traj, sl in zip(trajs[1:], slices[1:]):
                u = mda.Universe(traj.structure_file, traj.trajectory_file,
                                 tpr_resid_from_one=True)
                start, stop, step = sl
                for ts in u.trajectory[start:stop:step]:
                    if remove_double_frames:
                        if last_time_seen == ts.data["time"]:
                            # this is a no-op, as they are they same...
                            # last_time_seen = ts.data["time"]
                            continue  # skip this timestep/go to next iteration
                    if self.invert_v_for_negative_step and step < 0:
                        u.atoms.velocities *= -1
                    W.write(u.atoms)
                    if remove_double_frames:
                        last_time_seen = ts.data["time"]
                del u
        # return (file paths to) the finished trajectory
        return Trajectory(tra_out, struct_out)


class FrameExtractor(abc.ABC):
    # extract a single frame with given idx from a trajectory and write it out
    # simplest case is without modification, but useful modifications are e.g.
    # with inverted velocities, with random Maxwell-Boltzmann velocities, etc.

    @abc.abstractmethod
    def apply_modification(self, universe, ts):
        # this func will is called when the current timestep is at the choosen
        # frame and applies the subclass specific frame modifications to the
        # mdanalysis universe, after this function finishes the frames is
        # written out, i.e. with potential modifications applied
        # no return value is expected or considered,
        # the modifications in the universe are nonlocal anyway
        raise NotImplementedError

    def extract(self, outfile, traj_in, idx, struct_out=None, overwrite=False):
        # TODO: should we check that idx is an idx, i.e. an int?
        # TODO: make it possible to select a subset of atoms to write out
        #       and also for modification?
        # TODO: should we make it possible to extract multiple frames, i.e.
        #       enable the use of slices (and iterables of indices?)
        """
        Extract a single frame from trajectory and write it out.

        outfile - path to output file (relative or absolute)
        traj_in - `:class:Trajectory` from which the original frame is taken
        idx - index of the frame in the input trajectory
        struct_out - None or output structure filepath, if None we will take the
                     structure file of the input trajectory
        overwrite - bool (default=False), if True overwrite existing tra_out,
                    if False and the file exists raise an error
        """
        outfile = os.path.abspath(outfile)
        if os.path.exists(outfile) and not overwrite:
            raise ValueError(f"overwrite=False and outfile exists: {outfile}")
        struct_out = (traj_in.structure_file if struct_out is None
                      else os.path.abspath(struct_out))
        if not os.path.isfile(struct_out):
            # although we would expect that it exists if it comes from an
            # existing traj, we still check to catch other unrelated issues :)
            raise ValueError(f"Output structure file must exist ({struct_out}).")
        u = mda.Universe(traj_in.structure_file, traj_in.trajectory_file,
                         tpr_resid_from_one=True)
        with mda.Writer(outfile, n_atoms=u.trajectory.n_atoms) as W:
            ts = u.trajectory[idx]
            self.apply_modification(u, ts)
            W.write(u.atoms)
        return Trajectory(trajectory_file=outfile, structure_file=struct_out)


class NoModificationFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out without modification."""

    def apply_modification(self, universe, ts):
        pass


class InvertedVelocitiesFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out with inverted velocities."""

    def apply_modification(self, universe, ts):
        ts.velocities *= -1.


class RandomVelocitiesFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out with randomized velocities."""

    def __init__(self, T):
        """Temperature T must be given in degree Kelvin."""
        self.T = T  # in K
        self._rng = np.random.default_rng()

    def apply_modification(self, universe, ts):
        # m is in units of g / mol
        # v should be in units of \AA / ps = 100 m / s
        # which means m [10**-3 kg / mol] v**2 [10000 (m/s)**2]
        # is in units of [ 10 kg m**s / (mol * s**2) ]
        # so we use R = N_A * k_B [J / (mol * K) = kg m**2 / (s**2 * mol * K)]
        # and add in a factor 10 to get 1/σ**2 = m / (k_B * T) in the right units
        scale = np.empty((ts.n_atoms, 3), dtype=np.float64)
        s1d = np.sqrt((self.T * constants.R * 0.1)
                      / universe.atoms.masses
                      )
        # sigma is the same for all 3 cartesian dimensions
        for i in range(3):
            scale[:, i] = s1d
        ts.velocities = self._rng.normal(loc=0, scale=scale)
