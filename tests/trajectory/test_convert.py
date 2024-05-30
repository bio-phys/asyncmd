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
import pytest
import os
import MDAnalysis as mda
import numpy as np

import asyncmd
from asyncmd.trajectory.convert import (TrajectoryConcatenator,
                                        NoModificationFrameExtractor,
                                        InvertedVelocitiesFrameExtractor,
                                        RandomVelocitiesFrameExtractor,
                                        )


class TBase:
    def setup_method(self):
        # this ala traj has 18 frames
        self.ala_traj = asyncmd.Trajectory(
                trajectory_files="tests/test_data/trajectory/ala_traj.trr",
                structure_file="tests/test_data/trajectory/ala.tpr"
                                           )
        # some simple MDAnalysis trafos
        # these ones can just be attached (do not depend on the universe or
        #  atomgroups therein)
        def translate_in_x(x_trans=5):
            def wrapped(ts):
                ts.positions[:, 0] += x_trans
                return ts
            return wrapped
        def translate_in_y(y_trans=5):
            def wrapped(ts):
                ts.positions[:, 1] += y_trans
                return ts
            return wrapped
        self.mda_trafos = [translate_in_x(), translate_in_y()]

        # these ones depend on the universe
        def translate_prot_in_x(universe, x_trans=5):
            all_prot_ix = universe.select_atoms("protein").ix
            def wrapped(ts):
                ts.positions[all_prot_ix, 0] += x_trans
                return ts
            return wrapped
        def translate_prot_in_y(universe, y_trans=5):
            all_prot_ix = universe.select_atoms("protein").ix
            def wrapped(ts):
                ts.positions[all_prot_ix, 1] += y_trans
                return ts
            return wrapped
        # lets write an setup func
        def mda_trafo_setup_func(universe):
            trafos = [translate_prot_in_x(universe=universe),
                      translate_prot_in_y(universe=universe),
                      ]
            universe.trajectory.add_transformations(*trafos)
            return universe
        self.mda_trafo_setup_func = mda_trafo_setup_func


class Test_NoModificationFrameExtractor(TBase):
    @pytest.mark.parametrize("idx", [0, 5, 10, 17])
    @pytest.mark.parametrize("mda_transformations", [True, False])
    @pytest.mark.parametrize("mda_transformations_setup_func", [True, False])
    @pytest.mark.parametrize("use_async", [True, False])
    @pytest.mark.asyncio
    async def test_extract(
                self, idx, tmpdir, use_async,
                mda_transformations,  # whether we use simple mda trafos
                mda_transformations_setup_func,  # whetger we use a setup func for more complex trafos
                           ):
        # figure out if we use mda trafos
        if mda_transformations and not mda_transformations_setup_func:
            # simple trafos
            extractor = NoModificationFrameExtractor(
                    mda_transformations=self.mda_trafos,
                    mda_transformations_setup_func=None,
                                                     )
        elif not mda_transformations and mda_transformations_setup_func:
            # complex trafos
            extractor = NoModificationFrameExtractor(
                    mda_transformations=None,
                    mda_transformations_setup_func=self.mda_trafo_setup_func,
                                                     )
        elif not mda_transformations and not mda_transformations_setup_func:
            # no trafos at all
            extractor = NoModificationFrameExtractor(
                    mda_transformations=None,
                    mda_transformations_setup_func=None,
                                                     )
        elif mda_transformations and mda_transformations_setup_func:
            # if both ways of passing mda trafos are used simultaneously we
            # should get a ValueError
            with pytest.raises(ValueError):
                extractor = NoModificationFrameExtractor(
                    mda_transformations=self.mda_trafos,
                    mda_transformations_setup_func=self.mda_trafo_setup_func,
                                                         )
            return  # get out of here
        # actual extraction
        if use_async:
            out_frame = await extractor.extract_async(
                                        outfile=tmpdir + "/out_frame.trr",
                                        traj_in=self.ala_traj,
                                        idx=idx,
                                                      )
        else:
            out_frame = extractor.extract(outfile=tmpdir + "/out_frame.trr",
                                          traj_in=self.ala_traj,
                                          idx=idx,
                                          )
        # get mda universes of the original and written out to compare
        in_traj_universe = mda.Universe(self.ala_traj.structure_file,
                                        *self.ala_traj.trajectory_files,
                                        )
        out_frame_universe = mda.Universe(out_frame.structure_file,
                                          *out_frame.trajectory_files,
                                          )
        _ = in_traj_universe.trajectory[idx]
        _ = out_frame_universe.trajectory[0]
        all_atoms_original = in_traj_universe.select_atoms("all")
        all_atoms_extracted = out_frame_universe.select_atoms("all")
        all_pos_original = all_atoms_original.positions.copy()
        # figure out what our trafos did to the original/extracted
        if mda_transformations:
            # we shift x and y of all atoms by 5 \AA each
            all_pos_original[:, 0] += 5
            all_pos_original[:, 1] += 5
        elif mda_transformations_setup_func:
            # we shift all protein atoms by 5 \AA in x and y each
            all_prot_atoms_original = in_traj_universe.select_atoms("protein")
            protein_ixs = all_prot_atoms_original.ix
            all_pos_original[protein_ixs, 0] += 5
            all_pos_original[protein_ixs, 1] += 5
        # compare!
        assert np.allclose(all_pos_original,
                           all_atoms_extracted.positions,
                           )
        # and check for velocities too
        assert np.allclose(all_atoms_original.velocities,
                           all_atoms_extracted.velocities,
                           )
