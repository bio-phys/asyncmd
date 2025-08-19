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
    # class with common test methods useful for FrameExtractors and TrajectoryConcatenator
    def setup_method(self):
        # pylint: disable=attribute-defined-outside-init
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


class Test_TrajectoryConcatenator(TBase):
    def test_init_and_concatenate_raises(self, tmp_path):
        # test for errors raised by init
        # if both mda_trafos and mda_trafo_setup_func are give we should raise
        # a ValueError
        with pytest.raises(ValueError):
            concatenator = TrajectoryConcatenator(
                    mda_transformations=self.mda_trafos,
                    mda_transformations_setup_func=self.mda_trafo_setup_func,
                                                  )
        # now check for errors raised by concatenate method
        concatenator = TrajectoryConcatenator()
        # check for FileExistsError when we write the same file twice
        _ = concatenator.concatenate(trajs=[self.ala_traj],
                                     slices=[(0, 5, 1)],
                                     tra_out=os.path.join(tmp_path, "tra_out.trr"),
                                     )
        # this time must err
        with pytest.raises(FileExistsError):
            _ = concatenator.concatenate(trajs=[self.ala_traj],
                                         slices=[(0, 5, 1)],
                                         tra_out=os.path.join(tmp_path, "tra_out.trr"),
                                         )
        # this time it must not err since we pass overwrite=True
        _ = concatenator.concatenate(trajs=[self.ala_traj],
                                     slices=[(0, 5, 1)],
                                     tra_out=os.path.join(tmp_path, "tra_out.trr"),
                                     overwrite=True,
                                     )
        # check for FileNotFoundError raised when struct_out is not found
        with pytest.raises(FileNotFoundError):
            _ = concatenator.concatenate(trajs=[self.ala_traj],
                                         slices=[(0, 5, 1)],
                                         tra_out=os.path.join(tmp_path, "tra_out2.trr"),
                                         struct_out=os.path.join(tmp_path, "does_not_exist.tpr"),
                                         )

    @pytest.mark.parametrize("slices", [[(0, 18, 1)],
                                        # here the first slice will result in no frames
                                        [(0, 18, -1), (0, 18, 1)],
                                        [(5, 0, -1), (0, 5, 1)],
                                        # here we will have a double frame
                                        [(0, 6, 1), (5, 0, -1)],
                                        [(5, 2, -2), (0, 5, 2)],
                                        # here we will have two double frames
                                        [(0, 6, 1), (5, -1, -1), (0, 5, 1)],
                                        ])
    @pytest.mark.parametrize("invert_v_for_negative_step", [True, False])
    @pytest.mark.parametrize("remove_double_frames", [True, False])
    @pytest.mark.parametrize(["mda_transformations", "mda_transformations_setup_func"],
                             # they can not be both true at the same time (we test for the err above)
                             ([False, False],
                              [True, False],
                              [False, True],
                              )
                             )
    @pytest.mark.parametrize("use_async", [True, False])
    @pytest.mark.asyncio
    async def test_concatenate(self, tmp_path, slices, use_async,
                               invert_v_for_negative_step,
                               remove_double_frames,
                               mda_transformations,  # whether we use simple mda trafos
                               mda_transformations_setup_func,  # whether we use a setup func for more complex trafos
                               ):
        if mda_transformations and not mda_transformations_setup_func:
            concatenator = TrajectoryConcatenator(
                invert_v_for_negative_step=invert_v_for_negative_step,
                remove_double_frames=remove_double_frames,
                mda_transformations=self.mda_trafos,
                mda_transformations_setup_func=None,
                                                  )
        elif not mda_transformations and mda_transformations_setup_func:
            concatenator = TrajectoryConcatenator(
                invert_v_for_negative_step=invert_v_for_negative_step,
                remove_double_frames=remove_double_frames,
                mda_transformations=None,
                mda_transformations_setup_func=self.mda_trafo_setup_func,
                                                  )
        elif not mda_transformations and not mda_transformations_setup_func:
            concatenator = TrajectoryConcatenator(
                invert_v_for_negative_step=invert_v_for_negative_step,
                remove_double_frames=remove_double_frames,
                mda_transformations=None,
                mda_transformations_setup_func=None,
                                                  )
        # actual concatenation
        if use_async:
            out_traj = await concatenator.concatenate_async(
                    trajs=[self.ala_traj for _ in range(len(slices))],
                    slices=slices,
                    tra_out=os.path.join(tmp_path, "tra_out.trr"),
                    )
        else:
            out_traj = concatenator.concatenate(
                    trajs=[self.ala_traj for _ in range(len(slices))],
                    slices=slices,
                    tra_out=os.path.join(tmp_path, "tra_out.trr"),
                    )
        # get universes of in and out to compare
        u_original = mda.Universe(self.ala_traj.structure_file,
                                  *self.ala_traj.trajectory_files)
        u_written = mda.Universe(out_traj.structure_file,
                                 *out_traj.trajectory_files)
        # step through all the slices in the original and compare each frame to
        # what we have in the written universe
        written_frame_count = 0
        if remove_double_frames:
            last_time_seen = None
        for sl in slices:
            start, stop, step = sl
            for ts_original in u_original.trajectory[start:stop:step]:
                if remove_double_frames:
                    # the TrajectoryConcatenator removes frames with the same
                    # (integration) time as double frames
                    cur_time = ts_original.data["time"]
                    if last_time_seen is not None:
                        if last_time_seen == cur_time:
                            continue
                    last_time_seen = cur_time
                ts_written = u_written.trajectory[written_frame_count]
                all_atoms_original = u_original.select_atoms("all")
                all_atoms_extracted = u_written.select_atoms("all")
                all_pos_original = all_atoms_original.positions.copy()
                # figure out what our trafos did to the original/extracted
                if mda_transformations:
                    # we shift x and y of all atoms by 5 \AA each
                    all_pos_original[:, 0] += 5
                    all_pos_original[:, 1] += 5
                elif mda_transformations_setup_func:
                    # we shift all protein atoms by 5 \AA in x and y each
                    all_prot_atoms_original = u_original.select_atoms("protein")
                    protein_ixs = all_prot_atoms_original.ix
                    all_pos_original[protein_ixs, 0] += 5
                    all_pos_original[protein_ixs, 1] += 5
                # compare!
                # coordinates
                assert np.allclose(all_pos_original,
                                   all_atoms_extracted.positions)
                # and velocities
                vel_factor = -1. if invert_v_for_negative_step and step < 0 else 1.
                assert np.allclose(all_atoms_original.velocities,
                                   vel_factor * all_atoms_extracted.velocities)
                # increment written frame counter by one
                written_frame_count += 1
        # make sure that we have steped through the whole written trajectory
        assert written_frame_count == len(out_traj)


class TBase_FrameExtractors(TBase):
    # class for common test methods useful for all FrameExtractors
    async def test_extract(
                self, idx, tmp_path, use_async,
                mda_transformations,  # whether we use simple mda trafos
                mda_transformations_setup_func,  # whether we use a setup func for more complex trafos
                extractor_class,  # the FrameExtractor subclass to test
                extractor_init_kwargs: dict,  # the (additional) init kwargs for the extractor as a dict
                           ):
        """
        Instantiante a given extractor_class with given additional kwargs and
        call its extract or extract_async methods. Return original position and
        velocities and extacted atoms group for comparison in subclassed tests.
        If MDAnalysis transformations are requested the original atoms positions
        are modified according to the transformations before returning.

        Parameters
        ----------
        idx : int
            index to pass to extract
        tmp_path : Fixture
            pytest temp_path fixture
        use_async : bool
            whether to test the async or non-async version of extract method
        mda_transformations : bool
            whether to use MDAnalysis transformations (simple list)
        mda_transformations_setup_func : bool
            whether to use more complex MDAnalysis transformations with setup func
        extractor_class : FrameExtractor
            the FrameExtractor (subclass) to test
        extractor_init_kwargs : dict
            the (additional) init kwargs for the extractor as a dict

        Returns
        -------
        all_pos_original, all_vels_original, all_atoms_extracted
        """
        # figure out if we use mda trafos
        if mda_transformations and not mda_transformations_setup_func:
            # simple trafos
            extractor = extractor_class(
                    mda_transformations=self.mda_trafos,
                    mda_transformations_setup_func=None,
                    **extractor_init_kwargs
                                        )
        elif not mda_transformations and mda_transformations_setup_func:
            # complex trafos
            extractor = extractor_class(
                    mda_transformations=None,
                    mda_transformations_setup_func=self.mda_trafo_setup_func,
                    **extractor_init_kwargs
                                        )
        elif not mda_transformations and not mda_transformations_setup_func:
            # no trafos at all
            extractor = extractor_class(
                    mda_transformations=None,
                    mda_transformations_setup_func=None,
                    **extractor_init_kwargs
                                        )
        # actual extraction
        if use_async:
            out_frame = await extractor.extract_async(
                                        outfile=os.path.join(tmp_path, "out_frame.trr"),
                                        traj_in=self.ala_traj,
                                        idx=idx,
                                                      )
        else:
            out_frame = extractor.extract(outfile=os.path.join(tmp_path, "out_frame.trr"),
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
        return (all_pos_original,
                all_atoms_original.velocities,
                all_atoms_extracted,
                )


class Test_NoModificationFrameExtractor(TBase_FrameExtractors):
    def test_init_and_extract_raises(self, tmp_path):
        # check for the errors raised by extract here once as they are the same
        # in all other classes that actually modify, its just that we can not
        # instanstiate the ABC that implements the extract method that raises
        # these errors
        # check for error if both mda_trafos and mda_trafos_setup_func are given
        # if both ways of passing mda trafos are used simultaneously we
        # should get a ValueError from init
        with pytest.raises(ValueError):
            extractor = NoModificationFrameExtractor(
                mda_transformations=self.mda_trafos,
                mda_transformations_setup_func=self.mda_trafo_setup_func,
                                                     )
        # now check for raises from extract method
        extractor = NoModificationFrameExtractor()
        # extract the same frame twice to raise FileExistsError
        _ = extractor.extract(outfile=os.path.join(tmp_path, "out_frame.trr"),
                              traj_in=self.ala_traj,
                              idx=0,
                              )
        with pytest.raises(FileExistsError):
            _ = extractor.extract(outfile=os.path.join(tmp_path, "out_frame.trr"),
                                  traj_in=self.ala_traj,
                                  idx=0,
                                  )
        # check that it works if we pass overwrite=True
        _ = extractor.extract(outfile=os.path.join(tmp_path, "out_frame.trr"),
                              traj_in=self.ala_traj,
                              idx=0,
                              overwrite=True,
                              )
        # finally check for the FileNotFoundError raised if the structure file
        # does not exist
        with pytest.raises(FileNotFoundError):
            _ = extractor.extract(outfile=os.path.join(tmp_path, "out_frame2.trr"),
                                  traj_in=self.ala_traj,
                                  idx=0,
                                  struct_out=os.path.join(tmp_path, "does_not_exist.tpr"),
                                  )

    @pytest.mark.parametrize("idx", [0, 5, 10, 17])
    @pytest.mark.parametrize(["mda_transformations", "mda_transformations_setup_func"],
                             # they can not be both true at the same time (we test for the err above)
                             ([False, False],
                              [True, False],
                              [False, True],
                              )
                             )
    @pytest.mark.parametrize("use_async", [True, False])
    @pytest.mark.asyncio
    # pylint: disable-next=arguments-differ
    async def test_extract(
                self, idx, tmp_path, use_async,
                mda_transformations,  # whether we use simple mda trafos
                mda_transformations_setup_func,  # whether we use a setup func for more complex trafos
                           ):
        (all_pos_original,
         all_vels_original,
         all_atoms_extracted,
         ) = await super().test_extract(
                    idx=idx, tmp_path=tmp_path, use_async=use_async,
                    mda_transformations=mda_transformations,
                    mda_transformations_setup_func=mda_transformations_setup_func,
                    extractor_class=NoModificationFrameExtractor,
                    extractor_init_kwargs={},  # no additional init kwargs
                                        )
        # compare!
        assert np.allclose(all_pos_original,
                           all_atoms_extracted.positions,
                           )
        # and check for velocities too
        assert np.allclose(all_vels_original,
                           all_atoms_extracted.velocities,
                           )


class Test_InvertedVelocitiesFrameExtractor(TBase_FrameExtractors):
    @pytest.mark.parametrize("idx", [0, 5, 10, 17])
    @pytest.mark.parametrize(["mda_transformations", "mda_transformations_setup_func"],
                             # they can not be both true at the same time (we test for the err above)
                             ([False, False],
                              [True, False],
                              [False, True],
                              )
                             )
    @pytest.mark.parametrize("use_async", [True, False])
    @pytest.mark.asyncio
    # pylint: disable-next=arguments-differ
    async def test_extract(
                self, idx, tmp_path, use_async,
                mda_transformations,  # whether we use simple mda trafos
                mda_transformations_setup_func,  # whether we use a setup func for more complex trafos
                           ):
        (all_pos_original,
         all_vels_original,
         all_atoms_extracted,
         ) = await super().test_extract(
                    idx=idx, tmp_path=tmp_path, use_async=use_async,
                    mda_transformations=mda_transformations,
                    mda_transformations_setup_func=mda_transformations_setup_func,
                    extractor_class=InvertedVelocitiesFrameExtractor,
                    extractor_init_kwargs={},  # no additional init kwargs
                                        )
        # compare!
        # positions should be unmodified
        assert np.allclose(all_pos_original,
                           all_atoms_extracted.positions,
                           )
        # and check for inverted velocities too
        assert np.allclose(all_vels_original,
                           -1. * all_atoms_extracted.velocities,
                           )


class Test_RandomVelocitiesFrameExtractor(TBase_FrameExtractors):
    # NOTE: no need to test the full array of stuff we already tested for the
    #       other FrameExtractor subclasses
    @pytest.mark.parametrize("idx", [0, #5, 10, 17
                                     ]
                             )
    @pytest.mark.parametrize(["mda_transformations", "mda_transformations_setup_func"],
                             # they can not be both true at the same time (we test for the err above)
                             ([False, False],
                              #[True, False],
                              #[False, True],
                              )
                             )
    @pytest.mark.parametrize("use_async", [True, False])
    @pytest.mark.asyncio
    # pylint: disable-next=arguments-differ
    async def test_extract(
                self, idx, tmp_path, use_async,
                mda_transformations,  # whether we use simple mda trafos
                mda_transformations_setup_func,  # whether we use a setup func for more complex trafos
                           ):
        (all_pos_original,
         all_vels_original,
         all_atoms_extracted,
         ) = await super().test_extract(
                    idx=idx, tmp_path=tmp_path, use_async=use_async,
                    mda_transformations=mda_transformations,
                    mda_transformations_setup_func=mda_transformations_setup_func,
                    extractor_class=RandomVelocitiesFrameExtractor,
                    extractor_init_kwargs={"T": 303.},  # Temperature as init kwarg needed
                                        )
        # compare!
        # positions should be unmodified
        assert np.allclose(all_pos_original,
                           all_atoms_extracted.positions,
                           )
        # TODO: check for normal distribution of veloctities?!
        #       we could check for normality with scipy.stats.normaltest but
        #       that would only provide evidence against but not for the null
        #       hypothesis that the data is from a normal distribution
        #       also we would expect it to fail once in a while statistically...
        #       so for now we just ignore the velocities....
