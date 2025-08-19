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
import logging

import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations

from asyncmd.gromacs import GmxEngine, MDP
from asyncmd import Trajectory

# custom decorator for tests that need gromacs installed
from conftest import needs_gmx_install


class Test_GmxEngine:
    def setup_method(self):
        # some useful files
        self.gro = "tests/test_data/gromacs/conf.gro"
        self.ndx = "tests/test_data/gromcas/index.ndx"
        self.top = "tests/test_data/gromacs/topol_amber99sbildn.top"
        self.mdp_md_compressed_out = MDP("tests/test_data/gromacs/md_compressed_out.mdp")
        self.mdp_md_full_prec_out = MDP("tests/test_data/gromacs/md_full_prec_out.mdp")

    @pytest.mark.parametrize("integrator", ["steep", "cg", "l-bfgs"])
    def test_mpd_check_minimize(self, integrator, monkeypatch):
        # init an engine so we can use its mdconfig property (which does the checks)
        with monkeypatch.context() as m:
            # monkeypatch so we dont need to find a gromacs executable
            m.setattr("asyncmd.gromacs.mdengine.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                               gro_file=self.gro,
                               top_file=self.top)
        self.mdp_md_compressed_out["integrator"] = integrator
        with pytest.raises(ValueError):
            engine.mdp = self.mdp_md_compressed_out
        # this should work
        engine.output_traj_type = "trr"
        self.mdp_md_full_prec_out["integrator"] = integrator
        engine.mdp = self.mdp_md_full_prec_out

    def test_mdp_check_nsteps(self, caplog, monkeypatch):
        # init an engine so we can use its mdconfig property (which does the checks)
        with monkeypatch.context() as m:
            # monkeypatch so we dont need to find a gromacs executable
            m.setattr("asyncmd.gromacs.mdengine.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                               gro_file=self.gro,
                               top_file=self.top)
        # check that an nsteps value that is not -1 (but set) is changed
        self.mdp_md_compressed_out["nsteps"] = 100
        with caplog.at_level(logging.INFO):
            engine.mdp = self.mdp_md_compressed_out
        # make sure nsteps is now set to -1
        assert engine.mdp["nsteps"] == -1
        # and check the log
        info_str = "Changing nsteps from 100 to -1 (infinite), the run "
        info_str += "length is controlled via arguments of the run method."
        assert info_str in caplog.text

        # check that we set nsteps if it is unset
        del self.mdp_md_compressed_out["nsteps"]
        with caplog.at_level(logging.INFO):
            engine.mdp = self.mdp_md_compressed_out
        # make sure nsteps is now set to -1
        assert engine.mdp["nsteps"] == -1
        # and check the log
        info_str = "Setting previously undefined nsteps to -1 (infinite)."
        assert info_str in caplog.text

    def test_mdp_check_no_mdp_class(self, monkeypatch):
        # init should already fail
        with monkeypatch.context() as m:
            # monkeypatch so we dont need to find a gromacs executable
            m.setattr("asyncmd.gromacs.mdengine.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            with pytest.raises(TypeError):
                engine = GmxEngine(mdconfig=None,
                                   gro_file=self.gro,
                                   top_file=self.top)

    @pytest.mark.parametrize(["conversion_factor", "raises"],
                             [(-1., True),
                              (1.1, True),
                              (0., True),
                              (0.9, False),
                              (0.1, False),
                              ]
                             )
    def test_check_valid_and_invalid_mdrun_time_conversion_factor(
                    self, monkeypatch, conversion_factor, raises,
                    ):
        with monkeypatch.context() as m:
            # monkeypatch so we dont need to find a gromacs executable
            m.setattr("asyncmd.gromacs.mdengine.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            if raises:
                # init should already fail
                with pytest.raises(ValueError):
                    engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                                       gro_file=self.gro,
                                       top_file=self.top,
                                       mdrun_time_conversion_factor=conversion_factor,
                                       )
            else:
                engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                                   gro_file=self.gro,
                                   top_file=self.top,
                                   mdrun_time_conversion_factor=conversion_factor,
                                   )
                assert engine.mdrun_time_conversion_factor == conversion_factor

    @pytest.mark.parametrize(["output_traj_type", "raises"],
                             [("NOT_A_TRAJ_TYPE", True),
                              ("also not a traj type", True),
                              ("neither", True),
                              ("xtc", False),
                              ("XTC", False),
                              ("trr", False),
                              ("TRR", False),
                              ]
                             )
    def test_check_valid_and_invalid_output_traj_type(
                    self, monkeypatch, output_traj_type, raises,
                    ):
        if output_traj_type.lower() == "trr":
            # use a MDP that has trajectory output configured for TRR
            mdconfig = self.mdp_md_full_prec_out
        else:
            # use a MDP that has trajectory output configured for XTC
            # for all other output_traj_type values (including "xtc")
            mdconfig = self.mdp_md_compressed_out
        with monkeypatch.context() as m:
            # monkeypatch so we dont need to find a gromacs executable
            m.setattr("asyncmd.gromacs.mdengine.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            if raises:
                with pytest.raises(ValueError):
                    # init should already fail
                    engine = GmxEngine(mdconfig=mdconfig,
                                       gro_file=self.gro,
                                       top_file=self.top,
                                       output_traj_type=output_traj_type,
                                       )
            else:
                engine = GmxEngine(mdconfig=mdconfig,
                                   gro_file=self.gro,
                                   top_file=self.top,
                                   output_traj_type=output_traj_type,
                                   )
                assert engine.output_traj_type == output_traj_type.lower()

    @pytest.mark.slow
    @needs_gmx_install
    @pytest.mark.parametrize("starting_conf",
                             [None,
                              Trajectory("tests/test_data/trajectory/ala_traj.trr",
                                         "tests/test_data/trajectory/ala.tpr")
                              ]
                             )
    @pytest.mark.asyncio
    async def test_run_MD_compressed_out(self, tmp_path, starting_conf):
        engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                           gro_file=self.gro,
                           top_file=self.top)
        await engine.prepare(starting_configuration=starting_conf,
                             workdir=tmp_path,
                             deffnm="test")
        nsteps = 10
        traj = await engine.run(nsteps=nsteps)
        # some basic checks
        assert len(traj) == engine.nstout / nsteps + 1
        assert engine.steps_done == nsteps
        assert np.isclose(engine.time_done, nsteps * engine.dt)

    @pytest.mark.slow
    @needs_gmx_install
    @pytest.mark.parametrize("starting_conf",
                             [None,
                              Trajectory("tests/test_data/trajectory/ala_traj.trr",
                                         "tests/test_data/trajectory/ala.tpr")
                              ]
                             )
    @pytest.mark.asyncio
    async def test_run_MD_full_prec_out(self, tmp_path, starting_conf):
        engine = GmxEngine(mdconfig=self.mdp_md_full_prec_out,
                           gro_file=self.gro,
                           top_file=self.top,
                           output_traj_type="trr")
        await engine.prepare(starting_configuration=starting_conf,
                             workdir=tmp_path,
                             deffnm="test")
        nsteps = 10
        traj = await engine.run(nsteps=nsteps)
        # some basic checks
        assert len(traj) == engine.nstout / nsteps + 1
        assert engine.steps_done == nsteps
        assert np.isclose(engine.time_done, nsteps * engine.dt)

    @pytest.mark.slow
    @needs_gmx_install
    @pytest.mark.asyncio
    async def test_generate_velocities(self, tmp_path):
        initial_conf = Trajectory("tests/test_data/trajectory/ala_traj.trr",
                                  "tests/test_data/trajectory/ala.tpr")
        engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                           gro_file=self.gro,
                           top_file=self.top)
        new_conf = await engine.generate_velocities(
                                    conf_in=initial_conf,
                                    conf_out_name=os.path.join(tmp_path, "out.trr"),
                                    workdir=tmp_path,
                                    )
        # TODO: this is essentially a smoke test...
        #       ... what else can we test for except the length?!
        assert len(new_conf) == 1

    @pytest.mark.slow
    @needs_gmx_install
    @pytest.mark.asyncio
    async def test_apply_constraints(self, tmp_path):
        initial_conf = Trajectory("tests/test_data/trajectory/ala_traj.trr",
                                  "tests/test_data/trajectory/ala.tpr")
        engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                           gro_file=self.gro,
                           top_file=self.top)
        new_conf = await engine.apply_constraints(
                                    conf_in=initial_conf,
                                    conf_out_name=os.path.join(tmp_path, "out.trr"),
                                    workdir=tmp_path,
                                    )
        # TODO: this is essentially a smoke test...
        #       ... what else can we test for except the length?!
        assert len(new_conf) == 1
