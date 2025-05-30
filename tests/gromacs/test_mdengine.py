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
import logging
import shutil

import numpy as np

from asyncmd.gromacs import GmxEngine, MDP
from asyncmd import Trajectory


# Decorator for test that need gmx grompp and/or gmx mdrun
needs_gmx_install = pytest.mark.skipif(shutil.which("gmx") is None,
                                       reason="Need gromacs (gmx) executable.")


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
        info_str = "Changing nsteps from 100 to -1 (infinte), the run "
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

    @pytest.mark.parametrize("conversion_factor", [-1., 1.1, 0.])
    def test_check_invalid_mdrun_time_conversion_factor(self, monkeypatch,
                                                        conversion_factor):
        # init should already fail
        with monkeypatch.context() as m:
            # monkeypatch so we dont need to find a gromacs executable
            m.setattr("asyncmd.gromacs.mdengine.ensure_executable_available",
                      lambda _: "/usr/bin/true")
            with pytest.raises(ValueError):
                engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                                   gro_file=self.gro,
                                   top_file=self.top,
                                   mdrun_time_conversion_factor=conversion_factor,
                                   )

    @pytest.mark.slow
    @needs_gmx_install
    @pytest.mark.parametrize("starting_conf",
                             [None,
                              Trajectory("tests/test_data/trajectory/ala_traj.trr",
                                         "tests/test_data/trajectory/ala.tpr")
                              ]
                             )
    @pytest.mark.asyncio
    async def test_run_MD_compressed_out(self, tmpdir, starting_conf):
        engine = GmxEngine(mdconfig=self.mdp_md_compressed_out,
                           gro_file=self.gro,
                           top_file=self.top)
        await engine.prepare(starting_configuration=starting_conf,
                             workdir=tmpdir,
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
    async def test_run_MD_full_prec_out(self, tmpdir, starting_conf):
        engine = GmxEngine(mdconfig=self.mdp_md_full_prec_out,
                           gro_file=self.gro,
                           top_file=self.top,
                           output_traj_type="trr")
        await engine.prepare(starting_configuration=starting_conf,
                             workdir=tmpdir,
                             deffnm="test")
        nsteps = 10
        traj = await engine.run(nsteps=nsteps)
        # some basic checks
        assert len(traj) == engine.nstout / nsteps + 1
        assert engine.steps_done == nsteps
        assert np.isclose(engine.time_done, nsteps * engine.dt)
