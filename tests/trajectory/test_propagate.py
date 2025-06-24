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
import os

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals

from asyncmd import Trajectory
from asyncmd import gromacs as asyncgmx
from asyncmd.trajectory import (PyTrajectoryFunctionWrapper,
                                InPartsTrajectoryPropagator,
                                ConditionalTrajectoryPropagator,
                                )

# marker for tests that need gromacs installed
from conftest import needs_gmx_install


def dummy_condition_func(traj, return_len, true_on_frames):
    """
    Construct a condition function return for ConditionalTrajectoryPropagators.

    Actually construct a fixed return value for the function according to given
    specifications.

    Parameters
    ----------
    return_len : int
        The number of frames/values the function returns.
    true_on_frames : list[int]
        Which of these frames should be true (all others will be false).
        Zero-indexed!
    """
    # NOTE: we add the return_len and true_on_frames (which fully specify the
    # return value) as call_kwargs, such that the cached values will always
    # match the specified return (since call_kwargs is part of the function
    # hash which we use to cache results)
    ret = np.full((return_len,), False)
    for frame in true_on_frames:
        ret[frame] = True
    return ret


def ala_alpha_R(traj, skip=1):
    """
    Calculate alpha_R state function for capped alaninie dipeptide.

    The alpha_R state is defined in the space of the two dihedral angles psi
    and phi, for a configuration to belong to the state:
        phi: -pi < phi < 0
        psi: -50 degree < psi < 30 degree

    Parameters
    ----------
    traj : asyncmd.Trajectory
        The trajectory for which the state function is calculated.
    skip : int, optional
        stride for trajectory iteration, by default 1

    Returns
    -------
    numpy.ndarray, shape=(n_frames,)
        Array with boolean values for every configuration on the trajectory
        indicating if a configuration falls into the state or not.
    """
    u = mda.Universe(traj.structure_file, *traj.trajectory_files)
    psi_ag = u.select_atoms("resname ALA and name N")  # idx 6
    psi_ag += u.select_atoms("resname ALA and name CA")  # idx 8
    psi_ag += u.select_atoms("resname ALA and name C")  # idx 14
    psi_ag += u.select_atoms("resname NME and name N")  # idx 16
    phi_ag = u.select_atoms("resname ACE and name C")  # idx 4
    phi_ag += u.select_atoms("resname ALA and name N")  # idx 6
    phi_ag += u.select_atoms("resname ALA and name CA")  # idx 8
    phi_ag += u.select_atoms("resname ALA and name C")  # idx 14
    # empty arrays to fill
    state = np.full((len(u.trajectory[::skip]),), False, dtype=bool)
    phi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # make sure MDAnalysis closes the underlying trajectory files directly
    u.trajectory.close()
    # phi: -pi -> 0
    # psi: > -50 but smaller 30 degree
    deg = 180/np.pi
    state[(phi <= 0) & (-50/deg <= psi) & (psi <= 30/deg)] = True
    return state


def ala_C7_eq(traj, skip=1):
    """
    Calculate C7_eq state function for capped alanine dipeptide.

    The C7_eq state is defined in the space of the two dihedral angles psi
    and phi, for a configuration to belong to the state:
        phi: -pi < phi < 0
        psi: 120 degree < psi < 200 degree

    Parameters
    ----------
    traj : asyncmd.Trajectory
        The trajectory for which the state function is calculated.
    skip : int, optional
        stride for trajectory iteration, by default 1

    Returns
    -------
    numpy.ndarray, shape=(n_frames,)
        Array with boolean values for every configuration on the trajectory
        indicating if a configuration falls into the state or not.
    """
    u = mda.Universe(traj.structure_file, *traj.trajectory_files)
    psi_ag = u.select_atoms("resname ALA and name N")  # idx 6
    psi_ag += u.select_atoms("resname ALA and name CA")  # idx 8
    psi_ag += u.select_atoms("resname ALA and name C")  # idx 14
    psi_ag += u.select_atoms("resname NME and name N")  # idx 16
    phi_ag = u.select_atoms("resname ACE and name C")  # idx 4
    phi_ag += u.select_atoms("resname ALA and name N")  # idx 6
    phi_ag += u.select_atoms("resname ALA and name CA")  # idx 8
    phi_ag += u.select_atoms("resname ALA and name C")  # idx 14
    # empty arrays to fill
    state = np.full((len(u.trajectory[::skip]),), False, dtype=bool)
    phi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # make sure MDAnalysis closes the underlying trajectory files directly
    u.trajectory.close()
    # phi: -pi -> 0
    # psi: 120 -> 200 degree
    deg = 180/np.pi
    state[(phi <= 0) & ((120/deg <= psi) | (-160/deg >= psi))] = True
    return state


class Test_InPartsTrajectoryPropagator:
    def setup_method(self):
        self.empty_mdp = asyncgmx.MDP("tests/test_data/gromacs/empty.mdp")
        self.xtc_mdp = asyncgmx.MDP("tests/test_data/gromacs/md_compressed_out.mdp")
        self.ala_gro = "tests/test_data/gromacs/conf.gro"
        self.ala_top = "tests/test_data/gromacs/topol_amber99sbildn.top"

    @pytest.mark.slow
    @needs_gmx_install
    @pytest.mark.parametrize(["n_steps", "continuation_n_steps"],
                             [(0, 0), (10, 10),
                              (0, 10), (0, 200),
                              (10, 0), (200, 0),
                              (10, 100),
                              (300, 100),
                              ])
    @pytest.mark.parametrize("starting_conf",
                             [Trajectory("tests/test_data/trajectory/ala_traj.trr",
                                         "tests/test_data/trajectory/ala.tpr")
                              ])
    @pytest.mark.asyncio
    async def test_run_continue_remove_with_gromacs(
                self, tmp_path, starting_conf, n_steps, continuation_n_steps,
                                                    ):
        propa = InPartsTrajectoryPropagator(
                        n_steps=n_steps,
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.xtc_mdp,
                                       "gro_file": self.ala_gro,
                                       "top_file": self.ala_top,
                                       },
                        walltime_per_part=1 / (60 * 60 * 100),  # 0.01 s
                                            )
        traj = await propa.propagate_and_concatenate(
                                starting_configuration=starting_conf,
                                workdir=tmp_path,
                                deffnm="test",
                                tra_out=os.path.join(tmp_path, "out_traj.xtc"),
                                )
        if n_steps == 0:
            assert traj is None
        else:
            assert len(traj) == ((n_steps / self.xtc_mdp["nstxout-compressed"])
                                 + 1)
        # reset n_steps
        propa.n_steps = continuation_n_steps
        # and do a continuation
        traj_cont = await propa.propagate_and_concatenate(
                            starting_configuration=starting_conf,
                            workdir=tmp_path,
                            deffnm="test",
                            tra_out=os.path.join(tmp_path, "out_traj2.xtc"),
                            continuation=True,
                            )
        if continuation_n_steps == 0:
            assert traj_cont is None
        else:
            assert len(traj_cont) == ((continuation_n_steps
                                       / self.xtc_mdp["nstxout-compressed"]
                                       )
                                      + 1)
        # Tests for remove files
        # we have the two mdp-files in the directory (the one we write and the
        # mdp.out that gmx grompp writes),
        # the tpr file should also be there in any case
        n_files_beauty = 3
        # and we should have 0-2 concatenated out-trajs in the directory
        # depending on how many times we ran with n_steps != 0
        # for each run/out_traj we the hidden offsets file and the hidden lock
        # file from mdanalysis
        n_files_beauty += sum(3 * int(steps != 0)
                              for steps in [n_steps, continuation_n_steps]
                              )
        # the times we actually did a call to gmx mdrun, i.e. for the cases
        # where nsteps > 0 (for the first run) and nsteps < continuation_n_steps
        # (for the second run) we also have the confout.gro
        n_files_beauty += (int(n_steps != 0)  # first round
                           + int(n_steps < continuation_n_steps)  # second run
                           )
        if n_steps == 0:
            # in this case we have two tpr files because the second run will
            # not find any trajectories and call grompp again, gmx grompp will
            # move the old tpr and generate a new one
            n_files_beauty += 1
        await propa.remove_parts(workdir=tmp_path, deffnm="test",
                                 file_endings_to_remove=[
                                     # trajs and log are the default
                                     "trajectories", "log",
                                     # remove edr files so we can count easily
                                     "edr",
                                                         ]
                                 )
        all_files = os.listdir(tmp_path)
        # we filter out the checkpoint files because we can not know if there
        # is one or two (i.e. if we ran long enough to write a `test_prev.cpt`)
        all_files = [f for f in all_files if not f.lower().endswith(".cpt")]
        assert len(all_files) == n_files_beauty


class Test_ConditionalPropagator:
    def setup_method(self):
        self.empty_mdp = asyncgmx.MDP("tests/test_data/gromacs/empty.mdp")
        self.cond_wrap_1frame_true1 = PyTrajectoryFunctionWrapper(
                                            function=dummy_condition_func,
                                            call_kwargs={"return_len": 1,
                                                         "true_on_frames": [0],
                                                         }
                                                                  )
        self.cond_wrap_1frame_never_true = PyTrajectoryFunctionWrapper(
                                            function=dummy_condition_func,
                                            call_kwargs={"return_len": 1,
                                                         "true_on_frames": [],
                                                         }
                                                                       )
        self.xtc_mdp = asyncgmx.MDP("tests/test_data/gromacs/md_compressed_out.mdp")
        self.ala_gro = "tests/test_data/gromacs/conf.gro"
        self.ala_top = "tests/test_data/gromacs/topol_amber99sbildn.top"
        self.ala_conf_in_no_state = Trajectory("tests/test_data/trajectory/ala_conf_in_no_state.trr",
                                               "tests/test_data/trajectory/ala.tpr"
                                               )
        self.ala_cond_func_alphaR = PyTrajectoryFunctionWrapper(
                                        ala_alpha_R, call_kwargs={"skip": 1},
                                                                )
        self.ala_cond_func_C7eq = PyTrajectoryFunctionWrapper(
                                        ala_C7_eq, call_kwargs={"skip": 1},
                                                              )

    @pytest.mark.parametrize(["max_frames", "max_steps", "beauty_max_steps"],
                             [(None, None, float("inf")),  # no max steps/frames
                              (200, 10, 10),  # max_steps takes precedence
                              (10, None, 10),  # nstout=1, max_steps=max_frames
                              (None, 10, 10),
                              ])
    def test_init_max_steps_vs_max_frames(self, monkeypatch, max_frames,
                                          max_steps, beauty_max_steps):
        with monkeypatch.context() as m:
            # monkeypatch to make sure nstout will be equal to 1
            m.setattr("asyncmd.trajectory.propagate.nstout_from_mdconfig",
                      lambda mdconfig, output_traj_type: 1)
            propa = ConditionalTrajectoryPropagator(
                        conditions=[self.cond_wrap_1frame_true1,
                                    self.cond_wrap_1frame_never_true],
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.empty_mdp},
                        walltime_per_part=0.01,
                        max_steps=max_steps,
                        max_frames=max_frames,
                                                    )
        assert propa.max_steps == beauty_max_steps

    def test_conditions_setter_no_condition_raises(self, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr("asyncmd.trajectory.propagate.nstout_from_mdconfig",
                      lambda mdconfig, output_traj_type: 1)
            propa = ConditionalTrajectoryPropagator(
                        conditions=[self.cond_wrap_1frame_true1],
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.empty_mdp},
                        walltime_per_part=0.01,
                                                    )
        with pytest.raises(ValueError):
            propa.conditions = []

    def test_conditions_setter_not_all_coro(self, monkeypatch, caplog):
        with monkeypatch.context() as m:
            m.setattr("asyncmd.trajectory.propagate.nstout_from_mdconfig",
                      lambda mdconfig, output_traj_type: 1)
            propa = ConditionalTrajectoryPropagator(
                        conditions=[self.cond_wrap_1frame_true1],
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.empty_mdp},
                        walltime_per_part=0.01,
                                                    )
        with caplog.at_level(logging.WARNING):
            # just add one func that is not a coroutine to the list
            propa.conditions = (propa.conditions
                                + [lambda traj: dummy_condition_func(
                                                    traj,
                                                    return_len=1,
                                                    true_on_frames=[],
                                                                     )
                                   ])
        # NOTE: not the full warn string, but the important part ;)
        warn_str = "It is recommended to use coroutinefunctions for all conditions."
        assert warn_str in caplog.text

    @pytest.mark.parametrize(["conditions", "error"],
                             # NOTE: the trajectory we apply the func to has
                             # length 18 frames
                             [
                              # two wrapped conditions
                              ([PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [17]}),
                                PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": []}),
                                ],
                               False
                               ),
                              # three wrapped conditions
                              ([PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [16]}),
                                PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [17]}),
                                PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": []}),
                                ],
                               False
                               ),
                              # one wrapped func, one blocking/non-wrapped func
                              ([PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [17]}),
                                lambda traj: dummy_condition_func(
                                                    traj,
                                                    return_len=18,
                                                    true_on_frames=[],
                                                                  )
                                ],
                               False
                               ),
                              # two blocking/non-wrapped func
                              ([lambda traj: dummy_condition_func(
                                                    traj,
                                                    return_len=18,
                                                    true_on_frames=[12],
                                                                  ),
                                lambda traj: dummy_condition_func(
                                                    traj,
                                                    return_len=18,
                                                    true_on_frames=[],
                                                                  )
                                ],
                               False
                               ),
                              # These two funcs do not have the same return len
                              # so we expect a (hopefully helpful) error
                              # that the length does not match the trajectory
                              ([PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [1]}),
                                PyTrajectoryFunctionWrapper(
                                    dummy_condition_func,
                                  call_kwargs={"return_len": 2,
                                               "true_on_frames": []}),
                                ],
                               True
                               ),
                              ]
                             )
    @pytest.mark.asyncio
    async def test__condition_vals_for_traj(self, monkeypatch, conditions, error):
        with monkeypatch.context() as m:
            m.setattr("asyncmd.trajectory.propagate.nstout_from_mdconfig",
                      lambda mdconfig, output_traj_type: 1)
            propa = ConditionalTrajectoryPropagator(
                        conditions=conditions,
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.empty_mdp},
                        walltime_per_part=0.01,
                                                    )
        traj = Trajectory(
                    trajectory_files="tests/test_data/trajectory/ala_traj.xtc",
                    structure_file="tests/test_data/trajectory/ala.tpr"
                          )
        if error:
            with pytest.raises(ValueError):
                _ = await propa._condition_vals_for_traj(traj=traj)
        else:
            condition_vals = await propa._condition_vals_for_traj(traj=traj)
            assert len(condition_vals) == len(propa.conditions)

    @pytest.mark.asyncio
    async def test__condition_vals_for_traj_single_condition_changed(
                        self, monkeypatch,
                                                                      ):
        conditions = [PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [17]}),
                      PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": []}),
                      ]
        with monkeypatch.context() as m:
            m.setattr("asyncmd.trajectory.propagate.nstout_from_mdconfig",
                      lambda mdconfig, output_traj_type: 1)
            propa = ConditionalTrajectoryPropagator(
                        conditions=conditions,
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.empty_mdp},
                        walltime_per_part=0.01,
                                                    )
        # change one of the conditions
        propa.conditions[1] = PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [1]})
        traj = Trajectory(
                    trajectory_files="tests/test_data/trajectory/ala_traj.xtc",
                    structure_file="tests/test_data/trajectory/ala.tpr"
                          )
        # and call it
        condition_vals = await propa._condition_vals_for_traj(traj=traj)
        assert len(condition_vals) == len(propa.conditions)

    @pytest.mark.asyncio
    async def test_propagate(self):
        mdp = asyncgmx.MDP("tests/test_data/gromacs/empty.mdp")
        mdp["nstxout-compressed"] = 1
        mdp["nstxtcout"] = 1

        condition_function_wrapped = PyTrajectoryFunctionWrapper(
            dummy_condition_func,
            call_kwargs={"return_len": 18,
                         "true_on_frames": [17]}
        )
        propa_somewhere = ConditionalTrajectoryPropagator(
            conditions=[condition_function_wrapped],
            engine_cls=asyncgmx.GmxEngine,
            engine_kwargs={
                "mdconfig": mdp,
            },
            walltime_per_part=0.01,
        )
        starting_configuration = Trajectory(
            trajectory_files="tests/test_data/trajectory/ala_traj.xtc",
            structure_file="tests/test_data/trajectory/ala.gro",
        )
        workdir = "tests/trajectory"
        deffnm = "test_deffnm"

        trajectories, cond_fullfilled = await propa_somewhere.propagate(
            starting_configuration=starting_configuration, workdir=workdir, deffnm=deffnm
        )

        assert len(trajectories) > 0
        assert cond_fullfilled is not None

    @pytest.mark.slow
    @needs_gmx_install
    @pytest.mark.asyncio
    async def test_run_continue_with_gromacs(self, tmp_path):
        conditions = [self.ala_cond_func_alphaR, self.ala_cond_func_C7eq]
        propa = ConditionalTrajectoryPropagator(
                        conditions=conditions,
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.xtc_mdp,
                                       "gro_file": self.ala_gro,
                                       "top_file": self.ala_top,
                                       },
                        walltime_per_part=1 / (60 * 60 * 100),  # 0.01 s
                                                )
        traj, state_reached = await propa.propagate_and_concatenate(
                                    starting_configuration=self.ala_conf_in_no_state,
                                    workdir=tmp_path,
                                    deffnm="test",
                                    tra_out=os.path.join(tmp_path, "test_out.xtc"),
                                                                    )
        # some basic check(s)
        cond_vals = await conditions[state_reached](traj)
        assert cond_vals[-1]
        assert not any(cond_vals[:-1])
        # and "continue"
        traj2, state_reached2 = await propa.propagate_and_concatenate(
                                    starting_configuration=self.ala_conf_in_no_state,
                                    workdir=tmp_path,
                                    deffnm="test",
                                    tra_out=os.path.join(tmp_path, "test_out2.xtc"),
                                    continuation=True
                                                                      )
        # again some basic check(s)
        assert state_reached2 == state_reached  # consistency :)
        # and this should also still be true...
        cond_vals = await conditions[state_reached](traj2)
        assert cond_vals[-1]
        assert not any(cond_vals[:-1])
