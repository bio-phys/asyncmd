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

import logging

import numpy as np
import pytest

from asyncmd import Trajectory
from asyncmd import gromacs as asyncgmx
from asyncmd import trajectory as asynctraj


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


class Test_ConditionalPropagator:
    def setup_method(self):
        self.empty_mdp = asyncgmx.MDP("tests/test_data/gromacs/empty.mdp")
        self.cond_wrap_1frame_true1 = asynctraj.PyTrajectoryFunctionWrapper(
                                            function=dummy_condition_func,
                                            call_kwargs={"return_len": 1,
                                                         "true_on_frames": [0],
                                                         }
                                                                            )
        self.cond_wrap_1frame_never_true = asynctraj.PyTrajectoryFunctionWrapper(
                                            function=dummy_condition_func,
                                            call_kwargs={"return_len": 1,
                                                         "true_on_frames": [],
                                                         }
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
            propa = asynctraj.ConditionalTrajectoryPropagator(
                        conditions=[self.cond_wrap_1frame_true1,
                                    self.cond_wrap_1frame_never_true],
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.empty_mdp},
                        walltime_per_part=0.01,
                        max_steps=max_steps,
                        max_frames=max_frames,
                                                              )
        assert propa.max_steps == beauty_max_steps

    def test_conditions_setter_not_all_coro(self, monkeypatch, caplog):
        with monkeypatch.context() as m:
            m.setattr("asyncmd.trajectory.propagate.nstout_from_mdconfig",
                      lambda mdconfig, output_traj_type: 1)
            propa = asynctraj.ConditionalTrajectoryPropagator(
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
                              ([asynctraj.PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [17]}),
                                asynctraj.PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": []}),
                                ],
                               False
                               ),
                              # three wrapped conditions
                              ([asynctraj.PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [16]}),
                                asynctraj.PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [17]}),
                                asynctraj.PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": []}),
                                ],
                               False
                               ),
                              # one wrapped func, one blocking/non-wrapped func
                              ([asynctraj.PyTrajectoryFunctionWrapper(
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
                              ([asynctraj.PyTrajectoryFunctionWrapper(
                                  dummy_condition_func,
                                  call_kwargs={"return_len": 18,
                                               "true_on_frames": [1]}),
                                asynctraj.PyTrajectoryFunctionWrapper(
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
            propa = asynctraj.ConditionalTrajectoryPropagator(
                        conditions=conditions,
                        engine_cls=asyncgmx.GmxEngine,
                        engine_kwargs={"mdconfig": self.empty_mdp},
                        walltime_per_part=0.01,
                                                                )
        traj = Trajectory(
                    trajectory_files="tests/test_data/trajectory/ala_traj.xtc",
                    structure_file="tests/test_data/trajectory/ala.tpr"
                          )
        print("traj len", len(traj))
        if error:
            with pytest.raises(ValueError):
                _ = await propa._condition_vals_for_traj(traj=traj)
        else:
            condition_vals = await propa._condition_vals_for_traj(traj=traj)
            assert len(condition_vals) == len(propa.conditions)


@pytest.mark.asyncio
async def test_propagate():
    mdp = asyncgmx.MDP("tests/test_data/gromacs/empty.mdp")
    mdp["nstxout-compressed"] = 1
    mdp["nstxtcout"] = 1

    condition_function_wrapped = asynctraj.PyTrajectoryFunctionWrapper(
        dummy_condition_func,
        call_kwargs={"return_len": 18,
                     "true_on_frames": [17]}
    )
    propa_somewhere = asynctraj.ConditionalTrajectoryPropagator(
        conditions=[condition_function_wrapped],
        engine_cls=asyncgmx.GmxEngine,
        engine_kwargs={
            "mdconfig": mdp,
        },
        walltime_per_part=0.01,
    )
    starting_configuration = asynctraj.trajectory.Trajectory(
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
