import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box


class BlimpEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self, "blimp.xml", 5, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, a):
        pos_before = self.get_body_com("")
        self.do_simulate(a, self.frame_skip)
        pos_after = self.get_body_com("")

        forward_reward = (xposafter - xposbefore) / self.dt
        # TODO update the state_vector 
        state = self.state_vector()
        not_terminated = (
            np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        )
        terminated = not not_terminated
        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            ob,
            reward,
            terminated,
            False,
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5


    
