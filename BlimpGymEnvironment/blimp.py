import numpy as np
import mujoco
import cv2
import os
import pkg_resources

class Blimp():
    metadata = {
        "render_modes": [
            "human",
            "blimp",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }



    # TODO add action space
    def __init__(self, modelPath: str = "diff.xml", render_mode: str = ""):
        # Probably make it a bit more modular.
        DATA_PATH = pkg_resources.resource_filename('BlimpGymEnvironment',modelPath)
        self.m = mujoco.MjModel.from_xml_path(DATA_PATH)
        self.d = mujoco.MjData(self.m)
        # if render_mode == "human":
        self.renderer = mujoco.Renderer(self.m)
        self.render_mode : str = render_mode

    # TODO might need to add more sensor based on our real world sensor
    def get_obs(self):

        self.renderer.update_scene(self.d, camera="blimpCamera")
        pixels = self.renderer.render()
        return [
                self.d.geom("mylar").xpos,
                pixels.shape,
                pixels.flatten()
            ]


    # Update data is a private function
    def _update_data(self,action):
        self.d.actuator('motor1').ctrl = [action[0]]
        self.d.actuator('motor2').ctrl = [action[1]]
        self.d.actuator('servo1').ctrl = [action[2]]
        self.d.actuator('servo2').ctrl = [action[3]]


    """
    Observation space is basically what the neural network observes.
    For our reward function we use more than just the observation space.
    """
    def step(self, a):
        # Observation space

        observation = self.get_obs()

        # TODO use the action
        self._update_data(a)
        
        t1 = self.d.time
        pos_before = self.d.geom('mylar').xpos
        step = 10 # compute the reward function after 10 steps
        for i in range(step):
            mujoco.mj_step(self.m, self.d) 
        pos_after = self.d.geom('mylar').xpos
        t2 = self.d.time



        # print(forward_reward)
        #reward = pos_after[0] + pos_after[1] + pos_after[2]
        reward = -(abs(observation[0][0]) + abs(observation[0][1]) + abs(observation[0][2]))
        # for i in pos_after:
        #     if i > 0.3:
        #         reward = reward - 1
        #     else:
        #         reward = reward + 1
        # print(reward)

        state = observation
        # print(pos_after)

        # TODO Come up a condition for termination
        terminated = True if (pos_after[0] > 1 or pos_after[0] <-1 or pos_after[1]>1 or pos_after[1]<-1 or pos_after[2]>50 or reward < -2 ) else False# Keeping it true for now
       # terminated = not not_terminated
        # print(terminated )
        ob = state

        if self.render_mode == "human":
            self.renderer.update_scene(self.d)
            pixels = self.renderer.render()
            cv2.imshow("blimp",pixels)
            cv2.waitKey(10)

        if self.render_mode == "blimp":
            self.renderer.update_scene(self.d, camera="blimpCamera")
            pixels = self.renderer.render()
            cv2.imshow("blimp",pixels)
            cv2.waitKey(10)
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            ob,
            reward,
            terminated,
            False,
        )



    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        # TODO Add more info for rest
        return (self.get_obs(),[])
        
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5




