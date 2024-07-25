import numpy as np
import mujoco
import cv2
import os
import pkg_resources
import time

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
    def __init__(self, modelPath: str = "diff.xml", render_mode: str = "" , videoFile: str = "video.mp4", height: int = 480, width:int =620):
        # Probably make it a bit more modular.
        DATA_PATH = pkg_resources.resource_filename('BlimpGymEnvironment',modelPath)
        self.m = mujoco.MjModel.from_xml_path(DATA_PATH)
        self.d = mujoco.MjData(self.m)
        # if render_mode == "human":
        self.renderer = mujoco.Renderer(self.m, height, width)
        self.render_mode : str = render_mode
        size = (620, 480)
        self.videoWriter = cv2.VideoWriter(videoFile, cv2.VideoWriter_fourcc(*'MJPG'), 60, size)
        self.waypoint = (1,1,1)
        self.terminationTime = 200
        self.startTime = time.time()



    def update_waypoint(self, waypoint:(int, int, int)):
        """
        Updates the waypoint of the system
        """
        self.waypoint = waypoint

    def update_termination_time(self, time:int):
        """
        Updates the termination time of the environment

        Arguments:
        time : times in seconds
        """
        self.terminationTime = time

    # TODO might need to add more sensor based on our real world sensor
    def get_obs(self):
        # Observation should return a values observed by the sensors
        self.renderer.update_scene(self.d, camera="blimpCamera")
        pixels = self.renderer.render()
        return [
                self.d.sensor("body_linacc").data.copy(),
                pixels.shape,
                pixels.flatten(),
                self.d.sensor("body_gyro").data.copy()
            ]


    # Get ground truth is used to get the ground truth from eiter the simulation or the
    # motion capture system
    def get_ground_truth(self):
        return [self.d.geom("controller").xpos,self.d.geom("controller").xmat]
        

    # Update data is a private function
    def _update_data(self,action):
        self.d.actuator('motor1').ctrl = [action[0]]
        self.d.actuator('motor2').ctrl = [action[1]]
        self.d.actuator('servo1').ctrl = [action[2]]
        self.d.actuator('servo2').ctrl = [action[3]]

    def reward_calculation(self) -> float:
        """
        Calculate reward

        Should override the function for different tasks while
        implementing the function.

        Return:
        reward - float 

        """

        loc = self.get_ground_truth()[0]

        err_x = loc[0] - self.waypoint[0] 
        err_y = loc[1] - self.waypoint[1] 
        err_z = loc[2] - self.waypoint[2] 
        
        return err_x + err_y + err_z


    def _termination(self) -> bool:
        """
        Implementation of the termination logic

        Returns
        terminated - if the simulation should terminate or not
        """

        return (time.time() - self.startTime) > self.terminationTime
        

    """
    Observation space is basically what the neural network observes.
    For our reward function we use more than just the observation space.
    """
    def step(self, a):
        # Observation space
        ob = self.get_obs()

        # TODO use the action
        self._update_data(a)
        
        loc = self.get_ground_truth()

        reward = self.reward_calculation()

        mujoco.mj_step(self.m, self.d) 

        terminated = self._termination()

        return (
            ob,
            reward,
            terminated,
            False,
        )


    def render(self):
        if self.render_mode == "human":
            self.renderer.update_scene(self.d)
            pixels = self.renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB) 
            self.videoWriter.write(pixels)
            cv2.imshow("blimp",pixels)
            cv2.waitKey(10)


        elif self.render_mode == "blimp":
            self.renderer.update_scene(self.d, camera="blimpCamera")
            pixels = self.renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB) 
            self.videoWriter.write(pixels)
            cv2.imshow("blimp",pixels)
            cv2.waitKey(10)
        else :
            self.renderer.update_scene(self.d, camera="followCamera")
            pixels = self.renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB) 
            self.videoWriter.write(pixels)
            cv2.imshow("blimp",pixels)
            cv2.waitKey(10)



    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.startTime = time.time()
        # TODO Add more info for rest
        return (self.get_obs(),[])
        
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5




