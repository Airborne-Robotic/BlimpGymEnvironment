import numpy as np
import mujoco
import cv2
import os
import pkg_resources

import glob
import torch
import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

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
    def __init__(self, modelPath: str = "sano.xml", render_mode: str = ""):
        # Probably make it a bit more modular.
        # DATA_PATH = pkg_resources.resource_filename('BlimpGymEnvironment',modelPath)
        # self.m = mujoco.MjModel.from_xml_path(DATA_PATH)
        self.m = mujoco.MjModel.from_xml_path(modelPath)
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
        self.d.actuator('prop_joint').ctrl = [ action[0]]
        self.d.actuator('prop_joint2').ctrl = [ action[1]]
        self.d.actuator('prop_joint3').ctrl = [ action[2]]
        self.d.actuator('prop_joint4').ctrl = [ action[3]]
        self.d.actuator('prop_joint5').ctrl = [ action[4]]
        self.d.actuator('prop_joint6').ctrl = [ action[5]]


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




if __name__ == '__main__':
    env = Blimp("sano.xml", render_mode="blimp")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)


    model = DPTDepthModel(
        path="DPT_hybrid.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    net_w = net_h = 384
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()


    a = [0,0,0,0,0,0]
    while True:
        ob, reward, terminated,_ = env.step(a)
        img = ob[2].reshape(ob[1])
        img_input = transform({"image": img})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            util.io.disp_depth("out",prediction,bits=2)
            # print(prediction)

        # cv2.waitKey(10)
        # print(ob[0])
        # print(ob)
