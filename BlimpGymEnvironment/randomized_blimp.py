"""
Enhanced Blimp Environment with Randomization Support

This module extends the base Blimp class to support comprehensive
environment randomization for robust training.
"""

import numpy as np
import mujoco
import cv2
import time
from typing import Optional, Dict, List, Tuple
import pkg_resources

from BlimpGymEnvironment.environment_randomizer import (
    EnvironmentRandomizer,
    ObstacleCourseGenerator,
    WeatherRandomizer,
)


class RandomizedBlimp:
    """
    Enhanced Blimp environment with full randomization support.

    Features:
    - Randomized waypoints
    - Procedural obstacle courses
    - Dynamic weather conditions
    - Physics domain randomization
    - Multiple training scenarios
    """

    metadata = {
        "render_modes": [
            "human",
            "blimp",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        modelPath: str = "diff.xml",
        render_mode: str = "",
        videoFile: str = "video.mp4",
        height: int = 480,
        width: int = 620,
        randomize: bool = True,
        randomization_config: Optional[Dict] = None,
        course_type: Optional[str] = None,
        weather_preset: str = "calm",
        seed: Optional[int] = None,
    ):
        """
        Initialize the randomized blimp environment.

        Args:
            modelPath: Path to base XML model
            render_mode: Rendering mode
            videoFile: Output video file path
            height: Render height
            width: Render width
            randomize: Whether to enable randomization
            randomization_config: Custom randomization configuration
            course_type: Type of obstacle course ('corridor', 'slalom', 'tower', 'maze', None)
            weather_preset: Weather conditions ('calm', 'windy', 'turbulent', 'dense', 'thin')
            seed: Random seed for reproducibility
        """
        # Get full path to model
        DATA_PATH = pkg_resources.resource_filename("BlimpGymEnvironment", modelPath)

        self.base_xml_path = DATA_PATH
        self.randomize_enabled = randomize
        self.course_type = course_type
        self.weather_preset = weather_preset
        self.seed = seed

        # Initialize tracking variables
        self.episode_count = 0
        self.randomization_history = []

        # Initialize randomizer
        if randomize:
            self.randomizer = EnvironmentRandomizer(DATA_PATH, seed=seed)
            if randomization_config:
                self.randomizer.update_config(randomization_config)
        else:
            self.randomizer = None

        # Generate initial environment
        if randomize:
            self.m, self.d = self._generate_randomized_environment()
        else:
            self.m = mujoco.MjModel.from_xml_path(DATA_PATH)
            self.d = mujoco.MjData(self.m)

        # Rendering setup
        self.render_mode = render_mode
        if render_mode != "":
            self.renderer = mujoco.Renderer(self.m, height, width)
            size = (width, height)
            self.videoWriter = cv2.VideoWriter(
                videoFile, cv2.VideoWriter_fourcc(*"MJPG"), 60, size
            )

        # Environment state
        self.waypoint = self._get_current_waypoint()
        self.terminationTime = 200
        self.startTime = time.time()

    def _generate_randomized_environment(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Generate a new randomized environment."""
        if self.randomizer is None:
            # Fallback to non-randomized
            m = mujoco.MjModel.from_xml_path(self.base_xml_path)
            d = mujoco.MjData(m)
            return m, d

        # Generate base randomized environment
        m, d = self.randomizer.generate_environment(
            randomize_waypoint=True,
            randomize_obstacles=(self.course_type is None),
            randomize_physics=True,
            randomize_lighting=True,
        )

        # Apply weather preset
        WeatherRandomizer.apply_weather(m, self.weather_preset, randomize=True)

        # Store randomization info
        self.randomization_history.append(
            {
                "episode": self.episode_count,
                "waypoint": self._get_current_waypoint(),
                "viscosity": m.opt.viscosity,
                "wind": m.opt.wind.copy(),
                "density": m.opt.density,
            }
        )

        return m, d

    def _get_current_waypoint(self) -> Tuple[float, float, float]:
        """Get current waypoint position."""
        try:
            waypoint_id = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_GEOM, "waypoint"
            )
            if waypoint_id >= 0:
                return tuple(self.m.geom_pos[waypoint_id])
        except:
            pass
        return (1, 1, 1)  # Default

    def update_waypoint(self, waypoint: Tuple[float, float, float]):
        """Update the waypoint position."""
        try:
            waypoint_id = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_GEOM, "waypoint"
            )
            if waypoint_id >= 0:
                self.m.geom_pos[waypoint_id] = waypoint
                self.waypoint = waypoint
        except:
            pass

    def randomize_waypoint(self):
        """Randomize waypoint position on the fly."""
        if self.randomizer:
            x = np.random.uniform(*self.randomizer.config["waypoint"]["x_range"])
            y = np.random.uniform(*self.randomizer.config["waypoint"]["y_range"])
            z = np.random.uniform(*self.randomizer.config["waypoint"]["z_range"])
            self.update_waypoint((x, y, z))

    def update_termination_time(self, time_val: int):
        """Update the termination time."""
        self.terminationTime = time_val

    def get_obs(self):
        """Get observations from the environment."""
        if hasattr(self, "renderer"):
            self.renderer.update_scene(self.d, camera="blimpCamera")
            pixels = self.renderer.render()
        else:
            pixels = None

        return [
            self.d.geom("controller").xpos,
            pixels,
            self.d.sensor("body_gyro").data.copy(),
        ]

    def get_ground_truth(self):
        """Get ground truth state."""
        return [self.d.geom("controller").xpos, self.d.geom("controller").xmat]

    def _update_data(self, action):
        """Update actuator controls."""
        self.d.actuator("motor1").ctrl = [2 * action[0]]
        self.d.actuator("motor2").ctrl = [2 * action[1]]
        self.d.actuator("servo1").ctrl = [action[2]]
        self.d.actuator("servo2").ctrl = [action[3]]

    def reward_calculation(self) -> float:
        """Calculate reward based on distance to waypoint."""
        loc = self.get_ground_truth()[0]
        err_x = loc[0] - self.waypoint[0]
        err_y = loc[1] - self.waypoint[1]
        err_z = loc[2] - self.waypoint[2]
        return -np.linalg.norm([err_x, err_y, err_z])

    def _termination(self) -> bool:
        """Check if episode should terminate."""
        return (time.time() - self.startTime) > self.terminationTime

    def step(self, action):
        """Step the simulation."""
        ob = self.get_obs()
        self._update_data(action)

        reward = self.reward_calculation()
        mujoco.mj_step(self.m, self.d)

        terminated = self._termination()

        return (ob, reward, terminated, False)

    def render(self):
        """Render the environment."""
        # Check if renderer exists
        if not hasattr(self, "renderer"):
            return None

        if self.render_mode == "human":
            self.renderer.update_scene(self.d)
            pixels = self.renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            self.videoWriter.write(pixels)
            cv2.imshow("blimp", pixels)
            cv2.waitKey(10)
            return pixels
        elif self.render_mode == "blimp":
            self.renderer.update_scene(self.d, camera="blimpCamera")
            pixels = self.renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            self.videoWriter.write(pixels)
            cv2.imshow("blimp", pixels)
            cv2.waitKey(10)
            return pixels
        elif self.render_mode == "rgb_array":
            # For rgb_array mode, just render and return pixels without display
            self.renderer.update_scene(self.d, camera="followCamera")
            pixels = self.renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            return pixels
        else:
            self.renderer.update_scene(self.d, camera="followCamera")
            pixels = self.renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            self.videoWriter.write(pixels)
            cv2.imshow("blimp", pixels)
            cv2.waitKey(10)
            return pixels

    def reset(self, regenerate_environment: bool = False):
        """
        Reset the environment.

        Args:
            regenerate_environment: Whether to generate a completely new environment

        Returns:
            Initial observation and info
        """
        self.episode_count += 1

        if regenerate_environment and self.randomize_enabled:
            # Generate completely new environment
            self.m, self.d = self._generate_randomized_environment()

            # Update renderer if needed
            if hasattr(self, "renderer"):
                # Renderer needs to be recreated for new model
                height = self.renderer.height
                width = self.renderer.width
                self.renderer = mujoco.Renderer(self.m, height, width)
        else:
            # Just reset the simulation state
            mujoco.mj_resetData(self.m, self.d)

            # Optionally randomize waypoint
            if self.randomize_enabled:
                self.randomize_waypoint()

        self.waypoint = self._get_current_waypoint()
        self.startTime = time.time()

        return (
            self.get_obs(),
            {
                "episode": self.episode_count,
                "waypoint": self.waypoint,
                "weather": self.weather_preset,
            },
        )

    def get_randomization_info(self) -> Dict:
        """Get information about current randomization state."""
        return {
            "waypoint": self.waypoint,
            "viscosity": float(self.m.opt.viscosity),
            "wind": self.m.opt.wind.copy().tolist(),
            "density": float(self.m.opt.density),
            "gravity": self.m.opt.gravity.copy().tolist(),
            "episode_count": self.episode_count,
            "weather_preset": self.weather_preset,
        }

    def set_weather(self, preset: str):
        """Change weather conditions on the fly."""
        self.weather_preset = preset
        WeatherRandomizer.apply_weather(self.m, preset, randomize=True)

    def viewer_setup(self):
        """Setup viewer configuration."""
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.cam.distance = self.m.stat.extent * 0.5
