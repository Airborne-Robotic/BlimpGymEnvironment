"""
Environment Randomization System for Blimp Simulation

This module provides comprehensive environment randomization including:
- Waypoint randomization
- Obstacle course generation
- Weather/physics randomization
- Lighting variations
"""

import numpy as np
import mujoco
from xml.etree import ElementTree as ET
from typing import Dict, List, Optional, Tuple
import tempfile
import os


class EnvironmentRandomizer:
    """
    Comprehensive environment randomizer for MuJoCo blimp simulation.

    Features:
    - Dynamic waypoint generation
    - Procedural obstacle courses
    - Physics randomization (wind, viscosity, gravity)
    - Lighting variations
    - Terrain modifications
    """

    def __init__(self, base_xml_path: str, seed: Optional[int] = None):
        """
        Initialize the environment randomizer.

        Args:
            base_xml_path: Path to the base MuJoCo XML file
            seed: Random seed for reproducibility
        """
        self.base_xml_path = base_xml_path
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Configuration for randomization ranges
        self.config = {
            "waypoint": {"x_range": (-4, 4), "y_range": (-4, 4), "z_range": (0.5, 2.5)},
            "obstacles": {
                "count_range": (3, 10),
                "x_range": (-5, 5),
                "y_range": (-5, 5),
                "z_range": (0.2, 2.0),
                "size_range": (0.1, 0.8),
                "types": ["box", "sphere", "cylinder"],
            },
            "physics": {
                "viscosity_range": (1e-6, 5e-5),
                "wind_range": (-1.0, 1.0),
                "gravity_scale_range": (0.9, 1.1),
                "density_range": (1.0, 1.5),
            },
            "lighting": {"pos_range": (-5, 5), "intensity_range": (0.5, 1.5)},
        }

    def generate_environment(
        self,
        randomize_waypoint: bool = True,
        randomize_obstacles: bool = True,
        randomize_physics: bool = True,
        randomize_lighting: bool = True,
        num_obstacles: Optional[int] = None,
    ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """
        Generate a randomized environment.

        Args:
            randomize_waypoint: Whether to randomize waypoint position
            randomize_obstacles: Whether to add random obstacles
            randomize_physics: Whether to randomize physics parameters
            randomize_lighting: Whether to randomize lighting
            num_obstacles: Specific number of obstacles (None for random)

        Returns:
            tuple: (MjModel, MjData) for the randomized environment
        """
        # Parse base XML
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()

        # Apply randomizations
        if randomize_waypoint:
            self._randomize_waypoint(root)

        if randomize_obstacles:
            self._add_random_obstacles(root, num_obstacles)

        if randomize_lighting:
            self._randomize_lighting(root)

        # Fix compiler assetdir to use absolute path
        compiler = root.find(".//compiler[@assetdir]")
        if compiler is not None:
            # Get the directory of the base XML file
            base_dir = os.path.dirname(os.path.abspath(self.base_xml_path))
            assets_dir = os.path.join(base_dir, compiler.get("assetdir"))
            # Set absolute path
            compiler.set("assetdir", assets_dir)

        # Save modified XML to temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".xml", text=True)
        try:
            tree.write(temp_path)
            os.close(temp_fd)

            # Load model
            model = mujoco.MjModel.from_xml_path(temp_path)
            data = mujoco.MjData(model)

            # Apply physics randomization (done at model level)
            if randomize_physics:
                self._randomize_physics(model)

            return model, data

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _randomize_waypoint(self, root: ET.Element):
        """Randomize waypoint position in XML."""
        waypoint = root.find(".//geom[@name='waypoint']")
        if waypoint is not None:
            x = np.random.uniform(*self.config["waypoint"]["x_range"])
            y = np.random.uniform(*self.config["waypoint"]["y_range"])
            z = np.random.uniform(*self.config["waypoint"]["z_range"])
            waypoint.set("pos", f"{x} {y} {z}")

    def _add_random_obstacles(
        self, root: ET.Element, num_obstacles: Optional[int] = None
    ):
        """Add random obstacles to the environment."""
        if num_obstacles is None:
            num_obstacles = np.random.randint(*self.config["obstacles"]["count_range"])

        # Find worldbody to add obstacles
        worldbody = root.find(".//worldbody")
        if worldbody is None:
            return

        # Generate obstacles
        for i in range(num_obstacles):
            obstacle_type = np.random.choice(self.config["obstacles"]["types"])
            x = np.random.uniform(*self.config["obstacles"]["x_range"])
            y = np.random.uniform(*self.config["obstacles"]["y_range"])
            z = np.random.uniform(*self.config["obstacles"]["z_range"])
            size = np.random.uniform(*self.config["obstacles"]["size_range"])

            # Random color
            r, g, b = np.random.uniform(0.2, 0.9, 3)

            # Create obstacle element
            obstacle = ET.SubElement(worldbody, "geom")
            obstacle.set("name", f"random_obstacle_{i}")
            obstacle.set("type", obstacle_type)
            obstacle.set("pos", f"{x} {y} {z}")
            obstacle.set("rgba", f"{r} {g} {b} 0.8")

            # Set size based on type
            if obstacle_type == "box":
                obstacle.set("size", f"{size} {size} {size}")
            elif obstacle_type == "sphere":
                obstacle.set("size", f"{size}")
            elif obstacle_type == "cylinder":
                obstacle.set("size", f"{size} {size * 1.5}")

    def _randomize_lighting(self, root: ET.Element):
        """Randomize lighting conditions."""
        lights = root.findall(".//light")
        for light in lights:
            # Randomize position
            x = np.random.uniform(*self.config["lighting"]["pos_range"])
            y = np.random.uniform(*self.config["lighting"]["pos_range"])
            z = np.random.uniform(1, 5)
            light.set("pos", f"{x} {y} {z}")

            # Randomize direction (point downward-ish)
            dx = np.random.uniform(-0.3, 0.3)
            dy = np.random.uniform(-0.3, 0.3)
            dz = np.random.uniform(-1.2, -0.8)
            light.set("dir", f"{dx} {dy} {dz}")

    def _randomize_physics(self, model: mujoco.MjModel):
        """Randomize physics parameters at the model level."""
        # Viscosity (air resistance)
        model.opt.viscosity = np.random.uniform(
            *self.config["physics"]["viscosity_range"]
        )

        # Wind
        wind_x = np.random.uniform(*self.config["physics"]["wind_range"])
        wind_y = np.random.uniform(*self.config["physics"]["wind_range"])
        wind_z = np.random.uniform(-0.2, 0.2)  # Less vertical wind
        model.opt.wind[:] = [wind_x, wind_y, wind_z]

        # Gravity scale
        gravity_scale = np.random.uniform(
            *self.config["physics"]["gravity_scale_range"]
        )
        model.opt.gravity[2] *= gravity_scale

        # Density
        model.opt.density = np.random.uniform(*self.config["physics"]["density_range"])

    def update_config(self, config_updates: Dict):
        """
        Update randomization configuration.

        Args:
            config_updates: Dictionary with configuration updates
        """
        for key, value in config_updates.items():
            if key in self.config:
                if isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value


class ObstacleCourseGenerator:
    """
    Generate specific obstacle course layouts.

    Provides predefined course types:
    - Corridor: Navigate through a narrow passage
    - Slalom: Weave through obstacles
    - Tower: Obstacles at different heights
    - Maze: Complex 3D maze
    """

    @staticmethod
    def generate_corridor(root: ET.Element, length: float = 8.0, width: float = 2.0):
        """Generate a corridor course."""
        worldbody = root.find(".//worldbody")
        if worldbody is None:
            return

        # Left wall
        for i in range(int(length)):
            wall = ET.SubElement(worldbody, "geom")
            wall.set("name", f"corridor_left_{i}")
            wall.set("type", "box")
            wall.set("pos", f"{-width / 2} {i} 1")
            wall.set("size", "0.1 0.5 1")
            wall.set("rgba", "0.6 0.3 0.1 1")

        # Right wall
        for i in range(int(length)):
            wall = ET.SubElement(worldbody, "geom")
            wall.set("name", f"corridor_right_{i}")
            wall.set("type", "box")
            wall.set("pos", f"{width / 2} {i} 1")
            wall.set("size", "0.1 0.5 1")
            wall.set("rgba", "0.6 0.3 0.1 1")

    @staticmethod
    def generate_slalom(root: ET.Element, num_gates: int = 5, spacing: float = 2.0):
        """Generate a slalom course with alternating obstacles."""
        worldbody = root.find(".//worldbody")
        if worldbody is None:
            return

        for i in range(num_gates):
            x_offset = 1.5 * (1 if i % 2 == 0 else -1)
            y_pos = i * spacing

            obstacle = ET.SubElement(worldbody, "geom")
            obstacle.set("name", f"slalom_gate_{i}")
            obstacle.set("type", "cylinder")
            obstacle.set("pos", f"{x_offset} {y_pos} 1")
            obstacle.set("size", "0.3 1.5")
            obstacle.set("rgba", "1 0.5 0 0.8")

    @staticmethod
    def generate_tower(
        root: ET.Element, num_levels: int = 4, obstacles_per_level: int = 3
    ):
        """Generate a tower course with obstacles at different heights."""
        worldbody = root.find(".//worldbody")
        if worldbody is None:
            return

        for level in range(num_levels):
            z = 0.5 + level * 0.8
            for i in range(obstacles_per_level):
                angle = 2 * np.pi * i / obstacles_per_level
                radius = 2.0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

                obstacle = ET.SubElement(worldbody, "geom")
                obstacle.set("name", f"tower_l{level}_o{i}")
                obstacle.set("type", "sphere")
                obstacle.set("pos", f"{x} {y} {z}")
                obstacle.set("size", "0.3")
                obstacle.set(
                    "rgba", f"{level / num_levels} 0.5 {1 - level / num_levels} 0.7"
                )

    @staticmethod
    def generate_maze(root: ET.Element, grid_size: int = 5, cell_size: float = 1.5):
        """Generate a simple 3D maze."""
        worldbody = root.find(".//worldbody")
        if worldbody is None:
            return

        # Simple maze generation (random walls)
        for i in range(grid_size):
            for j in range(grid_size):
                if np.random.random() < 0.3:  # 30% chance of wall
                    x = (i - grid_size / 2) * cell_size
                    y = (j - grid_size / 2) * cell_size
                    z = np.random.uniform(0.5, 2.0)

                    wall = ET.SubElement(worldbody, "geom")
                    wall.set("name", f"maze_wall_{i}_{j}")
                    wall.set("type", "box")
                    wall.set("pos", f"{x} {y} {z}")
                    wall.set("size", f"{cell_size / 3} {cell_size / 3} {z}")
                    wall.set("rgba", "0.4 0.4 0.4 0.9")


class WeatherRandomizer:
    """
    Randomize weather conditions in the simulation.

    Weather presets:
    - Calm: Minimal wind, normal conditions
    - Windy: Strong variable wind
    - Turbulent: Chaotic wind conditions
    - Dense: High air density (harder to fly)
    - Thin: Low air density (less drag)
    """

    PRESETS = {
        "calm": {"viscosity": 1.8e-5, "wind": (0.0, 0.0, 0.0), "density": 1.293},
        "windy": {"viscosity": 2e-5, "wind": (1.5, 0.8, 0.1), "density": 1.293},
        "turbulent": {"viscosity": 3e-5, "wind": (2.0, 1.5, 0.3), "density": 1.4},
        "dense": {"viscosity": 5e-5, "wind": (0.5, 0.3, 0.0), "density": 1.8},
        "thin": {"viscosity": 5e-6, "wind": (0.3, 0.2, 0.0), "density": 0.8},
    }

    @staticmethod
    def apply_weather(
        model: mujoco.MjModel, preset: str = "calm", randomize: bool = True
    ):
        """
        Apply weather conditions to the model.

        Args:
            model: MuJoCo model
            preset: Weather preset name
            randomize: Add random variations to the preset
        """
        if preset not in WeatherRandomizer.PRESETS:
            preset = "calm"

        weather = WeatherRandomizer.PRESETS[preset].copy()

        if randomize:
            # Add random variations
            weather["viscosity"] *= np.random.uniform(0.9, 1.1)
            weather["wind"] = tuple(
                w * np.random.uniform(0.8, 1.2) for w in weather["wind"]
            )
            weather["density"] *= np.random.uniform(0.95, 1.05)

        # Apply to model
        model.opt.viscosity = weather["viscosity"]
        model.opt.wind[:] = weather["wind"]
        model.opt.density = weather["density"]

    @staticmethod
    def apply_random_gusts(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gust_probability: float = 0.1,
        gust_strength: float = 2.0,
    ):
        """
        Apply random wind gusts during simulation.

        Call this method at each step to potentially add gusts.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            gust_probability: Probability of gust per step
            gust_strength: Maximum gust strength
        """
        if np.random.random() < gust_probability:
            gust = np.random.uniform(-gust_strength, gust_strength, 3)
            model.opt.wind[:] += gust
