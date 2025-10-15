# Blimp Gym Environment

A MuJoCo-based simulation environment for autonomous blimp control with comprehensive domain randomization support.

## Features

- **High-fidelity physics simulation** using MuJoCo
- **Differential thrust control** with tiltable propellers
- **Full domain randomization** for robust policy training
- **Multiple weather conditions** (calm, windy, turbulent, dense, thin)
- **Procedural obstacle generation**
- **Flexible observation and reward system**
- **Compatible with reinforcement learning frameworks**

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Basic Usage](#basic-usage)
- [Randomization Features](#randomization-features)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd BlimpGymEnvironment

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Verify Installation

```bash
python -c "from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp; print('Installation successful!')"
```

---

## Quick Start

### Basic Simulation

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp

# Create environment (no randomization)
env = RandomizedBlimp(
    modelPath='diff.xml',
    render_mode='',  # Set to 'human' to visualize
    randomize=False
)

# Reset environment
obs, info = env.reset()

# Run simulation
for step in range(200):
    # Action: [motor1, motor2, servo1, servo2]
    # motor1, motor2: thrust [-1, 1] (scaled to [-2, 2] N)
    # servo1, servo2: torque [-1, 1] (Nm)
    action = [0.5, 0.5, 0, 0]  # Forward thrust
    
    obs, reward, done, _ = env.step(action)
    
    if done:
        break

print(f"Final reward: {reward:.3f}")
```

### With Randomization

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp

# Create randomized environment
env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    weather_preset='windy',
    seed=42  # For reproducibility
)

# Training loop
for episode in range(100):
    obs, info = env.reset(regenerate_environment=True)
    episode_reward = 0
    
    for step in range(200):
        action = your_policy(obs)  # Your RL policy
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}: Reward={episode_reward:.2f}")
```

---

## Project Structure

```
BlimpGymEnvironment/
├── BlimpGymEnvironment/
│   ├── __init__.py                    # Package initialization
│   ├── blimp.py                       # Original Blimp class (legacy)
│   ├── randomized_blimp.py            # Enhanced Blimp with randomization
│   ├── environment_randomizer.py      # Core randomization engine
│   ├── examples_randomization.py      # Example usage scripts
│   ├── diff.xml                       # Blimp model definition
│   └── assets/                        # 3D models and textures
│       ├── BalloonBody.obj
│       ├── Frame.obj
│       ├── PropGuard.obj
│       └── ...
├── README.md                          # This file
├── RANDOMIZATION_GUIDE.md             # Detailed randomization documentation
└── pyproject.toml                     # Project configuration
```

---

## Basic Usage

### 1. Environment Creation

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp

# Deterministic environment
env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=False
)

# Randomized environment
env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    weather_preset='windy'
)

# With custom configuration
custom_config = {
    'waypoint': {
        'x_range': (-2, 2),
        'y_range': (-2, 2),
        'z_range': (1.0, 1.5)
    }
}

env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    randomization_config=custom_config
)
```

### 2. Observation Space

The environment returns observations as a list:

```python
obs = [
    position,   # [x, y, z] - Blimp position in world frame
    pixels,     # Camera image (if renderer enabled, else None)
    gyro_data   # [wx, wy, wz] - Angular velocity
]

# Access observations
position = obs[0]     # numpy array shape (3,)
image = obs[1]        # numpy array shape (height, width, 3) or None
angular_vel = obs[2]  # numpy array shape (3,)
```

### 3. Action Space

Actions control motors and servos:

```python
action = [motor1, motor2, servo1, servo2]

# motor1, motor2: Thrust control
#   Range: [-1, 1]
#   Scaled to: [-2, 2] Newtons
#   motor1: Left motor
#   motor2: Right motor

# servo1, servo2: Servo torque for propeller tilt
#   Range: [-1, 1]
#   Units: Newton-meters
#   servo1: Left servo
#   servo2: Right servo

# Examples:
forward_thrust = [1, 1, 0, 0]      # Both motors forward
turn_left = [1, -1, 0, 0]          # Differential thrust
tilt_forward = [0, 0, 0.5, 0.5]    # Tilt propellers forward
```

### 4. Reward Function

Default reward is negative distance to waypoint:

```python
# Reward = -distance to waypoint
reward = -sqrt((x - x_goal)^2 + (y - y_goal)^2 + (z - z_goal)^2)

# Higher reward = closer to goal
# Maximum reward = 0 (at goal)
```

### 5. Episode Termination

Episodes terminate based on time:

```python
# Default: 200 seconds
env.update_termination_time(300)  # Change to 300 seconds
```

---

## Randomization Features

See [RANDOMIZATION_GUIDE.md](RANDOMIZATION_GUIDE.md) for complete documentation.

### Quick Overview

| Feature | Description | Default Range |
|---------|-------------|---------------|
| **Waypoint** | Random goal position | X[-4,4], Y[-4,4], Z[0.5,2.5] |
| **Obstacles** | Procedural obstacles | 3-10 objects |
| **Wind** | Dynamic wind force | [-1, 1] m/s |
| **Viscosity** | Air resistance | [1e-6, 5e-5] |
| **Density** | Air density | [1.0, 1.5] kg/m³ |
| **Lighting** | Light position/direction | Randomized |

### Weather Presets

```python
# Calm: No wind, normal conditions
env = RandomizedBlimp(modelPath='diff.xml', weather_preset='calm')

# Windy: Moderate wind
env = RandomizedBlimp(modelPath='diff.xml', weather_preset='windy')

# Turbulent: Strong chaotic wind
env = RandomizedBlimp(modelPath='diff.xml', weather_preset='turbulent')

# Dense: High air density (harder to move)
env = RandomizedBlimp(modelPath='diff.xml', weather_preset='dense')

# Thin: Low air density (less drag)
env = RandomizedBlimp(modelPath='diff.xml', weather_preset='thin')
```

---

## API Reference

### `RandomizedBlimp` Class

#### Constructor Parameters

```python
RandomizedBlimp(
    modelPath='diff.xml',           # XML model file
    render_mode='',                  # '', 'human', 'blimp', 'rgb_array'
    videoFile='video.mp4',           # Output video path
    height=480,                      # Render height
    width=620,                       # Render width
    randomize=True,                  # Enable randomization
    randomization_config=None,       # Custom config dict
    course_type=None,                # Obstacle course type
    weather_preset='calm',           # Weather conditions
    seed=None                        # Random seed
)
```

#### Key Methods

**`reset(regenerate_environment=False)`**
- Reset environment to initial state
- `regenerate_environment=True`: New obstacles, waypoint, physics
- `regenerate_environment=False`: Keep same environment, reset state
- Returns: `(observation, info_dict)`

**`step(action)`**
- Execute one simulation step
- `action`: List/array of 4 values [motor1, motor2, servo1, servo2]
- Returns: `(observation, reward, terminated, info)`

**`render()`**
- Render current state (if render_mode enabled)
- Displays window and/or saves to video

**`get_randomization_info()`**
- Get current randomization parameters
- Returns: Dict with waypoint, physics params, episode count

**`set_weather(preset)`**
- Change weather on-the-fly
- `preset`: 'calm', 'windy', 'turbulent', 'dense', 'thin'

**`randomize_waypoint()`**
- Generate new random waypoint immediately

**`update_waypoint(waypoint)`**
- Set specific waypoint position
- `waypoint`: Tuple (x, y, z)

**`update_termination_time(time)`**
- Set episode duration
- `time`: Seconds (int)

**`get_obs()`**
- Get current observation
- Returns: [position, pixels, gyro_data]

**`get_ground_truth()`**
- Get true blimp state
- Returns: [position, rotation_matrix]

---

## Examples

### Example 1: Simple Flight

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp

env = RandomizedBlimp(modelPath='diff.xml', randomize=False)
obs, _ = env.reset()

# Fly forward for 5 seconds
for i in range(100):  # 100 steps ≈ 5 seconds
    action = [1, 1, 0, 0]  # Both motors forward
    obs, reward, done, _ = env.step(action)
    print(f"Position: {obs[0]}, Reward: {reward:.3f}")
```

### Example 2: Waypoint Navigation

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp
import numpy as np

env = RandomizedBlimp(modelPath='diff.xml', randomize=True, seed=42)

for episode in range(10):
    obs, info = env.reset(regenerate_environment=True)
    target = info['waypoint']
    print(f"Episode {episode}: Target = {target}")
    
    for step in range(200):
        # Simple proportional control
        position = obs[0]
        error = np.array(target) - position
        
        # Thrust based on distance
        thrust = np.clip(np.linalg.norm(error[:2]) * 0.5, 0, 1)
        action = [thrust, thrust, 0, 0]
        
        obs, reward, done, _ = env.step(action)
        
        if done or reward > -0.5:  # Close enough
            print(f"  Reached target! Final reward: {reward:.3f}")
            break
```

### Example 3: Training with RL (Pseudocode)

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp
import your_rl_library as rl

env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    weather_preset='windy'
)

# Initialize RL agent
agent = rl.PPO(env)

# Training loop
for episode in range(1000):
    obs, _ = env.reset(regenerate_environment=True)
    episode_reward = 0
    
    for step in range(200):
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.store_transition(obs, action, reward, done)
        episode_reward += reward
        
        if done:
            break
    
    agent.update()
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

### Example 4: Curriculum Learning

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp

# Stage 1: Easy (100 episodes)
easy_config = {
    'waypoint': {'x_range': (-1, 1), 'y_range': (-1, 1), 'z_range': (0.8, 1.2)},
    'obstacles': {'count_range': (0, 2)},
    'physics': {'wind_range': (-0.2, 0.2)}
}

env_easy = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    randomization_config=easy_config,
    weather_preset='calm'
)

# Train on easy for 100 episodes...

# Stage 2: Medium (200 episodes)
medium_config = {
    'waypoint': {'x_range': (-3, 3), 'y_range': (-3, 3), 'z_range': (0.5, 2.0)},
    'obstacles': {'count_range': (3, 6)},
    'physics': {'wind_range': (-0.8, 0.8)}
}

env_medium = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    randomization_config=medium_config,
    weather_preset='windy'
)

# Continue training...
```

### More Examples

See [examples_randomization.py](BlimpGymEnvironment/examples_randomization.py) for 7 detailed examples covering:
- Basic randomization
- Custom configurations
- Weather presets
- Dynamic waypoints
- Training curriculum
- Statistics tracking
- Deterministic mode

---

## Advanced Topics

### Custom Reward Functions

Subclass `RandomizedBlimp` and override `reward_calculation()`:

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp
import numpy as np

class CustomRewardBlimp(RandomizedBlimp):
    def reward_calculation(self) -> float:
        loc = self.get_ground_truth()[0]
        vel = self.d.qvel[3:6]
        
        # Distance penalty
        distance = np.linalg.norm(loc - self.waypoint)
        distance_reward = -distance
        
        # Velocity penalty (discourage fast movement)
        velocity_penalty = -0.1 * np.linalg.norm(vel)
        
        # Altitude bonus (prefer staying at target height)
        altitude_bonus = -abs(loc[2] - self.waypoint[2])
        
        return distance_reward + velocity_penalty + altitude_bonus

env = CustomRewardBlimp(modelPath='diff.xml', randomize=True)
```

### Multi-Waypoint Navigation

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp

env = RandomizedBlimp(modelPath='diff.xml', randomize=True)

waypoints = [(1, 1, 1), (2, 2, 1.5), (-1, 3, 1), (0, 0, 1.2)]

obs, _ = env.reset()

for wp in waypoints:
    print(f"Navigating to {wp}")
    env.update_waypoint(wp)
    
    for step in range(100):
        action = your_navigation_policy(obs, wp)
        obs, reward, done, _ = env.step(action)
        
        if reward > -0.3:  # Close enough
            break
```

### Rendering and Visualization

```python
# Human view (external camera)
env = RandomizedBlimp(
    modelPath='diff.xml',
    render_mode='human',
    videoFile='output.mp4'
)

# Blimp camera (first-person view)
env = RandomizedBlimp(
    modelPath='diff.xml',
    render_mode='blimp',
    height=480,
    width=640
)

# Render each step
for step in range(200):
    action = [0.5, 0.5, 0, 0]
    obs, reward, done, _ = env.step(action)
    env.render()  # Display and save to video
```

---

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'mujoco'`**
```bash
pip install mujoco
```

**Issue: `ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install opencv-python
```

**Issue: Assets not found**
- Ensure you're using the correct `modelPath` parameter
- Check that `BlimpGymEnvironment/assets/` directory exists
- Verify all .obj files are present

**Issue: Simulation unstable / NaN warnings**
- Reduce physics randomization ranges
- Use 'calm' weather preset
- Increase joint damping in diff.xml

**Issue: Slow training**
- Use `regenerate_environment=False` more frequently
- Reduce number of obstacles
- Disable rendering during training (`render_mode=''`)

**Issue: Policy not generalizing**
- Increase randomization ranges
- Use curriculum learning
- Train on multiple weather presets
- Ensure sufficient training episodes

### Performance Tips

1. **Disable rendering during training**: Set `render_mode=''`
2. **Batch resets**: Use `regenerate_environment=False` for faster resets
3. **Reduce obstacles**: Lower `count_range` in config
4. **Use deterministic mode** for debugging
5. **Set random seed** for reproducible results

---

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{blimp_gym_environment,
  title={Blimp Gym Environment},
  author={Your Name},
  year={2025},
  description={MuJoCo-based simulation environment for autonomous blimp control with domain randomization}
}
```

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## License

[Add your license here]

---

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

## Acknowledgments

- MuJoCo physics engine
- OpenCV for rendering
- NumPy for numerical computations

---

**Happy Flying! 🎈**
