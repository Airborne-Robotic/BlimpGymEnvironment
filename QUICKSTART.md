# Quick Start Guide

Get started with the Blimp Gym Environment in 5 minutes!

## Installation

```bash
cd BlimpGymEnvironment
pip install -e .
```

## Your First Simulation

```python
from BlimpGymEnvironment import RandomizedBlimp

# Create environment
env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=False  # Start without randomization
)

# Reset environment
obs, info = env.reset()

# Run simulation
for step in range(200):
    # Action: [motor1, motor2, servo1, servo2]
    action = [0.5, 0.5, 0, 0]  # Forward thrust
    obs, reward, done, _ = env.step(action)
    print(f"Step {step}: Reward = {reward:.3f}")
    
    if done:
        break
```

## Understanding the Environment

### Observation
```python
obs = [position, pixels, gyro_data]

position = obs[0]     # [x, y, z] - Blimp position
image = obs[1]        # Camera image (or None)
angular_vel = obs[2]  # [wx, wy, wz] - Angular velocity
```

### Action
```python
action = [motor1, motor2, servo1, servo2]

# motor1, motor2: Range [-1, 1] → Thrust force
# servo1, servo2: Range [-1, 1] → Tilt torque

# Examples:
forward = [1, 1, 0, 0]        # Go forward
turn_left = [1, -1, 0, 0]     # Turn left
tilt_up = [0, 0, -0.5, -0.5]  # Tilt up
```

### Reward
```python
# Reward = -distance to waypoint
# Higher is better (0 = at goal)
reward = -sqrt((x - x_goal)² + (y - y_goal)² + (z - z_goal)²)
```

## With Randomization

```python
from BlimpGymEnvironment import RandomizedBlimp

# Enable randomization for robust training
env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    weather_preset='windy',
    seed=42  # Optional: for reproducibility
)

# Training loop
for episode in range(10):
    # Generate new environment each episode
    obs, info = env.reset(regenerate_environment=True)
    print(f"Episode {episode}: Target = {info['waypoint']}")
    
    episode_reward = 0
    for step in range(200):
        # Your policy here
        action = [0.5, 0.5, 0, 0]
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"  Total reward: {episode_reward:.2f}")
```

## Common Actions

```python
# Basic maneuvers
hover = [0.4, 0.4, 0, 0]           # Maintain altitude
forward = [1, 1, 0, 0]             # Maximum forward thrust
backward = [-1, -1, 0, 0]          # Reverse
turn_left = [1, -1, 0, 0]          # Differential thrust
turn_right = [-1, 1, 0, 0]         # Differential thrust
ascend = [0.8, 0.8, 0, 0]          # Go up
descend = [0.2, 0.2, 0, 0]         # Go down
tilt_forward = [1, 1, 0.5, 0.5]    # Forward with tilt
```

## Customization

### Change Waypoint
```python
env.update_waypoint((2, 3, 1.5))  # Set specific target
env.randomize_waypoint()           # Random target
```

### Change Weather
```python
env.set_weather('turbulent')  # Make it harder
env.set_weather('calm')       # Make it easier
```

### Custom Ranges
```python
config = {
    'waypoint': {
        'x_range': (-2, 2),
        'y_range': (-2, 2),
        'z_range': (1.0, 1.5)
    },
    'obstacles': {
        'count_range': (5, 8)
    }
}

env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    randomization_config=config
)
```

## Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Explore randomization**: [RANDOMIZATION_GUIDE.md](RANDOMIZATION_GUIDE.md)
3. **Run examples**: `python -m BlimpGymEnvironment.examples_randomization`
4. **Build your RL agent**: Use with Stable-Baselines3, RLlib, or your own RL code

## Common Issues

**Import Error?**
```bash
pip install -e .
```

**Missing dependencies?**
```bash
pip install mujoco opencv-python numpy
```

**Can't see visualization?**
```python
env = RandomizedBlimp(
    modelPath='diff.xml',
    render_mode='human'  # Enable visualization
)

# Call render() each step
env.render()
```

## Tips

- Start with `randomize=False` to test your policy
- Use `seed` parameter for reproducible results
- Set `render_mode=''` during training for speed
- Use `regenerate_environment=True` for maximum diversity
- Try different weather presets: calm → windy → turbulent

---

**Ready to fly? Good luck! 🎈**
