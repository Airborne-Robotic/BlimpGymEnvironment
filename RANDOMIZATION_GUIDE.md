# Environment Randomization Guide

Complete guide for using the environment randomization system in the Blimp simulation.

## Table of Contents
- [Quick Start](#quick-start)
- [Features](#features)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Quick Start

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp

# Create a fully randomized environment
env = RandomizedBlimp(
    modelPath='diff.xml',
    render_mode='',
    randomize=True,
    seed=42  # Optional: for reproducibility
)

# Train with randomization
for episode in range(100):
    obs, info = env.reset(regenerate_environment=True)
    
    for step in range(200):
        action = your_policy(obs)
        obs, reward, done, _ = env.step(action)
        
        if done:
            break
```

---

## Features

### 1. **Waypoint Randomization**
- Random goal positions in 3D space
- Configurable position ranges
- Dynamic waypoint changes during episodes

### 2. **Obstacle Course Generation**
- Procedural obstacle placement
- Multiple obstacle types (box, sphere, cylinder)
- Configurable density and sizes
- Predefined courses (corridor, slalom, tower, maze)

### 3. **Weather/Physics Randomization**
- Air viscosity variations
- Dynamic wind conditions
- Gravity scaling
- Air density changes
- Weather presets (calm, windy, turbulent, dense, thin)

### 4. **Lighting Variations**
- Random light positions
- Varying light directions
- Intensity changes

---

## Basic Usage

### Creating a Randomized Environment

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp

# Basic randomization (all features enabled)
env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True
)

# With specific weather
env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    weather_preset='windy'  # Options: calm, windy, turbulent, dense, thin
)

# No randomization (deterministic)
env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=False
)
```

### Custom Randomization Ranges

```python
custom_config = {
    'waypoint': {
        'x_range': (-2, 2),      # Narrower X range
        'y_range': (-2, 2),      # Narrower Y range
        'z_range': (1.0, 1.5)    # Keep at mid-height
    },
    'obstacles': {
        'count_range': (5, 8),   # More obstacles
        'size_range': (0.2, 0.5) # Smaller obstacles
    },
    'physics': {
        'wind_range': (-0.5, 0.5),  # Less wind
        'viscosity_range': (1e-5, 3e-5)
    }
}

env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    randomization_config=custom_config
)
```

---

## Advanced Features

### 1. Progressive Difficulty Curriculum

```python
# Stage 1: Easy
easy_config = {
    'waypoint': {'x_range': (-1, 1), 'y_range': (-1, 1), 'z_range': (0.8, 1.2)},
    'obstacles': {'count_range': (1, 3)},
    'physics': {'wind_range': (-0.2, 0.2)}
}

env_easy = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    randomization_config=easy_config,
    weather_preset='calm'
)

# Stage 2: Medium
medium_config = {
    'waypoint': {'x_range': (-3, 3), 'y_range': (-3, 3), 'z_range': (0.5, 2.0)},
    'obstacles': {'count_range': (4, 7)},
    'physics': {'wind_range': (-0.8, 0.8)}
}

env_medium = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    randomization_config=medium_config,
    weather_preset='windy'
)

# Stage 3: Hard
hard_config = {
    'waypoint': {'x_range': (-5, 5), 'y_range': (-5, 5), 'z_range': (0.3, 2.5)},
    'obstacles': {'count_range': (8, 15)},
    'physics': {'wind_range': (-1.5, 1.5)}
}

env_hard = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    randomization_config=hard_config,
    weather_preset='turbulent'
)
```

### 2. Dynamic Environment Changes

```python
# Change weather during training
env.set_weather('turbulent')

# Randomize waypoint on-the-fly
env.randomize_waypoint()

# Set specific waypoint
env.update_waypoint((2, 3, 1.5))
```

### 3. Monitoring Randomization

```python
# Get current randomization state
info = env.get_randomization_info()
print(f"Waypoint: {info['waypoint']}")
print(f"Viscosity: {info['viscosity']}")
print(f"Wind: {info['wind']}")
print(f"Episode: {info['episode_count']}")

# Access randomization history
print(f"Total episodes: {len(env.randomization_history)}")
```

### 4. Episode Reset Strategies

```python
# Reset with same environment (just reset physics state)
obs, info = env.reset(regenerate_environment=False)

# Reset with completely new environment
obs, info = env.reset(regenerate_environment=True)
```

---

## API Reference

### `RandomizedBlimp` Class

#### Constructor

```python
RandomizedBlimp(
    modelPath: str = "diff.xml",
    render_mode: str = "",
    videoFile: str = "video.mp4",
    height: int = 480,
    width: int = 620,
    randomize: bool = True,
    randomization_config: Optional[Dict] = None,
    course_type: Optional[str] = None,
    weather_preset: str = 'calm',
    seed: Optional[int] = None
)
```

**Parameters:**
- `modelPath`: Path to base XML model
- `render_mode`: Rendering mode ("human", "blimp", "rgb_array", "")
- `randomize`: Enable/disable randomization
- `randomization_config`: Custom configuration dictionary
- `weather_preset`: Weather conditions ('calm', 'windy', 'turbulent', 'dense', 'thin')
- `seed`: Random seed for reproducibility

#### Methods

**`reset(regenerate_environment: bool = False)`**
- Reset the environment
- `regenerate_environment=True`: Generate new obstacles, waypoints, physics
- `regenerate_environment=False`: Just reset state, keep same environment
- Returns: (observation, info_dict)

**`step(action)`**
- Step the simulation
- `action`: [motor1, motor2, servo1, servo2]
- Returns: (observation, reward, terminated, info)

**`get_randomization_info()`**
- Get current randomization parameters
- Returns: Dictionary with waypoint, physics parameters, etc.

**`set_weather(preset: str)`**
- Change weather conditions on-the-fly
- `preset`: 'calm', 'windy', 'turbulent', 'dense', 'thin'

**`randomize_waypoint()`**
- Randomize waypoint position immediately

**`update_waypoint(waypoint: Tuple[float, float, float])`**
- Set specific waypoint position

---

## Configuration Schema

### Default Configuration

```python
{
    'waypoint': {
        'x_range': (-4, 4),
        'y_range': (-4, 4),
        'z_range': (0.5, 2.5)
    },
    'obstacles': {
        'count_range': (3, 10),
        'x_range': (-5, 5),
        'y_range': (-5, 5),
        'z_range': (0.2, 2.0),
        'size_range': (0.1, 0.8),
        'types': ['box', 'sphere', 'cylinder']
    },
    'physics': {
        'viscosity_range': (1e-6, 5e-5),
        'wind_range': (-1.0, 1.0),
        'gravity_scale_range': (0.9, 1.1),
        'density_range': (1.0, 1.5)
    },
    'lighting': {
        'pos_range': (-5, 5),
        'intensity_range': (0.5, 1.5)
    }
}
```

### Weather Presets

| Preset | Viscosity | Wind | Density | Use Case |
|--------|-----------|------|---------|----------|
| **calm** | 1.8e-5 | 0.0 | 1.293 | Testing, early training |
| **windy** | 2e-5 | 1.5 | 1.293 | Moderate challenge |
| **turbulent** | 3e-5 | 2.0 | 1.4 | High difficulty |
| **dense** | 5e-5 | 0.5 | 1.8 | Heavy air resistance |
| **thin** | 5e-6 | 0.3 | 0.8 | Low drag |

---

## Examples

### Example 1: Training Loop with Randomization

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp
import numpy as np

env = RandomizedBlimp(
    modelPath='diff.xml',
    randomize=True,
    weather_preset='windy',
    seed=42
)

# Training loop
for episode in range(1000):
    obs, info = env.reset(regenerate_environment=True)
    episode_reward = 0
    
    for step in range(200):
        # Your policy here
        action = np.random.uniform(-1, 1, 4)
        
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}: Reward={episode_reward:.2f}, Waypoint={info['waypoint']}")
```

### Example 2: Curriculum Learning

```python
# Start easy, progressively increase difficulty
stages = [
    {'name': 'easy', 'config': easy_config, 'episodes': 100},
    {'name': 'medium', 'config': medium_config, 'episodes': 200},
    {'name': 'hard', 'config': hard_config, 'episodes': 300}
]

for stage in stages:
    print(f"Training stage: {stage['name']}")
    
    env = RandomizedBlimp(
        modelPath='diff.xml',
        randomize=True,
        randomization_config=stage['config']
    )
    
    for episode in range(stage['episodes']):
        obs, _ = env.reset(regenerate_environment=True)
        # ... training code ...
```

### Example 3: Testing on Different Conditions

```python
# Test trained policy on various conditions
test_conditions = ['calm', 'windy', 'turbulent', 'dense', 'thin']
results = {}

for condition in test_conditions:
    env = RandomizedBlimp(
        modelPath='diff.xml',
        randomize=True,
        weather_preset=condition,
        seed=42
    )
    
    # Run 10 test episodes
    rewards = []
    for episode in range(10):
        obs, _ = env.reset(regenerate_environment=True)
        episode_reward = 0
        
        for step in range(200):
            action = trained_policy(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        
        rewards.append(episode_reward)
    
    results[condition] = {
        'mean': np.mean(rewards),
        'std': np.std(rewards)
    }

# Print results
for condition, stats in results.items():
    print(f"{condition}: {stats['mean']:.2f} ± {stats['std']:.2f}")
```

---

## Tips for Best Results

1. **Start Simple**: Begin with `randomize=False` to verify your policy works in deterministic settings

2. **Progressive Training**: Use curriculum learning - start with easy randomization, gradually increase difficulty

3. **Seed for Reproducibility**: Use `seed` parameter for debugging and comparison

4. **Monitor Performance**: Track rewards across different randomization settings to identify weaknesses

5. **Regenerate Strategically**: 
   - Use `regenerate_environment=True` every episode for maximum diversity
   - Use `regenerate_environment=False` for faster resets when testing

6. **Custom Ranges**: Tailor randomization to your specific training goals

7. **Weather Variety**: Train on multiple weather presets for robust policies

---

## Troubleshooting

### Issue: Environment too difficult
**Solution**: Reduce randomization ranges, use 'calm' weather preset

### Issue: Policy not generalizing
**Solution**: Increase randomization ranges, use curriculum learning

### Issue: Slow training
**Solution**: Use `regenerate_environment=False` more frequently, reduce obstacle count

### Issue: Inconsistent performance
**Solution**: Ensure sufficient training episodes, verify random seed is set for testing

---

## Files

- `environment_randomizer.py` - Core randomization logic
- `randomized_blimp.py` - Enhanced Blimp class with randomization
- `examples_randomization.py` - Usage examples
- `RANDOMIZATION_GUIDE.md` - This guide

---

## Citation

If you use this randomization system in your research, please cite:

```
Blimp Gym Environment with Domain Randomization
MuJoCo-based simulation environment for autonomous blimp control
```
