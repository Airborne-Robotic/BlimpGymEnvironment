# Project Structure

Overview of the Blimp Gym Environment codebase.

## Directory Layout

```
BlimpGymEnvironment/
├── BlimpGymEnvironment/           # Main package
│   ├── __init__.py                # Package exports and version
│   ├── blimp.py                   # Original Blimp class (backward compatibility)
│   ├── randomized_blimp.py        # Enhanced Blimp with randomization (MAIN)
│   ├── environment_randomizer.py  # Core randomization logic
│   ├── examples_randomization.py  # Usage examples
│   ├── diff.xml                   # Blimp model definition (MuJoCo XML)
│   └── assets/                    # 3D models and textures
│       ├── BalloonBody.obj        # Helium balloon mesh
│       ├── Frame.obj              # Gondola frame mesh
│       ├── PropGuard.obj          # Propeller guard mesh
│       ├── ServoMount.obj         # Servo mount mesh
│       ├── CarbonFiber.obj        # Carbon fiber rod mesh
│       ├── House.obj              # Environment obstacle
│       ├── Desk.obj               # Environment obstacle
│       └── *.mtl                  # Material files
├── README.md                      # Main documentation
├── QUICKSTART.md                  # 5-minute getting started guide
├── RANDOMIZATION_GUIDE.md         # Detailed randomization docs
├── PROJECT_STRUCTURE.md           # This file
├── test_environment.py            # Automated test suite
└── pyproject.toml                 # Project configuration
```

## Core Files

### `randomized_blimp.py` (Main Environment)

The primary environment class with full randomization support.

**Key Components:**
- `RandomizedBlimp` class - Main environment interface
- Episode management with regeneration
- Integration with randomization system
- Observation/action/reward interfaces

**Usage:**
```python
from BlimpGymEnvironment import RandomizedBlimp
env = RandomizedBlimp(modelPath='diff.xml', randomize=True)
```

### `environment_randomizer.py` (Randomization Engine)

Core randomization logic separated from environment.

**Key Components:**
- `EnvironmentRandomizer` - Main randomizer class
  - Waypoint randomization
  - Obstacle generation
  - Physics randomization
  - Lighting variations

- `ObstacleCourseGenerator` - Predefined course layouts
  - Corridor course
  - Slalom course
  - Tower course
  - Maze course

- `WeatherRandomizer` - Weather conditions
  - 5 presets (calm, windy, turbulent, dense, thin)
  - Dynamic gust simulation

**Usage:**
```python
from BlimpGymEnvironment import EnvironmentRandomizer
randomizer = EnvironmentRandomizer('diff.xml', seed=42)
model, data = randomizer.generate_environment()
```

### `blimp.py` (Legacy)

Original Blimp class maintained for backward compatibility.

**Status:** Legacy - use `RandomizedBlimp` instead
**Purpose:** Existing code compatibility

### `diff.xml` (MuJoCo Model)

Physics simulation model definition.

**Key Elements:**
- Balloon body (helium-filled ellipsoid)
- Gondola frame with electronics
- 2 differential thrust motors
- 2 servo-controlled propellers
- Sensors (gyro, accelerometer, camera)
- Environment (floor, lighting)

**Recent Updates:**
- Fixed motor thrust controls (gear="0 0 7 0 0 0")
- Stabilized servo joints (damping=0.2, armature=0.005)
- Added joint limits for servos (0-180°)
- Optimized for stable simulation

## Data Flow

```
User Code
    ↓
RandomizedBlimp.__init__()
    ↓
EnvironmentRandomizer.generate_environment()
    ↓
[Parse XML] → [Randomize] → [Save Temp] → [Load MuJoCo Model]
    ↓
RandomizedBlimp.reset()
    ↓
RandomizedBlimp.step(action)
    ↓
[Update Actuators] → [MuJoCo Step] → [Compute Reward] → [Get Observation]
    ↓
Return (obs, reward, done, info)
```

## Class Hierarchy

```
RandomizedBlimp
├── Inherits: None (standalone)
├── Uses: EnvironmentRandomizer
├── Contains: MuJoCo Model & Data
└── Methods:
    ├── __init__()
    ├── reset()
    ├── step()
    ├── render()
    ├── get_obs()
    ├── get_ground_truth()
    ├── reward_calculation()
    └── Randomization methods:
        ├── randomize_waypoint()
        ├── set_weather()
        └── get_randomization_info()

EnvironmentRandomizer
├── Inherits: None
├── Uses: xml.etree.ElementTree, mujoco
└── Methods:
    ├── generate_environment()
    ├── _randomize_waypoint()
    ├── _add_random_obstacles()
    ├── _randomize_lighting()
    └── _randomize_physics()

ObstacleCourseGenerator
└── Static Methods:
    ├── generate_corridor()
    ├── generate_slalom()
    ├── generate_tower()
    └── generate_maze()

WeatherRandomizer
└── Static Methods:
    ├── apply_weather()
    └── apply_random_gusts()
```

## Configuration System

### Default Configuration (in `environment_randomizer.py`)

```python
config = {
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

### Custom Configuration

Users can override any part:

```python
custom_config = {
    'waypoint': {
        'x_range': (-2, 2)  # Only override x_range
    }
}

env = RandomizedBlimp(
    modelPath='diff.xml',
    randomization_config=custom_config
)
```

## Observation Space

```python
observation = [
    position,    # numpy array shape (3,)   - [x, y, z]
    pixels,      # numpy array shape (H,W,3) or None
    gyro_data    # numpy array shape (3,)   - [wx, wy, wz]
]
```

## Action Space

```python
action = [motor1, motor2, servo1, servo2]

# motor1, motor2: float in [-1, 1]
#   Scaled to [-2, 2] Newtons thrust
#   Controls left and right motors

# servo1, servo2: float in [-1, 1]
#   Newton-meters torque
#   Controls propeller tilt angle
```

## Reward Function

Default implementation in `randomized_blimp.py`:

```python
def reward_calculation(self) -> float:
    loc = self.get_ground_truth()[0]
    err_x = loc[0] - self.waypoint[0]
    err_y = loc[1] - self.waypoint[1]
    err_z = loc[2] - self.waypoint[2]
    return -np.linalg.norm([err_x, err_y, err_z])
```

Can be overridden in custom subclass.

## Testing

### `test_environment.py`

Automated test suite covering:
1. Module imports
2. Deterministic environment
3. Randomized environment
4. Weather presets
5. Observation space structure
6. Action space functionality
7. Episode reset

**Run tests:**
```bash
python test_environment.py
```

## Dependencies

From `pyproject.toml`:

- **mujoco** - Physics simulation engine
- **opencv-python** - Rendering and video output
- **numpy** - Numerical computations
- **setuptools** - Package management

## Version History

- **v1.0.0** - Initial release with full randomization
  - Comprehensive domain randomization
  - Weather presets
  - Obstacle generation
  - Fixed motor controls
  - Stabilized servos
  - Complete documentation

## Development Guidelines

### Adding New Features

1. **New randomization feature:**
   - Add to `environment_randomizer.py`
   - Update config schema
   - Add to documentation

2. **New reward function:**
   - Subclass `RandomizedBlimp`
   - Override `reward_calculation()`
   - Document in docstring

3. **New observation:**
   - Modify `get_obs()` in `randomized_blimp.py`
   - Update documentation
   - Update tests

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Document all public methods
- Add examples for complex features

### Testing

Before committing:
1. Run `python test_environment.py`
2. Test with `randomize=True` and `randomize=False`
3. Verify all weather presets work
4. Check documentation is updated

## File Sizes

Approximate sizes:
- `randomized_blimp.py`: ~10 KB
- `environment_randomizer.py`: ~15 KB
- `blimp.py`: ~8 KB
- `diff.xml`: ~5 KB
- `assets/`: ~50 MB (3D models)

## Future Enhancements

Potential additions:
- [ ] More obstacle course types
- [ ] Multi-blimp scenarios
- [ ] Advanced sensor models
- [ ] Vision-based observations
- [ ] Collision detection
- [ ] Battery simulation
- [ ] Communication delays
- [ ] Wind turbulence models

## Maintenance

**Active files:**
- `randomized_blimp.py` ✓
- `environment_randomizer.py` ✓
- `diff.xml` ✓

**Legacy (maintain for compatibility):**
- `blimp.py`

**Documentation:**
- `README.md` - Keep updated
- `RANDOMIZATION_GUIDE.md` - Update with new features
- `QUICKSTART.md` - Keep simple and concise

## Contact & Support

For issues, questions, or contributions:
- Check documentation first
- Run test suite
- Review examples
- Open GitHub issue if needed

---

**Last Updated:** 2025-10-15
