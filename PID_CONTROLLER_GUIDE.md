# PID Controller Guide

## Overview

This guide explains how to use the PID-based waypoint controllers for autonomous blimp navigation.

## Controller Classes

### `BlimpWaypointController`
Default PID controller for waypoint navigation with moderate gains.

### `AggressivePIDController`  
Higher-gain variant for faster response (may overshoot).

### `ConservativePIDController`
Lower-gain variant for smooth, stable flight.

## Important Limitations

### Underactuated System
The blimp is an **underactuated system** with only 4 actuators:
- 2 motors (differential thrust for yaw + vertical thrust)
- 2 servos (tilt forward/backward only)

**Key constraint**: The blimp **cannot move sideways**. To reach a waypoint, it must:
1. Rotate (yaw) to face the target
2. Tilt forward and thrust to move toward it  
3. Adjust altitude with vertical thrust

This makes waypoint navigation inherently slow compared to fully-actuated systems like quadcopters.

### Servo Torque Control
The servos use **torque control**, not position control. This means:
- Servo commands apply torque to tilt the motors
- Higher torque values are needed compared to position control
- The tilt angle depends on the physics (torque vs drag vs inertia)

### Typical Performance
With default gains on a waypoint 3.2m away:
- **Best distance achieved**: ~1.97m at step 800 (1500 steps = 30 seconds)
- **Common issues**: Altitude overshoot, slow convergence
- **Recommendation**: Use multiple intermediate waypoints rather than direct long-distance navigation

## Usage Examples

### Basic Single Waypoint

```python
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp
from BlimpGymEnvironment.controllers import BlimpWaypointController
import numpy as np

env = RandomizedBlimp(modelPath='diff.xml', render_mode='', randomize=False)
controller = BlimpWaypointController(dt=0.02)

obs, _ = env.reset()
waypoint = np.array([1.0, 1.0, 1.2])  # Nearby waypoint
controller.set_waypoint(waypoint)

for step in range(1000):
    # Extract state
    position = obs[0]
    rot_matrix = env.d.geom("controller").xmat.reshape(3, 3)
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    
    # Compute control
    action = controller.compute_control(position, yaw)
    obs, reward, done, _ = env.step(action)
    
    if controller.at_waypoint(threshold=0.5):
        print(f"Reached waypoint at step {step}!")
        break
```

### Multi-Waypoint Navigation

For distant targets, use multiple intermediate waypoints:

```python
waypoints = [
    np.array([0.5, 0.5, 1.0]),
    np.array([1.0, 1.0, 1.2]),
    np.array([1.5, 1.5, 1.3]),
    np.array([2.0, 2.0, 1.5]),  # Final target
]

for wp in waypoints:
    controller.set_waypoint(wp)
    
    while not controller.at_waypoint(threshold=0.4):
        position, yaw = extract_state(env, obs)
        action = controller.compute_control(position, yaw)
        obs, _, _, _ = env.step(action)
```

### Custom PID Gains

Tune gains for your specific application:

```python
position_gains = {
    "x": (0.4, 0.005, 0.15),  # (kp, ki, kd) for X axis
    "y": (0.4, 0.005, 0.15),  # (kp, ki, kd) for Y axis  
    "z": (0.3, 0.0, 0.1),     # (kp, ki, kd) for altitude
}

yaw_gains = (1.0, 0.0, 0.2)  # (kp, ki, kd) for yaw

controller = BlimpWaypointController(
    position_gains=position_gains,
    yaw_gains=yaw_gains,
    dt=0.02,
    heading_control=True  # Automatically face waypoint
)
```

## Tuning Guidelines

### Position Gains (kp, ki, kd)
- **kp (Proportional)**: Higher = faster response, but may overshoot
  - Recommended range: 0.2 - 1.0
- **ki (Integral)**: Eliminates steady-state error, but can cause windup
  - Recommended range: 0.0 - 0.05 (often better to keep at 0)
- **kd (Derivative)**: Dampens oscillations
  - Recommended range: 0.1 - 0.4

### Altitude Control Special Considerations
- **Avoid integral term** (set ki=0) to prevent altitude windup
- Use lower kp than horizontal axes (0.2-0.5 instead of 0.4-1.0)
- The buoyancy compensation (gravcomp in XML) means you don't need much thrust

### Yaw Control
- Can use higher kp (1.0-2.0) since differential thrust is very effective
- Usually don't need integral term
- Moderate kd (0.2-0.4) for smooth turns

## Troubleshooting

### Problem: Blimp overshoots in altitude
**Solution**: Reduce Z axis kp gain, set ki=0, reduce thrust compensation in control allocation

### Problem: Blimp doesn't reach waypoint
**Solution**: 
- Use smaller waypoint spacing (0.5-1.0m increments)
- Increase episode length (try 2000-3000 steps)
- Check if threshold is too strict (try 0.5m instead of 0.3m)

### Problem: Blimp oscillates around waypoint
**Solution**: Increase kd (derivative) gain to add damping, reduce kp gain

### Problem: Slow convergence
**Solution**: 
- Increase kp gains (but watch for overshoot)
- Use AggressivePIDController instead of default
- Ensure servo torque is sufficient (check servo_torque scaling in _allocate_controls)

## Controller Architecture

```
Position Error → PID Controllers (X, Y, Z, Yaw) → Control Allocation → Motors & Servos
```

### Control Allocation Strategy
- **Altitude**: Symmetric motor thrust (motor1 + motor2)
- **Yaw**: Differential thrust (motor1 - motor2)
- **Forward motion**: Servo tilt (both servos forward)
- **Heading control**: Automatically rotates to face waypoint

### Coordinate Frames
- **World frame**: Fixed XYZ coordinates
- **Body frame**: Blimp's local coordinate system
- Controller operates in world frame, actions in body frame

## Advanced Usage

### Disable Heading Control
If you want manual yaw control:

```python
controller = BlimpWaypointController(heading_control=False)
# Now you manually control desired yaw instead of auto-facing waypoint
```

### Monitor Controller Status

```python
status = controller.get_status()
print(f"Distance to waypoint: {status['distance_to_waypoint']:.2f}m")
print(f"Integral terms: X={status['integral_x']:.2f}, Z={status['integral_z']:.2f}")
```

### Check Waypoint Arrival

```python
if controller.at_waypoint(threshold=0.5):
    print("Within 0.5m of waypoint!")
```

## Performance Expectations

| Waypoint Distance | Expected Time | Success Rate | Notes |
|------------------|---------------|--------------|-------|
| 0.5 - 1.0m | 10-20 sec | High (~90%) | Recommended range |
| 1.0 - 2.0m | 20-40 sec | Medium (~60%) | May need tuning |
| 2.0 - 3.0m | 40-60 sec | Low (~30%) | Use intermediate waypoints |
| > 3.0m | 60+ sec | Very Low | Strongly recommend waypoint chaining |

*Based on default controller with 0.5m threshold at 50Hz (dt=0.02)*

## Examples

See `examples_pid.py` for:
1. Single waypoint navigation
2. Multi-waypoint sequential navigation
3. Controller comparison (Conservative vs Standard vs Aggressive)
4. Random waypoints with obstacles
5. Custom PID gains
6. Performance metrics tracking

## Future Improvements

Potential enhancements:
- Model Predictive Control (MPC) for trajectory optimization
- Adaptive gain scheduling based on distance
- Wind disturbance compensation
- Learn optimal gains via RL
- Trajectory smoothing with splines

## References

- MuJoCo blimp model: `diff.xml`
- Environment implementation: `blimp.py`, `randomized_blimp.py`
- Controller implementation: `controllers.py`
- Blimp dynamics: `DYNAMICS_COMPARISON.md`
