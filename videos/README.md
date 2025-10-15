# Blimp Navigation Videos

This directory contains recorded videos and animated GIFs of the blimp navigating to waypoints using PID control.

## Videos

### Demo 1: Very Close Waypoint (0.8, 0.8, 1.0m) - Standard Controller
- **Files**: `demo1_very_close_standard_0.8_0.8_1.0.mp4` / `.gif`
- **Duration**: 16 seconds (800 steps)
- **Controller**: Standard PID
- **Result**: Got within 0.73m of target
- **Notes**: Shows navigation to a nearby waypoint. The blimp makes good progress initially but doesn't quite reach the target due to altitude overshoot.

### Demo 2: Close Waypoint (1.0, 1.0, 1.2m) - Standard Controller
- **Files**: `demo2_close_standard_1.0_1.0_1.2.mp4` / `.gif`
- **Duration**: 24 seconds (1200 steps)
- **Controller**: Standard PID
- **Result**: Got within 0.82m of target
- **Notes**: Demonstrates navigation to a slightly farther waypoint. The blimp approaches within 0.82m at step 777 but then overshoots in altitude, illustrating the challenge of controlling an underactuated system.

### Demo 3: Conservative Controller (0.9, 0.9, 1.1m)
- **Files**: `demo3_conservative_conservative_0.9_0.9_1.1.mp4` / `.gif`
- **Duration**: 30 seconds (1500 steps)
- **Controller**: Conservative PID (lower gains)
- **Result**: Got within 0.95m of target
- **Notes**: Shows smoother but slower navigation with the conservative controller. Less aggressive control reduces overshoot but takes longer to converge.

## File Formats

- **MP4 videos**: Smaller file size (~4-6MB), good for playback
- **GIF animations**: Larger file size (~15-24MB), good for embedding in documentation

## How to Generate More Videos

Use the `record_video.py` module to create your own recordings:

```python
from BlimpGymEnvironment.record_video import record_waypoint_navigation
import numpy as np

record_waypoint_navigation(
    waypoint=np.array([1.0, 1.0, 1.2]),
    controller_type="standard",  # or "aggressive", "conservative"
    max_steps=1200,
    filename_prefix="my_demo",
    fps=20,
    save_video=True,
    save_gif=True,
)
```

## Observations

### What Works Well
- The blimp successfully navigates toward waypoints
- Heading control automatically rotates the blimp to face the target
- Differential thrust provides effective yaw control
- The visualizations clearly show the green waypoint cylinder as the target

### Challenges
- **Altitude control**: The blimp tends to overshoot in the Z axis after getting close to the waypoint
- **Underactuated dynamics**: Cannot move sideways, must rotate then thrust forward
- **Convergence time**: Takes 15-30 seconds to get within ~1m of target
- **Final approach**: Difficult to reach the exact waypoint due to coupled dynamics

### Recommendations
For better performance:
1. Use **waypoint chaining** with multiple intermediate waypoints (0.5-1.0m apart)
2. Increase the **waypoint threshold** to 0.5m or larger
3. Allow **more time** (2000-3000 steps) for distant targets
4. Consider **tuning PID gains** for your specific use case

## Technical Details

- **Simulation frequency**: 50 Hz (dt=0.02s)
- **Video frame rate**: 20 FPS
- **Camera**: Follow camera (third-person view)
- **Resolution**: 620x480 pixels (resized to 624x480 for codec compatibility)
- **Rendering**: MuJoCo renderer with OpenCV color conversion

## Related Documentation

- `PID_CONTROLLER_GUIDE.md` - Detailed guide on using and tuning PID controllers
- `record_video.py` - Source code for video recording
- `controllers.py` - PID controller implementation
- `examples_pid.py` - Example scripts using PID controllers
