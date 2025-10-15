"""
PID Controllers for Blimp Waypoint Navigation

This module provides PID-based controllers for autonomous blimp navigation.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque


class PIDController:
    """
    Single-axis PID controller.

    Implements discrete PID control with anti-windup and derivative filtering.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limits: Tuple[float, float] = (-1.0, 1.0),
        integral_limits: Tuple[float, float] = (-10.0, 10.0),
        dt: float = 0.02,
        derivative_filter_alpha: float = 0.1,
    ):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Min/max output values
            integral_limits: Min/max integral accumulation (anti-windup)
            dt: Time step (seconds)
            derivative_filter_alpha: Low-pass filter coefficient for derivative (0-1)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.dt = dt
        self.derivative_filter_alpha = derivative_filter_alpha

        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def update(self, setpoint: float, measurement: float) -> float:
        """
        Compute PID control output.

        Args:
            setpoint: Desired value
            measurement: Current measured value

        Returns:
            Control output
        """
        # Compute error
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, *self.integral_limits)
        i_term = self.ki * self.integral

        # Derivative term with filtering
        derivative = (error - self.prev_error) / self.dt
        filtered_derivative = (
            self.derivative_filter_alpha * derivative
            + (1 - self.derivative_filter_alpha) * self.prev_derivative
        )
        d_term = self.kd * filtered_derivative

        # Compute output
        output = p_term + i_term + d_term
        output = np.clip(output, *self.output_limits)

        # Update state
        self.prev_error = error
        self.prev_derivative = filtered_derivative

        return output


class BlimpWaypointController:
    """
    Cascaded PID controller for 3D waypoint navigation.

    Uses separate PID controllers for:
    - X, Y, Z position control
    - Yaw heading control

    Outputs motor and servo commands for the blimp.
    """

    def __init__(
        self,
        position_gains: Optional[Dict[str, Tuple[float, float, float]]] = None,
        yaw_gains: Optional[Tuple[float, float, float]] = None,
        dt: float = 0.02,
        max_tilt_angle: float = 30.0,
        heading_control: bool = True,
    ):
        """
        Initialize waypoint controller.

        Args:
            position_gains: Dict with keys 'x', 'y', 'z' containing (kp, ki, kd) tuples
            yaw_gains: Tuple of (kp, ki, kd) for yaw control
            dt: Time step (seconds)
            max_tilt_angle: Maximum propeller tilt angle (degrees)
            heading_control: Whether to use heading control for forward flight
        """
        # Default gains tuned for blimp dynamics
        if position_gains is None:
            position_gains = {
                "x": (0.6, 0.01, 0.2),  # Forward/backward
                "y": (0.6, 0.01, 0.2),  # Left/right
                "z": (
                    0.4,
                    0.0,
                    0.2,
                ),  # Up/down (altitude) - no integral to prevent windup
            }

        if yaw_gains is None:
            yaw_gains = (1.5, 0.0, 0.3)  # Yaw rotation

        self.dt = dt
        self.max_tilt_angle = max_tilt_angle
        self.heading_control = heading_control

        # Position controllers (world frame)
        self.pid_x = PIDController(
            kp=position_gains["x"][0],
            ki=position_gains["x"][1],
            kd=position_gains["x"][2],
            output_limits=(-1.0, 1.0),
            dt=dt,
        )

        self.pid_y = PIDController(
            kp=position_gains["y"][0],
            ki=position_gains["y"][1],
            kd=position_gains["y"][2],
            output_limits=(-1.0, 1.0),
            dt=dt,
        )

        self.pid_z = PIDController(
            kp=position_gains["z"][0],
            ki=position_gains["z"][1],
            kd=position_gains["z"][2],
            output_limits=(-1.0, 1.0),
            integral_limits=(-5.0, 5.0),  # Tighter integral limits for altitude
            dt=dt,
        )

        # Yaw controller
        self.pid_yaw = PIDController(
            kp=yaw_gains[0],
            ki=yaw_gains[1],
            kd=yaw_gains[2],
            output_limits=(-1.0, 1.0),
            dt=dt,
        )

        # State tracking
        self.current_waypoint = None
        self.total_distance = 0.0
        self.distance_history = deque(maxlen=10)

    def reset(self):
        """Reset all controllers."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()
        self.distance_history.clear()

    def set_waypoint(self, waypoint: np.ndarray):
        """
        Set new target waypoint.

        Args:
            waypoint: [x, y, z] target position
        """
        self.current_waypoint = np.array(waypoint)
        self.reset()  # Reset controller state for new waypoint

    def compute_control(
        self,
        current_position: np.ndarray,
        current_yaw: float,
        waypoint: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute control commands to reach waypoint.

        Args:
            current_position: [x, y, z] current position
            current_yaw: Current yaw angle (radians)
            waypoint: Optional [x, y, z] target (uses stored waypoint if None)

        Returns:
            action: [motor1, motor2, servo1, servo2] control commands
        """
        if waypoint is not None:
            self.current_waypoint = np.array(waypoint)

        if self.current_waypoint is None:
            return np.array([0.0, 0.0, 0.0, 0.0])

        current_position = np.array(current_position)

        # Compute position error
        error = self.current_waypoint - current_position
        distance = np.linalg.norm(error)
        self.distance_history.append(distance)

        # PID control for each axis
        u_x = self.pid_x.update(self.current_waypoint[0], current_position[0])
        u_y = self.pid_y.update(self.current_waypoint[1], current_position[1])
        u_z = self.pid_z.update(self.current_waypoint[2], current_position[2])

        # Compute desired heading from position error
        if self.heading_control and np.linalg.norm(error[:2]) > 0.1:
            desired_yaw = np.arctan2(error[1], error[0])
        else:
            desired_yaw = current_yaw

        # Normalize angle difference to [-pi, pi]
        yaw_error = self._normalize_angle(desired_yaw - current_yaw)
        u_yaw = self.pid_yaw.update(0, yaw_error)  # setpoint=0 means we want error=0

        # Allocate controls to motors and servos
        action = self._allocate_controls(u_x, u_y, u_z, u_yaw, error)

        return action

    def _allocate_controls(
        self, u_x: float, u_y: float, u_z: float, u_yaw: float, error: np.ndarray
    ) -> np.ndarray:
        """
        Allocate control efforts to motor thrust and servo angles.

        Control strategy:
        - Z-axis (altitude): Controlled by symmetric motor thrust
        - XY-plane (horizontal): Controlled by propeller tilt
        - Yaw: Controlled by differential thrust

        Args:
            u_x, u_y, u_z: Position control outputs
            u_yaw: Yaw control output
            error: 3D position error vector

        Returns:
            action: [motor1, motor2, servo1, servo2]
        """
        # Base thrust for altitude control
        # Positive u_z means we want to go up (increase thrust)
        # Start with moderate base thrust (buoyancy compensates most of gravity)
        base_thrust = 0.5 + 0.3 * np.clip(u_z, -1, 1)

        # Horizontal control magnitude
        horizontal_command = np.sqrt(u_x**2 + u_y**2)
        horizontal_command = np.clip(horizontal_command, 0, 1)

        # Servo tilt for horizontal motion
        # The servos control tilt in the X direction (forward/back)
        # Since the blimp rotates to face the waypoint, we primarily use
        # the magnitude of horizontal error for forward thrust
        #
        # Servo action range: [-1, 1] maps to servo joint angles
        # Positive tilt should push blimp forward (positive X)

        # For horizontal motion, apply servo torque
        # Since servos use torque control, we need stronger values
        if horizontal_command > 0.05:
            # Apply torque proportional to horizontal error magnitude
            # Torque control needs much higher values than position control
            servo_torque = np.clip(horizontal_command * 3.0, 0, 1.0)
            servo1_tilt = servo_torque
            servo2_tilt = servo_torque

            # Add minimal thrust compensation
            # Too much compensation causes altitude overshoot
            thrust_compensation = 0.1 * horizontal_command
            base_thrust = np.clip(base_thrust + thrust_compensation, 0, 1)
        else:
            # Apply small holding torque to keep servos vertical
            servo1_tilt = 0.0
            servo2_tilt = 0.0

        # Differential thrust for yaw control (increased strength)
        yaw_differential = 0.5 * u_yaw

        # Motor commands with differential for yaw
        motor1 = np.clip(base_thrust - yaw_differential, 0, 1)
        motor2 = np.clip(base_thrust + yaw_differential, 0, 1)

        # Servo commands
        servo1 = servo1_tilt
        servo2 = servo2_tilt

        return np.array([motor1, motor2, servo1, servo2])

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_status(self) -> Dict:
        """
        Get controller status information.

        Returns:
            Dict with status information
        """
        if len(self.distance_history) > 0:
            current_distance = self.distance_history[-1]
            avg_distance = np.mean(self.distance_history)
        else:
            current_distance = 0.0
            avg_distance = 0.0

        return {
            "current_waypoint": self.current_waypoint,
            "distance_to_waypoint": current_distance,
            "average_distance": avg_distance,
            "integral_x": self.pid_x.integral,
            "integral_y": self.pid_y.integral,
            "integral_z": self.pid_z.integral,
            "integral_yaw": self.pid_yaw.integral,
        }

    def at_waypoint(self, threshold: float = 0.5) -> bool:
        """
        Check if blimp has reached the waypoint.

        Args:
            threshold: Distance threshold (meters)

        Returns:
            True if within threshold of waypoint
        """
        if len(self.distance_history) == 0:
            return False
        return self.distance_history[-1] < threshold


class AggressivePIDController(BlimpWaypointController):
    """
    Aggressive PID controller for faster waypoint reaching.

    Uses higher gains and more aggressive control allocation.
    """

    def __init__(self, dt: float = 0.02):
        position_gains = {
            "x": (0.8, 0.05, 0.3),
            "y": (0.8, 0.05, 0.3),
            "z": (1.2, 0.08, 0.4),
        }
        yaw_gains = (1.5, 0.02, 0.4)

        super().__init__(
            position_gains=position_gains,
            yaw_gains=yaw_gains,
            dt=dt,
            max_tilt_angle=45.0,
            heading_control=True,
        )


class ConservativePIDController(BlimpWaypointController):
    """
    Conservative PID controller for gentle, stable flight.

    Uses lower gains for smooth trajectories.
    """

    def __init__(self, dt: float = 0.02):
        position_gains = {
            "x": (0.15, 0.005, 0.05),
            "y": (0.15, 0.005, 0.05),
            "z": (0.25, 0.01, 0.08),
        }
        yaw_gains = (0.4, 0.0, 0.1)

        super().__init__(
            position_gains=position_gains,
            yaw_gains=yaw_gains,
            dt=dt,
            max_tilt_angle=20.0,
            heading_control=True,
        )


def auto_tune_pid(
    env,
    initial_gains: Tuple[float, float, float] = (0.5, 0.01, 0.1),
    n_episodes: int = 10,
    waypoint: Optional[np.ndarray] = None,
) -> Dict:
    """
    Simple auto-tuning for PID gains using Ziegler-Nichols-inspired approach.

    Args:
        env: Blimp environment
        initial_gains: Starting (kp, ki, kd) values
        n_episodes: Number of test episodes
        waypoint: Test waypoint (random if None)

    Returns:
        Dict with tuned gains and performance metrics
    """
    # This is a placeholder for a more sophisticated auto-tuning algorithm
    # In practice, you would implement:
    # 1. Relay feedback test to find ultimate gain Ku and period Tu
    # 2. Apply Ziegler-Nichols rules
    # 3. Fine-tune based on overshoot/settling time

    print("Auto-tuning not fully implemented. Use manual tuning or the default gains.")
    return {
        "position_gains": {"x": initial_gains, "y": initial_gains, "z": initial_gains},
        "yaw_gains": initial_gains,
    }
