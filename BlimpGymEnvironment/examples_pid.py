"""
Example usage of PID controllers for blimp waypoint navigation.

Run this file to see the PID controller in action.
"""

import numpy as np
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp
from BlimpGymEnvironment.controllers import (
    BlimpWaypointController,
    AggressivePIDController,
    ConservativePIDController,
)


def extract_state(env, obs):
    """
    Extract position and yaw from environment observations.

    Args:
        env: Blimp environment
        obs: Observation from environment

    Returns:
        position: [x, y, z]
        yaw: Current yaw angle (radians)
    """
    # Position from observation
    position = obs[0]  # [x, y, z]

    # Get yaw from rotation matrix
    # For a rotation matrix R, yaw = atan2(R[1,0], R[0,0])
    rot_matrix = env.d.geom("controller").xmat.reshape(3, 3)
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

    return position, yaw


def example_1_single_waypoint():
    """Example 1: Navigate to a single waypoint."""
    print("=" * 60)
    print("Example 1: Single Waypoint Navigation")
    print("=" * 60)

    # Create environment
    env = RandomizedBlimp(
        modelPath="diff.xml",
        render_mode="",  # Set to 'human' to visualize
        randomize=False,
    )

    # Create controller
    controller = BlimpWaypointController(dt=0.02)

    # Reset environment
    obs, _ = env.reset()
    position, yaw = extract_state(env, obs)

    # Set waypoint
    waypoint = np.array([2.0, 2.0, 1.5])
    controller.set_waypoint(waypoint)

    print(f"Starting position: {position}")
    print(f"Target waypoint: {waypoint}")
    print()

    # Control loop
    max_steps = 1500
    for step in range(max_steps):
        # Get current state
        position, yaw = extract_state(env, obs)

        # Compute control action
        action = controller.compute_control(position, yaw)

        # Step environment
        obs, reward, done, _ = env.step(action)

        # Print status every 50 steps
        if step % 50 == 0:
            status = controller.get_status()
            print(
                f"Step {step:3d}: Distance = {status['distance_to_waypoint']:.3f} m, "
                f"Position = [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"
            )

        # Check if reached
        if controller.at_waypoint(threshold=0.3):
            print(f"\n✓ Reached waypoint at step {step}!")
            print(f"Final position: {position}")
            print(f"Final distance: {status['distance_to_waypoint']:.3f} m")
            break

    if not controller.at_waypoint(threshold=0.3):
        print(f"\n✗ Did not reach waypoint within {max_steps} steps")
        print(
            f"Final distance: {controller.get_status()['distance_to_waypoint']:.3f} m"
        )

    print()


def example_2_multi_waypoint():
    """Example 2: Sequential waypoint navigation."""
    print("=" * 60)
    print("Example 2: Multi-Waypoint Navigation")
    print("=" * 60)

    # Create environment
    env = RandomizedBlimp(modelPath="diff.xml", render_mode="", randomize=False)

    # Create controller
    controller = BlimpWaypointController(dt=0.02)

    # Define waypoint sequence
    waypoints = [
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 1.0, 1.5]),
        np.array([2.0, 2.0, 1.0]),
        np.array([0.0, 0.0, 1.2]),
    ]

    # Reset environment
    obs, _ = env.reset()

    print("Waypoint sequence:")
    for i, wp in enumerate(waypoints):
        print(f"  {i + 1}. {wp}")
    print()

    # Navigate through waypoints
    for wp_idx, waypoint in enumerate(waypoints):
        print(f"Navigating to waypoint {wp_idx + 1}: {waypoint}")
        controller.set_waypoint(waypoint)

        max_steps = 300
        for step in range(max_steps):
            position, yaw = extract_state(env, obs)
            action = controller.compute_control(position, yaw)
            obs, reward, done, _ = env.step(action)

            if controller.at_waypoint(threshold=0.3):
                print(f"  ✓ Reached at step {step}")
                break

        if not controller.at_waypoint(threshold=0.3):
            print(f"  ✗ Timeout")

        print()


def example_3_controller_comparison():
    """Example 3: Compare different controller types."""
    print("=" * 60)
    print("Example 3: Controller Comparison")
    print("=" * 60)

    waypoint = np.array([2.0, 2.0, 1.5])

    controllers = [
        ("Conservative", ConservativePIDController()),
        ("Standard", BlimpWaypointController()),
        ("Aggressive", AggressivePIDController()),
    ]

    results = []

    for name, controller in controllers:
        print(f"\nTesting {name} Controller...")

        # Create fresh environment
        env = RandomizedBlimp(
            modelPath="diff.xml", render_mode="", randomize=False, seed=42
        )

        obs, _ = env.reset()
        controller.set_waypoint(waypoint)

        max_steps = 500
        reached = False

        for step in range(max_steps):
            position, yaw = extract_state(env, obs)
            action = controller.compute_control(position, yaw)
            obs, reward, done, _ = env.step(action)

            if controller.at_waypoint(threshold=0.3):
                reached = True
                break

        status = controller.get_status()
        results.append(
            {
                "name": name,
                "reached": reached,
                "steps": step + 1 if reached else max_steps,
                "final_distance": status["distance_to_waypoint"],
            }
        )

        if reached:
            print(f"  ✓ Reached in {step + 1} steps")
        else:
            print(
                f"  ✗ Did not reach (final distance: {status['distance_to_waypoint']:.3f} m)"
            )

    # Summary
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"{'Controller':<15} {'Reached':<10} {'Steps':<10} {'Final Dist (m)':<15}")
    print("-" * 60)
    for result in results:
        reached_str = "Yes" if result["reached"] else "No"
        print(
            f"{result['name']:<15} {reached_str:<10} {result['steps']:<10} {result['final_distance']:<15.3f}"
        )
    print()


def example_4_random_waypoints():
    """Example 4: Navigate to random waypoints with obstacles."""
    print("=" * 60)
    print("Example 4: Random Waypoints with Obstacles")
    print("=" * 60)

    # Create environment with randomization
    env = RandomizedBlimp(
        modelPath="diff.xml",
        render_mode="",
        randomize=True,
        weather_preset="windy",
        seed=42,
    )

    controller = BlimpWaypointController(dt=0.02)

    n_waypoints = 5
    success_count = 0

    for episode in range(n_waypoints):
        print(f"\nEpisode {episode + 1}/{n_waypoints}")

        # Reset with new environment
        obs, info = env.reset(regenerate_environment=True)
        waypoint = np.array(info["waypoint"])

        print(f"  Target: {waypoint}")
        print(f"  Weather: {info['weather']}")

        controller.set_waypoint(waypoint)

        max_steps = 400
        for step in range(max_steps):
            position, yaw = extract_state(env, obs)
            action = controller.compute_control(position, yaw)
            obs, reward, done, _ = env.step(action)

            if controller.at_waypoint(threshold=0.5):
                print(f"  ✓ Reached at step {step}")
                success_count += 1
                break

        if not controller.at_waypoint(threshold=0.5):
            status = controller.get_status()
            print(
                f"  ✗ Timeout (final distance: {status['distance_to_waypoint']:.3f} m)"
            )

    print(
        f"\nSuccess rate: {success_count}/{n_waypoints} ({100 * success_count / n_waypoints:.1f}%)"
    )
    print()


def example_5_custom_gains():
    """Example 5: Custom PID gains."""
    print("=" * 60)
    print("Example 5: Custom PID Gains")
    print("=" * 60)

    # Define custom gains
    custom_gains = {
        "x": (0.5, 0.02, 0.15),  # (kp, ki, kd)
        "y": (0.5, 0.02, 0.15),
        "z": (0.8, 0.03, 0.2),
    }
    custom_yaw_gains = (1.0, 0.01, 0.3)

    # Create controller with custom gains
    controller = BlimpWaypointController(
        position_gains=custom_gains, yaw_gains=custom_yaw_gains, dt=0.02
    )

    # Create environment
    env = RandomizedBlimp(modelPath="diff.xml", render_mode="", randomize=False)

    obs, _ = env.reset()

    waypoint = np.array([2.5, 2.5, 1.8])
    controller.set_waypoint(waypoint)

    print(f"Custom gains:")
    print(
        f"  Position (x): kp={custom_gains['x'][0]}, ki={custom_gains['x'][1]}, kd={custom_gains['x'][2]}"
    )
    print(
        f"  Position (z): kp={custom_gains['z'][0]}, ki={custom_gains['z'][1]}, kd={custom_gains['z'][2]}"
    )
    print(
        f"  Yaw: kp={custom_yaw_gains[0]}, ki={custom_yaw_gains[1]}, kd={custom_yaw_gains[2]}"
    )
    print(f"\nTarget waypoint: {waypoint}")
    print()

    max_steps = 500
    for step in range(max_steps):
        position, yaw = extract_state(env, obs)
        action = controller.compute_control(position, yaw)
        obs, reward, done, _ = env.step(action)

        if step % 50 == 0:
            status = controller.get_status()
            print(f"Step {step:3d}: Distance = {status['distance_to_waypoint']:.3f} m")

        if controller.at_waypoint(threshold=0.3):
            print(f"\n✓ Reached waypoint at step {step}!")
            break

    print()


def example_6_performance_metrics():
    """Example 6: Detailed performance metrics."""
    print("=" * 60)
    print("Example 6: Performance Metrics")
    print("=" * 60)

    env = RandomizedBlimp(modelPath="diff.xml", render_mode="", randomize=False)

    controller = BlimpWaypointController(dt=0.02)

    obs, _ = env.reset()
    position, yaw = extract_state(env, obs)

    waypoint = np.array([2.0, 2.0, 1.5])
    controller.set_waypoint(waypoint)

    # Track metrics
    distances = []
    actions_history = []
    positions = []

    print(f"Starting position: {position}")
    print(f"Target waypoint: {waypoint}")
    print(f"Initial distance: {np.linalg.norm(waypoint - position):.3f} m")
    print()

    max_steps = 400
    for step in range(max_steps):
        position, yaw = extract_state(env, obs)
        action = controller.compute_control(position, yaw)
        obs, reward, done, _ = env.step(action)

        # Record metrics
        distance = np.linalg.norm(waypoint - position)
        distances.append(distance)
        actions_history.append(action.copy())
        positions.append(position.copy())

        if controller.at_waypoint(threshold=0.3):
            break

    # Compute metrics
    distances = np.array(distances)
    actions_history = np.array(actions_history)
    positions = np.array(positions)

    print("Performance Metrics:")
    print(f"  Steps to reach: {step + 1}")
    print(f"  Final distance: {distances[-1]:.3f} m")
    print(f"  Min distance: {distances.min():.3f} m")
    print(f"  Avg distance: {distances.mean():.3f} m")
    print(
        f"  Distance reduction rate: {(distances[0] - distances[-1]) / step:.4f} m/step"
    )
    print()
    print(
        f"  Total path length: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.3f} m"
    )
    print(f"  Direct distance: {np.linalg.norm(waypoint - positions[0]):.3f} m")
    print(
        f"  Path efficiency: {np.linalg.norm(waypoint - positions[0]) / np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.3f}"
    )
    print()
    print(f"  Avg motor command: {actions_history[:, :2].mean():.3f}")
    print(f"  Avg servo command: {actions_history[:, 2:].mean():.3f}")
    print(f"  Max motor command: {actions_history[:, :2].max():.3f}")
    print(f"  Max servo command: {actions_history[:, 2:].max():.3f}")
    print()


def run_all_examples():
    """Run all examples sequentially."""
    examples = [
        example_1_single_waypoint,
        example_2_multi_waypoint,
        example_3_controller_comparison,
        example_4_random_waypoints,
        example_5_custom_gains,
        example_6_performance_metrics,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback

            traceback.print_exc()
        print("\n")


if __name__ == "__main__":
    # Run individual example:
    example_1_single_waypoint()

    # Or run all examples:
    # run_all_examples()
