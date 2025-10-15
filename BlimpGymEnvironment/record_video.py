"""
Record videos and GIFs of blimp waypoint navigation.

This script uses the PID controller to navigate to waypoints and records
the visualization as video files and animated GIFs.
"""

import numpy as np
import imageio
import os
from pathlib import Path
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp
from BlimpGymEnvironment.controllers import (
    BlimpWaypointController,
    AggressivePIDController,
    ConservativePIDController,
)


def extract_state(env, obs):
    """Extract position and yaw from environment observations."""
    position = obs[0]
    rot_matrix = env.d.geom("controller").xmat.reshape(3, 3)
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return position, yaw


def record_waypoint_navigation(
    waypoint,
    controller_type="standard",
    max_steps=1500,
    output_dir="videos",
    filename_prefix="blimp_nav",
    camera_name="followCamera",
    fps=30,
    save_video=True,
    save_gif=True,
):
    """
    Record blimp navigation to a waypoint.

    Args:
        waypoint: Target [x, y, z] position
        controller_type: "standard", "aggressive", or "conservative"
        max_steps: Maximum simulation steps
        output_dir: Directory to save outputs
        filename_prefix: Prefix for output files
        camera_name: MuJoCo camera to use ("followCamera" or "blimpCamera")
        fps: Frames per second for video
        save_video: Whether to save as MP4 video
        save_gif: Whether to save as animated GIF

    Returns:
        dict with recording statistics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environment with rendering
    print(f"Creating environment with {camera_name} camera...")
    env = RandomizedBlimp(
        modelPath="diff.xml", render_mode="rgb_array", randomize=False
    )
    print(
        f"Environment render_mode: {env.render_mode if hasattr(env, 'render_mode') else 'NOT SET'}"
    )
    print(f"Has renderer: {hasattr(env, 'renderer')}")

    # Create controller
    if controller_type == "aggressive":
        controller = AggressivePIDController(dt=0.02)
        controller_name = "Aggressive"
    elif controller_type == "conservative":
        controller = ConservativePIDController(dt=0.02)
        controller_name = "Conservative"
    else:
        controller = BlimpWaypointController(dt=0.02)
        controller_name = "Standard"

    print(f"Using {controller_name} PID Controller")

    # Reset environment
    obs, _ = env.reset()
    position, yaw = extract_state(env, obs)

    # Set waypoint
    controller.set_waypoint(waypoint)
    env.update_waypoint(waypoint)  # Visualize waypoint in MuJoCo

    print(f"Starting position: {position}")
    print(f"Target waypoint: {waypoint}")
    print(f"Initial distance: {np.linalg.norm(waypoint - position):.2f}m")
    print()

    # Recording setup
    frames = []
    frame_skip = max(1, int(50 / fps))  # Simulation runs at 50Hz
    min_distance = float("inf")
    min_distance_step = 0
    reached_waypoint = False

    # Control loop
    print("Recording navigation...")
    for step in range(max_steps):
        # Get current state
        position, yaw = extract_state(env, obs)

        # Compute control action
        action = controller.compute_control(position, yaw)

        # Step environment
        obs, reward, done, _ = env.step(action)

        # Record frame
        if step % frame_skip == 0:
            # Render the scene
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            elif step == 0:
                print(f"Warning: render() returned None at step {step}")

        # Track statistics
        distance = controller.get_status()["distance_to_waypoint"]
        if distance < min_distance:
            min_distance = distance
            min_distance_step = step

        # Check if reached
        if controller.at_waypoint(threshold=0.5):
            if not reached_waypoint:
                print(f"✓ Reached waypoint at step {step} ({step * 0.02:.1f}s)!")
                reached_waypoint = True
            # Continue recording for a bit after reaching
            if step > min_distance_step + 100:
                break

        # Progress updates
        if step % 250 == 0:
            print(
                f"Step {step:4d}: Distance = {distance:.2f}m, "
                f"Position = [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"
            )

    # Final statistics
    final_distance = controller.get_status()["distance_to_waypoint"]
    print()
    print("=" * 60)
    print("Recording Statistics:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Total steps: {step + 1}")
    print(f"  Duration: {(step + 1) * 0.02:.1f}s")
    print(f"  Min distance: {min_distance:.2f}m at step {min_distance_step}")
    print(f"  Final distance: {final_distance:.2f}m")
    print(f"  Reached waypoint: {reached_waypoint}")
    print("=" * 60)

    # Save video
    if save_video and frames:
        video_filename = f"{filename_prefix}_{controller_type}_{waypoint[0]:.1f}_{waypoint[1]:.1f}_{waypoint[2]:.1f}.mp4"
        video_path = output_path / video_filename
        print(f"\nSaving video to {video_path}...")

        # Use imageio to save video
        imageio.mimsave(
            video_path,
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        print(f"✓ Video saved: {video_path}")

    # Save GIF
    if save_gif and frames:
        gif_filename = f"{filename_prefix}_{controller_type}_{waypoint[0]:.1f}_{waypoint[1]:.1f}_{waypoint[2]:.1f}.gif"
        gif_path = output_path / gif_filename
        print(f"Saving GIF to {gif_path}...")

        # Downsample frames for smaller GIF
        gif_frames = frames[::2]  # Use every other frame
        # Optionally resize frames for smaller file
        # gif_frames = [frame[::2, ::2] for frame in gif_frames]

        imageio.mimsave(gif_path, gif_frames, fps=fps // 2, loop=0)
        print(f"✓ GIF saved: {gif_path}")

    # Note: Blimp environment doesn't have close() method
    # env.close()

    return {
        "reached_waypoint": reached_waypoint,
        "min_distance": min_distance,
        "final_distance": final_distance,
        "total_steps": step + 1,
        "total_frames": len(frames),
    }


def main():
    """Record several example navigations."""
    print("=" * 60)
    print("Blimp Waypoint Navigation Video Recording")
    print("=" * 60)
    print()

    # Check for imageio-ffmpeg
    try:
        import imageio_ffmpeg

        print("✓ imageio-ffmpeg found")
    except ImportError:
        print("⚠ imageio-ffmpeg not found. Installing...")
        import subprocess

        subprocess.check_call(["pip", "install", "imageio-ffmpeg"])
        print("✓ imageio-ffmpeg installed")

    print()

    # Example 1: Close waypoint with standard controller
    print("\n" + "=" * 60)
    print("Example 1: Close waypoint (1m, 1m, 1.2m) - Standard Controller")
    print("=" * 60)
    record_waypoint_navigation(
        waypoint=np.array([1.0, 1.0, 1.2]),
        controller_type="standard",
        max_steps=1000,
        filename_prefix="example1_close",
        fps=20,
    )

    # Example 2: Medium waypoint with aggressive controller
    print("\n" + "=" * 60)
    print("Example 2: Medium waypoint (1.5, 1.5, 1.3m) - Aggressive Controller")
    print("=" * 60)
    record_waypoint_navigation(
        waypoint=np.array([1.5, 1.5, 1.3]),
        controller_type="aggressive",
        max_steps=1200,
        filename_prefix="example2_medium",
        fps=20,
    )

    # Example 3: Conservative controller
    print("\n" + "=" * 60)
    print("Example 3: Close waypoint (0.8, 0.8, 1.0m) - Conservative Controller")
    print("=" * 60)
    record_waypoint_navigation(
        waypoint=np.array([0.8, 0.8, 1.0]),
        controller_type="conservative",
        max_steps=1500,
        filename_prefix="example3_conservative",
        fps=20,
    )

    # Example 4: Different camera angle
    print("\n" + "=" * 60)
    print("Example 4: Blimp camera view (1.2, 1.2, 1.2m)")
    print("=" * 60)
    record_waypoint_navigation(
        waypoint=np.array([1.2, 1.2, 1.2]),
        controller_type="standard",
        max_steps=1200,
        filename_prefix="example4_blimp_cam",
        camera_name="blimpCamera",
        fps=20,
    )

    print("\n" + "=" * 60)
    print("All recordings complete! Check the 'videos' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
