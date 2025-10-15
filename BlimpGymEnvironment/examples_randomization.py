"""
Example usage of the Environment Randomization System

This file demonstrates various ways to use the randomization features.
"""

import numpy as np
from BlimpGymEnvironment.randomized_blimp import RandomizedBlimp
from BlimpGymEnvironment.environment_randomizer import (
    EnvironmentRandomizer,
    ObstacleCourseGenerator,
    WeatherRandomizer,
)


def example_1_basic_randomization():
    """Example 1: Basic environment randomization"""
    print("=" * 60)
    print("Example 1: Basic Randomization")
    print("=" * 60)

    # Create environment with full randomization
    env = RandomizedBlimp(
        modelPath="diff.xml",
        render_mode="",
        randomize=True,
        seed=42,  # For reproducibility
    )

    print("Initial environment created with:")
    info = env.get_randomization_info()
    print(f"  Waypoint: {info['waypoint']}")
    print(f"  Viscosity: {info['viscosity']:.6f}")
    print(f"  Wind: {info['wind']}")
    print(f"  Density: {info['density']:.3f}")
    print()

    # Run a few episodes with different randomizations
    for episode in range(3):
        print(f"Episode {episode + 1}:")
        obs, info = env.reset(regenerate_environment=True)

        print(f"  New waypoint: {env.waypoint}")
        print(f"  Wind: {env.m.opt.wind}")

        # Run 50 steps
        for step in range(50):
            action = [0.5, 0.5, 0, 0]  # Simple forward thrust
            obs, reward, done, _ = env.step(action)

        print(f"  Final reward: {reward:.3f}")
        print()


def example_2_custom_ranges():
    """Example 2: Custom randomization ranges"""
    print("=" * 60)
    print("Example 2: Custom Randomization Ranges")
    print("=" * 60)

    # Define custom randomization ranges
    custom_config = {
        "waypoint": {
            "x_range": (-2, 2),  # Narrower range
            "y_range": (-2, 2),
            "z_range": (1.0, 1.5),  # Keep waypoint at mid-height
        },
        "obstacles": {
            "count_range": (5, 8),  # More obstacles
            "size_range": (0.2, 0.5),  # Smaller obstacles
        },
        "physics": {
            "wind_range": (-0.5, 0.5),  # Less wind
        },
    }

    env = RandomizedBlimp(
        modelPath="diff.xml",
        render_mode="",
        randomize=True,
        randomization_config=custom_config,
        seed=123,
    )

    print("Custom configuration applied:")
    info = env.get_randomization_info()
    print(f"  Waypoint: {info['waypoint']}")
    print(f"  (Should be in narrower range)")
    print()


def example_3_weather_presets():
    """Example 3: Different weather conditions"""
    print("=" * 60)
    print("Example 3: Weather Presets")
    print("=" * 60)

    weather_conditions = ["calm", "windy", "turbulent", "dense", "thin"]

    for weather in weather_conditions:
        env = RandomizedBlimp(
            modelPath="diff.xml",
            render_mode="",
            randomize=True,
            weather_preset=weather,
            seed=42,
        )

        print(f"{weather.upper()} conditions:")
        print(f"  Viscosity: {env.m.opt.viscosity:.6f}")
        print(f"  Wind: {env.m.opt.wind}")
        print(f"  Density: {env.m.opt.density:.3f}")
        print()


def example_4_dynamic_waypoints():
    """Example 4: Dynamic waypoint changes during episode"""
    print("=" * 60)
    print("Example 4: Dynamic Waypoint Changes")
    print("=" * 60)

    env = RandomizedBlimp(modelPath="diff.xml", render_mode="", randomize=True, seed=42)

    obs, _ = env.reset()
    print(f"Initial waypoint: {env.waypoint}")

    # Simulate multi-waypoint navigation
    waypoints = [(1, 1, 1), (2, 2, 1.5), (-1, 3, 1), (0, 0, 1.2)]

    for i, wp in enumerate(waypoints):
        print(f"\nNavigating to waypoint {i + 1}: {wp}")
        env.update_waypoint(wp)

        # Run 30 steps
        for step in range(30):
            action = [0.5, 0.5, 0, 0]
            obs, reward, done, _ = env.step(action)

        print(f"  Final reward: {reward:.3f}")


def example_5_training_curriculum():
    """Example 5: Progressive difficulty curriculum"""
    print("=" * 60)
    print("Example 5: Training Curriculum")
    print("=" * 60)

    # Easy stage: calm weather, close waypoints, few obstacles
    print("STAGE 1: Easy")
    easy_config = {
        "waypoint": {"x_range": (-1, 1), "y_range": (-1, 1), "z_range": (0.8, 1.2)},
        "obstacles": {
            "count_range": (1, 3),
        },
        "physics": {
            "wind_range": (-0.2, 0.2),
        },
    }

    env_easy = RandomizedBlimp(
        modelPath="diff.xml",
        render_mode="",
        randomize=True,
        randomization_config=easy_config,
        weather_preset="calm",
        seed=1,
    )
    print(f"  Waypoint range: narrow")
    print(f"  Obstacles: few (1-3)")
    print(f"  Wind: minimal")
    print()

    # Medium stage: moderate wind, medium distance, more obstacles
    print("STAGE 2: Medium")
    medium_config = {
        "waypoint": {"x_range": (-3, 3), "y_range": (-3, 3), "z_range": (0.5, 2.0)},
        "obstacles": {
            "count_range": (4, 7),
        },
        "physics": {
            "wind_range": (-0.8, 0.8),
        },
    }

    env_medium = RandomizedBlimp(
        modelPath="diff.xml",
        render_mode="",
        randomize=True,
        randomization_config=medium_config,
        weather_preset="windy",
        seed=2,
    )
    print(f"  Waypoint range: medium")
    print(f"  Obstacles: moderate (4-7)")
    print(f"  Wind: moderate")
    print()

    # Hard stage: turbulent weather, far waypoints, many obstacles
    print("STAGE 3: Hard")
    hard_config = {
        "waypoint": {"x_range": (-5, 5), "y_range": (-5, 5), "z_range": (0.3, 2.5)},
        "obstacles": {
            "count_range": (8, 15),
        },
        "physics": {
            "wind_range": (-1.5, 1.5),
        },
    }

    env_hard = RandomizedBlimp(
        modelPath="diff.xml",
        render_mode="",
        randomize=True,
        randomization_config=hard_config,
        weather_preset="turbulent",
        seed=3,
    )
    print(f"  Waypoint range: wide")
    print(f"  Obstacles: many (8-15)")
    print(f"  Wind: strong and turbulent")
    print()


def example_6_statistics_tracking():
    """Example 6: Track randomization statistics"""
    print("=" * 60)
    print("Example 6: Statistics Tracking")
    print("=" * 60)

    env = RandomizedBlimp(modelPath="diff.xml", render_mode="", randomize=True, seed=42)

    # Run multiple episodes and collect stats
    waypoint_positions = []
    rewards = []

    for episode in range(10):
        obs, info = env.reset(regenerate_environment=True)
        waypoint_positions.append(env.waypoint)

        # Run episode
        episode_rewards = []
        for step in range(100):
            action = np.random.uniform(-1, 1, 4)  # Random actions
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            if done:
                break

        rewards.append(np.mean(episode_rewards))

    # Print statistics
    waypoint_positions = np.array(waypoint_positions)
    print(f"Episodes run: {len(rewards)}")
    print(f"\nWaypoint statistics:")
    print(
        f"  X range: [{waypoint_positions[:, 0].min():.2f}, {waypoint_positions[:, 0].max():.2f}]"
    )
    print(
        f"  Y range: [{waypoint_positions[:, 1].min():.2f}, {waypoint_positions[:, 1].max():.2f}]"
    )
    print(
        f"  Z range: [{waypoint_positions[:, 2].min():.2f}, {waypoint_positions[:, 2].max():.2f}]"
    )
    print(f"\nPerformance statistics:")
    print(f"  Mean reward: {np.mean(rewards):.3f}")
    print(f"  Std reward: {np.std(rewards):.3f}")
    print(f"  Best episode: {np.max(rewards):.3f}")
    print(f"  Worst episode: {np.min(rewards):.3f}")
    print()


def example_7_no_randomization():
    """Example 7: Disable randomization for testing"""
    print("=" * 60)
    print("Example 7: No Randomization (Deterministic)")
    print("=" * 60)

    env = RandomizedBlimp(
        modelPath="diff.xml",
        render_mode="",
        randomize=False,  # Disable all randomization
    )

    print("Running deterministic environment:")

    # Run same scenario multiple times
    for trial in range(3):
        obs, _ = env.reset()
        print(f"\nTrial {trial + 1}:")
        print(f"  Waypoint: {env.waypoint}")

        # Run 50 steps with fixed actions
        total_reward = 0
        for step in range(50):
            action = [0.5, 0.5, 0, 0]
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"  Total reward: {total_reward:.3f}")

    print("\nNote: All trials should be identical!")


def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_basic_randomization,
        example_2_custom_ranges,
        example_3_weather_presets,
        example_4_dynamic_waypoints,
        example_5_training_curriculum,
        example_6_statistics_tracking,
        example_7_no_randomization,
    ]

    for example in examples:
        try:
            example()
            print("\n")
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()


if __name__ == "__main__":
    print("Environment Randomization Examples")
    print("=" * 60)
    print()

    # You can run individual examples or all of them
    # Uncomment the one you want to run:

    # Run single example:
    example_1_basic_randomization()

    # Or run all examples:
    # run_all_examples()
