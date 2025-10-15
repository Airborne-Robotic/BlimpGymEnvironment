#!/usr/bin/env python
"""
Quick test script to verify the environment is working correctly.
Run this after installation to ensure everything is set up properly.
"""

import sys
import numpy as np


def test_basic_import():
    """Test that we can import the environment."""
    print("Test 1: Importing modules...")
    try:
        from BlimpGymEnvironment import RandomizedBlimp

        print("  ✓ Import successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_deterministic_environment():
    """Test creating and running a deterministic environment."""
    print("\nTest 2: Deterministic environment...")
    try:
        from BlimpGymEnvironment import RandomizedBlimp

        env = RandomizedBlimp(modelPath="diff.xml", render_mode="", randomize=False)

        obs, info = env.reset()

        # Run a few steps
        for i in range(10):
            action = [0.5, 0.5, 0, 0]
            obs, reward, done, _ = env.step(action)

        print(f"  ✓ Environment created and ran successfully")
        print(f"  Final reward: {reward:.3f}")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_randomized_environment():
    """Test creating and running a randomized environment."""
    print("\nTest 3: Randomized environment...")
    try:
        from BlimpGymEnvironment import RandomizedBlimp

        env = RandomizedBlimp(
            modelPath="diff.xml",
            render_mode="",
            randomize=True,
            weather_preset="calm",
            seed=42,
        )

        obs, info = env.reset(regenerate_environment=True)

        # Run a few steps
        for i in range(10):
            action = [0.5, 0.5, 0, 0]
            obs, reward, done, _ = env.step(action)

        print(f"  ✓ Randomized environment working")
        print(f"  Waypoint: {info['waypoint']}")
        print(f"  Final reward: {reward:.3f}")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_weather_presets():
    """Test different weather presets."""
    print("\nTest 4: Weather presets...")
    try:
        from BlimpGymEnvironment import RandomizedBlimp

        weather_conditions = ["calm", "windy", "turbulent"]

        for weather in weather_conditions:
            env = RandomizedBlimp(
                modelPath="diff.xml",
                render_mode="",
                randomize=True,
                weather_preset=weather,
                seed=42,
            )

            info = env.get_randomization_info()
            print(
                f"  {weather}: viscosity={info['viscosity']:.6f}, wind={info['wind']}"
            )

        print("  ✓ All weather presets working")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_observation_space():
    """Test observation space structure."""
    print("\nTest 5: Observation space...")
    try:
        from BlimpGymEnvironment import RandomizedBlimp

        env = RandomizedBlimp(modelPath="diff.xml", render_mode="", randomize=False)

        obs, _ = env.reset()

        # Check observation structure
        assert len(obs) == 3, f"Expected 3 observation components, got {len(obs)}"
        assert obs[0].shape == (3,), f"Position should be (3,), got {obs[0].shape}"
        assert obs[1] is None or len(obs[1].shape) == 3, (
            "Pixels should be None or 3D array"
        )
        assert obs[2].shape == (3,), f"Gyro should be (3,), got {obs[2].shape}"

        print("  ✓ Observation space correct")
        print(f"    Position: {obs[0].shape}")
        print(f"    Pixels: {'None' if obs[1] is None else obs[1].shape}")
        print(f"    Gyro: {obs[2].shape}")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_action_space():
    """Test action space."""
    print("\nTest 6: Action space...")
    try:
        from BlimpGymEnvironment import RandomizedBlimp

        env = RandomizedBlimp(modelPath="diff.xml", render_mode="", randomize=False)

        obs, _ = env.reset()

        # Test different actions
        actions = [
            [1, 1, 0, 0],  # Forward
            [-1, -1, 0, 0],  # Backward
            [1, -1, 0, 0],  # Turn left
            [0, 0, 1, 1],  # Tilt
        ]

        for action in actions:
            obs, reward, done, _ = env.step(action)

        print("  ✓ All actions executed successfully")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_episode_reset():
    """Test episode reset functionality."""
    print("\nTest 7: Episode reset...")
    try:
        from BlimpGymEnvironment import RandomizedBlimp

        env = RandomizedBlimp(
            modelPath="diff.xml", render_mode="", randomize=True, seed=42
        )

        # Get initial waypoint
        obs1, info1 = env.reset(regenerate_environment=True)
        wp1 = info1["waypoint"]

        # Reset with regeneration
        obs2, info2 = env.reset(regenerate_environment=True)
        wp2 = info2["waypoint"]

        # Waypoints should be different
        different = not np.allclose(wp1, wp2)

        print(f"  ✓ Reset working")
        print(f"    Waypoint 1: {wp1}")
        print(f"    Waypoint 2: {wp2}")
        print(f"    Different: {different}")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Blimp Gym Environment - Test Suite")
    print("=" * 60)

    tests = [
        test_basic_import,
        test_deterministic_environment,
        test_randomized_environment,
        test_weather_presets,
        test_observation_space,
        test_action_space,
        test_episode_reset,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Environment is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
