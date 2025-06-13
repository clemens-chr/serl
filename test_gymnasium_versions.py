#!/usr/bin/env python3
import subprocess
import sys
import time

PYTHON_PATH = "/home/ccc/miniforge3/envs/serl/bin/python"
TEST_SCRIPT = "/home/ccc/orca_ws/src/serl/franka_sim/franka_sim/envs/orca_rotate_gym_env.py"

def run_test(version):
    print(f"\n=== Testing Gymnasium version {version} ===")
    
    # Install specific version
    install_cmd = f"pip install gymnasium=={version}"
    print(f"Running: {install_cmd}")
    try:
        subprocess.run(install_cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to install Gymnasium {version}")
        return False
    
    # Run test script
    print(f"\nTesting with {TEST_SCRIPT}")
    try:
        subprocess.run([PYTHON_PATH, TEST_SCRIPT], timeout=10)
        print(f"Success with Gymnasium {version}")
        return True
    except subprocess.TimeoutExpired:
        print(f"Test timed out with Gymnasium {version}")
        return False
    except subprocess.CalledProcessError:
        print(f"Test failed with Gymnasium {version}")
        return False

# Available Gymnasium versions
versions = [
    "0.26.1", "0.26.2", "0.26.3",
    "0.27.0", "0.27.1",
    "0.28.0", "0.28.1",
    "0.29.0", "0.29.1",
    "1.0.0", "1.1.0", "1.1.1"
]

working_versions = []
failing_versions = []

for version in versions:
    success = run_test(version)
    if success:
        working_versions.append(version)
    else:
        failing_versions.append(version)
    time.sleep(1)  # Small delay between tests

print("\n=== Summary ===")
print("\nWorking versions:", working_versions)
print("\nFailing versions:", failing_versions) 