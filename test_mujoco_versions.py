#!/usr/bin/env python3
import subprocess
import sys
import time

PYTHON_PATH = "/home/clemens/miniforge3/envs/serl/bin/python"
TEST_SCRIPT = "/home/clemens/serl/serl/franka_sim/franka_sim/envs/orca_rotate_gym_env.py"

def run_test(version):
    print(f"\n=== Testing Mujoco version {version} ===")
    
    # Install specific version
    install_cmd = f"pip install mujoco=={version}"
    print(f"Running: {install_cmd}")
    try:
        subprocess.run(install_cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to install Mujoco {version}")
        return False
    
    # Run test script
    print(f"\nTesting with {TEST_SCRIPT}")
    try:
        subprocess.run([PYTHON_PATH, TEST_SCRIPT], timeout=10)
        print(f"Success with Mujoco {version}")
        return True
    except subprocess.TimeoutExpired:
        print(f"Test timed out with Mujoco {version}")
        return False
    except subprocess.CalledProcessError:
        print(f"Test failed with Mujoco {version}")
        return False

# Only available versions according to pip
versions = [
    "2.1.2", "2.1.3", "2.1.4", "2.1.5", 
    "2.2.0", "2.2.1", "2.2.2", 
    "2.3.0", "2.3.1", "2.3.2", "2.3.3", "2.3.5", "2.3.6", "2.3.7",
    "3.0.0", "3.0.1", 
    "3.1.0", "3.1.1", "3.1.2", "3.1.3", "3.1.4", "3.1.5", "3.1.6"
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