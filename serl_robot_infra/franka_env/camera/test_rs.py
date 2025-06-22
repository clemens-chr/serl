#!/usr/bin/env python3
"""
Test script for RealSense cameras
Checks which cameras are active and tests their connections
"""

import time
import numpy as np
from rs_capture import RSCapture
from video_capture import VideoCapture
from termcolor import colored


def test_camera_connection(serial_number, name="test_camera", dim=(640, 480), fps=15, depth=False):
    """
    Test connection to a specific RealSense camera
    
    Args:
        serial_number (str): Camera serial number
        name (str): Camera name for identification
        dim (tuple): Image dimensions (width, height)
        fps (int): Frames per second
        depth (bool): Whether to enable depth stream
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    print(colored(f"\n=== Testing Camera: {name} (Serial: {serial_number}) ===", "cyan"))
    
    try:
        # Initialize camera
        print(colored("Initializing camera...", "yellow"))
        camera = RSCapture(name, serial_number, dim=dim, fps=fps, depth=depth)
        
        # Test basic read
        print(colored("Testing basic frame capture...", "yellow"))
        ret, frame = camera.read()
        
        if ret and frame is not None:
            print(colored(f"✓ Basic capture successful - Frame shape: {frame.shape}", "green"))
            
            # Test multiple frames
            print(colored("Testing multiple frame capture...", "yellow"))
            successful_frames = 0
            total_frames = 10
            
            for i in range(total_frames):
                ret, frame = camera.read()
                if ret and frame is not None:
                    successful_frames += 1
                    print(colored(f"  Frame {i+1}/{total_frames}: ✓", "green"))
                else:
                    print(colored(f"  Frame {i+1}/{total_frames}: ✗", "red"))
                time.sleep(0.1)  # Small delay between frames
            
            success_rate = (successful_frames / total_frames) * 100
            print(colored(f"Frame capture success rate: {success_rate:.1f}% ({successful_frames}/{total_frames})", "green" if success_rate > 80 else "yellow"))
            
            # Test with VideoCapture wrapper
            print(colored("Testing with VideoCapture wrapper...", "yellow"))
            try:
                video_cap = VideoCapture(camera, name)
                time.sleep(0.5)  # Give time for thread to start
                
                # Test a few frames with wrapper
                for i in range(5):
                    try:
                        frame = video_cap.read()
                        print(colored(f"  Wrapper frame {i+1}/5: ✓", "green"))
                    except Exception as e:
                        print(colored(f"  Wrapper frame {i+1}/5: ✗ - {str(e)}", "red"))
                    time.sleep(0.1)
                
                # Close video capture wrapper but don't close the camera
                video_cap.enable = False
                video_cap.t.join(timeout=1.0)  # Wait for thread to finish
                print(colored("✓ VideoCapture wrapper test successful", "green"))
                
            except Exception as e:
                print(colored(f"✗ VideoCapture wrapper test failed: {str(e)}", "red"))
            
            # Clean up camera only once
            camera.close()
            print(colored(f"✓ Camera {name} test completed successfully", "green"))
            return True
            
        else:
            print(colored("✗ Basic frame capture failed", "red"))
            camera.close()
            return False
            
    except Exception as e:
        print(colored(f"✗ Camera initialization failed: {str(e)}", "red"))
        return False


def main():
    """Main test function"""
    print(colored("=== RealSense Camera Connection Test ===", "cyan", attrs=["bold"]))
    print(colored("This script will test all available RealSense cameras", "cyan"))
    
    # Create a dummy camera to get device list
    dummy_camera = RSCapture("dummy", "dummy", dummy_mode=True)
    
    # Get available cameras
    try:
        available_serials = dummy_camera.get_device_serial_numbers()
    except Exception as e:
        print(colored(f"Error getting device list: {str(e)}", "red"))
        return
    
    if not available_serials:
        print(colored("No RealSense cameras found!", "red"))
        return
    
    print(colored(f"\nFound {len(available_serials)} RealSense camera(s):", "green"))
    for i, serial in enumerate(available_serials):
        print(colored(f"  {i+1}. Serial: {serial}", "green"))
    
    # Test each camera
    results = {}
    
    for i, serial in enumerate(available_serials):
        camera_name = f"Camera_{i+1}"
        
        # Test RGB only first
        print(colored(f"\n{'='*60}", "cyan"))
        rgb_success = test_camera_connection(serial, camera_name, depth=False)
        
        # Test RGBD if RGB was successful
        rgbd_success = False
        if rgb_success:
            print(colored(f"\n{'='*60}", "cyan"))
            rgbd_success = test_camera_connection(serial, f"{camera_name}_RGBD", depth=True)
        
        results[serial] = {
            'name': camera_name,
            'rgb_success': rgb_success,
            'rgbd_success': rgbd_success
        }
    
    # Print summary
    print(colored(f"\n{'='*60}", "cyan"))
    print(colored("=== TEST SUMMARY ===", "cyan", attrs=["bold"]))
    
    all_successful = True
    for serial, result in results.items():
        status = "✓" if result['rgb_success'] else "✗"
        rgbd_status = "✓" if result['rgbd_success'] else "✗"
        
        print(colored(f"Camera: {result['name']} (Serial: {serial})", "cyan"))
        print(colored(f"  RGB Stream: {status}", "green" if result['rgb_success'] else "red"))
        print(colored(f"  RGBD Stream: {rgbd_status}", "green" if result['rgbd_success'] else "red"))
        
        if not result['rgb_success']:
            all_successful = False
    
    if all_successful:
        print(colored("\n🎉 All cameras are working properly!", "green", attrs=["bold"]))
    else:
        print(colored("\n⚠️  Some cameras have issues. Check the details above.", "yellow", attrs=["bold"]))
    
    print(colored(f"\n{'='*60}", "cyan"))


if __name__ == "__main__":
    main() 