#!/usr/bin/env python3
"""
Debug script for RealSense depth stream issues
"""

import numpy as np
import pyrealsense2 as rs
import time
from termcolor import colored


def check_camera_capabilities(serial_number):
    """Check what streams and formats the camera supports"""
    print(colored(f"\n=== Camera Capabilities Check (Serial: {serial_number}) ===", "cyan"))
    
    try:
        # Get device info
        ctx = rs.context()
        devices = ctx.query_devices()
        
        device = None
        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == serial_number:
                device = dev
                break
        
        if not device:
            print(colored(f"✗ Device with serial {serial_number} not found", "red"))
            return False
            
        print(colored(f"✓ Found device: {device.get_info(rs.camera_info.name)}", "green"))
        print(colored(f"  Firmware: {device.get_info(rs.camera_info.firmware_version)}", "green"))
        
        # Check supported streams
        sensors = device.query_sensors()
        print(colored(f"\nSensors found: {len(sensors)}", "yellow"))
        
        for i, sensor in enumerate(sensors):
            print(colored(f"\nSensor {i+1}: {sensor.get_info(rs.camera_info.name)}", "yellow"))
            
            # Get supported stream profiles
            profiles = sensor.get_stream_profiles()
            print(colored(f"  Stream profiles: {len(profiles)}", "yellow"))
            
            # Group by stream type
            stream_types = {}
            for profile in profiles:
                stream_type = profile.stream_type()
                if stream_type not in stream_types:
                    stream_types[stream_type] = []
                stream_types[stream_type].append(profile)
            
            for stream_type, profiles_list in stream_types.items():
                stream_name = str(stream_type).split('.')[-1]
                print(colored(f"  {stream_name}: {len(profiles_list)} profiles", "green"))
                
                # Show some example profiles
                for j, profile in enumerate(profiles_list[:3]):  # Show first 3
                    if hasattr(profile, 'get_format'):
                        format_name = str(profile.get_format()).split('.')[-1]
                        if hasattr(profile, 'get_resolution'):
                            width, height = profile.get_resolution()
                            fps = profile.get_framerate()
                            print(colored(f"    {j+1}. {width}x{height} @ {fps}fps ({format_name})", "white"))
        
        return True
        
    except Exception as e:
        print(colored(f"✗ Error checking capabilities: {str(e)}", "red"))
        return False


def test_depth_stream_only(serial_number, name="test_depth"):
    """Test depth stream in isolation"""
    print(colored(f"\n=== Testing Depth Stream Only (Serial: {serial_number}) ===", "cyan"))
    
    try:
        # Initialize pipeline
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial_number)
        
        # Try different depth configurations
        depth_configs = [
            (640, 480, 30),
            (640, 480, 15),
            (640, 480, 6),
            (848, 480, 30),
            (848, 480, 15),
            (848, 480, 6),
        ]
        
        for width, height, fps in depth_configs:
            print(colored(f"\nTrying depth: {width}x{height} @ {fps}fps", "yellow"))
            
            try:
                # Clear previous config
                cfg.disable_all_streams()
                cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                
                # Start pipeline
                profile = pipe.start(cfg)
                print(colored(f"  ✓ Pipeline started", "green"))
                
                # Try to get frames
                for i in range(5):
                    try:
                        frames = pipe.wait_for_frames(timeout_ms=3000)
                        depth_frame = frames.get_depth_frame()
                        
                        if depth_frame:
                            depth_data = np.asarray(depth_frame.get_data())
                            print(colored(f"  ✓ Frame {i+1}: {depth_data.shape}, min={depth_data.min()}, max={depth_data.max()}", "green"))
                        else:
                            print(colored(f"  ✗ Frame {i+1}: No depth frame", "red"))
                            
                    except Exception as e:
                        print(colored(f"  ✗ Frame {i+1}: {str(e)}", "red"))
                        break
                
                pipe.stop()
                print(colored(f"  ✓ Depth stream {width}x{height} @ {fps}fps works!", "green"))
                return True
                
            except Exception as e:
                print(colored(f"  ✗ Failed: {str(e)}", "red"))
                try:
                    pipe.stop()
                except:
                    pass
                continue
        
        print(colored("✗ No depth configuration worked", "red"))
        return False
        
    except Exception as e:
        print(colored(f"✗ Error in depth test: {str(e)}", "red"))
        return False


def test_simultaneous_streams(serial_number, name="test_simultaneous"):
    """Test RGB and depth streams simultaneously"""
    print(colored(f"\n=== Testing Simultaneous RGB + Depth (Serial: {serial_number}) ===", "cyan"))
    
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial_number)
        
        # Try different configurations
        configs = [
            # (color_width, color_height, color_fps, depth_width, depth_height, depth_fps)
            (640, 480, 30, 640, 480, 30),
            (640, 480, 15, 640, 480, 15),
            (640, 480, 30, 640, 480, 15),
            (640, 480, 15, 640, 480, 30),
            (848, 480, 30, 848, 480, 30),
            (848, 480, 15, 848, 480, 15),
        ]
        
        for c_w, c_h, c_fps, d_w, d_h, d_fps in configs:
            print(colored(f"\nTrying: Color {c_w}x{c_h}@{c_fps}fps + Depth {d_w}x{d_h}@{d_fps}fps", "yellow"))
            
            try:
                cfg.disable_all_streams()
                cfg.enable_stream(rs.stream.color, c_w, c_h, rs.format.bgr8, c_fps)
                cfg.enable_stream(rs.stream.depth, d_w, d_h, rs.format.z16, d_fps)
                
                profile = pipe.start(cfg)
                print(colored(f"  ✓ Pipeline started", "green"))
                
                # Test alignment
                align = rs.align(rs.stream.color)
                
                # Try to get aligned frames
                for i in range(3):
                    try:
                        frames = pipe.wait_for_frames(timeout_ms=5000)
                        aligned_frames = align.process(frames)
                        
                        color_frame = aligned_frames.get_color_frame()
                        depth_frame = aligned_frames.get_depth_frame()
                        
                        if color_frame and depth_frame:
                            color_data = np.asarray(color_frame.get_data())
                            depth_data = np.asarray(depth_frame.get_data())
                            print(colored(f"  ✓ Frame {i+1}: Color {color_data.shape}, Depth {depth_data.shape}", "green"))
                        else:
                            missing = []
                            if not color_frame: missing.append("color")
                            if not depth_frame: missing.append("depth")
                            print(colored(f"  ✗ Frame {i+1}: Missing {', '.join(missing)}", "red"))
                            
                    except Exception as e:
                        print(colored(f"  ✗ Frame {i+1}: {str(e)}", "red"))
                        break
                
                pipe.stop()
                print(colored(f"  ✓ Simultaneous streams work!", "green"))
                return True
                
            except Exception as e:
                print(colored(f"  ✗ Failed: {str(e)}", "red"))
                try:
                    pipe.stop()
                except:
                    pass
                continue
        
        print(colored("✗ No simultaneous configuration worked", "red"))
        return False
        
    except Exception as e:
        print(colored(f"✗ Error in simultaneous test: {str(e)}", "red"))
        return False


def main():
    """Main diagnostic function"""
    print(colored("=== RealSense Depth Stream Diagnostic ===", "cyan", attrs=["bold"]))
    
    # Get available cameras
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        
        if not serials:
            print(colored("No RealSense cameras found!", "red"))
            return
            
        print(colored(f"Found {len(serials)} RealSense camera(s): {serials}", "green"))
        
    except Exception as e:
        print(colored(f"Error getting device list: {str(e)}", "red"))
        return
    
    # Test each camera
    for serial in serials:
        print(colored(f"\n{'='*80}", "cyan"))
        
        # Check capabilities
        check_camera_capabilities(serial)
        
        # Test depth only
        depth_only_works = test_depth_stream_only(serial)
        
        # Test simultaneous streams
        simultaneous_works = test_simultaneous_streams(serial)
        
        # Summary for this camera
        print(colored(f"\n--- Summary for Camera {serial} ---", "cyan"))
        print(colored(f"Depth-only stream: {'✓' if depth_only_works else '✗'}", "green" if depth_only_works else "red"))
        print(colored(f"Simultaneous RGB+Depth: {'✓' if simultaneous_works else '✗'}", "green" if simultaneous_works else "red"))
        
        if not depth_only_works:
            print(colored("  → Depth stream has fundamental issues", "red"))
        elif not simultaneous_works:
            print(colored("  → Depth works alone but not with RGB (resource conflict)", "yellow"))
        else:
            print(colored("  → All tests passed!", "green"))
    
    print(colored(f"\n{'='*80}", "cyan"))
    print(colored("Diagnostic complete!", "cyan", attrs=["bold"]))


if __name__ == "__main__":
    main() 