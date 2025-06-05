import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import time
import json
from threading import Thread, Lock
from datetime import datetime
from PIL import Image, ImageTk

class RealSenseRecorderApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("RealSense Recorder v3") # Updated title

        # --- Configuration ---
        self.color_width = 640
        self.color_height = 480
        self.depth_width = 640
        self.depth_height = 480
        self.fps = 30

        # --- State Variables ---
        self.pipeline = None
        self.align = None
        self.is_recording = False
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        self.recording_name_var = tk.StringVar(value="recording") 
        self.current_session_path = "" 
        self.rgb_path = ""
        self.depth_path = ""
        self.jpg_path = "" # Path for JPG images
        self.frame_count = 0
        self.video_thread = None
        self.stop_event = False 
        self.img_lock = Lock() 
        self.current_color_image_display = None
        self.associations_file = None

        # --- UI Elements ---
        # Output Directory
        tk.Label(root_window, text="Output Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dir_entry = tk.Entry(root_window, textvariable=self.output_dir_var, width=50)
        self.dir_entry.grid(row=0, column=1, padx=5, pady=5)
        self.browse_button = tk.Button(root_window, text="Browse", command=self.browse_directory)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # Recording Name
        tk.Label(root_window, text="Recording Name:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.recording_name_entry = tk.Entry(root_window, textvariable=self.recording_name_var, width=50)
        self.recording_name_entry.grid(row=1, column=1, padx=5, pady=5)

        # Video Preview
        self.video_label = tk.Label(root_window) 
        self.video_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5)
        blank_image = np.zeros((self.color_height, self.color_width, 3), dtype=np.uint8)
        self.update_video_preview_tk(blank_image) 

        # Controls
        self.start_stop_button = tk.Button(root_window, text="Start Recording", command=self.toggle_recording, width=20)
        self.start_stop_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

        self.status_label = tk.Label(root_window, text="Status: Not Connected")
        self.status_label.grid(row=3, column=2, padx=5, pady=5, sticky="e")

        # --- Initialize RealSense ---
        self.init_realsense()
        self.start_video_thread()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_video_preview_tk(self, frame_bgr):
        """Updates the Tkinter video preview label with a new BGR frame."""
        try:
            if frame_bgr is None: # Handle cases where frame might be None
                # Optionally display a placeholder or clear the label
                blank_image_arr = np.zeros((self.color_height, self.color_width, 3), dtype=np.uint8)
                img_pil = Image.fromarray(blank_image_arr)
            else:
                img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk 
            self.video_label.configure(image=imgtk)
        except Exception as e:
            print(f"Error updating Tkinter video preview: {e}")


    def init_realsense(self):
        """Initializes the RealSense camera pipeline and streams."""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            ctx = rs.context()
            devices = ctx.query_devices()
            if not devices:
                self.status_label.config(text="Status: No RealSense device connected!")
                messagebox.showerror("Error", "No RealSense device connected. Please connect a camera and restart.")
                self.root.after(100, self.root.quit) 
                return

            # Configure and enable depth and color streams
            config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.rgb8, self.fps) # Using RGB8
            
            # Start the pipeline
            profile = self.pipeline.start(config)
            # Create an align object to align depth frames to color frames
            self.align = rs.align(rs.stream.color)
            self.status_label.config(text="Status: Camera Connected")
            print("RealSense camera initialized.")
        except Exception as e:
            self.status_label.config(text=f"Status: Error: {e}")
            messagebox.showerror("RealSense Error", f"Failed to initialize RealSense camera: {e}")
            self.pipeline = None # Ensure pipeline is None if initialization fails

    def video_processing_loop(self):
        """Handles frame grabbing, processing, display, and saving during recording."""
        while not self.stop_event and self.pipeline:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000) 
                if not frames:
                    print("Warning: No frames received from pipeline.")
                    time.sleep(0.1) 
                    continue

                # Align depth frame to color frame
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue # Skip if either frame is missing

                # Convert frames to NumPy arrays
                color_image_rgb = np.asanyarray(color_frame.get_data()) # Native RGB format from sensor
                depth_image_raw = np.asanyarray(depth_frame.get_data()) # Native 16-bit depth

                # Convert RGB to BGR for OpenCV display and saving
                color_image_bgr_display = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
                
                # Thread-safe update of the image to be displayed in the UI
                with self.img_lock:
                    self.current_color_image_display = color_image_bgr_display.copy()

                # If recording is active and session path is set
                if self.is_recording and self.current_session_path: 
                    # Get timestamp from the color frame (in milliseconds)
                    timestamp_ms = color_frame.get_timestamp() 
                    timestamp_sec = timestamp_ms / 1000.0 # Convert to seconds

                    # Create dot-less timestamp string for filenames
                    # Example: 1678886400.123456 -> "1678886400123456"
                    filename_base = f"{timestamp_sec:.6f}".replace('.', '')

                    # Define relative paths for the associations file
                    relative_rgb_filename = os.path.join("rgb", f"{filename_base}.png")
                    relative_depth_filename = os.path.join("depth", f"{filename_base}.png")
                    # (JPG path is not typically in TUM format associations, but we save the file)

                    # Define full paths for saving image files
                    full_rgb_png_filename = os.path.join(self.rgb_path, f"{filename_base}.png")
                    full_depth_filename = os.path.join(self.depth_path, f"{filename_base}.png")
                    full_rgb_jpg_filename = os.path.join(self.jpg_path, f"{filename_base}.jpg") # JPG filename
                    
                    # Save RGB image as PNG
                    cv2.imwrite(full_rgb_png_filename, color_image_bgr_display)
                    # Save Depth image as 16-bit PNG
                    cv2.imwrite(full_depth_filename, depth_image_raw.astype(np.uint16))
                    # Save RGB image as JPG
                    cv2.imwrite(full_rgb_jpg_filename, color_image_bgr_display)


                    # Write to associations file (TUM format)
                    if self.associations_file:
                        depth_ts_sec = depth_frame.get_timestamp() / 1000.0 # Get depth timestamp
                        # Write: original_timestamp_sec path_to_rgb original_depth_timestamp_sec path_to_depth
                        self.associations_file.write(f"{timestamp_sec:.6f} {relative_rgb_filename} {depth_ts_sec:.6f} {relative_depth_filename}\n")
                    
                    self.frame_count += 1
                    # Update status label periodically (e.g., once per second)
                    if self.frame_count % self.fps == 0 : 
                         self.status_label.config(text=f"Status: Recording... Frames: {self.frame_count}")

            except RuntimeError as e:
                if "Frame didn't arrive within" in str(e):
                    print("Warning: Frame didn't arrive within timeout in video loop.")
                    continue
                else: # Other runtime errors
                    print(f"Runtime error in video loop: {e}")
                    self.status_label.config(text="Status: RealSense Error!")
                    break 
            except Exception as e: # Catch any other exceptions
                print(f"Error in video loop: {e}")
                self.status_label.config(text="Status: Error!")
                break 
        
        if self.pipeline: 
            print("Video processing loop stopped.")


    def update_gui_preview(self):
        """Periodically updates the GUI video preview with the latest frame."""
        if not self.stop_event: # Continue updating if not stopped
            with self.img_lock: # Thread-safe access to the shared image
                if self.current_color_image_display is not None:
                    self.update_video_preview_tk(self.current_color_image_display)
            # Schedule the next update (matches configured FPS)
            self.root.after(int(1000/self.fps), self.update_gui_preview) 

    def start_video_thread(self):
        """Starts the background thread for video processing and GUI updates."""
        if not self.pipeline:
            messagebox.showwarning("Camera Issue", "Cannot start video: RealSense camera not initialized.")
            return
        self.stop_event = False # Reset stop event flag
        # Create and start the daemon thread for video processing
        self.video_thread = Thread(target=self.video_processing_loop, daemon=True)
        self.video_thread.start()
        # Start the GUI preview update loop
        self.update_gui_preview()
        print("Video processing thread started.")

    def browse_directory(self):
        """Opens a dialog to choose the output directory."""
        dir_name = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if dir_name: # If a directory was selected
            self.output_dir_var.set(dir_name)

    def save_full_intrinsics_json(self, output_path, color_intr, depth_intr, d_scale):
        """Saves detailed camera intrinsics (color, depth, depth_scale) to a JSON file."""
        intrinsics_data = {
            "color_camera_intrinsics": {
                "width": color_intr.width, "height": color_intr.height,
                "fx": color_intr.fx, "fy": color_intr.fy,
                "ppx": color_intr.ppx, "ppy": color_intr.ppy,
                "model": str(color_intr.model), "coeffs": list(color_intr.coeffs)
            },
            "depth_camera_intrinsics": {
                "width": depth_intr.width, "height": depth_intr.height,
                "fx": depth_intr.fx, "fy": depth_intr.fy,
                "ppx": depth_intr.ppx, "ppy": depth_intr.ppy,
                "model": str(depth_intr.model), "coeffs": list(depth_intr.coeffs)
            },
            "depth_scale": d_scale
        }
        intrinsics_file_path = os.path.join(output_path, "intrinsics.json")
        try:
            with open(intrinsics_file_path, 'w') as f:
                json.dump(intrinsics_data, f, indent=4)
            print(f"Full intrinsics saved to {intrinsics_file_path}")
        except Exception as e:
            print(f"Error saving intrinsics.json: {e}")
            messagebox.showerror("File Error", f"Could not save intrinsics.json: {e}")

    def save_cam_k_txt(self, output_path, color_intr):
        """Saves the 3x3 color camera intrinsic matrix K to cam_K.txt."""
        # K matrix format: [fx, 0, ppx; 0, fy, ppy; 0, 0, 1]
        cam_k_matrix_flat = [
            color_intr.fx, 0.0, color_intr.ppx,
            0.0, color_intr.fy, color_intr.ppy,
            0.0, 0.0, 1.0
        ]
        # Format each number to scientific notation with high precision
        cam_k_str = " ".join(["{:.17e}".format(val) for val in cam_k_matrix_flat])
        
        cam_k_file_path = os.path.join(output_path, "cam_K.txt")
        try:
            with open(cam_k_file_path, 'w') as f:
                f.write(cam_k_str + "\n")
            print(f"cam_K.txt saved to {cam_k_file_path}")
        except Exception as e:
            print(f"Error saving cam_K.txt: {e}")
            messagebox.showerror("File Error", f"Could not save cam_K.txt: {e}")

    def perform_countdown(self, count):
        """Performs a UI countdown before starting the actual recording."""
        if count > 0:
            self.status_label.config(text=f"Recording starts in {count}...")
            # Schedule the next countdown step after 1 second
            self.root.after(1000, lambda: self.perform_countdown(count - 1))
        else: # Countdown finished
            self.is_recording = True # Set recording flag to true
            self.frame_count = 0 # Reset frame counter
            self.start_stop_button.config(text="Stop Recording", state=tk.NORMAL) # Update button
            self.status_label.config(text="Status: Recording...")
            print(f"Recording started. Saving to: {self.current_session_path}")


    def toggle_recording(self):
        """Handles the Start/Stop recording button action."""
        if not self.pipeline:
            messagebox.showerror("Error", "RealSense camera not initialized. Cannot record.")
            return

        if self.is_recording: # --- If currently recording, stop it ---
            self.is_recording = False # Set recording flag to false
            self.start_stop_button.config(text="Start Recording") # Update button text
            self.status_label.config(text=f"Status: Idle. Total Frames: {self.frame_count}")
            if self.associations_file: # Close associations file if open
                self.associations_file.close()
                self.associations_file = None
            print(f"Recording stopped. Total frames saved: {self.frame_count}")
            # Re-enable UI input fields
            self.dir_entry.config(state=tk.NORMAL)
            self.browse_button.config(state=tk.NORMAL)
            self.recording_name_entry.config(state=tk.NORMAL)
            self.current_session_path = "" # Reset current session path
        
        else: # --- If not recording, start it ---
            base_output_dir = self.output_dir_var.get()
            recording_name = self.recording_name_var.get().strip()
            if not recording_name: # Use default if name is empty
                recording_name = "recording" 
            
            # Create a unique session folder name with date and time
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_folder_name = f"{recording_name}_{datetime_str}"
            self.current_session_path = os.path.join(base_output_dir, session_folder_name)

            try:
                # Create the main session folder and subfolders
                os.makedirs(self.current_session_path, exist_ok=True)
                self.rgb_path = os.path.join(self.current_session_path, "rgb") # For PNGs
                self.depth_path = os.path.join(self.current_session_path, "depth")
                self.jpg_path = os.path.join(self.current_session_path, "img_jpg") # For JPGs
                os.makedirs(self.rgb_path, exist_ok=True)
                os.makedirs(self.depth_path, exist_ok=True)
                os.makedirs(self.jpg_path, exist_ok=True) # Create img_jpg folder
            except OSError as e:
                messagebox.showerror("Error", f"Cannot create directory: {self.current_session_path}\n{e}")
                return

            # Fetch camera intrinsics for saving
            try:
                profile = self.pipeline.get_active_profile()
                color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
                depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
                color_intr = color_profile.get_intrinsics()
                depth_intr = depth_profile.get_intrinsics()
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            except Exception as e:
                messagebox.showerror("Intrinsics Error", f"Could not retrieve camera intrinsics: {e}")
                return

            # Save intrinsic files
            self.save_full_intrinsics_json(self.current_session_path, color_intr, depth_intr, depth_scale)
            self.save_cam_k_txt(self.current_session_path, color_intr)

            # Open associations file for writing
            try:
                assoc_file_path = os.path.join(self.current_session_path, "associations.txt")
                self.associations_file = open(assoc_file_path, 'w')
                # Write header to associations file
                self.associations_file.write("# association file: timestamp_rgb path_rgb timestamp_depth path_depth\n")
            except Exception as e:
                messagebox.showerror("File Error", f"Could not open associations.txt: {e}")
                self.associations_file = None # Ensure it's None if opening failed
                return

            # Disable UI input elements and start countdown
            self.start_stop_button.config(state=tk.DISABLED) 
            self.dir_entry.config(state=tk.DISABLED)
            self.browse_button.config(state=tk.DISABLED)
            self.recording_name_entry.config(state=tk.DISABLED)
            
            self.perform_countdown(3) # Start 3-second countdown

    def on_closing(self):
        """Handles application closing event."""
        print("Closing application...")
        self.stop_event = True # Signal video thread to stop
        if self.video_thread and self.video_thread.is_alive():
            print("Waiting for video thread to join...")
            self.video_thread.join(timeout=2.5) 
            if self.video_thread.is_alive(): # Check if thread actually stopped
                print("Warning: Video thread did not terminate cleanly.")
        
        if self.pipeline: # Stop RealSense pipeline if active
            try:
                print("Stopping RealSense pipeline...")
                self.pipeline.stop()
                print("RealSense pipeline stopped.")
            except Exception as e:
                print(f"Error stopping RealSense pipeline: {e}")

        if self.is_recording and self.associations_file: # Ensure associations file is closed
            self.associations_file.close()

        self.root.destroy() # Close Tkinter window
        print("Application closed.")

if __name__ == '__main__':
    root = tk.Tk() # Create the main Tkinter window
    app = RealSenseRecorderApp(root) # Instantiate the application
    root.mainloop() # Start the Tkinter event loop


# 1) Run joint tracking and reconstruction. 
