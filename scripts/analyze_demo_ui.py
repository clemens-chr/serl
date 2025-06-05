import os
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import json
import ast # For safely evaluating string representation of list

# --- Configuration ---
DEFAULT_FPS = 10
MAX_ARRAY_DISPLAY_ELEMENTS = 50 # Limit elements shown for large arrays

# --- Helper Functions ---
def format_value(value):
    """Formats a value for display, handling large arrays and 0-d arrays."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
             # Display 0-d array's scalar content
             return f"ndarray (0-d scalar) | dtype: {value.dtype}\n{value.item()}"
        elif value.size > MAX_ARRAY_DISPLAY_ELEMENTS:
            return f"ndarray | Shape: {value.shape} | dtype: {value.dtype} | First {MAX_ARRAY_DISPLAY_ELEMENTS} flat elements:\n{value.flatten()[:MAX_ARRAY_DISPLAY_ELEMENTS]}..."
        else:
            return f"ndarray | Shape: {value.shape} | dtype: {value.dtype}\n{value}"
    elif isinstance(value, (list, tuple)):
        if len(value) > MAX_ARRAY_DISPLAY_ELEMENTS:
             return f"{type(value).__name__} | Length: {len(value)} | First {MAX_ARRAY_DISPLAY_ELEMENTS} elements:\n{value[:MAX_ARRAY_DISPLAY_ELEMENTS]}..."
        else:
             # Handle empty list/tuple specifically
             if not value:
                 return f"{type(value).__name__} | Length: 0\n[]" if isinstance(value, list) else f"{type(value).__name__} | Length: 0\n()"
             return f"{type(value).__name__} | Length: {len(value)}\n{value}"
    elif isinstance(value, dict):
         # Handle empty dict specifically
         if not value:
             return "dict | Empty\n{}"
         return f"dict | Keys: {list(value.keys())}"
    else:
        # Handle NoneType specifically
        if value is None:
            return "NoneType\nNone"
        return f"{type(value).__name__}\n{value}"

def get_data_from_path(data, path_list):
    """Access nested data using a list of keys/indices."""
    current_data = data
    try:
        for key in path_list:
            current_data = current_data[key]
        return current_data
    except (KeyError, IndexError, TypeError, ValueError): # Added ValueError for safety
        # This can happen if the path becomes invalid during interaction
        print(f"Warning: Could not retrieve data at path {path_list}")
        return None # Path is invalid

# --- Main Application Class ---
class PklExplorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pickle Data Explorer")
        self.root.geometry("1200x800")

        self.data = None
        self.filepath = None
        self.image_sequences = {} # Stores {'seq_name': [img_array, ...]}
        self.other_data = {}     # Stores {'data_name': [value, ...]}
        self.current_frame_index = 0
        self.video_playing = False
        self.video_job = None
        self.num_frames = 0
        self.tk_image = None # Keep a reference to avoid garbage collection

        # --- Setup Paned Window Layout ---
        self.paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Pane: File Info and Tree ---
        self.left_frame = ttk.Frame(self.paned_window, width=400, height=800)
        self.paned_window.add(self.left_frame, weight=1)

        # File Controls
        self.file_frame = ttk.Frame(self.left_frame)
        self.file_frame.pack(pady=5, padx=5, fill=tk.X)
        self.load_button = ttk.Button(self.file_frame, text="Load .pkl File", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(self.file_frame, text="No file loaded.", wraplength=300)
        self.file_label.pack(side=tk.LEFT, padx=5)

        # Structure Tree
        self.tree_label = ttk.Label(self.left_frame, text="Data Structure:")
        self.tree_label.pack(anchor=tk.W, padx=5)

        self.tree_frame = ttk.Frame(self.left_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.tree = ttk.Treeview(self.tree_frame, selectmode='browse')
        # Place tree using grid within its frame to manage scrollbars correctly
        self.tree.grid(row=0, column=0, sticky='nsew')

        # Tree Scrollbars
        tree_vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        tree_hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        tree_vsb.grid(row=0, column=1, sticky='ns')
        tree_hsb.grid(row=1, column=0, sticky='ew')

        self.tree.configure(yscrollcommand=tree_vsb.set, xscrollcommand=tree_hsb.set)

        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)

        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)


        # --- Right Pane: Details and Visualization ---
        self.right_frame = ttk.Frame(self.paned_window, width=800, height=800)
        self.paned_window.add(self.right_frame, weight=3)

        # Details Area
        self.details_frame = ttk.LabelFrame(self.right_frame, text="Selected Item Details")
        self.details_frame.pack(pady=5, padx=5, fill=tk.X)
        self.details_text = tk.Text(self.details_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, font=('Courier New', 9)) # Monospace font
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Visualization Area (Video + Other Data)
        self.viz_frame = ttk.LabelFrame(self.right_frame, text="Trajectory Visualization")
        self.viz_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        # Video Controls
        self.controls_frame = ttk.Frame(self.viz_frame)
        self.controls_frame.pack(pady=5, fill=tk.X) # Fill X
        self.play_button = ttk.Button(self.controls_frame, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.prev_button = ttk.Button(self.controls_frame, text="<< Prev", command=self.prev_frame, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(self.controls_frame, text="Next >>", command=self.next_frame, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.frame_slider = ttk.Scale(self.controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.slider_update, length=300, state=tk.DISABLED)
        self.frame_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True) # Fill X and expand
        self.frame_label = ttk.Label(self.controls_frame, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        # Image Display
        self.image_canvas = tk.Canvas(self.viz_frame, bg='#CCCCCC') # Lighter gray
        self.image_canvas.pack(pady=5, fill=tk.BOTH, expand=True)
        self.image_display_id = None # To hold the canvas item id

        # Other Data Display
        self.other_data_label = ttk.Label(self.viz_frame, text="Other Data (at current frame):")
        self.other_data_label.pack(anchor=tk.W, padx=5, pady=(10,0))
        self.other_data_text = tk.Text(self.viz_frame, height=5, wrap=tk.WORD, state=tk.DISABLED, font=('Courier New', 9)) # Monospace font
        self.other_data_text.pack(fill=tk.X, padx=5, pady=5)

        # --- Status Bar ---
        self.status_bar = ttk.Label(root, text="Ready.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, text):
        self.status_bar.config(text=text)

    def load_file(self):
        """Opens file dialog, loads pickle, processes, and updates GUI."""
        filepath = filedialog.askopenfilename(
            title="Select Pickle File",
            filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*"))
        )
        if not filepath:
            return

        self.filepath = filepath
        self.set_status(f"Loading {os.path.basename(filepath)}...")
        self.root.update_idletasks() # Update GUI to show status

        try:
            with open(self.filepath, 'rb') as f:
                # Ensure compatibility with different pickle protocols if needed
                self.data = pickle.load(f) # Or add encoding='latin1' etc. if required
            self.file_label.config(text=os.path.basename(filepath))
            self.set_status("Loading complete. Processing data structure...")
            self.root.update_idletasks()

            self.reset_visualization()
            self.clear_tree()
            self.clear_details()
            self.populate_tree(self.data) # Build the tree first

            self.set_status("Processing for visualization...")
            self.root.update_idletasks()
            self.process_for_visualization() # Find image sequences and other data

            self.set_status("Data loaded and processed successfully.")

        except FileNotFoundError:
             messagebox.showerror("Error Loading File", f"File not found:\n{self.filepath}")
             self.set_status("Error: File not found.")
             self.reset_for_new_file()
        except pickle.UnpicklingError as e:
            messagebox.showerror("Error Loading File", f"Failed to unpickle file (corrupted or incompatible format):\n{e}")
            self.set_status("Error: Failed to unpickle file.")
            self.reset_for_new_file()
        except Exception as e:
            messagebox.showerror("Error Loading File", f"An unexpected error occurred:\n{e}")
            self.set_status(f"Error: {e}")
            self.reset_for_new_file()

    def reset_for_new_file(self):
        """Clears data and resets GUI state when loading fails."""
        self.data = None
        self.filepath = None
        self.file_label.config(text="No file loaded.")
        self.clear_tree()
        self.clear_details()
        self.reset_visualization()

    def clear_tree(self):
        """Removes all items from the treeview."""
        if self.tree.get_children(): # Check if tree has items before deleting
             for i in self.tree.get_children():
                 self.tree.delete(i)

    def clear_details(self):
        """Clears the details text area."""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete('1.0', tk.END)
        self.details_text.config(state=tk.DISABLED)

    def reset_visualization(self):
        """Resets video/image display and controls."""
        self.stop_video()
        self.image_sequences = {}
        self.other_data = {}
        self.current_frame_index = 0
        self.num_frames = 0
        self.frame_slider.set(0)
        self.frame_slider.config(to=0, state=tk.DISABLED)
        self.play_button.config(text="Play", state=tk.DISABLED)
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.frame_label.config(text="Frame: 0 / 0")
        if self.image_display_id:
            self.image_canvas.delete(self.image_display_id)
            self.image_display_id = None
        self.image_canvas.config(bg='#CCCCCC') # Reset background
        # Clear other data text
        self.other_data_text.config(state=tk.NORMAL)
        self.other_data_text.delete('1.0', tk.END)
        self.other_data_text.config(state=tk.DISABLED)
        # Clear canvas placeholder text (find and delete text items)
        text_items = self.image_canvas.find_withtag("placeholder_text")
        for item_id in text_items:
            self.image_canvas.delete(item_id)


    def _add_node(self, parent_id, key, value, path_list):
        """
        Recursive helper to add nodes to the tree.
        Handles 0-d numpy arrays correctly.
        Stores path as a string representation for robustness.
        """
        node_text = f"{key}"
        # Store the path as a string representation to avoid issues with Tkinter value types
        path_str = str(path_list)
        node_id = self.tree.insert(parent_id, 'end', text=node_text, open=False, values=[path_str]) # Store path_str in a list

        if isinstance(value, dict):
            # Add specific handling for empty dicts
            if not value:
                self.tree.item(node_id, text=f"{key} (dict, empty)", tags=('dict', 'empty'))
            else:
                self.tree.item(node_id, tags=('dict',))
                for sub_key, sub_value in value.items():
                    self._add_node(node_id, sub_key, sub_value, path_list + [sub_key])

        elif isinstance(value, np.ndarray):
            # *** FIX: Handle 0-d arrays explicitly ***
            if value.ndim == 0:
                scalar_value = value.item() # Extract the scalar value
                self.tree.item(node_id, text=f"{key} (ndarray scalar): {type(scalar_value).__name__}", tags=('leaf', 'ndarray_scalar'))
                # No children to add for a scalar
            else:
                # Handle arrays with ndim > 0
                tag = 'ndarray'
                shape_info = f"Shape: {value.shape}"
                dtype_info = f" | dtype: {value.dtype}"
                self.tree.item(node_id, text=f"{key} ({tag} {shape_info}{dtype_info})", tags=(tag,))

                # Heuristic: Avoid expanding large multi-dimensional arrays in the tree
                should_expand_elements = value.ndim == 1 and value.size < 100 # Only expand 1D arrays under a size limit

                if should_expand_elements:
                    for i, item in enumerate(value):
                        if i >= 50 and len(value) > 55:
                             self.tree.insert(node_id, 'end', text="...", open=False, values=str(path_list + ['...']), tags=('ellipsis',))
                             break
                        self._add_node(node_id, f"[{i}]", item, path_list + [i])
                # else: (Don't add children nodes for large/multi-dim arrays)

        elif isinstance(value, (list, tuple)):
            # Add specific handling for empty lists/tuples
            if not value:
                 self.tree.item(node_id, text=f"{key} ({type(value).__name__}, empty)", tags=('list' if isinstance(value, list) else 'tuple', 'empty'))
            else:
                tag = 'list' if isinstance(value, list) else 'tuple'
                len_info = f"Len: {len(value)}"
                self.tree.item(node_id, text=f"{key} ({tag} {len_info})", tags=(tag,))

                # Expand reasonably sized lists/tuples
                should_expand_elements = len(value) < 100

                if should_expand_elements:
                    for i, item in enumerate(value):
                        if i >= 50 and len(value) > 55:
                            self.tree.insert(node_id, 'end', text="...", open=False, values=str(path_list + ['...']), tags=('ellipsis',))
                            break
                        self._add_node(node_id, f"[{i}]", item, path_list + [i])
                # else: (Don't add children nodes for very long lists/tuples)

        else:
            # Leaf node (simple type like int, float, str, bool, None)
            type_name = type(value).__name__
            # Add representation for None
            display_text = f"{key}: {type_name}" if value is not None else f"{key}: NoneType"
            self.tree.item(node_id, text=display_text, tags=('leaf',))


    def populate_tree(self, data):
        """Fills the treeview with the structure of the loaded data."""
        self.clear_tree()
        if data is None:
            self.tree.insert('', 'end', text="Data is None", open=False, values="[]", tags=('empty',))
            return

        # Use a dummy root node to handle different top-level types consistently
        root_node_id = self.tree.insert('', 'end', text="Loaded Data", open=True, values="[]") # Represents the root access

        if isinstance(data, list):
            self.tree.item(root_node_id, text=f"Data Root (List, Len: {len(data)})")
            self.tree.item(root_node_id, tags=('list',))
            if not data: # Handle empty top-level list
                 self.tree.item(root_node_id, text="Data Root (List, empty)")
            else:
                for i, entry in enumerate(data):
                    # Pass 'i' as the key for list elements, starting path is [i]
                    self._add_node(root_node_id, f"[{i}] (Entry/Timestep)", entry, [i])
        elif isinstance(data, dict):
            self.tree.item(root_node_id, text=f"Data Root (dict, Keys: {len(data)})")
            self.tree.item(root_node_id, tags=('dict',))
            if not data: # Handle empty top-level dict
                 self.tree.item(root_node_id, text="Data Root (dict, empty)")
            else:
                 for key, value in data.items():
                    # Starting path is [key]
                    self._add_node(root_node_id, key, value, [key])
        else:
            # Handle cases where the root is a simple type or ndarray
            self.tree.item(root_node_id, text=f"Data Root ({type(data).__name__})")
            # Treat the root itself as the only node to add details for below the dummy root
            self._add_node(root_node_id, "Value", data, ["Value"]) # Use placeholder key


    def on_tree_select(self, event):
        """Handles selection changes in the treeview. Uses ast.literal_eval for paths."""
        selected_iid = self.tree.focus()
        if not selected_iid:
            return

        # Retrieve the path string stored in the first column of 'values'
        values = self.tree.item(selected_iid, 'values')
        path_list = []
        if values:
            path_str = values[0] # Get the first (and only) element which is the path string
            try:
                # Safely evaluate the string representation of the list
                path_list = ast.literal_eval(path_str)
                if not isinstance(path_list, list): # Ensure it evaluated to a list
                     path_list = []
                     print(f"Warning: Evaluated path is not a list: {path_str}")
            except (ValueError, SyntaxError, TypeError) as e:
                 path_list = [] # Fallback if eval fails
                 print(f"Warning: Could not evaluate path string: {path_str}, Error: {e}")
        else:
             # Should not happen if nodes are added correctly, but handle defensively
             path_list = []
             print(f"Warning: No path data found for selected tree item {selected_iid}")


        # --- Retrieve Data ---
        # Special case: If path_list is empty, it refers to the root data itself
        if not path_list:
             selected_data = self.data
             # If we added a dummy root node with a specific value path like ['Value']
             # check if this is the actual data root or the dummy node
             if self.tree.item(selected_iid, 'text') == "Loaded Data":
                  selected_data = self.data # Show overall data for the top node
             else:
                  # Try to get data based on the structure if root wasn't list/dict
                  # This part might need adjustment based on how _add_node handles non-list/dict roots
                  selected_data = get_data_from_path(self.data, path_list)

        else:
             selected_data = get_data_from_path(self.data, path_list)

        # --- Update Details View ---
        self.clear_details()
        self.details_text.config(state=tk.NORMAL)
        if selected_data is not None or not path_list: # Allow showing info for root even if None initially
            data_info = f"Path: {path_list}\n"
            # Handle root data display slightly differently
            current_data_to_format = self.data if not path_list else selected_data
            if current_data_to_format is not None:
                 data_info += f"Type: {type(current_data_to_format).__name__}\n\nValue:\n"
                 data_info += format_value(current_data_to_format)
            else:
                 data_info += "Type: NoneType\n\nValue:\nNone" # If data itself is None

            self.details_text.insert('1.0', data_info)
        else:
             # If get_data_from_path returned None for a non-empty path
             self.details_text.insert('1.0', f"Path: {path_list}\n\nCould not retrieve data for this path (might be invalid or data structure changed).")
        self.details_text.config(state=tk.DISABLED)

        # --- Update Visualization Focus ---
        # If a top-level list item (trajectory step) is selected, jump the visualizer
        parent_iid = self.tree.parent(selected_iid)
        is_top_level_entry = False
        frame_index_to_set = -1

        # Check if the selected node's parent is the effective root node AND the original data was a list
        if parent_iid == self.tree.get_children('')[0]: # Parent is the 'Loaded Data' node
             root_tags = self.tree.item(parent_iid, 'tags')
             if isinstance(self.data, list): # Original data must be a list
                 # Extract index from the node text like "[0] (Entry/Timestep)"
                 node_text = self.tree.item(selected_iid, 'text')
                 import re
                 match = re.match(r'\[(\d+)\]', node_text) # Check if text starts with "[index]"
                 if match:
                     try:
                         index = int(match.group(1))
                         if 0 <= index < self.num_frames:
                             is_top_level_entry = True
                             frame_index_to_set = index
                     except (ValueError, IndexError):
                          pass # Ignore if index parsing fails


        if is_top_level_entry and self.num_frames > 0 and frame_index_to_set != -1:
            # Stop playback if user selects a frame manually
            if self.video_playing:
                 self.stop_video()
            self.set_frame(frame_index_to_set)
            self.frame_slider.set(frame_index_to_set) # Update slider position


    def process_for_visualization(self):
        """
        Analyzes the loaded data (assumed to be a list of dicts/steps)
        to find image sequences and corresponding other data for playback.
        """
        self.reset_visualization() # Clear previous viz data

        if not isinstance(self.data, list) or not self.data:
            self.set_status("Visualization requires data to be a non-empty list of steps.")
            # Display placeholder on canvas if no viz possible
            self.image_canvas.create_text(10, 10, anchor=tk.NW, text="Load a list-based dataset for visualization.", tags="placeholder_text")
            return

        self.num_frames = len(self.data)
        if self.num_frames == 0:
             self.image_canvas.create_text(10, 10, anchor=tk.NW, text="Dataset list is empty.", tags="placeholder_text")
             return

        # --- Heuristic to find image sequences and other data ---
        first_entry = self.data[0]
        potential_image_paths = [] # Store lists of keys, e.g., [['obs', 'front'], ['wrist']]
        potential_other_data_paths = []

        def find_paths(current_data, current_path, img_paths, other_paths):
            """Recursive helper to find paths to images and other data."""
            if isinstance(current_data, dict):
                for key, value in current_data.items():
                    new_path = current_path + [key]
                    if isinstance(value, np.ndarray):
                         # Image Check: ndim 3 (HWC) or 4 (BHWC), last dim 1, 3, or 4
                         # Also check shape is reasonable (e.g., not 1x1x3)
                         is_image_like = False
                         if value.ndim in [3, 4] and value.shape[-1] in [1, 3, 4]:
                             if value.ndim == 3 and value.shape[0] > 1 and value.shape[1] > 1: # Min H, W > 1 for HWC
                                 is_image_like = True
                             elif value.ndim == 4 and value.shape[0] == 1 and value.shape[1] > 1 and value.shape[2] > 1: # Min H, W > 1 for BHWC (B=1)
                                 is_image_like = True

                         if is_image_like:
                             img_paths.append(new_path)
                         else: # Array that's not image-like goes to other data
                             other_paths.append(new_path)
                    elif isinstance(value, dict):
                        # Recurse into sub-dictionaries
                        find_paths(value, new_path, img_paths, other_paths)
                    elif not isinstance(value, (list, tuple)): # Add non-iterable leaves as other data
                        other_paths.append(new_path)
                    # We generally ignore lists/tuples within the structure for top-level viz data
            # We don't traverse into lists/tuples from the root by default here

        if isinstance(first_entry, dict):
            find_paths(first_entry, [], potential_image_paths, potential_other_data_paths)
        # If first_entry isn't a dict, we can't easily find named sequences

        # --- Validate and Extract ---
        valid_image_sequences = {}
        valid_other_data = {}

        # Validate image sequences across all frames
        for path_list in potential_image_paths:
            seq_name = "_".join(map(str, path_list)) # Create name like 'observations_front'
            images = []
            valid = True
            expected_shape = None
            expected_dtype = None # Check dtype consistency too

            for i in range(self.num_frames):
                img_data = get_data_from_path(self.data[i], path_list)

                if not isinstance(img_data, np.ndarray):
                    valid = False; break # Must be numpy array

                # Handle potential batch dimension (e.g., shape (1, H, W, C)) -> (H, W, C)
                processed_img = img_data
                if img_data.ndim == 4 and img_data.shape[0] == 1:
                    processed_img = img_data[0]
                elif img_data.ndim != 3 and img_data.ndim != 2: # Expect HWC or HW
                    valid = False; break

                # Check shape consistency (after removing potential batch dim)
                current_shape = processed_img.shape
                if expected_shape is None:
                    expected_shape = current_shape
                    expected_dtype = processed_img.dtype
                elif current_shape != expected_shape:
                    print(f"Shape mismatch in sequence '{seq_name}' at frame {i}: Expected {expected_shape}, got {current_shape}")
                    valid = False; break
                elif processed_img.dtype != expected_dtype:
                    # Allow safe casting between int types or float types, but warn
                    # print(f"Dtype mismatch in sequence '{seq_name}' at frame {i}: Expected {expected_dtype}, got {processed_img.dtype}")
                    pass # Might need stricter check depending on data


                # --- Image Normalization/Conversion ---
                try:
                    # Convert to uint8 for display
                    if processed_img.dtype != np.uint8:
                        if np.issubdtype(processed_img.dtype, np.floating):
                            processed_img = (processed_img * 255).clip(0, 255).astype(np.uint8)
                        elif np.issubdtype(processed_img.dtype, np.integer):
                             # Scale if not already 0-255 range (basic check)
                             min_val, max_val = np.min(processed_img), np.max(processed_img)
                             if max_val > 255 or min_val < 0:
                                 processed_img = ((processed_img.astype(np.float32) - min_val) / max(1e-6, max_val - min_val) * 255).astype(np.uint8)
                             else:
                                 processed_img = processed_img.astype(np.uint8)
                        else: # Boolean or other types - convert T/F to 255/0
                             processed_img = processed_img.astype(np.uint8) * 255


                    # Convert grayscale (HW) to RGB (HWC)
                    if processed_img.ndim == 2:
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                    # Ensure 3 channels if not grayscale
                    elif processed_img.ndim == 3 and processed_img.shape[-1] == 1:
                         processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                    # Assume BGR if 3 channels and convert to RGB (common OpenCV convention)
                    # Comment out if your data is already RGB
                    elif processed_img.ndim == 3 and processed_img.shape[-1] == 3:
                         processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    # Handle 4 channels (RGBA), just take RGB
                    elif processed_img.ndim == 3 and processed_img.shape[-1] == 4:
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGBA2RGB)


                    images.append(processed_img)

                except Exception as e:
                    print(f"Error processing image in sequence '{seq_name}' at frame {i}: {e}")
                    valid = False; break

            if valid and images:
                valid_image_sequences[seq_name] = images


        # Extract other data (check presence across all frames)
        for path_list in potential_other_data_paths:
            data_name = "_".join(map(str, path_list)) # Create name
            data_list = []
            valid = True
            for i in range(self.num_frames):
                 value = get_data_from_path(self.data[i], path_list)
                 if value is None and path_list not in self.data[i]: # Check if path was actually invalid vs value being None
                      print(f"Other data path {path_list} invalid at frame {i}")
                      valid = False; break
                 # We store the value even if it's None, as long as the path was valid
                 data_list.append(value)

            if valid:
                valid_other_data[data_name] = data_list


        self.image_sequences = valid_image_sequences
        self.other_data = valid_other_data

        if not self.image_sequences:
            self.set_status("No suitable image sequences found for visualization.")
            # Add placeholder text to canvas
            self.image_canvas.create_text(self.image_canvas.winfo_width()//2 if self.image_canvas.winfo_width()>1 else 10,
                                         self.image_canvas.winfo_height()//2 if self.image_canvas.winfo_height()>1 else 10,
                                         anchor=tk.CENTER, text="No displayable image sequences found.", tags="placeholder_text")
            return

        # Prepare visualization controls
        self.frame_slider.config(to=self.num_frames - 1, state=tk.NORMAL if self.num_frames > 1 else tk.DISABLED) # Disable slider if only 1 frame
        self.play_button.config(state=tk.NORMAL if self.num_frames > 1 else tk.DISABLED)
        self.prev_button.config(state=tk.NORMAL if self.num_frames > 1 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.num_frames > 1 else tk.DISABLED)

        # Display the first frame
        self.set_frame(0)
        status_msg = f"Ready. Found {len(self.image_sequences)} sequence(s), {len(self.other_data)} data track(s)."
        self.set_status(status_msg)


    def combine_images(self, frame_index):
        """Combines images from different sequences for the current frame vertically."""
        images_to_show = []
        max_width = 0

        seq_keys = sorted(list(self.image_sequences.keys())) # Consistent order

        for seq_name in seq_keys:
             if frame_index < len(self.image_sequences[seq_name]):
                 img = self.image_sequences[seq_name][frame_index]
                 if img is not None and isinstance(img, np.ndarray) and img.ndim == 3: # Ensure valid RGB image
                     images_to_show.append(img)
                     max_width = max(max_width, img.shape[1])
                 #else: print(f"Warning: Invalid image data for {seq_name} at frame {frame_index}")


        if not images_to_show:
            # print(f"No images found for frame {frame_index}")
            return None

        # Pad images to max_width for vertical stacking
        padded_images = []
        for img in images_to_show:
            h, w, c = img.shape
            if w < max_width:
                 pad_width = max_width - w
                 # Pad on the right side
                 padded_img = cv2.copyMakeBorder(img, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=[0,0,0]) # Black padding
                 padded_images.append(padded_img)
            else:
                 padded_images.append(img)

        # Stack vertically
        try:
            combined_image = cv2.vconcat(padded_images)
            return combined_image
        except Exception as e:
             print(f"Error during vconcat for frame {frame_index}: {e}")
             # Fallback: return the first image if stacking fails
             return padded_images[0] if padded_images else None


    def update_display(self):
        """Updates the image canvas and other data text for the current frame."""
        if self.num_frames == 0: # Guard against no data loaded
            return
        if not self.image_sequences and not self.other_data: # Guard if processing failed
             return

        # --- Update Image ---
        # Clear previous placeholder text
        text_items = self.image_canvas.find_withtag("placeholder_text")
        for item_id in text_items:
            self.image_canvas.delete(item_id)

        combined_img_np = self.combine_images(self.current_frame_index)

        if combined_img_np is not None:
            try:
                # Convert numpy array (RGB expected from combine_images) to PIL Image
                pil_image = Image.fromarray(combined_img_np)

                # Resize image to fit canvas while maintaining aspect ratio
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()

                # Wait if canvas not realized yet
                if canvas_width < 2 or canvas_height < 2:
                    self.root.after(50, self.update_display) # Try again shortly
                    return

                img_aspect = pil_image.width / pil_image.height
                canvas_aspect = canvas_width / canvas_height

                if img_aspect > canvas_aspect: # Image wider than canvas -> fit width
                    new_width = canvas_width
                    new_height = int(new_width / img_aspect)
                else: # Image taller than canvas (or same aspect) -> fit height
                    new_height = canvas_height
                    new_width = int(new_height * img_aspect)

                # Ensure dimensions are at least 1x1
                new_width = max(1, new_width)
                new_height = max(1, new_height)

                resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                # Keep a reference to the PhotoImage object!
                self.tk_image = ImageTk.PhotoImage(resized_image)

                # Display the image on the canvas (centered)
                x_pos = (canvas_width - new_width) // 2
                y_pos = (canvas_height - new_height) // 2

                if self.image_display_id:
                    # Update existing image item
                    self.image_canvas.coords(self.image_display_id, x_pos, y_pos)
                    self.image_canvas.itemconfig(self.image_display_id, image=self.tk_image)
                else:
                    # Create new image item
                    self.image_display_id = self.image_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_image)

                # Set background color (useful if image doesn't fill canvas)
                self.image_canvas.config(bg='#333333') # Dark gray background

            except Exception as e:
                 print(f"Error updating image display for frame {self.current_frame_index}: {e}")
                 # Clean up canvas on error
                 if self.image_display_id: self.image_canvas.delete(self.image_display_id); self.image_display_id = None
                 self.image_canvas.config(bg='#CCCCCC') # Reset background
                 self.image_canvas.create_text(10, 10, anchor=tk.NW, text=f"Error displaying frame {self.current_frame_index}", tags="placeholder_text")

        else: # No combined image available for this frame
             if self.image_display_id: self.image_canvas.delete(self.image_display_id); self.image_display_id = None
             self.image_canvas.config(bg='#CCCCCC') # Reset background
             # Display placeholder text if canvas is realized
             if self.image_canvas.winfo_width() > 1 and self.image_canvas.winfo_height() > 1:
                  self.image_canvas.create_text(self.image_canvas.winfo_width()//2, self.image_canvas.winfo_height()//2,
                                                anchor=tk.CENTER, text=f"No Image Data\nFrame {self.current_frame_index}", tags="placeholder_text")


        # --- Update Other Data Text ---
        other_data_str = ""
        # Sort keys for consistent display order
        sorted_other_keys = sorted(self.other_data.keys())
        for data_name in sorted_other_keys:
            data_list = self.other_data[data_name]
            if self.current_frame_index < len(data_list):
                value = data_list[self.current_frame_index]
                # Use format_value helper, maybe just first line for brevity
                formatted_val = format_value(value)
                # Simple heuristic: if formatted value has multiple lines, take first line + "..."
                val_lines = formatted_val.split('\n')
                if len(val_lines) > 2: # e.g., Type line + Value line + more
                    val_repr = val_lines[0] + " " + val_lines[1] # Combine type/shape and maybe start of value
                    if len(val_repr) > 100: val_repr = val_repr[:97] + "..."
                elif len(val_lines) == 2: # Type + Value (short)
                    val_repr = val_lines[0] + " " + val_lines[1]
                    if len(val_repr) > 100: val_repr = val_repr[:97] + "..."
                else: # Single line format
                     val_repr = formatted_val
                     if len(val_repr) > 100: val_repr = val_repr[:97] + "..."

                other_data_str += f"{data_name}: {val_repr}\n"
            # else: (Data missing for this frame - already checked during processing)

        self.other_data_text.config(state=tk.NORMAL)
        self.other_data_text.delete('1.0', tk.END)
        self.other_data_text.insert('1.0', other_data_str if other_data_str else "No other data tracked.")
        self.other_data_text.config(state=tk.DISABLED)

        # --- Update Frame Label ---
        self.frame_label.config(text=f"Frame: {self.current_frame_index} / {max(0, self.num_frames - 1)}")


    def set_frame(self, frame_index):
        """Sets the current frame index and updates the display."""
        if self.num_frames == 0: return # No data loaded
        new_index = max(0, min(frame_index, self.num_frames - 1))
        if new_index != self.current_frame_index or not self.image_display_id: # Update only if index changed or image not shown yet
             self.current_frame_index = new_index
             self.update_display()

    def slider_update(self, value):
        """Callback when the slider value changes."""
        # Stop playback if user interacts with slider
        if self.video_playing:
            self.stop_video()
        # Value from slider is a float string, convert to int
        self.set_frame(int(float(value)))

    def next_frame(self):
        """Moves to the next frame."""
        if self.video_playing: self.stop_video() # Stop playback on manual advance
        if self.current_frame_index < self.num_frames - 1:
            new_index = self.current_frame_index + 1
            self.set_frame(new_index)
            self.frame_slider.set(new_index) # Keep slider in sync

    def prev_frame(self):
        """Moves to the previous frame."""
        if self.video_playing: self.stop_video() # Stop playback on manual advance
        if self.current_frame_index > 0:
            new_index = self.current_frame_index - 1
            self.set_frame(new_index)
            self.frame_slider.set(new_index) # Keep slider in sync

    def toggle_play(self):
        """Starts or stops the video playback."""
        if self.video_playing:
            self.stop_video()
        else:
            # Start from current frame if stopped, or from 0 if at end
            if self.current_frame_index == self.num_frames - 1:
                 self.set_frame(0)
                 self.frame_slider.set(0)
            self.play_video()

    def play_video(self):
        """Starts video playback using root.after()."""
        if self.num_frames <= 1: return # Cannot play single frame or no data
        self.video_playing = True
        self.play_button.config(text="Stop")
        self.set_status("Playing...")
        # Disable manual controls during playback
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.frame_slider.config(state=tk.DISABLED)
        self.playback_loop()

    def stop_video(self):
        """Stops video playback."""
        was_playing = self.video_playing
        self.video_playing = False
        if self.video_job:
            self.root.after_cancel(self.video_job)
            self.video_job = None
        # Re-enable controls only if there's valid data
        if self.num_frames > 0:
             self.play_button.config(text="Play", state=tk.NORMAL if self.num_frames > 1 else tk.DISABLED)
             self.prev_button.config(state=tk.NORMAL if self.num_frames > 1 else tk.DISABLED)
             self.next_button.config(state=tk.NORMAL if self.num_frames > 1 else tk.DISABLED)
             self.frame_slider.config(state=tk.NORMAL if self.num_frames > 1 else tk.DISABLED)
             if was_playing: # Only change status if it was actually playing
                  self.set_status("Stopped.")


    def playback_loop(self):
        """The core loop for video playback, called repeatedly."""
        if not self.video_playing:
            return

        # Update display for current frame *before* incrementing
        self.update_display()
        # self.frame_slider.set(self.current_frame_index) # Don't update slider during play, too slow

        # Move to next frame or stop at the end
        if self.current_frame_index < self.num_frames - 1:
            self.current_frame_index += 1
            delay = max(10, int(1000 / DEFAULT_FPS)) # ms per frame, minimum delay 10ms
            self.video_job = self.root.after(delay, self.playback_loop)
        else:
            # Reached the end
            self.stop_video()
            # Keep slider at the end frame
            self.frame_slider.set(self.current_frame_index)
            self.set_status("Playback finished.")


# --- Run the Application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PklExplorerApp(root)
    # Handle window closing gracefully
    def on_closing():
        app.stop_video() # Stop any background tasks
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()