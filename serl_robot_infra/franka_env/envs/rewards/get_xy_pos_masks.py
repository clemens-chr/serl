import numpy as np
import cv2
import os
from typing import Tuple, Dict, List, Optional


def get_white_pixel_center(mask_image: np.ndarray, threshold: int = 127) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the average x and y position of white pixels in a mask image.
    
    Args:
        mask_image: Input mask image (grayscale or BGR)
        threshold: Threshold value to consider a pixel as "white" (default: 127)
        
    Returns:
        Tuple of (avg_x, avg_y) or (None, None) if no white pixels found
    """
    # Convert to grayscale if image is BGR
    if len(mask_image.shape) == 3:
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_image
    
    # Find white pixels (above threshold)
    white_pixels = np.where(gray > threshold)
    
    if len(white_pixels[0]) == 0:
        print("No white pixels found in the image")
        return None, None
    
    # Calculate average position
    # Note: white_pixels[0] is y coordinates, white_pixels[1] is x coordinates
    avg_y = np.mean(white_pixels[0])
    avg_x = np.mean(white_pixels[1])
    
    return float(avg_x), float(avg_y)


def get_white_pixel_center_normalized(mask_image: np.ndarray, threshold: int = 127) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the average x and y position of white pixels in a mask image, normalized to [0, 1].
    
    Args:
        mask_image: Input mask image (grayscale or BGR)
        threshold: Threshold value to consider a pixel as "white" (default: 127)
        
    Returns:
        Tuple of (normalized_avg_x, normalized_avg_y) or (None, None) if no white pixels found
    """
    avg_x, avg_y = get_white_pixel_center(mask_image, threshold)
    
    if avg_x is None or avg_y is None:
        return None, None
    
    height, width = mask_image.shape[:2]
    normalized_x = avg_x / width
    normalized_y = avg_y / height
    
    return float(normalized_x), float(normalized_y)



def main():
    """
    Main function to demonstrate usage with example images.
    Modify the image_paths list to include your specific images.
    """
    
    base_path = "/home/ccc/orca_ws/src/"
    # Example image paths - modify these to match your actual image locations
    image_paths = [
        os.path.join(base_path, "1-tl.png"),
        os.path.join(base_path, "1-tr.png"), 
        os.path.join(base_path, "2-tl.png"),
        os.path.join(base_path, "sgm.png")
    ]
    
    print("Extracting white pixel centers from mask images...")
    print("=" * 50)
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        result = get_white_pixel_center_normalized(image)
        print(f"{os.path.basename(image_path)}: ({result[0]:.2f}, {result[1]:.2f})")
    
 
if __name__ == "__main__":
    main()
