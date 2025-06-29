import cv2
import time
import numpy as np
from typing import Tuple, Optional

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



def is_left_cube(sgm):
    avg_x, avg_y = get_white_pixel_center_normalized(sgm)
    print(f'avg_x: {avg_x}, avg_y: {avg_y}')
    if avg_x > 0.35 and avg_x < 0.68 and avg_y < 0.53:
        return 1
    else:
        return 0
    

def is_right_cube(sgm):
    avg_x, avg_y = get_white_pixel_center_normalized(sgm)
    if avg_x < 0.5 and avg_x > 0.25 and avg_y < 0.53:
        return 1
    else:
        return 0
    
def is_cube_lifted(sgm):
    avg_x, avg_y = get_white_pixel_center_normalized(sgm)
    return avg_x > 0.35 and avg_x < 0.75 and avg_y < 0.58
    
if __name__ == "__main__":
    
    while True:
        sgm = cv2.imread("/home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/sgm.png")
        if sgm is None:
            continue
        
        print(is_cube_lifted(sgm))
        time.sleep(0.05)