import cv2
import numpy as np
import json
import os
import sys
from tkinter import filedialog, Tk

# Global variables for trackbar callbacks
img_original = None
img_display = None
window_name = "Fisheye Calibration"
trackbar_window = "Parameters"

# Default parameters
default_params = {
    'k1': -30,      # Multiplied by 100 for trackbar precision
    'k2': 10,       # Multiplied by 100 for trackbar precision  
    'k3': 0,        # Multiplied by 100 for trackbar precision
    'k4': 0,        # Multiplied by 100 for trackbar precision
    'alpha': 100,   # Multiplied by 100 for trackbar precision (0-100%)
    'fx_scale': 50, # Focal length scale (20-100%)
    'fy_scale': 50, # Focal length scale (20-100%)
}

def create_camera_matrix(width, height, fx_scale, fy_scale):
    """Create camera matrix based on image dimensions and scale factors."""
    fx = width * fx_scale / 100.0
    fy = height * fy_scale / 100.0
    cx = width / 2.0
    cy = height / 2.0
    
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

def undistort_fisheye(img, k1, k2, k3, k4, alpha, fx_scale, fy_scale):
    """Apply fisheye correction with given parameters."""
    h, w = img.shape[:2]
    
    # Convert trackbar values to actual parameters
    dist_coeffs = np.array([
        k1 / 100.0,  # Convert back from trackbar range
        k2 / 100.0,
        k3 / 100.0, 
        k4 / 100.0
    ], dtype=np.float32)
    
    alpha_val = alpha / 100.0  # Convert to 0-1 range
    
    # Create camera matrix
    camera_matrix = create_camera_matrix(w, h, fx_scale, fy_scale)
    
    try:
        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha_val, (w, h)
        )
        
        # Undistort the image
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop if ROI is valid
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0 and x >= 0 and y >= 0:
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]
            # Resize back to original size for comparison
            undistorted = cv2.resize(undistorted, (w, h))
        
        return undistorted, True
        
    except Exception as e:
        print(f"Error in undistortion: {e}")
        return img.copy(), False

def update_display(val=None):
    """Update the display with current parameter values."""
    global img_original, img_display
    
    if img_original is None:
        return
    
    # Get current trackbar values
    k1 = cv2.getTrackbarPos('K1 (*100)', trackbar_window)
    k2 = cv2.getTrackbarPos('K2 (*100)', trackbar_window)
    k3 = cv2.getTrackbarPos('K3 (*100)', trackbar_window)
    k4 = cv2.getTrackbarPos('K4 (*100)', trackbar_window)
    alpha = cv2.getTrackbarPos('Alpha (%)', trackbar_window)
    fx_scale = cv2.getTrackbarPos('FX Scale (%)', trackbar_window)
    fy_scale = cv2.getTrackbarPos('FY Scale (%)', trackbar_window)
    
    # Apply undistortion
    undistorted, success = undistort_fisheye(img_original, k1, k2, k3, k4, alpha, fx_scale, fy_scale)
    
    # Create side-by-side comparison
    h, w = img_original.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Original image on the left
    comparison[:, :w] = img_original
    
    # Undistorted image on the right
    comparison[:, w:] = undistorted
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, 'Corrected', (w + 10, 30), font, 1, (0, 255, 0), 2)
    
    # Add parameter values as text
    param_text = [
        f'K1: {k1/100.0:.3f}',
        f'K2: {k2/100.0:.3f}',
        f'K3: {k3/100.0:.3f}',
        f'K4: {k4/100.0:.3f}',
        f'Alpha: {alpha/100.0:.2f}',
        f'FX Scale: {fx_scale}%',
        f'FY Scale: {fy_scale}%'
    ]
    
    for i, text in enumerate(param_text):
        y_pos = h - 150 + i * 20
        cv2.putText(comparison, text, (w + 10, y_pos), font, 0.5, (255, 255, 255), 1)
    
    # Status indicator
    status_color = (0, 255, 0) if success else (0, 0, 255)
    status_text = "OK" if success else "ERROR"
    cv2.putText(comparison, status_text, (w + 10, 60), font, 0.7, status_color, 2)
    
    # Resize for display if image is too large
    display_h, display_w = comparison.shape[:2]
    max_display_width = 1400
    if display_w > max_display_width:
        scale = max_display_width / display_w
        new_w = int(display_w * scale)
        new_h = int(display_h * scale)
        comparison = cv2.resize(comparison, (new_w, new_h))
    
    cv2.imshow(window_name, comparison)

def save_parameters():
    """Save current parameters to a JSON file."""
    # Get current trackbar values
    k1 = cv2.getTrackbarPos('K1 (*100)', trackbar_window) / 100.0
    k2 = cv2.getTrackbarPos('K2 (*100)', trackbar_window) / 100.0
    k3 = cv2.getTrackbarPos('K3 (*100)', trackbar_window) / 100.0
    k4 = cv2.getTrackbarPos('K4 (*100)', trackbar_window) / 100.0
    alpha = cv2.getTrackbarPos('Alpha (%)', trackbar_window) / 100.0
    fx_scale = cv2.getTrackbarPos('FX Scale (%)', trackbar_window)
    fy_scale = cv2.getTrackbarPos('FY Scale (%)', trackbar_window)
    
    # Create parameter dictionary
    params = {
        'FISHEYE_CONFIG': {
            'enabled': True,
            'k1': k1,
            'k2': k2,
            'k3': k3,
            'k4': k4,
            'alpha': alpha,
            'new_camera_matrix_scale': 0.8
        },
        'CAMERA_MATRIX_CONFIG': {
            'fx_scale_percent': fx_scale,
            'fy_scale_percent': fy_scale,
            'use_calibrated_matrix': True,
            'auto_focal_length': True
        },
        'RESOLUTION_OVERRIDES': {
            'apply_to_all_resolutions': True,
            'preserve_aspect_ratio': True
        },
        'NOTE': 'Use these parameters in your main script FISHEYE_CONFIG section'
    }
    
    # Save to file
    output_file = 'fisheye_calibration_params.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"\nParameters saved to: {output_file}")
        print("You can copy these values to your main script's FISHEYE_CONFIG section.")
        print(f"K1: {k1:.3f}, K2: {k2:.3f}, K3: {k3:.3f}, K4: {k4:.3f}")
        print(f"Alpha: {alpha:.3f}, FX Scale: {fx_scale}%, FY Scale: {fy_scale}%")
        print(f"Camera Matrix: Auto-generated with FX/FY scale factors")
    except Exception as e:
        print(f"Error saving parameters: {e}")

def select_image():
    """Select an image file using file dialog."""
    root = Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Test Image for Fisheye Calibration",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def main():
    global img_original, img_display
    
    print("=== Fisheye Calibration Tool ===")
    print("This tool helps you calibrate fisheye distortion parameters.")
    print("Instructions:")
    print("1. Select a test image with visible distortion")
    print("2. Adjust parameters using trackbars")
    print("3. Press 's' to save parameters")
    print("4. Press 'r' to reset to defaults")
    print("5. Press 'q' or ESC to quit")
    print()
    
    # Select image
    image_path = select_image()
    if not image_path:
        print("No image selected. Exiting.")
        return
    
    # Load image
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Loaded image: {image_path}")
    print(f"Image dimensions: {img_original.shape[1]}x{img_original.shape[0]}")
    
    # Create windows
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(trackbar_window, cv2.WINDOW_AUTOSIZE)
    
    # Create trackbars with extended ranges
    cv2.createTrackbar('K1 (*100)', trackbar_window, default_params['k1'] + 100, 500, update_display)  # -100 to +100
    cv2.createTrackbar('K2 (*100)', trackbar_window, default_params['k2'] + 50, 300, update_display)   # -50 to +50
    cv2.createTrackbar('K3 (*100)', trackbar_window, default_params['k3'] + 50, 100, update_display)   # -50 to +50
    cv2.createTrackbar('K4 (*100)', trackbar_window, default_params['k4'] + 50, 100, update_display)   # -50 to +50
    cv2.createTrackbar('Alpha (%)', trackbar_window, default_params['alpha'], 100, update_display)      # 0 to 100%
    cv2.createTrackbar('FX Scale (%)', trackbar_window, default_params['fx_scale'], 100, update_display) # 0 to 100%
    cv2.createTrackbar('FY Scale (%)', trackbar_window, default_params['fy_scale'], 100, update_display) # 0 to 100%
    
    # Initial display
    update_display()
    
    print("\nUse trackbars to adjust parameters. Watch the real-time preview.")
    print("Press 's' to save, 'r' to reset, 'q' or ESC to quit.")
    
    # Main loop
    while True:
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s'):  # Save parameters
            save_parameters()
        elif key == ord('r'):  # Reset to defaults
            cv2.setTrackbarPos('K1 (*100)', trackbar_window, default_params['k1'] + 100)
            cv2.setTrackbarPos('K2 (*100)', trackbar_window, default_params['k2'] + 50)
            cv2.setTrackbarPos('K3 (*100)', trackbar_window, default_params['k3'] + 50)
            cv2.setTrackbarPos('K4 (*100)', trackbar_window, default_params['k4'] + 50)
            cv2.setTrackbarPos('Alpha (%)', trackbar_window, default_params['alpha'])
            cv2.setTrackbarPos('FX Scale (%)', trackbar_window, default_params['fx_scale'])
            cv2.setTrackbarPos('FY Scale (%)', trackbar_window, default_params['fy_scale'])
            update_display()
            print("Parameters reset to defaults.")
    
    cv2.destroyAllWindows()
    print("Calibration tool closed.")

if __name__ == "__main__":
    main()
