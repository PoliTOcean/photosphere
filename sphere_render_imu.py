import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import json
import piexif
import piexif.helper
import os # Added for directory listing

# === CONFIGURATION ===
path = "photos_imu/" # Reinstated path global

# Resolution-specific settings
RESOLUTION_SETTINGS = {
    (1600, 1200): {
        'yaw_span': 190,  # Much wider field of view for 1600x1200
        'pitch_span': 150,
        'fisheye_correction': True,
        'camera_matrix': np.array([[800.0, 0, 800.0],
                                  [0, 800.0, 600.0],
                                  [0, 0, 1]], dtype=np.float32),
        'distortion_coeffs': np.array([-0.3, 0.1, 0, 0], dtype=np.float32)  # k1, k2, p1, p2
    },
    'default': {
        'yaw_span': 80,
        'pitch_span': 50,
        'fisheye_correction': False,
        'camera_matrix': None,
        'distortion_coeffs': None
    }
}

# Fisheye correction parameters (configurable)
FISHEYE_CONFIG = {
    'enabled': True,  # Global enable/disable
    'k1': -0.3,       # Barrel distortion coefficient
    'k2': 0.1,        # Barrel distortion coefficient  
    'k3': 0.0,        # Higher order coefficient
    'k4': 0.0,        # Higher order coefficient
    'alpha': 1.0,     # Free scaling parameter (0=all pixels, 1=no black pixels)
    'new_camera_matrix_scale': 0.8,  # Scale factor for new camera matrix
    'camera_matrix_config': None  # Will be populated by calibration file if available
}

sector_crop = 0.8

# Circle mask radius for 1600x1200 display (in normalized [0,0.5], default 0.5 for inscribed circle)
circle_mask_radius = 0.4  # Increase up to ~0.6 for a larger circle, but not >0.707

# === GLOBALS ===
camera_azimuth = 0.0
camera_elevation = 0.0
last_x, last_y = 0, 0
mouse_down = False

sector_textures = []
image_processing_queue = [] # New global list

debug_active = True

sector_data = []
sector_settings = []  # New: Store settings per image

initial_yaw = 0.0
initial_pitch = 0.0
initial_roll = 0.0

def crop_to_largest_centered_circle(img):
    """
    Crop the input image to the largest possible centered circle and set outside pixels to black.
    Args:
        img: Input image (numpy array)
    Returns:
        Cropped image with outside pixels set to black (same shape as input)
    """
    h, w = img.shape[:2]
    radius = min(w, h) // 2
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = dist_from_center <= radius
    output = np.zeros_like(img)
    if img.ndim == 3:
        for c in range(img.shape[2]):
            output[..., c][mask] = img[..., c][mask]
    else:
        output[mask] = img[mask]
    return output

def load_calibration_parameters(resolution=None):
    """
    Load fisheye calibration parameters from the calibration tool's output file.
    Only loads for 1600x1200 resolution.
    """
    global FISHEYE_CONFIG
    if resolution != (1600, 1200):
        print("Skipping fisheye calibration parameters for non-1600x1200 resolution.")
        return False
    calibration_file = 'fisheye_calibration_params.json'
    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)
            if 'FISHEYE_CONFIG' in calibration_data:
                calibrated_config = calibration_data['FISHEYE_CONFIG']
                FISHEYE_CONFIG.update(calibrated_config)
                if 'CAMERA_MATRIX_CONFIG' in calibration_data:
                    FISHEYE_CONFIG['camera_matrix_config'] = calibration_data['CAMERA_MATRIX_CONFIG']
                print(f"Loaded calibration parameters from {calibration_file}:")
                print(f"  K1: {FISHEYE_CONFIG['k1']:.3f}")
                print(f"  K2: {FISHEYE_CONFIG['k2']:.3f}")
                print(f"  K3: {FISHEYE_CONFIG['k3']:.3f}")
                print(f"  K4: {FISHEYE_CONFIG['k4']:.3f}")
                print(f"  Alpha: {FISHEYE_CONFIG['alpha']:.3f}")
                if 'camera_matrix_config' in FISHEYE_CONFIG:
                    cam_config = FISHEYE_CONFIG['camera_matrix_config']
                    print(f"  Camera Matrix: FX Scale={cam_config.get('fx_scale_percent', 50)}%, FY Scale={cam_config.get('fy_scale_percent', 50)}%")
                return True
            else:
                print(f"Warning: No FISHEYE_CONFIG found in {calibration_file}")
                return False
        except Exception as e:
            print(f"Error loading calibration parameters from {calibration_file}: {e}")
            print("Using default parameters instead.")
            return False
    else:
        print(f"No calibration file found at {calibration_file}")
        print("Using default fisheye parameters. Run fisheye_calibration.py to create calibrated parameters.")
        return False

# Add the new function to read IMU data from image EXIF
def read_imu_from_image(image_path):
    """
    Reads IMU data embedded in the EXIF UserComment tag of an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the IMU data, or None if not found or an error occurs.
    """
    try:
        print(f"  [read_imu] Attempting piexif.load for {image_path}")
        exif_dict = piexif.load(image_path)
        print(f"  [read_imu] piexif.load successful for {image_path}")
        user_comment_bytes = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)

        if user_comment_bytes:
            try:
                user_comment_str = piexif.helper.UserComment.load(user_comment_bytes)
            except UnicodeDecodeError:
                try:
                    user_comment_str = user_comment_bytes.decode('utf-8', errors='ignore')
                    if user_comment_str.startswith("UNICODE"):
                        user_comment_str = user_comment_str.split('\x00\x00', 1)[-1].strip('\x00')
                    elif user_comment_str.startswith("ASCII"):
                         user_comment_str = user_comment_str.split('\x00\x00', 1)[-1].strip('\x00')
                except Exception as e_decode_fallback:
                    print(f"Error decoding UserComment with fallback: {e_decode_fallback}")
                    return None
            except Exception as e_load_comment: # Catch other piexif.helper.UserComment.load errors
                print(f"Error loading UserComment: {e_load_comment}")
                return None


            try:
                imu_data = json.loads(user_comment_str)
                return imu_data
            except json.JSONDecodeError as e_json:
                print(f"Failed to parse UserComment as JSON: {e_json}")
                print(f"UserComment content was: {user_comment_str}")
                return None
        else:
            print(f"No EXIF UserComment tag found in the image: {image_path}")
            return None

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except piexif.InvalidImageDataError:
        print(f"Error: Invalid image data or no EXIF data in {image_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading EXIF from {image_path}: {e}")
        return None

def undistort_fisheye_image(img, camera_matrix, dist_coeffs, alpha=1.0):
    """
    Correct fisheye distortion in an image.
    
    Args:
        img: Input image (numpy array)
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2] or [k1, k2, k3, k4]
        alpha: Free scaling parameter (0=all pixels visible, 1=no black pixels)
    
    Returns:
        Undistorted image
    """
    h, w = img.shape[:2]
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )
    
    # Undistort the image
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop the image if ROI is valid
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
    
    return undistorted

def create_calibrated_camera_matrix(width, height):
    """
    Create camera matrix using calibrated parameters if available.
    
    Args:
        width: Image width
        height: Image height
    
    Returns:
        Camera matrix (3x3 numpy array)
    """
    if 'camera_matrix_config' in FISHEYE_CONFIG:
        cam_config = FISHEYE_CONFIG['camera_matrix_config']
        fx_scale = cam_config.get('fx_scale_percent', 50) / 100.0
        fy_scale = cam_config.get('fy_scale_percent', 50) / 100.0
        
        # Use calibrated scale factors
        fx = width * fx_scale
        fy = height * fy_scale
        cx = width * 0.5
        cy = height * 0.5
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"Created calibrated camera matrix: FX={fx:.1f}, FY={fy:.1f}, CX={cx:.1f}, CY={cy:.1f}")
        return camera_matrix
    else:
        # Fallback to default method
        fx = fy = min(width, height) * 0.5
        cx, cy = width * 0.5, height * 0.5
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"Created default camera matrix: FX={fx:.1f}, FY={fy:.1f}, CX={cx:.1f}, CY={cy:.1f}")
        return camera_matrix

def get_resolution_settings(width, height):
    """
    Get appropriate settings based on image resolution.
    Only enable fisheye correction and calibration for 1600x1200.
    """
    resolution_key = (width, height)
    if resolution_key == (1600, 1200):
        settings = RESOLUTION_SETTINGS[resolution_key].copy()
        print(f"Using specific settings for resolution {width}x{height}")
        # Load calibration only for 1600x1200
        load_calibration_parameters(resolution=resolution_key)
        if settings['fisheye_correction'] and FISHEYE_CONFIG['enabled']:
            settings['distortion_coeffs'] = np.array([
                FISHEYE_CONFIG['k1'], FISHEYE_CONFIG['k2'], 
                FISHEYE_CONFIG['k3'], FISHEYE_CONFIG['k4']
            ], dtype=np.float32)
            settings['camera_matrix'] = create_calibrated_camera_matrix(width, height)
            print(f"Applied fisheye correction with K1={FISHEYE_CONFIG['k1']:.3f}, K2={FISHEYE_CONFIG['k2']:.3f}")
        return settings
    else:
        print(f"Using default settings for resolution {width}x{height}")
        default_settings = RESOLUTION_SETTINGS['default'].copy()
        # Do NOT enable fisheye correction or calibration for other resolutions
        default_settings['fisheye_correction'] = False
        default_settings['distortion_coeffs'] = None
        default_settings['camera_matrix'] = None
        return default_settings

def load_texture_from_file(filepath):
    print(f"  [load_texture] Attempting cv2.imread for {filepath}")
    img = cv2.imread(filepath)
    print(f"  [load_texture] cv2.imread status for {filepath}: {'OK' if img is not None else 'Failed - img is None'}")
    if img is None:
        print(f"Failed to load image at {filepath}")
        sys.exit(1)

    # Get image dimensions before any processing
    original_h, original_w = img.shape[:2]
    print(f"  [load_texture] Original image dimensions for {filepath}: w={original_w}, h={original_h}")
    
    # Get resolution-specific settings
    settings = get_resolution_settings(original_w, original_h)
    
    # Apply fisheye correction if enabled
    if settings['fisheye_correction'] and FISHEYE_CONFIG['enabled']:
        print(f"  [load_texture] Applying fisheye correction for {filepath}")
        try:
            img = undistort_fisheye_image(
                img, 
                settings['camera_matrix'], 
                settings['distortion_coeffs'], 
                FISHEYE_CONFIG['alpha']
            )
            print(f"  [load_texture] Fisheye correction applied successfully for {filepath}")
        except Exception as e:
            print(f"  [load_texture] Warning: Fisheye correction failed for {filepath}: {e}")
            print(f"  [load_texture] Continuing with original image...")

    # Crop to largest centered circle for 1600x1200 images (after fisheye correction)
    h, w = img.shape[:2]
    print("wioddt:", w, "h:", h)
    if w < 1100:
        print(f"  [load_texture] Cropping to largest centered circle for {filepath}")
        img = crop_to_largest_centered_circle(img)
        print(f"  [load_texture] Cropping to circle complete for {filepath}")

    print(f"  [load_texture] Attempting cv2.flip for {filepath}")
    img = cv2.flip(img, 0)
    print(f"  [load_texture] cv2.flip successful for {filepath}")

    print(f"  [load_texture] Attempting cv2.cvtColor for {filepath}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  [load_texture] cv2.cvtColor successful for {filepath}")

    print(f"  [load_texture] Attempting np.ascontiguousarray for {filepath}")
    img = np.ascontiguousarray(img, dtype=np.uint8)
    print(f"  [load_texture] np.ascontiguousarray successful for {filepath}")

    h, w, channels = img.shape
    print(f"  [load_texture] Final image dimensions for {filepath}: w={w}, h={h}, channels={channels}")

    # Store settings for this image (could be used later if needed per-image settings)
    # For now, we'll update global settings based on the first image processed
    
    # ...existing OpenGL texture creation code...
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    print(f"  [load_texture] glPixelStorei successful for {filepath}")

    print(f"  [load_texture] Attempting glGenTextures for {filepath}")
    texture_id = glGenTextures(1)
    print(f"  [load_texture] glGenTextures successful, ID: {texture_id} for {filepath}")

    print(f"  [load_texture] Attempting glBindTexture for {filepath}")
    glBindTexture(GL_TEXTURE_2D, texture_id)
    print(f"  [load_texture] glBindTexture successful for {filepath}")

    print(f"  [load_texture] Attempting glTexParameteri (MIN_FILTER) for {filepath}")
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    print(f"  [load_texture] glTexParameteri (MIN_FILTER) successful for {filepath}")

    print(f"  [load_texture] Attempting glTexParameteri (MAG_FILTER) for {filepath}")
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    print(f"  [load_texture] glTexParameteri (MAG_FILTER) successful for {filepath}")

    print(f"  [load_texture] Attempting img.tobytes() for {filepath}")
    img_bytes = img.tobytes()
    print(f"  [load_texture] img.tobytes() successful, byte length: {len(img_bytes)} for {filepath}")

    print(f"  [load_texture] Attempting glTexImage2D for {filepath}")
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_bytes)
    print(f"  [load_texture] glTexImage2D successful for {filepath}")

    print(f"  [load_texture] Attempting glBindTexture(0) for {filepath}")
    glBindTexture(GL_TEXTURE_2D, 0)
    print(f"  [load_texture] glBindTexture(0) successful for {filepath}")

    return texture_id, settings

# Modified: This function now collects image paths and IMU data from the 'path' directory.
# It does not perform OpenGL operations.
def collect_image_paths_and_imu():
    global image_processing_queue, path # To store paths and IMU data

    # # Initialize Tkinter root (it won't be shown) # Removed Tkinter
    # root = Tk() # Removed Tkinter
    # root.withdraw() # Hide the main window # Removed Tkinter

    # # Open file dialog to select images # Removed Tkinter
    # image_paths_tuple = askopenfilenames( # Renamed to avoid conflict if needed, though it's removed
    #     title="Select Images",
    #     filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*"))
    # )
    # # Tkinter GUI is destroyed here, before this function returns. # Removed Tkinter
    # root.destroy() # Destroy the root window after selection # Removed Tkinter

    # if not image_paths_tuple: # Changed from image_paths
    #     print("No images selected. Exiting.")
    #     sys.exit(0)

    # image_paths = list(image_paths_tuple) # Convert tuple to list

    if not os.path.isdir(path):
        print(f"Error: The specified path '{path}' is not a directory or does not exist.")
        sys.exit(1)

    image_filenames = []
    try:
        for filename in os.listdir(path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_filenames.append(os.path.join(path, filename))
    except OSError as e:
        print(f"Error reading directory {path}: {e}")
        sys.exit(1)

    if not image_filenames:
        print(f"No image files (.jpg, .png) found in directory: {path}")
        sys.exit(0)
    
    print(f"Found {len(image_filenames)} images in directory: {path}")

    image_processing_queue = [] # Clear any previous queue

    for image_path in image_filenames: # Iterate over found image files
        print(f"Queueing for processing: {image_path}")
        
        print(f"  Attempting to read IMU data for {image_path}...")
        imu_data = read_imu_from_image(image_path)
        print(f"  Finished reading IMU data for {image_path}. IMU data is: {'present' if imu_data else 'absent'}")

        if imu_data:
            image_processing_queue.append({'path': image_path, 'imu': imu_data})
        else:
            print(f"Could not load IMU data for {image_path}. Skipping this image for texture loading.")

    if not image_processing_queue:
        print("No images with valid IMU data were queued for processing. Exiting.")
        sys.exit(1)

# New function to perform OpenGL texture loading and finalize data
def finalize_texture_loading():
    global sector_textures, sector_data, initial_yaw, initial_pitch, initial_roll, image_processing_queue
    global sector_settings

    sector_textures = []
    sector_data = []
    sector_settings = []  # Initialize per-image settings
    first_image = True

    print("Starting final texture loading and data setup...")
    for item in image_processing_queue:
        image_path = item['path']
        imu_data = item['imu']

        print(f"  Processing from queue: {image_path}")
        sector_data.append(imu_data) # Add IMU data
        
        print(f"  Attempting to load texture for {image_path}...")
        texture_id, settings = load_texture_from_file(image_path) # Now returns settings too
        sector_textures.append(texture_id)
        sector_settings.append(settings)  # Store settings per image
        print(f"  Finished loading texture for {image_path}. Texture ID: {texture_id}")
        print(f"  Settings for this image: Yaw span={settings['yaw_span']}, Pitch span={settings['pitch_span']}")

        if first_image:
            initial_yaw = float(imu_data.get("yaw", 0.0))
            initial_pitch = float(imu_data.get("pitch", 0.0))
            initial_roll = float(imu_data.get("roll", 0.0))
            
            print(f"Initial orientation set from {image_path}: Yaw={initial_yaw}, Pitch={initial_pitch}, Roll={initial_roll}")
            first_image = False
            
    if not sector_textures:
        print("No valid textures were loaded into OpenGL. Exiting.")
        sys.exit(1)
    print("All textures loaded and data finalized.")


def init():
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    glClearColor(0.5, 0.5, 0.5, 1.0)

def reshape(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(40, width / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def print_debug_info():
    if debug_active:
        print(f"[DEBUG] Sectors Crop: {sector_crop:.2f}")
        print(f"[DEBUG] ")
        
def draw_textured_patch(texture_id, yaw_deg, pitch_deg, roll_deg, radius=2.0, yaw_span=30, pitch_span=20, image_shape=None):
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Convert angles to radians
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)
    yaw_span_rad = np.radians(yaw_span)
    pitch_span_rad = np.radians(pitch_span)

    # Compute patch center direction vector
    cx = np.cos(pitch_rad) * np.sin(yaw_rad)
    cy = np.sin(pitch_rad)
    cz = -np.cos(pitch_rad) * np.cos(yaw_rad)
    center = np.array([cx, cy, cz])

    # Build local coordinate axes for the patch
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(center, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, center)
    up = up / np.linalg.norm(up)

    # Apply roll around center axis
    cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
    up_rot = cos_r * up + sin_r * right
    right_rot = -sin_r * up + cos_r * right

    # Draw the patch
    rows, cols = 20, 20
    # Determine if we need to apply a circular mask (only for 1600x1200)
    apply_circle_mask = False
    mask_r = 0.0
    if image_shape is not None and image_shape[1] == 1600 and image_shape[0] == 1200:
        apply_circle_mask = True
        mask_r = circle_mask_radius  # Use configurable parameter
    for i in range(rows):
        v0 = i / rows
        v1 = (i + 1) / rows
        pitch0 = (v0 - 0.5) * pitch_span_rad
        pitch1 = (v1 - 0.5) * pitch_span_rad
        glBegin(GL_QUAD_STRIP)
        for j in range(cols + 1):
            u = j / cols
            yaw_offset = (u - 0.5) * yaw_span_rad
            for v, pitch_offset in zip([v0, v1], [pitch0, pitch1]):
                # Apply circular mask in texture space for 1600x1200
                if apply_circle_mask:
                    du = u - 0.5
                    dv = v - 0.5
                    if du * du + dv * dv > mask_r * mask_r:
                        continue
                # Direction in local patch space
                # Corrected the sign of the center component
                dir = (
                    np.cos(pitch_offset) * np.sin(yaw_offset) * right_rot +
                    np.sin(pitch_offset) * up_rot +
                    np.cos(pitch_offset) * np.cos(yaw_offset) * center
                )
                dir = dir / np.linalg.norm(dir)
                pos = radius * dir
                glTexCoord2f(u, (pitch_offset + pitch_span_rad / 2) / pitch_span_rad)
                glVertex3f(*pos)
        glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)

def display():
    #print("[display] Display function called.") # Unconditional print

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glRotatef(camera_elevation, 1.0, 0.0, 0.0)
    glRotatef(camera_azimuth, 0.0, 1.0, 0.0)
    #print(f"[display] Camera Azimuth: {camera_azimuth:.2f}, Elevation: {camera_elevation:.2f}")
    gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0)

    for i, imu_entry in enumerate(sector_data):
        if i >= len(sector_textures) or i >= len(sector_settings):
            #print(f"[display] Error: Skipping sector {i}, missing texture or settings.")
            continue
            
        texture_id = sector_textures[i]
        settings = sector_settings[i]  # Get settings for this specific image
        
        # Use per-image settings instead of global ones
        scaled_yaw_span = settings['yaw_span'] * sector_crop
        scaled_pitch_span = settings['pitch_span'] * sector_crop
        
        yaw = float(imu_entry.get("yaw", 0.0))
        pitch = float(imu_entry.get("pitch", 0.0))
        pitch = max(min(pitch, 89), -89)  # Clamp pitch between -89 and 89 degrees          
        roll = float(imu_entry.get("roll", 0.0))

        #print(f"[display] Drawing patch {i+1}/{len(sector_data)}: TexID={texture_id}, Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}, YawSpan={scaled_yaw_span:.2f}, PitchSpan={scaled_pitch_span:.2f}")

        # Pass image shape for masking
        image_shape = None
        if settings.get('camera_matrix') is not None:
            # Only 1600x1200 will have camera_matrix set
            image_shape = (1200, 1600, 3)
        draw_textured_patch(texture_id, yaw, pitch, roll,
                            radius=2.0-i*0.01,
                            yaw_span=scaled_yaw_span,
                            pitch_span=scaled_pitch_span,
                            image_shape=image_shape)

    glutSwapBuffers()
    #print("[display] SwapBuffers called.") # Confirm end of display function

def mouse(button, state, x, y):
    global mouse_down, last_x, last_y, sector_crop
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            mouse_down = True
            last_x, last_y = x, y
        elif state == GLUT_UP:
            mouse_down = False
    elif button == 4:  # Scroll down
        sector_crop = max(sector_crop - 0.02, 0.3)
    elif button == 3:  # Scroll up
        sector_crop = min(sector_crop + 0.02, 0.8)
    
    print_debug_info();

def keyboard(key, x, y):
    if key == b'\x1b':  # ESC
        sys.exit(0)
    
    print_debug_info();

def motion(x, y):
    global camera_azimuth, camera_elevation, last_x, last_y
    if mouse_down:
        dx = x - last_x
        dy = y - last_y
        camera_azimuth += dx * 0.5
        camera_elevation += dy * 0.5
        last_x, last_y = x, y

def main():
    print("=== 360 Sphere Viewer with IMU ===")
    # Do not load calibration globally here, only per-image in get_resolution_settings
    print()
    collect_image_paths_and_imu() 

    # Step 2: Initialize GLUT and create the OpenGL window (and context)
    # This happens *after* collect_image_paths_and_imu() has completed.
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1600, 900)
    glutCreateWindow(b"360 Sphere Viewer")
    
    # Step 3: Initialize OpenGL states (glEnable, glShadeModel, etc.)
    init()  
    
    # Step 4: Now that OpenGL context is ready, load textures into OpenGL
    # and finalize related data.
    finalize_texture_loading()
    
    # Register GLUT callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse)
    glutKeyboardFunc(keyboard)
    glutMotionFunc(motion)
    glutIdleFunc(display) # This should keep calling display

    print("OpenGL setup complete. Requesting initial display.")
    glutPostRedisplay() # Ensure the first frame is drawn
    
    glutMainLoop()

if __name__ == "__main__":
    main()

