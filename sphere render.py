import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import json

# === CONFIGURATION ===
path = "C:\\Users\\ginof\\photosphere\\photos\\"

base_yaw_span  = 70  # degrees
base_pitch_span   = 45

sector_crop = 0.8

# === GLOBALS ===
camera_azimuth = 0.0
camera_elevation = 0.0
last_x, last_y = 0, 0
mouse_down = False

sector_textures = []

debug_active = True

sector_data = []

initial_yaw = 0.0
initial_pitch = 0.0
initial_roll = 0.0

def load_imu_data():
    global sector_data, initial_yaw, initial_pitch, initial_roll
    imu_path = path + "IMU_data.json"
    with open(imu_path, "r") as f:
        sector_data = json.load(f) 

    entry = sector_data[0]
    initial_yaw = entry.get("yaw", 0.0)
    initial_pitch = entry.get("pitch", 0.0)
    initial_roll = entry.get("roll", 0.0)

def load_texture_from_file(filepath):
    img = cv2.imread(filepath)
    if img is None:
        print(f"Failed to load image at {filepath}")
        sys.exit(1)
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w, _ = img.shape

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
    glBindTexture(GL_TEXTURE_2D, 0)

    return texture_id

def load_textures():
    global sector_textures

    for i in range(len(sector_data)):
        tex_path = path + f"{i+1}.jpg"
        sector_textures.append(load_texture_from_file(tex_path))

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
    gluPerspective(45, width / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def print_debug_info():
    if debug_active:
        print(f"[DEBUG] Sectors Crop: {sector_crop:.2f}")
        print(f"[DEBUG] ")
        
def draw_textured_patch(texture_id, yaw_deg, pitch_deg, roll_deg, radius=2.0, yaw_span=30, pitch_span=20):
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
    for i in range(rows):
        v0 = i / rows
        v1 = (i + 1) / rows
        pitch0 = (v0 - 0.5) * pitch_span_rad
        pitch1 = (v1 - 0.5) * pitch_span_rad
        glBegin(GL_QUAD_STRIP)
        for j in range(cols + 1):
            u = j / cols
            yaw_offset = (u - 0.5) * yaw_span_rad

            for pitch_offset in [pitch0, pitch1]:
                # Direction in local patch space
                dir = (
                    np.cos(pitch_offset) * np.sin(yaw_offset) * right_rot +
                    np.sin(pitch_offset) * up_rot +
                    -np.cos(pitch_offset) * np.cos(yaw_offset) * center
                )
                dir = dir / np.linalg.norm(dir)
                pos = radius * dir
                glTexCoord2f(u, (pitch_offset + pitch_span_rad / 2) / pitch_span_rad)
                glVertex3f(*pos)
        glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)

def display():
    global base_yaw_span , base_pitch_span 
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glRotatef(camera_elevation, 1.0, 0.0, 0.0)
    glRotatef(camera_azimuth, 0.0, 1.0, 0.0)
    gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0)

    # Scale spans by sector_crop
    scaled_yaw_span = base_yaw_span * sector_crop
    scaled_pitch_span = base_pitch_span * sector_crop

    for i, imu_entry in enumerate(sector_data):
        if i >= len(sector_textures):
            continue
        texture_id = sector_textures[i]
        yaw = float(imu_entry.get("yaw", 0.0))
        pitch = float(imu_entry.get("pitch", 0.0))
        pitch = max(min(pitch, 89), -89)  # Clamp pitch between -89 and 89 degrees          
        roll = float(imu_entry.get("roll", 0.0))

        draw_textured_patch(texture_id, yaw, pitch, roll,
                            yaw_span=scaled_yaw_span,
                            pitch_span=scaled_pitch_span)

    glutSwapBuffers()

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
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1600, 900)
    glutCreateWindow(b"360 Sphere Viewer")
    init()  
    load_imu_data()
    load_textures()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse)
    glutKeyboardFunc(keyboard)
    glutMotionFunc(motion)
    glutIdleFunc(display)
    glutMainLoop()

if __name__ == "__main__":
    main()