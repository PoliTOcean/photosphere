import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import json

# === CONFIGURATION ===
path = "C:\\Users\\ginof\\photosphere\\photos\\"

# === GLOBALS ===
camera_azimuth = 0.0
camera_elevation = 0.0
last_x, last_y = 0, 0
mouse_down = False

texture_north = None
texture_south = None
sector_textures = []

north_rotation_angle = 0.0
south_rotation_angle = 0.0

equator_lat_extent = 70

debug_active = True

sector_crop = 1.0

sector_data = []

def load_imu_data():
    global sector_data
    imu_path = path + "IMU_data.json"
    with open(imu_path, "r") as f:
        sector_data = json.load(f)

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
    global texture_north, texture_south, sector_textures
    texture_north = load_texture_from_file(path + "north.jpg")
    texture_south = load_texture_from_file(path + "south.jpg")

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

def draw_curved_patch_on_sphere(radius, pitch_deg, yaw_deg, roll_deg, crop=0.2, resolution=16):
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    roll = np.radians(roll_deg)

    # Compute central direction vector
    center = np.array([
        np.cos(pitch) * np.cos(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.sin(yaw)
    ])

    # Build local tangent space
    up = np.array([0, 1, 0])
    if np.allclose(center, up):
        up = np.array([0, 0, 1])
    right = np.cross(up, center)
    right /= np.linalg.norm(right)
    up = np.cross(center, right)
    up /= np.linalg.norm(up)

    # Apply roll rotation to tangent basis
    rot = np.array([
        [np.cos(roll), -np.sin(roll)],
        [np.sin(roll),  np.cos(roll)]
    ])

    # Grid patch around center direction
    glBegin(GL_QUADS)
    for i in range(resolution):
        for j in range(resolution):
            for dx, dy in [(0,0), (1,0), (1,1), (0,1)]:
                u = (i + dx) / resolution
                v = (j + dy) / resolution
                local = np.array([u - 0.5, v - 0.5]) * 2 * crop
                local_rot = rot @ local
                offset_dir = right * local_rot[0] + up * local_rot[1]
                dir_vec = center + offset_dir
                dir_vec /= np.linalg.norm(dir_vec)
                glTexCoord2f(u, v)
                glVertex3f(*(radius * dir_vec))
    glEnd()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glRotatef(camera_elevation, 1.0, 0.0, 0.0)
    glRotatef(camera_azimuth, 0.0, 1.0, 0.0)
    gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0)

    for i, imu in enumerate(sector_data):
        glBindTexture(GL_TEXTURE_2D, sector_textures[i])
        draw_curved_patch_on_sphere(
            radius=2.01,
            pitch_deg=imu["pitch"],
            yaw_deg=imu["yaw"],
            roll_deg=imu["roll"],
            crop=sector_crop
        )
    glBindTexture(GL_TEXTURE_2D, 0)
    glutSwapBuffers()

def mouse(button, state, x, y):
    global mouse_down, last_x, last_y, sector_crop, pole_crop
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            mouse_down = True
            last_x, last_y = x, y
        elif state == GLUT_UP:
            mouse_down = False
    elif button == 4:  # Scroll down
        sector_crop = max(sector_crop - 0.02, 0.0)
        pole_crop = min(pole_crop + 0.05, 1.0)
    elif button == 3:  # Scroll up
        sector_crop = min(sector_crop + 0.02, 0.49)
        pole_crop = max(pole_crop - 0.05, 0.0)
    
    print_debug_info();

def keyboard(key, x, y):
    global north_rotation_angle, south_rotation_angle
    global equator_lat_extent

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