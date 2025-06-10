import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

# === CONFIGURATION ===
NUM_EQUATOR_SECTORS = 4

path = "D:\\Produzioni e progetti\\PoliTOcean\\Sphere rendering\\photos\\"

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

sector_crop = 0.0
pole_crop = 0.0

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

    for i in range(NUM_EQUATOR_SECTORS):
        tex_path = path + f"sector_{i+1}.jpg"
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
        print(f"[DEBUG] North Pole Rotation: {north_rotation_angle:.2f}°, South Pole Rotation: {south_rotation_angle:.2f}°")
        print(f"[DEBUG] Poles Crop: {pole_crop:.2f}")
        print(f"[DEBUG] Sectors Crop: {sector_crop:.2f}")
        print(f"[DEBUG] Sectors Max Latitude: {equator_lat_extent:.2f}")
        print(f"[DEBUG] ")

def draw_sphere_sector(radius, slices, stacks, lat_min, lat_max, lon_min, lon_max):
    lat_min_rad = np.radians(lat_min)
    lat_max_rad = np.radians(lat_max)
    lon_min_rad = np.radians(lon_min)
    lon_max_rad = np.radians(lon_max)

    for i in range(stacks):
        lat0 = lat_min_rad + (lat_max_rad - lat_min_rad) * i / stacks
        lat1 = lat_min_rad + (lat_max_rad - lat_min_rad) * (i + 1) / stacks
        y0 = np.sin(lat0)
        y1 = np.sin(lat1)
        r0 = np.cos(lat0)
        r1 = np.cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = lon_min_rad + (lon_max_rad - lon_min_rad) * j / slices
            x = np.cos(lng)
            z = np.sin(lng)

            u_min = sector_crop
            u_max = 1.0 - sector_crop
            u = u_min + (u_max - u_min) * j / slices
            v0 = i / stacks
            v1 = (i + 1) / stacks

            glTexCoord2f(u, v0)
            glVertex3f(radius * r0 * x, radius * y0, radius * r0 * z)

            glTexCoord2f(u, v1)
            glVertex3f(radius * r1 * x, radius * y1, radius * r1 * z)
        glEnd()

def draw_partial_textured_sphere_sectorized(radius=2, slices=16, stacks=16):
    lon_per_sector = 360.0 / NUM_EQUATOR_SECTORS
    for i in range(NUM_EQUATOR_SECTORS):
        lon_min = i * lon_per_sector - 180
        lon_max = (i + 1) * lon_per_sector - 180
        glBindTexture(GL_TEXTURE_2D, sector_textures[i])
        draw_sphere_sector(radius, slices, stacks, -equator_lat_extent, equator_lat_extent, lon_min, lon_max)
    glBindTexture(GL_TEXTURE_2D, 0)

def draw_pole_cap(radius=2, slices=60, rings=30, north=True, texture_radius=0.5, rotation_angle_deg=0.0):
    rotation_rad = np.radians(rotation_angle_deg)

    for i in range(rings):
        r0 = texture_radius * (i / rings)
        r1 = texture_radius * ((i + 1) / rings)

        theta0 = (np.pi / 2) * (1 - r0) if north else (-np.pi / 2) * (1 - r0)
        theta1 = (np.pi / 2) * (1 - r1) if north else (-np.pi / 2) * (1 - r1)

        y0 = radius * np.sin(theta0)
        y1 = radius * np.sin(theta1)
        rad0 = radius * np.cos(theta0)
        rad1 = radius * np.cos(theta1)

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            phi = 2 * np.pi * j / slices
            x = np.cos(phi)
            z = np.sin(phi)

            uv_phi0 = phi + rotation_rad
            uv_phi1 = phi + rotation_rad

            crop_scale = pole_crop   # User-defined parameter

            # Offset to center crop
            offset = (1.0 - crop_scale) / 2.0

            u0 = offset + crop_scale * (0.5 + 0.5 * np.cos(uv_phi0) * r0)
            v0 = offset + crop_scale * (0.5 + 0.5 * np.sin(uv_phi0) * r0)

            u1 = offset + crop_scale * (0.5 + 0.5 * np.cos(uv_phi1) * r1)
            v1 = offset + crop_scale * (0.5 + 0.5 * np.sin(uv_phi1) * r1)

            glTexCoord2f(u0, v0)
            glVertex3f(rad0 * x, y0, rad0 * z)

            glTexCoord2f(u1, v1)
            glVertex3f(rad1 * x, y1, rad1 * z)
        glEnd()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glRotatef(camera_elevation, 1.0, 0.0, 0.0)
    glRotatef(camera_azimuth, 0.0, 1.0, 0.0)
    gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0)

    draw_partial_textured_sphere_sectorized(radius=2, slices=16, stacks=16)

    glBindTexture(GL_TEXTURE_2D, texture_north)
    draw_pole_cap(radius=2.1, slices=60, rings=30, north=True,
    texture_radius=1.0 * ((90 - equator_lat_extent) / 90),
    rotation_angle_deg=north_rotation_angle)

    glBindTexture(GL_TEXTURE_2D, texture_south)
    draw_pole_cap(radius=2.1, slices=60, rings=30, north=False,
    texture_radius=1.0 * ((90 - equator_lat_extent) / 90),
    rotation_angle_deg=south_rotation_angle)


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
    
    if key == b'q':
        south_rotation_angle = (south_rotation_angle - 10) % 360
    elif key == b'w':
        south_rotation_angle = (south_rotation_angle + 10) % 360
    elif key == b'e':
        north_rotation_angle = (north_rotation_angle - 10) % 360
    elif key == b'r':
        north_rotation_angle = (north_rotation_angle + 10) % 360
    elif key == b'-':
        equator_lat_extent = max(10, equator_lat_extent - 5)
    elif key == b'+':
        equator_lat_extent = min(90, equator_lat_extent + 5)
    elif key == b'\x1b':  # ESC
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