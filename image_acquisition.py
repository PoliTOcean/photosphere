import cv2
import os
import numpy as np

PATH = "photos/"  # Replace with your images path


def stitch_images(image_paths, output_path="output.jpg"):
    images = []

    # Load images
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading image: {path}")
            return
        else: images.append(img)

    # Create stitcher object and stitch images
    stitcher = cv2.Stitcher.create(mode=cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images) #merge the photos only if there is at least the 30% of overlap between 2 photos

    if status == cv2.Stitcher_OK:
        path = PATH
        # Try to add boat.jpg to the right of the panorama
        boat_path = os.path.join(path, "boat.jpg")
        boat_img = cv2.imread(boat_path)
        if boat_img is not None:
            try:
                # Resize boat to match panorama height
                pano_h, pano_w = panorama.shape[:2]
                boat_h, boat_w = boat_img.shape[:2]
                if boat_h != pano_h:
                    scale = pano_h / boat_h * 0.75
                    boat_img = cv2.resize(boat_img, (int(boat_w * scale), pano_h))
                    boat_h, boat_w = boat_img.shape[:2]
                # Try to blend right edge of panorama with left edge of boat
                blend_width = min(100, pano_w, boat_w)  # Blend over 100px or less if images are small
                if blend_width > 0:
                    # Linear alpha blend
                    alpha = np.linspace(1, 0, blend_width).reshape(1, -1, 1)
                    pano_edge = panorama[:, -blend_width:]
                    boat_edge = boat_img[:, :blend_width]
                    blended = (pano_edge * alpha + boat_edge * (1 - alpha)).astype(np.uint8)
                    # Concatenate: panorama (except last blend_width), blended, boat (except first blend_width)
                    final_img = np.concatenate([
                        panorama[:, :-blend_width],
                        blended,
                        boat_img[:, blend_width:]
                    ], axis=1)
                else:
                    # Fallback: just concatenate
                    final_img = np.concatenate([panorama, boat_img], axis=1)
                cv2.imwrite(output_path, final_img)
                print(f"Panorama with boat saved as {output_path}")
            except Exception as e:
                print(f"Blending failed: {e}, concatenating instead.")
                try:
                    final_img = np.concatenate([panorama, boat_img], axis=1)
                    cv2.imwrite(output_path, final_img)
                    print(f"Panorama with boat saved as {output_path}")
                except Exception as e2:
                    print(f"Concatenation failed: {e2}")
                    cv2.imwrite(output_path, panorama)
                    print(f"Panorama saved as {output_path}")
        else:
            cv2.imwrite(output_path, panorama)
            print(f"Panorama saved as {output_path}")
    else:
        print("Error: Unable to stitch images")


if __name__ == "__main__":
    path = PATH #Replace with your images path
    valid_extensions = ['.jpg', '.jpeg']
    image_paths = [path + f for f in sorted(os.listdir(path)) 
                  if os.path.isfile(os.path.join(path, f)) and 
                  os.path.splitext(f.lower())[1] in valid_extensions and
                  not any(x in f.lower() for x in ["south", "north", "stitched", "boat"])]
    
    stitch_images(image_paths, path + "stitched.jpg")
