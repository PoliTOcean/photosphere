import cv2
import os


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
    stitcher = cv2.Stitcher.create()
    status, panorama = stitcher.stitch(images) #merge the photos only if there is at least the 30% of overlap between 2 photos

    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, panorama)
        print(f"Panorama saved as {output_path}")
    else:
        print("Error: Unable to stitch images")


if __name__ == "__main__":
    path = "D:\\Produzioni e progetti\\PoliTOcean\\Sphere rendering\\photos\\" #Replace with your images path
    image_paths = os.listdir(path)
    image_paths = [path + p for p in image_paths] 
    stitch_images(image_paths, path + "stitched.jpg")
