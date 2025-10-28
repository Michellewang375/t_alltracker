import os
import argparse
import numpy as np
from PIL import Image
import glob

def convert_images_to_npy(input_dir, output_dir):
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all jpg images in the input directory
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    if not image_paths:
        print(f"No .jpg images found in {input_dir}")
        return

    for img_path in image_paths:
        # Load image in grayscale
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        # Create output filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        npy_path = os.path.join(output_dir, f"{base_name}.npy")

        # Save as .npy
        np.save(npy_path, img_array)
        print(f"Saved {npy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JPG images to NPY arrays.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input JPG images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save NPY files")
    args = parser.parse_args()

    convert_images_to_npy(args.input_dir, args.output_dir)
