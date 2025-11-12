#coverts images to numpy format

import os
import numpy as np
from PIL import Image

# --- CONFIG ---
input_folder = "./Database/keypoints"      # in folder
output_folder = "npy_files"  # out folder
resize_to = (None)       # optional: resize images, set None to skip

os.makedirs(output_folder, exist_ok=True)

# --- PROCESS IMAGES ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('RGB')  # convert to RGB
        if resize_to:
            img = img.resize(resize_to)
        img_array = np.array(img)

        # save as .npy
        npy_filename = os.path.splitext(filename)[0] + '.npy'
        np.save(os.path.join(output_folder, npy_filename), img_array)

        print(f"Saved {npy_filename}")
