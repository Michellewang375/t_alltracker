#Extracts ORB features and feature descriptors, and put them in respective folders
import os
import numpy as np
import cv2
from PIL import Image

# --- CONFIG ---
input_folder = "./All_t/masked"   # folder with images
output_kp_folder = "./Database/key_npy"
output_desc_folder = "./Database/desc_npy"
resize_to = None  # (width, height) or None to skip
feat_shape = 6    # must match your DB

os.makedirs(output_kp_folder, exist_ok=True)
os.makedirs(output_desc_folder, exist_ok=True)

# --- ORB extractor ---
orb = cv2.ORB_create(nfeatures=1000)

# --- PROCESS IMAGES ---
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path).convert('RGB')
    if resize_to:
        img = img.resize(resize_to)
    img_array = np.array(img)

    # convert to grayscale for ORB
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # detect keypoints and descriptors
    kps = orb.detect(gray, None)
    kps, desc = orb.compute(gray, kps)

    if kps is None or len(kps) == 0:
        print(f"No keypoints found for {filename}")
        continue

    # convert keypoints to Nx2 array (x, y)
    kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)

    # pad columns if needed
    if kp_array.shape[1] < feat_shape:
        padding = np.zeros((kp_array.shape[0], feat_shape - kp_array.shape[1]), dtype=np.float32)
        kp_array = np.hstack((kp_array, padding))

    # ensure correct type
    kp_array = kp_array.astype(np.float32)

    # save
    kp_npy = os.path.splitext(filename)[0] + "_kps.npy"
    np.save(os.path.join(output_kp_folder, kp_npy), kp_array)

    # save descriptors (Nx32 for ORB)
    if desc is not None:
        desc_npy = os.path.splitext(filename)[0] + "_desc.npy"
        np.save(os.path.join(output_desc_folder, desc_npy), desc)
    else:
        print(f"No descriptors found for {filename}")

    print(f"Saved {kp_npy} and descriptors for {filename}")
