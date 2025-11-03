#!/usr/bin/env python3
"""
Standalone DINOv3 pipeline.
Loads images, DINOv3 model, foreground classifier, computes patch features,
and visualizes results, all using local files.
"""

import os
import glob
import pickle
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import signal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


import shutil
import os

CHECKPOINT_FILE = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
CHECKPOINT_PATH = Path(__file__).parent / CHECKPOINT_FILE

# Make sure torch hub checkpoints directory exists
hub_cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(hub_cache_dir, exist_ok=True)

# Move checkpoint into hub cache if not already there
dest_path = Path(hub_cache_dir) / CHECKPOINT_FILE
if not dest_path.exists() and CHECKPOINT_PATH.exists():
    shutil.copy2(CHECKPOINT_PATH, dest_path)
    print(f"Copied checkpoint to hub cache: {dest_path}")
elif dest_path.exists():
    print(f"Checkpoint already exists in hub cache: {dest_path}")
else:
    raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")


# --------------------------
# Config
# --------------------------
PATCH_SIZE = 16
IMAGE_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SCRIPT_DIR = Path(__file__).parent.resolve()
IMAGES_DIR = SCRIPT_DIR / "dino_img"
PICKLE_PATH = SCRIPT_DIR / "fg_classifier.pkl"

# Use local DINOv3 clone (must exist)
LOCAL_DINOV3_DIR = SCRIPT_DIR / "dinov3_local"
MODEL_NAME = "dinov3_vits16"

if not LOCAL_DINOV3_DIR.exists():
    raise FileNotFoundError(f"Local DINOv3 repo not found: {LOCAL_DINOV3_DIR}. "
                            f"Clone it via: git clone https://github.com/facebookresearch/dinov3.git {LOCAL_DINOV3_DIR}")

# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Image preprocessing
# --------------------------
def resize_transform(img: Image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    w, h = img.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(img, (h_patches * patch_size, w_patches * patch_size)))

def load_and_preprocess_images(images_dir: Path):
    image_paths = sorted(list(images_dir.glob("*.png")) +
                         list(images_dir.glob("*.jpg")) +
                         list(images_dir.glob("*.jpeg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    processed_images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_resized = resize_transform(img)
        img_resized_norm = TF.normalize(img_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        processed_images.append(img_resized_norm)

    return torch.stack(processed_images)

# --------------------------
# Load DINOv3 model (local)
# --------------------------
print(f"Loading DINOv3 model '{MODEL_NAME}' from local clone...")
model = torch.hub.load(
    repo_or_dir=str(LOCAL_DINOV3_DIR),
    model=MODEL_NAME,
    source="local",
)
model.eval()
model = model.to(device)

MODEL_TO_NUM_LAYERS = {
    'dinov3_vits16': 12,
    'dinov3_vitsp16': 12,
    'dinov3_vitb16': 12,
    'dinov3_vitl16': 24,
    'dinov3_vithp14': 32,
    'dinov3_vit7b14': 40,
}
n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

# --------------------------
# Load foreground classifier
# --------------------------
if not PICKLE_PATH.exists():
    raise FileNotFoundError(f"Pickle file not found: {PICKLE_PATH}")
with open(PICKLE_PATH, 'rb') as f:
    clf = pickle.load(f)

# --------------------------
# Load and preprocess images
# --------------------------
images_tensor = load_and_preprocess_images(IMAGES_DIR)
image_resized_norm = images_tensor[0].to(device).unsqueeze(0)  # pick first image
h_patches, w_patches = [int(d / PATCH_SIZE) for d in images_tensor[0].shape[1:]]

# --------------------------
# Compute DINOv3 features
# --------------------------
with torch.inference_mode():
    autocast_ctx = torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
    with autocast_ctx:
        feats = model.get_intermediate_layers(image_resized_norm, n=range(n_layers), reshape=True, norm=True)
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

# --------------------------
# Compute foreground score
# --------------------------
fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

# --------------------------
# PCA projection
# --------------------------
foreground_selection = fg_score_mf.view(-1) > 0.5
fg_patches = x[foreground_selection]

pca = PCA(n_components=3, whiten=True)
pca.fit(fg_patches)
projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)
projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
projected_image *= (fg_score_mf.unsqueeze(0) > 0.5)

# --------------------------
# Plot results
# --------------------------
first_image_path = sorted(list(IMAGES_DIR.glob("*.*")))[0]
image = Image.open(first_image_path).convert("RGB")

plt.rcParams.update({"xtick.labelsize": 5, "ytick.labelsize": 5, "axes.labelsize": 5, "axes.titlesize": 4})
plt.figure(figsize=(4, 2), dpi=300)

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')
plt.title(f"Image, Size {image.size}")

plt.subplot(1, 2, 2)
plt.imshow(fg_score_mf)
plt.title(f"Foreground Score, Size {tuple(fg_score_mf.shape)}")
plt.colorbar()
plt.axis('off')
plt.show()

# PCA visualization
plt.figure(dpi=300)
plt.imshow(projected_image.permute(1, 2, 0))
plt.axis('off')
plt.show()

print("Done!")
