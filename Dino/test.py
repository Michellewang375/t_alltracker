#!/usr/bin/env python3
import os
import subprocess
import glob
import shutil
import pickle
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA
from scipy import signal
import matplotlib.pyplot as plt
import requests
import os
import requests

import gdown
from pathlib import Path

CHECKPOINT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
GDRIVE_FILE_ID = "1OGQbGfSHRr3npK9AONpx96Xe2s06oWg1"

if not CHECKPOINT_PATH.exists():
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading checkpoint from Google Drive to {CHECKPOINT_PATH}...")
    gdown.download(url, str(CHECKPOINT_PATH), quiet=False)
else:
    print(f"Checkpoint already exists: {CHECKPOINT_PATH}")

hub_cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(hub_cache_dir, exist_ok=True)
dest_path = os.path.join(hub_cache_dir, CHECKPOINT_PATH.name)
if not os.path.exists(dest_path):
    shutil.copy2(CHECKPOINT_PATH, dest_path)
    print(f"Copied checkpoint to hub cache: {dest_path}")



# -----------------------------
# Config
# -----------------------------
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Paths
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "dino_img"
LOCAL_DINOV3_DIR = BASE_DIR / "dinov3_local"
CHECKPOINT_FILE = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
CHECKPOINT_PATH = BASE_DIR / CHECKPOINT_FILE
PKL_FILE = BASE_DIR / "fg_classifier.pkl"

# Google Drive checkpoint file ID
GDRIVE_FILE_ID = "1OGQbGfSHRr3npK9AONpx96Xe2s06oWg1"

# -----------------------------
# Helpers
# -----------------------------
def download_from_gdrive(file_id, dest_path):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print(f"Downloaded checkpoint to {dest_path}")


def resize_transform(img: Image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    w, h = img.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(img, (h_patches * patch_size, w_patches * patch_size)))


def load_and_preprocess_images(images_dir: str):
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    processed_images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_resized = resize_transform(img)
        img_resized_norm = TF.normalize(img_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        processed_images.append(img_resized_norm)

    return torch.stack(processed_images)


# -----------------------------
# Ensure DINOv3 repo exists
# -----------------------------
if not LOCAL_DINOV3_DIR.exists():
    print(f"Local DINOv3 repo not found. Cloning into {LOCAL_DINOV3_DIR}...")
    subprocess.run(
        ["git", "clone", "https://github.com/facebookresearch/dinov3.git", str(LOCAL_DINOV3_DIR)],
        check=True
    )
else:
    print(f"Using existing local DINOv3 clone at {LOCAL_DINOV3_DIR}")

# -----------------------------
# Ensure checkpoint exists
# -----------------------------
if not CHECKPOINT_PATH.exists():
    download_from_gdrive(GDRIVE_FILE_ID, CHECKPOINT_PATH)

hub_cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(hub_cache_dir, exist_ok=True)
dest_path = os.path.join(hub_cache_dir, CHECKPOINT_FILE)
if not os.path.exists(dest_path):
    shutil.copy2(CHECKPOINT_PATH, dest_path)
    print(f"Copied checkpoint to hub cache: {dest_path}")

# -----------------------------
# Load DINOv3 model
# -----------------------------
MODEL_NAME = "dinov3_vits16"
print(f"Loading DINOv3 model '{MODEL_NAME}' from local repo...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load(
    repo_or_dir=str(LOCAL_DINOV3_DIR),
    model=MODEL_NAME,
    source="local"
)
model.eval()
model = model.to(device)
print(f"Model loaded: {MODEL_NAME}, embed_dim={model.embed_dim}")

# -----------------------------
# Load foreground classifier
# -----------------------------
with open(PKL_FILE, 'rb') as f:
    clf = pickle.load(f)

# -----------------------------
# Load images
# -----------------------------
images_tensor = load_and_preprocess_images(IMAGES_DIR)
image_resized_norm = images_tensor[0].to(device)  # pick first image

h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized_norm.shape[1:]]

# -----------------------------
# Extract features
# -----------------------------
MODEL_TO_NUM_LAYERS = {
    'dinov3_vits16': 12,
    'dinov3_vitsp16': 12,
    'dinov3_vitb16': 12,
    'dinov3_vitl16': 24,
    'dinov3_vithp14': 32,
    'dinov3_vit7b14': 40,
}
n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

with torch.inference_mode():
    feats = model.get_intermediate_layers(
        image_resized_norm.unsqueeze(0),
        n=range(n_layers),
        reshape=True,
        norm=True
    )
    x = feats[-1].squeeze().detach().cpu()
    dim = x.shape[0]
    x = x.view(dim, -1).permute(1, 0)

# -----------------------------
# Compute foreground score
# -----------------------------
fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

# -----------------------------
# Plot original image & foreground
# -----------------------------
first_image_path = sorted(
    glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
    glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
)[0]
image = Image.open(first_image_path).convert("RGB")

plt.rcParams.update({
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.labelsize": 5,
    "axes.titlesize": 4,
})

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

# -----------------------------
# PCA projection for foreground patches
# -----------------------------
foreground_selection = fg_score_mf.view(-1) > 0.5
fg_patches = x[foreground_selection]

pca = PCA(n_components=3, whiten=True)
pca.fit(fg_patches)
projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)

projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
projected_image *= (fg_score_mf.unsqueeze(0) > 0.5)

plt.figure(dpi=300)
plt.imshow(projected_image.permute(1, 2, 0))
plt.axis('off')
plt.show()
