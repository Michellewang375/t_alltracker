#---------------------------IMPORTS------------------------------
import os
import sys
import cv2
import numpy as np
import torch
from types import SimpleNamespace
import sqlite3

# Import alltracker
from all_t_git.nets.alltracker import Net
from all_t_git.demo import run, read_image_folder, forward_video

# import db
from Database.Db import COLMAPDatabase


#---------------------------CONFIG------------------------------
# Add all_t_git folder to Python path
sys.path.append(os.path.abspath("./all_t_git"))
RAW_IMG_DIR = "test_sin"
DB_PATH = "./Database/alltracker.db"
VIS_OUT_DIR = "./Database"
os.makedirs(VIS_OUT_DIR, exist_ok=True)

#---------------------------ALLTRACKER-----------------------------
# Run AllTracker on all frames in input_dir using the same API as demo.py
#  Returns a dictionary of keypoints per frame (or trajectories)
def run_alltracker(model, input_dir, args):
    # Load images
    rgbs, framerate = read_image_folder(
        input_dir, image_size=args.image_size, max_frames=args.max_frames
    )

    if len(rgbs) == 0:
        print("[AllTracker] No images found in", input_dir)
        return {}

    # Forward through AllTracker
    print("[AllTracker] Running forward pass...")
    with torch.no_grad():
        traj_maps, vis_maps, *_ = model.forward_sliding(
            rgbs, iters=args.inference_iters, window_len=args.window_len
        )

    # Convert output to per-frame keypoints (rough)
    results = {}
    num_frames = traj_maps.shape[1]  # T
    for t in range(num_frames):
        traj_frame = traj_maps[0, t].cpu().numpy()  # HxW
        yx = np.argwhere(traj_frame > 0)  # list of (y, x)
        keypoints = yx[:, ::-1]  # convert to (x, y)
        descriptors = np.ones((len(keypoints), 32), dtype=np.float32)  # placeholder
        results[f"frame_{t:04d}.png"] = (keypoints, descriptors)

    print(f"[AllTracker] Extracted keypoints for {len(results)} frames.")
    return results


#---------------------------ORB TRACKING------------------------------
#Use ORB to refine and track keypoints between consecutive frames.
def orb_track(results, input_dir):
    orb = cv2.ORB_create(nfeatures=1000)
    tracked_results = {}
    
    filenames = sorted(list(results.keys()))
    prev_kps, prev_desc = None, None
    prev_img = None

    for i, filename in enumerate(filenames):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        kps_all, desc_all = results[filename]
        kps_all_cv2 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in kps_all]

        # compute ORB descriptors using AllTracker keypoints
        kps_refined, desc_refined = orb.compute(img, kps_all_cv2)
        if desc_refined is None:
            desc_refined = np.zeros((len(kps_refined), 32), dtype=np.uint8)

        # if previous frame exists, match to track
        if prev_kps is not None and prev_desc is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(prev_desc, desc_refined)
            matches = sorted(matches, key=lambda x: x.distance)
            print(f"[ORB Track] {len(matches)} matches between frames {filenames[i-1]} and {filename}")

        # update
        tracked_results[filename] = (
            np.array([[kp.pt[0], kp.pt[1]] for kp in kps_refined], dtype=np.float32),
            desc_refined
        )
        prev_kps, prev_desc, prev_img = kps_refined, desc_refined, img

    return tracked_results


#---------------------------SAVE TO DATABASE------------------------------
#Store refined keypoints and descriptors in database.
def save_to_database(db_path, results):
    db = COLMAPDatabase.connect(db_path)
    db.clear_keypoints()
    db.clear_descriptors()
    db.create_tables() 

    for i, (filename, (kps, desc)) in enumerate(results.items()):
        image_id = i + 1
        db.add_keypoints(image_id, kps)
        db.add_descriptors(image_id, desc)
        print(f"[DB] Added {filename} (ID {image_id})")

    db.commit()
    db.close()
    print("[DB] Successfully saved all features.")


#---------------------------VISUALIZE DB------------------------------
def visualize_from_db(db_path, img_dir, out_dir):
    db = COLMAPDatabase.connect(db_path)
    keypoints_dict = db.read_all_keypoints()

    for image_id, keypoints in keypoints_dict.items():
        img_name = f"{str(image_id).zfill(3)}.png"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        for (x, y) in keypoints[:, :2]:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        cv2.imwrite(os.path.join(out_dir, f"vis_{img_name}"), img)
        print(f"[VIS] Saved vis_{img_name}")

    db.close()
    print("[VIS] Visualization done.")


#---------------------------PIPLINE------------------------------
# 0. Alltracker model
def main():
    print("Creating AllTracker model")
    args = SimpleNamespace(
        ckpt_init="./checkpoints/alltracker.pth",
        image_folder=RAW_IMG_DIR,
        query_frame=0,
        image_size=1024,
        max_frames=400,
        inference_iters=4,
        window_len=16,
        rate=2,
        conf_thr=0.1,
        bkg_opacity=0.5,
        vstack=False,
        hstack=False,
        tiny=False,
        mask=None
    )

# 0. DB tables
    MAX_IMAGE_ID = 2**31 - 1
    CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
        camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        model INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params BLOB,
        prior_focal_length INTEGER NOT NULL
    );"""

    CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name TEXT NOT NULL UNIQUE,
        camera_id INTEGER NOT NULL,
        CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
    );""".format(MAX_IMAGE_ID)

    CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
    );"""

    CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
    );"""

    CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB
    );"""

    CREATE_TWO_VIEW_GEOMETRIES_TABLE = """CREATE TABLE IF NOT EXISTS two_view_geometries (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        config INTEGER NOT NULL,
        F BLOB,
        E BLOB,
        H BLOB,
        qvec BLOB,
        tvec BLOB
    );"""

    CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
        image_id INTEGER PRIMARY KEY NOT NULL,
        position BLOB,
        coordinate_system INTEGER NOT NULL,
        position_covariance BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
    );"""

    CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);"

    CREATE_ALL = "; ".join([
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_POSE_PRIORS_TABLE,
        CREATE_NAME_INDEX
    ])
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(CREATE_ALL)
    conn.commit()
    conn.close()

    # create model
    model = Net(args.window_len)
    model.cuda()
    for p in model.parameters():
        p.requires_grad = False

    # load checkpoint
    if args.ckpt_init:
        checkpoint = torch.load(args.ckpt_init, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"[MODEL] Loaded checkpoint from {args.ckpt_init}")
    else:
        print("[MODEL] Using default weights")

# 1. all tracker
    print("Running AllTracker")
    alltracker_results = run_alltracker(model, RAW_IMG_DIR, args)

# 2. orb tracking
    print("ORB tracking")
    orb_results = orb_track(alltracker_results, RAW_IMG_DIR)

# 3. saving to db
    print("Saving to database")
    save_to_database(DB_PATH, orb_results)

# 4. visulization
    print("Visualizing from DB")
    visualize_from_db(DB_PATH, RAW_IMG_DIR, VIS_OUT_DIR)

if __name__ == "__main__":
    main()
