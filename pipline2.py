# how to run:
#1. conda activate alltracker
#2. python pipline.py

#---------------------------IMPORTS------------------------------
import os
import sys
import cv2
import numpy as np
import torch
from types import SimpleNamespace
import sqlite3
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from Database.visualize import visualize_from_db




#---------------------------CONFIG------------------------------
sys.path.append(os.path.abspath("./all_t_git"))
# Import alltracker after path is set
from nets.alltracker import Net
from demo import read_image_folder
# Import database module
from Database.Db import COLMAPDatabase


#---------------------------DIRECTORIES-------------------------
RAW_IMG_DIR = "test_sin"
MASK_ALL_DIR = "./testing2/all_fmap"
ALLT_IMG_DIR = "./testing2/masked"
DB_PATH = "./testing2/testing.db"
ORB_IMG_DIR = "./testing2/vis"
MASK_PATH = "/mnt/data1/michelle/t_alltracker/All_t/undistorted_mask.bmp"
PCA_DIR = "./testing2/pca"
os.makedirs(PCA_DIR, exist_ok=True)




#---------------------------ALLTRACKER-----------------------------
#import desired tracking algo --> using alltracker
def run_alltracker(model, input_dir, args):
    rgbs, framerate = read_image_folder(
        input_dir, image_size=args.image_size, max_frames=args.max_frames
    )
    if len(rgbs) == 0:
        print("[AllTracker] No images found in", input_dir)
        return {}, {}
    # Move to GPU and eval
    rgbs = rgbs.cuda()
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()
    # Forward through AllTracker (same as demo)
    print("[AllTracker] Running forward pass")
    with torch.no_grad():
        traj_maps, vis_maps, *_ = model.forward_sliding(
            rgbs, iters=args.inference_iters, window_len=args.window_len
        )
    # Convert outputs
    traj_maps = traj_maps[0].cpu().numpy()   # (T, 2, H, W)
    vis_maps  = vis_maps[0].cpu().numpy()    # (T, 1, H, W)
    T, _, H, W = traj_maps.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    base_pts = np.stack([xs.flatten(), ys.flatten()], axis=-1).astype(np.float32)  # (H*W, 2)
    results = {}
    correspondences = {}

    for t in range(T):
        flow = traj_maps[t]        # (2, H, W)
        conf = vis_maps[t, 0]      # (H, W)
        mask = conf > args.conf_thr

        # keypoints in frame t (x,y)
        ft_xy = np.stack([xs[mask], ys[mask]], axis=1).astype(np.float32)

        # corresponding coordinates
        flow_x = flow[0]
        flow_y = flow[1]
        f0_xy = np.stack([
            xs[mask] - flow_x[mask],
            ys[mask] - flow_y[mask]
        ], axis=1).astype(np.float32)

        # indices into flattened base_pts
        idx0 = np.flatnonzero(mask.flatten()).astype(np.uint32)
        idxt = np.arange(len(ft_xy), dtype=np.uint32)

        # Save keypoints (descriptors r empty)
        fname = f"frame_{t:04d}.png"
        results[fname] = (ft_xy, np.zeros((len(ft_xy), 32), dtype=np.uint8))

        # Save correspondence
        correspondences[t] = {
            "idx0": idx0,
            "idxt": idxt,
            "f0_xy": f0_xy,
            "ft_xy": ft_xy,
            "mask": mask  # keep 2D
        }
    print(f"[AllTracker] finished.")
    return results, correspondences

#---------------------------ORB TRACKING------------------------------
#Use ORB to refine and track keypoints and descriptors between consecutive frames
def orb_track(results, input_dir, mask=None, max_kp=5000, image_size=1024):
    tracked_results = {}
    orb = cv2.ORB_create(nfeatures=max_kp)
    filenames = sorted(list(results.keys()))

    # Resize mask if provided
    mask_r_dict = {}
    if mask is not None:
        for filename in filenames:
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if mask.shape != img.shape:
                mask_r_dict[filename] = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_r_dict[filename] = mask

    for filename in filenames:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ORB Track]: Could not read {filename}, skipping")
            continue

        # resize
        H0, W0 = img.shape[:2]
        scale = min(image_size / H0, image_size / W0)
        H = int((H0 * scale) // 8 * 8)
        W = int((W0 * scale) // 8 * 8)
        img_r = cv2.resize(img, (W, H))

        kps_all, desc_all = results[filename]
        kps_all = np.array(kps_all, dtype=np.float32)

        # Filter keypoints using mask
        if mask is not None:
            mask_r = mask_r_dict[filename]
            keep = [
                i for i, (x, y) in enumerate(kps_all)
                if 0 <= int(round(x)) < mask_r.shape[1] and 0 <= int(round(y)) < mask_r.shape[0] and mask_r[int(round(y)), int(round(x))] > 0
            ]
            kps_all = kps_all[keep]
            desc_all = desc_all[keep]

        # convert to cv2 KeyPoints
        kps_cv2 = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kps_all] if len(kps_all) else []

        # subsample
        if len(kps_cv2) > max_kp:
            idxs = np.linspace(0, len(kps_cv2)-1, max_kp).astype(int)
            kps_cv2 = [kps_cv2[i] for i in idxs]
        if kps_cv2:
            kps_refined, desc_refined = orb.compute(img_r, kps_cv2)
            if desc_refined is None:
                desc_refined = np.zeros((0, 32), dtype=np.uint8)
                kps_refined = []
            else:
                desc_refined = np.asarray(desc_refined, dtype=np.uint8)
                if desc_refined.ndim == 1:
                    desc_refined = desc_refined.reshape(1, -1)
        else:
            kps_refined, desc_refined = [], np.zeros((0, 32), dtype=np.uint8)

        # scale back to original image size
        if kps_refined:
            pts_orig = np.array([[kp.pt[0], kp.pt[1]] for kp in kps_refined], dtype=np.float32)
            pts_orig[:, 0] *= W0 / W
            pts_orig[:, 1] *= H0 / H
        else:
            pts_orig = np.zeros((0, 2), dtype=np.float32)

        tracked_results[filename] = (pts_orig, desc_refined)

    return tracked_results

#---------------------------SAVE TO DB------------------------------
#saving keypoints and descriptors to database
def save_to_db(db_path, results):
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()
    for filename, (kps, desc) in results.items():
        # add image
        try:
            db.add_image(name=filename, camera_id=1)
        except sqlite3.IntegrityError:
            pass
        image_id = db.execute("SELECT image_id FROM images WHERE name=?", (filename,)).fetchone()[0]

        # keypoints
        colmap_kps = normalize_to_colmap_kp(kps)
        db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id,))
        if colmap_kps.shape[0] > 0:
            db.add_keypoints(image_id, colmap_kps)

        # descriptors
        db.execute("DELETE FROM descriptors WHERE image_id=?", (image_id,))
        if desc is not None and desc.shape[0] > 0:
            db.add_descriptors(image_id, desc.astype(np.uint8))

    db.commit()
    db.close()
    print("[DB] Saved keypoints and descriptors")

#-------------------------VISUALIZE KEYPOINTS--------------------------
# visualzing keypoints in database 
def key_visualize(db_path, img_dir, out_dir, image_size=1024):
    os.makedirs(out_dir, exist_ok=True)
    db = COLMAPDatabase.connect(db_path)
    rows = db.execute("SELECT image_id, name FROM images").fetchall()
    image_map = {row[0]: row[1] for row in rows}
    keypoints_dict = db.read_all_keypoints()
    def preprocess_img(img):
        H0, W0 = img.shape[:2]
        scale = min(image_size / H0, image_size / W0)
        H = int(H0 * scale)
        W = int(W0 * scale)
        H = (H // 8) * 8
        W = (W // 8) * 8
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    for image_id, kps in keypoints_dict.items():
        if kps is None or len(kps) == 0:
            continue
        filename = image_map.get(image_id, None)
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_resized = preprocess_img(img)
        vis = img_resized.copy()
        kps_2d = np.array(kps, dtype=np.float32)[:, :2]
        for x, y in kps_2d:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        out_path = os.path.join(out_dir, f"vis_{filename}")
        cv2.imwrite(out_path, vis)
    db.close()
    print("[VIS] Done.")

#---------------------------FEATURE DESCRIPTOR------------------------------
# visualizing descriptors from database using PCA
def pca_visualization(db_path, filenames, pca_out_dir):
    os.makedirs(pca_out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    #select descriptors
    for fname in filenames:
        cursor.execute("SELECT image_id FROM images WHERE name=?", (fname,))
        row = cursor.fetchone()
        if row is None:
            print(f"[PCA] Skipping {fname}, no image in DB")
            continue
        image_id = row[0]
        cursor.execute(
            "SELECT rows, cols, data FROM descriptors WHERE image_id=?",
            (image_id,)
        )
        #finding descriptors
        row = cursor.fetchone()
        if row is None or row[2] is None:
            print(f"[PCA] Skipping {fname}, no descriptors found")
            continue
        rows, cols, blob = row
        desc = np.frombuffer(blob, dtype=np.uint8)  # OR float32 if saved as float
        if desc.size == 0:
            print(f"[PCA] Skipping {fname}, empty descriptor blob")
            continue
        if desc.size % cols != 0:
            print(f"[PCA] Warning {fname}: blob size {desc.size} not divisible by {cols}, truncating")
            desc = desc[:(desc.size // cols) * cols]

        rows_actual = desc.size // cols
        desc = desc.reshape(rows_actual, cols)

        # DEBUG: print first 5 descriptors to check values
        # print(f"[DEBUG] {fname}: first 5 descriptors (rows x cols = {desc.shape}):")
        # print(desc[:5])
        
        if np.var(desc) == 0:
            print(f"[PCA] Skipping {fname}, constant descriptors")
            continue
        # Map PCA to keypoints
        desc_f = desc.astype(np.float32) / 255.0
        cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,))
        row_kp = cursor.fetchone()
        if row_kp is None or row_kp[2] is None:
            print(f"[PCA] Skipping {fname}, no keypoints")
            continue
        kp_rows, kp_cols, kp_blob = row_kp
        kps = np.frombuffer(kp_blob, dtype=np.float32).reshape(kp_rows, kp_cols)

        #running PCA vis
        pca = PCA(n_components=3)
        projected = pca.fit_transform(desc_f)  # shape: (num_kps, 3)
        # normalize for RGB
        colors = (projected - projected.min(0)) / (np.ptp(projected, axis=0) + 1e-8)
        img_path = os.path.join(ALLT_IMG_DIR, fname) 
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        H_img, W_img = img.shape[:2]
        viz = np.zeros((H_img, W_img, 3), dtype=np.float32)
        # square size
        square_size = 5
        half = square_size // 2
        for i in range(kps.shape[0]):
            x, y = int(round(kps[i, 0])), int(round(kps[i, 1]))
            x0, x1 = max(x - half, 0), min(x + half + 1, W_img)
            y0, y1 = max(y - half, 0), min(y + half + 1, H_img)
            viz[y0:y1, x0:x1, :] = colors[i]

        plt.figure(figsize=(12, 8))
        plt.imshow(viz)
        plt.axis('off')
        out_path = os.path.join(pca_out_dir, f"{os.path.splitext(fname)[0]}_pca.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[PCA] Saved visualization to {out_path}")

#------------------------FEATURE CORRESPONDANCE#------------------------
# feature correspondance between images from database
def f_matches(
    img0_path,
    img1_path,
    kps0,
    kps1,
    matches,
    mask=None,
    max_vis=2000,
    out_path="vis_matches.png"
):
    # img loading
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    if img0 is None or img1 is None:
        print(f"[WARN] Could not read {img0_path} or {img1_path}")
        return

    # masking
    if mask is not None:
        # masking size and match imgs same size
        mask_r0 = cv2.resize(mask.astype(np.uint8), (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_r1 = cv2.resize(mask.astype(np.uint8), (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
        # both keypoints must be in mask
        keep = [
            i for i, (i0, i1) in enumerate(matches)
            if mask_r0[int(kps0[i0][1]), int(kps0[i0][0])] > 0 and
               mask_r1[int(kps1[i1][1]), int(kps1[i1][0])] > 0
        ]
        if len(keep) == 0:
            print("[WARN] No matches left after masking")
            return
        matches = matches[keep]

    # adjust how much to match here (max 2000)
    if matches.shape[0] > max_vis:
        idx = np.random.choice(matches.shape[0], max_vis, replace=False)
        matches = matches[idx]

    # visualize
    h = max(img0.shape[0], img1.shape[0])
    w = img0.shape[1] + img1.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:img0.shape[0], :img0.shape[1]] = img0
    canvas[:img1.shape[0], img0.shape[1]:] = img1
    for i0, i1 in matches:
        x0, y0 = kps0[i0]
        x1, y1 = kps1[i1]
        x1 += img0.shape[1]
        cv2.circle(canvas, (int(x0), int(y0)), 2, (0, 255, 0), -1)
        cv2.circle(canvas, (int(x1), int(y1)), 2, (0, 255, 0), -1)
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 1)

    cv2.imwrite(out_path, canvas)
    print(f"[VIS] Saved {out_path}")




#------------------------HELPER FUNCTIONS------------------------

#resizing img for alltracker, same as alltracker's read_img_folder
def img_resize(img, image_size=1024):
    import cv2
    H0, W0 = img.shape[:2]
    scale = min(image_size / H0, image_size / W0)
    H = int(H0 * scale)
    W = int(W0 * scale)
    H = (H // 8) * 8
    W = (W // 8) * 8
    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img_resized

# visualzing keypoints after orb is ran
def visualize_keypoints(img_path, kps, mask=None, out_path=None, show=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path}")
        return

    vis = img.copy()

    # Draw keypoints
    for x, y in kps:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Apply mask if available
    if mask is not None:
        mask_r = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        vis[mask_r == 0] = 0

    # Save to file
    if out_path is not None:
        cv2.imwrite(out_path, vis)
        print(f"[VIS] Saved masked keypoints to {out_path} ({len(kps)} points)")

    # Show image
    if show:
        # Convert BGR -> RGB for matplotlib
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return vis

# return Nx6 array for COLMAP, used for saving keypoint data to database
def normalize_to_colmap_kp(pts):
    if pts.shape[1] == 2:  
        N = pts.shape[0]
        scales = np.ones((N, 1), np.float32)
        orientations = np.zeros((N, 1), np.float32)
        a4 = np.zeros((N, 1), np.float32)
        scores = np.ones((N, 1), np.float32)
        return np.hstack([pts, scales, orientations, a4, scores]).astype(np.float32)
    return pts.astype(np.float32)




#---------------------------MAIN PIPLINE------------------------------
def main():
    # 0. creating db and loading in Alltracker model
    print("0. Creating Database and loading in Alltracker")
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
    image_id INTEGER NOT NULL UNIQUE,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
    """

    CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER NOT NULL UNIQUE,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
    """

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
        #print(f"[MODEL] Loaded checkpoint from {args.ckpt_init}")
    else:
        print("[MODEL] Using default weights")

 # 1. Running Alltracker and apply masking
    print("\n1. Running AllTracker")
    alltracker_results_and_corr = run_alltracker(model, RAW_IMG_DIR, args)
    results, correspondences = alltracker_results_and_corr
    #masking
    os.makedirs(ALLT_IMG_DIR, exist_ok=True)
    # load mask
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("[WARN] Mask not found, skipping image masking")
    else:
        mask = (mask > 0).astype(np.uint8)
        for fname in results.keys():
            in_path = os.path.join(MASK_ALL_DIR, fname)
            out_path = os.path.join(ALLT_IMG_DIR, fname)
            img = cv2.imread(in_path)
            if img is None:
                print(f"[WARN] Could not read {in_path}")
                continue
            # resize mask if needed
            if mask.shape[:2] != img.shape[:2]:
                mask_r = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_r = mask
            masked_img = img.copy()
            masked_img[mask_r == 0] = 0
            cv2.imwrite(out_path, masked_img)

# 2. Orb and masking
    print("\n2. ORB and masking")
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)
    else:
        print("[WARN] Mask not found, skipping masking")
    orb_results = orb_track(results, ALLT_IMG_DIR, mask=mask, max_kp=5000)
    #visualize
    vis_dir = os.path.join(os.path.dirname(DB_PATH), "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for fname, (kps, _) in orb_results.items():
        out_path = os.path.join(vis_dir, f"vis_{fname}")
        visualize_keypoints(
            img_path=os.path.join(ALLT_IMG_DIR, fname),
            kps=kps,
            mask=mask,
            out_path=out_path,
            show=True
        )
        
# 3. Saving to database
    print("\n3. Saving to database")
    save_to_db(DB_PATH, orb_results)

# 4. Visualize feature descirptor using PCA
    print("\n4. Feature Desc w/ PCA")
    pca_visualization(DB_PATH, list(orb_results.keys()), PCA_DIR)

# 5. feature correspondance with masking
    print("\n5. Visualizing feature correspondences with masking")
    results, correspondences = alltracker_results_and_corr
    matches_dir = os.path.join(os.path.dirname(DB_PATH), "matches")
    os.makedirs(matches_dir, exist_ok=True)
    # loading mask
    mask = None
    if MASK_PATH is not None and os.path.exists(MASK_PATH):
        mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = (mask > 0).astype(np.uint8)
        else:
            print("[WARN] Mask not found or invalid")
    frame_names = sorted(results.keys())
    for i in range(len(frame_names) - 1):
        fname0 = frame_names[i]
        fname1 = frame_names[i + 1]
        kps0, _ = results[fname0]
        kps1, _ = results[fname1]
        # build matches
        corr = correspondences.get(i+1, None)
        if corr is None:
            continue
        matches = np.stack([corr["idx0"], corr["idxt"]], axis=1).astype(np.int32)
        # only valid matches
        valid = (matches[:,0] < len(kps0)) & (matches[:,1] < len(kps1))
        matches = matches[valid]
        
        img0_path = os.path.join(ORB_IMG_DIR, f"vis_{fname0}")
        img1_path = os.path.join(ORB_IMG_DIR, f"vis_{fname1}")
        out_path = os.path.join(matches_dir, f"all_match_{i:04d}.png")

        f_matches(
            img0_path=img0_path,
            img1_path=img1_path,
            kps0=kps0,
            kps1=kps1,
            matches=matches,
            mask=mask,
            max_vis=2000,
            out_path=out_path
        )   
if __name__ == "__main__":
    main()