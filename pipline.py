#---------------------------IMPORTS------------------------------
import os
import sys
import cv2
import numpy as np
import torch
from types import SimpleNamespace
import sqlite3
import matplotlib.pyplot as plt
#from Database.visualize import visualize_from_db




#---------------------------CONFIG------------------------------
sys.path.append(os.path.abspath("./all_t_git"))
# Import alltracker after path is set
from nets.alltracker import Net
from demo import run, read_image_folder, forward_video

# Import database module
from Database.Db import COLMAPDatabase




#---------------------------DIRECTORIES-------------------------
# "./All_t/masked"
RAW_IMG_DIR = "test_sin"
ALLT_IMG_DIR = "./All_t/masked"
DB_PATH = "./Database/alltracker.db"
ORB_IMG_DIR = "./Database/vis"
MASK_PATH = "/mnt/data1/michelle/t_alltracker/All_t/undistorted_mask.bmp"






#---------------------------ALLTRACKER-----------------------------
# def run_alltracker(model, input_dir, args):
#     # Load images
#     rgbs, framerate = read_image_folder(
#         input_dir, image_size=args.image_size, max_frames=args.max_frames
#     )

#     if len(rgbs) == 0:
#         print("[AllTracker] No images found in", input_dir)
#         return {}

#     # Forward through AllTracker
#     print("[AllTracker] Running forward pass")
#     with torch.no_grad():
#         traj_maps, vis_maps, *_ = model.forward_sliding(
#             rgbs, iters=args.inference_iters, window_len=args.window_len
#         )

#     # Convert output to per-frame keypoints
#     results = {}
#     num_frames = traj_maps.shape[1]  # T
#     for t in range(num_frames):
#         traj_frame = traj_maps[0, t].cpu().numpy()  # HxW
#         yx = np.argwhere(traj_frame > 0)  # list of (y, x)
#         keypoints = yx[:, ::-1]  # convert to (x, y)
#         descriptors = np.ones((len(keypoints), 32), dtype=np.float32)  # placeholder
#         results[f"frame_{t:04d}.png"] = (keypoints, descriptors)

#     print(f"[AllTracker] Extracted keypoints for {len(results)} frames.")
#     return results

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


def img_resize(img, image_size=1024): #same as alltracker read_image_folder
    import cv2
    H0, W0 = img.shape[:2]
    scale = min(image_size / H0, image_size / W0)
    H = int(H0 * scale)
    W = int(W0 * scale)
    H = (H // 8) * 8
    W = (W // 8) * 8
    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img_resized


def resize_like_alltracker(img, image_size=1024):
    H0, W0 = img.shape[:2]
    scale = min(image_size / H0, image_size / W0)
    H = int(H0 * scale)
    W = int(W0 * scale)
    H = (H // 8) * 8
    W = (W // 8) * 8
    img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img_r, (H0, W0), (H, W)


def undo_resize_pts(pts, orig_shape, resized_shape):
    H0, W0 = orig_shape
    Hr, Wr = resized_shape
    sx = W0 / Wr
    sy = H0 / Hr
    pts = pts.copy()
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts


#---------------------------ORB TRACKING------------------------------
#Use ORB to refine and track keypoints between consecutive frames.
def orb_track(results, input_dir, max_kp=5000, image_size=1024):
    import cv2
    tracked_results = {}
    orb = cv2.ORB_create(nfeatures=max_kp)
    filenames = sorted(list(results.keys()))
    for i, filename in enumerate(filenames):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ORB Track]: Could not read {filename}, skipping")
            continue
        #resizing
        H0, W0 = img.shape[:2]
        scale = min(image_size / H0, image_size / W0)
        H = int(H0 * scale)
        W = int(W0 * scale)
        H = (H // 8) * 8
        W = (W // 8) * 8
        img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        
        kps_all, desc_all = results[filename]
        kps_cv2 = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kps_all] if len(kps_all) > 0 else []

        # Subsample if too many
        if len(kps_cv2) > max_kp:
            idxs = np.linspace(0, len(kps_cv2)-1, max_kp).astype(int)
            kps_cv2 = [kps_cv2[idx] for idx in idxs]

        #computing orb
        if len(kps_cv2) > 0:
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
        #resizing
        if kps_refined:
            pts_orig = np.array([[kp.pt[0], kp.pt[1]] for kp in kps_refined], dtype=np.float32)
            sx = W0 / W
            sy = H0 / H
            pts_orig[:, 0] *= sx
            pts_orig[:, 1] *= sy
        else:
            pts_orig = np.zeros((0, 2), dtype=np.float32)
        tracked_results[filename] = (pts_orig, desc_refined)
    return tracked_results


def apply_mask_to_keypoints(kps, mask):
    if kps is None or len(kps) == 0:
        return kps

    H, W = mask.shape
    inside = []
    for x, y in kps:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H and mask[iy, ix] > 0:
            inside.append([x, y])
    return np.array(inside, dtype=np.float32) if inside else np.zeros((0, 2), dtype=np.float32)


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


#---------------------------SAVE TO DB------------------------------
# return Nx6 array for COLMAP
def normalize_to_colmap_kp(pts):
    if pts.shape[1] == 2:  
        N = pts.shape[0]
        scales = np.ones((N, 1), np.float32)
        orientations = np.zeros((N, 1), np.float32)
        a4 = np.zeros((N, 1), np.float32)
        scores = np.ones((N, 1), np.float32)
        return np.hstack([pts, scales, orientations, a4, scores]).astype(np.float32)
    return pts.astype(np.float32)


# filtering keypoints to demo the matching
def filter_keypoints(pts, desc=None, max_kps=2000):
    if pts is None or len(pts) == 0:
        return pts, None, np.array([], dtype=np.int32)
    N, C = pts.shape
    if N <= max_kps:
        return pts.astype(np.float32), desc, np.arange(N)
    if C > 2:  # has scores
        scores = pts[:, 2]
        sel_idx = np.argsort(scores)[-max_kps:] # best ones
    else:
        sel_idx = np.random.choice(N, max_kps, replace=False)
    pts_f = pts[sel_idx].astype(np.float32)
    
    # handling descr = none
    if desc is not None and desc.shape[0] > 0:
        desc_f = desc[sel_idx]
    else:
        desc_f = None

    return pts_f, desc_f, sel_idx

# saving to db
def save_to_database(db_path, results_and_corr, max_kps=2000): # max keypoints
    results, correspondences = results_and_corr
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()
    image_id_map = {}
    keypoints_map = {}  # for matching
    
    # Saving keypts
    for filename, (pts, desc) in results.items():
        # filter keypoints
        pts_f, desc_f, sel_idx = filter_keypoints(pts, desc, max_kps=max_kps)
        keypoints_map[filename] = sel_idx
        # convert to COLMAP format
        colmap_kps = normalize_to_colmap_kp(pts_f)
        # create img
        try:
            db.add_image(name=filename, camera_id=1)
        except sqlite3.IntegrityError:
            pass
        image_id = db.execute(
            "SELECT image_id FROM images WHERE name=?", (filename,)
        ).fetchone()[0]
        image_id_map[filename] = image_id
        # save keypoints
        db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id,))
        if colmap_kps.shape[0] > 0:
            db.add_keypoints(image_id, colmap_kps)
            print(f"[DB] Added {colmap_kps.shape[0]} keypoints for {filename}")
        else:
            print(f"[DB] No keypoints for {filename}")
        # save descriptors
        db.execute("DELETE FROM descriptors WHERE image_id=?", (image_id,))
        if desc_f is not None and desc_f.shape[0] > 0:
            db.add_descriptors(image_id, desc_f.astype(np.uint8))

    # save matches (for fmap correspondance)
    id0 = image_id_map.get("frame_0000.png", None)
    sel0 = keypoints_map.get("frame_0000.png", None)
    if id0 is None or sel0 is None:
        print("[DB] ERROR: frame_0000.png missing")
    else:
        for t, corr in correspondences.items():
            fname = f"frame_{t:04d}.png"
            idt = image_id_map.get(fname, None)
            selt = keypoints_map.get(fname, None)
            if idt is None or selt is None:
                continue
            idx0, idxt = corr["idx0"].astype(np.uint32), corr["idxt"].astype(np.uint32)
            # keep only matches with filtered keypoints
            mask0 = np.isin(idx0, sel0)
            maskt = np.isin(idxt, selt)
            mask = mask0 & maskt
            if not np.any(mask):
                continue
            # remap
            idx_map0 = {old: new for new, old in enumerate(sel0)}
            idx_mapt = {old: new for new, old in enumerate(selt)}
            idx0_f = np.array([idx_map0[i] for i in idx0[mask]], dtype=np.uint32)
            idxt_f = np.array([idx_mapt[i] for i in idxt[mask]], dtype=np.uint32)

            match_arr = np.stack([idx0_f, idxt_f], axis=1)
            pair_id = (id0 << 32) + idt
            db.execute(
                "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?,?,?,?)",
                (pair_id, match_arr.shape[0], match_arr.shape[1], match_arr.tobytes())
            )
            print(f"[DB] Added {match_arr.shape[0]} matches 0 to {fname}")
    db.commit()
    db.close()
    print("[DB] Finished saving keypoints + matches.")



# ------------------VISUALIZE DB------------------
# visualzing database by plotting out keypoints
def visualize_from_db(db_path, img_dir, out_dir, image_size=1024):
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


def apply_mask_to_kps_and_matches(kps0, kps1, matches, mask):
    H, W = mask.shape
    def valid(pt):
        x, y = int(pt[0]), int(pt[1])
        return 0 <= x < W and 0 <= y < H and mask[y, x] > 0
    keep = []
    for i, (i0, i1) in enumerate(matches):
        if valid(kps0[i0]) and valid(kps1[i1]):
            keep.append(i)
    if not keep:
        return None, None, None
    keep = np.array(keep)
    return kps0, kps1, matches[keep]





#---------------------------FEATURE CORRESPONDANCE------------------------------
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

    
    
    

#---------------------------PIPLINE------------------------------
# 0. Alltracker model
def main():
    print("0. Creating AllTracker model")
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

 # -------------------RUN ALLTRACKER-------------------------
    print("\n1. Running AllTracker")
    alltracker_results_and_corr = run_alltracker(model, RAW_IMG_DIR, args)

# -------------------SAVE TO DATABASE-----------------------
    print("\n2. Saving to database")
    save_to_database(DB_PATH, alltracker_results_and_corr)

# -------------------RUN ORB AND MASK-----------------------
    print("\n3. ORB and masking")
    orb_results = orb_track(alltracker_results_and_corr[0], ALLT_IMG_DIR)
    #loading mask
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)
    else:
        print("[WARN] Mask not found, skipping masking")
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
        
        
# -------------------VISUALIZE FEATURE MATCHES--------------
    print("\n4. Visualizing feature correspondences with masking")
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