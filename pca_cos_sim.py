import os
import cv2
import numpy as np
import sqlite3

# --------------------------- CONFIG ------------------------------
RGB_DIR   = "./All_t/masked"
PCA_DIR_1 = "./Database/pca"
PCA_DIR_2 = "./Database2/pca"
DB_PATH_1 = "./Database/alltracker.db"
DB_PATH_2 = "./Database2/testing.db"
OUT_DIR   = "./comparison_results"

N_COMPARE = 5
TOP_K = 50

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------- VISUALIZE LOW AND HIGHEST ------------------------------
# high is green, low is red
def visualize_extremes(rgb, kps, cos_values, mask=None, out_path=None, k=50, erosion_size=5):
    vis = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    # erosion
    if mask is not None:
        # resize mask if needed
        if mask.shape != rgb.shape:
            mask_r = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_r = mask.copy()
        if erosion_size > 0:
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            mask_r = cv2.erode(mask_r, kernel, iterations=1)
        # keep only keypoints in mask
        keep = []
        for i, (x, y) in enumerate(kps):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < mask_r.shape[1] and 0 <= yi < mask_r.shape[0] and mask_r[yi, xi] > 0:
                keep.append(i)
        if len(keep) == 0:
            print("[VIS] No keypoints inside eroded mask, skipping")
            return vis 
        kps = kps[keep]
        cos_values = cos_values[keep]
    sorted_idx = np.argsort(cos_values)
    lowest_idx = sorted_idx[:k]
    highest_idx = sorted_idx[-k:]
    for idx in lowest_idx:
        x, y = kps[idx]
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
    for idx in highest_idx:
        x, y = kps[idx]
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
    # optional save
    if out_path is not None:
        cv2.imwrite(out_path, vis)
    return vis


   
# --------------------------- PCA COMPARISON VIS ------------------------------
def pca_comparison(fname1, fname2, rgb_dir, mask=None, top_k=TOP_K):
    rgb_path1 = os.path.join(rgb_dir, fname1.replace("_pca.png", ".png"))
    rgb_path2 = os.path.join(rgb_dir, fname2.replace("_pca.png", ".png"))
    rgb1 = cv2.imread(rgb_path1, cv2.IMREAD_GRAYSCALE)
    rgb2 = cv2.imread(rgb_path2, cv2.IMREAD_GRAYSCALE)
    if rgb1 is None or rgb2 is None:
        print("[PCA COMP] RGB image missing, skipping.")
        return None
    #comparison
    pca_path1 = os.path.join(PCA_DIR_1, fname1)
    pca_path2 = os.path.join(PCA_DIR_2, fname2)
    img1 = cv2.imread(pca_path1)
    img2 = cv2.imread(pca_path2)
    if img1 is None or img2 is None:
        print("[PCA COMP] PCA image missing, skipping.")
        return None
    # resize to same height
    h = max(img1.shape[0], img2.shape[0])
    img1_r = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2_r = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    combined = np.hstack([img1_r, img2_r])
    return combined
       
       
       
       
       
          
# --------------------------- MAIN ------------------------------
def main():
    conn1 = sqlite3.connect(DB_PATH_1)
    conn2 = sqlite3.connect(DB_PATH_2)
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    files1 = sorted([f for f in os.listdir(PCA_DIR_1) if f.endswith("_pca.png")])
    files2 = sorted([f for f in os.listdir(PCA_DIR_2) if f.endswith("_pca.png")])
    files1_sel = files1[-N_COMPARE:]
    files2_sel = files2[:N_COMPARE]
    print("\nImage pairing (last 5 of Dir1 vs first 5 of Dir2):")

    scores = []
    # Load mask
    mask_path = "/mnt/data1/michelle/t_alltracker/All_t/undistorted_mask.bmp"
    mask_global = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_global is not None:
        mask_global = (mask_global > 0).astype(np.uint8)

    #looping over selected pairs
    for fname1, fname2 in zip(files1_sel, files2_sel):
        print(f"\nComparison: {fname1} <--> {fname2}")
        frame_id_1 = int(fname1.split("_")[1])
        frame_id_2 = int(fname2.split("_")[1])

        #load desc
        cursor1.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=? ORDER BY rowid", (frame_id_1,))
        row1 = cursor1.fetchone()
        cursor2.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=? ORDER BY rowid", (frame_id_2,))
        row2 = cursor2.fetchone()
        if row1 is None or row2 is None:
            print("Descriptor blob missing, skipping.")
            continue
        num_pixels1, desc_dim1, blob1 = row1
        num_pixels2, desc_dim2, blob2 = row2
        if num_pixels1 != num_pixels2 or desc_dim1 != desc_dim2:
            print("Descriptor shape mismatch, skipping.")
            continue
        desc1 = np.frombuffer(blob1, dtype=np.uint8).reshape(num_pixels1, desc_dim1).astype(np.float32)
        desc2 = np.frombuffer(blob2, dtype=np.uint8).reshape(num_pixels2, desc_dim2).astype(np.float32)
        # cosine similarity
        dot  = np.sum(desc1 * desc2, axis=1)
        magA = np.linalg.norm(desc1, axis=1)
        magB = np.linalg.norm(desc2, axis=1)
        cos = dot / (magA * magB + 1e-8)
        scores.append(np.mean(cos))
        print(f"Mean cosine similarity = {np.mean(cos):.6f}")

        # load keypoints
        cursor1.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=? ORDER BY rowid", (frame_id_1,))
        row_kp1 = cursor1.fetchone()
        cursor2.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=? ORDER BY rowid", (frame_id_2,))
        row_kp2 = cursor2.fetchone()
        if row_kp1 is None or row_kp2 is None:
            print("Keypoints missing, skipping visualization.")
            continue

        kp_rows1, kp_cols1, kp_blob1 = row_kp1
        kp_rows2, kp_cols2, kp_blob2 = row_kp2
        kps1 = np.frombuffer(kp_blob1, dtype=np.float32).reshape(kp_rows1, kp_cols1)[:, :2]
        kps2 = np.frombuffer(kp_blob2, dtype=np.float32).reshape(kp_rows2, kp_cols2)[:, :2]

        # pca side by side vis
        pca_vis = pca_comparison(fname1, fname2, RGB_DIR)
        if pca_vis is not None:
            out_path_pca = os.path.join(OUT_DIR, f"{fname1[:-8]}_pca_comp.png")
            cv2.imwrite(out_path_pca, pca_vis)
            print(f"[PCA COMP] Saved {out_path_pca}")

        # highest and lowest dp
        rgb_path1 = os.path.join(RGB_DIR, fname1.replace("_pca.png", ".png"))
        rgb_path2 = os.path.join(RGB_DIR, fname2.replace("_pca.png", ".png"))
        rgb1 = cv2.imread(rgb_path1, cv2.IMREAD_GRAYSCALE)
        rgb2 = cv2.imread(rgb_path2, cv2.IMREAD_GRAYSCALE)
        if rgb1 is None or rgb2 is None:
            print("[VIS HL] RGB missing, skipping extremes overlay.")
            continue
        # mask + erosion
        mask_local = mask_global.copy() if mask_global is not None else None
        if mask_local is not None:
            kernel = np.ones((5,5), np.uint8)
            mask_local = cv2.erode(mask_local, kernel, iterations=1)

        vis1 = visualize_extremes(rgb1, kps1, cos, mask=mask_local, k=TOP_K)
        vis2 = visualize_extremes(rgb2, kps2, cos, mask=mask_local, k=TOP_K)
        combined_extremes = np.hstack([vis1, vis2])
        out_path_ext = os.path.join(OUT_DIR, f"{fname1[:-8]}__extremes_overlay.png")
        cv2.imwrite(out_path_ext, combined_extremes)
        print(f"[EVIS HL] Saved {out_path_ext}")

    # final stats
    if scores:
        print("\n----------------------------------")
        print("FINAL MEAN COSINE SIMILARITY ACROSS ALL FRAMES:", np.mean(scores))
    else:
        print("No valid comparisons.")

    conn1.close()
    conn2.close()


if __name__ == "__main__":
    main()