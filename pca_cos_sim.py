import os
import cv2
import numpy as np
import sqlite3

# --------------------------- CONFIG ------------------------------
RGB_DIR   = "./All_t/masked"
PCA_DIR_1 = "./Database/pca"
PCA_DIR_2 = "./testing2/pca"
DB_PATH_1 = "./Database/alltracker.db"
DB_PATH_2 = "./testing2/testing.db"
OUT_DIR   = "./comparison_results"

N_COMPARE = 5
TOP_K = 50

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------- VISUALIZE LOW AND HIGHEST ------------------------------
def visualize_extremes(rgb, kps, cos_values, mask=None, out_path=None, k=50):
    vis = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    # resize mask to match image
    if mask is not None:
        if mask.shape != rgb.shape:
            mask_r = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_r = mask
        keep = [i for i, (x, y) in enumerate(kps) 
                if 0 <= int(round(x)) < mask_r.shape[1] 
                and 0 <= int(round(y)) < mask_r.shape[0] 
                and mask_r[int(round(y)), int(round(x))] > 0]
        if len(keep) == 0:
            print("[VIS] No keypoints inside mask, skipping")
            return
        kps = kps[keep]
        cos_values = cos_values[keep]
    sorted_idx = np.argsort(cos_values)
    lowest_idx  = sorted_idx[:k]
    highest_idx = sorted_idx[-k:]
    for idx in lowest_idx:
        x, y = kps[idx]
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
    for idx in highest_idx:
        x, y = kps[idx]
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
    return vis
        
        
          
# --------------------------- MAIN ------------------------------
def main():
    conn1 = sqlite3.connect(DB_PATH_1)
    conn2 = sqlite3.connect(DB_PATH_2)
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()

    # select PCA images
    files1 = sorted([f for f in os.listdir(PCA_DIR_1) if f.endswith("_pca.png")])
    files2 = sorted([f for f in os.listdir(PCA_DIR_2) if f.endswith("_pca.png")])
    files1_sel = files1[-N_COMPARE:]
    files2_sel = files2[:N_COMPARE]

    print("\nImage pairing:")
    for a, b in zip(files1_sel, files2_sel):
        print(f"   {a}  <-->  {b}")

    scores = []

    mask_path = "./All_t/mask.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)

    # selecting descriptors from db
    query_desc = "SELECT rows, cols, data FROM descriptors WHERE image_id=? ORDER BY rowid"
    query_kp   = "SELECT rows, cols, data FROM keypoints WHERE image_id=? ORDER BY rowid"

    for fname1, fname2 in zip(files1_sel, files2_sel):
        print("\n---------------------------------------")
        print(f"DIR1: {fname1}")
        print(f"DIR2: {fname2}")

        frame_id_1 = int(fname1.split("_")[1])
        frame_id_2 = int(fname2.split("_")[1])

        # descriptors
        cursor1.execute(query_desc, (frame_id_1,))
        row1 = cursor1.fetchone()
        cursor2.execute(query_desc, (frame_id_2,))
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

        #debugging here!!!!!!
        print("Descriptors identical:", np.array_equal(desc1, desc2))
        print("Descriptor mean abs diff:", np.mean(np.abs(desc1 - desc2)))
        print("Descriptor max abs diff:", np.max(np.abs(desc1 - desc2)))

        # cosine similarity
        dot  = np.sum(desc1 * desc2, axis=1)
        magA = np.linalg.norm(desc1, axis=1)
        magB = np.linalg.norm(desc2, axis=1)
        cos = dot / (magA * magB + 1e-8)
        scores.append(np.mean(cos))
        print(f"Mean cosine similarity = {np.mean(cos):.6f}")

        # keypoints
        cursor1.execute(query_kp, (frame_id_1,))
        row_kp1 = cursor1.fetchone()
        cursor2.execute(query_kp, (frame_id_2,))
        row_kp2 = cursor2.fetchone()
        if row_kp1 is None or row_kp2 is None:
            print("No keypoints found, skipping visualization.")
            continue
        kp_rows1, kp_cols1, kp_blob1 = row_kp1
        kp_rows2, kp_cols2, kp_blob2 = row_kp2
        kps1 = np.frombuffer(kp_blob1, dtype=np.float32).reshape(kp_rows1, kp_cols1)[:, :2]
        kps2 = np.frombuffer(kp_blob2, dtype=np.float32).reshape(kp_rows2, kp_cols2)[:, :2]
        #Debugging here!!!!!
        print("Keypoints identical:", np.array_equal(kps1, kps2))
        print("Keypoint mean diff:", np.mean(np.abs(kps1 - kps2)))
        
        # visualization
        rgb_path1 = os.path.join(RGB_DIR, fname1.replace("_pca.png", ".png"))
        rgb_path2 = os.path.join(RGB_DIR, fname2.replace("_pca.png", ".png"))

        rgb1 = cv2.imread(rgb_path1, cv2.IMREAD_GRAYSCALE)
        rgb2 = cv2.imread(rgb_path2, cv2.IMREAD_GRAYSCALE)

        if rgb1 is None or rgb2 is None:
            print("RGB image missing, skipping visualization.")
            continue

        vis1 = visualize_extremes(rgb1, kps1, cos, mask=mask, k=TOP_K)
        vis2 = visualize_extremes(rgb2, kps2, cos, mask=mask, k=TOP_K)
        
        combined = np.hstack([vis1, vis2])

        out_path = os.path.join(OUT_DIR, f"{fname1[:-8]}__comparison.png")
        cv2.imwrite(out_path, combined)

        print(f"[VIS] Saved {out_path}")


    if scores:
        print("\n=======================================")
        print("FINAL MEAN ACROSS ALL FRAMES:", np.mean(scores))
    else:
        print("No valid comparisons.")

    conn1.close()
    conn2.close()

if __name__ == "__main__":
    main()