import os
import cv2
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import csv

# --------------------------- CONFIG ------------------------------
RGB_DIR   = "./p01_left_preop_scope01/window1/masked1"
DB_PATH_1 = "./p01_left_preop_scope01/window1/alltracker.db"
DB_PATH_2 = "./p01_left_preop_scope01/window2/db2.db"
OUT_DIR   = "./p01_left_preop_scope01/comparison_results"
sequence_name = "p01_left_preop_scope01"

N_COMPARE = 5
TOP_K = 50
OFFSET = 0
os.makedirs(OUT_DIR, exist_ok=True)
MASK_PATH = "/mnt/data1/michelle/t_alltracker/All_t/undistorted_mask.bmp"


# --------------------------- VISUALIZE EXTREMES ------------------------------
def visualize_extremes(rgb, kps, cos_values, mask=None, k=TOP_K):
    vis = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    if mask is not None:
        mask_r = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        keep = [i for i, (x, y) in enumerate(kps) if mask_r[int(round(y)), int(round(x))] > 0]
        if len(keep) == 0:
            return vis
        kps = kps[keep]
        cos_values = cos_values[keep]
    sorted_idx = np.argsort(cos_values)
    lowest_idx = sorted_idx[:k]
    highest_idx = sorted_idx[-k:]
    for idx in lowest_idx:
        x, y = kps[idx]
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0,0,255), -1)
    for idx in highest_idx:
        x, y = kps[idx]
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0,255,0), -1)
    return vis

# --------------------------- PCA ------------------------------
def pca_from_db(cursor, image_id, img_path, mask=None, square_size=5):
    # descriptors
    cursor.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (image_id,))
    row = cursor.fetchone()
    if row is None or row[2] is None:
        print(f"[PCA] No descriptors for image_id {image_id}")
        return None
    rows, cols, blob = row
    desc = np.frombuffer(blob, dtype=np.uint8)
    if desc.size % cols != 0:
        desc = desc[:(desc.size // cols) * cols]
    desc = desc.reshape(desc.size // cols, cols)
    if np.var(desc) == 0:
        return None
    desc_f = desc.astype(np.float32) / 255.0

    # keypoints
    cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,))
    row_kp = cursor.fetchone()
    if row_kp is None or row_kp[2] is None:
        return None
    kp_rows, kp_cols, kp_blob = row_kp
    kps = np.frombuffer(kp_blob, dtype=np.float32).reshape(kp_rows, kp_cols)

    # PCA
    pca = PCA(n_components=3)
    projected = pca.fit_transform(desc_f)
    colors = (projected - projected.min(0)) / (np.ptp(projected, axis=0) + 1e-8)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    H_img, W_img = img.shape[:2]
    viz = np.zeros((H_img, W_img, 3), dtype=np.float32)
    half = square_size // 2

    # mask and erode
    if mask is None:
        mask = np.ones((H_img, W_img), dtype=np.uint8)
    else:
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=10)
        if mask.shape != (H_img, W_img):
            mask = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    for i in range(kps.shape[0]):
        x, y = int(round(kps[i,0])), int(round(kps[i,1]))
        if mask[y, x] == 0:
            continue
        x0, x1 = max(x-half, 0), min(x+half+1, W_img)
        y0, y1 = max(y-half, 0), min(y+half+1, H_img)
        viz[y0:y1, x0:x1, :] = colors[i]

    return (viz*255).astype(np.uint8)

# --------------------------- COSINE SIMILARITY PLOT ------------------------------
def plot_cosine_similarity(frame_indices, scores, out_path):
    plt.figure()
    plt.plot(frame_indices, scores, marker='o')
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Cosine Similarity Over Time")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
    print(f"[GRAPH] Saved {out_path}")

# --------------------------- SAVE TO CSV ------------------------------
def save_to_csv(cos_values, out_path, sequence_name):
    if len(cos_values) == 0:
        print("[CSV] No cosine values to save.")
        return
    cos_values = np.array(cos_values)
    stats = {
        "sequence": sequence_name,
        "min": np.min(cos_values),
        "max": np.max(cos_values),
        "mean": np.mean(cos_values),
        "std": np.std(cos_values),
        "median": np.median(cos_values),
    }
    # Check if file exists 
    file_exists = os.path.isfile(out_path)
    with open(out_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        # write header only once
        if not file_exists:
            writer.writerow(stats.keys())
        writer.writerow(stats.values())
    print(f"[CSV] Appended stats to {out_path}")
    
# --------------------------- MAIN ------------------------------
def main():
    conn1 = sqlite3.connect(DB_PATH_1)
    conn2 = sqlite3.connect(DB_PATH_2)
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()

    # image ids
    cursor1.execute("SELECT image_id FROM images ORDER BY image_id")
    ids1 = [row[0] for row in cursor1.fetchall()]
    cursor2.execute("SELECT image_id FROM images ORDER BY image_id")
    ids2 = [row[0] for row in cursor2.fetchall()]

    num_pairs = min(len(ids1)-OFFSET, len(ids2))
    ids1_sel = ids1[OFFSET:OFFSET+num_pairs]
    ids2_sel = ids2[:num_pairs]

    # load mask
    mask_global = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_global is not None:
        mask_global = (mask_global>0).astype(np.uint8)

    # random 10 frames for visualization
    vis_indices = random.sample(range(num_pairs), min(10,num_pairs))

    scores = []
    frame_indices = []
    all_cos_values = []

    for idx, (frame_id_1, frame_id_2) in enumerate(zip(ids1_sel, ids2_sel)):

        # DESCRIPTORS
        cursor1.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (frame_id_1,))
        row1 = cursor1.fetchone()
        cursor2.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (frame_id_2,))
        row2 = cursor2.fetchone()
        if row1 is None or row2 is None:
            continue

        desc1 = np.frombuffer(row1[2], dtype=np.uint8).reshape(row1[0], row1[1]).astype(np.float32)
        desc2 = np.frombuffer(row2[2], dtype=np.uint8).reshape(row2[0], row2[1]).astype(np.float32)
        min_len = min(len(desc1), len(desc2))
        if min_len==0: continue
        desc1, desc2 = desc1[:min_len], desc2[:min_len]

        # COSINE SIMILARITY
        dot = np.sum(desc1 * desc2, axis=1)
        magA = np.linalg.norm(desc1, axis=1)
        magB = np.linalg.norm(desc2, axis=1)
        cos = dot / (magA * magB + 1e-8)
        mean_cos = np.mean(cos)
        min_cos = np.min(cos)
        max_cos = np.max(cos)

        # APPEND TO ALL SCORES
        scores.append(mean_cos)
        frame_indices.append(idx)
        all_cos_values.extend(cos.tolist())

        print(f"Frame {frame_id_1} vs {frame_id_2}: mean={mean_cos:.6f}, min={min_cos:.6f}, max={max_cos:.6f}")

        # VISUALIZATION ONLY FOR RANDOM SAMPLE
        if idx in vis_indices:
            # paths
            fname1 = f"frame_{frame_id_1:04d}.png"
            fname2 = f"frame_{frame_id_2:04d}.png"
            rgb_path1 = os.path.join(RGB_DIR, fname1)
            rgb_path2 = os.path.join(RGB_DIR, fname2)

            # PCA from DB
            pca1 = pca_from_db(cursor1, frame_id_1, rgb_path1, mask_global)
            pca2 = pca_from_db(cursor2, frame_id_2, rgb_path2, mask_global)
            if pca1 is not None and pca2 is not None:
                combined_pca = np.hstack([pca1,pca2])
                out_path_pca = os.path.join(OUT_DIR, f"{fname1[:-4]}_pca_comp.png")
                cv2.imwrite(out_path_pca, combined_pca)
                print(f"[PCA] Saved {out_path_pca}")

            # EXTREMES
            cursor1.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (frame_id_1,))
            cursor2.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (frame_id_2,))
            row_kp1 = cursor1.fetchone()
            row_kp2 = cursor2.fetchone()
            if row_kp1 is None or row_kp2 is None: continue
            kps1 = np.frombuffer(row_kp1[2], dtype=np.float32).reshape(row_kp1[0], row_kp1[1])[:,:2]
            kps2 = np.frombuffer(row_kp2[2], dtype=np.float32).reshape(row_kp2[0], row_kp2[1])[:,:2]
            
            kps1 = kps1[:min_len]
            kps2 = kps2[:min_len]

            vis1 = visualize_extremes(cv2.imread(rgb_path1, cv2.IMREAD_GRAYSCALE), kps1, cos, mask_global, TOP_K)
            vis2 = visualize_extremes(cv2.imread(rgb_path2, cv2.IMREAD_GRAYSCALE), kps2, cos, mask_global, TOP_K)
            combined_ext = np.hstack([vis1, vis2])
            out_path_ext = os.path.join(OUT_DIR, f"{fname1[:-4]}_extremes_overlay.png")
            cv2.imwrite(out_path_ext, combined_ext)
            print(f"[EXTREMES] Saved {out_path_ext}")

    # plot cosine graph and save to csv
    if scores and len(all_cos_values) > 0:
        graph_path = os.path.join(OUT_DIR, "cosine_similarity_over_time.png")
        plot_cosine_similarity(frame_indices, scores, graph_path)
        # Save CSV stats
        save_to_csv(all_cos_values, "./stats.csv", sequence_name)

    else:
        print("No valid comparisons.")
            
        
        

    conn1.close()
    conn2.close()

if __name__ == "__main__":
    main()