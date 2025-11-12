#Load the orb and descirptors into the database
import os
import numpy as np
from Db import COLMAPDatabase

def main():
    db_path = "./../database.db"
    keypoints_dir = "key_npy"
    desc_dir = "desc_npy"
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = COLMAPDatabase.connect(db_path)
    db.clear_keypoints()  # deletes all existing keypoints
    db.clear_descriptors()  # deletes all descriptors


    # iterate through keypoint/descriptor files
    for filename in sorted(os.listdir(keypoints_dir)):
            if not filename.endswith("_kps.npy"):
                continue

            base = filename.replace("_kps.npy", "")  # e.g., "043"
            image_id = int(base.lstrip("0"))         # ensures "043" -> 43

            kp_path = os.path.join(keypoints_dir, filename)
            desc_path = os.path.join(desc_dir, f"{base}_desc.npy")

            print(f"Inserting image {base} (ID {image_id}) into DB...")

            db.add_keypoints_npy(image_id, kp_path)
            db.add_descriptors_npy(image_id, desc_path)

    db.commit()
    db.close()
    print("Successfully inserted all keypoints + descriptors into the database.")

if __name__ == "__main__":
    main()
