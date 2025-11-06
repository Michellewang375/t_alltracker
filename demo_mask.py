# places a mask so it only reads images inside the mask, then extract kepyoints using orb



#----------------------------KEYPOINTS--------------------------------
# Just keypoints, not visuals and prints out how many keypoints are in a single image
# import cv2
# import numpy as np
# import os
# import argparse
# from natsort import natsorted

# def load_mask(mask_path, frame_shape):
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask.shape != frame_shape[:2]:
#         mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
#     return (mask > 0).astype(np.uint8)  # binary mask

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--frames', type=str, required=True, help="Folder containing video frames")
#     parser.add_argument('--mask', type=str, required=True, help="Path to undistorted_mask.bmp")
#     parser.add_argument('--prints', action='store_true', help="Show masked frames and keypoints")
#     args = parser.parse_args()

#     frame_files = natsorted([os.path.join(args.frames, f) for f in os.listdir(args.frames)
#                              if f.lower().endswith(('.png', '.jpg', '.bmp'))])
#     if len(frame_files) == 0:
#         raise ValueError("No frames found in folder")

#     # Initialize ORB
#     orb = cv2.ORB_create(1000)

#     for frame_file in frame_files:
#         frame = cv2.imread(frame_file)
#         if frame is None:
#             continue

#         # Load/resize mask
#         mask = load_mask(args.mask, frame.shape)

#         # Apply mask
#         masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

#         # Convert to grayscale
#         gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

#         # Detect features inside mask
#         keypoints, descriptors = orb.detectAndCompute(gray, mask)

#         # detects ORB keypoints inside your mask and prints how many per frame
#         print(f"{frame_file}: {len(keypoints)} keypoints detected inside mask")

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



#----------------------------KEYPOINTS--------------------------------
# Keypoints and visual (matplotlib), prints how many keypoints are in a single image and stores imgs to folder
import cv2
import numpy as np
import os
import argparse
from natsort import natsorted
import matplotlib.pyplot as plt

def load_mask(mask_path, frame_shape):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask file at {mask_path}")
    if mask.shape != frame_shape[:2]:
        mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=str, required=True, help="Folder containing video frames")
    parser.add_argument('--mask', type=str, required=True, help="Path to undistorted_mask.bmp")
    parser.add_argument('--prints', action='store_true', help="Save masked frames with keypoints using matplotlib")
    args = parser.parse_args()

    frame_files = natsorted([os.path.join(args.frames, f) for f in os.listdir(args.frames)
                             if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    if len(frame_files) == 0:
        raise ValueError("No frames found in folder")

    # Initialize ORB
    orb = cv2.ORB_create(1000)

    # output folder name
    os.makedirs("keypoints", exist_ok=True)

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is None:
            continue

        mask = load_mask(args.mask, frame.shape)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, mask)

        print(f"{frame_file}: {len(keypoints)} keypoints detected inside mask")

        if args.prints:
            # Draw keypoints
            vis = cv2.drawKeypoints(masked_frame, keypoints, None, color=(0,255,0))
            # Convert BGR to RGB for matplotlib
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10,6))
            plt.imshow(vis_rgb)
            plt.axis('off')
            plt.title(os.path.basename(frame_file))
            plt.savefig(f"keypoints/{os.path.basename(frame_file)}")
            plt.close()

if __name__ == "__main__":
    main()
