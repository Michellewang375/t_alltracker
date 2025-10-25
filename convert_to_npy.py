import os
import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert images to .npy")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--grayscale", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.input_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(args.input_dir, fname)
            if args.grayscale:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            npy_path = os.path.join(args.output_dir, os.path.splitext(fname)[0] + ".npy")
            np.save(npy_path, img)
            print(f"Saved {npy_path}")

    print(f"Converted {len(os.listdir(args.output_dir))} frames to .npy to {args.output_dir}/")

if __name__ == "__main__":
    main()
#run command:
# python convert_to_npy.py --input_dir img_no_fmap --output_dir feature_maps
    # if want grayscale version: python convert_to_npy.py --input_dir img_no_fmap --output_dir feature_maps --grayscale