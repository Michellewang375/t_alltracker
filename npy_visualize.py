import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_map(np_path, output_dir):
    """Visualize a single .npy feature map as a full image."""
    fmap = np.load(np_path)

    # Normalize to 0-1 for better visualization
    fmap_min, fmap_max = fmap.min(), fmap.max()
    fmap_norm = (fmap - fmap_min) / (fmap_max - fmap_min + 1e-8)

    base_name = os.path.splitext(os.path.basename(np_path))[0]

    plt.imshow(fmap_norm)
    plt.axis('off')
    plt.title(base_name)
    out_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize feature maps from .npy files.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".npy")])
    for npy_file in npy_files:
        np_path = os.path.join(args.input_dir, npy_file)
        print(f"Processing {npy_file}...")
        visualize_feature_map(np_path, args.output_dir)

    print(f"\nVisualization complete. Images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

# run command: python DINO_vis.py --input_dir feature_maps --output_dir dino_converted
