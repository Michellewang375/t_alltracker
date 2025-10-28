import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_map(np_path, output_dir, channel_mode="mean"):
    fmap = np.load(np_path)

    # If 3D (C, H, W), reduce to 2D
    if fmap.ndim == 3:
        if channel_mode == "mean":
            fmap_2d = fmap.mean(axis=0)
        elif channel_mode == "max":
            fmap_2d = fmap.max(axis=0)
        else:
            raise ValueError(f"Unknown channel: {channel_mode}")
    elif fmap.ndim == 2:
        fmap_2d = fmap
    else:
        raise ValueError(f"Unsupported shape {fmap.shape}")

    # Normalize to 0-1
    fmap_min, fmap_max = fmap_2d.min(), fmap_2d.max()
    fmap_norm = (fmap_2d - fmap_min) / (fmap_max - fmap_min + 1e-8)

    base_name = os.path.splitext(os.path.basename(np_path))[0]

    plt.imshow(fmap_norm, cmap="viridis")  # or "gray" for graysclae
    plt.axis('off')
    plt.title(base_name)
    out_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize fmaps from .npy files.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--channel_mode", default="mean", choices=["mean", "max"],
                        help="How to reduce channels to 2D (mean or max)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".npy")])
    for npy_file in npy_files:
        np_path = os.path.join(args.input_dir, npy_file)
        print(f"Processing {npy_file}...")
        visualize_feature_map(np_path, args.output_dir, channel_mode=args.channel_mode)

    print(f"\nVisualization complete. Images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
