import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_map(np_path, output_dir):
    """Visualize a single .npy feature map and save to output_dir."""
    fmap = np.load(np_path)
    base_name = os.path.splitext(os.path.basename(np_path))[0]
    
    # Handle 2D or 3D maps
    if fmap.ndim == 2:
        plt.imshow(fmap, cmap='viridis')
        plt.colorbar()
        plt.title(f"{base_name}")
        plt.axis('off')
        out_path = os.path.join(output_dir, f"{base_name}.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    elif fmap.ndim == 3:
        n_channels = fmap.shape[0]
        for i in range(n_channels):
            plt.imshow(fmap[i], cmap='viridis')
            plt.colorbar()
            plt.title(f"{base_name}_ch{i}")
            plt.axis('off')
            out_path = os.path.join(output_dir, f"{base_name}_ch{i}.png")
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
    else:
        print(f"Skipping {np_path}: unexpected shape {fmap.shape}")

def main():
    parser = argparse.ArgumentParser(description="Visualize feature maps from .npy files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .npy files.")
    parser.add_argument("--output_dir", default="feature_map_images", help="Directory to save .png images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(args.input_dir) if f.endswith(".npy")]
    for npy_file in npy_files:
        np_path = os.path.join(args.input_dir, npy_file)
        print(f"Processing {npy_file}...")
        visualize_feature_map(np_path, args.output_dir)

    print(f"\n✅ Visualization complete. Images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
