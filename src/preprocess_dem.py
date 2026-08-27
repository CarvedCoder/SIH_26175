import rasterio
import numpy as np

input_path = "../data/process/tiles/tile_0_0.tif"
output_path = "../data/process/tiles/tile_0_0_normalized.tif"

with rasterio.open(input_path) as src:

    elevation = src.read(1).astype(np.float32)

    # Min-Max normalization
    min_value = np.min(elevation)
    max_value = np.max(elevation)

    normalized = (elevation - min_value) / (max_value - min_value)

    print("Original minimum:", min_value)
    print("Original maximum:", max_value)

    print("Normalized minimum:", np.min(normalized))
    print("Normalized maximum:", np.max(normalized))

    profile = src.profile.copy()

    profile.update(dtype="float32")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(normalized, 1)

print("\nPreprocessing completed.")
print("Saved to:", output_path)