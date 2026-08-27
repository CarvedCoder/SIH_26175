import rasterio
import numpy as np

input_path = "../data/process/tiles/tile_0_0_normalized.tif"

with rasterio.open(input_path) as src:

    data = src.read(1).astype(np.float32)

    # Add channel dimension
    data = np.expand_dims(data, axis=0)

    # Add batch dimension
    data = np.expand_dims(data, axis=0)

print("========== MODEL INPUT ==========")
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Minimum:", np.min(data))
print("Maximum:", np.max(data))

print("\nModel input preparation completed.")