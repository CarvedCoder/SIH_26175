import rasterio
import numpy as np

input_path = "../data/raw/286_5630_dem.tif/286_5630_dem.tif"
output_path = "../data/process/cleaned_286_5630_dem.tif"

with rasterio.open(input_path) as src:

    elevation = src.read(1)

    # Create mask for invalid values
    invalid_mask = np.isnan(elevation) | np.isinf(elevation)

    if src.nodata is not None:
        invalid_mask |= elevation == src.nodata

    # Count invalid pixels
    invalid_count = np.sum(invalid_mask)

    print("Invalid pixels:", invalid_count)

    # Copy original metadata
    profile = src.profile.copy()

    # Write cleaned raster
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(elevation, 1)

print("Cleaning completed.")
print("Saved to:", output_path)

import rasterio
import numpy as np

input_path = "../data/raw/286_5630_dem.tif/286_5630_dem.tif"
output_path = "../data/process/cleaned_286_5630_dem.tif"

with rasterio.open(input_path) as src:

    elevation = src.read(1)

    # Create mask for invalid values
    invalid_mask = np.isnan(elevation) | np.isinf(elevation)

    if src.nodata is not None:
        invalid_mask |= elevation == src.nodata

    # Count invalid pixels
    invalid_count = np.sum(invalid_mask)

    print("Invalid pixels:", invalid_count)

    # Copy original metadata
    profile = src.profile.copy()

    # Write cleaned raster
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(elevation, 1)

print("Cleaning completed.")
print("Saved to:", output_path)