import rasterio
import numpy as np

path = "../data/process/tiles/tile_0_0_normalized.tif"

with rasterio.open(path) as src:

    data = src.read(1)

    print("========== Preprocessed Tile ==========")
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Data type:", src.dtypes)

    print("\n--- Normalized Values ---")
    print("Minimum:", np.min(data))
    print("Maximum:", np.max(data))
    print("Mean:", np.mean(data))

print("\nPreprocessing verification completed.")