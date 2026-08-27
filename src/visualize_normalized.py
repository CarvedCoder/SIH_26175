import rasterio
import matplotlib.pyplot as plt

path = "../data/process/tiles/tile_0_0_normalized.tif"

with rasterio.open(path) as src:
    data = src.read(1)

plt.figure(figsize=(10, 8))

plt.imshow(data)

plt.colorbar(label="Normalized Elevation (0–1)")

plt.title("Normalized DEM - tile_0_0")

plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

plt.show()