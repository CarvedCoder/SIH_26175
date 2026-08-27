import rasterio
from rasterio.windows import Window
import os

input_path = "../data/process/cleaned_286_5630_dem.tif"
output_dir = "../data/process/tiles"

os.makedirs(output_dir, exist_ok=True)

tile_size = 256

with rasterio.open(input_path) as src:

    for row in range(0, src.height, tile_size):
        for col in range(0, src.width, tile_size):

            width = min(tile_size, src.width - col)
            height = min(tile_size, src.height - row)

            window = Window(col, row, width, height)

            data = src.read(1, window=window)

            transform = src.window_transform(window)

            profile = src.profile.copy()
            profile.update({
                "height": height,
                "width": width,
                "transform": transform
            })

            tile_name = f"tile_{row}_{col}.tif"
            output_path = os.path.join(output_dir, tile_name)

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data, 1)

print("Tiling completed.")
print("Tiles saved to:", output_dir)