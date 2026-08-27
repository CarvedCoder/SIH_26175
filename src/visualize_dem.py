import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Path to the original GeoTIFF
tiff_path = "../data/raw/286_5630_dem.tif/286_5630_dem.tif"

# Open the GeoTIFF
with rasterio.open(tiff_path) as src:

    # Read the elevation band
    elevation = src.read(1)

    # Get NoData value
    nodata = src.nodata

    # Create a mask for invalid/NoData pixels
    if nodata is not None:
        elevation = np.ma.masked_equal(elevation, nodata)

    # Create the plot
    plt.figure(figsize=(10, 8))

    image = plt.imshow(elevation)

    # Add colorbar
    plt.colorbar(image, label="Elevation")

    # Add title
    plt.title("286_5630 DEM - Elevation")

    # Label axes
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")

    # Display
    plt.show()