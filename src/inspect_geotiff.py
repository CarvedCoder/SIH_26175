import rasterio
import numpy as np

# Path to our original GeoTIFF
tiff_path = "../data/raw/286_5630_dem.tif/286_5630_dem.tif"

# Open the GeoTIFF
with rasterio.open(tiff_path) as src:

    print("========== GeoTIFF Information ==========")

    print("File:", tiff_path)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Number of bands:", src.count)

    print("\n--- Coordinate Information ---")
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Bounds:", src.bounds)

    print("\n--- Raster Information ---")
    print("Data type:", src.dtypes)
    print("NoData value:", src.nodata)
    print("Resolution:", src.res)

    # Read the first band
    elevation = src.read(1)

    # Check data quality
    nodata_count = np.sum(elevation == src.nodata)
    nan_count = np.sum(np.isnan(elevation))
    inf_count = np.sum(np.isinf(elevation))

    total_pixels = elevation.size
    valid_pixels = total_pixels - nodata_count - nan_count

    print("\n--- Data Quality ---")
    print("Total pixels:", total_pixels)
    print("NoData pixels:", nodata_count)
    print("NaN pixels:", nan_count)
    print("Inf pixels:", inf_count)
    print("Valid pixels:", valid_pixels)

    print("\n--- Elevation Information ---")

    # Remove NoData values before calculating statistics
    if src.nodata is not None:
        valid_values = elevation[elevation != src.nodata]
    else:
        valid_values = elevation

    print("Minimum:", np.min(valid_values))
    print("Maximum:", np.max(valid_values))
    print("Mean:", np.mean(valid_values))

    print("\n--- Sample Values ---")
    print(elevation[:5, :5])