import rasterio

path = "../data/process/cleaned_286_5630_dem.tif"

with rasterio.open(path) as src:
    print("========== Cleaned DEM ==========")
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
    print("Data type:", src.dtypes)
    print("NoData:", src.nodata)

print("\nVerification completed.")