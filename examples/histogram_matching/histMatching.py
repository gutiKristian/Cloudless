import rasterio
import numpy
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

reference = rasterio.open("sk.tif")
target_img = rasterio.open("eq.tif")

ref_raster = reference.read()
target_raster = target_img.read()

matched = match_histograms(target_raster, ref_raster).astype(numpy.uint8)

# Save the result
with rasterio.open("MATCHED_EQ.tif", "w", **reference.profile) as dst:
    dst.write(matched[0], 1)
    dst.write(matched[1], 2)
    dst.write(matched[2], 3)
