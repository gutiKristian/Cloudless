"""
Do the median filtering on each band -  R, G, B
Do the filtering only on pixels with values 0 or 255 this is not the case (I'll update it later)
Please also note that this aims to improve only the visual aspect of created mosaic, if you want to
commit a further analysis on the data do not do this! This operation alters the values and may devalue the analysis.
"""
import rasterio
from skimage.morphology import disk
from skimage.filters import median
from Pipeline.utils import rescale_intensity


# Load files
red = rasterio.open("B04.tif")
green = rasterio.open("B03.tif")
blue = rasterio.open("B02.tif")

# Read rasters
red_raster = red.read(1)
green_raster = green.read(1)
blue_raster = blue.read(1)

new_rgb = [median(red_raster, disk(1)), median(green_raster, disk(1)), median(blue_raster, disk(1))]

# Enhance rgb, *1.5 is gain
new_rgb[0] = rescale_intensity(new_rgb[0] * 1.5, 0, 4096)
new_rgb[1] = rescale_intensity(new_rgb[1] * 1.5, 0, 4096)
new_rgb[2] = rescale_intensity(new_rgb[2] * 1.5, 0, 4096)

# create new rgb profile for our new image
rgb_profile = red.profile
rgb_profile['dtype'] = 'uint8'
rgb_profile['count'] = 3
rgb_profile['photometric'] = "RGB"
rgb_profile['driver'] = "GTiff"
rgb_profile['interleave'] = "PIXEL"
rgb_profile['compress'] = "JPEG"
rgb_profile['photometric'] = "YCBCR"
rgb_profile['blockxsize'] = 256
rgb_profile['blockysize'] = 256
rgb_profile['nodata'] = 0

with rasterio.open("NEW_RGB.tif", 'w', **rgb_profile) as dst:
    for count, band in enumerate(new_rgb, 1):
        dst.write(band, count)


red.close()
blue.close()
green.close()
