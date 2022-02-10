import os
import re
from typing import List

import numpy
import rasterio
import glob
from skimage import exposure
from Pipeline.Granule import S2Granule
from Pipeline.Task import S2CloudlessPerPixel
from Pipeline.logger import log
import subprocess
from rasterio.enums import Resampling
import Download.Sentinel2 as Download
from skimage.morphology import disk, square, dilation
from scipy.ndimage.filters import convolve

USER_NAME = "kristianson12"
PASSWORD = "mosvegczzz"


# --------------- FILE UTILS ---------------

def is_dir_valid(path: str) -> bool:
    return os.path.isdir(path)


def format_path(path: str) -> str:
    if path is None or len(path) == 0:
        raise ValueError("Path is none or length is 0")
    return path if path[-1] == os.path.sep else path + os.path.sep


def is_file_valid(path: str) -> bool:
    return os.path.isfile(path)


def get_subdirectories(path: str) -> List[str]:
    return [el.path for el in os.scandir(path) if el.is_dir()]


def get_files_in_directory(path: str, _type: str = None) -> List[str]:
    result = []
    for f in os.listdir(path):
        f = path + os.path.sep + f
        if is_file_valid(f):
            if _type and os.path.splitext(f)[1] == _type:
                result.append(f)
            elif not _type:
                result.append(f)
    return result


def extract_rgb_paths(path):
    r = glob.glob(path + os.path.sep + "*B04*")[0]
    g = glob.glob(path + os.path.sep + "*B03*")[0]
    b = glob.glob(path + os.path.sep + "*B02*")[0]
    return r, g, b


# --------------- S2WORKER UTILS ---------------

def s2_is_spatial_correct(resolution: int) -> bool:
    return resolution in [10, 20, 60]


def s2_is_safe_format(name: str) -> bool:
    """
    Checks whether the dataset is in sentinel 2 safe format.
    Actually checks if the name of the folder hasn't been modified.
    :param name: file name
    :return: true/false
    """
    return name.split(".")[-1] == "SAFE"


def extract_mercator(path: str) -> str:
    g = re.search('(_?)(T[0-9]+[a-zA-Z]+)(_?)', path)
    if g:
        return g.group(2)
    return ""


def s2_get_resolution(spatial):
    """
    Return image size for that spatial res.
    @param spatial:
    @return:
    """
    if spatial == 10:
        return 10980, 10980
    if spatial == 20:
        return 5490, 5490
    elif spatial == 60:
        return 1830, 1830
    log.info("Spatial not present, calculating on the fly...")
    val = spatial / 10
    return 10980 / val, 10980 / val


def look_up_raster(node, element):
    item = node.findall(element)
    for child in node:
        if len(item) > 0:
            return item
        item = look_up_raster(child, element)
    return item


def bands_for_resolution(spatial_resolution):
    if not s2_is_spatial_correct(spatial_resolution):
        raise Exception("Wrong spatial resolution")
    if spatial_resolution == 20:
        return ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "AOT", "SCL"]
    elif spatial_resolution == 10:
        return ["B02", "B03", "B04", "B08", "AOT"]
    return ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12", "AOT", "SCL"]  # 60


def s2_l1c_bands():
    return ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


# --------------- BAND UTILS ---------------

def is_supported_slice(index: int):
    """
    Slice index: 5 = 20x20km
                10 = 10x10km
                15 =  6x6km (exactly 6.6)
                # Only for 10m, 20m
                18 =  5x5km (exactly 5.5)
                45 =  2x2km (exactly  2.2)
    """
    return index in [5, 10, 15, 18, 45]


def find_closest_slice(index: int):
    current_index = 0
    arr = [5, 10, 15, 18, 45]
    _x = 10000
    for i in range(len(arr)):
        x = abs(index - arr[i])
        if x < _x:
            _x = x
            current_index = i
    return arr[current_index]


# --------------- RASTER UTILS ---------------


def ndvi(red: numpy.ndarray, nir: numpy.ndarray) -> numpy.ndarray:
    ndvi1 = (nir - red)
    ndvi2 = (nir + red)
    return numpy.divide(ndvi1, ndvi2, out=numpy.zeros_like(ndvi1).astype(float), where=ndvi2 != 0).squeeze()


def slice_raster(index: int, image: numpy.ndarray) -> numpy.ndarray:
    """
    Modifies the image.
    :param index - slicing index
    :param image - reference to the base image
    :return: sliced image
    """
    res_x, res_y = image.shape
    if res_x % index != 0:
        raise Exception("Raster slice index is not correct!")
    return (image.reshape(index, res_y // index, -1, res_x // index)
            .swapaxes(1, 2)
            .reshape(-1, res_y // index, res_x // index))


def glue_raster(image: numpy.ndarray, res_y: int, res_x: int):
    """
    Return an array of shape (res_x, res_y) where
    """
    n, old_y, old_x = image.shape
    if res_x % old_x != 0:
        raise Exception(f"Cannot glue raster image with shape: ({n},{old_y},{old_x})"
                        f" to an img with res: ({res_x},{res_y})")
    return image.reshape(res_y // old_y, -1, old_y, old_x).swapaxes(1, 2).reshape(res_y, res_x)


def mark_file_sentinel2_bands(filepath, band_names, band_wavelengths):
    if not band_names:
        band_names = [f"name:B{'0' if i != 8 or i < 10 else ''}{i}" for i in range(13)]
        band_names[8] += "a"
    if not band_wavelengths:
        band_wavelengths = [443, 490, 560, 665, 705, 740, 783, 865, 945, 1375, 1610, 2190]

    with rasterio.open(filepath) as dataset:
        for band_i in dataset.count:
            dataset.set_band_description(band_i,
                                         band_names[band_i - 1] + " wavelength:{}".format(band_wavelengths[band_i - 1]))
        print("New band descriptions:", dataset.descriptions)


def rescale_intensity(image, _min, _max):
    return exposure.rescale_intensity(image, in_range=(_min, _max), out_range=(0, 255)).astype(numpy.uint8)


def create_rgb_uint8(r, g, b, path, tile):
    gain = 1.5
    red = rescale_intensity(rasterio.open(r).read(1) * gain, 0, 4096)
    green = rescale_intensity(rasterio.open(g).read(1) * gain, 0, 4096)
    blue = rescale_intensity(rasterio.open(b).read(1) * gain, 0, 4096)

    rgb_profile = rasterio.open(r).profile
    rgb_profile['dtype'] = 'uint8'
    rgb_profile['count'] = 3
    rgb_profile['photometric'] = "RGB"
    rgb_profile['driver'] = "GTiff"
    rgb_profile['interleave'] = "PIXEL"
    rgb_profile['photometric'] = "YCBCR"
    rgb_profile['compress'] = "JPEG"
    rgb_profile['blockxsize'] = 256
    rgb_profile['blockysize'] = 256
    rgb_profile['nodata'] = 0
    log.debug(f"RGB PROFILE:\n{rgb_profile}")
    with rasterio.open(path + os.path.sep + f"{tile}_rgb.tif", 'w', **rgb_profile) as dst:
        for count, band in enumerate([red, green, blue], 1):
            dst.write(band, count)
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
        dst.update_tags(ns='rio_overview', resampling='nearest')


def build_mosaic(destination: str, paths: List[str], name: str = "mosaic", rgb=False, **kwargs) -> None:
    """
    Build mosaic from files. Using gdal vrt.
    @param destination - path where the mosaic will be available
    @param paths - paths to tiff files
    @param name - name of the result file
    @param rgb - yep
    """
    _destination = format_path(destination) + "mosaic.vrt"
    final_image = format_path(destination) + name + ".tif"
    escaped_paths = " ".join(f'"{p}"' for p in paths)
    process = subprocess.Popen(f"gdalbuildvrt -q \"{_destination}\" {escaped_paths}", shell=True,
                               stdout=subprocess.PIPE)
    process.wait()
    #  Experiment, NOTE: TILED is making artefacts on monochromatic pictures!
    if rgb:
        process = subprocess.Popen(f"gdal_translate -of GTiff --config GDAL_CACHE_MAX 128 -co \"TILED=YES\" "
                                   f"-co \"COMPRESS=JPEG\" -co "
                                   f"\"PHOTOMETRIC=YCBCR\" \"{_destination}\" \"{final_image}\" -q", shell=True,
                                   stdout=subprocess.PIPE)
        process.wait()
    else:
        process = subprocess.Popen(
            f"gdal_translate -of GTiff -co \"TILED=YES\" \"{_destination}\" \"{final_image}\" -q",
            shell=True, stdout=subprocess.PIPE)
        process.wait()
    process = subprocess.Popen(f"rm \"{_destination}\"", shell=True, stdout=subprocess.PIPE)
    process.wait()


# --------------- GRANULE UTILS ---------------

def verify_bands(img_paths: List[str], found_imgs: List[str], desired_bands: List[str], spatial: int) -> List[str]:
    """
    If user wants band that is not present for 10m spatial res. this finds it (either in 20m or 60m) and 
    after in the process this raster is resampled to the correct resolution.
    """
    # Helper function
    def grab_imgs(p, desired, _spatial):
        #  Check whether we would find something in this spatial res.
        res_bands = set(bands_for_resolution(_spatial)).intersection(desired)
        desired.difference(res_bands)
        result = []
        #  If not, return
        if len(res_bands) == 0:
            return []
        #  Traverse images for current spatial resolution and if we find band we were looking for
        #  add it's path to the list
        for _p in p:
            reg = re.findall('B[0-9]+A?|TCI|AOT|WVP|SCL', _p)
            if len(reg) != 0:
                reg = reg[-1]
            if reg in res_bands:
                result.append(_p)
                desired -= {reg}
        return result

    #  Current images for granule
    found_imgs = found_imgs
    desired_bands = set(desired_bands)
    #  First we will check what particular bands we are missing, then we look for it
    for found in found_imgs:
        f = re.findall('B[0-9]+A?|TCI|AOT|WVP|SCL', found)
        if len(f) != 0:
            f = f[-1]
        desired_bands -= {f}
    #  desired_bands => bands we want to find, if it's length is 0 it means we already have everything in found_imgs
    if len(desired_bands) == 0:
        return found_imgs
    #  [0:7] - 10m, [7:20] - 20m, [20::] - 60m
    if spatial != 10:
        p = img_paths[0:7]
        found_imgs += grab_imgs(p, desired_bands, 10)
        if len(desired_bands) == 0:
            return found_imgs
    if spatial != 20:
        p = img_paths[7:20]
        found_imgs += grab_imgs(p, desired_bands, 20)
        if len(desired_bands) == 0:
            return found_imgs
    if spatial != 60:
        p = img_paths[20::]
        found_imgs += grab_imgs(p, desired_bands, 60)
        if len(desired_bands) == 0:
            return found_imgs
    return found_imgs


def profile_for_rgb(profile: rasterio.profiles.Profile) -> rasterio.profiles.Profile:
    """
    Modify rasterio profile for RGB image.
    """
    profile['dtype'] = 'uint8'
    profile['count'] = 3
    profile['photometric'] = "RGB"
    profile['driver'] = "GTiff"
    profile['blockxsize'] = 256
    profile['blockysize'] = 256
    profile['nodata'] = 0
    return profile


def supported_granule_type(_type: str) -> bool:
    return _type in ["L1C", "L2A"]


def get_metadata_path(granule_path: str, granule_type: str) -> str:
    if not supported_granule_type(granule_type):
        raise NotImplemented
    return granule_path + os.path.sep + ("MTD_MSIL2A.xml" if granule_type == "L2A" else "MTD_MSIL1C.xml")


# --------------- S2CLOUDLESS UTILS ------------------
def download_l1c(granule: S2Granule) -> S2Granule:
    """
    Used with S2CLDLESS.
    """
    working_path_1c = granule.path
    downloader = Download.Downloader(USER_NAME, PASSWORD, working_path_1c, granule_identifier=[granule.l1c_identifier],
                                     product_type="S2MSI1C")
    p = list(downloader.download_granule_bands(S2CloudlessPerPixel.bands_l1c))[-1]
    l1c_raster = p + os.path.sep + os.listdir(p)[0]
    # Data will be automatically resampled during the creation of the granule
    return S2Granule(l1c_raster, 160, S2CloudlessPerPixel.bands_l1c, granule_type="L1C")


def create_cloud_product(granule: S2Granule, mask=False) -> str:
    """
    Create cloud mask with s2cldless.
    """
    # Input is l1c granule and if we should compute binary mask or prob mask
    assert granule.granule_type == "L1C"
    #  Cloudless time, compute mask
    data = granule.stack_bands(desired_order=S2CloudlessPerPixel.bands_l1c, dstack=True) / 10000.0
    granule.free_resources()
    product = None
    if mask:
        product = S2CloudlessPerPixel.cloud_detector.get_cloud_masks(data)
    else:
        product = S2CloudlessPerPixel.cloud_detector.get_cloud_probability_maps(data)
        _conv = disk(5) / numpy.sum(disk(5)) # avg. filter
        product = convolve(product, _conv)
    # The flaw of this prob mask is that it has some points that are marked as clear and yet they are cloudy (it's 
    # sourroundings aswell), maybe try median or avg filtering, maybe max filt.

    # Since s2cloudless does not detect no-data value, we need to filter them out here
    # How to get no-data in "data" variable => transform array (120, 120, 10) to (10, 120, 120)
    # then compare =>  == 0, it gives us 1 where it is zero (no-data value), squeeze along dimension, it will give us (120, 120)
    # numpy.all => If the image has first pixel in every band 0 then after first two steps we receive for
    # data[:, 0, 0] = [True, True, ..., True], numpy all squeezes it to True, after all multiply these data with 2
    data = numpy.all(numpy.transpose(data, (2, 0, 1)) == 0, axis=0)
    #  mask product with no-data values, no-data -> 2 and cloud. prob. in interval [0, 1]
    product = numpy.ma.masked_array(data=product, mask=data, fill_value=int(2)).filled()

    # obtain profile from parsed band
    band = granule.get_initialized_bands()[-1]
    if len(band) < 1:
        raise NotImplemented
    prof = granule[band].profile
    print(prof)
    prof['driver'] = "GTiff"
    prof['dtype'] = "int" if mask else "float64"
    path = os.path.join(granule.path, "CLD.tif")
    with rasterio.open(path, "w", **prof) as cld:
        cld.write(product, 1)
    return path
