import os
import re
import numpy
from typing import *


# --------------- FILE UTILS ---------------

def is_dir_valid(path: str) -> bool:
    return os.path.isdir(path)


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
    if not s2_is_spatial_correct(spatial):
        raise Exception("This spatial resolution does not exist in the sentinel 2 context")
    if spatial == 10:
        return 10980, 10980
    if spatial == 20:
        return 5490, 5490
    return 1830, 1830


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
        return ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "AOT"]
    elif spatial_resolution == 10:
        return ["B02", "B03", "B04", "B08", "AOT"]
    return ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12", "AOT"]  # 60


# --------------- RASTER UTILS ---------------


def ndvi(red: numpy.ndarray, nir: numpy.ndarray) -> numpy.ndarray:
    ndvi1 = (nir - red)
    ndvi2 = (nir + red)
    return numpy.divide(ndvi1, ndvi2, out=numpy.zeros_like(ndvi1), where=ndvi2 != 0).squeeze()


def slice_raster(index: int, image: numpy.ndarray) -> None:
    """
    Modifies the image, in-situ function.
    :param index - slicing index
    :param image - reference to the base image
    :return: sliced image
    """
    pass
