import os
import re
from typing import *


def is_dir_valid(path: str) -> bool:
    return os.path.isdir(path)


def is_file_valid(path: str) -> bool:
    return os.path.isfile(path)


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


def get_subdirectories(path: str) -> List[str]:
    return [el.path for el in os.scandir(path) if el.is_dir()]


def get_files_in_directory(path: str) -> List[str]:
    return [file for file in os.listdir(path) if is_file_valid(file)]


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
