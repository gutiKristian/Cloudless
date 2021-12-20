import logging
import os
import skimage.transform

import Pipeline.utils
from Download.DownloadExceptions import IncorrectInput
from Pipeline.Granule import S2Granule
from Pipeline.GranuleCalculator import GranuleCalculator
import numpy as np
from Pipeline.logger import log
from typing import List
from Pipeline.utils import extract_mercator, s2_get_resolution, slice_raster, glue_raster
from Download.Sentinel2 import Downloader
from skimage.exposure import rescale_intensity
from s2cloudless import S2PixelCloudDetector
from Pipeline.Plotting import Plot
from Pipeline.utils import USER_NAME, PASSWORD, download_l1c, create_cloud_product


class S2Detectors:
    """
    Detectors for Per-Tile.
    """

    @staticmethod
    def scl(g: S2Granule) -> np.ndarray:
        """
        Detector based on SCL mask provided by ESA.
        :param g: granule which contains the data
        :return: array of zeros and ones, based on this array best tile is selected
        """
        # filter clouds
        a = g["SCL"] > 7
        b = g["SCL"] < 11
        c = (g["SCL"] < 1) * 10  # no data - punish more
        return (a & b) | c

    @staticmethod
    def max_ndvi(g: S2Granule) -> np.ndarray:
        """
        Experimental...shows how easy it is to add new detectors.
        Detector based on the vegetation index in the area.
        For each tile veg. index is computed and normalized from <-1,1> to <0,3>.
        Based on this the area where the sum of the ndvi is biggest is chosen.
        :param g: granule which contains the data
        :return: numpy array
        """
        ndvi = GranuleCalculator.s2_ndvi(g) * (-1)

        # no data values
        mask = (g['B8A'].raster() == 0) & (g['B04'].raster() == 0)
        ndvi = np.ma.masked_array(data=ndvi, mask=mask, fill_value=float(4)).filled()
        # linear transformation
        return 1.5 * (ndvi + 1)

    @staticmethod
    def sentinel_cloudless(g: S2Granule, probability: bool = False) -> np.ndarray:
        """
        Cloud detection based on machine learning algorithm by SentinelHub.
        Granule is identified and accompanying L1C dataset is downloaded and mas computed.
        In future we might save these mask and use them but for now we will download the data and compute the mask
        over and over.
        @param g - granule.
        @param probability - if we want the function to return probability mask instead of 0,1,255 mask
        :return: based on the probability parameter, we return either mask of 0,1,255 or probability mask <0, 255>
        """
        l1c_granule = download_l1c(g)

        l1c_granule.add_another_band(create_cloud_product(l1c_granule, mask=True), "CLD")

        # nodata = g["SCL"] < 1
        # product = np.ma.masked_array(data=product, mask=nodata, fill_value=255).filled()

        # Slicing is done automatically
        return l1c_granule["CLD"].raster()

