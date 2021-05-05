import logging
import os
import skimage.transform

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
        c = g["SCL"] < 1  # no data
        return (a & b) | c

    @staticmethod
    def max_ndvi(g: S2Granule) -> np.ndarray:
        """
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
    def sentinel_cloudless(g: S2Granule) -> np.ndarray:
        """
        Cloud detection based on machine learning algorithm by SentinelHub.
        Granule is identified and accompanying L1C dataset is downloaded and mas computed.
        In future we might save these mask and use them but for now we will download the data and compute the mask
        over and over.
        @param g - granule.
        """
        # Workspace preparation phase
        working_path = g.path + os.path.sep + "L1C"
        try:
            os.mkdir(working_path)
        except NotImplementedError:
            log.error("Error while creating mask for {}".format(g.path))
            #  Automatically discarded (taken as cloudy)
            return np.ones(shape=(s2_get_resolution(g.spatial_resolution))) * 255  # no data

        #  Data preparation phase
        #  We find the accompanying tile with data-take and mercator
        mercator = extract_mercator(g.path)
        # get credentials from file
        try:
            downloader = Downloader("kristianson12", "mosvegcz", root_path=working_path,
                                    date=(g.data_take, g.data_take),
                                    product_type="S2MSI1C", mercator_tiles=[mercator])
        except IncorrectInput:
            log.error("Did not find corresponding l1c this dataset wont be taken")
            res = np.ones(shape=(s2_get_resolution(g.spatial_resolution))) * 255
            if g.slice_index > 1:
                return slice_raster(g.slice_index, res)
            return res

        l1c_raster = None
        necessary_bands = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
        #  This is generalized download, in this case we expect only one iteration
        for p in downloader.download_l1c(necessary_bands):
            l1c_raster = p + os.path.sep + os.listdir(p)[0]

        # Data will be automatically resampled during the creation of the granule
        l1c_granule = S2Granule(l1c_raster, 160, necessary_bands, granule_type="L1C")

        #  Cloudless time, compute mask
        data = l1c_granule.stack_bands(necessary_bands, dstack=True) / 10000.0
        l1c_granule.free_resources()

        cloud_detector = S2PixelCloudDetector()
        # CPL = cloud_detector.get_cloud_probability_maps(data)
        cml = cloud_detector.get_cloud_masks(data)
        #  Mask is in 160m spatial resolution, we need to up-sample to working spatial res.
        #  0 (no clouds), 1 (clouds), 255 (no data)
        cml = skimage.transform.resize(cml, order=0, output_shape=s2_get_resolution(g.spatial_resolution))
        # For some reason it marks no data as no cloud therefore we will filter them out with SCL
        nodata = g["SCL"] < 1
        if g.slice_index != 1:
            x, y = s2_get_resolution(g.spatial_resolution)
            nodata = glue_raster(nodata, y, x)
        cml = np.ma.masked_array(data=cml, mask=nodata, fill_value=255).filled()
        #  Since we can make use of this detector in per pixel let's not slice it automatically but based on granule
        if g.slice_index > 1:
            return slice_raster(g.slice_index, cml)
        return cml
