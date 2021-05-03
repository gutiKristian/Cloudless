from Pipeline.Granule import S2Granule
from Pipeline.GranuleCalculator import GranuleCalculator
import numpy as np
from typing import List
from skimage.exposure import rescale_intensity


class S2Detectors:
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
        c = g["SCL"] == 0  # no data
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
        @param g - granule.
        """
        pass

