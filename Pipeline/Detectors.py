from Pipeline.Granule import S2Granule
from Pipeline.GranuleCalculator import GranuleCalculator
import numpy as np
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
        ndvi = GranuleCalculator.s2_ndvi(g)
        # linear transformation
        return 1.5 * (ndvi + 1)
