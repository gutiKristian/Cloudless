"""
Faster calculations with numba
"""
from numba import njit
import math


class S2JIT:

    @staticmethod
    @njit
    def s2_ndvi_pixel_analysis(ndvi, data, doys, result, doy, res_x, res_y):
        """
        Per pixel analysis for NDVI masking
        :param ndvi: List[numpy.ndarray] - list of 2D arrays
        :param data: List[numpy.ndarray] - list of 3D arrays, stacked data
        :param doys: numpy.array - 1D array
        :param result: numpy.ndarray - 3D array
        :param doy: numpy.ndarray - 2D array
        :param res_x: int, different for 10, 20, 60
        :param res_y: int, different for 10, 20, 60
        :return: masked numpy.ndarray (3D) and (2D) raster with 'day of the year' values
        """
        for y in range(res_y):
            for x in range(res_x):
                _max_val = -math.inf
                index = 0
                # i - worker index, y,x are coords
                for i in range(len(data)):
                    if ndvi[i][y, x] > _max_val:
                        _max_val = ndvi[i][y, x]
                        index = i
                doy[y, x] = doys[index]
                #  Access worker bands, j - band index
                for j in range(len(data[index])):
                    result[j, y, x] = data[index][j, y, x]  # TODO: check this result[:, y, x] = data[index][:, y, x]
        return result, doy                                  # TODO : no loop needed
