"""
Faster calculations with numba
"""
from numba import njit, prange
import math
import numpy as np


class S2JIT:

    @staticmethod
    @njit(parallel=True)
    def s2_ndvi_pixel_masking(ndvi, offset=0):
        """
        Per pixel masking of the best ndvi pixels.
        @param ndvi: ndvi of each worker
        @param offset: offsetting workers, making ndvi with workers[10:20] -> offset = 9
        :return: mask of indices of the workers and worker map of used workers
        """
        result = np.zeros(shape=(ndvi.shape[1], ndvi.shape[2]))
        workers = np.zeros(ndvi.shape[0] + offset)
        x_res = ndvi.shape[1]
        y_res = ndvi.shape[2]
        for y in prange(y_res):
            for x in prange(x_res):
                r = ndvi[:, x, y].argmax() + offset  # if we had to split this operation
                if workers[r] == 0:
                    workers[r] = 1
                result[x, y] = r
        return result, workers

    @staticmethod
    @njit
    def s2_ndvi_pixel_analysis(ndvi, ndvi_res, data, doys, result, doy, res_x, res_y):
        """
        Per pixel analysis for NDVI masking
        :param ndvi: List[numpy.ndarray] - list of 2D arrays
        :param ndvi_res: numpy.array - 2D array which holds current maximum value of ndvi for pixel in that position
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
                # i - worker index; y,x are coords
                for i in range(len(data)):
                    if ndvi[i][y, x] > _max_val:
                        _max_val = ndvi[i][y, x]
                        index = i
                if ndvi_res[y, x] <= ndvi[index][y, x]:
                    doy[y, x] = doys[index]
                    result[:, y, x] = data[index][:, y, x]
        return result, doy
