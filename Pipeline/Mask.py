"""
Faster calculations with numba
"""
from numba import njit, prange
import math
import numpy as np


class S2JIT:

    # @staticmethod
    # @njit(parallel=True)
    # def s2_ndvi_pixel_masking(ndvi, offset=0):
    #     """
    #     Per pixel masking of the best ndvi pixels.
    #     @param ndvi: ndvi of each worker
    #     @param offset: offsetting granules, making ndvi with granules[10:20] -> offset = 9
    #     :return: mask of indices of the granules and worker map of used granules
    #     """
    #     result = np.zeros(shape=(ndvi.shape[1], ndvi.shape[2]))
    #     workers = np.zeros(ndvi.shape[0] + offset)
    #     x_res = ndvi.shape[1]
    #     y_res = ndvi.shape[2]
    #     for y in prange(y_res):
    #         for x in prange(x_res):
    #             r = ndvi[:, x, y].argmax() + offset  # if we had to split this operation
    #             if workers[r] == 0:
    #                 workers[r] = 1
    #             result[x, y] = r
    #     return result, workers

    @staticmethod
    @njit
    def s2_median_analysis(data, median_values):
        """
        JIT-ed method for picking median, purpose is to avoid picking no data values with basic np.median.
        """
        res_x, res_y = median_values.shape
        for y in range(res_y):
            for x in range(res_x):
                if median_values[y][x] == 0:
                    pick = 0
                    inspected_arr = data[:, y, x]
                    for i in range(len(inspected_arr) // 2 - 1, len(inspected_arr)):
                        if inspected_arr[i] != 0:
                            pick = inspected_arr[i]
                            break
                    median_values[y][x] = pick
        return median_values

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
                # np.sum is no data indicator, we don't want pixels with no data in image
                for i in range(len(data)):
                    if ndvi[i][y, x] > _max_val and np.sum(data[i][:, y, x]) > 0:
                        _max_val = ndvi[i][y, x]
                        index = i
                if ndvi_res[y, x] <= ndvi[index][y, x]:
                    ndvi_res[y, x] = ndvi[index][y, x]
                    doy[y, x] = doys[index]
                    result[:, y, x] = data[index][:, y, x]
        return result, doy

    @staticmethod
    @njit
    def s2_cloud_probability_analysis(current_data, current_masks, current_doy, result, doy, final_mask) -> None:
        """
        Run over the data and pick the best pixel.
        :param current_data: stacked data, sorted by the doy(date, ascending)
        :param current_masks: L1C masks for the current data, for current_data[i] we have got current_masks[i]
        :param current_doy: doy for current_data
        :param result: intermediate array for result
        :param doy: intermediate array for doy result
        :param final_mask: intermediate array for probability mask result
        :return: None, we directly manipulate the result, doy and final_mask
        """
        #  This is somewhat similar to the ndvi function
        res_x, res_y = result.shape[1], result.shape[2]
        for y in range(res_y):
            for x in range(res_x):
                _min_val = math.inf
                index = 0
                for i in range(len(current_masks)):
                    #  do not take no data pixels and take the most recent pixels
                    if 255 > current_masks[i][y, x] <= _min_val:
                        _min_val = current_masks[i][y, x]
                        index = i
                if abs(_min_val - final_mask[y, x]) <= 20:
                    final_mask[y, x] = _min_val
                    result[:, y, x] = current_data[index][:, y, x]
                    doy[y, x] = current_doy[index]
