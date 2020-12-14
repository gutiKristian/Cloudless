"""
Faster calculations with numba
"""
from numba import njit
import math


class S2JIT:

    @staticmethod
    @njit
    def s2_pixel_analysis(ndvi, data, doys, result, doy, res_x, res_y):
        for y in range(res_y):
            for x in range(res_x):
                _max_val = -math.inf
                index = 0
                # i - worker index, y,x are coords
                for i in range(len(data)):
                    if ndvi[i][y][x] > _max_val:
                        _max_val = ndvi[i][y][x]
                        index = i
                doy[y][x] = doys[index]
                #  Access worker bands, j - band index
                for j in range(len(data[index])):
                    result[j][y][x] = data[index][j][y][x]
        return result, doy
