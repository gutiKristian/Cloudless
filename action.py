from Pipeline import Sentinel2Runner
from Pipeline.logger import log
from Pipeline.utils import *
from timeit import default_timer as timer
import gdal
import numpy as np
from Pipeline.Plotting import Plot

if __name__ == '__main__':
    start = timer()
    path = "/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/data/33UXQ"
    s2 = Sentinel2Runner.S2Runner(path, 20, 18)
    s2.per_tile(constraint=4)
    end = timer()
    print(end - start)
    # set1 = '/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/data/33UXQ/S2A_MSIL2A_20210301T100031_N0214_R122_T33UXQ_20210301T115651.SAFE/T33UXQ_20210301T100031_B04_20m.jp2'
    # set2 = '/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/data/33UXQ/result/B05_20.tif'
    # set3 = '/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/data/33UXQ/S2A_MSIL2A_20210226T095031_N0214_R079_T33UXQ_20210226T122801.SAFE/T33UXQ_20210226T095031_B04_20m.jp2'
    # s1 = gdal.Open(set1)
    # s2 = gdal.Open(set2)
    # s3 = gdal.Open(set3)
    # n1 = s1.GetRasterBand(1).ReadAsArray()
    # n2 = s2.GetRasterBand(1).ReadAsArray()
    # n3 = s3.GetRasterBand(1).ReadAsArray()
    # from Pipeline import Plotting
    # Plotting.Plot.plot_mask(n1 == n3)
    # Plotting.Plot.plot_mask(n1 == n2)
    # Plotting.Plot.plot_mask(n2 == n3)
    # Plotting.Plot.plot_mask((n1 == n2) | (n3 == n2))
    # print(n3.shape)
    # print(np.sum(n1 == n3))
    # print(np.sum(n1 == n2))
    # print(np.sum(n3 == n2))
