from joblib import Parallel, delayed

from Pipeline import Worker
from Pipeline.Granule import S2Granule
from Pipeline.GranuleCalculator import GranuleCalculator
from Pipeline.logger import log
from Pipeline.utils import *
from timeit import default_timer as timer
import gdal
import numpy as np
from Pipeline.Plotting import Plot
import pandas
import requests
from PIL import Image, Jpeg2KImagePlugin
import glob
import rasterio
import cv2
import concurrent.futures
from matplotlib.pyplot import imread
import multiprocessing as mp
import threading
from Download import Sentinel2
from Pipeline.Worker import S2Worker
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime
from Pipeline.Task import PerTile, NdviPerPixel


def a(path):
    PerTile.perform_computation(
        S2Worker(path=path, spatial_resolution=20, output_bands=bands_for_resolution(20), slice_index=45))


if __name__ == '__main__':
    # paths = glob.glob(
    #     "/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/data/33UXQ/S2A_MSIL2A_20210226T095031_N0214_R079_T33UXQ_20210226T122801.SAFE/*.jp2")
    # # nprocs = mp.cpu_count()
    # start = timer()
    # # imgs = []
    # print("RUNNING")
    # PerTile.perform_computation(worker1)
    # PerTile.perform_computation(worker2)
    # for p in ["/home/xgutic/Desktop/wrk/"]:
    #     a(p)
    # p = Pool(cpu_count())    # all_data = p.map(a, ["/home/xgutic/Desktop/wrk/", "/home/xgutic/Desktop/wrk2/",
    #                      "/home/xgutic/Desktop/wrk2 (copy)/", "/home/xgutic/Desktop/wrk (copy)/"])
    # with concurrent.futures.ThreadPoolExecutor() as e:
    #     e.map(a, ["/home/xgutic/Desktop/wrk/", "/home/xgutic/Desktop/wrk2/"])
    # path = "/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/data/33UXQ/S2A_MSIL2A_20210226T095031_N0214_R079_T33UXQ_20210226T122801.SAFE/T33UXQ_20210226T095031_B07_20m.jp2"
    # i = Image.open(path)
    # a = i.load()
    # # s2 = Worker.S2Worker(path, 20, 18)
    # # s2.per_tile(constraint=4)
    # print(type(a))
    # a = gdal.Open(path, gdal.gdalconst.GA_ReadOnly)
    # a = a.GetRasterBand(1).ReadAsArray()
    # print(a)
    # with rasterio.open(path, driver='JP2OpenJPEG') as dataset:
    #     print(dataset.profile)
    #     dataset.profile.update(blockxsize=256, blockysize=256, tiled=True)
    #     print(dataset.profile)
    #     a = dataset.read(1)
    #     print(a)
    # end = timer()
    # print(end - start)
    # from Pipeline import Granule
    # Granule.S2Granule("/media/xgutic/ECF1-46C6/pertile/33UUS/result/", 20, bands_for_resolution(20))
    # aa = [(16.44978919891358, 49.301000145032816), (16.424463393402544, 49.00468664741558),
    #      (16.995701006596047, 49.054502687187835),
    #      (17.023840790497204, 49.266123693767184), (16.44978919891358, 49.301000145032816)]
    # d = Sentinel2.Downloader('kristianson12', 'kikaakiko', '/home/xgutic', polygon=aa,
    #                          date=(datetime.datetime(2019, 5, 1), datetime.datetime(2019, 8, 1)))
    # paths = d.download_all_bands('20m')
    # for p in paths:
    #     a(p)
    g = S2Granule("/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/T33UWR/result", 20,
                  ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "AOT", "SCL"])
    GranuleCalculator.build_mosaics([g], "/home/xgutic")
