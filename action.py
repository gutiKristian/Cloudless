from joblib import Parallel, delayed

from Pipeline import Worker
from Pipeline.Granule import S2Granule
from Pipeline.GranuleCalculator import GranuleCalculator
from Pipeline.logger import log
from Pipeline.utils import *
from timeit import default_timer as timer
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
        S2Worker(path=path, spatial_resolution=20, output_bands=bands_for_resolution(20), slice_index=5))


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
    # p = "/home/xgutic/Desktop/T33UXR/S2A_MSIL2A_20200912T100031_N0214_R122_T33UXR_20200912T114911.SAFE/GRANULE/L2A_T33UXR_A027289_20200912T100044/IMG_DATA/R20m/T33UXR_20200912T100031_AOT_20m.jp2"
    # p2 = "/home/xgutic/Desktop/T33UXR/result/AOT_20.tif"
    # a = rasterio.open(p)
    # b = a.profile
    # a.close()
    # a = rasterio.open(p2)
    # c = a.profile
    # a.close()
    # path = "/home/xgutic/Desktop/T33UXR"
    # # i = Image.open(path)
    # # a = i.load()
    # s2 = Worker.S2Worker(path, 20, 1)
    # NdviPerPixel.perform_computation(s2)
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
    # d = Sentinel2.Downloader('kristianson12', 'mosvegcz', '/home/xgutic/Desktop', polygon=aa,
    #                          date=(datetime.datetime(2021, 3, 24), datetime.datetime(2021, 3, 30)))
    # paths = d.download_all_bands('20m')
    import time
    start = time.time()
    p = "/home/xgutic/Desktop/UXR"
    PerTile.perform_computation(S2Worker(p, 20, 45))
    end = time.time()
    print(end - start)
    # for p in ["/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/T33UXQ","/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/T33UWQ"]:
    #     a(p)
    # g = S2Granule("/home/xgutic/dev/mosveg/temp_jobs/KRWyfMs6kW9f4frgu2rY24/T33UWR/result", 20,
    #               ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "AOT", "SCL"])
    # GranuleCalculator.build_mosaics([g], "/home/xgutic")
    # manifest parsing
    # a = "/home/xgutic/Downloads/manifest.safe"
    # from xml.dom import minidom
    # xmldoc = minidom.parse(a)
    # items = xmldoc.getElementsByTagName('dataObject')
    # for item in items:
    #     print(item.attributes['ID'].value)
    #     x = item.childNodes[1].childNodes[3].firstChild.nodeValue
    #     print(x)
    #
    # process = subprocess.Popen(f"md5sum /home/xgutic/T33UXQ/S2B_MSIL2A_20190801T095039_N0213_R079_T33UXQ_20190801T124448.SAFE/T33UXQ_20190801T095039_TCI_60m.jp2", shell=True,
    #                            stdout=subprocess.PIPE)
    # out = process.communicate()
    # print(out[])
    # # print(out.decode().split(" ")[0])
