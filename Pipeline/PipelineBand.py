import numpy as np
import gc
from osgeo import gdal
from Pipeline.utils import *

gdal.UseExceptions()


class Band:
    def __init__(self, path: str, load_on_init: bool = False, slice_index: int = 1):
        if not is_file_valid(path):
            raise FileNotFoundError("Raster does not exist!")
        self.path = path
        self.slice_index = slice_index
        self._gdal = None
        self.raster_image = None
        self.geo_transform = None
        self.projection = None
        self._was_raster_read = False
        self.is_opened = False
        if load_on_init:
            self.load_raster()

    def init_gdal(self) -> gdal.Dataset:
        if self.is_opened:
            return self._gdal
        try:
            self._gdal = gdal.Open(self.path)
            self.projection = self._gdal.GetProjection()
            self.geo_transform = self._gdal.GetGeoTransform()
            self.is_opened = True
        except Exception as e:
            raise Exception("GDAL thrown an error: ", e)
        return self._gdal

    def load_raster(self) -> None:
        if self._was_raster_read:
            return
        if not self.is_opened:
            self.init_gdal()
        self._was_raster_read = True
        self.raster_image = self._gdal.GetRasterBand(1).ReadAsArray()
        if self.slice_index > 1:
            slice_raster(self.slice_index, self.raster_image)

    def raster(self) -> np.array:
        if self._was_raster_read:
            return self.raster_image
        self.load_raster()
        return self.raster_image

    def free_resources(self) -> None:
        del self._gdal
        del self.raster_image
        self._was_raster_read = False
        self.is_opened = False
        gc.collect()  # Garbage collector

    def __gt__(self, other: int):
        #  call raster method in case raster image has not been initialized
        return np.copy(self.raster()) > other

    def __lt__(self, other: int):
        #  call raster method in case raster image has not been initialized
        return np.copy(self.raster()) < other
