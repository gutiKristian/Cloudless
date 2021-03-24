import numpy as np
import gc
from osgeo import gdal
from Pipeline.utils import *
import numpy as np
from Pipeline.logger import log
gdal.UseExceptions()


class Band:
    def __init__(self, path: str, load_on_init: bool = False, slice_index: int = 1):
        if not is_file_valid(path):
            raise FileNotFoundError("Raster does not exist!")
        if slice_index > 1:
            if not is_supported_slice(slice_index):
                log.warning("Unsupported slice index, choosing the closest one..")
                slice_index = find_closest_slice(slice_index)
                log.info(f"New slice index: {slice_index}")
            log.info("Raster is going to be sliced")

        self.path = path
        self.slice_index = slice_index
        self.gdal_dataset = None
        self.raster_image = None
        self.geo_transform = None
        self.projection = None
        self._was_raster_read = False
        self.is_opened = False
        if load_on_init:
            self.load_raster()

    def init_gdal(self) -> None:
        """
        This is method initializes the raster image (opens it with gdal and grabs the meta-data).
        :return: None
        """
        if self.is_opened:
            return
        try:
            self.gdal_dataset = gdal.Open(self.path)
            self.projection = self.gdal_dataset.GetProjection()
            self.geo_transform = self.gdal_dataset.GetGeoTransform()
            self.is_opened = True
        except Exception as e:
            raise Exception("GDAL thrown an error: ", e)

    def load_raster(self) -> None:
        """
        Since the raster images might be quite big,
        we do not want them casually lay in our memory therefore we can choose when we want them.
        """
        if self._was_raster_read:
            return
        if not self.is_opened:
            self.init_gdal()
        self._was_raster_read = True
        self.raster_image = self.gdal_dataset.GetRasterBand(1).ReadAsArray()
        if self.slice_index > 1:
            log.debug(f"Slicing raster with slice index: {self.slice_index}")
            self.raster_image = slice_raster(self.slice_index, self.raster_image)
            log.debug(f"Slicing successful, shape:({self.raster_image.shape})")

    def raster(self) -> np.array:
        """
        Basic getter.
        """
        if self._was_raster_read:
            return self.raster_image
        self.load_raster()
        return self.raster_image

    def free_resources(self) -> None:
        """
        Delete the pointers to the data and call garbage collector to free the memory.
        """
        if self.gdal_dataset is not None:
            self.gdal_dataset = None
        if self.raster_image is not None:
            self.raster_image = None
        self._was_raster_read = False
        self.is_opened = False

    def __gt__(self, other: int):
        """
        :param other: threshold value
        """
        return np.copy(self.raster()) > other

    def __lt__(self, other: int):
        """
        :param other: threshold value
        """
        return np.copy(self.raster()) < other
