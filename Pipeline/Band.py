import numpy as np
import gc
from osgeo import gdal
from Pipeline.utils import *
import numpy as np
from Pipeline.logger import log
import rasterio

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
        self.profile = None
        with rasterio.open(self.path) as dataset:
            self.profile = dataset.profile
        self.slice_index = slice_index
        self.raster_image = None
        self._was_raster_read = False
        if load_on_init:
            self.load_raster()

    def load_raster(self) -> None:
        """
        Since the raster images might be quite big,
        we do not want them casually lay in our memory therefore we can choose when we want them.
        """
        if self._was_raster_read:
            return
        with rasterio.open(self.path) as dataset:
            self.raster_image = dataset.read()
        self._was_raster_read = True
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
        self.raster_image = None
        self._was_raster_read = False

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
