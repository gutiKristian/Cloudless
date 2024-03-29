import shutil

from osgeo import gdal
from Pipeline.utils import *
import numpy as np
from Pipeline.logger import log
import rasterio
from rasterio import Affine, MemoryFile
from rasterio.warp import calculate_default_transform, reproject
import subprocess
from shapely.geometry import Polygon
from rasterio.mask import mask
gdal.UseExceptions()


class Band:
    def __init__(self, path: str, load_on_init: bool = False, slice_index: int = 1):
        if not is_file_valid(path):
            raise FileNotFoundError("Raster does not exist!")
        if slice_index > 1:
            if not is_supported_slice(slice_index):
                log.warning("Unsupported slice index, choosing the closest one..")
                slice_index = find_closest_slice(slice_index)

        self.path = path
        self.profile = None
        with rasterio.open(self.path) as dataset:
            self.profile = dataset.profile
        self.slice_index = slice_index
        self.raster_image = None
        self._was_raster_read = False
        if load_on_init:
            self.load_raster()
        self.__rasterio_reference = None
        self.polygon = None

    def load_raster(self) -> None:
        """
        Since the raster images might be quite big,
        we do not want them casually lay in our memory therefore we can choose when we want them.
        """
        if self._was_raster_read:
            return
        with rasterio.open(self.path) as dataset:
            if self.polygon is None:
                self.raster_image = dataset.read(1)
            else:
                # Expects iterable and Polygon ain't iterable therefore [pol]
                self.raster_image, _ = mask(dataset, [self.polygon], crop=True)
                self.raster_image = self.raster_image.squeeze()
        self._was_raster_read = True
        if self.slice_index > 1:
            self.raster_image = slice_raster(self.slice_index, self.raster_image)

    def rasterio_ref(self):
        """
        Used within median to avoid frequent opening and closing.
        """
        if self.__rasterio_reference is None:
            self.__rasterio_reference = rasterio.open(self.path)
        return self.__rasterio_reference

    def raster(self) -> np.array:
        """
        Basic getter.
        """
        if self._was_raster_read:
            return self.raster_image
        self.load_raster()
        return self.raster_image

    def band_reproject(self, t_srs='EPSG:32633', delete=True) -> str:
        """
        Reproject band to the given band to the other UTM zone.
        REPROJECTION SHOULD BE DONE AFTER THE PIPELINE BECAUSE REPROJECTION MIGHT CHANGE THE RESOLUTION OF RASTER.
        @param t_srs: target srs
        @param delete: delete source file after the reprojection
        """
        if self.profile['crs'] == t_srs:
            return self.path
        new_path, _ = os.path.splitext(self.path)
        new_path += '.tif'
        # GDAL version
        process = subprocess.Popen(
            f"gdalwarp \"{self.path}\" -s_srs {self.profile['crs']} -t_srs {t_srs} \"{new_path}\"",
            shell=True)
        process.wait()
        # rasterio version
        # with rasterio.open(self.path) as src:
        #     transform, width, height = calculate_default_transform(src.crs, t_srs, src.width, src.height, *src.bounds)
        #     kwargs = src.meta.copy()
        #     kwargs.update({
        #         'crs': t_srs,
        #         'transform': transform,
        #         'width': width,
        #         'height': height
        #     })
        #     with rasterio.open(new_path, 'w', **kwargs) as dst:
        #         for i in range(1, src.count + 1):
        #             reproject(
        #                 source=rasterio.band(src, i),
        #                 destination=rasterio.band(dst, i),
        #                 src_transform=src.transform,
        #                 src_crs=src.crs,
        #                 dst_transform=transform,
        #                 dst_crs=t_srs)
        if delete:
            os.remove(self.path)
        self.path = new_path
        with rasterio.open(self.path) as dataset:
            self.profile = dataset.profile
        return new_path

    def resample(self, sample_factor, delete=False):
        transform = None
        width = 0
        height = 0
        data = None
        with rasterio.open(self.path, "r") as dataset:
            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * sample_factor),
                    int(dataset.width * sample_factor)
                ),
                resampling=rasterio.enums.Resampling.nearest
            )
            height = dataset.height * sample_factor
            width = dataset.width * sample_factor
            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
            )
            self.profile = dataset.profile
        _, ext = os.path.splitext(self.path)
        self.profile.update(transform=transform, width=width, height=height)
        with rasterio.open(self.path + "_res" + ext, 'w', **self.profile) as dataset:
            for i in range(dataset.count):
                dataset.write(data[i], i + 1)
        if delete:
            os.remove(self.path)
        self.path = self.path + "_res" + ext

    def free_resources(self) -> None:
        """
        Delete the pointers to the data and call garbage collector to free the memory.
        """
        self.raster_image = None
        self._was_raster_read = False

    def __del__(self):
        if self.__rasterio_reference is not None:
            self.__rasterio_reference.close()

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
