import shutil

from osgeo import gdal
from Pipeline.utils import *
import numpy as np
from Pipeline.logger import log
import rasterio
from rasterio import Affine, MemoryFile
from rasterio.warp import calculate_default_transform, reproject
import subprocess

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

    def load_raster(self) -> None:
        """
        Since the raster images might be quite big,
        we do not want them casually lay in our memory therefore we can choose when we want them.
        """
        if self._was_raster_read:
            return
        with rasterio.open(self.path) as dataset:
            self.raster_image = dataset.read(1)
        self._was_raster_read = True
        if self.slice_index > 1:
            self.raster_image = slice_raster(self.slice_index, self.raster_image)

    def raster(self) -> np.array:
        """
        Basic getter.
        """
        if self._was_raster_read:
            return self.raster_image
        self.load_raster()
        return self.raster_image

    def band_reproject(self, t_srs='EPSG:32633', delete=True):
        """
        Reproject band to the given band to the other UTM zone.
        @param t_srs: target srs
        @param delete: delete source file after the reprojection
        """
        # GDAL version
        # process = subprocess.Popen(f"gdalwarp \"{self.path}\" -s_srs  -t_srs {t_srs} -co TILED=TRUE \"{new_path}\" ")
        # process.wait()
        # rasterio version
        new_path, _ = os.path.splitext(self.path)
        new_path += '.tif'
        with rasterio.open(self.path) as src:
            transform, width, height = calculate_default_transform(src.crs, t_srs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': t_srs,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open(new_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=t_srs)
        if delete:
            shutil.rmtree(self.path)
        self.path = new_path

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
                resampling=rasterio.enums.Resampling.bilinear
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
