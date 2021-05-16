import numpy as np

from Pipeline.Granule import *
from Pipeline.utils import *
from osgeo import gdal
from typing import Callable
import rasterio
import os
from rasterio.profiles import Profile as RasterioProfile
from Pipeline.utils import profile_for_rgb


class GranuleCalculator:

    @staticmethod
    def save_band_rast(raster: np.ndarray, path: str, prof: RasterioProfile = None, dtype: np.dtype = None,
                       driver: str = None) -> str:
        """
        Save numpy array as raster image.
        Returns path to the file.
        """
        if dtype is not None:
            prof.update(dtype=dtype)
        if driver is not None:
            prof.update(driver=driver)

        if prof['driver'] == "JP2OpenJPEG":
            path += '.jp2'
        elif prof['driver'] == "GTiff":
            path += '.tif'
            prof.update(blockxsize=256, blockysize=256, compress='lzw')

        dim = raster.ndim
        if dim != 2 and dim != 3:
            log.error("Raster has bad dimensions.")
            raise ValueError

        iterations = 1
        if dim == 3:
            iterations = len(raster)
        prof.update(count=iterations)

        with rasterio.open(path, 'w', **prof) as dst:
            for i in range(1, iterations + 1):
                if dim == 3:
                    dst.write(raster[i - 1], i)
                else:
                    dst.write(raster, i)
        return path

    @staticmethod
    def save_band(raster_img, name: str, granule: S2Granule = None, path: str = None, driver: str = "GTiff",
                  d_type: gdal.Dataset = gdal.GDT_UInt16, geo_transform: gdal.Dataset = None,
                  projection: gdal.Dataset = None):

        # 2 dimensional array means there's only one raster image to save
        dim = raster_img.ndim
        if dim == 2:
            dim = 1
        # 3 dim array means this is going to be multi-band image
        elif dim == 3:
            dim = len(raster_img)
        else:
            raise Exception("Bad dimension of the raster image")

        if granule is not None:
            # saving connected with worker
            if path is None:
                path = "/".join(granule.paths_to_raster[:-1].split(os.path.sep)) + "/" + name
            x_res, y_res = granule.get_image_resolution()
        else:
            x_res, y_res = raster_img.shape[0], raster_img.shape[1]

        if path is None:
            raise Exception("Please provide a valid path")

        if driver == "GTiff":
            path += ".tif"
        elif driver == "OpenJPEG200":
            path += ".jp2"
        else:
            raise Exception("Format is not yet supported!")

        if dim == 1 and (x_res, y_res) != (raster_img.shape[0], raster_img.shape[1]):
            raise Exception("The dimensions of saved image do not correspond with the raster image")
        elif dim > 1:
            for i in range(dim):
                if (x_res, y_res) != (raster_img[i].shape[0], raster_img[i].shape[1]):
                    raise Exception("The dimensions of saved image do not correspond with one of the band from raster "
                                    "image")

        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(
            path,
            x_res,
            y_res,
            dim,
            d_type
        )

        if geo_transform is not None:
            dataset.SetGeoTransform(geo_transform)
        if projection is not None:
            dataset.SetProjection(projection)
        #  TODO: catch gdal exceptions ?
        if dim == 1:
            dataset.GetRasterBand(1).WriteArray(raster_img)
        else:
            for d in range(dim):
                dataset.GetRasterBand(d + 1).WriteArray(raster_img[d])
                # dataset.SetNoDataValue(0)
        dataset.FlushCache()
        # Update worker if everything has been done correctly and worker is available
        if granule is not None:
            granule.update_worker(name, path)
        del dataset, driver
        return path  # path where it is saved

    @staticmethod
    def s2_agriculture(granule: S2Granule, save: bool = False):
        stacked = granule.stack_bands(desired_order=["B11", "B08", "B02"])
        path = granule.path + os.path.sep + f"agriculture_{granule.spatial_resolution}"
        if not save:
            return stacked
        for i in range(len(stacked)):
            stacked[i] = rescale_intensity(stacked[i], 0, 4096)
        stacked = stacked.astype(numpy.uint8)
        profile = granule['B02'].profile
        profile = profile_for_rgb(profile)
        path = GranuleCalculator.save_band_rast(stacked, path, prof=profile, driver="GTiff")
        granule.add_another_band(path, "agriculture")
        return stacked

    @staticmethod
    def s2_color_infrared(granule: S2Granule, save: bool = False):
        stacked = granule.stack_bands(desired_order=["B8A", "B04", "B03"])
        path = granule.path + os.path.sep + f"infrared_{granule.spatial_resolution}"
        if not save:
            return stacked
        for i in range(len(stacked)):
            stacked[i] = rescale_intensity(stacked[i], 0, 4096)
        stacked = stacked.astype(numpy.uint8)
        profile = granule['B02'].profile
        profile = profile_for_rgb(profile)
        path = GranuleCalculator.save_band_rast(stacked, path, prof=profile)
        granule.add_another_band(path, "infrared")
        return stacked

    @staticmethod
    def s2_moisture_index(granule: S2Granule, save: bool = False):
        path = granule.path + os.path.sep + f"moisture_index_{granule.spatial_resolution}"
        b8 = granule['B8A'].raster().astype(float)
        b11 = granule['B11'].raster().astype(float)
        m1 = (b8 - b11)
        m2 = (b8 + b11)
        numpy.divide(m1, m2, out=m1, where=m2 != 0).squeeze()
        if not save:
            return m1
        profile = granule['B8A'].profile
        path = GranuleCalculator.save_band_rast(m1, path, prof=profile, dtype=np.dtype('float'), driver="GTiff")
        granule.add_another_band(path, "moisture_index")
        return m1

    @staticmethod
    def s2_ndvi(granule: S2Granule, save: bool = False):
        """
        Calculates the Normalized difference vegetation index
        :param granule: worker that provides us data
        :param save: if user wants to save the result inside the working dir of the worker
        :return: numpy array
        """
        nir = granule['B8A'].raster().astype(float)
        red = granule['B04'].raster().astype(float)
        _ndvi = ndvi(red=red, nir=nir)
        granule.temp["NDVI"] = _ndvi
        if not save:
            return _ndvi
        path = granule.path + os.path.sep + f"ndvi_{granule.spatial_resolution}"
        path = GranuleCalculator.save_band_rast(_ndvi, path=path, prof=granule['B8A'].profile,
                                                dtype=np.dtype('float'), driver="GTiff")
        # initialize new band
        granule.add_another_band(path, "ndvi")
        del nir, red
        return _ndvi

    @staticmethod
    def s2_ari1(granule: S2Granule, save: bool = False):
        """
        ARI - Anthocyanin Reflectance Index
        Anthocyanins are pigments common in higher plants, causing their red, blue and purple coloration.
        They provide valuable information about the physiological status of plants,
        as they are considered indicators of various types of plant stresses.

        The reflectance of anthocyanin is highest around 550nm. However, the same wavelengths are reflected by
        chlorophyll as well. To isolate the anthocyanins, the 700nm spectral band, that reflects only chlorophyll and
        not anthocyanins, is subtracted.
        """
        if granule is None or granule["B03"] is None or granule["B05"] is None:
            raise ValueError("granule is none")
        b03 = 1 / granule["B03"].raster().astype(float)
        b05 = 1 / granule["B05"].raster().astype(float)
        ari1 = b03 - b05
        granule.temp["ARI1"] = ari1
        if not save:
            return ari1
        path = granule.path + os.path.sep + f"ari1_{granule.spatial_resolution}"
        path = GranuleCalculator.save_band_rast(ari1, path=path, prof=granule["B03"].profile,
                                                dtype=np.dtype('float'), driver="GTiff")
        granule.add_another_band(path, "ari1")
        return ari1

    @staticmethod
    def s2_pertile_cloud_index_mask(granule: S2Granule, detector: Callable) -> np.array:
        log.debug(f"Worker {granule.doy}, cloud index mask.")
        #  Compute cloud mask for each tile
        arr = detector(granule)  # (slices, res_x, res_y)
        #  Each index represents one tile and her cloud percentage
        # result = np.zeros(shape=granule.slice_index ** 2)
        shp1 = arr.shape[1]
        shp2 = arr.shape[2]

        arr = np.sum(arr, axis=(1, 2)) / (shp1 * shp2)
        # for i in range(granule.slice_index ** 2):
        #     result[i] = np.sum(arr[i]) / (arr.shape[1] * arr.shape[2])
        # return result
        return arr

    @staticmethod
    def build_mosaics(granules: List[S2Granule], path: str, name: str = "_mosaic", **kwargs) -> None:
        """
        Method gathers all initialized bands inside a Granule and compares it with the others.
        For each Band that is present in every Granule, mosaic is built.
        """
        if len(granules) == 0:
            log.warning("Empty list of granules. Terminating...")
            return
        bands = set(granules[0].get_initialized_bands())  # init values
        #  Get bands that are present in every granule
        for granule in granules:
            b = set(granule.get_initialized_bands())
            bands = bands.intersection(b)
        log.info(f"Bands in each granule : {bands}")
        for band in bands:
            paths = [g.bands[g.spatial_resolution][band].path for g in granules]  # Paths to raster data
            log.info(f"Band: {band}, paths: {paths}")
            build_mosaic(path, paths, band + name, band == "rgb", **kwargs)

    @staticmethod
    def info():
        print("Bands calculator supports GeoTiff and Jpeg2000")
