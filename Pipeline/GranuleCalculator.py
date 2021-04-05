from Pipeline.Granule import *
from Pipeline.utils import *
from osgeo import gdal
from typing import Callable
import rasterio
import os
from rasterio.profiles import Profile as RasterioProfile


class GranuleCalculator:

    @staticmethod
    def save_band_rast(raster: np.ndarray, path: str, prof: RasterioProfile = None, dtype: type = None,
                       driver: str = None):
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
            iterations = len(raster)[0]
        prof.update(count=iterations)

        with rasterio.open(path, 'w', **prof) as dst:
            for i in range(1, iterations + 1):
                if dim == 3:
                    dst.write(raster[i - 1], i)
                else:
                    dst.write(raster, i)

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
        GranuleCalculator.save_band(granule=granule, raster_img=_ndvi, geo_transform=granule['B04'].geotransform,
                                    projection=granule['B04'].projection)
        del nir, red
        return _ndvi

    @staticmethod
    def s2_cloud_mask_scl(w: S2Granule) -> np.ndarray:
        a = w["SCL"] > 7
        b = w["SCL"] < 11
        c = w["SCL"] < 1
        return (a & b) | c

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
    def build_mosaics(granules: List[S2Granule], path: str, name: str = "_mosaic",  **kwargs):
        bands = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12", "AOT", "RGB", "SCL",
                 "WVP", "DOY", "rgb"}
        #  Get bands that are present in every granule
        for granule in granules:
            b = set(granule.get_initialized_bands())
            bands = bands.intersection(b)
        for band in bands:
            paths = [g.bands[g.spatial_resolution][band].path for g in granules]  # Paths to raster data
            build_mosaic(path, paths, band + name, band == "rgb", **kwargs)

    @staticmethod
    def info():
        print("Bands calculator supports GeoTiff and Jpeg2000")
