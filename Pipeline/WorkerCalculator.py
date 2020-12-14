import numpy as np
import os
from Pipeline.PipelineWorker import *
from osgeo import gdal


class WorkerCalculator:

    #  TODO: Make worker argument optional, to generalize this method
    @staticmethod
    def save_band(raster_img, name: str, worker: S2Worker = None, path: str = None, driver: str = "GTiff",
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

        if worker is not None:
            # saving connected with worker
            if path is None:
                path = "/".join(worker.paths_to_raster[:-1].split(os.path.sep)) + "/" + name
            x_res, y_res = worker.get_image_resolution()
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
        dataset.FlushCache()
        # Update worker if everything has been done correctly and worker is available
        if worker is not None:
            worker.update_worker(name, path)
        del dataset, driver
        return path  # path where it is saved

    @staticmethod
    def ndvi(red: np.ndarray, nir: np.ndarray):
        ndvi1 = (nir - red)
        ndvi2 = (nir + red)
        return np.divide(ndvi1, ndvi2, out=np.zeros_like(ndvi1), where=ndvi2 != 0).squeeze()

    @staticmethod
    def s2_ndvi(worker: S2Worker, save: bool = False):
        nir = worker['B8A'].raster().astype(float)
        red = worker['B04'].raster().astype(float)
        ndvi = WorkerCalculator.ndvi(red, nir)
        worker.temp["NDVI"] = ndvi
        if not save:
            return ndvi
        WorkerCalculator.save_band(worker, ndvi, "NDVI", geo_transform=worker['B04'].geotransform,
                                   projection=worker['B04'].projection)
        del nir, red
        return ndvi

    @staticmethod
    def info():
        print("Bands calculator supports GeoTiff and Jpeg2000")
