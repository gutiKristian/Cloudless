from Pipeline.PipelineWorker import *
from Pipeline.utils import *
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
    def s2_ndvi(worker: S2Worker, save: bool = False):
        """
        Calculates the Normalized difference vegetation index
        :param worker: worker that provides us data
        :param save: if user wants to save the result inside the working dir of the worker
        :return: numpy array
        """
        nir = worker['B8A'].raster().astype(float)
        red = worker['B04'].raster().astype(float)
        _ndvi = ndvi(red=red, nir=nir)
        worker.temp["NDVI"] = _ndvi
        if not save:
            return _ndvi
        WorkerCalculator.save_band(worker, _ndvi, "NDVI", geo_transform=worker['B04'].geotransform,
                                   projection=worker['B04'].projection)
        del nir, red
        return _ndvi

    @staticmethod
    def s2_cloud_mask(w: S2Worker) -> np.ndarray:
        a = w["SCL"] > 7
        b = w["SCL"] < 11
        c = w["SCL"] < 1
        return (a & b) | c

    @staticmethod
    def s2_pertile_cloud_index_mask(worker: S2Worker) -> np.array:
        arr = WorkerCalculator.s2_cloud_mask(worker)
        result = np.zeros(shape=worker.slice_index**2)
        for i in range(worker.slice_index**2):
            log.info(f"Sum: {np.sum(arr)}")
            result[i] = np.sum(arr[i]) / (arr.shape[1] * arr.shape[2])
            log.info(f"Worker {worker.doy}, slice_index: {i}, cloud % : {result[i] * 100}")
        return result

    @staticmethod
    def info():
        print("Bands calculator supports GeoTiff and Jpeg2000")
