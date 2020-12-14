import threading
import time

from numba.typed import List as LIST

from Pipeline.BandsCalculator import *
from Pipeline.Masks import *
from Pipeline.Plotting import *


class S2Runner:
    def __init__(self, path: str, spatial_resolution: int):
        if not is_dir_valid(path):
            raise FileNotFoundError("{} may not exist\nPlease check if file exists".format(path))
        if not sentinel2_is_spatial_correct(spatial_resolution):
            raise Exception("Wrong spatial resolution, please choose between 10, 20 and 60m")
        #  Path to directory that represents one big tile for instance T33UXW
        self.main_dataset_path = path
        #  Supported spatial resolutions for sentinel 2 are 10m,20m and 60m
        self.spatial_resolution = spatial_resolution
        self.mercator = extract_mercator(path)
        #  Datasets in SAFE format
        self.datasets = get_subdirectories(path)
        self._validate_files_by_mercator()
        # Initialize workers
        self.workers = [S2Worker(_path, spatial_resolution) for _path in self.datasets]
        #  The result of masking is stored in this variable
        self.result = {}
        self.save_result_path = self.main_dataset_path + "/result"

    def _validate_files_by_mercator(self) -> None:
        if len(self.datasets) < 2:
            raise Exception("Not enough files to execute, exactly: {}".format(len(self.datasets)))
        if self.mercator == "":
            self.mercator = extract_mercator(self.datasets[0])
        for file in self.datasets:
            if extract_mercator(file) != self.mercator:
                raise Exception("Tiles with different area detected")

    def _save_result(self) -> None:
        os.mkdir(self.save_result_path)
        # TODO : Fix setting projections and geo-transforms
        # projection = list(self.workers[-1].bands.values())[0].projection
        # geo_transform = list(self.workers[-1].bands.values())[0].geo_transform
        for key in self.result.keys():
            path = self.save_result_path + "/" + key + "_" + str(self.spatial_resolution)
            BandCalculator.save_band(raster_img=self.result[key], name=key + "_" + str(self.spatial_resolution),
                                     path=path)

    def run_ndvi_cloud_masking(self) -> int:
        """
        1. Calculate NDVI for every worker
        2. Calculate binary mask
        3. Apply it on the ndvi
        4. Do the masking
        :return: code 0 for success, results will be saved in the workers path and final directory
        """
        print("start masking")
        for worker in self.workers:
            BandCalculator.s2_ndvi(worker)
            mask = (worker["B02"] > 100) & (worker["B04"] > 100) & (worker["B8A"] > 500) & \
                   (worker["B8A"] < 8000) & (worker["AOT"] < 100)
            # Plot.plot_mask(mask)
            Plot.plot_image(worker.temp["NDVI"])
            worker.temp["NDVI"] = np.ma.array(worker.temp["NDVI"], mask=~mask, fill_value=0).filled()
            # Plot.plot_image(worker.temp["NDVI"])
            del mask
        print("done")
        start = time.time()
        # self._s2_pixel_analysis()
        self._s2_jit_pixel_analysis()
        end = time.time()
        print("Elapsed time - masking = %s" % (end - start))
        self._save_result()
        return 0

    def _s2_pixel_analysis(self):
        print("starting analysis")
        result_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "AOT"]
        res_x, res_y = s2_get_resolution(self.spatial_resolution)
        self.result = {key: np.zeros(shape=(res_x, res_y), dtype=np.uint16) for key in result_bands}
        doy = np.zeros(shape=(res_x, res_y), dtype=np.uint16)
        # proceed to masking
        for y in range(res_y):
            for x in range(res_x):
                _max_val = -math.inf
                index = 0
                for i, worker in enumerate(self.workers, 0):
                    if worker.temp["NDVI"][y][x] > _max_val:
                        _max_val = worker.temp["NDVI"][y][x]
                        index = i
                doy[y][x] = self.workers[index].doy
                for band in result_bands:
                    self.result[band][y][x] = self.workers[index][band].raster()[y][x]
            if y % 100 == 0 and y != 0:
                print("Masked {} bands".format(y))
        self.result["DOY"] = doy

    def _load_bands(self, desired_bands: List[str] = None):
        threads = []
        for worker in self.workers:
            t = threading.Thread(target=worker.load_bands, args=[desired_bands])
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            print("done-opening")

    def _s2_jit_pixel_analysis(self):
        # Prepare the data
        result_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "AOT"]
        res_x, res_y = s2_get_resolution(self.spatial_resolution)

        # Using multithreading
        start = time.time()
        self._load_bands(result_bands)
        end = time.time()
        print("Elapsed time - opening datasets = %s" % (end - start))

        doys = np.array([w.doy for w in self.workers])
        ndvi = LIST()
        data_bands = LIST()
        start = time.time()
        for i, worker in enumerate(self.workers, 0):
            ndvi.append(worker.temp["NDVI"])
            data_bands.append(worker.stack_bands(result_bands))
        end = time.time()
        print("Elapsed time - stacking= %s" % (end - start))

        result = np.zeros(shape=(len(result_bands), res_x, res_y), dtype=np.uint16)
        doy = np.zeros(shape=(res_x, res_y), dtype=np.uint16)

        start = time.time()
        S2JIT.s2_pixel_analysis(ndvi, data_bands, doys, result, doy, res_x, res_y)
        end = time.time()
        print("Elapsed time - masking = %s" % (end - start))

        # Init
        for i, band in enumerate(result_bands, 0):
            self.result[band] = result[i]
        self.result["DOY"] = doy
