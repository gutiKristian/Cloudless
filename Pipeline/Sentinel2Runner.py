import threading
import time
import shutil

from numba.typed import List as LIST
from Pipeline.logger import log
from Pipeline.WorkerCalculator import *
from Pipeline.Masks import *
from Pipeline.Plotting import *


class S2Runner:
    def __init__(self, path: str, spatial_resolution: int, slice_index: int = 1, output_bands: str = []):
        """
        :param path: to the dataset
        :param spatial_resolution: on which we are going to operate on
        :param slice_index: per-pixel: always 1, per-tile from pre-defined choices
        :param output_bands: defined by user on the frontend side
        """
        if not is_dir_valid(path):
            raise FileNotFoundError("{} may not exist\nPlease check if file exists".format(path))
        if not s2_is_spatial_correct(spatial_resolution):
            raise Exception("Wrong spatial resolution, please choose between 10, 20 and 60m")
        if s2_get_resolution(spatial_resolution)[0] % slice_index != 0:
            raise Exception("Unable to evenly slice the image! Please use different value, the working resolution is "
                            "{0} and slicing index is {1}")
        self.output_bands = output_bands
        if len(output_bands) == 0:
            log.info(
                f"Output bands was {output_bands}, using pre-defined bands for spatial resolution {spatial_resolution}")
            self.output_bands = bands_for_resolution(spatial_resolution)
        #  Path to directory that represents one big tile for instance T33UXW
        self.main_dataset_path = path
        #  Supported spatial resolutions for sentinel 2 are 10m,20m and 60m
        self.spatial_resolution = spatial_resolution
        self.mercator = extract_mercator(path)  # TODO: CHECK THIS
        #  Datasets in SAFE format
        self.datasets = get_subdirectories(path)
        self._validate_files_by_mercator()
        # Initialize workers
        self.workers = [S2Worker(_path, spatial_resolution, self.output_bands, slice_index) for _path in self.datasets
                        if s2_is_safe_format(_path)]
        #  The result of masking is stored in this variable, "B01": numpy.array, etc.
        self.result = {}
        self.save_result_path = self.main_dataset_path + os.path.sep + "result"
        self.result_worker = None
        log.info(f"Initialized S2Runner:\n{self}")

    def get_save_path(self) -> str:
        return self.save_result_path

    def _validate_files_by_mercator(self) -> None:
        if len(self.datasets) < 2:
            raise Exception("Not enough files to execute, exactly: {}".format(len(self.datasets)))
        if self.mercator == "":
            self.mercator = extract_mercator(self.datasets[0])
        for file in self.datasets:
            if extract_mercator(file) != self.mercator:
                raise Exception("Tiles with different area detected")

    def _save_result(self) -> None:
        try:
            os.mkdir(self.save_result_path)
        except FileExistsError:
            shutil.rmtree(self.save_result_path)
            os.mkdir(self.save_result_path)
        projection = list(self.workers[-1].bands[self.spatial_resolution].values())[0].projection
        geo_transform = list(self.workers[-1].bands[self.spatial_resolution].values())[0].geo_transform
        for key in self.result.keys():
            path = self.save_result_path + "/" + key + "_" + str(self.spatial_resolution)
            WorkerCalculator.save_band(raster_img=self.result[key], name=key + "_" + str(self.spatial_resolution),
                                       path=path, projection=projection, geo_transform=geo_transform)

    def _load_bands(self, desired_bands: List[str] = None):
        for worker in self.workers:
            worker.load_bands(desired_bands)

    def _release_bands(self):
        """
        Free the memory. Set the references for the numpy arrays to None.
        """
        for worker in self.workers:
            worker.free_resources()
        self.result = {}

    def optimized_ndvi(self, constraint=5) -> int:
        """
        :param constraint: how many tiles are allowed to be opened simultaneously
        :return: exit code, success-0, error anything else might depend on the error
        """
        log.info(f"Running optimised ndvi masking. Dataset {self.main_dataset_path}")
        res_x, res_y = s2_get_resolution(self.spatial_resolution)
        # we don't need to stack all ndvi arrays, we need just the constraint and result
        ndvi_arrays = np.zeros(shape=(constraint, res_x, res_y), dtype=np.float)
        # this ndvi array serves as a holder of the current max ndvi value for this pixel
        ndvi_result = np.zeros(shape=(res_x, res_y), dtype=np.float)
        result = np.zeros(shape=(len(self.output_bands), res_x, res_y), dtype=np.uint16)
        doy = np.zeros(shape=(res_x, res_y), dtype=np.uint16)
        log.info(f"{(len(self.workers) - 1) // constraint + 1} iteration(s) expected!")
        for iteration in range((len(self.workers) - 1) // constraint + 1):
            # compute NDVI
            current_doy = LIST()
            current_data = LIST()
            log.info(f"Calculating NDVI arrays for iteration {iteration}")
            # Acquire first batch of workers, for instance constraint=4, workers=[0,1,2,3]
            workers = self.workers[iteration * constraint: (iteration + 1) * constraint]
            for i, w in enumerate(workers, 0):
                current_doy.append(w.doy)
                # Prepare data
                w.load_bands()
                WorkerCalculator.s2_ndvi(w)
                mask = (w["B02"] > 100) & (w["B04"] > 100) & (w["B8A"] > 500) & (w["B8A"] < 8000) & (w["AOT"] < 100)
                ndvi_arrays[i] = np.ma.array(w.temp["NDVI"], mask=mask, fill_value=0).filled()
                del mask
                current_data.append(w.stack_bands())
                w.free_resources()  # We have copied the resources to the new numpy array inside current_data
            log.debug(f"Masking {constraint} tiles.")
            S2JIT.s2_ndvi_pixel_analysis(ndvi_arrays, ndvi_result, current_data, current_doy, result, doy, res_x, res_y)
            log.debug(f"Done!")
            current_data = LIST()  # last iteration, del the pointer to the arrays

        # Init
        for i, band in enumerate(self.output_bands, 0):
            self.result[band] = result[i]
        self.result["DOY"] = doy
        gc.collect()
        log.info("Saving result to the files...")
        self._save_result()  # Save result into files
        log.info("Done!")
        # If there's an intention to work further with the files
        self.result_worker = S2Worker(self.save_result_path, self.spatial_resolution, self.output_bands)
        log.debug("New self.worker attribute initialized.")
        log.info("Masking done!")
        return 0

    def __str__(self):
        return f"Dataset: {self.main_dataset_path}\n" \
               f"Spatial resolution: {self.spatial_resolution}\n" \
               f"Tile datasets: {self.datasets}\n" \
               f"The result will be saved at: {self.save_result_path}\n"
