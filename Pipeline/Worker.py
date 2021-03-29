import threading
import time
import shutil

from numba.typed import List as LIST
from Pipeline.logger import log
from Pipeline.GranuleCalculator import *
from Pipeline.Mask import *
from Pipeline.Plotting import *


class S2Worker:

    def __init__(self, path: str, spatial_resolution: int, slice_index: int = 1, output_bands: List[str] = []):
        """
        :param path: to the dataset
        :param spatial_resolution: on which we are going to operate on
        :param slice_index: per-pixel: always 1, per-tile from pre-defined choices
        :param output_bands: bands we work with
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
        # Initialize granules
        self.granules = [S2Granule(_path, spatial_resolution, self.output_bands, slice_index) for _path in self.datasets
                         if s2_is_safe_format(_path)]
        #  The result of masking is stored in this variable, "B01": numpy.array, etc.
        self.result = {}
        self.save_result_path = self.main_dataset_path + os.path.sep + "result"
        self.result_worker = None
        self.slice_index = slice_index
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

    # TODO: CHANGE TO "PUBLIC" after refactor
    def _save_result(self) -> None:
        try:
            os.mkdir(self.save_result_path)
        except FileExistsError:
            shutil.rmtree(self.save_result_path)
            os.mkdir(self.save_result_path)
            log.warning("Result directory already exists. File will be deleted.")
        # projection = list(self.granules[-1].bands[self.spatial_resolution].values())[0].projection
        # geo_transform = list(self.granules[-1].bands[self.spatial_resolution].values())[0].geo_transform
        profile = list(self.granules[-1].bands[self.spatial_resolution].values())[0].profile
        for key in self.result.keys():
            path = self.save_result_path + os.path.sep + key + "_" + str(self.spatial_resolution)
            GranuleCalculator.save_band_rast(self.result[key], path=path, prof=profile, driver="GTiff")
            # GranuleCalculator.save_band(raster_img=self.result[key], name=key + "_" + str(self.spatial_resolution),
            #                             path=path, projection=projection, geo_transform=geo_transform)

    def _load_bands(self, desired_bands: List[str] = None):
        """
        Load each band in each granule.
        """
        for worker in self.granules:
            worker.load_bands(desired_bands)

    def release_bands(self):
        """
        Free the memory. Set the references for the numpy arrays to None.
        """
        for granule in self.granules:
            granule.free_resources()
        self.result = {}
        gc.collect()

    def __str__(self):
        return f"Dataset: {self.main_dataset_path}\n" \
               f"Spatial resolution: {self.spatial_resolution}\n" \
               f"Tile datasets: {self.datasets}\n" \
               f"The result will be saved at: {self.save_result_path}\n"
