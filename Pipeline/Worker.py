import shutil
import gc

import Download.Sentinel2
from Pipeline.logger import log
from Pipeline.GranuleCalculator import *
from rasterio import dtypes as rastTypes
from concurrent.futures import ThreadPoolExecutor


class S2Worker:

    def __init__(self, path: str, spatial_resolution: int, slice_index: int = 1, output_bands: List[str] = [],
                 target_projection='EPSG:32633', polygon: List = None):
        """
        :param path: to the dataset
        :param spatial_resolution: on which we are going to operate on
        :param slice_index: per-pixel: always 1, per-tile from pre-defined choices
        :param output_bands: bands we work with
        :param polygon: polygon that crops out our data
        """
        if not is_dir_valid(path):
            raise FileNotFoundError("{} may not exist\nPlease check if file exists".format(path))
        if not s2_is_spatial_correct(spatial_resolution):
            raise Exception("Wrong spatial resolution, please choose between 10, 20 and 60m")
        if s2_get_resolution(spatial_resolution)[0] % slice_index != 0:
            raise Exception("Unable to evenly slice the image! Please use different value, the working resolution is "
                            "{0} and slicing index is {1}")
        self.polygon = polygon
        if self.polygon is not None:
            self.polygon = Download.Sentinel2.Downloader.create_polygon(polygon)
            if polygon is None:
                log.warning("Invalid polygon, 100x100 tiles are going to be used")
        self.output_bands = output_bands
        if len(output_bands) == 0:
            log.info(
                f"Output bands was {output_bands}, using pre-defined bands for spatial resolution {spatial_resolution}")
            self.output_bands = bands_for_resolution(spatial_resolution)
        #  Path to directory that represents one big tile for instance T33UXW
        self.main_dataset_path = path
        #  Supported spatial resolutions for sentinel 2 are 10m,20m and 60m
        self.spatial_resolution = spatial_resolution
        self.mercator = extract_mercator(path)
        #  Datasets in SAFE format
        self.datasets = get_subdirectories(path)
        self._validate_files_by_mercator()
        # Initialize granules
        self.granules = []
        for _path in self.datasets:
            try:
                if s2_is_safe_format(_path):
                    gr = S2Granule(_path, spatial_resolution, self.output_bands, slice_index, target_projection,
                                   polygon=self.polygon)
                    self.granules.append(gr)
            except Exception as _:
                log.error(f"Did not find raster dataset in {_path}")
        if len(self.granules) == 0:
            raise Exception("Initialization failed")
        #  The result of masking is stored in this variable, "B01": numpy.array, etc.
        self.result = {}
        self.save_result_path = self.main_dataset_path + os.path.sep + "result"
        self.result_worker = None
        self.slice_index = slice_index
        self.t_srs = target_projection
        log.info(f"Initialized S2Runner:\n{self}")

    def get_save_path(self) -> str:
        return self.save_result_path

    def _validate_files_by_mercator(self) -> None:
        if len(self.datasets) < 2:
            log.warning("Not enough files to execute cloudless jobs")
        if self.mercator == "":
            self.mercator = extract_mercator(self.datasets[0])
        for file in self.datasets:
            if extract_mercator(file) != self.mercator:
                raise Exception("Tiles with different area detected")

    def _save_result(self) -> None:
        """
        Save the results inside result dict to raster files.
        @return: None
        """
        try:
            os.mkdir(self.save_result_path)
        except FileExistsError:
            log.warning("Result directory already exists. File will be deleted.")
            shutil.rmtree(self.save_result_path)
            os.mkdir(self.save_result_path)
        # projection = list(self.granules[-1].bands[self.spatial_resolution].values())[0].projection
        # geo_transform = list(self.granules[-1].bands[self.spatial_resolution].values())[0].geo_transform
        # Fetch random profile from the bands
        profile = list(self.granules[-1].bands[self.spatial_resolution].values())[0].profile
        log.debug(f"Profile: {profile}")
        log.debug(f"Loaded from  {list(self.granules[-1].bands[self.spatial_resolution].values())[0].path}")
        with ThreadPoolExecutor(max_workers=10) as executor:
            for key in self.result.keys():
                path = self.save_result_path + os.path.sep + key + "_" + str(self.spatial_resolution)
                executor.submit(GranuleCalculator.save_band_rast, self.result[key], path=path, prof=profile,
                                driver="GTiff",
                                dtype=rastTypes.uint16)

    def _load_bands(self, desired_bands: List[str] = None):
        """
        Load each band in each granule.
        """
        for granule in self.granules:
            granule.load_bands(desired_bands)

    def release_bands(self):
        """
        Free the memory. Set the references for the numpy arrays to None.
        """
        for granule in self.granules:
            granule.free_resources()
        self.result = {}
        gc.collect()

    def get_res(self) -> (int, int):
        if self.polygon is None:
            return s2_get_resolution(self.spatial_resolution)
        granule = self.granules[-1]
        band = list(granule.bands[self.spatial_resolution].values())[-1]
        band.load_raster()
        return band.raster_image.shape

    def __str__(self):
        return f"Dataset: {self.main_dataset_path}\n" \
               f"Spatial resolution: {self.spatial_resolution}\n" \
               f"Tile datasets: {self.datasets}\n" \
               f"The result will be saved at: {self.save_result_path}\n"
