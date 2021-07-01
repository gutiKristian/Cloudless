from datetime import datetime
from xml.etree import ElementTree  # xml

import numpy as np
from osgeo import gdal
from Pipeline.Band import *
from Pipeline.utils import *
from Pipeline.logger import log

gdal.UseExceptions()


class S2Granule:

    def __init__(self, path: str, spatial_res: int, desired_bands: List[str], slice_index: int = 1,
                 t_srs: str = 'EPSG:32633', granule_type: str = "L2A"):
        if not is_dir_valid(path):
            raise FileNotFoundError("Dataset has not been found !")
        if not supported_granule_type(granule_type):
            raise ValueError("This granule type is not supported !")
        self.path = path
        self.granule_type = granule_type
        self.spatial_resolution = spatial_res
        self.desired_bands = desired_bands
        self.t_srs = t_srs
        self.meta_data_path = self.path + os.path.sep + (
            "MTD_MSIL2A.xml" if granule_type == "L2A" else "MTD_MSIL1C.xml")
        self.meta_data_gdal = None
        self.meta_data = None
        self.data_take = None
        self.doy = 0
        self.paths_to_raster = None
        self.__initialize_meta()
        self.__find_images()
        if self.paths_to_raster is None or len(self.paths_to_raster) < 2:
            raise Exception("None or not enough datasets have been provided!")
        self.slice_index = slice_index
        self.bands = self.__to_band_dictionary()
        self.cloud_index = 0
        self.temp = {}
        log.info(f"Initialized granule:\n{self}")

    def __find_images(self) -> None:
        """
        Methods tries to find images in the granule directory.
        If meta data file is present, it grabs paths and validate if they exist.
        Also at the end checks whether we have desired bands.
        If meta data file is present but the paths to the raster images are incorrect, it grabs all available
        images in the directory. The same is done if the meta data file was not provided.
        """
        if self.meta_data_gdal is None:
            log.info(f"{self.granule_type} meta-data has not been found in {self.path}")
            # Sentinel images are encoded with JPEG2000 but some processing might have been done so we check for tif too
            self.paths_to_raster = get_files_in_directory(self.path, '.jp2')
            if len(self.paths_to_raster) == 0:
                self.paths_to_raster = get_files_in_directory(self.path, '.tif')
            return
        log.info(f"{os.path.basename(self.meta_data_path)} has been found in {self.path}")
        #  Extract images from xml
        tree = ElementTree.parse(self.meta_data_path)
        root = tree.getroot()
        images = []
        for image in look_up_raster(root, 'Granule')[0]:
            text = image.text
            if os.name == 'nt':
                text = text.replace('/', '\\')
            images.append(self.path + os.sep + text + '.jp2')
        #  Take only bands we are "looking for"
        self.paths_to_raster = self.__extract_bands(images)

        #  Especially useful when we download meta-data file and only selected bands
        if not is_file_valid(self.paths_to_raster[0]):
            log.info("File found in meta-data do not exist...\nChecking the directory...")
            self.paths_to_raster = get_files_in_directory(self.path, '.jp2')
        else:
            self.paths_to_raster = verify_bands(images, self.paths_to_raster,
                                                self.desired_bands, self.spatial_resolution)
        log.info(f"Final paths to raster data {self.paths_to_raster}")

    def __to_band_dictionary(self) -> dict:
        """
        Match list of paths of bands with desired bands to dictionary of bands.
        @return: dictionary of Band class elements
        """
        if not self.paths_to_raster or len(self.paths_to_raster) == 0:
            log.error(f"Raster images not present in {self.path}")
            raise FileNotFoundError("Images not present")
        e_dict = dict()  # create new dictionary
        # spatial_res = int(re.findall(r'\d+', array[0])[-1])
        if self.spatial_resolution not in e_dict:
            e_dict[self.spatial_resolution] = {}
        for band in self.paths_to_raster:
            key = re.findall('B[0-9]+A?|TCI|AOT|WVP|SCL|rgb|DOY', band)
            if len(key) == 0:
                pass
            key = key[-1]
            if key in self.desired_bands:
                b = Band(band, slice_index=self.slice_index)
                #  Automatically resample band to working spatial resolution
                if b.profile["width"] != s2_get_resolution(self.spatial_resolution)[0]:
                    b.resample(s2_get_resolution(self.spatial_resolution)[0] / b.profile["width"], delete=True)
                e_dict[self.spatial_resolution][key] = b
        for band in self.desired_bands:
            if band not in e_dict[self.spatial_resolution]:
                raise Exception(f"Band {band} is missing in the dataset, terminating")
        log.info("All necessary bands are present...Continue")
        return e_dict

    def __initialize_meta(self) -> None:
        """
        If the metadata file is present we extract the DATATAKE to find out the DOY for this dataset.
        If there's no metadata file the DOY isn't acquired and by default is set to 0, what means no-data.
        @return: None
        """
        try:
            self.meta_data_gdal = gdal.Open(self.meta_data_path)
            self.meta_data = self.meta_data_gdal.GetMetadata()
            self.data_take = datetime.strptime(self.meta_data["DATATAKE_1_DATATAKE_SENSING_START"],
                                               "%Y-%m-%dT%H:%M:%S.%fZ")
            self.doy = self.data_take.timetuple().tm_yday
            log.info("Meta data initialized.")
        except Exception as e:
            log.warning("Worker continues without metadata file!")

    def __extract_bands(self, paths) -> List[str]:
        """
        Util method for extracting bands that have been found inside the granule folder,
        for L2A it is done based on the spatial resolution and for L1C data all of available bands are picked.
        """
        if self.granule_type == "L1C":
            return paths
        #  L2A [0:7] - 10m, [7:20] - 20m, [20::] - 60m
        if self.spatial_resolution == 10:
            return paths[0:7]
        elif self.spatial_resolution == 20:
            return paths[7:20]
        return paths[20::]

    def add_another_band(self, path_to_band: str, key: str) -> None:
        """
        :param path_to_band - path to the raster data
        :param key - used for lookup inside granule
        """
        b = Band(path_to_band, slice_index=self.slice_index)
        if b.profile["width"] != s2_get_resolution(self.spatial_resolution)[0]:
            b.resample(s2_get_resolution(self.spatial_resolution)[0] / b.profile["width"], delete=True)
        self.bands[self.spatial_resolution][key] = b

    def load_bands(self, desired_bands: List[str] = None) -> None:
        """
        Method for loading the raster data into memory. Exists to avoid loading at the
        instantiation.
        :param desired_bands: user might specify which bands should be loaded
        """
        if desired_bands is None:
            desired_bands = list(self.bands[self.spatial_resolution])
        for _key in desired_bands:
            self.bands[self.spatial_resolution][_key].load_raster()

    def free_resources(self) -> None:
        """
        Release the bands numpy arrays! With the TEMPS.
        :return: None
        """
        for band in self.bands[self.spatial_resolution].values():
            band.free_resources()
        self.temp = {}

    def update_granule(self, name: str, path: str) -> None:
        """
        Register new file in worker.
        :param name: represents the file in the worker class
        :param path: file
        """
        self.bands[self.spatial_resolution][name] = Band(path)

    def stack_bands(self, desired_order: List[str] = None, dstack: bool = False) -> np.ndarray:
        """
        Methods stacks all available bands.
        It forms a cube of bands.
        @param desired_order: user may set his order
        @param dstack: whether to stack array along the third axis (used in ML predictions)
        WARNING: if no desired_order is specified, the order is random therefore might cause problems with the masking.
        """
        stack = []
        if desired_order is None:
            desired_order = list(self.bands[self.spatial_resolution].keys())
        for key in desired_order:
            stack.append(self.bands[self.spatial_resolution][key].raster())
        log.info(f"STACK ORDER: {desired_order}")
        return np.dstack(stack) if dstack else np.stack(stack)

    def get_initialized_bands(self) -> List[str]:
        if len(self.bands) == 0:
            return []
        return [key for key in self.bands[self.spatial_resolution]]

    def get_projection(self):
        return list(self.bands[self.spatial_resolution].values())[-1].profile["crs"]

    def reproject_bands(self, target_projection: str):
        for band in self.bands[self.spatial_resolution].values():
            band.band_reproject(t_srs=target_projection)

    def __getitem__(self, item) -> Band:
        """
        The band that will be returned is band with the active
        spatial resolution
        :param item: band for instance 'B01' or 'B8A'
        :return: instance of a Band class
        """
        return self.bands[self.spatial_resolution][item]

    def __str__(self):
        return f"Granule: {os.path.basename(os.path.normpath(self.path))}\n" \
               f"Bands: {self.desired_bands}\n Doy: {self.doy}" \
               f"Granule type: {self.granule_type}"
