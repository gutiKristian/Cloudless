from datetime import datetime
from xml.etree import ElementTree  # xml
from osgeo import gdal
from Pipeline.PipelineBand import *
from Pipeline.utils import *
from colorama import Back
gdal.UseExceptions()


class S2Worker:
    """
    This class can be instantiated only inside S2Runner.
    """

    def __init__(self, path: str, spatial_res: int):
        if not is_dir_valid(path):
            raise FileNotFoundError(Back.RED + "Dataset has not been found !")
        self.path = path
        self.spatial_resolution = spatial_res
        self.meta_data_path = self.path + os.path.sep + "MTD_MSIL2A.xml"
        try:
            self.meta_data_gdal = gdal.Open(self.meta_data_path)
            self.meta_data = self.meta_data_gdal.GetMetadata()
            self.data_take = datetime.strptime(self.meta_data["DATATAKE_1_DATATAKE_SENSING_START"],
                                               "%Y-%m-%dT%H:%M:%S.%fZ")
            self.doy = self.data_take.timetuple().tm_yday
        except Exception as e:
            print('Opening meta data file raised an exception', e, "\nWorker continues without metadata file!")
        self.paths_to_raster = self._find_images()
        if self.paths_to_raster is None or len(self.paths_to_raster) < 2:
            raise Exception("None or not enough datasets have been provided!")
        self.bands = self._to_band_dictionary()
        self.temp = {}

    def _find_images(self):
        if self.meta_data_gdal is None:
            return get_files_in_directory(self.path)
        tree = ElementTree.parse(self.meta_data_path)
        root = tree.getroot()
        images = []
        for image in look_up_raster(root, 'Granule')[0]:
            text = image.text
            if os.name == 'nt':
                text = text.replace('/', '\\')
            images.append(self.path + os.sep + text + '.jp2')
        if self.spatial_resolution == 10:
            return images[0:7]
        elif self.spatial_resolution == 20:
            return images[7:20]
        return images[20::]

    def _to_band_dictionary(self) -> dict:
        """
        Match list of paths of bands to dictionary for better access.
        """
        if not self.paths_to_raster or len(self.paths_to_raster) == 0:
            return {}
        e_dict = dict()  # create new dictionary
        # spatial_res = int(re.findall(r'\d+', array[0])[-1])
        if self.spatial_resolution not in e_dict:
            e_dict[self.spatial_resolution] = {}
        for band in self.paths_to_raster:
            key = re.findall('B[0-9]+A?|TCI|AOT|WVP|SCL', band)[-1]
            e_dict[self.spatial_resolution][key] = Band(band)
        return e_dict

    def get_image_resolution(self) -> Tuple[int, int]:
        if self.spatial_resolution == 10:
            return 10980, 10980
        elif self.spatial_resolution == 20:
            return 5490, 5490
        return 1830, 1830

    def add_another_band(self) -> None:
        pass  # TODO: RESAMPLE AND UPDATE THE DICT

    def load_bands(self, desired_bands: List[str] = None):
        if desired_bands is None:
            desired_bands = list(self.bands[self.spatial_resolution])
        for _key in desired_bands:
            self.bands[self.spatial_resolution][_key].load_raster()

    def free_resources(self) -> None:
        for key, band in self.bands[self.spatial_resolution]:
            band.free_resources()

    def update_worker(self, name: str, path: str):
        self.bands[self.spatial_resolution][name] = Band(path)

    def stack_bands(self, desired_order: List[str] = None) -> np.ndarray:
        """
        Methods stacks all available bands.
        It forms a cube of bands.
        @param desired_order: user may set his order
        """
        stack = []
        if desired_order is None:
            desired_order = list(self.bands[self.spatial_resolution].keys())
        for key in desired_order:
            stack.append(self.bands[self.spatial_resolution][key].raster())
        return np.array(stack)

    def __getitem__(self, item):
        """
        The band that will be returned is band with the active
        spatial resolution
        :param item: band for instance 'B01' or 'TCI'
        :return: instance of a Band class
        """
        return self.bands[self.spatial_resolution][item]
