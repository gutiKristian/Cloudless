import json
import os
import re
from typing import *
import shutil
import subprocess
import threading
from zipfile import ZipFile
import pandas
import requests
import json
import re
from shapely.geometry import Polygon
from DownloadExceptions import *
from Pipeline.logger import log
import datetime


# Download bands or full file option ..... bands need to be provided...
# If bands are not present in the spatial res they are downloaded with other spat. res and resampled

class Downloader:
    def __init__(self, user_name: str, password: str, root_path: str = None, polygon: List = None,
                 date: datetime = (datetime.datetime.now() - datetime.timedelta(days=50), datetime.datetime.now()),
                 uuid: List[str] = None, cloud_coverage: List[int] = None, product_type: str = "S2MSI2A",
                 mercator_tiles: List[str] = None, text_search: str = None, platform_name: str = "Sentinel-2"):
        """
        It is recommended to initialize object via class methods to prevent unexpected results for the user.
        Rules:
            - polygon is always taken as the highest priority argument for the datasets look up, following uuid and
            last is mercator
        @param: user_name - copernicus account login
        @param: password - copernicus account password
        @param: root_path - where files are going to be downloaded BY DEFAULT
        @param: polygon - polygon that defines the area of interest (AOI)
        @param: date - in what time range we should search, by default is two weeks from now
        @param: uuid - id's of products to download
        @param: cloud_coverage - [min, max], default is set to [0, 95]
        @param: product_type - what sentinel-2 products we want to consume, default: L2A products
        @param: mercator_tiles - 100x100 km2 ortho-images in UTM/WGS84 projection
        @param: text_search - regex search
        """
        self.url = "https://dhr1.cesnet.cz/"
        if not user_name or not password:
            raise CredentialsNotProvided()
        self.user_name = user_name
        self.password = password
        self.root_path = root_path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.session = requests.Session()
        self.session.auth = (self.user_name, self.password)
        self.polygon = Downloader.create_polygon(polygon)
        if not Downloader.is_valid_cloud_cov(cloud_coverage):
            raise IncorrectInput("Bad format: cloud coverage.")
        self.cloud_coverage = cloud_coverage
        if not Downloader.is_valid_date_range(date):
            raise IncorrectInput("Bad format: date")
        self.date = date
        self.product_type = product_type
        self.tile_uuids = uuid
        self.mercator_tiles = Downloader.validate_mercator_tiles(mercator_tiles)
        self.text_search = text_search
        self.platform_name = platform_name  # even though this is Sentinel2.py we may extract this class someday
        #  After successful initialization of Downloader, obj_cache contains: formatted string for cloud, time and
        #  cached requests
        self.__obj_cache = {'request': {}}
        self.__cache = {}
        # Check
        self.__minimum_requirements()

    #  'Public'

    # downloading is triggered by the user, each time he calls this method
    def download_tile_whole(self):
        pass

    def download_tile_bands(self, bands: List[str], primary_spatial_res: int):
        pass

    # all_... methods download everything at once
    def download_all_whole(self, unzip: bool = False):
        pass

    def download_all_bands(self, bands: List[str], primary_spatial_res: int):
        pass

    #  Mangled

    def __build_info_queries(self) -> List[str]:
        """
        Build initial request for the required aoi or tiles. Method returns list of 1 url for polygon
        and list of urls for each tile if it is preferred method.
        """
        self.__obj_cache['cloud'] = "[{} TO {}]".format(self.cloud_coverage[0], self.cloud_coverage[1])
        self.__obj_cache['time'] = "[" + self.date[0].strftime("%Y-%m-%d") + "T00:00:00.000Z" + " TO " + self.date[
            1].strftime(
            "%Y-%m-%d") + "T23:59:59.999Z]"
        result = self.url + "?=( platformname:{} AND producttype:{} AND cloudcoverpercentage:{} " \
                            "AND beginposition:{}".format(self.platform_name, self.product_type,
                                                          self.__obj_cache['cloud'], self.__obj_cache['time'])
        if self.polygon:
            return [result + ' footprint:"Intersects(Polygon(({})))"'.format(
                ",".join("{} {}".format(p[0], p[1]) for p in list(self.polygon.exterior.coords)))]

    def __minimum_requirements(self):
        """
        If the downloader was provided enough information to do his job.
        """
        if not self.polygon and not self.tile_uuids and not self.mercator_tiles and not self.tile_uuids:
            log.warning("Area was not defined")
            if self.text_search:
                log.info("I'll try to find some area indicators in text search")
                r = re.compile("^[*][\d]{1,2}[\w]{0,3}[*]$")
                if not r.search(self.text_search):
                    raise IncorrectInput("Area was not specified!")
                log.info("Found regex for the tile!")
            log.info("Area defined with regex in text search")
        log.info("Downloader has enough information, performing initial request")
        #  A bit clumsy but it's made to speed up the response when user is creating job, it's just a confirmation
        # that the job is valid and we've got dataset s to work with
        self.__obj_cache['urls'] = self.__build_info_queries()
        overall_datasets = 0
        for url in self.__obj_cache['urls']:
            if overall_datasets > 0:
                log.info("Required minimum of datasets achieved.")
                return
            self.__obj_cache['requests'][url] = pandas.read_json(self.session.get(url).content)['feed']
            overall_datasets += self.__obj_cache['requests'][url]['opensearch:totalResults']
        raise IncorrectInput("Not enough datasets to download or run the pipeline for this input")

    def __before_download(self):
        """
        This method gathers the remaining data that weren't downloaded/requested during the initialization
        because of response time optimization.
        """
        pass

    #  Static

    @staticmethod
    def create_polygon(polygon: List) -> Optional[Polygon]:
        try:
            p = Polygon(polygon)
            if p.is_valid():
                return p
            return None
        except Exception as _:
            log.error("Couldn't create polygon instance from provided coordinates", exc_info=True)
        return None

    @staticmethod
    def is_valid_cloud_cov(cloud_cov) -> bool:
        if cloud_cov is None or len(cloud_cov) < 2:
            log.error("Cloud coverage takes list of 2, [min, max]")
            return False
        if cloud_cov[0] > cloud_cov[1]:
            log.error(f"Bad interval, did you mean {[cloud_cov[1], cloud_cov[0]]} ?")
            return False
        if abs(cloud_cov[0] - cloud_cov[1]) < 5:
            log.warning(f"Provided cloud coverave interval is small, there may be fewer result or even None")
        return True

    @staticmethod
    def is_valid_date_range(date: Tuple) -> bool:
        if date is None or len(date) != 2 or (date[1] - date[0]).days > 0:
            return False
        return True

    @staticmethod
    def validate_mercator_tiles(tiles: List[str]) -> List[str]:
        r = re.compile("^[\d]{1,2}[A-Z]{0,3}")
        validated = []
        for entry in tiles:
            if len(entry) == 5 and r.search(entry):
                validated.append(entry)
            else:
                log.warning(f"BAD FORMAT: {entry} is not a tile, it won't be included in the tiles to download!")
        return validated

    @staticmethod
    def un_zip(path):
        """
        Function finds all .zip files and unzip them.
        """
        for item in os.listdir(path):
            if item.endswith(".zip"):
                abs_path = os.path.abspath(path + os.path.sep + item)
                with ZipFile(abs_path, 'r') as zip_ref:
                    zip_ref.extractall(path)
                os.remove(abs_path)
                print("UnZipped : {}".format(path))
