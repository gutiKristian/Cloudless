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
import hashlib
import re
from shapely.geometry import Polygon
from Pipeline.logger import log
from Pipeline.utils import bands_for_resolution
import datetime
from Pipeline.utils import extract_mercator, is_dir_valid
from Download.DownloadExceptions import *


# Download bands or full file option ..... bands need to be provided...
# If bands are not present in the spatial res they are downloaded with other spat. res and resampled

class Downloader:
    meta_url = "https://dhr1.cesnet.cz/odata/v1/Products('{}')/Nodes('{}.SAFE')/Nodes('MTD_MSIL2A.xml')/$value"
    raster_url = "https://dhr1.cesnet.cz/odata/v1/Products('{}')/Nodes('{}.SAFE')/Nodes('GRANULE')/" \
                 "Nodes('{}')/Nodes('IMG_DATA')/Nodes('R{}')/Nodes('{}.jp2')/$value"

    def __init__(self, user_name: str, password: str, root_path: str = None, polygon: List = None,
                 date: datetime = (datetime.datetime.now() - datetime.timedelta(days=14), datetime.datetime.now()),
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
        self.root_path = self.root_path if self.root_path[-1] == os.path.sep else self.root_path + os.path.sep
        self.session = requests.Session()
        self.session.auth = (self.user_name, self.password)
        self.polygon = None
        if polygon is not None:
            self.polygon = Downloader.create_polygon(polygon)
        cloud_coverage = cloud_coverage if cloud_coverage else [0, 95]  # default value
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
        #  cached requests NOT ALL! all requests are cached after before_download is ran
        self.__obj_cache = {'requests': {}, 'polygon': False}
        self.__cache = {}
        # Check
        self.__minimum_requirements()

    #  'Public'

    def download_meta_data(self, url, filepath) -> str:
        """
        Downloads the meta-data MTD_MSIL2A.xml and saves it under filepath + MTD_MSIL2A.xml
        """
        response = self.session.get(url)
        if response.status_code != 200:
            raise Exception("Could not donwload the meta-data file.")
        content = response.content
        with open(filepath + "MTD_MSIL2A.xml", 'wb') as f:
            f.write(content)
            f.flush()
        return response.text

    def __get_group_matches(self, spatial: str, meta: str, result, bands, entry) -> set:
        pattern = re.compile(
            r'<IMAGE_FILE>GRANULE/([0-9A-Z_]+)/IMG_DATA/R{}/([0-9A-Z_]+_(.*)_{})</IMAGE_FILE>'.format(spatial,
                                                                                                       spatial))
        for m in re.finditer(pattern, meta):
            print(m.group(3))
            if not len(bands.intersection({m.group(3)})) > 0:
                continue
            bands = bands - {m.group(3)}
            result.append((Downloader.raster_url.format(entry["id"], entry["title"], m.group(1),
                                                         spatial, m.group(2)), m.group(2)))
        return bands

    def get_raster_urls(self, meta, entry, spatial_res, bands):
        """
        @param: meta - content of xml in string form
        @param: entry - data of the tile from the request
        @param: what spatial resolution should be
        :return: tuple - url and the name of the raster
        """
        bands = set(bands)
        raster_urls = []
        bands = self.__get_group_matches(spatial_res, meta, raster_urls, bands, entry)

        if len(bands) == 0:
            return raster_urls

        if spatial_res != "10m":
            bands = self.__get_group_matches("10m", meta, raster_urls, bands, entry)
            if len(bands) == 0:
                return raster_urls
        if spatial_res != "20m":
            bands = self.__get_group_matches("20m", meta, raster_urls, bands, entry)
            if len(bands) == 0:
                return raster_urls
        if spatial_res != "60m":
            bands = self.__get_group_matches("10m", meta, raster_urls, bands, entry)
        log.warning(f"Could not find {bands}.")
        return raster_urls

    def download_file(self, url, path, chunk_size=8192, check_sum=None) -> bool:
        """
        Downloads the file. Returns False on failure.
        """
        with self.session.get(url, stream=True) as req:
            req.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                f.flush()
        if not check_sum:
            log.warning("Check sum not provided")
            return True
        return hashlib.md5(path) == check_sum

    # downloading is triggered by the user, each time he calls this method
    def download_tile_whole(self, unzip: bool = True):
        """
        Downloads entire granule dataset, all meta data and spatial resolution images.
        File will be structured in .SAFE format. Yields path to the downloaded content.
        """
        self.__before_download()

        for mercator, entries in self.__get_next_download():
            working_path = self.root_path + os.path.sep + mercator
            # create directory mercator and download the entries
            Downloader.prepare_dir(working_path)
            threads = []
            for entry in entries:
                # 'link':[{'href': "https://dhr1.cesnet.cz/odata/v1/Products('')/$value"}, link for checksum ]
                zip_file = working_path + os.path.sep + entry['title'] + ".zip"
                link = entry['link'][0]['href']
                check_sum = self.session.get(entry['link'][1]['href'] + "/Checksum/Value/$value")
                if not self.download_file(link, zip_file, check_sum=check_sum):
                    log.error(f"File: {zip_file} wasn't downloaded properly")
                    os.remove(zip_file)
                else:
                    if unzip:
                        t = threading.Thread(target=Downloader.un_zip, args=(zip_file,))
                        threads.append(t)
                        t.start()
                        log.info("Started unzipping in another thread.")
                    log.info(f"File {working_path} successfully downloaded")
            for thread in threads:
                thread.join()
            yield working_path

        log.info("Everything downloaded")

    def download_tile_bands(self, primary_spatial_res: str, bands: List[str] = None):
        """
        @param primary_spatial_res: 20 -> 20m, 10 -> 10m, 60 -> 60m
        @param bands: ["B01", ... ]
        """
        self.__before_download()
        if bands is None:
            bands = bands_for_resolution(primary_spatial_res)

        for mercator, entries in self.__get_next_download():
            working_path = self.root_path + os.path.sep + mercator  # path where all datasets are going to be downloaded
            Downloader.prepare_dir(working_path)
            for entry in entries:
                data_set_path = working_path + os.path.sep + entry['title'] + os.path.sep
                Downloader.prepare_dir(data_set_path)
                meta_data = self.download_meta_data(Downloader.meta_url.format(entry["id"], entry["title"]),
                                                    data_set_path)
                raster_urls = self.get_raster_urls(meta_data, entry, primary_spatial_res, bands)
                for url, name in raster_urls:
                    self.download_file(url, data_set_path + name + ".jp2")
                yield data_set_path
        log.info("All downloaded")

    # all_... methods download everything at once
    def download_all_whole(self, unzip: bool = False) -> List[str]:
        """
        Returns list of paths to the downloaded content.
        """
        paths = []
        for e in self.download_tile_whole(unzip=unzip):
            paths.append(e)
        return paths

    def download_all_bands(self, primary_spatial_res: str, bands: List[str] = None):
        paths = []
        for e in self.download_tile_bands(primary_spatial_res, bands):
            paths.append(e)
        return paths

    #  Mangled

    def __get_next_download(self):
        """
        Traverse __cache and return meta data for mercator tiles that haven't been downloaded.
        """
        if len(self.__cache.keys()) == 0:
            return None
        for merc, feed in self.__cache.items():
            yield merc, feed

    def __build_info_queries(self) -> List[str]:
        """
        Build initial request for the required aoi or tiles. Method returns list of 1 url for polygon
        and list of urls for each tile if it is preferred method.
        """
        self.__obj_cache['cloud'] = "[{} TO {}]".format(self.cloud_coverage[0], self.cloud_coverage[1])
        self.__obj_cache['time'] = "[" + self.date[0].strftime("%Y-%m-%d") + "T00:00:00.000Z" + " TO " + self.date[1] \
            .strftime("%Y-%m-%d") + "T23:59:59.999Z]"
        result = self.url + "search?q=( platformname:{} AND producttype:{} AND cloudcoverpercentage:{} " \
                            "AND beginposition:{}".format(self.platform_name, self.product_type,
                                                          self.__obj_cache['cloud'], self.__obj_cache['time'])
        suffix = "&rows=100&format=json"
        if self.polygon:
            self.__obj_cache['polygon'] = True
            return [result + ' AND footprint:"Intersects(Polygon(({})))")'.format(
                ",".join("{} {}".format(p[0], p[1]) for p in list(self.polygon.exterior.coords))) + suffix]
        if self.mercator_tiles:
            return [result + " AND *{}*)".format(tile) + suffix for tile in self.mercator_tiles]
        if self.tile_uuids:
            return [self.url + "?=(uuid:{})".format(uuid) for uuid in self.tile_uuids]
        return [result + " AND {} ".format(self.text_search) + suffix]

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
            overall_datasets += int(self.__obj_cache['requests'][url]['opensearch:totalResults'])
        if overall_datasets > 0:
            log.info("Required minimum of datasets achieved.")
            return
        raise IncorrectInput("Not enough datasets to download or run the pipeline for this input")

    def __parse_cached_response(self):
        """
        Response for Polygon is unstructured and messy for us, since we download the datasets by tiles
        and therefore we are able to run pipeline meanwhile another granule(tile) dataset is downloading.
        """
        entries = list(self.__obj_cache['requests'].values())[0]['entry']
        for entry in entries:
            mercator = extract_mercator(entry['title'])
            if mercator not in self.__cache:
                self.__cache[mercator] = []
            self.__cache[mercator].append(entry)
        log.info(f"Mercator tiles extraced from the response: {self.__cache.keys()}")

    def __before_download(self):
        """
        This method gathers the remaining data that weren't downloaded/requested during the initialization
        because of response time optimization.
        """
        for url in self.__obj_cache['urls']:
            if self.__obj_cache['polygon']:
                # Already have full request just parse it
                self.__parse_cached_response()
            if url not in self.__obj_cache['requests']:
                self.__obj_cache['requests'][url] = pandas.read_json(self.session.get(url).content)['feed']
            # entry for uuid is just dict
        self.__parse_cached_response()

    #  Static

    @staticmethod
    def prepare_dir(path: str):
        if is_dir_valid(path):
            log.warning(f"Path: {path} already exists, the files in the directory will be deleted.")
            shutil.rmtree(path)
        os.mkdir(path)

    @staticmethod
    def create_polygon(polygon: List) -> Optional[Polygon]:
        try:
            p = Polygon(polygon)
            if p.is_valid:
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
        if date is None or len(date) != 2 or (date[1] - date[0]).days < 0:
            return False
        return True

    @staticmethod
    def validate_mercator_tiles(tiles: List[str]) -> List[str]:
        if tiles is None:
            return []
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
