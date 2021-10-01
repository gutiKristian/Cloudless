import json
import os
import re
import sys
from sys import platform
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
from xml.dom import minidom
from concurrent.futures import ThreadPoolExecutor


# Download bands or full file option ..... bands need to be provided...
# If bands are not present in the spatial res they are downloaded with other spat. res and resampled

class Downloader:
    manifest_url = "https://dhr1.cesnet.cz/odata/v1/Products('{}')/Nodes('{}.SAFE')/Nodes('manifest.safe')/$value"
    meta_url = "https://dhr1.cesnet.cz/odata/v1/Products('{}')/Nodes('{}.SAFE')/Nodes('{}')/$value"
    raster_url = "https://dhr1.cesnet.cz/odata/v1/Products('{}')/Nodes('{}.SAFE')/Nodes('GRANULE')/" \
                 "Nodes('{}')/Nodes('IMG_DATA')/Nodes('R{}')/Nodes('{}.jp2')/$value"
    raster_url_l1c = "https://dhr1.cesnet.cz/odata/v1/Products('{}')/Nodes('{}.SAFE')/Nodes('GRANULE')/" \
                     "Nodes('{}')/Nodes('IMG_DATA')/Nodes('{}.jp2')/$value"

    def __init__(self, user_name: str, password: str, root_path: str = None, polygon: List = None,
                 date: datetime = (datetime.datetime.now() - datetime.timedelta(days=14), datetime.datetime.now()),
                 uuid: List[str] = None, cloud_coverage: List[int] = None, product_type: str = "S2MSI2A",
                 mercator_tiles: List[str] = None, text_search: str = None, platform_name: str = "Sentinel-2",
                 time_str: str = None):
        """
        TODO: path to credentials folder
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
        if not Downloader.is_valid_sentinel_type(product_type):
            raise IncorrectInput("Invalid sentinel type, supported: S2MSI2A, S2MSI1C")
        self.cloud_coverage = cloud_coverage
        # if not Downloader.is_valid_date_range(date):
        #     raise IncorrectInput("Bad format: date")
        self.date = date
        self.product_type = product_type
        self.meta_data_name = "MTD_MSIL2A.xml"
        if self.product_type == "S2MSI1C":
            self.meta_data_name = "MTD_MSIL1C.xml"
        self.tile_uuids = uuid
        self.mercator_tiles = []
        self.validate_mercator_tiles(mercator_tiles)
        self.text_search = text_search
        self.platform_name = platform_name  # even though this is Sentinel2.py we may extract this class someday
        #  After successful initialization of Downloader, obj_cache contains: formatted string for cloud, time and
        #  cached requests NOT ALL! all requests are cached after before_download is ran
        self.__obj_cache = {'requests': {}, 'polygon': False}
        self.__cache = {}
        self.time_str = time_str
        self.overall_datasets = 0  # If > 100, we must do some paging
        # Check
        self.__minimum_requirements()

    #  'Public'
    def download_meta_data(self, url, filepath) -> str:
        """
        FILEPATH MUST INCLUDE THE NAME OF THE FILE! FOR INSTANCE DOWNLOADING
        THE MTD FILE -> <path>/MTD_MSIL2A.xml, DOWNLOADING THE MANIFEST FILE <path>/manifest.safe
        Downloads the meta-data MTD_MSIL2A.xml and saves it under filepath + MTD_MSIL2A.xml
        """
        response = self.session.get(url)
        if response.status_code != 200:
            raise Exception("Could not donwload the meta-data file.")
        content = response.content
        with open(filepath, 'wb') as f:
            f.write(content)
            f.flush()
        return response.text

    @staticmethod
    def extract_l2a_urls(spatial: str, meta: str, result, bands, entry) -> set:
        """
        Extract image file data information from metadata file
        """
        pattern = re.compile(
            r'<IMAGE_FILE>GRANULE/([0-9A-Z_]+)/IMG_DATA/R{}/([0-9A-Z_]+_(.*)_{})</IMAGE_FILE>'.format(spatial,
                                                                                                      spatial))
        for m in re.finditer(pattern, meta):
            if not len(bands.intersection({m.group(3)})) > 0:
                continue
            bands = bands - {m.group(3)}
            result.append((Downloader.raster_url.format(entry["id"], entry["title"], m.group(1),
                                                        spatial, m.group(2)), m.group(2)))
        return bands

    @staticmethod
    def get_raster_urls_l1c(meta, entry, bands):
        bands = set(bands)
        raster_urls = []
        pattern = re.compile(
            r'<IMAGE_FILE>GRANULE/(L1C_[0-9A-Z_]+)/IMG_DATA/(([0-9A-Z_]+)_([0-9A-Z_]{03}))</IMAGE_FILE>')
        for m in re.finditer(pattern, meta):
            if not len(bands.intersection({m.group(4)})) > 0:
                continue
            bands = bands - {m.group(4)}
            raster_urls.append(
                (Downloader.raster_url_l1c.format(entry["id"], entry["title"], m.group(1), m.group(2)), m.group(2)))
        return raster_urls

    @staticmethod
    def get_raster_urls_l2a(meta, entry, spatial_res, bands):
        """
        Function tries to extract and create urls for desired bands and spatial resolution.
        If there's a band that is not present in specified spatial resolution, this band is going to be pulled either
        from lower or higher spatial resolution.
        For instance band B08 is only present in 10m spatial res. and therefore when we ask for this band with spatial
        resolution set to 60m, 10m band is downloaded.
        @param: meta - content of xml in string form
        @param: entry - data of the tile from the request
        @param: what spatial resolution should be
        :return: tuple - url and the name of the raster
        """
        bands = set(bands)
        raster_urls = []
        bands = Downloader.extract_l2a_urls(spatial_res, meta, raster_urls, bands, entry)

        if len(bands) == 0:
            return raster_urls

        if spatial_res != "10m":
            bands = Downloader.extract_l2a_urls("10m", meta, raster_urls, bands, entry)
            if len(bands) == 0:
                return raster_urls
        if spatial_res != "20m":
            bands = Downloader.extract_l2a_urls("20m", meta, raster_urls, bands, entry)
            if len(bands) == 0:
                return raster_urls
        if spatial_res != "60m":
            bands = Downloader.extract_l2a_urls("60m", meta, raster_urls, bands, entry)
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
        return Downloader.calculate_hash_unix(path, 'md5sum') == check_sum or Downloader.calculate_hash_unix(path,
                                                                                                             'sha3sum -a 256') == check_sum

    def download_granule_full(self, unzip: bool = True):
        """
        Generator implementation. This downloads the granule only when you ask for it.
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

    def download_granule_bands(self, bands: List[str] = None, primary_spatial_res: Optional[str] = None):
        """
        Generator implementation. This downloads the granule only when you ask for it.
        @param primary_spatial_res: 20 -> 20m, 10 -> 10m, 60 -> 60m
        @param bands: ["B01", ... ]
        """
        bands = self.__download_bands_checker(bands, primary_spatial_res)
        self.__before_download()
        for mercator, entries in self.__get_next_download():
            working_path = self.root_path + mercator  # path where all datasets are going to be downloaded
            status = self.__download_data(entries, working_path, bands, primary_spatial_res)
            if status:
                yield working_path
        log.info("All downloaded")

    def download_granule_bands_threads(self, bands: List[str] = None, primary_spatial_res: str = None):
        """
        Downloads all the desired data at once with support of threads.
        """
        bands = self.__download_bands_checker(bands, primary_spatial_res)
        self.__before_download()
        data = list(self.__get_next_download())
        with ThreadPoolExecutor(max_workers=5) as executor:
            for mercator, entries in data:
                executor.submit(self.__download_data, entries, self.root_path + mercator, bands, primary_spatial_res)

    def __download_bands_checker(self, bands: List[str] = None, primary_spatial_res: str = None):
        # Check if spatial resolution for L2A is Ok
        if self.product_type == "S2MSI2A" and \
                (primary_spatial_res is None or
                 (primary_spatial_res is not None and primary_spatial_res not in ["10m", "20m", "60m"])):
            raise ValueError("Bad spatial resolution or None")
        # Check if provided bands are Ok
        if bands is None and self.product_type == "S2MSI2A":
            prim = 60 if primary_spatial_res == "60m" else (20 if primary_spatial_res == "20m" else 10)
            bands = bands_for_resolution(prim)
        elif bands is None:
            raise IncorrectInput("Please specify L1C bands")
        return bands

    def __download_data(self, entries, working_path, bands, primary_spatial_res):
        Downloader.prepare_dir(working_path)
        status = False
        for entry in entries:
            data_set_path = working_path + os.path.sep + entry['title'] + ".SAFE" + os.path.sep
            Downloader.prepare_dir(data_set_path)
            manifest = self.download_meta_data(Downloader.manifest_url.format(entry["id"], entry["title"]),
                                               data_set_path + "manifest.safe")
            manifest_imgs = Downloader.parse_manifest(manifest)
            meta_data = self.download_meta_data(
                Downloader.meta_url.format(entry["id"], entry["title"], self.meta_data_name),
                data_set_path + self.meta_data_name)
            raster_urls = None
            if self.product_type == "S2MSI2A":
                raster_urls = Downloader.get_raster_urls_l2a(meta_data, entry, primary_spatial_res, bands)
            else:
                raster_urls = Downloader.get_raster_urls_l1c(meta_data, entry, bands)
            results = []
            for url, name in raster_urls:
                check_sum = Downloader.extract_check_sum(name, manifest_imgs)
                results.append(self.download_file(url, data_set_path + name + ".jp2", check_sum=check_sum))
            #  Check if there's corrupted file (all check sums must match otherwise this dataset will be discarded)
            status = all(results)
            if not status:
                log.warning("Corrupted download this dataset will be discarded.")
                shutil.rmtree(data_set_path)
            else:
                log.debug("Check sum OK!")
        return status

    # all_... methods download everything at once
    def download_full_all(self, unzip: bool = False) -> List[str]:
        """
        Downloads whole dataset and every requested tile.
        Returns list of paths to the downloaded content.
        """
        return list(self.download_granule_full(unzip=unzip))

    def download_bands_all(self, primary_spatial_res: str, bands: List[str] = None):
        """
        Downloads every granule and in each granule required band.
        @param primary_spatial_res: spatial
        @param bands: required bands
        @return: paths to the files
        """
        return list(self.download_granule_bands(bands, primary_spatial_res))

    #  Mangled

    @staticmethod
    def extract_check_sum(name, manifest_imgs) -> Optional[str]:
        """
        index and res attributes are there for backward compatibility idk why are the manifests different
        """

        # calculating checksum on windows not yet supported
        if sys.platform == "win32":
            return None

        for img in manifest_imgs:
            try:
                index = 1
                res = 3
                if len(img.childNodes) == 1:
                    index = 0
                    res = 1
                if len(img.childNodes) == 0:
                    continue
                children = img.childNodes[index].childNodes
                if re.search(name, children[index].attributes["href"].value):
                    return children[res].firstChild.nodeValue
            except Exception as _:
                log.error("Extract check sum error...Recoverable")
        return None

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
        self.__obj_cache['time'] = self.time_str if self.time_str is not None else \
            "[" + self.date[0].strftime("%Y-%m-%d") + "T00:00:00.000Z" + " TO " + self.date[1] \
                .strftime("%Y-%m-%d") + "T23:59:59.999Z]"
        result = self.url + "search?q=( platformname:{} AND producttype:{} AND cloudcoverpercentage:{} " \
                            "AND beginposition:{}".format(self.platform_name, self.product_type,
                                                          self.__obj_cache['cloud'], self.__obj_cache['time'])
        # suffix = "&rows=100&format=json"
        if self.polygon:
            self.__obj_cache['polygon'] = True
            return [result + ' AND footprint:"Intersects(Polygon(({})))")'.format(
                ",".join("{} {}".format(p[0], p[1]) for p in list(self.polygon.exterior.coords)))]  # + suffix
        if self.mercator_tiles:
            return [result + " AND *{}*)".format(tile) for tile in self.mercator_tiles]  # + suffix
        if self.tile_uuids:
            return [self.url + "?=(uuid:{})".format(uuid) for uuid in self.tile_uuids]
        return [result + " AND {} ".format(self.text_search)]  # + suffix

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
        #  A bit clumsy but it's made to speed up the response of the server when user is creating job,
        #  it's just a confirmation that the job is valid and we've got datasets to work with
        #  NEW! : URLS ARE WITHOUT &rows=100&format=json SUFFIX FOR EASIER PAGING IF NEEDED
        self.__obj_cache['urls'] = self.__build_info_queries()
        for url in self.__obj_cache['urls']:
            url += "&start=0&rows=100&format=json"  # Add suffix | in post init queries add suffix based on overall_datasets
            self.__obj_cache['requests'][url] = pandas.read_json(self.session.get(url).content)['feed']
            self.overall_datasets += int(self.__obj_cache['requests'][url]['opensearch:totalResults'])
            if self.overall_datasets > 0:
                log.info("Required minimum of datasets achieved: {}".format(self.overall_datasets))
                return

        raise IncorrectInput("Not enough datasets to download or run the pipeline for this input")

    def __parse_cached_response(self):
        """
        Method parses responses for initial queries and sort them based on the mercator id (position)
        and after we download these data "mercator by mercator".
        Response for Polygon is unstructured and messy for us, since we download the datasets by tiles
        and therefore we are able to run pipeline meanwhile another granule(tile) dataset is downloading.
        """
        #  If there is only one result the request contains only dict to follow the pattern we will wrap it with list
        urls = list(self.__obj_cache['requests'].values())
        entries = []
        for url in urls:
            if type(url['entry']) == list:
                entries += url['entry']
            else:
                entries.append(url['entry'])

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
        base = 0
        suffix = "&start={}&rows=100&format=json"
        for url in self.__obj_cache['urls']:
            if self.__obj_cache['polygon']:
                while self.overall_datasets > 100:
                    base += 1
                    n_url = url
                    n_url += suffix.format(base)
                    self.overall_datasets -= 100
                    self.__obj_cache['requests'][n_url] = pandas.read_json(self.session.get(n_url).content)['feed']
                else:
                    # Already have full request just parse it
                    break
            url += suffix.format(0)  # not supported outside polygon definition
            if url not in self.__obj_cache['requests']:
                self.__obj_cache['requests'][url] = pandas.read_json(self.session.get(url).content)['feed']
            # entry for uuid is just dict
        self.__parse_cached_response()

    #  Static

    @staticmethod
    def calculate_hash_unix(path: str, _hash: str):
        process = None
        if platform == "win32":
            # process = subprocess.Popen(f"CertUtil -hashfile {path} MD5", shell=True,
            #                            stdout=subprocess.PIPE)
            return None
        else:
            process = subprocess.Popen(f"{_hash} {path}", shell=True,
                                       stdout=subprocess.PIPE)
        out = process.communicate()
        if len(out) < 1:
            return None
        try:
            out = out[0].decode().split(" ")[0]
            return out
        except Exception as _:
            log.error("Exception during checksum parsing skipping.")
        return None

    @staticmethod
    def parse_manifest(manifest):
        regex = re.compile("IMG_DATA")
        xmldom = minidom.parseString(manifest)
        items = xmldom.getElementsByTagName('dataObject')
        img_elements = []
        for item in items:
            id = item.attributes['ID'].value
            if regex.search(id):
                img_elements.append(item)
        #  Items now consist of img elements where is the checksum
        return img_elements

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

    # @staticmethod
    # def is_valid_date_range(date: Tuple) -> bool:
    #     if date is None or len(date) != 2 or (date[1] - date[0]).days < 0:
    #         return False
    #     return True

    def validate_mercator_tiles(self, tiles: List[str]) -> List[str]:
        if tiles is None:
            return []
        r = re.compile("^(T?)[\\d]{1,2}[A-Z]{0,3}")
        validated = []
        for entry in tiles:
            if r.search(entry):
                validated.append(entry)
            else:
                log.warning(f"BAD FORMAT: {entry} is not a tile, it won't be included in the tiles to download!")
        self.mercator_tiles = validated

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

    @staticmethod
    def is_valid_sentinel_type(_type):
        return _type in ["S2MSI2A", "S2MSI1C"]
