"""
First step is to download the data, we will use our Download module.
"""
import datetime

from Download.Sentinel2 import Downloader
from Pipeline.Worker import S2Worker

# We will download tiles over Brno,
# mercator tile 33UXQ is tile in UTM33, srs is EPSG:32633

download = Downloader("ultra", "secret", mercator_tiles=["33UXQ"],
                      date=(datetime.datetime(2021, 9, 1), datetime.datetime(2021, 9, 30)))

# download 60m rasters - not whole dataset (around 1GB), good for slow internet
paths = download.download_granule_bands_threads("60m", ["B04", "B03", "B02", "SCL", "B8A"])
# where are the datasets stored
for path in paths:
    print(path)

# Run computations on datasets
worker = S2Worker(paths[0], 60, output_bands=["B04", "B03", "B02", "SCL", "B8A"], slice_index=10)
from Pipeline.Task import PerTile
PerTile.perform_computation(worker)
