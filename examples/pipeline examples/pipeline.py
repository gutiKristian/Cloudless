"""
First step is to download the data, we will use our Download module.
"""
import datetime

from Download.Sentinel2 import Downloader

# We will download tiles over Brno,
# mercator tile 33UXQ is tile in UTM33, srs is EPSG:32633

download = Downloader("example", "thesis", mercator_tiles=["33UXQ"],
                      date=(datetime.datetime(2021, 5, 8), datetime.datetime(2021, 5, 23)))

# download 60m rasters - not whole dataset (around 1GB), good for slow internet
paths = download.download_bands_all("60m", ["B04", "B03", "B02", "SCL", "B8A"])

# where are the datasets stored
for path in paths:
    print(path)

# Run computations on datasets
