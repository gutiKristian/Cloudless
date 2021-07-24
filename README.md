# Sentinel pipeline

![](https://github.com/gutiKristian/Cloudless/actions/workflows/main.yml/badge.svg?branch=dev)

Sentinel pipeline allows you to create beautiful cloudless image mosaics from
Sentinel 2 imagery. 
You may pick from the variety of algorithms.


### Per-pixel algorithms

- Max ndvi
- Median
- S2Cloudless ML model - requires L1C dataset

### Per-tile algorithms with the masking based on

- S2Cloudless ML model - requires L1C dataset
- Sentinel's 2 SCL
- Average NDVI values in the tile - experimental


### Sentinel pipeline Downloader

This repository also includes the Download utility that simplifies the process
of downloading these datasets.

- Download module supports L1C and L2A datasets
- Download only data you desire and speed up the creation by not downloading whole datasets
- Filter datasets with many accessible properties such as
    - Time (ingestion date, FROM - TO)
    - AOI (area of interest) defined by polygon, tile id (mercator)
    - Product type
    - Cloud coverage
    - Custom text search within the rules of Copernicus api
- Download data from any Copernicus collaboration server (default - Cesnet), WARNING: In order to download data from the server you will have to set up and account first


### Other usage
Pipeline does not have to be used only to create cloud mosaics, it could be used as a tool to manipulate
the sentinel's 2 data as a whole or individually. For instance, you might perform:
- re-projection to custom CRS
- up-sampling or down-sampling to custom spatial resolution
- easily compute intermediate products such as NDVI, Agriculture index and more (more can be easily added)
- easily compute masks/thresholding on custom band with a few lines of code, 
    - it's easy as : mask = GRANULE["B01"] > 100

### Speed
Even though Python does not belong to the set of fast langues. The pipeline is from the very beginning being built
with aimt to reach maximum speed and memory optimizations within the computations. This allows us to run multiple pipelines workers simultaneously and push the effectivity even further. Speaking of speed pipeline uses rasterio library built on top of GDAL, to open the JPEG2000 imagery. Decoding is done by openjpeg library witihin these modules and supports multithreading.
Regarding memory, optimizations have been made in the process of creating cloudless products, user is able to pick the number of 
maximum opened datasets within the computation. This is big news for users with laptops that have limited RAM and as we know these datasets are sometimes pretty big even for 16GB of RAM.
