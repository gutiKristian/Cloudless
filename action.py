from Pipeline import Sentinel2Runner
from osgeo import gdal

if __name__ == '__main__':
    path = "C:\\Users\\krist\\Desktop\\T33UXQ"
    s2 = Sentinel2Runner.S2Runner(path, 20)
    s2.run_ndvi_cloud_masking()