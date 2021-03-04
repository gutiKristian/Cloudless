from Pipeline import Sentinel2Runner
from Pipeline.logger import log
from Pipeline.utils import *
if __name__ == '__main__':
    path = "C:\\Users\\krist\\Desktop\\T33UXQ"
    s2 = Sentinel2Runner.S2Runner(path, 20)
    s2.optimized_ndvi(constraint=2)
