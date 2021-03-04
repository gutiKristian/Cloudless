from Pipeline import Sentinel2Runner
from Pipeline.logger import log
from Pipeline.utils import *
from timeit import default_timer as timer

if __name__ == '__main__':
    start = timer()
    path = "D:\\Work\\T33UXQ"
    s2 = Sentinel2Runner.S2Runner(path, 20)
    s2.optimized_ndvi(constraint=2)
    end = timer()
    print(end - start)