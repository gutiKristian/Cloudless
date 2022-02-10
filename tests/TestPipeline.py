import os

import numpy as np
import pytest
from Pipeline.utils import *
from Pipeline.Worker import S2Worker
from Pipeline.Granule import S2Granule
from Pipeline.Band import Band
from Pipeline.Mask import S2JIT
from Pipeline.Detectors import S2Detectors
import pathlib


class TestPipeline:
    supported = [5, 10, 15, 18, 45]
    dataset = "T33UXQ"
    images = []
    path = str(pathlib.Path(__file__).parent.absolute()) + os.path.sep + dataset
    worker: S2Worker = None
    granule: S2Granule = None

    @classmethod
    def setup_class(cls):
        TestPipeline.worker = S2Worker(TestPipeline.path, 60, output_bands=["B02", "B03", "B04", "B8A", "SCL"])
        TestPipeline.granule = TestPipeline.worker.granules[0]
        return cls()

    """
    WORKER
    """

    def test_bad_slice_index(self):
        with pytest.raises(Exception):
            S2Worker(path=self.path, spatial_resolution=60, slice_index=3)

    def test_extracted_mercator(self):
        assert self.worker.mercator == TestPipeline.dataset

    def test_save_path(self):
        assert self.worker.save_result_path == self.path + os.path.sep + "result"

    def test_number_of_datasets(self):
        assert len(self.worker.datasets) == len(os.listdir(self.path))
        assert len(self.worker.granules) == len(self.worker.datasets)

    """
    GRANULE
    """

    def test_no_path(self):
        with pytest.raises(FileNotFoundError):
            S2Granule("test", 10, ["B02"])

    def test_spatial(self):
        assert TestPipeline.granule.spatial_resolution == 60

    def test_granule_type(self):
        assert TestPipeline.granule.granule_type == "L2A"

    def test_meta_data(self):
        assert TestPipeline.granule.meta_data is not None
        assert TestPipeline.granule.meta_data_gdal is not None
        assert TestPipeline.granule.data_take is not None

    def test_doy(self):
        assert TestPipeline.granule.doy is not None

    def test_slice_index(self):
        assert TestPipeline.granule.slice_index == 1

    def test_parsed_bands(self):
        assert TestPipeline.granule.bands != {}

    def test_parsed_projection(self):
        assert TestPipeline.granule.get_projection() == "EPSG:32633"

    def test_bands_stacking(self):
        assert TestPipeline.granule.stack_bands().shape == (5, 1830, 1830)
        assert TestPipeline.granule.stack_bands(dstack=True).shape == (1830, 1830, 5)

    def test_get_bands(self):
        assert type(TestPipeline.granule["B03"]) == Band
        assert type(TestPipeline.granule["B04"]) == Band
        assert type(TestPipeline.granule["B02"]) == Band

    """
    BANDS
    """

    def test_band_dataset_manipulation(self):
        TestPipeline.granule["B03"].load_raster()
        assert type(TestPipeline.granule["B03"].raster()) == numpy.ndarray

    """
    JIT-ed computations
    """

    def test_median_computation_anomaly(self):
        data = np.ones(shape=(3, 10, 10))
        data[0] = data[0] * 5
        data[2] = data[2] * 9
        median = np.median(data, axis=0)
        median[0, 0] = 0  # no data anomaly
        result = S2JIT.s2_median_analysis(data, median)
        expected = np.median(data, axis=0)
        assert np.array_equal(result, expected)

    def test_median_computation_no_anomaly(self):
        #  Without no data values
        data = np.ones(shape=(3, 10, 10))
        data[0] = data[0] * 10
        data[2] = data[2] * 18
        median = np.median(data, axis=0)
        result = S2JIT.s2_median_analysis(data, median)
        expected = np.median(data, axis=0)
        assert np.array_equal(result, expected)

    def test_ndvi_pixel_analysis(self):
        #  CURRENT NDVI SETUPS
        ndvi = [np.ones(shape=(10, 10)), np.ones(shape=(10, 10))]
        ndvi[0] = ndvi[0] * (-1)
        ndvi_current_max = np.ones(shape=(10, 10)) * (-10)
        #  CURRENT DATA SETUP
        data = [np.ones(shape=(3, 10, 10)), np.ones(shape=(3, 10, 10)) * 10]
        #  DOY SETUPS
        doy = np.zeros(shape=(10, 10))
        doys = np.array([1, 2])
        expected_doy = np.ones(shape=(10, 10)) * 2
        # Preallocated result data
        result_data = np.zeros(shape=(3, 10, 10))

        S2JIT.s2_ndvi_pixel_analysis(ndvi, ndvi_current_max, data, doys, result_data, doy, 10, 10)
        assert np.array_equal(result_data, data[1])  # TEST IF FINAL DATA HAS BEEN CHANGED, SAME FOR DOY
        assert np.array_equal(doy, expected_doy)
        assert np.array_equal(ndvi_current_max, ndvi[1])  # TEST IF THE CURRENT MAX NDVI HAS BEEN CHANGED AS WELL

    """
    DETECTORS
    """

    def test_scl_detector(self):
        """
        Test if we if the function filters out correct number of pixels.
        """
        #  Make a copy
        c = TestPipeline.granule["SCL"].raster()
        #  All no data
        TestPipeline.granule.bands[60]["SCL"].raster_image = c * 0
        expected_sum = 33489000
        assert expected_sum == S2Detectors.scl(TestPipeline.granule).sum()
        #  No cloud test
        TestPipeline.granule.bands[60]["SCL"].raster_image = c * 0 + 3
        assert 0 == S2Detectors.scl(TestPipeline.granule).sum()
