import os
import pytest
from Pipeline.utils import *
from Pipeline.Granule import S2Granule
import numpy as np
from xml.etree import ElementTree  # xml
import pathlib
import re


class TestUtils:
    safe_format = "S2A_MSIL2A_20201108T095221_N0214_R079_T33UXQ_20201108T121244.SAFE"
    supported = [5, 10, 15, 18, 45]
    images = []
    path = str(pathlib.Path(__file__).parent.absolute())

    @classmethod
    def setup_class(cls):
        tree = ElementTree.parse(TestUtils.path + os.path.sep + "MTD_MSIL2A.xml")
        root = tree.getroot()
        images = []
        for image in look_up_raster(root, 'Granule')[0]:
            text = image.text
            if os.name == 'nt':
                text = text.replace('/', '\\')
            images.append(text + '.jp2')
        TestUtils.images = images
        return cls()

    def test_s2_safe_format(self):
        assert s2_is_safe_format(TestUtils.safe_format)
        assert not s2_is_safe_format("S2A_MSIL2A_20201108T095221_N0214_R079_T33UXQ_20201108T121244")

    def tests_s2_get_resolution(self):
        assert s2_get_resolution(10) == (10980, 10980)
        assert s2_get_resolution(20) == (5490, 5490)
        assert s2_get_resolution(60) == (1830, 1830)

    def test_bands_for_resolution(self):
        with pytest.raises(Exception):
            bands_for_resolution(30)
        assert bands_for_resolution(20) == ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "AOT", "SCL"]
        assert bands_for_resolution(10) == ["B02", "B03", "B04", "B08", "AOT"]
        assert bands_for_resolution(60) == ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12",
                                            "AOT"]

    def test_supported_slice_index(self):
        for s in TestUtils.supported:
            assert is_supported_slice(s)
        assert not is_supported_slice(12)

    def test_find_closest_slice(self):
        actual = [6, 11, 16, 19, 46, 35, 8]
        supported = [5, 10, 15, 18, 45, 45, 10]
        for i in range(len(actual)):
            print(actual[i])
            assert find_closest_slice(actual[i]) == supported[i]

    def test_slice_raster(self):
        expected = [
            (25, 1098, 1098),
            (100, 549, 549),
            (225, 366, 366),
            (324, 305, 305),
            (2025, 122, 122)
        ]
        for i in range(len(TestUtils.supported)):
            arr = np.zeros(shape=(5490, 5490))
            arr = slice_raster(TestUtils.supported[i], arr)
            assert arr.shape == expected[i]

    def test_found_bands_spatial_res(self):
        """
        Tests whether the intervals are correct
        TestUtils.images -> [0:7] = 10m; [7:20] = 20m; [20::] = 60m;
        """
        pattern = re.compile("_(\\d+)(m)")

        def _check_res(arr, res):
            for b in arr:
                assert res == pattern.findall(b)[0][0]

        _check_res(TestUtils.images[0:7], '10')
        _check_res(TestUtils.images[7:20], '20')
        _check_res(TestUtils.images[20::], '60')

    def test_verify_bands(self):
        # 10m
        img_paths = TestUtils.images[0:7]
        # Bands we want the path to
        desired_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B08", "TCI",
                         "WVP", "B11", "B12", "AOT", "SCL"]
        # ["B02", "B03", "B04", "B08", "AOT", "WVP", "TCI"] are expected top be 10m
        #  expected 20m
        m20 = ["B05", "B06", "B07", "B8A", "B11", "B12", "SCL"]
        found = set(verify_bands(TestUtils.images, img_paths, desired_bands, 10))
        assert len(found) == len(desired_bands)
        spatial = re.compile("_(\\d+)(m)")
        band = re.compile("B[0-9]+A?|TCI|AOT|WVP|SCL")
        for img in found:
            sp = spatial.findall(img)[0][0]
            bn = band.findall(img)[0]
            if sp == '10':
                assert bn not in m20
            else:
                assert bn in m20
