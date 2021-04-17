import pytest
from Pipeline.utils import *


class TestUtils:
    safe_format = "S2A_MSIL2A_20201108T095221_N0214_R079_T33UXQ_20201108T121244.SAFE"

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
        assert bands_for_resolution(60) == ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12", "AOT"]

    def test_supported_slice_index(self):
        supported = [5, 10, 15, 18, 45]
        for s in supported:
            assert is_supported_slice(s)
        assert not is_supported_slice(12)

    def test_find_closest_slice(self):
        actual = [6, 11, 16, 19, 46, 35, 8]
        supported = [5, 10, 15, 18, 45, 45, 10]
        for i in range(len(actual)):
            print(actual[i])
            assert find_closest_slice(actual[i]) == supported[i]
