import pytest
from Pipeline.Granule import S2Granule


class TestGranule:

    def test_no_path(self):
        with pytest.raises(FileNotFoundError):
            S2Granule("test", 10, ["B02"])
