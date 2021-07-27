import os


class S2Parser:

    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def get_metadata_path(granule_path: str, granule_type: str) -> str:
        if S2Parser.supported_granule_type(granule_type):
            raise NotImplemented
        return granule_path + os.path.sep + ("MTD_MSIL2A.xml" if granule_type == "L2A" else "MTD_MSIL1C.xml")

    @staticmethod
    def supported_granule_type(_type: str) -> bool:
        return _type in ["L1C", "L2A"]
