from enum import Enum


class StrEnum(Enum):
    def __str__(self):
        return self.value    


class DownsampleScale(StrEnum):
    X2 = "X2"
    X4 = "X4"


class DownsampleMethod(StrEnum):
    DEFAULT = "default"
    UNKNOWN = "unknown"


class PaddingMode(StrEnum):
    REFLECT = "reflect"
    REPLICATE = "replicate"
