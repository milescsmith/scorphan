# orphan utility functions for working wtih single cell multiomics data
from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from scorphan import aggregation as ag
from scorphan import preprocessing as pp
from scorphan import tools as tl

logger.disable("scorphan")

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__ = ["pp", "tl", "ag"]
