# orphan utility functions for working wtih single cell multiomics data
from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from scorphan import _aggregation as ag
from scorphan import _preprocessing as pp
from scorphan import _tools as tl
from scorphan import _utils as ut

logger.disable("scorphan")

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__ = ["pp", "tl", "ag", "ut"]
