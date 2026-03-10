# orphan utility functions for working wtih single cell multiomics data
# import delayed_import

# delayed_import.enable(__name__)

from importlib.metadata import PackageNotFoundError, metadata, version

from loguru import logger

from . import _aggregation as ag
from . import _plotting as pl
from . import _preprocessing as pp
from . import _tools as tl
from . import _utils as ut
from . import gpu, pathways

logger.disable("scorphan")

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

try:
    __author__, __email__ = [x.strip("> ") for x in metadata(__name__)["Author-email"].split("<")]

except KeyError:  # pragma: no cover
    __email__ = "unknown"


__all__ = ["ag", "gpu", "pathways", "pl", "pp", "tl", "ut"]

__doc__ = """\
scorphan
--------

Collection of orphan helper fuctions for single cell omics data analysis:

"""
