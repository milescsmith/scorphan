# orphan utility functions for working wtih single cell multiomics data
# import delayed_import

# delayed_import.enable(__name__)

from importlib.metadata import PackageNotFoundError, version, metadata

from loguru import logger

from scorphan import _aggregation as ag
from scorphan import _preprocessing as pp
from scorphan import _tools as tl
from scorphan import _utils as ut
from scorphan import gpu

logger.disable("scorphan")

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

try:
    __author__, __email__ = [x.strip("> ") for x in metadata(__name__)["Author-email"].split("<")]

except KeyError:  # pragma: no cover
    __email__ = "unknown"


__all__ = ["ag", "gpu", "pp", "tl", "ut"]

__doc__ = """\
scorphan
------------

Collection of orphan helper fuctions for single cell omics data analysis:

.. autosummary::
   :toctree: .

   atac
   gpu
   pp
   tl
   ut


atac
----

Utilities for working with scATAC-seq data

Note this is largely disabled at the moment due to
issues related to building certain rust dependencies
of snapatac2

.. autosummary::
   :toctree: .

   EnsemblRestClient
   available_species
   get_resources

Prepare a new species for use in MerryCRISPR

.. autosummary::
   :toctree: .

   build_bowtie_index


seqextractor
-------------

Scan GTF for information.

.. autosummary::
   :toctree: .

   display_gtf_features
   display_gtf_genes
   display_gtf_geneids

Create FASTAs to search for spacers.

.. autosummary::
   :toctree: .

   extract
   extract_for_tss_adjacent

Utilities for parsing FASTAs or matching annotations

.. autosummary::
   :toctree: .

   match_seq
   split_records

find_spacers
------------

Description

.. autosummary::
   :toctree: .

   find_spacers


on_target_scoring
-----------------

Description

.. autosummary::
   :toctree: .

   on_target_scoring
   score_entry

off_target_scoring
------------------

Find and score potential off-targets

.. autosummary::
   :toctree: .

   hsu_offtarget_score
   sumofftargets
   off_target_discovery
   off_target_scoring


library_assembly
----------------

Description

.. autosummary::
   :toctree: .

   assemble_library
   assemble_paired_library
"""

