import re
from collections.abc import Sequence
from functools import partial, wraps
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import polars as pl
import scipy as sp
import sparse
from loguru import logger
from numba import float32, float64, guvectorize, int32, int64, vectorize
from rich.console import Console
from scipy.sparse import issparse
from scipy.stats import median_abs_deviation

console = Console()

def not_yet_implemented(func):
    @wraps(func)
    def __inner(*args, **kwargs):
        msg = f"`{func.__name__}` is not yet fully implemented and does not work at this time"
        raise NotImplementedError(msg)
    return __inner


def is_integer_array(arr) -> bool:
    """
    Test if an array is really all integers
    """
    if issparse(arr):
        # have to convert to `sparse.COO` here as `scipy.csr_matrix` either no longer or never did support numpy
        # calculations on it
        return not (np.mod(sparse.asCOO(arr), 1) != 0).any()
    else:
        return not (np.mod(arr, 1) != 0).any()


def value_percentile(arr: np.ndarray) -> np.ndarray:
    # hacky way to loop over the array of counts and calculate each's quantile.
    # not sure why percentileofscore isn't already vectorized
    # and we have to use partial here because percentileofscore's function
    # signature is "iter, item" instead of "item, iter", meaning I cannot just pass
    # the array to score as the vectorized first argument
    return np.vectorize(partial(sp.stats.percentileofscore, a=arr))(score=arr)


# using the numba.vectorize decorator speeds this up about 13x
@vectorize(
    [
        float64(float64, float64, float64),
        float32(float32, float32, float32),
        int64(int64, int64, int64),
        int32(int32, int32, int32),
    ],
    nopython=True,
    fastmath=True,
)
def above_below(x: float, lower: float, upper: float) -> float:
    """
    Parameters
    ----------
    x : float

    lower : float

    upper : float

    Returns
    -------
    """
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x


@guvectorize([(float64[:, :], float64, float64, float64[:, :])], "(m,n),(),()->(m,n)")
def percentile_trim_cols(
    arr: npt.ArrayLike, lower: float = 0.10, upper: float = 0.99, res: npt.ArrayLike | None = None
) -> npt.ArrayLike:
    """
    Row-by-row, calculate the lower and upper percentiles and then use those to replace values that are
    below or above them, respectively
    """
    if res is None:
        res = np.zeros_like(res)
    for i in range(arr.shape[1]):
        lower_bounds = np.quantile(arr[:, i], lower)
        upper_bounds = np.quantile(arr[:, i], upper)
        res[:, i] = above_below(arr[:, i], lower_bounds, upper_bounds)


@guvectorize([(float64[:, :], float64, float64, float64[:, :])], "(m,n),(),()->(m,n)")
def percentile_trim_rows(
    arr: npt.ArrayLike, lower: float = 0.10, upper: float = 0.99, res: npt.ArrayLike | None = None
) -> npt.ArrayLike:
    """
    Row-by-row, calculate the lower and upper percentiles and then use those to replace values that are
    below or above them, respectively

    NOTE: even though there are defaults listed here, they DO NOT WORK
    I don't yet know why numba ignores them.
    """
    if res is None:
        res = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        lower_bounds = np.quantile(arr[:, i], lower)
        upper_bounds = np.quantile(arr[:, i], upper)
        res[:, i] = above_below(arr[:, i], lower_bounds, upper_bounds)


# stolen from https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#filtering-low-quality-cells
def is_outlier(adata, metric: str, nmads: int):
    met = adata.obs[metric]
    outlier = (met < np.median(met) - nmads * median_abs_deviation(met)) | (
        np.median(met) + nmads * median_abs_deviation(met) < met
    )
    return outlier


def repair_anndataset(
    bad_file: Path,
    new_file: Path,
    backing_files_dir: Path,
    sample_name_pattern: str,
    overwrite: bool = False,
) -> None:
    """
        For whatever reason, the AnnDataSet files produced by SnapATAC2 seem to rot quickly. Or, SnapATAC2 might crash.
        Even closed properly, they rapidly develop a problem with the adatas.uns["AnnDataSet"]
        polars DataFrame - something about wrong length or a bad EOF

        I don't want to keep spending time either reproducing this thing or remembering how to repair it.

        Parameters
        ----------
        bad_file : Path
            Path to the file that needs repairing
        new_file : Path
            Path to where a new object will be written
        backing_files_dir: Path
            Path to where the individual backing files are located
        sample_name_pattern : str
            A regular expression corresponding to the sample name used for the individual backing files
        overwrite : bool, default False
            If the new_file already exists, should it be overwritten?

        Returns:
        --------
        None
            A new file is created.

        Example
        -------
        >>> repair_anndataset(
        >>> bad_file=data_folder.joinpath(
        >>>     "bad_file.h5ads"
        >>> ),
        >>> new_file=data_folder.joinpath(
        >>>     "new_good_file.h5ads"
        >>> ),
        >>> backing_files_dir=data_folder.joinpath("atacseq_backing_files"),
        >>> sample_name_pattern=r"Sample[0-9]+",
        >>> overwrite=True,
    )
    """

    if overwrite and new_file.exists():
        new_file.unlink()

    with (
        h5py.File(
            bad_file,
            mode="r",
        ) as f,
        h5py.File(
            new_file,
            mode="a",
        ) as g,
    ):
        for i in list(f):
            if i != "uns":
                g.copy(source=f[i], dest=i)

        g.create_group("uns")
        for j in list(f["uns"]):
            if j != "AnnDataSet":
                g["uns"].copy(source=f["uns"][j], dest=j)

        backing_files = pl.from_dict(
            {
                "keys": [
                    re.findall(pattern=sample_name_pattern, string=str(_))[0] for _ in backing_files_dir.glob("*.h5ad")
                ],
                "file_path": list(backing_files_dir.glob("*.h5ad")),
            }
        )
        backing_files = backing_files.with_row_index().cast({"index": str})

        g["uns"].create_group("AnnDataSet")

        for col in backing_files.columns:
            g["uns"]["AnnDataSet"].create_dataset(
                name=col,
                shape=(backing_files.shape[0],),
                data=backing_files[col].to_list(),
            )

# Source - https://stackoverflow.com/a/73686304
# Posted by Alex44, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-05, License - CC BY-SA 4.0

def h5_tree(val: h5py.Group, pre=""):
    items = len(val)
    for key, _val in val.items():
        items -= 1
        if isinstance(_val, h5py.Group):
            if items == 0:
                print(f"{pre}└── {key}")
                h5_tree(_val, f"{pre}    ")
            else:
                print(f"{pre}├── {key}")
                h5_tree(_val, f"{pre}│   ")
        elif items == 0:
            try:
                print(f"{pre}└── {key} ({len(_val)})")
            except TypeError:
                print(f"{pre}└── {key} (scalar)")
        else:
            try:
                print(f"{pre}├── {key} ({len(_val)})")
            except TypeError:
                print(f"{pre}├── {key} (scalar)")


class ArgumentError(Exception):
    pass

def make_list_if_not(obj: Any) -> list[Any]:
    """make_list_if_not: is it a list? Make it one!

    Parameters
    ----------
    obj : Any
        Any ol' thing you want to test to see if it is a list and, if not,
        make it one

    Returns
    -------
    list[Any]
    """
    return obj if isinstance(obj, Sequence) else [obj]
