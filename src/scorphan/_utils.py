from functools import partial

import numpy as np
import numpy.typing as npt
import scipy as sp
import sparse
from numba import float32, float64, guvectorize, int32, int64, vectorize
from scipy.sparse import issparse
from scipy.stats import median_abs_deviation


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


def value_quantile(arr: np.ndarray) -> np.ndarray:
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
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x


@guvectorize([(float64[:, :], float64, float64, float64[:, :])], "(m,n),(),()->(m,n)")
def percentile_trim_rows(
    arr: npt.ArrayLike, lower: float = 0.10, upper: float = 0.99, res: npt.ArrayLike = None
) -> npt.ArrayLike:
    """
    Row-by-row, calculate the lower and upper percentiles and then use those to replace values that are
    below or above them, respectively
    """
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
