import numpy as np
import sparse
from scipy.sparse import issparse


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
