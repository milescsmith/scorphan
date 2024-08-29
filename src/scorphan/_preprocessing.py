import numbers
import warnings
from enum import StrEnum
from functools import partial
from typing import Final, Literal

import muon as mu
import numpy as np
import numpy.typing as npt
import scanpy as sc
import scipy as sp
import vaeda
from loguru import logger
from mudata import MuData
from muon import prot as pt
from muon._core.preproc import (
    _jaccard_euclidean_metric,
    _jaccard_sparse_euclidean_metric,
    _make_slice_intervals,
    _sparse_csr_fast_knn,
    _sparse_csr_ptp,
)
from numba import float32, float64, guvectorize, int32, int64, vectorize
from scanpy import logging
from scanpy.neighbors._connectivity import umap as _compute_connectivities_umap
from scanpy.tools._utils import _choose_representation
from scipy.sparse import (
    SparseEfficiencyWarning,
    csr_matrix,
    issparse,
)
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import median_abs_deviation
from umap.umap_ import nearest_neighbors

LARGE_NUMBER_OF_OBSERVATIONS: Final[int] = 50000
LOW_MEMORY_SPARSE_SPLITS: Final[int] = 10000
LARGE_MEMORY_SPARSE_SPLITS: Final[int] = 30000

# revised processing function, including rudimentary protein scrubbing


class DoubletFilter(StrEnum):
    scrublet = "scrublet"
    vaeda = "vaeda"


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
def quantile_trim_rows(
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


# @typechecked
def std_process_run(
    filtered: MuData,
    raw: MuData | None = None,
    min_genes: int = 500,
    max_genes_quantile: float = 0.95,
    min_gene_total_counts: int = 1000,
    min_cells: int = 3,
    max_percent_mt: float = 10.0,
    remove_isotype_outliers: bool = True,
    remove_all_feature_outliers: bool = False,
    max_isotype_counts_quantile: float = 0.95,
    doublet_algorithm: DoubletFilter = DoubletFilter.scrublet,
):
    filtered.var_names_make_unique()
    filtered["rna"].var["mt"] = filtered["rna"].var_names.str.startswith("MT-")
    filtered["rna"].var["ribo"] = filtered["rna"].var_names.str.startswith(("RPS", "RPL"))
    filtered["rna"].var["hb"] = filtered["rna"].var_names.str.contains(r"^HB[^(P)]", regex=True)

    logger.info("Calculating QC metrics")

    sc.pp.calculate_qc_metrics(
        adata=filtered["rna"],
        qc_vars=["mt", "ribo", "hb"],
        percent_top=(20, 50, 100, 200, 500),
        log1p=True,
        inplace=True,
    )

    logger.info("Filtering cells and genes")
    filtered["rna"].obs["outlier"] = (
        is_outlier(filtered["rna"], "log1p_total_counts", 5)
        | is_outlier(filtered["rna"], "log1p_n_genes_by_counts", 5)
        | is_outlier(filtered["rna"], "pct_counts_in_top_20_genes", 5)
        | is_outlier(filtered["rna"], "pct_counts_mt", 3)
        | (filtered["rna"].obs["pct_counts_mt"] > max_percent_mt)
    )

    # mu.pp.filter_var(filtered["rna"], "n_cells_by_counts", lambda x: x >= min_cells)
    # mu.pp.filter_obs(filtered["rna"], "n_genes_by_counts", lambda x: x >= min_genes)
    mu.pp.filter_obs(filtered["rna"], "outlier", lambda x: ~x)

    match doublet_algorithm:
        case DoubletFilter.scrublet:
            logger.info("Running Scrublet")
            sc.pp.scrublet(filtered["rna"])
            mu.pp.filter_obs(filtered["rna"], "predicted_doublet", lambda x: ~x)
        case DoubletFilter.vaeda:
            logger.info("Running vaeda")
            adata = vaeda.vaeda(filtered["rna"].copy(), seed=20150318)
            filtered["rna"].obs = filtered["rna"].obs.join(adata.obs[["vaeda_scores", "vaeda_calls"]], how="left")
            mu.pp.filter_obs(filtered["rna"], "vaeda_calls", lambda x: x == "singlet")

    mu.pp.intersect_obs(filtered)
    logger.info("finding isotype controls")
    isotype_controls = raw.mod["prot"].var.index[raw.mod["prot"].var.index.str.match("Isotype")]

    if remove_isotype_outliers:
        isotype_idx = {_: filtered["prot"].var_names.tolist().index(_) for _ in isotype_controls}
        for _ in isotype_idx:
            if issparse(filtered["prot"].X):
                filtered["prot"].obs[f"{_} percentile"] = value_quantile(
                    filtered["prot"].X.toarray()[:, isotype_idx[_]]
                )
            else:
                filtered["prot"].obs[f"{_} percentile"] = value_quantile(filtered["prot"].X[:, isotype_idx[_]])

        mu.pp.intersect_obs(filtered)
        for _ in isotype_idx:
            # do this twice because if it is all in the same loop, the quantile
            # calculations are affected by the removal of the first noisy samples
            mu.pp.filter_obs(
                filtered,
                f"prot:{_} percentile",
                lambda x: x < max_isotype_counts_quantile * 100,
            )

    filtered["prot"].layers["counts"] = filtered["prot"].X.copy()
    if raw is not None:
        logger.info("running dsb")
        pt.pp.dsb(
            data=filtered,
            data_raw=raw,
            isotype_controls=isotype_controls,
            add_layer=False,
        )

    logger.info("Normalizing")
    sc.pp.normalize_total(adata=filtered["rna"], target_sum=1e4)

    logger.info("Log-transforming")
    sc.pp.log1p(filtered["rna"])

    logger.info("Finding variable genes")
    sc.pp.highly_variable_genes(filtered["rna"], min_mean=0.0125, max_mean=3, min_disp=0.5)

    filtered["rna"].raw = filtered["rna"]

    logger.info("Scaling data")
    sc.pp.scale(filtered["rna"], max_value=10)


def neighbors(
    mdata: MuData,
    modality_weights: dict[str, int] | None = None,
    manual_weights: dict[str, npt.ArrayLike] | None = None,
    n_neighbors: int | None = None,
    n_bandwidth_neighbors: int = 20,
    n_multineighbors: int = 200,
    neighbor_keys: dict[str, str | None] | None = None,
    metric: Literal[
        "euclidean",
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "hamming",
        "jaccard",
        "jensenshannon",
        "kulsinski",
        "mahalanobis",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "wminkowski",
        "yule",
    ] = "euclidean",
    low_memory: bool | None = None,
    key_added: str | None = None,
    weight_key: str = "mod_weight",
    add_weights_to_modalities: bool = False,
    eps: float = 1e-4,
    copy: bool = False,
    random_state: int | np.random.RandomState | None = 42,
) -> MuData | None:
    """
    Multimodal nearest neighbor search.

    This implements the multimodal nearest neighbor method of Hao et al. and Swanson et al. The neighbor search
    efficiency on this heavily relies on UMAP. In particular, you may want to decrease n_multineighbors for large
    data set to avoid excessive peak memory use. Note that to achieve results as close as possible to the Seurat
    implementation, observations must be normalized to unit L2 norm (see :func:`l2norm`) prior to running per-modality
    nearest-neighbor search.

    References:
        Hao et al, 2020 (`doi:10.1101/2020.10.12.335331 <https://dx.doi.org/10.1101/2020.10.12.335331>`_)
        Swanson et al, 2020 (`doi:10.1101/2020.09.04.283887 <https://dx.doi.org/10.1101/2020.09.04.283887>`_)

    Args:
        mdata: MuData object. Per-modality nearest neighbor search must have already been performed for all modalities
            that are to be used for multimodal nearest neighbor search.
        modality_weights: A factor by which to weigh the modalities when contructing the joint neighborhood graph.
        manual_weights: Supply your own weights!
        n_neighbors: Number of nearest neighbors to find. If ``None``, will be set to the arithmetic mean of per-modality
            neighbors.
        n_bandwidth_neighbors: Number of nearest neighbors to use for bandwidth selection.
        n_multineighbors: Number of nearest neighbors in each modality to consider as candidates for multimodal nearest
            neighbors. Only points in the union of per-modality nearest neighbors are candidates for multimodal nearest
            neighbors. This will use the same metric that was used for the nearest neighbor search in the respective modality.
        neighbor_keys: Keys in .uns where per-modality neighborhood information is stored. Defaults to ``"neighbors"``.
            If set, only the modalities present in ``neighbor_keys`` will be used for multimodal nearest neighbor search.
        metric: Distance measure to use. This will only be used in the final step to search for nearest neighbors in the set
            of candidates.
        low_memory: Whether to use the low-memory implementation of nearest-neighbor descent. If not set, will default to True
            if the data set has more than 50 000 samples.
        key_added: If not specified, the multimodal neighbors data is stored in ``.uns["neighbors"]``, distances and
            connectivities are stored in ``.obsp["distances"]`` and ``.obsp["connectivities"]``, respectively. If specified, the
            neighbors data is added to ``.uns[key_added]``, distances are stored in ``.obsp[key_added + "_distances"]`` and
            connectivities in ``.obsp[key_added + "_connectivities"]``.
        weight_key: Weight key to add to each modality's ``.obs`` or to ``mdata.obs``. By default, it is ``"mod_weight"``.
        add_weights_to_modalities: If to add weights to individual modalities. By default, it is ``False``
            and the weights will be added to ``mdata.obs``.
        eps: Small number to avoid numerical errors.
        copy: Return a copy instead of writing to ``mdata``.
        random_state: Random seed.

    Returns: Depending on ``copy``, returns or updates ``mdata``. Cell-modality weights will be stored in
        ``.obs["modality_weight"]`` separately for each modality.
    """
    randomstate = np.random.RandomState(random_state)
    mdata = mdata.copy() if copy else mdata
    if neighbor_keys is None:
        modalities = mdata.mod.keys()
        neighbor_keys = {}
    else:
        modalities = neighbor_keys.keys()
    neighbors_params = {}
    reps = {}
    observations = mdata.obs.index

    if low_memory or low_memory is None and observations.size > LARGE_NUMBER_OF_OBSERVATIONS:
        sparse_matrix_assign_splits = LOW_MEMORY_SPARSE_SPLITS
    else:
        sparse_matrix_assign_splits = LARGE_MEMORY_SPARSE_SPLITS

    mod_neighbors = np.empty((len(modalities),), dtype=np.uint16)
    mod_reps = {}
    mod_n_pcs = {}
    for i, mod in enumerate(modalities):
        nkey = neighbor_keys.get(mod, "neighbors")
        try:
            nparams = mdata.mod[mod].uns[nkey]
        except KeyError as err:
            msg = f'Did not find .uns["{nkey}"] for modality "{mod}". Run `sc.pp.neighbors` on all modalities first.'
            raise ValueError(msg) from err

        use_rep = nparams["params"].get("use_rep", None)
        n_pcs = nparams["params"].get("n_pcs", None)
        mod_neighbors[i] = nparams["params"].get("n_neighbors", 0)

        neighbors_params[mod] = nparams
        reps[mod] = _choose_representation(mdata.mod[mod], use_rep, n_pcs)
        mod_reps[mod] = use_rep if use_rep is not None else -1  # otherwise this is not saved to h5mu
        mod_n_pcs[mod] = n_pcs if n_pcs is not None else -1

    if n_neighbors is None:
        mod_neighbors = mod_neighbors[mod_neighbors > 0]
        n_neighbors = int(round(np.mean(mod_neighbors), 0))

    ratios = np.full((len(observations), len(modalities)), -np.inf, dtype=np.float64)
    sigmas = {}

    for i1, mod1 in enumerate(modalities):
        observations1 = observations.intersection(mdata.mod[mod1].obs.index)
        ratioidx = np.where(observations.isin(observations1))[0]
        nparams1 = neighbors_params[mod1]
        X = reps[mod1]  # noqa: N806
        neighbordistances = mdata.mod[mod1].obsp[nparams1["distances_key"]]
        nndistances = np.empty((neighbordistances.shape[0],), neighbordistances.dtype)
        # neighborsdistances is a sparse matrix, we can either convert to dense, or loop
        for i in range(neighbordistances.shape[0]):
            nndist = neighbordistances[i, :].data
            if nndist.size == 0:
                msg = (
                    f"Cell {i} in modality {mod1} does not have any neighbors. "
                    "This could be due to subsetting after nearest neighbors calculation. "
                    "Make sure to subset before calculating nearest neighbors."
                )
                raise ValueError(msg)
            nndistances[i] = nndist.min()

        # We want to get the k-nn with the largest Jaccard distance, but break ties using
        # Euclidean distance. Largest Jaccard distance corresponds to lowest Jaccard index,
        # i.e. 1 - Jaccard distance. The naive version would be to compute pairwise Jaccard and
        # Euclidean distances for all points, but this is low and needs lots of memory. We
        # want to use an efficient k-nn algorithm, however no package that I know of supports
        # tie-breaking k-nn, so we use a custom distance. Jaccard index is always between 0 and 1
        # and has discrete increments of at least 1/N, where N is the number of data points.
        # If we scale the Jaccard indices by N, the minimum Jaccard index will be 1. If we scale
        # all Euclidean distances to be less than one, we can define a combined distance as the
        # sum of the scaled Jaccard index and one minus the Euclidean distances. This is not a
        # proper metric, but UMAP's nearest neighbor search uses NN-descent, which works with
        # arbitrary similarity measures.
        # The scaling factor for the Euclidean distance is given by the length of the diagonal
        # of the bounding box of the data. This can be computed in linear time by just taking
        # the minimal and maximal coordinates of each dimension.
        num_obs = X.shape[0]
        bbox_norm = np.linalg.norm(_sparse_csr_ptp(X) if issparse(X) else np.ptp(X, axis=0), ord=2)
        lmemory = low_memory if low_memory is not None else num_obs > LARGE_NUMBER_OF_OBSERVATIONS
        if issparse(X):
            X = X.tocsr()  # noqa: N806
            cmetric = _jaccard_sparse_euclidean_metric
            metric_kwds = {
                "X_indices": X.indices,
                "X_indptr": X.indptr,
                "X_data": X.data,
                "neighbors_indices": neighbordistances.indices,
                "neighbors_indptr": neighbordistances.indptr,
                "neighbors_data": neighbordistances.data,
                "N": num_obs,
                "bbox_norm": bbox_norm,
            }
        else:
            cmetric = _jaccard_euclidean_metric
            metric_kwds = {
                "X": X,
                "neighbors_indices": neighbordistances.indices,
                "neighbors_indptr": neighbordistances.indptr,
                "neighbors_data": neighbordistances.data,
                "N": num_obs,
                "bbox_norm": bbox_norm,
            }

        logging.info(f"Calculating kernel bandwidth for '{mod1}' modality...")
        nn_indices, _, _ = nearest_neighbors(
            np.arange(num_obs)[:, np.newaxis],
            n_neighbors=n_bandwidth_neighbors,
            metric=cmetric,
            metric_kwds=metric_kwds,
            random_state=randomstate,
            angular=False,
            low_memory=lmemory,
        )

        csigmas = np.empty((num_obs,), dtype=neighbordistances.dtype)
        if issparse(X):
            for i, neighbors in enumerate(nn_indices):
                csigmas[i] = cdist(X[i : (i + 1), :].toarray(), X[neighbors, :].toarray(), metric="euclidean").mean()
        else:
            for i, neighbors in enumerate(nn_indices):
                csigmas[i] = cdist(X[i : (i + 1), :], X[neighbors, :], metric="euclidean").mean()

        if not manual_weights:
            currtheta = None
            thetas = np.full((len(observations1), len(modalities) - 1), -np.inf, dtype=neighbordistances.dtype)

            lasti = 0

            logging.info(f"Calculating cell affinities for '{mod1} modality...")
            for i2, mod2 in enumerate(modalities):
                nparams2 = neighbors_params[mod2]
                neighbordistances = mdata.mod[mod2].obsp[nparams2["distances_key"]]
                observations2 = observations1.intersection(mdata.mod[mod2].obs.index)
                Xidx = np.where(observations1.isin(observations2))[0]  # noqa: N806
                r = np.empty(shape=(len(observations2), X.shape[1]), dtype=neighbordistances.dtype)
                # alternative to the loop would be broadcasting, but this requires converting the sparse
                # connectivity matrix to a dense ndarray and producing a temporary 3d array of size
                # n_cells x n_cells x n_genes => requires a lot of memory
                for i, cell in enumerate(Xidx):
                    r[i, :] = np.asarray(np.mean(X[neighbordistances[cell, :].nonzero()[1], :], axis=0)).squeeze()

                theta = np.exp(
                    -np.maximum(np.linalg.norm(X[Xidx, :] - r, ord=2, axis=-1) - nndistances[Xidx], 0)
                    / (csigmas[Xidx] - nndistances[Xidx])
                )
                if i1 == i2:
                    currtheta = theta
                else:
                    thetas[:, lasti] = theta
                    lasti += 1
            ratios[ratioidx, i1] = currtheta / (np.max(thetas, axis=1) + eps)
        sigmas[mod1] = csigmas

    if manual_weights:
        for mod in modalities:
            if mod not in manual_weights:
                msg = f"A weight was not suppied for the {mod} modality"
                raise ValueError(msg)
        weights = np.empty((len(mdata.obs), len(modalities)))
        for i, mod in enumerate(manual_weights):
            if isinstance(manual_weights[mod], numbers.Number):
                manual_weights[mod] = np.broadcast_to(manual_weights[mod], len(mdata[mod].obs))
            if len(manual_weights[mod]) != len(mdata[mod].obs):
                msg = f"The length of the weight array for {mod} does not match the number of cells. Either suppy a single value or one for each cell"
                raise ValueError(msg)
            weights[:, i] = manual_weights[mod]
    else:
        weights = softmax(ratios, axis=1)

    neighbordistances = csr_matrix((mdata.n_obs, mdata.n_obs), dtype=np.float64)
    largeidx = mdata.n_obs**2 > np.iinfo(np.int32).max
    if largeidx:  # work around scipy bug https://github.com/scipy/scipy/issues/13155
        neighbordistances.indptr = neighbordistances.indptr.astype(np.int64)
        neighbordistances.indices = neighbordistances.indices.astype(np.int64)
    for _, m in enumerate(modalities):
        cmetric = neighbors_params[m].get("metric", "euclidean")
        observations1 = observations.intersection(mdata.mod[m].obs.index)

        rep = reps[m]
        lmemory = low_memory if low_memory is not None else rep.shape[0] > LARGE_NUMBER_OF_OBSERVATIONS
        logger.info(f"Calculating nearest neighbor candidates for '{m}' modality...")
        logger.debug(f"Using low_memory={lmemory} for '{m}' modality")
        nn_indices, distances, _ = nearest_neighbors(
            rep,
            n_neighbors=n_multineighbors + 1,
            metric=cmetric,
            metric_kwds={},
            random_state=randomstate,
            angular=False,
            low_memory=lmemory,
        )
        graph = csr_matrix(
            (
                distances[:, 1:].reshape(-1),
                nn_indices[:, 1:].reshape(-1),
                np.concatenate((nn_indices[:, 0] * n_multineighbors, (nn_indices[:, 1:].size,))),
            ),
            shape=(rep.shape[0], rep.shape[0]),
        )
        with warnings.catch_warnings():
            # CSR is faster here than LIL, no matter what SciPy says
            warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
            if observations1.size == observations.size:
                if neighbordistances.size == 0:
                    neighbordistances = graph
                else:
                    neighbordistances += graph
            # the naive version of neighbordistances[idx[:, np.newaxis], idx[np.newaxis, :]] += graph
            else:
                # uses way too much memory
                if largeidx:
                    graph.indptr = graph.indptr.astype(np.int64)
                    graph.indices = graph.indices.astype(np.int64)
                fullstarts, fullstops = _make_slice_intervals(
                    np.where(observations.isin(observations1))[0], sparse_matrix_assign_splits
                )
                modstarts, modstops = _make_slice_intervals(
                    np.where(mdata.mod[m].obs.index.isin(observations1))[0],
                    sparse_matrix_assign_splits,
                )

                for fullidxstart1, fullidxstop1, modidxstart1, modidxstop1 in zip(
                    fullstarts, fullstops, modstarts, modstops, strict=False
                ):
                    for fullidxstart2, fullidxstop2, modidxstart2, modidxstop2 in zip(
                        fullstarts, fullstops, modstarts, modstops, strict=False
                    ):
                        neighbordistances[fullidxstart1:fullidxstop1, fullidxstart2:fullidxstop2] += graph[
                            modidxstart1:modidxstop1, modidxstart2:modidxstop2
                        ]

    neighbordistances.data[:] = 0
    logging.info("Calculating multimodal nearest neighbors...")
    if modality_weights is None:
        modality_weights = {_: 1 for _ in modalities}
    if len(modality_weights) != len(modalities):
        msg = "Number of weights in modality_weights does not match the actual number of modalities!"
        raise ValueError(msg)
    for i, m in enumerate(modalities):
        observations1 = observations.intersection(mdata.mod[m].obs.index)
        fullidx = np.where(observations.isin(observations1))[0]

        if weight_key:
            if add_weights_to_modalities:
                mdata.mod[m].obs[weight_key] = weights[fullidx, i] * modality_weights[m]
            else:
                # mod_weight -> mod:mod_weight
                mdata.obs[":".join([m, weight_key])] = weights[fullidx, i] * modality_weights[m]

        rep = reps[m]
        csigmas = sigmas[m]

        for cell, _ in enumerate(fullidx):
            row = slice(neighbordistances.indptr[cell], neighbordistances.indptr[cell + 1])
            nz = neighbordistances.indices[row]
            neighbordistances.data[row] += (
                np.exp(neighdist(rep, cell, nz, metric) / csigmas[cell]).squeeze()
                * weights[cell, i]
                * modality_weights[m]
            )
    neighbordistances.data = np.sqrt(0.5 * (1 - neighbordistances.data))

    neighbordistances = _sparse_csr_fast_knn(neighbordistances, n_neighbors + 1)

    logging.info("Calculating connectivities...")
    _, connectivities = _compute_connectivities_umap(
        knn_indices=neighbordistances.indices.reshape((neighbordistances.shape[0], n_neighbors + 1)),
        knn_dists=neighbordistances.data.reshape((neighbordistances.shape[0], n_neighbors + 1)),
        n_obs=neighbordistances.shape[0],
        n_neighbors=n_neighbors + 1,
    )

    if key_added is None:
        key_added = "neighbors"
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = key_added + "_connectivities"
        dists_key = key_added + "_distances"
    neighbors_dict = {"connectivities_key": conns_key, "distances_key": dists_key}
    neighbors_dict["params"] = {
        "n_neighbors": n_neighbors,
        "n_multineighbors": n_multineighbors,
        "metric": metric,
        "eps": eps,
        "random_state": random_state,
        "use_rep": mod_reps,
        "n_pcs": mod_n_pcs,
        "method": "umap",
    }
    mdata.obsp[dists_key] = neighbordistances
    mdata.obsp[conns_key] = connectivities
    mdata.uns[key_added] = neighbors_dict

    mdata.update_obs()

    return mdata if copy else None


def neighdist(rep, cell, nz, metric):
    if issparse(rep):
        return -cdist(rep[cell, :].toarray(), rep[nz, :].toarray(), metric=metric)
    else:
        return -cdist(rep[np.newaxis, cell, :], rep[nz, :], metric=metric)
