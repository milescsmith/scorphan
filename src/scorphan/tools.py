from copy import deepcopy
from typing import Literal

import numpy as np
import scanpy as sc
from anndata import AnnData
from scanpy.tools._utils import _choose_representation
from scipy.sparse import issparse


def muon_paga_umap(
    mdata: AnnData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Literal["spectral", "paga", "random"] = "paga",
    random_state: int = 42,
    a: float | None = None,
    b: float | None = None,
    copy: bool = False,
    method: Literal["umap", "rapids"] = "umap",
    neighbors_key: str = "neighbors",
) -> None:
    neighbors = mdata.uns[neighbors_key]
    reps = {}
    nfeatures = 0
    nparams = neighbors["params"]
    use_rep = {k: (v if v != -1 else None) for k, v in nparams["use_rep"].items()}
    n_pcs = {k: (v if v != -1 else None) for k, v in nparams["n_pcs"].items()}
    observations = mdata.obs.index

    for mod, rep in use_rep.items():
        nfeatures += rep.shape[1]
        reps[mod] = _choose_representation(mdata.mod[mod], rep, n_pcs[mod])

    rep = np.empty((len(observations), nfeatures), np.float32)
    nfeatures = 0

    for mod, crep in reps.items():
        cnfeatures = nfeatures + crep.shape[1]
        idx = observations.isin(mdata.mod[mod].obs.index)
        rep[idx, nfeatures:cnfeatures] = crep.toarray() if issparse(crep) else crep
        if np.sum(idx) < rep.shape[0]:
            imputed = crep.mean(axis=0)
            if issparse(crep):
                imputed = np.asarray(imputed).squeeze()
            rep[~idx, nfeatures : crep.shape[1]] = imputed
        nfeatures = cnfeatures

    adata = AnnData(X=rep, obs=mdata.obs)

    adata.uns[neighbors_key] = deepcopy(neighbors)
    adata.uns[neighbors_key]["params"]["use_rep"] = "X"
    del adata.uns[neighbors_key]["params"]["n_pcs"]
    adata.obsp[neighbors["connectivities_key"]] = mdata.obsp[neighbors["connectivities_key"]]
    adata.obsp[neighbors["distances_key"]] = mdata.obsp[neighbors["distances_key"]]

    adata.uns["paga"] = mdata.uns["paga"].copy()

    sc.tl.umap(
        adata=adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        maxiter=maxiter,
        alpha=alpha,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        init_pos=init_pos,
        random_state=random_state,
        a=a,
        b=b,
        copy=False,
        method=method,
        neighbors_key=neighbors_key,
    )

    mdata.obsm["X_umap"] = adata.obsm["X_umap"]
    mdata.uns["umap"] = adata.uns["umap"]
