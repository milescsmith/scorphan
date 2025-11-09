import anndata as ad
import muon as mu
from loguru import logger

try:
    import cupy as cu
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import svds
except ImportError as exc:
    msg = "Unable to import CuPy. Please make sure it is installed."
    raise ImportError(msg) from exc


def cu_lsi(data: ad.AnnData | mu.MuData, scale_embeddings: bool = True, n_comps: int = 50) -> None:
    """
    Run Latent Semantic Indexing using CuPy

    PARAMETERS
    ----------
    data:
            AnnData object or MuData object with 'atac' modality
    scale_embeddings: bool [default: True]
            Scale embeddings to zero mean and unit variance
    n_comps: int [default: 50]
            Number of components to calculate with SVD
    """
    if isinstance(data, ad.AnnData):
        adata = data
    elif isinstance(data, mu.MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        msg = "Expected AnnData or MuData object with 'atac' modality"
        raise TypeError(msg)

    # In an unlikely scnenario when there are less 50 features, set n_comps to that value
    n_comps = int(cu.minimum(n_comps, adata.X.shape[1]))

    logger.info("Performing SVD")
    cu_x = csr_matrix(adata.X)
    cell_embeddings, svalues, peaks_loadings = svds(cu_x, k=n_comps)

    # Re-order components in the descending order
    cell_embeddings = cell_embeddings[:, ::-1]
    svalues = svalues[::-1]
    peaks_loadings = peaks_loadings[::-1, :]

    if scale_embeddings:
        cell_embeddings = cu.divide(
            (cu.subtract(cell_embeddings, cell_embeddings.mean(axis=0))),
            cell_embeddings.std(axis=0),
        )

    stdev = cu.divide(svalues, cu.sqrt(adata.X.shape[0] - 1))

    adata.obsm["X_lsi"] = cell_embeddings.get()
    adata.uns["lsi"] = {"stdev": stdev.get()}
    adata.varm["LSI"] = peaks_loadings.T.get()
