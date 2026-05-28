# fuck pyright is so goddamned stupid about some things

# TODO: either break up this module or use that delayed import module to make this less painful
# pulling in torch is killing import speed
import re
import warnings
from collections.abc import Iterable, Sequence
from copy import deepcopy
from multiprocessing import cpu_count

# from copy import deepcopy
from pathlib import Path
from typing import Final, Literal

import anndata as ad
import decoupler as dc
import gseapy as gp
import h5py
import matplotlib.pyplot as plt

# import msgspec
import muon as mu
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import scanpy as sc
import scipy as sp

# import scvi
# import torch
from anndata import AnnData
from formulaic.errors import FormulaicError
from formulaic.parser import DefaultFormulaParser
from formulaic_contrasts import FormulaicContrasts
from loguru import logger
from matplotlib import colormaps
from mudata import MuData
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from rich.progress import Progress
from scanpy.tools._utils import _choose_representation

# from scanpy.tools._utils import _choose_representation
from scipy.sparse import issparse

from scorphan._utils import ArgumentError, is_integer_array, make_list_if_not, not_yet_implemented
from scorphan.log import init_logger

MAX_PVAL: Final[float] = 0.05


@not_yet_implemented(reason="Function needs to be adapted to work with new versions of scanpy/anndata/muon")
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
    """_summary_

    Parameters
    ----------
    mdata : AnnData
        _description_
    min_dist : float, optional
        _description_, by default 0.5
    spread : float, optional
        _description_, by default 1.0
    n_components : int, optional
        _description_, by default 2
    maxiter : int | None, optional
        _description_, by default None
    alpha : float, optional
        _description_, by default 1.0
    gamma : float, optional
        _description_, by default 1.0
    negative_sample_rate : int, optional
        _description_, by default 5
    init_pos : Literal[&quot;spectral&quot;, &quot;paga&quot;, &quot;random&quot;], optional
        _description_, by default "paga"
    random_state : int, optional
        _description_, by default 42
    a : float | None, optional
        _description_, by default None
    b : float | None, optional
        _description_, by default None
    copy : bool, optional
        _description_, by default False
    method : Literal[&quot;umap&quot;, &quot;rapids&quot;], optional
        _description_, by default "umap"
    neighbors_key : str, optional
        _description_, by default "neighbors"

    Raises
    ------
    NotImplementedError
        _description_
    """
    # msg = "This function does not currently work and is disabled for the time being."
    # logger.error(msg)
    # raise NotImplementedError(msg)
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


def GSEApy_process(
    adata: AnnData,
    groupby: str | Sequence[str],
    comparison_group: str | None = None,
    reference_group: str | None = None,
    obs_subset: pd.Series | pd.Index | list[str] | None = None,
    var_subset: pd.Series | pd.Index | list[str] | None = None,
    top_x_pathways: int = 5,
    top_pathway_type: Literal["heatmap", "dotplot"] = "dotplot",
    geneset: str | list[str] | dict[str, list[str]] | Path | None = None,  # pyright: ignore[reportRedeclaration]
    outdir_path: Path | None = None,
    verbose: bool = False,
):
    r"""_summary_

    Parameters
    ----------
    adata : :class:`AnnData
        Object containing scRNAseq data to perform analysis on
    groupby : :class:`str`, optional
        Column in adata.obs to use for grouping cells, by default None
    comparison_group : :class:`str`, optional
        Factor in adata.obs[groupby] to generate the analysis for. For example, the disease group of interest., by default None
    reference_group : :class:`str`, optional
        Factor in adata.obs[groupby] to set as the basis of comparison, such as the control group, by default None
    obs_subset : :class:`pd.Series` | :class:`pd.Index` | :class:`list[str]`, optional
        Names from adata.obs_names (i.e. cell barcodes) to use to subset the data, by default None
    var_subset : :class:`pd.Series` | :class:`pd.Index` | :class:`list[str]`, optional
        Names from adata.var_names (i.e. gene or protein names) to use to subset the data, by default None
    top_x_pathways : :class:`int`, optional
        Number of pathways to generate plots for, by default 5
    top_pathway_type : :class:`Literal["heatmap", "dotplot"], optional
        Type of plots to generate, by default "dotplot"
    geneset : :class:`str` | :class:`list[str]` | :class:`dict[str, list[str]]` | :class:`Path`, optional
        Gene set list to test for. Can either be the name of a pathway or the path to a gene matrix transposed (*.gmt)
        file, by default None, which results in using "GO_Biological_Process_2023", "GO_Cellular_Component_2023", and
        "GO_Molecular_Function_2023"
    outdir_path : :class:`Path`, optional
        Location to write output plots and spreadsheets to, by default None
    verbose : :class:`bool`, optional
        More or less feedback, by default False

    """
    if verbose:
        init_logger(3)
    for _ in ["groupby", "reference_group", "comparison_group"]:
        if _ is None:
            msg = f"A value for {_} was not given"
            raise SyntaxError(msg)
    if isinstance(geneset, Path):
        geneset: dict[str, list[str]] = gp.parser.read_gmt(str(geneset))  # pyright: ignore[reportRedeclaration, reportAssignmentType]
    elif geneset is None:
        geneset: list[str] = ["GO_Biological_Process_2023", "GO_Cellular_Component_2023", "GO_Molecular_Function_2023"]  # pyright: ignore[reportRedeclaration]
        logger.info("No geneset was specified. Using the three GeneOntology groups")
    else:
        geneset: list[str] = geneset  # noqa: PLW0127  # pyright: ignore[reportAssignmentType]

    # TODO: no need to repeat normalization/log-transformation. Wrap the next few lines in something like a
    # prep_adata() function
    outdir_path = Path().cwd().absolute() if outdir_path is None else outdir_path.absolute()

    obs_subset = adata.obs_names if obs_subset is None else obs_subset
    var_subset = adata.var_names if var_subset is None else var_subset

    if (len(obs_subset) < adata.shape[0]) or (len(var_subset) < adata.shape[1]):
        logger.info("subsetting anndata obj")
        adata = adata[obs_subset, var_subset].copy()

    if not is_integer_array(adata.X):
        if "counts" in adata.layers and is_integer_array(adata.layers["counts"]):
            adata.layers["lognorm"] = adata.X.copy()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            adata.X = adata.layers["counts"].copy()

            logger.info("normalizing and log transforming data")
            sc.pp.normalize_total(adata, target_sum=1e6)
            sc.pp.log1p(adata)
        else:
            msg = "This function requires starting with untransformed, integer counts."
            raise ValueError(msg)

    adata.layers["lognorm"] = adata.X.copy()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

    ####GSEA
    if issparse(adata.layers["counts"]):
        logger.info("inflating sparse counts")
        counts_df = adata.layers["counts"].toarray().transpose()  # pyright: ignore[reportAttributeAccessIssue]
    else:
        counts_df = adata.layers["counts"].transpose()
    logger.info("Performing GSEA")
    gs = gp.GSEA(
        data=pd.DataFrame(counts_df, index=var_subset, columns=obs_subset),  # row -> genes, column-> samples
        gene_sets=geneset,
        classes=adata.obs.loc[obs_subset, groupby].tolist(),  # pyright: ignore[reportAttributeAccessIssue]
        permutation_num=1000,
        permutation_type="phenotype",
        outdir=str(outdir_path.joinpath("GSEA_results")),
        method="s2n",  # signal_to_noise
        threads=16,
        verbose=True,
    )
    gs.pheno_pos = comparison_group  # pyright: ignore[reportAttributeAccessIssue]
    gs.pheno_neg = reference_group
    gs.run()

    # both sc.pl.heatmap and sc.pl.dotplot let you group variables *IF* you supply the var_names as a dict[str, list[str]]
    # this converts the "Term" and "Lead_genes" columns into a dict for the number of top pathways specified
    term_gene_dict = {
        re.sub(pattern=r"\s\(GO:[0-9]+\)", repl="", string=x[1]["Term"]): x[1]["Lead_genes"].split(";")
        for i, x in enumerate(gs.res2d.iterrows())  # pyright: ignore[reportOptionalMemberAccess]
        if i < top_x_pathways
    }

    with Progress() as progress:
        task = progress.add_task(description="Creating GSEA pathway plots for", total=top_x_pathways)
        progress.start()
        for i, _ in term_gene_dict.items():
            # term_name = re.sub(pattern=r"\s\(GO:[0-9]+\)", repl="", string=gs.res2d.Term.iloc[i])
            progress.update(task, description=f"Creating GSEA pathway plots for {i}")
            logger.info(f"generating plots for {i}")
            _, ax = plt.subplots(figsize=(9, 5))
            match top_pathway_type:
                case "heatmap":
                    sc.pl.heatmap(
                        adata=adata,
                        var_names=term_gene_dict[i],
                        standard_scale="var",
                        groupby=groupby,
                        # figsize=(9,5), # I don't like this. Replace with user-defined? Return values packaged so that one could rapidly recreate?
                        show=False,
                        ax=ax,
                        # save=str(outdir_path.joinpath(f"{i}_Heatmap.png")) # This DOES NOT WORK! WTF? It just reinterprets the path in some weird way. gonna have to make a pull request to scanpy
                    )
                case "dotplot":
                    sc.pl.dotplot(
                        adata=adata,
                        var_names=term_gene_dict[i],
                        standard_scale="var",
                        groupby=groupby,
                        # figsize=(9,5),
                        show=False,
                        # save=str(outdir_path.joinpath(f"{i}_Heatmap.png"))
                        # show_gene_labels=True,
                        ax=ax,  # pyright: ignore[reportArgumentType]
                    )
            ax.set_title(i)
            ax.figure.savefig(Path(outdir_path).absolute().joinpath("GSEA_results", f"{i}_Heatmap.png"))  # pyright: ignore[reportAttributeAccessIssue]
            progress.update(task, advance=1)
        term = gs.res2d.Term  # pyright: ignore[reportOptionalMemberAccess]
        # gp.gseaplot(res.ranking, term=term[i], **res.results[term[i]])
        gs.plot(terms=term[:top_x_pathways], ofname=str(outdir_path.joinpath("GSEA_results", "Top_GSEA_Terms.png")))

    #### DEG analysis
    if ("rank_genes_groups" not in adata.uns) or (adata.uns["rank_genes_groups"]["params"]["groupby"] != groupby):
        logger.info("Performing DEG analysis")
        sc.tl.rank_genes_groups(adata, groupby=groupby, reference=reference_group, layer="lognorm", use_raw=False)
    # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
    degs = sc.get.rank_genes_groups_df(adata, group=comparison_group)
    if np.all(degs["logfoldchanges"].isnull()):  # should this maybe be changed to np.any?
        msg = "There is something wrong with the data. All log fold changes are reported as NAs"
        logger.error(msg)
    else:
        degs_sig = degs[degs["pvals_adj"] < MAX_PVAL]
        degs_up = degs_sig[degs_sig["logfoldchanges"] > 0]
        degs_down = degs_sig[degs_sig["logfoldchanges"] < 0]

    #### Enrichment
    if degs_up.shape[0] > 0:
        logger.info(f"Performing Enrichr analysis of upregulated in {comparison_group} genes")
        enr_up = gp.enrichr(degs_up["names"], gene_sets=geneset, outdir=str(outdir_path.joinpath("Enrichr_results")))
        enr_up.res2d.Term = enr_up.res2d.Term.str.replace(pat=r"\s\(GO:[0-9]+\)", repl="", regex=True)
        try:
            gp.dotplot(enr_up.res2d, figsize=(3, 5), title="Up", cmap=plt.cm.autumn_r)
            fig1 = plt.gcf()
            fig1.savefig(str(outdir_path.joinpath("Enrichr_results", "ORA_UP.png")), bbox_inches="tight")
        except ValueError:
            logger.error("No enrich terms at current cutoff")
    else:
        logger.info(f"No significant upregulated in {comparison_group} genes were found")
        enr_up = gp.Enrichr(gene_list=[], gene_sets=geneset)  # make an empty object to prevent errors downstream

    if degs_up.shape[0] > 0:
        logger.info(f"Performing Enrichr analysis of downregulated in {comparison_group} genes")
        enr_down = gp.enrichr(
            degs_down["names"], gene_sets=geneset, outdir=str(outdir_path.joinpath("Enrichr_results"))
        )
        enr_down.res2d.Term = enr_down.res2d.Term.str.replace(pat=r"\s\(GO:[0-9]+\)", repl="", regex=True)
        try:
            gp.dotplot(
                enr_down.res2d,
                figsize=(3, 5),
                title="Down",
                cmap=plt.cm.winter_r,
                size=5,
            )
            fig1 = plt.gcf()
            fig1.savefig(str(outdir_path.joinpath("Enrichr_results", "ORA_DOWN.png")), bbox_inches="tight")
        except ValueError:
            logger.error("No enrich terms at current cutoff")
    else:
        logger.info(f"No significant downregulated in {comparison_group} genes were found")
        enr_down = gp.Enrichr(gene_list=[], gene_sets=geneset)  # make an empty object to prevent errors downstream

    enr_up.res2d["UP_DW"] = "UP"
    enr_down.res2d["UP_DW"] = "DOWN"
    enr_res = pd.concat([enr_up.res2d.head(), enr_down.res2d.head()])

    try:
        ax = gp.dotplot(
            enr_res,
            figsize=(3, 5),
            x="UP_DW",
            x_order=["UP", "DOWN"],
            title="GO_BP",
            cmap=colormaps["viridis"].reversed(),
            size=3,
            show_ring=True,
        )
        ax.set_xlabel("")
        fig1 = plt.gcf()
        fig1.savefig(str(outdir_path.joinpath("Enrichr_results", "ORA_Dotplot.png")), bbox_inches="tight")
    except ValueError:
        logger.error("No enrich terms at current cutoff")
    try:
        ax = gp.barplot(
            enr_res,
            figsize=(3, 5),
            group="UP_DW",
            title="GO_BP",
            color=["b", "r"],
            ofname=str(outdir_path.joinpath("Enrichr_results", "Enriched_Barplot.png")),
        )
    except ValueError:
        logger.error("No enrich terms at current cutoff")

    ##### Network Plot
    nodes, edges = gp.enrichment_map(gs.res2d)
    graph = nx.from_pandas_edgelist(
        edges, source="src_idx", target="targ_idx", edge_attr=["jaccard_coef", "overlap_coef", "overlap_genes"]
    )

    # Add missing node if there is any
    for node in nodes.index:
        if node not in graph.nodes():
            graph.add_node(node)
    _, ax = plt.subplots(figsize=(12, 12))

    # init node cooridnates
    pos = nx.layout.spiral_layout(graph)
    # node_size = nx.get_node_attributes()
    # draw node
    nx.draw_networkx_nodes(
        graph, pos=pos, cmap=plt.cm.RdYlBu, node_color=list(nodes.NES), node_size=list(nodes.Hits_ratio * 1000), ax=ax
    )
    # draw node label
    nx.draw_networkx_labels(graph, pos=pos, labels=nodes.Term.to_dict(), clip_on=False, ax=ax)
    # draw edge
    edge_weight = nx.get_edge_attributes(graph, "jaccard_coef").values()
    nx.draw_networkx_edges(graph, pos=pos, width=[x * 10 for x in edge_weight], edge_color="#CDDBD4", ax=ax)
    fig1 = plt.gcf()
    fig1.savefig(str(outdir_path.joinpath("GSEA_results", "Network.png")), bbox_inches="tight")
    nodes.to_csv(str(outdir_path.joinpath("GSEA_results", "Nodes.csv")))
    edges.to_csv(str(outdir_path.joinpath("GSEA_results", "Edges.csv")))


# TODO: why is this here?
def umap(
    mdata: MuData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Literal["spectral", "random"] | npt.ArrayLike | None = "spectral",
    random_state: int | np.random.RandomState | None = 42,
    a: float | None = None,
    b: float | None = None,
    copy: bool = False,
    method: Literal["umap", "rapids"] = "umap",
    neighbors_key: str | None = None,
) -> MuData | AnnData | None:
    r"""
    Embed the multimodal neighborhood graph using UMAP (McInnes et al, 2018).

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. We use ScanPy's
    implementation.

    References:
        McInnes et al, 2018 (`arXiv:1802.03426` <https://arxiv.org/abs/1802.03426>`_)

    Parameters
    ----------
    mdata: mu.MuData
        Multimodal nearest neighbor search must have already been performed.
    min_dist : float, default=5
        The effective minimum distance between embedded points. Smaller
        values will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded points
        will be spread out.
    spread : float, default=1.0
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.
    n_components : int, default=2
        The number of dimensions of the embedding
    maxiter : int | None, default=None
        The number of iterations (epochs) of the optimization. Called ``n_epochs`` in the original UMAP.
    alpha : float, default=1.0
        The initial learning rate for the embedding optimization.
    gamma : float, default=1.0
        Weighting applied to negative samples in low dimensional embedding optimization. Values higher than one will
        result in greater weight being given to negative samples.
    negative_sample_rate : int, default=5
        The number of negative edge/1-simplex samples to use per
        positive edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos : Literal["spectral", "random"] | npt.ArrayLike | None, default="spectral"
        How to initialize the low dimensional embedding. Called ``init`` in the original UMAP. Options are:
        - 'spectral': use a spectral embedding of the graph.
        - 'random': assign initial embedding positions at random.
        - A numpy array of initial embedding positions.
    random_state :
        Random seed.
    a :
        More specific parameters controlling the embedding. If ``None`` these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b :
        More specific parameters controlling the embedding. If ``None`` these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    copy :
        Return a copy instead of writing to mdata.
    method :
        Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
    neighbors_key :
        If not specified, umap looks in ``.uns['neighbors']`` for neighbors
        settings and ``.obsp['connectivities']`` for connectivities (default storage
        places for ``pp.neighbors``). If specified, umap looks ``.uns[neighbors_key]``
        for neighbors settings and ``.obsp[.uns[neighbors_key]['connectivities_key']]``
            for connectivities.

    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
        **X_umap** : ``mdata.obsm`` field holding UMAP coordinates of data.
    """
    if method == "rapids":
        try:
            import rapids_singlecell as usc
        except ImportError as e:
            msg = "the rapids_singlecell library is required for the 'rapids' method, but it was not able to be imported. Please install the correct version."
            raise ImportError(msg) from e
    else:
        import scanpy as usc

    if isinstance(mdata, AnnData):
        return usc.tl.umap(
            adata=mdata,
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
            copy=copy,
            neighbors_key=neighbors_key,
        )

    if neighbors_key is None:
        neighbors_key = "neighbors"

    try:
        neighbors = mdata.uns[neighbors_key]
    except KeyError as e:
        msg = f'Did not find .uns["{neighbors_key}"]. Run `muon.pp.neighbors` first.'
        raise ValueError(msg) from e

    from copy import deepcopy

    from scanpy.tools._utils import _choose_representation

    # we need a data matrix. This is used only for initialization and only if init_pos=="spectral"
    # and the graph has many connected components, so we can do very simple imputation
    reps = {}
    nfeatures = 0
    nparams = neighbors["params"]
    use_rep = {k: (v if v != -1 else None) for k, v in nparams["use_rep"].items()}
    n_pcs = {k: (v if v != -1 else None) for k, v in nparams["n_pcs"].items()}
    observations = mdata.obs.index
    for mod, rep in use_rep.items():
        _rep = _choose_representation(adata=mdata.mod[mod], use_rep=rep, n_pcs=n_pcs[mod])
        nfeatures += _rep.shape[1]
        reps[mod] = _rep
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

    usc.tl.umap(
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

    mdata = mdata.copy() if copy else mdata
    mdata.obsm["X_umap"] = adata.obsm["X_umap"]
    mdata.uns["umap"] = adata.uns["umap"]
    return mdata if copy else None


def extract_h5_obs(h5_file: Path) -> pd.DataFrame:
    """Read the obs attribute directly from the on-disk AnnData or MuData object
    without reading it into memory.

    Parameters
    ---------
    h5_file : Path
        A Path object pointing to the AnnData/MuData file with the obs attribute of interest

    Returns
    -------

    pandas.DataFrame
    """
    with h5py.File(h5_file) as f:
        obs_h5_dict = dict(f["obs"].items())
        index_col = "_index" if "_index" in obs_h5_dict.keys() else "index"
        index = obs_h5_dict[index_col][:]
        obs_df = pd.DataFrame(index=pd.Index(index).str.decode("utf-8"))

        for k, v in obs_h5_dict.items():
            if isinstance(v, h5py.Group):
                # need to handle an instance where we
                decoded = {j: i.decode() for j, i in enumerate(v["categories"][:])}
                try:
                    obs_df.insert(
                        loc=obs_df.shape[1], column=k, value=[decoded[_] if _ != 1 else np.nan for _ in v["codes"][:]]
                    )
                except KeyError as err:
                    msg = f"{v=}"
                    raise KeyError(msg) from err
            else:
                obs_df.insert(loc=obs_df.shape[1], column=k, value=v[:])
        obs_df.drop(columns=index_col, inplace=True)
    return obs_df


def pseudobulk_differential_expression(
    adata: ad.AnnData,
    compare_col: str,
    design: str,
    comparisons: dict[str, str] | Sequence[dict[str, str]],
    cell_type_of_interest: str | None = None,
    cell_type_obs_col: str | None = None,
    sample_name_col: str = "sample_name",
    min_sum_counts: float = 1.0,
    min_std: float = 0.1,
    layer: str | None = None,
    n_cpus: int = -1,
    output_dir: Path | None = None,
) -> dict[str, DeseqStats]:
    """
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to examine
    compare_col : str
        Column in `adata.obs` to use when grouping cells for comparison
    design : str
        Design formula to use for analysis. Like an R-style formula. For example: "~disease"
    comparisons : dict[str, str], Sequence[dict[str, str]]
        What groups to compare and how to compare them. This MUST be a dictionary in the form of
        `{"ref": "controls", "test": "test_group"}`
    cell_type_of_interest : str, optional
        A value in `cell_type_obs_col` to subset `adata` by
    cell_type_obs_col : str, optional
        A column in `adata.obs` to look for values by which to subset `adata`
    sample_name_col : str, default = "sample_name"
        Name of column in `adata.obs` to be used to group cells when pseudobulking the sample
    min_sum_counts : float, default = 1.0
        Count threshold above which genes much be for their inclusion in analysis.
        Genes with a count below this will be removed.
    min_std : float, default = 0.1
        Standard deviation threshold above which genes much be for their inclusion in analysis.
        Genes with a stdev below this will be removed.
    layer : str, optional
        Layer to use for the counts
    n_cpus : int, default = -1 (i.e. use all available cpus)
        Number of CPUs to use when running pydeseq2 functions.
    output_dir : Path, optional
        Path to where DEG results should be saved.

    Returns
    -------
    A dictionary with the comparison name as keys in the form of "ref_vs_test" and the DeseqStats object as value

    Example
    -------
    >>> res = pseudobulk_deg_process(
            adata=bcell_rna,
            design=~disease+age,
            comparisons=[
                {"ref": "Neg", "test": "Pos"},
                {"ref": "Neg", "test": "ILE"},
                {"ref": "Neg", "test": "SLE"},
                {"ref": "Pos", "test": "ILE"},
                {"ref": "Pos", "test": "SLE"},
                {"ref": "ILE", "test": "SLE"}
            ],
            cell_type_of_interest="Memory",
            cell_type_obs_col="type_label",
            layer="counts"
        )
    """
    # TODO: handle errors and a lot of checking and messages!

    if n_cpus == -1:
        n_cpus = cpu_count()

    _check_formula(adata.obs, design)

    if cell_type_of_interest:
        if cell_type_obs_col is None:
            msg = f"{cell_type_of_interest} was passed as the 'cell_type_of_interest' argument, but no 'cell_type_obs_col' was given!"
            raise KeyError(msg)
        if any(adata.obs[cell_type_obs_col] == cell_type_of_interest):
            subset: ad.AnnData = adata[adata.obs[cell_type_obs_col] == cell_type_of_interest, :]
        else:
            msg = f"There do not appear to be any cells where the value of '{cell_type_obs_col}' is '{cell_type_of_interest}'"
            raise ValueError(msg)
    else:
        subset = adata

    subset_bulked: ad.AnnData = dc.pp.pseudobulk(adata=subset, sample_col=sample_name_col, groups_col=None, layer=layer)

    # I don't *think* decoupler can return a sparse pseudobulked object, but do I know for sure? No.
    if issparse(subset_bulked.X):
        msg = "PyDESeq2 doesn't work with sparse matrices. Densifying X"
        warnings.warn(msg, stacklevel=2)
        subset_bulked.X = subset_bulked.X.toarray()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    exprs: pd.DataFrame = pd.DataFrame(
        data=subset_bulked.X, index=subset_bulked.obs_names, columns=subset_bulked.var_names
    )

    # dumping genes with no detectable expression and no variance
    for stat, func, thresh in zip(
        ["below_zero_thresh", "below_deviance_thresh"], [np.sum, np.std], [min_sum_counts, min_std], strict=True
    ):
        subset_bulked.var[stat] = (
            func(exprs, axis=0).where(lambda x: x < thresh).replace({np.nan: False, 0.0: True})[subset_bulked.var_names]  # noqa: B023
        )
        mu.pp.filter_var(subset_bulked, var=stat, func=lambda x: ~x)

    if not _full_rank_design(subset_bulked, design):
        msg = (
            "One or more terms in the design formula is causing the design matrix to not be full rank. "
            "Attempting to correct an issue with `decoupler` changing continuous variables to categorical..."
        )
        warnings.warn(msg, stacklevel=2)
        try:
            subset_bulked.obs = _fix_continuous_vars(subset_bulked.obs, adata.obs, design)
        except FormulaicError as err:
            msg = "Something is wrong with the design formula. Stopping as PyDESeq2 WILL fail"
            raise FormulaicError(msg) from err

    inference = DefaultInference(n_cpus=cpu_count())

    dds = DeseqDataSet(
        adata=subset_bulked,
        design=design,
        refit_cooks=True,
        inference=inference,
    )

    dds.deseq2()

    res_dict = {}
    # start looping through comparisons
    if isinstance(comparisons, Sequence):
        for compare in comparisons:
            stat_res = DeseqStats(
                dds, contrast=[compare_col, compare["test"], compare["ref"]], inference=inference, n_cpus=n_cpus
            )

            stat_res.summary()
            # add results to res_dict
            res_dict[f"{compare['ref']}_vs_{compare['test']}"] = stat_res

            if output_dir:
                stat_res.results_df.to_csv(
                    output_dir.joinpath(
                        f"{cell_type_of_interest}_{cell_type_obs_col}_{compare['ref']}{compare['test']}.csv".replace(
                            " ", "_"
                        )
                    )
                )
    else:
        stat_res = DeseqStats(
            dds, contrast=[compare_col, comparisons["test"], comparisons["ref"]], inference=inference, n_cpus=n_cpus
        )

        stat_res.summary()
        # add results to res_dict
        res_dict[f"{comparisons['ref']}_vs_{comparisons['test']}"] = stat_res

        if output_dir:
            stat_res.results_df.to_csv(
                output_dir.joinpath(
                    f"{cell_type_of_interest}_{cell_type_obs_col}_{comparisons['ref']}{comparisons['test']}.csv".replace(
                        " ", "_"
                    )
                )
            )
    return res_dict


# based on the version in PyDESeq2. Copied and not imported because it is a private method
def _full_rank_design(adata: ad.AnnData, formula: str) -> bool:
    r"""Check that the design matrix has full column rank."""
    design_matrix = FormulaicContrasts(data=adata.obs, design=formula).design_matrix
    rank = np.linalg.matrix_rank(A=design_matrix)
    num_vars = design_matrix.shape[1]

    if rank < num_vars:
        return False
    else:
        return True


def _extract_terms(formula: str) -> list[str]:
    return [
        token.token
        for token in DefaultFormulaParser().get_tokens(formula)
        if token.kind.value == "name"  # pyright: ignore[reportOptionalMemberAccess]
    ]


def _fix_continuous_vars(obs, prev_obs, formula) -> pd.DataFrame:
    design_terms = _extract_terms(formula)
    for term in design_terms:
        if (obs[term].dtype == "category") & (obs[term].dtype != prev_obs[term].dtype):
            obs[term] = obs[term].astype(prev_obs[term].dtype)
    return obs


def _check_formula(obs, formula) -> None:
    design_terms = _extract_terms(formula)
    not_found = [term for term in design_terms if term not in obs.columns]
    if len(not_found) > 0:
        msg = f"The terms {', '.join(not_found)} are not present in the anndata object's obs columns"
        raise pd.errors.InvalidColumnName(msg)


@not_yet_implemented
def transfer_to_asap(
    adata: ad.AnnData,
    label_key: str,
    source_key: str,
    source_label: str = "RNA",
    target_label: str = "ASAP",
    batch_key: str | None = None,
    covars: Iterable[str] | str | None = None,
    hyperparameters: Path | None = None,
    layer: str | None = None,
):
    r"""Use scVI and scANVI to transfer labels from the protein modality of CITE-seq data
    to the protein modality of ASAP-seq data

    This requires an anndata object with a raw counts in either the `X` attribute or as a layer.

    Parameters
    ----------
    adata : AnnData
        Concatenated object with raw protein counts from both the reference RNAseq and ASAPseq modalities.
    label_key : str
        Column in `obs` that has the labels to be transferred from the RNAseq to the ASAPseq modality
    source_key : str
        Column in `obs` that indicates which assay a cell comes from
    ref_label : str [default: "RNA"]
        Value in `source_key` that indicates the cell is from the reference (RNAseq) data
    target_label : str [default: "ASAP"]
        Value in `source_key` that indicates the cell is from the query (ASAPseq) data
    batch_key : str, optional [default: None]
        Column in `obs` that indicates batch information.
    covars : Iterable[str] | str, optional [default: None]
        Column(s) in `obs` containing covariates to control for.
    hyperparameters : Path, optional [default: None]
        The path to the `results.json` output from `ray.tune.Tuner`/`scvi.autotune.ModelTuner`
    layer : str, optional [default: None]
        Layer containing raw integer counts if the `X` attribute does not.
    """
    pass
    # torch.set_float32_matmul_precision("high")

    # hyperparams = msgspec.json.decode(hyperparameters.read_text())

    # subrna = sc.pp.sample(adata[adata.obs[source_key] == "RNA", :], fraction=0.05, copy=True)
    # subatac = sc.pp.sample(
    #     adata[adata.obs[source_key] == "ASAP", :], fraction=0.05, copy=True
    # )
    # refdata = ad.concat([subrna, subatac])

    # scvi.model.SCVI.setup_anndata(
    #     refdata,
    #     batch_key="source",
    #     categorical_covariate_keys=covars,
    #     layer="counts",
    # )

    # refined_model = scvi.model.SCVI(
    #     adata=refdata,
    #     n_hidden=hyperparams["config"]["model_params"]["n_hidden"],
    #     n_layers=hyperparams["config"]["model_params"]["n_layers"],
    #     n_latent=hyperparams["config"]["model_params"]["n_latent"],
    #     dropout_rate=hyperparams["config"]["model_params"]["dropout_rate"],
    # )

    # refined_model.train(
    #     max_epochs=int(hyperparams["config"]["train_params"]["max_epochs"]),
    #     check_val_every_n_epoch=1,
    #     plan_kwargs={
    #         "lr": hyperparams["config"]["train_params"]["plan_kwargs"]["lr"],
    #         "weight_decay": hyperparams["config"]["train_params"]["plan_kwargs"][
    #             "weight_decay"
    #         ],
    #         "eps": hyperparams["config"]["train_params"]["plan_kwargs"]["eps"],
    #     },
    # )

    # scvi.model.SCVI.prepare_query_anndata(querydata, refined_model)
    # type_query = scvi.model.SCVI.load_query_data(
    #     querydata,
    #     refined_model,
    # )

    # type_model = scvi.model.SCANVI.from_scvi_model(
    #     type_query,
    #     adata=querydata,
    #     labels_key="labels",
    #     unlabeled_category="unknown",
    # )
    # type_model.train(
    #     max_epochs=int(hyperparams["config"]["train_params"]["max_epochs"]),
    #     check_val_every_n_epoch=1,
    #     plan_kwargs={
    #         "lr": hyperparams["config"]["train_params"]["plan_kwargs"]["lr"],
    #         "weight_decay": hyperparams["config"]["train_params"]["plan_kwargs"][
    #             "weight_decay"
    #         ],
    #         "eps": hyperparams["config"]["train_params"]["plan_kwargs"]["eps"],
    #     },
    #     adversarial_classifier=True,
    # )
    # querydata.obsm["X_scANVI_scVI"] = type_model.get_latent_representation()
    # querydata.obs["scanvi_scvi_predict"] = type_model.predict()


def pseudobulk_and_correlate(
    adata: ad.AnnData,
    olink: ad.AnnData,
    cell_type_col: str,
    cell_types: str | list[str],
    group_col: str,
    groups: str | list[str],
    sample_col: str,
    layer: str = "counts",
    filter_highly_variable: bool = True,
    normalize_bulk: bool = True,
    scale_bulk: bool = False,
    remove_nulls: bool = False,
    axis: int = 0,
    nan_policy: Literal["propagate", "raise", "omit"] = "propagate",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""pseudobulk_and_correlate
    For a given cell_type and grouping, pseudobulk by sample and then perform a
    spearman's correlation calculation comparing the bulked adata to the olink data

    NOTE: if there are no overlapping samples, the function aborts and just returns `None`

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Source single-cell RNA-seq AnnData object to use for correlation analysis
    olink : :class:`~anndata.AnnData`
        Olink data for correlation analysis. MUST have at least some overlapping samples
        for each cell_type/group combination
    cell_type_col : str
        Column in `adata.obs` containing the cell type of interest
    cell_types : str | list[str]
        Particular cell type(s) to include in pseudobulked object
    group_col : str
        Column in `adata.obs` containing the groups of interest
    groups : str | list[str]
        Particular group(s) to include in pseudobulked object
    sample_col : str
        Column in `adata.obs` containing sample names. Used with grouping cells
        during pseudobulking
    layer : str, default="counts"
        Layer in `adata.layers` to take data from during pseudobulking
    filter_highly_variable : bool, default=True
        If `True`, use the information in `adata.var["highly_variable"]` to filter genes.
    normalize_bulk : bool, default=True
        If `True`, normalize and log-transform the pseudobulked data
    scale_bulk : bool, default=False
        If `True`, scale the pseudobulked data
    remove_nulls : bool, default=False
        If `True`, remove and RNA or Olink features that have all null values
    axis : int, default=0
        Passed to `scipy.stats.pearsonr`
    nan_policy : Literal["propagate", "raise", "omit"]='propagate'
        Passed to `scipy.stats.pearsonr`
    alternative : Literal["two-sided", "less", "greater"]="two-sided"
        Passed to `scipy.stats.pearsonr`

    Returns
    -------
    correlation coefficients : :class:`~pandas.DataFrame`

    associated p-values : :class:`~pandas.DataFrame`
    """
    cell_types = make_list_if_not(cell_types)
    groups = make_list_if_not(groups)

    adata_subset = adata[
        adata.obs[group_col].isin([groups]) & (adata.obs[cell_type_col].isin([cell_types])),
        :,
    ]
    pdata = dc.pp.pseudobulk(adata_subset, sample_col="sample_name", groups_col=None, layer=layer)

    if filter_highly_variable:
        if "highly_variable" in adata.var.columns:
            pdata = pdata[:, pdata.var_names[pdata.var["highly_variable"]]].copy()
        else:
            msg = "A True value was passed to the `highly_variable` argument, but no `highly_variable` column is present in `adata`. Aborting."
            raise ArgumentError(msg)

    if normalize_bulk:
        sc.pp.normalize_total(pdata, target_sum=1e4)
        sc.pp.log1p(pdata)
    if scale_bulk:
        sc.pp.scale(pdata, max_value=10)

    rna_df = pdata.to_df()

    olink_df = olink.to_df()

    # one dimension of the matrices can be different. In this case, we need to subset on common samples
    common_indices = rna_df.index.intersection(olink_df.index)
    rna_df = rna_df.loc[common_indices, :]
    olink_df = olink_df.loc[common_indices, :]

    corr_arr, p_arr = sp.stats.spearmanr(
        a=rna_df, b=olink_df, axis=axis, nan_policy=nan_policy, alternative=alternative
    )
    corr_df = pd.DataFrame(
        corr_arr,
        index=np.hstack(
            [
                rna_df.columns.to_series().add_suffix("_rna").index,
                olink_df.columns.to_series().add_suffix("_olink").index,
            ]
        ),
        columns=np.hstack(
            [
                rna_df.columns.to_series().add_suffix("_rna").index,
                olink_df.columns.to_series().add_suffix("_olink").index,
            ]
        ),
    )

    p_df = pd.DataFrame(
        p_arr,
        index=np.hstack(
            [
                rna_df.columns.to_series().add_suffix("_rna").index,
                olink_df.columns.to_series().add_suffix("_olink").index,
            ]
        ),
        columns=np.hstack(
            [
                rna_df.columns.to_series().add_suffix("_rna").index,
                olink_df.columns.to_series().add_suffix("_olink").index,
            ]
        ),
    )

    if remove_nulls:
        corr_df = corr_df.loc[~pd.isnull(corr_df).all(axis=1), ~pd.isnull(corr_df).all(axis=0)]

    return (corr_df, p_df)


def add_gene_set_score(
    adata: ad.AnnData,
    features: dict[str, Sequence[str]],
    pool: None = None,
    num_bins: int = 24,
    num_ctrls: int = 100,
    seed: int = 70,
    layer: str | None = None,
    inplace: bool = True,
):
    """add_gene_set_score
    Calculate and add a module score using the algorithm described in `Tirosh 2016`_
    (i.e. `{Seurat}::AddModuleScore`)

    Parameters
    ----------
    object : :class:`~anndata.AnnData`
    features : dict[str, list[str]]
        A dictionary of lists features for expression programs. Should have the form of 

    .. _python ::
        {
            "M1.1": ['GP9', 'VWF', 'ALOX12', ...],
            "M1.2": ['LY6E', 'IFIT1', 'OAS1', 'IFIT1', ...],
            ...
        }

    pool
        List of features to check expression levels against
    num_bins
        Number of bins of aggregate expression levels for all analyzed features
    num_ctrls
        Number of control features selected from the same bin per analyzed feature
    layer : str, default=None
        Layer to use for expression values. If `None`, use `adata.X`
    seed : int, default=70
        Number to use for numpy random number generator seed
    inplace : bool, default=True
        Write the results to `adata.obsm['score_tirosh']` or return a :class:~pandas.DataFrame of module scores

    Returns
    -------
    Depending on `inplace`, returns or updates `adata.obsm['score_tirosh']`


    Notes
    -----
    .. [1] I. Tirosh, et al. "Dissecting the multicellular ecosystem of
    metastatic melanoma by single-cell RNA-seq," Science, vol. 352,
    no. 6282, pp. 189-196, 2016, doi:10.1126/science.aad0501
    """
    if layer:
        obj = adata.layers[layer]
    else:
        obj = adata.X

    for key, values in features.items():
        if not_found := [x for x in values if x not in adata.var_names and x is not None]:
            logger.info(f"{', '.join(not_found)} were not found in the {key} and were dropped")
        features[key] = set(values).intersection(adata.var_names)

    pool = adata.var_names if pool is None else pool

    data_means_sr = pd.Series(obj[:, adata.var_names.get_indexer(pool)].mean(axis=0)).sort_values()

    rng = np.random.default_rng(seed=seed)

    data_cut = np.array_split(
        ary=np.add(data_means_sr, rng.normal(size=len(data_means_sr)) / 1e30),
        indices_or_sections=num_bins,
        axis=0,
    )
    data_cut = pd.concat([pd.Series(i, index=data_cut[i].index) for i, _ in enumerate(data_cut)])

    ctrl_use = {
        module: np.unique(
            np.hstack(
                [
                    data_cut[data_cut == data_cut[gene]].sample(n=num_ctrls, replace=False).index
                    for gene in features[module]
                ]
            )
        )
        for module in features.keys()
    }

    ctrl_scores_arr = np.vstack([obj[:, adata.var_names.get_indexer(ctrl_use[x])].mean(axis=1) for x in ctrl_use])
    features_scores_arr = np.vstack([obj[:, adata.var_names.get_indexer(features[x])].mean(axis=1) for x in features])

    features_scores_use_df = pd.DataFrame(
        np.subtract(features_scores_arr, ctrl_scores_arr).T,
        index=adata.obs_names,
        columns=features.keys(),
    )
    features_scores_use_df = features_scores_use_df.sub(features_scores_use_df.min(axis=0), axis=1).div(
        np.subtract(features_scores_use_df.max(axis=0), features_scores_use_df.min(axis=0)),
        axis=1,
    )

    if inplace:
        adata.obsm["score_tirosh"] = features_scores_use_df
    else:
        return features_scores_use_df
