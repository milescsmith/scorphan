import re

# from copy import deepcopy
from pathlib import Path
from typing import Final, Literal

import gseapy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from loguru import logger
from matplotlib import colormaps
from rich.progress import Progress

# from scanpy.tools._utils import _choose_representation
from scipy.sparse import issparse

from scorphan._utils import is_integer_array
from scorphan.logging import init_logger

MAX_PVAL: Final[float] = 0.05


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
    msg = "This function does not currently work and is disabled for the time being."
    logger.error(msg)
    raise RuntimeError(msg)
    # neighbors = mdata.uns[neighbors_key]
    # reps = {}
    # nfeatures = 0
    # nparams = neighbors["params"]
    # use_rep = {k: (v if v != -1 else None) for k, v in nparams["use_rep"].items()}
    # n_pcs = {k: (v if v != -1 else None) for k, v in nparams["n_pcs"].items()}
    # observations = mdata.obs.index

    # for mod, rep in use_rep.items():
    #     nfeatures += rep.shape[1]
    #     reps[mod] = _choose_representation(mdata.mod[mod], rep, n_pcs[mod])

    # rep = np.empty((len(observations), nfeatures), np.float32)
    # nfeatures = 0

    # for mod, crep in reps.items():
    #     cnfeatures = nfeatures + crep.shape[1]
    #     idx = observations.isin(mdata.mod[mod].obs.index)
    #     rep[idx, nfeatures:cnfeatures] = crep.toarray() if issparse(crep) else crep
    #     if np.sum(idx) < rep.shape[0]:
    #         imputed = crep.mean(axis=0)
    #         if issparse(crep):
    #             imputed = np.asarray(imputed).squeeze()
    #         rep[~idx, nfeatures : crep.shape[1]] = imputed
    #     nfeatures = cnfeatures

    # adata = AnnData(X=rep, obs=mdata.obs)

    # adata.uns[neighbors_key] = deepcopy(neighbors)
    # adata.uns[neighbors_key]["params"]["use_rep"] = "X"
    # del adata.uns[neighbors_key]["params"]["n_pcs"]
    # adata.obsp[neighbors["connectivities_key"]] = mdata.obsp[neighbors["connectivities_key"]]
    # adata.obsp[neighbors["distances_key"]] = mdata.obsp[neighbors["distances_key"]]

    # adata.uns["paga"] = mdata.uns["paga"].copy()

    # sc.tl.umap(
    #     adata=adata,
    #     min_dist=min_dist,
    #     spread=spread,
    #     n_components=n_components,
    #     maxiter=maxiter,
    #     alpha=alpha,
    #     gamma=gamma,
    #     negative_sample_rate=negative_sample_rate,
    #     init_pos=init_pos,
    #     random_state=random_state,
    #     a=a,
    #     b=b,
    #     copy=False,
    #     method=method,
    #     neighbors_key=neighbors_key,
    # )

    # mdata.obsm["X_umap"] = adata.obsm["X_umap"]
    # mdata.uns["umap"] = adata.uns["umap"]


def GSEApy_process(
    adata: AnnData,
    groupby: str | None = None,
    comparison_group: str | None = None,
    reference_group: str | None = None,
    obs_subset: pd.Series | pd.Index | list[str] | None = None,
    var_subset: pd.Series | pd.Index | list[str] | None = None,
    top_x_pathways: int = 5,
    top_pathway_type: Literal["heatmap", "dotplot"] = "dotplot",
    geneset: str | Path | None = None,
    outdir_path: Path | None = None,
    verbose: bool = False,
):
    """_summary_

    Parameters
    ----------
    adata : AnnData
        Object containing scRNAseq data to perform analysis on
    groupby : str | None, optional
        Column in adata.obs to use for grouping cells, by default None
    comparison_group : str | None, optional
        Factor in adata.obs[groupby] to generate the analysis for. For example, the disease group of interest., by default None
    reference_group : str | None, optional
        Factor in adata.obs[groupby] to set as the basis of comparison, such as the control group, by default None
    obs_subset : pd.Series | pd.Index | list[str] | None, optional
        Names from adata.obs_names (i.e. cell barcodes) to use to subset the data, by default None
    var_subset : pd.Series | pd.Index | list[str] | None, optional
        Names from adata.var_names (i.e. gene or protein names) to use to subset the data, by default None
    top_x_pathways : int, optional
        Number of pathways to generate plots for, by default 5
    top_pathway_type : Literal["heatmap", "dotplot"], optional
        Type of plots to generate, by default "dotplot"
    geneset : str | Path | None, optional
        Gene set list to test for. Can either be the name of a pathway or the path to a gene matrix transposed (*.gmt)
        file, by default None, which results in using "GO_Biological_Process_2023", "GO_Cellular_Component_2023", and
        "GO_Molecular_Function_2023"
    outdir_path : Path | None, optional
        Location to write output plots and spreadsheets to, by default None
    verbose : bool, optional
        More or less feedback, by default False

    """
    if verbose:
        init_logger(3)
    for _ in ["groupby", "reference_group", "comparison_group"]:
        if _ is None:
            msg = f"A value for {_} was not given"
            raise SyntaxError(msg)
    if geneset is None:
        geneset = ["GO_Biological_Process_2023", "GO_Cellular_Component_2023", "GO_Molecular_Function_2023"]
        logger.info("No geneset was specified. Using the three GeneOntology groups")

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
            adata.layers["lognorm"] = adata.X.copy()
            adata.X = adata.layers["counts"].copy()

            logger.info("normalizing and log transforming data")
            sc.pp.normalize_total(adata, target_sum=1e6)
            sc.pp.log1p(adata)
        else:
            msg = "This function requires starting with untransformed, integer counts."
            raise ValueError(msg)

    adata.layers["lognorm"] = adata.X.copy()

    ####GSEA
    if issparse(adata.layers["counts"]):
        logger.info("inflating sparse counts")
        counts_df = adata.layers["counts"].toarray().transpose()
    else:
        counts_df = adata.layers["counts"].transpose()
    logger.info("Performing GSEA")
    gs = gp.GSEA(
        data=pd.DataFrame(counts_df, index=var_subset, columns=obs_subset),  # row -> genes, column-> samples
        gene_sets=geneset,
        classes=adata.obs.loc[obs_subset, groupby].tolist(),
        permutation_num=1000,
        permutation_type="phenotype",
        outdir=str(outdir_path.joinpath("GSEA_results")),
        method="s2n",  # signal_to_noise
        threads=16,
        verbose=True,
    )
    gs.pheno_pos = comparison_group
    gs.pheno_neg = reference_group
    gs.run()

    # both sc.pl.heatmap and sc.pl.dotplot let you group variables *IF* you supply the var_names as a dict[str, list[str]]
    # this converts the "Term" and "Lead_genes" columns into a dict for the number of top pathways specified
    term_gene_dict = {
        re.sub(pattern=r"\s\(GO:[0-9]+\)", repl="", string=x[1]["Term"]): x[1]["Lead_genes"].split(";")
        for i, x in enumerate(gs.res2d.iterrows())
        if i < top_x_pathways
    }

    with Progress() as progress:
        task = progress.add_task(description="Creating GSEA pathway plots for", total=top_x_pathways)
        progress.start()
        for i in term_gene_dict:
            # term_name = re.sub(pattern=r"\s\(GO:[0-9]+\)", repl="", string=gs.res2d.Term.iloc[i])
            progress.update(task, description=f"Creating GSEA pathway plots for {i}")
            logger.info(f"generating plots for {i}")
            fig, ax = plt.subplots(figsize=(9, 5))
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
                        ax=ax,
                    )
            ax.set_title(i)
            ax.figure.savefig(Path(outdir_path).absolute().joinpath("GSEA_results", f"{i}_Heatmap.png"))
            progress.update(task, advance=1)
        term = gs.res2d.Term
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
    fig, ax = plt.subplots(figsize=(12, 12))

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
