import warnings
from collections.abc import Sequence
from importlib.util import find_spec

import anndata as ad
import decoupler as dc
import liana as li
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from rich.progress import track

# from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from .log import init_logger

warnings.simplefilter("ignore")


def calc_bulk_degs_per_group(
    pdata: ad.AnnData,
    groupby: str,
    ref_level: str,
    test_level: str,
    condition_key: str,
    skip_groups: list[str] | None = None,
    parallel: bool = False,
    min_number_of_samples: int = 2,
    force_recalc: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    r"""Using a pseudobulked object (preferably generated using :func:`decoupler.pp.pseudobulk`), will subset
    ``pdata`` by each value in ``groupby`` and compare the samples corresponding to the ``test_level`` to the
    ``ref_level`` of ``condition_key``

    Parameters
    ----------
    pdata : :class:`ad.AnnData`
        A anndata object already pseudobulked by ``groupby``
    groupby : str
        A column in ``pdata.obs`` by which to group data for comparisons.
    ref_level : str
        The category in ``groupby`` to use as the reference level in comparisons
    test_level : str
        The category in ``groupby`` to test against ``ref_level``
    condition_key : str
        A column in ``pdata.obs`` containing the ``ref_level`` and ``test_level`` identities
    skip_groups : list[str], optional
        Groups in ``groupby`` to skip when performing comparisons (i.e. there are too few samples of ``ref_level`` or
        ``test_level`` to enable comparisons)
    parallel : bool, default=False
        Perform tests for each ``groupby`` group in parallel? Should be vastly faster, though may cause error messages
        to be more confusing.
    min_number_of_samples : int, default=2
        If, after subsetting ``pdata`` by the current ``groupby`` category there are fewer than
        ``min_number_of_samples``, skip analysis of that category
    force_recalc : bool, default=False
        The results from previous ``calc_bulk_degs_per_group`` runs are stored within ``pdata`` in
        ``pdata.uns["{condition_key}({test_level}_vs_{ref_level})_for_{groupby}"]`` and a copy of the previously
        generated DataFrame will be returned. If ``force_recalc`` is set to ``True`` these previous values will be
        ignored and everything recalculated.
    verbose : bool, default=False
        Display extra information.

    Returns
    -------
    pd.DataFrame :
        A dataframe similar to the output of :func:`pydeseq2.ds.DeseqStats.results_df` with a column added for ``groupby``/
        For example:

        ====== ============= ========= ============== ======== ========= ======== ========
        index  level2_labels baseMean  log2FoldChange lfcSE    stat      pvalue   padj
        ====== ============= ========= ============== ======== ========= ======== ========
        HES4   IL1BCD14monos 12.787701 1.545927       0.743384 3.074723  0.002107 0.126105
        ISG15  IL1BCD14monos 68.270440 1.835120       0.614223 3.733109  0.000189 0.032866
        SDF4   IL1BCD14monos 28.175699 -0.029049      0.171900 -0.421481 0.673404 0.954327
        UBE2J2 IL1BCD14monos 17.744663 -0.066789      0.198866 -1.094360 0.273797 0.824913
        INTS11 IL1BCD14monos 12.376760 -0.022870      0.190244 -0.502349 0.615422 0.943669
        ====== ============= ========= ============== ======== ========= ======== ========

    Example
    -------
    >>> calc_bulk_degs_per_group(
        pdata=pdata,
        groupby="celltype",
        ref_level="controls",
        test_level="infected",
        condition_key="infection_status",
        skip_groups="monocytes",
        parallel=True,
        min_number_of_samples=2,
        force_recalc=False,
        verbose=False,
    )
    """
    if (f"{condition_key}({test_level}_vs_{ref_level})_for_{groupby}_degs" in pdata.uns) and (force_recalc is False):
        dea_df = pdata.uns[f"{condition_key}({test_level}_vs_{ref_level})_for_{groupby}_degs"].copy()

    # NOTE: uses HUMAN gene symbols!
    if not isinstance(skip_groups, list):
        skip_groups = [skip_groups]
    if verbose:
        init_logger(verbose=3, save=False)

    comparison = [ref_level, test_level]

    pdata.obs[condition_key] = pdata.obs[condition_key].cat.set_categories([ref_level, test_level], ordered=True)

    logger.debug("Calculating cell type DEGs...")
    calc_for_cell_types = {_ for _ in pdata.obs[groupby].unique() if _ not in skip_groups}

    if parallel:

        def ploop(x):
            # from functools import partial

            return _calc_bulk_group_degs(
                pdata=pdata,
                groupby=groupby,
                cell_group=x,
                condition_key=condition_key,
                comparison=comparison,
                ref_level=ref_level,
                test_level=test_level,
                min_number_of_samples=min_number_of_samples,
                verbose=verbose,
            )

        dea_results_list = Parallel(n_jobs=-2, prefer="threads", return_as="list")(
            delayed(ploop)(cell_group) for cell_group in calc_for_cell_types
        )
        dea_results = dict(zip(calc_for_cell_types, dea_results_list, strict=True))

    else:
        dea_results = {
            cell_group: _calc_bulk_group_degs(
                pdata,
                groupby,
                cell_group,
                condition_key,
                comparison,
                ref_level,
                test_level,
                verbose,
            )
            for cell_group in track(calc_for_cell_types)
        }

    logger.debug("Creating DEG table...")
    # Select cell profiles
    dea_df = (
        pd.concat(dea_results)
        .reset_index()
        .rename(columns={"level_0": groupby, "level_1": "index"})
        .set_index("index")
        .query("~@pd.isnull(padj)")
    )

    return dea_df


def _select_top_n(d: dict[str, float], n: int) -> dict[str, float]:
    """_summary_

    Parameters
    ----------
    d : dict[str, float]
        _description_
    n : int
        _description_

    Returns
    -------
    dict[str, float]
        _description_
    """
    d = dict(sorted(d.items(), key=lambda item: abs(item[1]), reverse=True))
    return {k: v for i, (k, v) in enumerate(d.items()) if i < n}


def _calc_receptor_tf_scores(
    lr_res: pd.DataFrame,
    target_group: str,
    source_group: str,
    groupby: str,
    dea_df: pd.DataFrame,
    net: pd.DataFrame | None = None,
    n_ligands: int = 10,
    n_tfs: int = 5,
) -> tuple[pd.DataFrame | pd.DataFrame]:
    """Using the results from `calc_lr_res`, determine the predicted most active receptors and the likely downstream
    transcription factors they are activating

    Note: Currently only works on a single source_celltype and a single target_celltype at a time

    Parameters
    ----------
    lr_res : pd.DataFrame
        _description_
    target_group : str
        _description_
    source_group : str
        _description_
    groupby : str
        _description_
    dea_df : pd.DataFrame
        _description_
    net : pd.DataFrame | None, optional
        _description_, by default None
    n_ligands : int, optional
        _description_, by default 10
    n_tfs : int, optional
        _description_, by default 5

    Returns
    -------
    receptor_scores : pd.DataFrame
    tf_scores : pd.DataFrame
    """

    # should we make this a parameter instead?
    net = (
        dc.op.collectri(organism="human", remove_complexes=False, license="academic", verbose=False)
        if net is None
        else net
    )

    # TODO: need a way of selecting the top n for multiple target cell types
    # NOTE: We sort by the absolute value of the interaction stat
    lr_stats = lr_res.query("source.isin([@source_celltype]) & target.isin([@target_celltype])")

    lr_stats = lr_stats.sort_values("interaction_stat", ascending=False, key=abs)
    lr_dict = lr_stats.set_index("receptor")["interaction_stat"].to_dict()
    receptor_scores = _select_top_n(lr_dict, n=n_ligands)

    dea_wide: pd.DataFrame = (
        dea_df.loc[:, [groupby, "stat"]]
        .reset_index(names="genes")
        .pivot(index=groupby, columns="genes", values="stat")
        .fillna(0)
    )

    estimates, _ = dc.mt.ulm(  # pyright: ignore[reportGeneralTypeIssues]
        data=dea_wide, net=net
    )
    tf_data: pd.DataFrame = (
        estimates.copy()
    )  # explicitly type since no linters seem to understand what dc.mt.ulm returns
    tf_dict = tf_data.loc[target_group].to_dict()
    tf_scores = _select_top_n(tf_dict, n=n_tfs)

    return (receptor_scores, tf_scores)


def _find_causal_network(
    adata: ad.AnnData,
    groupby: str,
    target_group: str,
    receptor_scores: pd.DataFrame,
    tf_scores: pd.DataFrame,
    solver: str | None = None,
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    adata : ad.AnnData
        _description_
    groupby : str
        _description_
    target_group : str
        _description_
    receptor_scores : pd.DataFrame
        _description_
    tf_scores : pd.DataFrame
        _description_
    solver : str | None, optional
        _description_, by default None

    Returns
    -------
    pd.DataFrame
        _description_
    """

    import omnipath as op

    ppis = op.interactions.OmniPath().get(genesymbols=True)

    ppis["mor"] = ppis["is_stimulation"].astype(int) - ppis["is_inhibition"].astype(int)
    ppis = ppis.query("(mor != 0) & (curation_effort >= 5) & (consensus_direction)")

    input_pkn = ppis[["source_genesymbol", "mor", "target_genesymbol"]].rename(
        columns={"source_genesymbol": "source", "target_genesymbol": "target"}
    )

    prior_graph = li.mt.build_prior_network(input_pkn, receptor_scores, tf_scores, verbose=True)
    temp = adata[adata.obs[groupby] == target_group].copy()
    node_weights = pd.DataFrame(temp.X.getnnz(axis=0) / temp.n_obs, index=temp.var_names)
    node_weights = node_weights.rename(columns={0: "props"})
    node_weights = node_weights["props"].to_dict()

    if (solver is None) or (solver == "gurobi"):
        solver = "scipy" if find_spec("gurobipy") is None else "gurobi"

    logger.debug(
        f"Running `liana.mt.find_causalnet` using {solver} as the solver. "
        f"Note that if it is using `scipy` instead of `gurobi`, things are going to take 10 times longer."
    )
    df_res, _ = li.mt.find_causalnet(
        prior_graph=prior_graph,
        input_node_scores=receptor_scores,
        output_node_scores=tf_scores,
        node_weights=node_weights,
        # penalize (max_penalty) nodes with counts in less than 0.1 of the cells
        node_cutoff=0.1,
        max_penalty=1,
        # the penaly of those in > 0.1 prop of cells set to:
        min_penalty=0.01,
        edge_penalty=0.1,
        verbose=False,
        max_runs=50,  # NOTE that this repeats the solving either until the max runs are reached
        stable_runs=10,  # or until X number of consequitive stable runs are reached (i.e. no new edges are added)
        solver=solver,  # 'scipy' is available by default, but often results in suboptimal solutions
    )
    return df_res


@logger.catch
def _calc_bulk_group_degs(
    pdata: ad.AnnData,
    groupby: str,
    cell_group: str,
    condition_key: str,
    comparison: Sequence[str],
    ref_level: str,
    test_level: str,
    min_number_genes: int = 50,
    min_number_of_samples: int = 2,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """_summary_

    Parameters
    ----------
    pdata : ad.AnnData
        _description_
    groupby : str
        _description_
    cell_group : str
        _description_
    condition_key : str
        _description_
    comparison : Sequence[str]
        _description_
    ref_level : str
        _description_
    test_level : str
        _description_
    min_number_genes : int, optional
        _description_, by default 50
    min_number_of_samples : int, optional
        _description_, by default 2
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # if verbose:
    #     init_logger(verbose=3, save=False)
    quiet = not verbose

    logger.debug(f"Subsetting on {cell_group} and {','.join(comparison)}...")
    ctdata = pdata[(pdata.obs[groupby] == cell_group) & (pdata.obs[condition_key].isin(comparison))]

    # counts = ctdata.obs.value_counts(condition_key)
    if ((counts := ctdata.obs.value_counts(condition_key)) <= min_number_of_samples).any():
        msg = f"There are {min_number_of_samples} or fewer samples for {', '.join(pd.DataFrame(counts).query('count <= @min_number_of_samples').index)}"
        warnings.warn(msg, stacklevel=2)
        return None

    logger.debug(f"Obtain genes that pass the edgeR-like thresholds for {cell_group}")
    # Obtain genes that pass the edgeR-like thresholds
    # NOTE: QC thresholds might differ between cell types, consider applying them by cell type
    genes = dc.pp.filter_by_expr(
        ctdata,
        group=condition_key,
        min_count=5,  # a minimum number of counts in a number of samples
        min_total_count=10,  # a minimum total number of reads across samples
    )

    logger.debug(f"Filtering genes for {cell_group}")
    # Filter by these genes
    if (genes is not None) and (len(genes) > min_number_genes):
        ctdata = ctdata[:, genes].copy()

    logger.debug(f"Building DESeq2 object for {cell_group}")
    # Build DESeq2 object
    dds = DeseqDataSet(
        adata=ctdata,
        design_factors=condition_key,
        ref_level=[condition_key, ref_level],  # set control as reference
        refit_cooks=True,
        quiet=quiet,
    )

    logger.debug(f"Computing LFCs for {cell_group}")
    # Compute LFCs
    dds.deseq2()

    logger.debug(f"Creating contrasts between {test_level} and {ref_level} for {cell_group}")
    # Contrast between stim and ctrl
    stat_res = DeseqStats(dds, contrast=[condition_key, test_level, ref_level], quiet=quiet)
    stat_res.quiet = quiet

    logger.debug(f"Computing Wald stats {cell_group}")
    # Compute Wald test
    stat_res.summary()

    logger.debug(f"Shrinking LFCs for {cell_group}")
    # Shrink LFCs
    stat_res.lfc_shrink(coeff=stat_res.contrast_vector.index[1])  # {condition_key}_cond_vs_ref

    return stat_res.results_df


def CCC_to_tf_network(
    adata: ad.AnnData,
    sample_key: str,
    groupby: str,
    ref_level: str,
    test_level: str,
    condition_key: str,
    source_celltype: str,
    target_celltype: str,
    pdata: ad.AnnData | None = None,
    min_cells: int = 10,
    min_counts: int = 1000,
    layer_use: str = "counts",
    net: pd.DataFrame | None = None,
    n_ligands: int = 10,
    n_tfs: int = 5,
    solver: str = "scipy",  # should be an enum or Literal
    skip_celltypes: list[str] | None = None,
    verbose: bool = False,
    prior_lr_res: pd.DataFrame | None = None,
    prior_dea_df: pd.DataFrame | None = None,
    force_recalc: bool = False,
) -> pd.DataFrame:
    """This will take an AnnData object all the way to predicting a causal signalling network to explain perturbances a target cell type caused by communincation
    from a source cell type.

    Note that if you are running some of this for the first time IT MAY TAKE A WHILE. In particular, creating a pseudobulked object and running the cell type differential
    expression analysis can take more than 30 minutes or so. I've attempted to make it so that the results from  longer calculations are stored

    Parameters
    ----------
    adata : ad.AnnData
        _description_
    sample_key : str
        _description_
    groupby : str
        _description_
    ref_level : str
        _description_
    test_level : str
        _description_
    condition_key : str
        _description_
    source_celltype : str
        _description_
    target_celltype : str
        _description_
    pdata : ad.AnnData | None, optional
        _description_, by default None
    min_cells : int, optional
        _description_, by default 10
    min_counts : int, optional
        _description_, by default 1000
    layer_use : str, optional
        _description_, by default "counts"
    net : pd.DataFrame | None, optional
        _description_, by default None
    n_ligands : int, optional
        _description_, by default 10
    n_tfs : int, optional
        _description_, by default 5
    solver : str, optional
        _description_, by default "scipy"
    verbose : bool, optional
        _description_, by default False
    prior_lr_res : pd.DataFrame | None, optional
        _description_, by default None
    prior_dea_df : pd.DataFrame | None, optional
        _description_, by default None
    force_recalc : bool, optional
        _description_, by default False

    Returns
    -------
    pd.DataFrame
        _description_

    """
    tdata = adata[adata.obs[condition_key] == test_level]

    for x in [groupby, condition_key]:
        if adata.obs[x].dtype == "category":
            logger.debug(f"Changing dtype of {x}")
            adata.obs[x] = adata.obs[x].astype(str)

    if pdata is None:
        pdata = dc.pp.pseudobulk(
            adata,
            sample_col=sample_key,
            groups_col=groupby,
            layer=layer_use,
            mode="sum",
        )

    dc.pp.filter_samples(pdata, min_cells=min_cells, min_counts=min_counts)

    if prior_dea_df:
        logger.debug("Using existing DEGs")
        dea_df = prior_dea_df
    elif (f"{condition_key}({test_level}_vs_{ref_level})_for_{groupby}_degs" in pdata.uns) and (force_recalc is False):
        logger.debug("Using existing DEGs")
        dea_df = pdata.uns["celltype_degs"]["deg_df"].copy()
    else:
        logger.debug("Calculating DEGs for each celltype")
        dea_df = calc_bulk_degs_per_group(
            pdata=pdata,
            sample_key=sample_key,
            groupby=groupby,
            ref_level=ref_level,
            test_level=test_level,
            condition_key=condition_key,
            min_cells=min_cells,
            min_counts=min_counts,
            layer_use=layer_use,
            skip_groups=skip_celltypes,
            verbose=verbose,
        )
        pdata.uns[f"{condition_key}({test_level}_vs_{ref_level})_for_{groupby}_degs"] = dea_df.copy()

    if prior_lr_res is None:
        logger.debug("Calculating `lr_res`")
        lr_res = li.multi.df_to_lr(
            adata=tdata,
            dea_df=dea_df,
            resource_name="consensus",
            expr_prop=0.1,
            groupby=groupby,
            stat_keys=["stat", "pvalue", "padj"],
            use_raw=False,
            complex_col="stat",
            verbose=True,
            return_all_lrs=False,
        )
    else:
        logger.debug("Using existing `lr_res`")
        lr_res = prior_lr_res

    logger.debug(f"Calcuating receptor:TF scores for {target_celltype} and {source_celltype}")
    receptor_scores, tf_scores = _calc_receptor_tf_scores(
        lr_res=lr_res,
        target_group=target_celltype,
        source_group=source_celltype,
        groupby=groupby,
        dea_df=dea_df,
        net=net,
        n_ligands=n_ligands,
        n_tfs=n_tfs,
    )
    logger.debug("Finding causal network")
    network_res = _find_causal_network(
        adata=tdata,
        groupby=groupby,
        target_group=target_celltype,
        tf_scores=tf_scores,
        receptor_scores=receptor_scores,
        solver=solver,
    )

    return (network_res, pdata, lr_res, receptor_scores, tf_scores)
