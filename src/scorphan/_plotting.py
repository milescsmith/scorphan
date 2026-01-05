import pandas as pd
import seaborn as sns
from multiprocessing import cpu_count

def density_heatmap(
    obs_df: pd.DataFrame,
    column_var: str,
    row_var: str,
    plot: bool = True,
    return_df: bool = False,
    cluster: bool = False,
    **kwargs,
) -> pd.DataFrame | None:
    """Produce a heatmap displaying the fraction of the total of column_var made up of
    each of the categories in row_var

    Parameters
    ----------
    obs_df: pd.DataFrame
        `obs` from a :class:`anndata.Anndata` or :class:`mudata.MuData` object
    column_var: str
        obs column to use as the primary grouping variable
    row_var: str
        obs column containing categories that the `column_var` is divided among
    plot: bool, default 'True'
        do you want to show a plot?
    return_df: bool, default 'False'
        would you like to get the :class:`pandas.DataFrame` version of the plot data?
    cluster : bool, default 'False'
        Use heirarchial clustering? (i.e. use :meth:`seaborn.clustermap` instead of :meth:`seaborn.heatmap`?).
    **kwargs
        Extra parameters to pass to :func:`seaborn.heatmap`

    Returns
    -------
    :class:`pandas.DataFrame`
        A dataframe containing percentage of each column_var group made up of each row_var
        group

    Example
    -------
    >>> density_heatmap(
            obs_df=mdata.obs,
            column_var="class",
            row_var="labels",
            plot=True,
            return_df=False,
            annot=True,
        )
    """
    total_counts: pd.DataFrame = obs_df.value_counts([column_var]).reset_index().set_index(column_var)
    percentage: pd.DataFrame = (
        (obs_df.groupby(column_var, observed=True).value_counts([row_var]).reset_index())
        .set_index(column_var)
        .merge(total_counts, left_index=True, right_index=True)
        .apply(lambda x: x["count_x"] / x["count_y"], axis=1)
    )

    plot_df: pd.DataFrame = obs_df.groupby(column_var, observed=True).value_counts([row_var]).reset_index()
    plot_df.insert(loc=plot_df.shape[1], value=percentage.to_list(), column="percentage")

    match plot:
        case plot if cluster:
            _ = sns.clustermap(
                plot_df.drop(columns="count").pivot(columns=row_var, index=column_var, values="percentage").transpose(),
                **kwargs,
            )
        case plot if not cluster:
            _ = sns.heatmap(
                plot_df.drop(columns="count").pivot(columns=row_var, index=column_var, values="percentage").transpose(),
                **kwargs,
            )
        case _:
            pass
    if return_df:
        return plot_df
    else:
        return None


def plot_marker_motif_enrichment(
    adata,
    groupby: str,
    pval: float = 0.05,
    genome: str = "hg38",
    repeat_macs: bool = False,
    max_fdr: float = 0.0001,
    height: int = 1000,
    return_data: bool = False,
    blacklist: Path | None = None,
    n_jobs=-1,
    plot_motifs: bool = True,
):
    """
    adata : 
    groupby : str
    pval : float
        0.05
    genome : str
        "hg38"
    repeat_macs : bool
        False
    max_fdr : float
        0.0001
    height : int
        1000
    return_data : bool
        False
    blacklist : Path | None
        None
    n_jobs 
        default 1
    plot_motifs : bool
        True
    """
    import snapatac2 as snap
    
    if n_jobs == -1:
        logger.info("setting n_jobs")
        n_jobs = cpu_count()
    
    logger.info("matching genome")
    match genome:
        case ("hg38" | "GRCh38" | "human"):
            genome = snap.genome.GRCh38
        case ("hg19" | "GRCh37"):
            genome = snap.genome.GRCh37
        case ("mm39" | "GRCm39" | "mouse"):
            genome = snap.genome.GRCm39
        case ("mm10" | "GRCm38"):
            genome = snap.genome.GRCm38
        case _:
            msg = f"{genome} does not match a built in genome"
            raise ValueError(msg)

    if "macs3" not in adata.uns.keys():
        logger.info("Running MACS3")
        snap.tl.macs3(
            adata,
            groupby=groupby,
            n_jobs=n_jobs,
            blacklist=blacklist,
        )
    elif ("macs3" in adata.uns.keys() and repeat_macs):
        logger.info("Running MACS3")
        snap.tl.macs3(
            adata,
            groupby=groupby,
            n_jobs=n_jobs,
            blacklist=blacklist,
        )
    else:
        logger.info("Not repeating MACS")

    logger.info("merging peaks")
    peaks = snap.tl.merge_peaks(adata.uns["macs3"], genome)

    logger.info("making a peak matrix")
    peaks_mat = snap.pp.make_peak_matrix(adata, use_rep=peaks["Peaks"])

    logger.info("calculating marker peaks")
    marker_peaks = snap.tl.marker_regions(peaks_mat, groupby=groupby, pvalue=pval)

    logger.info("calculating motif enrichment")
    # snap.pl.regions(peaks_mat, groupby=groupby, peaks=marker_peaks, interactive=False)
    motifs = snap.tl.motif_enrichment(
        motifs=snap.datasets.cis_bp(unique=True),
        regions=marker_peaks,
        genome_fasta=genome,
    )

    logger.info("plotting motif enrichment")
    if plot_motifs:
        p = snap.pl.motif_enrichment(motifs, max_fdr=max_fdr, height=height, interactive=False, show=False)
    else:
        p = None

    if return_data:
        return p, peaks, peaks_mat, marker_peaks, motifs
    else:
        return p
