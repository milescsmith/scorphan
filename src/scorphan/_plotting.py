import pandas as pd
import seaborn as sns
from multiprocessing import cpu_count
import warnings
from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal

import anndata as ad
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.cluster import hierarchy

A_GOOD_LINEWIDTH: float = 0.5
A_REASONABLE_WIDTH: int = 6
A_REASONABLE_HEIGHT: int = 3

class AxisVar(int, Enum):
    cells = 0
    features = 1

class ScaleMethod(str, Enum):
    standard_scale = "standard_scale"
    z_score = "z_score"

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
# adapted from the seaborn.matrix.ClusterGrid class methods `z_score()` and `standard_scale()`
def scale_data(
    data2d: pd.DataFrame | npt.NDArray,
    axis: Literal[0,1] | None = 0,
    method: ScaleMethod | None = ScaleMethod.standard_scale,
) -> pd.DataFrame | np.ndarray:
    """Standarize the mean and variance of the data axis

    Parameters
    ----------
    data2d : pandas.DataFrame
        Data to normalize
    axis : int
        Which axis to normalize across. If 0, normalize across rows, if 1,
        normalize across columns.

    Returns
    -------
    normalized : pandas.DataFrame
        Noramlized data with a mean of 0 and variance of 1 across the
        specified axis.
    """

    data_df: npt.ArrayLike = data2d if axis == 1 else np.transpose(data2d)

    match method:
        case ScaleMethod.standard_scale:
            subtract = data_df.min()
            data_df = (data_df - subtract) / (data_df.max() - data_df.min())
        case ScaleMethod.z_score:
            data_df = (data_df - data_df.mean()) / data_df.std()
        case _:
            msg = "That is not a scaling method I know."
            raise RuntimeError(msg)

    data_df = data_df if axis == 1 else np.transpose(data_df)
    return data_df

def prep_plot_df(
    adata: ad.AnnData,
    geneset: Sequence[str],
    group_by: str | None = None,
    cluster_cols: bool = True,
) -> pd.DataFrame:
    if len(adata.var_names.intersection(geneset)) != len(geneset):
        msg = f"{', '.join([_ for _ in geneset if _ not in adata.var_names.intersection(geneset)])} were not found in the data"
        warnings.warn(msg, stacklevel=2)

    if group_by is None:
        plot_df = (
            sc.get.obs_df(
                adata, keys=adata.var_names.intersection(geneset).to_list()
            )
        )
    else:
        plot_df = (
            sc.get.obs_df(
                adata, keys=[*adata.var_names.intersection(geneset).to_list(), group_by]
            )
            .groupby(group_by)
            .mean()
        )

    if cluster_cols and len(plot_df.columns[(np.std(plot_df) == 0)] != 0):
        msg = f"All values for {', '.join(plot_df.columns[(np.std(plot_df) == 0)])} were the same, which fill cause `sns.clustermap` to crash, so those have been removed"
        warnings.warn(msg, stacklevel=2)
        plot_df = plot_df.loc[:, (np.std(plot_df) != 0)]

    return plot_df

def pathway_matrixplot(
    adata: ad.AnnData,
    geneset: Sequence[str],
    group_by: str | None = None,
    width: int = A_REASONABLE_WIDTH,
    height: int = A_REASONABLE_HEIGHT,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    scale_by: AxisVar | None = None,
    scale_method: ScaleMethod = ScaleMethod.standard_scale,
    cmap: str = "viridis",
    linewidths: float = A_GOOD_LINEWIDTH,
    linecolor: str = "grey",
    **kwargs,
) -> None:

    if "groupby" in kwargs and group_by is None:
        msg = "`groupby` is not a valid parameter - did you mean `group_by`? Assuming you did and carrying on."
        warnings.warn(msg, stacklevel=2)
        group_by = kwargs.pop("groupby")

    plot_df = prep_plot_df(adata=adata, geneset=geneset, group_by=group_by, cluster_cols=cluster_cols)
    match scale_by:
        case AxisVar.features:
            scale_axis = 1
        case AxisVar.cells:
            scale_axis = 0
        case _:
            msg = f"You are attempting to scale by {scale_by}. Please scale either by 'features' or 'cells'. For not, not scaling"
            warnings.warn(msg, stacklevel=2)
            scale_axis = None
    plot_df = scale_data(data2d=plot_df, axis=scale_axis, method=scale_method)

    _ = sns.clustermap(
        data=plot_df,
        # standard_scale=standard_scale,
        figsize=(width, height),
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        cmap=cmap,
        linewidths=linewidths,
        linecolor=linecolor,
        robust=True,
        antialiased=True,
        **kwargs,
    )

def feature_hierarchy(
    adata: ad.AnnData,
    geneset: Sequence[str],
    scale_by: AxisVar = AxisVar.features,
    group_by: str | None = None,
    method: str = "average",
    metric: str = "euclidean",
    scaling_method: ScaleMethod | None = None,
    plot: bool = False,
    return_dendro_dict: bool = False,
    ax: mpl.axes.Axes | None = None,  # pyright: ignore[reportAttributeAccessIssue]
) -> pd.DataFrame | dict[str, Any] | None:
    """Given a cell-by-feature dataframe, calculate the grouping of items along the given axis (i.e. how genes cluster at the end of the dendrogram leaves)

    Parameters
    ----------
    adata: :class:`anndata.Anndata`
        Object containing expression values to use in clustering genes
    geneset: :class:`collections.abc.Sequence[str]`
        obs column to use as the primary grouping variable
    scale_by : :class:`AxisVar`
        How should the data be scaled, by cells or by features? Default: "features"
    group_by: str, Optional
        How should the cells be grouped, if they should be grouped. Default: None
    method: str
        Method to use when determining feature similarity. Default: "average"
    metric: str
        Metric to use in determining feature similary. Default: "euclidean"
    scaling_method: :class:`ScaleMethod`, Optional
        If the data is to be scaled, how should it be scaled? Using a "standard_scale" or "z_score"?. Default: None
    plot: bool 
        Show the dendrogram produced? Default: False
    return_dendro_dict: 
        Instead of a `feature | cluster` :class:`pd.DataFrame`, return the dictionary produced by scipy.hierarchy.dendrogram. Default = False
    ax: mpl.axes.Axes
        Axes object to pass when plotting the dendrogram.

    Returns
    -------
    By default, :class:`pandas.DataFrame`
        A dataframe containing percentage of each column_var group made up of each row_var
        group
    If `return_dendro_dict` is `True`, a `dict[str, Any]`

    Example
    -------
    >>> feature_hierarchy(
            adata=adata,
            geneset=["IFIT1", "IFIT2", "SIRT1", "SIRT2", "GAPDH"]
            axis="features",
            group_by="leiden",
            scaling: "standard_scale,
            plot=False,
            return_dendo_dict=False,
            ax=None,
        )
    """
    plot_df: pd.DataFrame = prep_plot_df(adata, geneset=geneset, group_by=group_by)

    if scale_by:
        match scale_by:
            case "features":
                scale_axis = 1
            case "cells":
                scale_axis = 0
            case _:
                msg = "You are attempting to scale by {standard_scale}. Please scale either by 'features' or 'cells'."
                warnings.warn(msg, stacklevel=2)
                scale_axis = None
        plot_df = scale_data(data2d=plot_df, axis=scale_axis, method=scaling_method)

    if plot:
        dendro = hierarchy.dendrogram(
            hierarchy.linkage(np.transpose(plot_df), method=method, metric=metric), no_plot=False, ax=ax
        )
        if return_dendro_dict:
            return dendro
    else:
        dendro = hierarchy.dendrogram(
            hierarchy.linkage(np.transpose(plot_df), method=method, metric=metric), no_plot=True
        )
        return pd.DataFrame(
            {"group": dendro["leaves_color_list"]},
            index=plot_df.columns.to_series().iloc[dendro["leaves"]],
        ).reset_index()
