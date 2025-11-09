import pandas as pd
import seaborn as sns


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
