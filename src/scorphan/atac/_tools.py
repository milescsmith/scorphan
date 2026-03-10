# import warnings
# from collections.abc import Sequence
# from pathlib import Path

# import bioframe as bf
# import numpy as np
# import pandas as pd
# import polars as pl
# import snapatac2 as snap
# from loguru import logger
# from pyfaidx import Fasta
# from snapatac2._snapatac2 import PyDNAMotif
# from snapatac2._utils import fetch_seq
# from snapatac2.datasets import cis_bp
# from snapatac2.genome import hg38


# def tfbs_for_group(
#     regions: dict[str, pl.DataFrame],
#     group_of_interest: str,
#     tf: str,
#     genome_file: Path | None = None,
#     motif_db: list[PyDNAMotif] | None = None,
# ) -> dict[str, list[pd.Series]]:
#     """
#     Parameters
#     ----------
#     regions : dict[str, pl.DataFrame]
#         Anything mimicing the output from `snapatac2.tl.marker_regions`
#     group_of_interest : str
#         One of the keys for the `regions` parameter above. If `regions` is the output from snapatac2.tl.marker_regions`, this would be one of the groups in the `groupby` parameter
#     tf : str
#         What transcription factor to filter on
#     genome : Path | None = None
#         The path to an uncompressed FASTA file. By default, it will use `snapatac2.genome.hg38.fasta`
#     motif_db : list[PyDNAMotif] = cis_bp()
#         Transcription factor binding motifs to search for. By default, it will use `snapatac2.datasets.cis_bp()`

#     Returns
#     -------
#     dict[str, list[str]]
#         It looks something like:
#         ```
#         {
#             'FOSL2+M07812_2.00': [
#                 'chr1:12153200-12153701',
#                 'chr1:178192601-178193102'
#             ]
#         }
#         ```
#     """
#     genome_file = hg38.fasta if genome_file is None else genome_file

#     motif_db= cis_bp() if motif_db is None else motif_db

#     genome = Fasta(genome_file, one_based_attributes=False)
#     sequences = {region: fetch_seq(genome, region) for region in regions[group_of_interest]}
#     motifs: dict[str, PyDNAMotif] = {_.id: _ for _ in motif_db if _.name == tf}

#     return {
#         motif_name: [
#             key
#             for key, value
#             in zip(
#                 sequences.keys(),
#                 motif_data.with_nucl_prob().exists(list(sequences.values())),
#                 strict=True
#             )
#             if value is True
#         ]
#         for motif_name, motif_data
#         in motifs.items()
#     }


# def closest_gene(
#     annotation_df: pd.DataFrame,
#     location: str | list[str] | dict[str, str | int] | pd.Series | pd.DataFrame | pl.DataFrame,
#     feature_type: str = "gene",
#     expand_upstream_range: int | None = None,
#     expand_downstream_range: int | None = None,
# ) -> pd.DataFrame:
#     """
#     annotation_df : pd.DataFrame
#         Your bog-standard GTF-derived 9-column dataframe.
#         I ain't holdin' your hand here - load it in before you try to run this. Use `bioframe.read_table(annot.gtf.gz)`
#         or similar. And trim off the header!
#     location: str | dict[str, str|int] | pd.Series | pd.DataFrame
#         Genomic coordinates to lookup. Can be
#             * a string in the form of "chr1:1-100"
#             * a list of the above strings
#             * a dict in the form of {"chrom": 1, "start": 1, "end": 100}
#             * a pd.Series equivalent of the above dict
#             * a pd.DataFrame or pl.DataFrame (autoconverts) with "chrom", "start" and "end" columns; in this case, all
#                 rows will be used
#     feature_type : str
#         Because `annotation_df` is big with lots of overlapping features AND this func is named "closest_gene", we filter
#         by the "feature" column. Default: "gene"
#     expand_upstream_range : int | None
#         Subtract the given value from `start` in `location` to look farther upstream
#     expand_downstream_range : int | None
#         Subtract the given value from `end` in `location` to look farther downstream
#     """
#     for x in ["start", "end"]:
#         if annotation_df[x].dtype != int:
#             annotation_df[x] = annotation_df[x].astype(int)

#     features_df = annotation_df.query("feature == @feature_type")

#     match location:
#         case str():
#             loc = genomic_coords_to_df([location])
#         case list():
#             loc = genomic_coords_to_df(location)
#         case dict():
#             loc = pd.DataFrame(pd.Series(location)).transpose()
#         case pd.Series():
#             loc = pd.DataFrame(location).transpose()
#         case pl.DataFrame():
#             loc = location.to_pandas()
#         case pd.DataFrame():
#             loc = location
#         case _:
#             msg = f"{type(location)} is not a valid option for `location`"
#             raise ValueError(msg)

#     for x in ["start", "end"]:
#         if loc[x].dtype != int:
#             loc[x] = loc[x].astype(int)

#     if expand_upstream_range:
#         loc["start"] = loc["start"] - expand_upstream_range
#     if expand_downstream_range:
#         loc["end"] = loc["end"] + expand_downstream_range

#     overlaps = bf.overlap(
#         df1=features_df,
#         df2=loc,
#         return_overlap=True,
#         suffixes=("", "_query")
#     ).query("~@pd.isnull(start_query)")

#     overlaps.insert(
#         loc=overlaps.shape[1],
#         column="closest_gene",
#         value=(
#             overlaps["attributes"]
#             .str.split("; ", expand=True)[2]
#             .str.split(" ", expand=True)[1]
#             .str.strip('"')
#         ),
#     )

#     return overlaps


# def genomic_coords_to_df(coords: Sequence[str]) -> pd.DataFrame:
#     coords_df = pd.DataFrame(
#         data=np.vstack([_.replace(":", "-").split("-") for _ in coords]),
#         columns=pd.Index(["chrom", "start", "end"])
#     )

#     coords_df["strand"] = ["+" if (_[1]["start"] < _[1]["end"]) else "-" for _ in coords_df.iterrows()]
#     return coords_df


# def groupwise_diff_peaks(
#     adata,
#     peak_mat,
#     groupby,
#     group1,
#     group2,
#     outdir: Path | None = None,
#     min_log_fc: float = 0.25,
#     min_pct: float = 0.1,
#     max_fdr: float = 0.01,
#     height: int = 600,
#     width: int = 400,
# ):
#     diff_peaks = snap.tl.diff_test(
#         data=peak_mat,
#         cell_group1=adata.obs[:][:, groupby] == group1,
#         cell_group2=adata.obs[:][:, groupby] == group2,
#         min_log_fc=min_log_fc,
#         # setting the min_pct to speed this along
#         min_pct=min_pct,
#     )

#     diff_peaks_filtered = diff_peaks.filter(pl.col("adjusted p-value") < max_fdr)

#     group2_vs_group1_motifs = snap.tl.motif_enrichment(
#         motifs=snap.datasets.cis_bp(unique=True),
#         regions={
#             group1: diff_peaks_filtered.filter(pl.col("log2(fold_change)") > 0.5)
#             .get_column("feature name")
#             .to_pandas(),
#             group2: diff_peaks_filtered.filter(pl.col("log2(fold_change)") < -0.5)
#             .get_column("feature name")
#             .to_pandas(),
#         },
#         genome_fasta=snap.genome.hg38,
#     )

#     if outdir:
#         group2_vs_group1_motifs[group1].write_csv(
#             outdir / f"{group1}_peaks_vs_{group2}_motifs.csv",
#         )
#         group2_vs_group1_motifs[group2].write_csv(
#             outdir / f"{group2}_peaks_vs_{group1}_motifs.csv",
#         )

#     p = None
#     try:
#         p = snap.pl.motif_enrichment(
#             group2_vs_group1_motifs, max_fdr=max_fdr, height=height, interactive=False
#         )
#     except ValueError:
#         msg = "Guess nothing survived significance. No plot for you!"
#         warnings.warn(msg, stacklevel=2)

#     return group2_vs_group1_motifs, p

# def plot_marker_motif_enrichment(
#     adata,
#     groupby: str,
#     pval: float = 0.05,
#     genome: str = "hg38",
#     repeat_macs: bool = False,
#     max_fdr: float = 0.0001,
#     height: int = 1000,
#     return_data: bool = False,
#     blacklist: Path | None = None,
#     n_jobs=-1,
#     plot_motifs: bool = True,
# ):
#     from multiprocessing import cpu_count

#     import snapatac2 as snap

#     if n_jobs == -1:
#         logger.info("setting n_jobs")
#         n_jobs = cpu_count()

#     logger.info("matching genome")
#     match genome:
#         case "hg38" | "GRCh38" | "human":
#             genome = snap.genome.GRCh38
#         case "hg19" | "GRCh37":
#             genome = snap.genome.GRCh37
#         case "mm39" | "GRCm39" | "mouse":
#             genome = snap.genome.GRCm39
#         case "mm10" | "GRCm38":
#             genome = snap.genome.GRCm38
#         case _:
#             msg = f"{genome} does not match a built in genome"
#             raise ValueError(msg)

#     if "macs3" not in adata.uns.keys():
#         logger.info("Running MACS3")
#         snap.tl.macs3(
#             adata,
#             groupby=groupby,
#             n_jobs=n_jobs,
#             blacklist=blacklist,
#         )
#     elif "macs3" in adata.uns.keys() and repeat_macs:
#         logger.info("Running MACS3")
#         snap.tl.macs3(
#             adata,
#             groupby=groupby,
#             n_jobs=n_jobs,
#             blacklist=blacklist,
#         )
#     else:
#         logger.info("Not repeating MACS")

#     logger.info("merging peaks")
#     peaks = snap.tl.merge_peaks(adata.uns["macs3"], genome)

#     logger.info("making a peak matrix")
#     peaks_mat = snap.pp.make_peak_matrix(adata, use_rep=peaks["Peaks"])

#     logger.info("calculating marker peaks")
#     marker_peaks = snap.tl.marker_regions(peaks_mat, groupby=groupby, pvalue=pval)

#     logger.info("calculating motif enrichment")
#     # snap.pl.regions(peaks_mat, groupby=groupby, peaks=marker_peaks, interactive=False)
#     motifs = snap.tl.motif_enrichment(
#         motifs=snap.datasets.cis_bp(unique=True),
#         regions=marker_peaks,
#         genome_fasta=genome,
#     )

#     logger.info("plotting motif enrichment")
#     if plot_motifs:
#         p = snap.pl.motif_enrichment(
#             motifs, max_fdr=max_fdr, height=height, interactive=False, show=False
#         )
#     else:
#         p = None

#     if return_data:
#         return p, peaks, peaks_mat, marker_peaks, motifs
#     else:
#         return p
