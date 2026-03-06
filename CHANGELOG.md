# scorphan

## [0.14.0] - 2026-03-06

### Added
- `._tools.pseudobulk_and_correlate` to do just that between RNA/CITE-seq and Olink
- A couple of helper functions to `.utils`

### Changed
- Replaced `orjson` with `msgspec`
- Fixed up/reformatted some docstrings

## [0.13.0] - 2026-01-09

### Added
- A `not_yet_implemented` decorator to disable functions I'm still working on

### Changed
- Copied latest version of the log submodule from `taudata`
- Renamed several functions in the `_tools` submodule to mark them as private
- Decorate `_tools.transfer_to_asap` with `not_yet_implimented`

## [0.12.0] - 2026-01-05

### Added
- plotting of motif accessibility analysis

## [0.11.0] - 2025-12-05

### Added
- `tl.pseudobulk_differential_expression` for doing pretty much what it says on the tin - groups and pseudobulks cells
    based on a column in `obs` and then performs differential expression analysis using `PyDESeq2`
- `pl.pathway_matrixplot` - given a set of genes, such as a pathway, group based on a column in `obs` and plot the 
    groups mean expression level of the genes within that set
- `pl.feature_hierarchy` - save as `pathway_matrixplot`, except that it returns the hierarchial clustering data

## [0.10.0] - 2025-08-08

### Added
- New `gpu` submodule for placing various functions I'm attemptint to gpu-accelerate
- optional `cupy` dependency
- `tl.extract_h5_obs`, which can read the `obs` attribute of an Anndata or MuData object directly from disk and
    into a pandas.DataFrame

## [0.9.2] - 2025-06-26

### Fixed
- added the `()` to a stray function call that was missing them

## [0.9.1] - 2025-06-24

### Fixed
- Added what is probably a dumb method to get around a bug I encountered in so.pp.neighbors where one cannot add the adjacency graphs for two different modalities because of their sparseness

## [0.9.0] - 2025-06-23

### Fixed

- Repaired so.pp.neighbors
    - RAPIDS support disabled as it isn't currently working

## [0.8.1] - 2025-06-20

### Changed

- Decrease required library versions

## [0.8.0] - 2025/01/21

### Changed

- Updated dependencies

## [0.7.0] - 2025/01/10

### Added

- `so.ut.percentile_trim_rows`

## [0.6.0] - 2024/09/16

### Added

- Docstring to `so.tl.GSEApy_process`

### Changed

- Temporarily disabled `so.tl.muon_paga_umap` since it doesn't currently function correctly

### Fixed

- Minor fixes to `so.tl.GSEApy_process` to get it to work

## [0.5.1] - 2024/09/05

### Changed

- Renamed `so.ut.percentile_trim_rows` to `so.ut.percentile_trim_cols` since... that is what it does.

## [0.5.0] - 2024/09/04

### Changed

- Moved `so.pp.std_process_run()` components over to the `dsd` module

## [0.4.1] - 2024/08/29

### Fixed

- Minor fix for `so.pp.std_process_run`

## [0.4.0] - 2024/08/29

### Changed

- Added an option to choose between scrublet and vaeda for doublet detection
- Moved doublet detection to after removal of low quality cells (as per https://www.sc-best-practices.org/)

## [0.3.0] - 2024/08/09

### Changed

- renamed the `processing` submodule to `preprocessing`
- made all of the submodules private

## [0.2.0] - 2024/08/09

### Added

- `easyGSEApy`, for more easily running GSEApy analysis

## [0.1.0] - 2024/08/08

### Added

- Created module

[0.14.0]: https://github.com/milescsmith/scorphan/releases/compare/0.13.0..0.14.0
[0.13.0]: https://github.com/milescsmith/scorphan/releases/compare/0.12.0..0.13.0
[0.12.0]: https://github.com/milescsmith/scorphan/releases/compare/0.11.0..0.12.0
[0.11.0]: https://github.com/milescsmith/scorphan/releases/compare/0.10.0..0.11.0
[0.10.0]: https://github.com/milescsmith/scorphan/releases/compare/0.9.2..0.10.0
[0.9.2]: https://github.com/milescsmith/scorphan/releases/compare/0.9.1..0.9.2
[0.9.1]: https://github.com/milescsmith/scorphan/releases/compare/0.9.0..0.9.1
[0.9.0]: https://github.com/milescsmith/scorphan/releases/compare/0.8.1..0.9.0
[0.8.1]: https://github.com/milescsmith/scorphan/releases/compare/0.8.0..0.8.1
[0.8.0]: https://github.com/milescsmith/scorphan/releases/compare/0.7.0..0.8.0
[0.7.0]: https://github.com/milescsmith/scorphan/releases/compare/0.6.0..0.7.0
[0.6.0]: https://github.com/milescsmith/scorphan/releases/compare/0.5.1..0.6.0
[0.5.1]: https://github.com/milescsmith/scorphan/releases/compare/0.5.0..0.5.1
[0.5.0]: https://github.com/milescsmith/scorphan/releases/compare/0.4.1..0.5.0
[0.4.1]: https://github.com/milescsmith/scorphan/releases/compare/0.4.0..0.4.1
[0.4.0]: https://github.com/milescsmith/scorphan/releases/compare/0.3.0..0.4.0
[0.3.0]: https://github.com/milescsmith/scorphan/releases/compare/0.2.0..0.3.0
[0.2.0]: https://github.com/milescsmith/scorphan/releases/compare/0.1.0..0.2.0
[0.1.0]: https://github.com/milescsmith/scorphan/releases/tag/v0.0.1
