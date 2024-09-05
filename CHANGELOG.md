# scorphan

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

[0.3.0]: https://github.com/milescsmith/scorphan/releases/compare/0.5.0..0.5.1
[0.3.0]: https://github.com/milescsmith/scorphan/releases/compare/0.4.1..0.5.0
[0.3.0]: https://github.com/milescsmith/scorphan/releases/compare/0.4.0..0.4.1
[0.3.0]: https://github.com/milescsmith/scorphan/releases/compare/0.3.0..0.4.0
[0.3.0]: https://github.com/milescsmith/scorphan/releases/compare/0.2.0..0.3.0
[0.2.0]: https://github.com/milescsmith/scorphan/releases/compare/0.1.0..0.2.0
[0.1.0]: https://github.com/milescsmith/scorphan/releases/tag/v0.0.1
