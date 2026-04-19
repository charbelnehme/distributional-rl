# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Fixed

- Added project packaging metadata and an explicit dependency list so the
  repository can be installed and its tests can run in a clean environment.
- Added a clear import-time error message when NGBoost is not installed instead
  of failing with an opaque dependency traceback.
- Added a `.gitignore` that excludes the local `.env` file and local Alpaca or
  market-data download directories from version control.
- Replaced the placeholder `metrics.py` and `strategy.py` implementations with
  working risk-metric and position-selection logic.

### Changed

- Replaced the placeholder README with concrete installation, test, and usage
  instructions.
- Replaced synthetic data generation with Alpaca historical-bar retrieval,
  parquet persistence, and feature engineering for model training.
- Added a `setup_env.sh` script to create a local virtual environment and
  install the project dependencies reproducibly.
