# Supporting info for paper Optimization of gradient pulse shape prediction using Temporal Convolutional Networks

## Quick Map

| Path | Purpose |
| --- | --- |
| `Figures_MeasuredVsTCN.py` | Main evaluation script for measured vs theoretical/GIRF/TCN gradient and trajectory predictions. Exports testing-set RMSE/NRMSE tables, example plots, and NRMSE boxplots. |
| `trainTCNmodel.py` | Trains the current full-waveform TCN models for gradient axes and B0 terms. |
| `Table_BenchmarkTCNPredictionSpeed.py` | Benchmarks prediction speed of current vs original TCN models and writes a LaTeX table. |
| `Figures_BallExperiment.py` | Reconstructs and compares radial ball-phantom images using theoretical, measured, GIRF, TCN, and B0-corrected trajectories. |
| `Figures_InputTestShapes.py` | Generates the input waveform overview used in the paper. |
| `Figures_ResponseComparison.py` | Generates transfer-response comparison figures. |
| `Figures_PreemphasisEddyCurrents.py` | Fits and plots gradient eddy-current/preemphasis response. |
| `Figures_B0EddyCurrents.py` | Fits and plots B0 eddy-current response. |
| `_tuneup_*.py` | Tune-up analysis scripts used to inspect ramp, preemphasis, and B0 correction behavior for the paper workflow. |
| `_check_*.py` | Check/inspection scripts for transfer-function and preemphasis settings used around the paper workflow. |
| `nn_models/` | TCN architectures, dataset generation, and training losses. |
| `utils/` | Bruker file readers, GIRF correction, ML correction, trajectory generation, BART helpers, and stored system response data. |
| `waveform_generation/` | Bruker method-side C++ code used to generate the test gradient waveform shapes. |
| `paper2026/` | Generated paper figures/tables/CSV metrics. |

## Environment

The project is configured with Poetry in `pyproject.toml`.

```bash
poetry install
poetry run python Figures_MeasuredVsTCN.py
```

Important notes:

- PyTorch wheels are configured from the CUDA 12.8 PyTorch index in `pyproject.toml`.
- `Figures_BallExperiment.py` requires BART 1.0, specifically the v1.0.00 release: https://codeberg.org/mrirecon/bart/releases/tag/v1.0.00. If a `bart` command is available on `PATH`, the script uses that installation.
- Trained models are expected in `utils/gradModels/`.


## Training Current TCN Models

Run:

```bash
poetry run python trainTCNmodel.py
```

What it does:

- Builds datasets from the paper data configured under `trainingData` in `config.json`.
- Trains `X`, `Y`, `Z`, `XB0`, `YB0`, and `ZB0` models.
- Uses model/training settings from `config.json`.
- Saves checkpoints to `utils/gradModels/`.

Core implementation files:

- `nn_models/dataset.py`: converts Bruker gradient-map scans into tensors.
- `nn_models/TCN.py`: TCN architectures, including `TCNFull` and `TCNFullSkip`.
- `nn_models/training.py`: train/validation split, loss computation, checkpoint saving.
- `nn_models/components.py`: differentiable integration/loss helper functions.

## Prediction Speed Benchmark

Run:

```bash
poetry run python Table_BenchmarkTCNPredictionSpeed.py
```

This compares:

- current full-waveform TCN
- original sliding-window TCN architecture
- original TCN with 4x window scale

The benchmark settings are fixed in `Table_BenchmarkTCNPredictionSpeed.py`, and the LaTeX output is written to `paper2026/TCNPredictionSpeedTable.tex`.

## Figure Scripts

| Script | Paper output focus |
| --- | --- |
| `Figures_MeasuredVsTCN.py` | Gradient/trajectory comparison, RMSE/NRMSE metrics, NRMSE boxplots. |
| `Figures_InputTestShapes.py` | Input waveform overview. |
| `Figures_ResponseComparison.py` | System transfer-response figure from measured gradient-map data. |
| `Figures_PreemphasisEddyCurrents.py` | Preemphasis/eddy-current fit and residual visualization. |
| `Figures_B0EddyCurrents.py` | B0 eddy-current fit and residual visualization. |
| `Figures_BallExperiment.py` | Radial ball-phantom reconstruction comparison and SSIM summary. |

Most figure scripts write `.png` and `.pdf` files to `paper2026/`.

## Tune-Up Scripts

| Script | Purpose |
| --- | --- |
| `_tuneup_preemphasiscorrection.py` | Preemphasis correction fit and residual inspection. |
| `_tuneup_B0correction.py` | B0 crossterm fit and residual inspection. |

## Check Scripts

| Script | Purpose |
| --- | --- |
| `_check_getSystemTransfer.py` | Transfer-function inspection and export. |

## Waveform Generation Code

`waveform_generation/` contains the Bruker method-side C++ waveform-design code used to create the gradient test shapes measured in the paper. `shapeDesign.cc/.h` define the shape families such as chirp, ramp, triangle, readout, EPI, MGE, PRGW, trapz series, rose, and spiral; `spiral.cc/.h` contains the analytic spiral helper.

## Runtime Correction Utilities

| File | Purpose |
| --- | --- |
| `utils/GradientCorrector.py` | Classical/GIRF-based correction and trajectory generation. |
| `utils/GradientCorrectorML.py` | Loads trained TCN models and generates ML-corrected trajectories/B0 terms. |
| `utils/BrukerMRI.py` | Bruker parameter/raw-data readers. |
| `utils/utils.py` | Shared data-processing utilities, including gradient-map extraction. |
| `utils/modelHelper.py` | Small helper for loading trained TCN models. |
| `utils/gradSystemInfo.json` | Stored measured system response/GIRF-style information. |

For applied use, `GradientCorrectorML` is the main entry point for generating TCN-corrected trajectories from Bruker method/acqp data.

## Configuration

`config.json` contains the model and training settings used by the scripts:

- model family: `TCNSkip`
- number of channels/layers
- kernel size
- dropout
- shift in samples
- output resolution
- training loss settings
- training/evaluation dataset paths, scan lists, and shape-held-out testing split under `trainingData`

## Generated Files

The repository can create intermediate outputs during local runs:

- `caloutputs/`, `images/`: generated outputs or local reconstruction/diagnostic artifacts.
- `paperData/`: local extraction of the Zenodo paper dataset.
- `trainX.png`, `trainY.png`, etc.: training-progress plots.

## Reproducibility Notes

To fully regenerate the paper outputs, a user needs:

1. The Zenodo paper dataset extracted into `paperData/` in the project root.
2. Trained checkpoints in `utils/gradModels/`, or the ability to retrain them with `trainTCNmodel.py`.
3. BART v1.0.00 installed and configured for `Figures_BallExperiment.py`: https://codeberg.org/mrirecon/bart/releases/tag/v1.0.00.
4. The Python/Poetry environment from `pyproject.toml`.

## Paper Data

Download the paper dataset from Zenodo:

https://doi.org/10.5281/zenodo.21283064

Extract it into the project root as:

```text
paperData/
  high_snr_response/
  before_adjustments_response_b0_preemphasis/
  preemphasis_tuneup/
  tcn_training_testing/
  after_adjustments_response/
  radial_ball_phantom/
```

## Acknowledgment

This repository also contains code adapted from the original TCN gradient-system modeling work by Johnatan B. Martin and coauthors:

J. B. Martin, H. E. Alderson, J. C. Gore, M. D. Does, and K. D. Harkins, "Modeling the MRI gradient system with a temporal convolutional network: Improved reconstruction by prediction of readout gradient errors," Magnetic Resonance in Medicine 95, no. 1 (2026): 286-298, https://doi.org/10.1002/mrm.70044.
