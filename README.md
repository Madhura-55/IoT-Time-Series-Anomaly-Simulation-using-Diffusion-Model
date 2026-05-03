# IoT Time Series Anomaly Simulation using Diffusion Models on Smart Grid Data

## Overview
This project trains a Denoising Diffusion Probabilistic Model (DDPM) on real-world smart grid electricity consumption data to learn normal consumption patterns, then uses the trained model to simulate realistic anomalous time series. The goal is to generate labeled anomaly data that can be used to train and benchmark anomaly detection systems — a common challenge in industrial IoT where real anomalies are rare.

## Problem Statement
Anomaly detection in smart grid data is hard because real anomalies are rare, unlabeled, and diverse. This project addresses the data scarcity problem by using a diffusion model trained on normal behavior to generate synthetic anomalous sequences across three failure types: point spikes, flat-line dropouts, and gradual drift.

## Dataset
- **Source:** UCI ElectricityLoadDiagrams2011-2014
- **Size:** 140,256 timesteps x 370 clients (15-min intervals, 2011-2015)
- **Selected clients:** 5 clients (MT_154, MT_302, MT_099, MT_093, MT_163) chosen to represent low, mid, high consumption, most volatile, and most stable patterns
- **Bad clients removed:** 43 clients with >50% zero readings

## Pipeline

| Stage | Notebook | Description |
|-------|----------|-------------|
| 1 | `01_Stage1.ipynb` | EDA, client selection, visualization |
| 2 | `02_Stage2.ipynb` | Normalization, sliding windows (192 steps / 48hrs), train/val split |
| 3 | `03_Stage3.ipynb` | DDPM training — 1D U-Net denoiser, 50 epochs |
| 4 | `04_Stage4.ipynb` | Anomaly simulation — spike, flatline, drift injection |
| 5 | `05_Stage5.ipynb` | Evaluation — reconstruction error scoring, detection metrics |

## Model Architecture
A 1D U-Net with residual blocks and time embeddings, trained using the DDPM objective (noise prediction loss) over T=1000 diffusion timesteps with a linear beta schedule.

- **Parameters:** 1,070,721
- **Training windows:** 11,676 (80/20 split)
- **Window size:** 192 timesteps (48 hours)
- **Best validation loss:** 0.0144

## Anomaly Types Simulated

| Type | Description | F1 Score |
|------|-------------|----------|
| Point Spike | 3 random timesteps multiplied by 2.5x | 0.262 |
| Flat-line Dropout | 48 consecutive steps set to zero | 0.658 |
| Gradual Drift | Linear upward drift of +0.3 over 48 hours | 0.441 |

Detection threshold set at the 95th percentile of clean reconstruction scores.

## Key Findings
- Flat-line dropouts are most detectable — a complete signal loss deviates strongly from learned patterns
- Point spikes are hardest to detect — only 3/192 timesteps are affected, so overall reconstruction error barely moves
- The model is client-unaware — low-consumption clients scoring near zero are harder to distinguish from true flatline anomalies
- High precision (0.73-0.90) across all types — when the model flags an anomaly, it is usually correct

## Repo Structure

    notebooks/          Stage-by-stage Colab notebooks
    data/
        raw/            Downloaded at runtime from UCI (not tracked)
        processed/      Normalized windows as .npy files
    artifacts/          Shared outputs between notebooks
    outputs/
        figures/        All visualization outputs
        models/         Trained model weights
    requirements.txt

## Reproducing Results
All notebooks are designed to run sequentially in Google Colab. The raw dataset is downloaded automatically from UCI at runtime — no manual download needed.

1. Open each notebook from GitHub in Colab
2. Run all cells in order (Stages 1 to 5)
3. Each notebook clones the repo, loads artifacts from previous stages, and pushes outputs back to GitHub

## Requirements
See `requirements.txt`. Key dependencies: `torch`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`.
