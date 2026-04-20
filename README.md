# IoT-Time-Series-Anomaly-Detection-using-Diffusion-Model-on-Smart-Grid-Data

complete Pipeline:

Raw UCI Smart Grid Data
        ↓
Stage 1 — EDA & Visualization
        ↓
Stage 2 — Preprocessing & Feature Engineering
        ↓
Stage 3 — Core Time Series Processing (Decomposition, Stationarity)
        ↓
Stage 4 — Train/Test Split (time-aware)
        ↓
Stage 5 — Anomaly Simulation (inject synthetic anomalies)
        ↓
Stage 6 — Diffusion Model Architecture & Training
        ↓
Stage 7 — Anomaly Detection & Evaluation
        ↓
Stage 8 — End-to-end Pipeline + Results


# IoT Time Series Anomaly Detection with Diffusion Models

Anomaly simulation and detection on the UCI Electricity Load Diagrams dataset
using a Denoising Diffusion Probabilistic Model (DDPM).

## Setup
pip install -r requirements.txt

## Data
Download `LD2011_2014.txt` from https://archive.ics.uci.edu/dataset/321  
Place it in `data/raw/`

## Stages
- 01 — EDA & Visualization
- 02 — Preprocessing (coming soon)
