# IoT-Time-Series-Anomaly-Simulation-using-Diffusion-Model-on-Smart-Grid-Data

Anomaly simulation on the UCI Electricity Load Diagrams dataset
using a Denoising Diffusion Probabilistic Model (DDPM).
        
Stage 1 → EDA, Visualization & Dataset Understanding
Stage 2 → Preprocessing & Train/Test Split
Stage 3 → Time Series Core Processing (Decomposition, Stationarity, etc.)
Stage 4 → Model Selection & Architecture Design (Diffusion Model)
Stage 5 → Training Loop
Stage 6 → Evaluation & Metrics
Stage 7 → Anomaly Simulation (Generation)


What the Pipeline Does — Stage by Stage
Stage 1 — Understand the data
Load the dataset, explore its structure, visualize consumption patterns across clients, identify inactive/faulty meters (43 out of 370), and select 5 representative clients covering low, medium, and high consumption profiles.
Stage 2 — Prepare the data
Normalize each client's values independently using robust scaling so the model can compare patterns across clients fairly. Slice the continuous time series into fixed 2-day windows (192 timesteps each). Split data temporally — 2011 to 2013 for training, 2014 for testing.
Stage 3 — Analyze time series properties
Decompose signals into trend, seasonality, and residual components. Test for stationarity. Analyze repeating daily and weekly patterns. This step confirms what structure the model needs to learn.
Stage 4 — Build the diffusion model
Design a 1D U-Net neural network as the denoising backbone. Define a noise schedule that gradually adds Gaussian noise to normal consumption windows (forward process) and trains the network to reverse that noise (reverse process).
Stage 5 — Train the model
Train the denoiser on normal consumption windows. The model learns what normal electricity patterns look like at every noise level.
Stage 6 — Evaluate the model
Measure how realistic the generated sequences are using distributional metrics — FID, DTW, and MMD. Compare generated sequences visually and statistically against real data.
Stage 7 — Generate anomalies
Condition the generation process to produce sequences that deviate from normal patterns in controlled ways — spikes, flatlines, gradual drifts. These synthetic anomalous sequences are the final deliverable.



## Setup
pip install -r requirements.txt

## Data
Download `LD2011_2014.txt` from https://archive.ics.uci.edu/dataset/321  
Place it in `data/raw/`

## Stages
- 01 — EDA & Visualization
- 02 — 
