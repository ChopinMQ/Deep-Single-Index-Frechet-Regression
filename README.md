# DeSI Code Submission

This repository contains the implementation and simulation code for the paper **Deep Single-Index $\mathrm{Fr\acute{e}chet}$ Regression**, accepted by the Forty-Third International Conference on Machine Learning (ICML 2026). The code covers several response spaces:

- SPD covariance matrices with log-Cholesky geometry
- Network or graph Laplacian matrices with Frobenius geometry
- Distributional responses represented by quantile functions
- Emotional composition data represented on the unit sphere after square-root transformation

The main simulation folders follow a common structure:

- `DeSI.py` contains the core local regression routine for the response space.
- `simu.py` contains neural-network single-index training and evaluation code.
- Generator files create synthetic data for the corresponding simulation setting.

## Repository Structure

```text
code_submission/
|-- simulation_SPD/
|   |-- DeSI.py
|   |-- generate_matrices.py
|   `-- simu.py
|-- simulation_network/
|   |-- DeSI.py
|   |-- generate_lap.py
|   `-- simu.py
|-- simulation_distribution/
|   |-- DeSI.py
|   |-- generate_dist.py
|   `-- simu.py
`-- Emotional/
    |-- application.py
    |-- loc_sphere.py
    `-- nn_loc_composition.py
```

## Main Methods

The DeSI neural-network model is implemented in the `ThetaMLP` classes inside each simulation or application driver. These models estimate the single-index direction used by the response-specific local $\mathrm{Fr\acute{e}chet}$ regression step.

- SPD simulation: `simulation_SPD/simu.py`
  - `ThetaMLP`: neural network for estimating the SPD single-index direction.
  - `run_single_sim(seed)`: trains the model and evaluates one simulation replicate.

- Network simulation: `simulation_network/simu.py`
  - `ThetaMLP`: neural network for estimating the network single-index direction.
  - `run_single_sim(seed)`: trains the model and evaluates one simulation replicate.

- Distribution simulation: `simulation_distribution/simu.py`
  - `ThetaMLP`: neural network for estimating the distribution single-index direction.
  - `run_single_sim(seed)`: trains the model and evaluates one simulation replicate.

- Emotional composition application: `Emotional/nn_loc_composition.py`
  - `ThetaMLP`: neural network for estimating the spherical-composition single-index direction.
  - `train_compositional_model(...)`: trains the neural DeSI model for emotional composition responses.

## Common Workflow

The simulation scripts share the same high-level workflow:

1. Generate synthetic predictors and responses.
2. Split data into training, validation, and test sets.
3. Standardize predictors using training statistics.
4. Train `ThetaMLP` to estimate a normalized single-index direction.
5. Learn or use a global bandwidth.
6. Use the corresponding `DeSI` local regression routine for response prediction.
7. Report test-set prediction error using the geometry appropriate for the response space.

## Main Dependencies

The code uses:

- `numpy`
- `torch`
- `scipy`
- `pandas`
- `cvxpy`
- `cvxpylayers`
- `matplotlib`
- `joblib`

Some scripts are computationally expensive because they train neural networks and repeatedly solve local regression problems. The core functions are CPU-compatible, and the simulation drivers generally set seeds for reproducibility.

## Naming Convention

The final naming convention uses one `DeSI.py` module per simulation type:

- `simulation_SPD/DeSI.py`: `DeSI_SPD`
- `simulation_network/DeSI.py`: `DeSI_net`
- `simulation_distribution/DeSI.py`: `DeSI_distribution`

This keeps the public-facing API consistent while preserving response-specific implementations.
