# Physics-Informed Neural Networks (PINNs) — Exam Assignment

A comprehensive implementation and comparative study of Physics-Informed Neural Networks (PINNs) for solving two classical PDEs: **damped spring-mass system** and the **Burgers' equation**. This project explores multiple PINN architectures, reverse-mode differentiation, and parameter identification techniques.

## Project Overview

Physics-Informed Neural Networks combine machine learning with physics constraints. This repository implements PINNs to solve:

1. **Damped Spring-Mass System** — A second-order ODE with parameter identification:

   ```math
   m·y''(t) + c·y'(t) + k·y(t) = 0
   ```

   - State characterization via natural frequency (ω₀) and damping ratio (ζ)
   - Parameter recovery: ω₀ and ζ

2. **Burgers' Equation** — A nonlinear partial differential equation commonly used as a benchmark:

   ```math
   ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
   ```

   - Initial conditions: Gaussian, Step or N-wave profiles
   - Fully implicit Crank-Nicolson solver for generating synthetic "truth" data
   - Tests neural network's ability to learn nonlinear dynamics
   - Parameter recovery: viscosity

## Directory Structure

```dir
ExProgrammeren/
├── BurgersPINN/                 # Solving Burgers' equation using PINNS and standard neural network
│   ├── model.py                 # Architectures, BurgerConfig, predictor
│   ├── train.py                 # Training loop with both PINN modes and standard neural network, argument parsing if run
│   ├── Burger_PDE.py            # Crank-Nicolson solver for synthetic data
│   ├── hyperparam_search.py     # Search for optimal hyperparameters
│   ├── plot.py                  # Plot over the epochs, comparison for several time slices and evolution of losses
│   └── Output/                  # Training results and saved models
│
├── BurgersReversePINN/          # Solving the reverse problem for the Burgers' equation
│   ├── model.py                 # PINN architecture, BurgerConfig, predictor
│   ├── train.py                 # Training loop with both PINN modes, argument parsing if run
│   ├── Burger_PDE.py            # Crank-Nicolson solver for synthetic data
│   ├── findParams.py            # Evaluation of the retrieval of the viscosity parameter
│   ├── hyperparam_search.py     # Search for optimal hyperparameters
│   ├── plot.py                  # Plot over the epochs, comparison for several time slices and evolution of losses
│   └── Output/                  # Training results and saved models
│
├── DampedPINN/                  # Solving the damped harmonic oscillator using PINNs and standard neural network
│   ├── model.py                 # Architectures, DamperConfig, predictor
│   ├── train.py                 # Training loop with both PINN modes and standard neural network, argument parsing if run
│   ├── plot.py                  # Plot over the epochs, comparison for several time slices and evolution of losses
│   ├── hyperparam_search.py     # Search for optimal hyperparameters
│   ├── Hyperparam_optim/        # Hyperparameter optimization results
│   └── Output/
│
├── DampedReversePINN/           # Solving the reverse problem for the damped harmonic oscillator
│   ├── model.py                 # PINN architecture, DamperConfig, predictor
│   ├── train.py                 # Training loop with both PINN modes, argument parsing if run
│   ├── findParams.py            # Evaluation of the retrieval of the natural frequency and damping ratio parameters
│   ├── hyperparam_search.py     # Search for optimal hyperparameters
│   ├── plot.py                  # Plot over the epochs, comparison for several time slices and evolution of losses
│   └── Output/                  # Training results and saved models
│
│
├── AnalyticBurger/              # Precomputed analytic solutions for Burgers' eq.
│   ├── Gaussian_*.npz           # Solutions for Gaussian IC with various viscosities
│   ├── N-wave_*.npz             # Solutions for N-wave IC with various viscosities
│   └── Step_*.npz               # Solutions for Step IC with various viscosities
│
├── Original/                    # Original reference implementations
│
├── hyperparam_tracking_burger.md        # Hyperparameter search logs & results for burgers' equation
├── hyperparam_tracking_damper.md        # Hyperparameter search logs & results for damped harmonic oscillator
├── Questions.md                         # Research questions & investigation notes
├── tips.txt                             # Best practices & implementation notes
└── README.md                            # This file
```

## Key Features

### 1. **Dual Physics Applications**

- **Damped System**: Solves a simple ODE with known analytical solution
- **Burgers' Equation**: Solves a nonlinear PDE with spatial-temporal structure

### 2. **Training Modes**

- **Standard ML**: Data loss only
- **PINN (Blind)**: Data + physics residual (on training region) + initial condition losses
- **PINN (Extended Physics)**: Physics-residual extended to the extrapolated region

### 3. **Automatic Differentiation**

- Uses PyTorch for computing derivatives (∂u/∂t, ∂u/∂x, ∂²u/∂x², etc.)
- Efficient reverse-mode AD for loss backpropagation

### 4. **Hyperparameter Optimization**

- Learning rate optimization (Adam optimizer with StepLR decay)
- Grid search for loss weights (λ_physics, λ_ic)
- Collocation point density tuning
- Best hyperparameters logged and tracked

### 5. **Robust Training Infrastructure**

- Early stopping based on validation loss
- Checkpoint saving (best models preserved)
- Device auto-detection (CUDA > MPS > CPU)

## Getting Started

### Requirements

- Python 3.8+
- PyTorch (CUDA-enabled recommended)
- NumPy, Matplotlib, pathlib

### Usage

Note: the paths of input and output files are all configurable by either directly modifying the config class in model.py or by adding it as an argument (e.g. --out_dir changes the output directory).

The default configuration works only if your working folder is the project folder.

```bash

# Navigate to the project directory
cd ExProgrammeren

# Install dependencies (if needed)
pip install torch numpy scipy matplotlib
```

### Quick Start

#### Training on Burgers' Equation

```bash
# Train with default parameters
python BurgersPINN/train.py

# Train with custom parameters
python BurgersPINN/train.py --n_epochs 5000 --lr 0.001 --lambda_phys 1.0 --lambda_ic 50.0

# Specify initial condition type
python BurgersPINN/train.py --situation "Gaussian"  # or "N-wave"

# Visualize results
python BurgersPINN/plot.py
```

#### Training on Damped System

```bash
# Train for a single damping/stiffness pair
python DampedPINN/train.py --mass 1.0 --damping 2.5 --stiffness 3.0

# Visualize results
python DampedPINN/plot.py
```

## Core Modules

### `model.py`

Defines the neural network architecture, configuration dataclass, and prediction wrapper.

- **FCNet**: Fully-connected neural network with configurable depth/width
  - Input: time (for damped) or time+space (for Burgers')
  - Hidden layers: Tanh activation, important for backpropagation of derivatives
  - Output: solution
  
- **Config Dataclass** (BurgerConfig / DamperConfig):
  - Physical parameters (viscosity ν, mass m, damping c, stiffness k)
  - Data generation settings (noise levels, sampling density)
  - Architecture choices (hidden layers, neurons)
  - Training hyperparameters (learning rate, loss weights)
  
- **Predictor**: Wraps the neural network for batch inference

### `train.py`

Main training loop implementing the PINN framework.

**Workflow:**

1. Parse CLI arguments into Config object
2. Generate synthetic training/validation data
3. Train Standard-ML model (data loss only) (only possible in normal case)
4. Train PINN models (data + physics + IC losses)
5. Apply early stopping and checkpoint management
6. Save results to `training_results.pt` for visualization

**Loss Functions:**

```math
L_total = L_data + λ_phys · L_physics + λ_ic · L_ic
```

Where:

- `L_data` = MSE between predictions and observations
- `L_physics` = MSR of PDE/ODE residual at collocation points
- `L_ic` = MSE of initial conditions

### `Burger_PDE.py` (Burgers' Equation only)

Fully implicit Crank-Nicolson solver for generating ground-truth synthetic data.

- Nonlinear solver using Newton's method
- Exact Jacobian construction for efficiency
- Upwind finite-difference advection stencil
- Central-difference diffusion stencil

### `findParams.py`

Parameter identification and model evaluation.

**For Damper System:**

- Evaluates PINN predictions for multiple damping/stiffness combinations
- Recovers estimates of ω₀ and ζ
- Generates comparison plots (true vs. predicted parameters)

**For Burgers:**

- Evaluates PINN predictions for multiple ICs and viscosities
- Recovers estimates of ν
- Generates comparison plots (true vs. predicted parameters)

### `hyperparam_search.py`

Search for hyperparameter optimization.

- Grid search over parameter ranges
- Tracks RMSE on training/validation splits
- Logs best configurations to markdown files
- Enables reproducible hyperparameter tuning

### `plot.py`

Visualization module for results.

**Features:**

- Overlay of PINN predictions vs. observations
- Heatmaps of spatial-temporal solutions
- Epoch evolution portraits
- Error visualizations (residuals, prediction errors)
- Parameter estimation plots (true vs. recovered)

## Hyperparameters

See `hyperparam_tracking_burger.md` and `hyperparam_tracking_damper.md` for detailed search logs.
