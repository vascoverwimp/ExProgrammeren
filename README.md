# Physics-Informed Neural Networks (PINNs) — Comparative Study

A comprehensive implementation and comparative study of Physics-Informed Neural Networks (PINNs) for solving two classical PDEs: **damped spring-mass system** and the **Burgers' equation**. This project explores multiple PINN architectures, reverse-mode differentiation, and parameter identification techniques.

## Project Overview

Physics-Informed Neural Networks combine machine learning with physics constraints. This repository implements PINNs to solve:

1. **Damped Spring-Mass System** — A second-order ODE with parameter identification:
   ```
   m·y''(t) + c·y'(t) + k·y(t) = 0
   ```
   - State characterization via natural frequency (ω₀) and damping ratio (ζ)
   - Parameter recovery: ω₀ and ζ

2. **Burgers' Equation** — A nonlinear partial differential equation commonly used as a benchmark:
   ```
   ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
   ```
   - Initial conditions: Gaussian, Step or N-wave profiles
   - Fully implicit Crank-Nicolson solver for generating synthetic "truth" data
   - Tests neural network's ability to learn nonlinear dynamics
   - Parameter recovery: viscosity



## Directory Structure

```
ExProgrammeren/
├── VascoReverseBurger/          # Burgers' equation with reverse-mode differentiation
│   ├── model.py                 # PINN architecture and BurgerConfig
│   ├── train.py                 # Training loop with ML/PINN modes
│   ├── Burger_PDE.py            # Crank-Nicolson solver for synthetic data
│   ├── findParams.py            # Hyperparameter search utilities
│   ├── hyperparam_search.py     # Grid/random search for optimal parameters
│   ├── plot_claude.py           # Visualization of predictions vs. ground truth
│   └── Output/                  # Training results and saved models
│
├── VascoReversePINNDamper/      # Damped system with reverse-mode differentiation
│   ├── model.py                 # PINN architecture and DamperConfig
│   ├── train.py                 # Training loop (blind PINN, ext phys PINN)
│   ├── findParams.py            # Parameter recovery and predictions
│   ├── hyperparam_search.py     # Hyperparameter optimization
│   ├── plot_claude.py           # Phase portraits and parameter estimates
│   └── Output/                  # Training results and saved models
│
├── VascoPINNDamper/             # Original implementation for damped system
│   ├── model.py
│   ├── train.py
│   ├── plot_claude.py
│   ├── Hyperparam_optim/        # Hyperparameter optimization results
│   └── Output/
│
├── VascoVersionBurger/          # Alternative PINN implementation for Burgers' eq.
│   └── [similar structure]
│
├── AnalyticBurger/              # Precomputed analytic solutions for Burgers' eq.
│   ├── Gaussian_*.npz           # Solutions for Gaussian IC with various viscosities
│   ├── N-wave_*.npz             # Solutions for N-wave IC with various viscosities
│   └── [many more configurations]
│
├── Original/                    # Original reference implementations
│
├── hyperparam_tracking_burger.md        # Hyperparameter search logs & results
├── hyperparam_tracking_damper.md        # Hyperparameter search logs & results
├── Questions.md                         # Research questions & investigation notes
├── tips.txt                             # Best practices & implementation notes
└── README.md                            # This file
```

## Key Features

### 1. **Dual Physics Applications**
   - **Burgers' Equation**: Solves a nonlinear PDE with spatial-temporal structure
   - **Damped System**: Identifies unknown physical parameters (damping, stiffness)

### 2. **Training Modes**
   - **Standard ML**: Data loss only
   - **PINN (Blind)**: Data + physics residual + initial condition losses
   - **PINN (Extended Physics)**: Physics-informed with additional constraints

### 3. **Automatic Differentiation**
   - Uses PyTorch for computing derivatives (∂u/∂t, ∂u/∂x, ∂²u/∂x², etc.)
   - Second-order derivatives via automatic differentiation
   - Efficient reverse-mode AD for loss backpropagation

### 4. **Hyperparameter Optimization**
   - Grid search for loss weights (λ_physics, λ_ic)
   - Learning rate optimization (Adam optimizer with StepLR decay)
   - Collocation point density tuning
   - Best hyperparameters logged and tracked

### 5. **Robust Training Infrastructure**
   - Early stopping based on validation loss
   - Checkpoint saving (best models preserved)
   - Mixed-precision training support (AMP)
   - Device auto-detection (CUDA > MPS > CPU)

## Getting Started

### Requirements
- Python 3.8+
- PyTorch (CUDA-enabled recommended)
- NumPy, SciPy, Matplotlib
- See individual directories for specific versions

### Installation

```bash
# Navigate to the project directory
cd ExProgrammeren

# Install dependencies (if needed)
pip install torch numpy scipy matplotlib

# Or use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Quick Start

#### Training on Burgers' Equation

```bash
cd VascoReverseBurger

# Train with default parameters
python train.py

# Train with custom parameters
python train.py --n_epochs 5000 --lr 0.001 --lambda_phys 1.0 --lambda_ic 50.0

# Specify initial condition type
python train.py --situation "Gaussian"  # or "N-wave"

# Visualize results
python plot_claude.py
```

#### Training on Damped System

```bash
cd VascoReversePINNDamper

# Train for a single damping/stiffness pair
python train.py --mass 1.0 --damping 2.5 --stiffness 3.0

# Perform parameter recovery across a grid
python findParams.py

# Hyperparameter optimization
python hyperparam_search.py
```

## Core Modules

### `model.py`
Defines the neural network architecture, configuration dataclass, and prediction wrapper.

- **FCNet**: Fully-connected neural network with configurable depth/width
  - Input: time (for ODE) or time+space (for PDE)
  - Hidden layers: ReLU activation (configurable)
  - Output: solution or solution+derivatives
  
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
3. Train Standard-ML model (data loss only)
4. Train PINN models (data + physics + IC losses)
5. Apply early stopping and checkpoint management
6. Save results to `training_results.pt` for visualization

**Loss Functions:**
```
L_total = L_data + λ_phys · L_physics + λ_ic · L_ic
```

Where:
- `L_data` = MSE between predictions and observations
- `L_physics` = MSE of PDE/ODE residual at collocation points
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

### `hyperparam_search.py`
Automated hyperparameter optimization.

- Grid or random search over parameter ranges
- Tracks RMSE on training/validation splits
- Logs best configurations to markdown files
- Enables reproducible hyperparameter tuning

### `plot_claude.py`
Visualization module for results.

**Features:**
- Overlay of PINN predictions vs. observations
- Heatmaps of spatial-temporal solutions
- Phase portraits (for damper system)
- Error visualizations (residuals, prediction errors)
- Parameter estimation plots (true vs. recovered)

## Configuration & Hyperparameters

### Training Hyperparameters (from tracking files)

**Burgers' Equation (N-wave, Gaussian ICs):**
- **Learning Rate**: 0.0018 (optimized via grid search)
- **Lambda Physics**: 0.794 (physics loss weight)
- **Lambda IC**: 1.259 (initial condition loss weight)
- **Collocation Points**: 200–500 (adaptive per configuration)
- **Epochs**: 40,000
- **Early Stopping Patience**: 30 (validation check intervals)

**Damped System:**
- Varies across mass, damping, stiffness combinations
- Typically: λ_phys ∈ [0.1, 1.0], λ_ic ∈ [10, 100]

See `hyperparam_tracking_burger.md` and `hyperparam_tracking_damper.md` for detailed search logs.

## Baseline Results

### Burgers' Equation
- **Base RMSE** (initial): 0.1248
- **After Learning Rate Optimization**: 0.012209
- **After Weight Optimization**: 0.004675

### Damped System
- Recovers ω₀ and ζ to ~4 significant figures with sufficient training data
- Physics-informed PINN outperforms blind ML, especially with limited observations

## Research Questions & Investigation

See `Questions.md` for active research topics:
1. Loss function design and weighted optimization
2. Physics residual scaling with ω₀
3. Role of initial condition enforcement
4. Network architecture impact on extrapolation
5. Trade-offs between data fit and physics satisfaction

## Best Practices (from `tips.txt`)

- **Data Preprocessing**: Split data → preprocess (never preprocess → split to avoid data leakage)
- **Model Architecture**: Store architecture and predictor in separate files
- **Configuration Management**: Use dataclass for all parameters (no magic numbers)
- **Checkpointing**: Save best models periodically; never lose progress to crashes
- **Early Stopping**: Monitor validation loss; stop if no improvement for N iterations
- **Data Consistency**: Arrange synthetic data consistent with real-world scenarios

## Performance Considerations

- **Compute Device**: Auto-selects GPU (CUDA/MPS) if available; falls back to CPU
- **Mixed Precision Training**: Enabled for faster convergence with lower memory
- **Batch Processing**: Collocation points and observations handled in configurable batches
- **Scalability**: Tested up to 10,000 collocation points and 1,000 observations

## Troubleshooting

| Issue | Solution |
|-------|----------|
| NaN losses | Reduce learning rate; check physics loss scaling |
| Poor extrapolation | Increase λ_phys; add more collocation points beyond training domain |
| Training plateau | Tune learning rate decay (lr_step, lr_gamma); increase network depth |
| Out of memory | Reduce batch sizes; decrease hidden layer width; use CPU |

## Citation & References

This project is part of a comparative study on Physics-Informed Neural Networks for educational purposes. References:

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems.
- Han et al. (2018). Solving high-dimensional PDEs using deep learning.

## License

This project is part of academic coursework. Modify and use freely for educational purposes.

## Contact & Questions

For questions or issues:
- Review existing research notes in `Questions.md`
- Check hyperparameter logs in `hyperparam_tracking_*.md`
- Refer to docstrings in individual modules

---

**Last Updated**: May 2026  
**Project Status**: Active development & experimentation
