# Physics-Informed Neural Networks (PINNs)

**Author:** Des De Borger  
**Email:** des.deborger@student.uantwerpen.be  
**Last modified:** May 2026

This project applies Physics-Informed Neural Networks (PINNs) to two classical problems in mathematical physics:

1. **Damped Harmonic Oscillator** — a second-order ODE
2. **Burgers' Equation** — a nonlinear PDE describing convection and diffusion

Both forward problems (predict the solution given known parameters) and inverse problems (recover unknown physical parameters from noisy observations) are implemented.

---

## Project structure

```
.
├── Damped_Oscillator/
│   ├── config.py       # Physical constants, hyperparameters, runtime settings
│   ├── data.py         # Analytic solution and data generation
│   ├── model.py        # FCNet and InverseFCNet architectures
│   ├── losses.py       # Loss functions (data, physics, initial condition)
│   ├── trainer.py      # Training loop with early stopping and snapshots
│   ├── plot.py         # All plotting functions
│   ├── utils.py        # Helper utilities (device, tensors, save/load)
│   └── scripts/        # Example scripts to train PINNs
│
└── Burgers_Equation/
    ├── config.py       # Physical constants, hyperparameters, runtime settings
    ├── analytic.py     # Ground truth solvers (Cole-Hopf, Euler, Lax-Wendroff)
    ├── data.py         # Data generation (observations, collocation, validation)
    ├── model.py        # FCNet and InverseFCNet architectures
    ├── losses.py       # Loss functions (data, physics, IC, boundary condition)
    ├── trainer.py      # Training loop with early stopping and snapshots
    ├── plot.py         # All plotting functions
    ├── utils.py        # Helper utilities (device, tensors, save/load)
    └── scripts/        # Example scripts to train PINNs
```

Each module is self-contained and importable. Training experiments are written as separate scripts that import from these modules.

---

## Requirements

```
torch
numpy
matplotlib
scikit-learn
pandas
tqdm
optuna
```

Install with:

```bash
pip install torch numpy matplotlib scikit-learn pandas tqdm optuna
```

A CUDA-capable GPU is supported but not required. The code automatically selects CUDA > MPS > CPU.

---

## Chapter 1 — Damped Harmonic Oscillator

### Physics

The model solves the second-order linear ODE:

$$my''(t) + cy'(t) + ky(t) = 0$$

with initial conditions $y(0) = y_0$ and $y'(0) = dy_0$. The behaviour depends on the damping ratio $\zeta = c / (2\sqrt{mk})$ and natural frequency $\omega_0 = \sqrt{k/m}$:

| Regime | Condition | Solution form |
|---|---|---|
| Overdamped | $\zeta > 1$ | Sum of two real exponentials |
| Critically damped | $\zeta = 1$ | $(A + Bt)e^{-\omega_0 t}$ |
| Underdamped | $\zeta < 1$ | Exponentially decaying sinusoid |

### File descriptions

#### `config.py`
Defines the `Config` dataclass with all parameters. Physical parameters `m`, `c`, `k` are stored directly; `omega_0`, `zeta`, and `omega_d` are computed as `@property` fields. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `m`, `c`, `k` | `1.0`, `0.5`, `4.0` | Physical constants |
| `t_dom` | `6.0` | End of the training domain |
| `t_extrap` | `10.0` | End of the extrapolation domain |
| `n_obs` | `15` | Noisy observation points per epoch |
| `n_col_dom` | `500` | Collocation points in the training domain |
| `hidden` | `64` | Nodes per hidden layer |
| `n_layers` | `5` | Number of hidden layers |
| `lr` | `5e-3` | Initial learning rate |
| `lambda_phys` | `1e-2` | Weight for the physics loss term |
| `lambda_ic` | `10` | Weight for the initial condition loss term |
| `patience` | `1000` | Early stopping patience (epochs) |

#### `data.py`
Generates all data required for training:

- `analytic(t, cfg)` — closed-form solution for $y(t)$ supporting all three damping regimes.
- `make_train_observation(cfg)` — generates `n_obs` noisy observation points drawn uniformly from $(0.1,\ t_{dom})$.
- `make_validation(cfg)` — generates a dense, evenly spaced validation set on the training domain.
- `make_collocation(cfg)` — generates collocation points on the training domain and (optionally) the extrapolation domain.
- `generate_data(cfg)` — pipeline that calls all of the above and returns a single `dict`.

#### `model.py`
Two network architectures are defined:

- **`FCNet`**: maps $t \to y(t)$. Architecture: `1 → [hidden × n_layers] → 1` with `Tanh` activations and optional dropout. `Tanh` is required (not `ReLU`) because the network output is differentiated twice via `autograd`.
- **`InverseFCNet`**: same architecture as `FCNet` but adds learnable scalar parameters `zeta_hat` and `omega_0_hat` that are optimised alongside the network weights.

Helper functions:

- `predict(model, t)` — runs inference on a numpy array of times, returns a numpy array.
- `predict_from_state(state_dict, t, cfg)` — restores a saved state dict and predicts. Automatically detects whether the state dict belongs to a forward or inverse model.

#### `losses.py`
Three loss functions, each returning a scalar `torch.Tensor`:

- **`loss_data`** — MSE between predicted and observed $y$ values:
$$\mathcal{L}_{data} = \frac{1}{M}\sum_{j=1}^{M}(\hat{y}(t_j) - y_j)^2$$

- **`loss_physics`** — mean squared ODE residual at collocation points. Both $\hat{y}'$ and $\hat{y}''$ are computed with `torch.autograd.grad` using `create_graph=True` so the second derivative can be backpropagated through:
$$\mathcal{L}_{physics} = \frac{1}{N}\sum_{i=1}^{N}(m\hat{y}''(t_i) + c\hat{y}'(t_i) + k\hat{y}(t_i))^2$$

- **`loss_physics_inverse`** — same as above but uses `model.zeta_hat` and `model.omega_0_hat` instead of fixed `cfg.c` and `cfg.k`.

- **`loss_ic`** — squared error on both initial conditions, evaluated at $t=0$ using `autograd` for $\hat{y}'(0)$:
$$\mathcal{L}_{ic} = (\hat{y}(0) - y_0)^2 + (\hat{y}'(0) - dy_0)^2$$

The total loss is:
$$\mathcal{L}_{tot} = \mathcal{L}_{data} + \lambda_{phys}\mathcal{L}_{physics} + \lambda_{ic}\mathcal{L}_{ic}$$

#### `trainer.py`
The `train()` function runs the full training loop. Key features:

- **Automatic device selection** — model and tensors are moved to the best available device.
- **Inverse mode detection** — checks for `zeta_hat` attribute; if present, uses separate learning rates for the network weights (`lr`) and the physical parameters (`lr_inverse`).
- **Randomised training data** — observation and collocation points are regenerated every epoch if `cfg.randomise_observation` / `cfg.randomise_collocation` is `True`.
- **Early stopping** — training halts when the validation loss has not improved by more than `patience_threshold` for `patience` consecutive epochs.
- **Snapshots** — the model state dict is saved at epochs listed in `cfg.snapshot_epochs`.
- **Optuna integration** — pass an `optuna.Trial` object to enable pruning of unpromising hyperparameter trials.

Returns `(history, snapshots, best_state)`:
- `history`: dict of loss curves logged every `log_every` epochs.
- `snapshots`: dict mapping epoch number to CPU state dict.
- `best_state`: state dict corresponding to the lowest validation loss seen during training.

#### `utils.py`
Shared helper functions:

- `get_device()` — returns the best available `torch.device`.
- `set_seed(seed)` — sets seeds for `random`, `numpy`, and `torch`.
- `to_tensor(arr, requires_grad, unsqueeze)` — converts a numpy array to a `(N, 1)` float32 tensor on the active device.
- `rmse(pred, true)` — computes root mean squared error.
- `save_model(best_state, history, snapshots, cfg, output_path)` — saves model weights (`.pt`), loss history (`.csv`), snapshots (`.pt`), and config (`.json`) to a folder.
- `load_model(model, folder_path)` — restores a model and all associated data from a saved folder.
- `load_best_cfg(json_path)` — loads the best Optuna trial parameters from a JSON file as a `Config` object.

---

## Chapter 2 — Burgers' Equation

### Physics

The viscid Burgers' equation is:

$$u_t + u u_x = \nu u_{xx}$$

where $\nu$ is the kinematic viscosity. The solution $u(x, t)$ represents a 1D velocity field. The nonlinear advection term causes wave steepening and eventual shock formation; the viscosity term prevents true discontinuities from forming.

Three initial conditions are supported: Gaussian, N-wave, and step functions. The shock formation time can be estimated analytically via the method of characteristics as $t_{shock} = -1/\min_x u_x(x, 0)$.

### File descriptions

#### `config.py`
Similar structure to the DHO config. Key additional parameters:

| Parameter | Default | Description |
|---|---|---|
| `nu` | `0.05` | Kinematic viscosity |
| `ic` | `"Gauss"` | Initial condition type |
| `L` | `15` | Spatial domain length |
| `n_x` | `1000` | Spatial grid resolution |
| `t_dom` | `5.0` | End of the training domain |
| `t_extrap` | `7.0` | End of the extrapolation domain |
| `n_col_dom` | `200` | Collocation points |
| `n_bc` | `50` | Boundary condition points per epoch |
| `lambda_phys` | `5` | Weight for physics loss |
| `lambda_ic` | `10` | Weight for IC loss |
| `lambda_bc` | `10` | Weight for boundary condition loss |

`__post_init__` automatically sets IC-specific defaults (amplitude, width, boundary values) based on the `ic` field.

Supported initial conditions:

| `ic` | Shape |
|---|---|
| `"Gauss"` | Gaussian bell curve centred at $x = L/2$ |
| `"N_wave"` | Antisymmetric linear ramp (forms a bilateral shock) |
| `"N_wave_chop"` | Half N-wave rearranged to create a single steep front |
| `"Step_up"` | Step from 0 to `height` at $x = L/2$ (rarefaction, no shock) |
| `"Step_down"` | Step from `height` to 0 at $x = L/2$ |
| `"Slope"` | Linear decrease to zero over a finite width |

#### `analytic.py`
Provides ground truth solutions via three strategies:

**Cole-Hopf transformation** (primary method, used for training data):  
The viscid Burgers' equation is transformed to the heat equation via $u = -2\nu \phi_x / \phi$. The heat equation is solved by convolving the initial condition $\phi_0$ with the Gaussian heat kernel. A padding region (800 cells for step-like ICs, 400 cells otherwise) is applied to each side before convolution to suppress boundary artefacts, then removed afterwards. The solution is computed on a full time grid and stored as a `(n_t, n_x)` array.

**Forward Euler with upwind advection** (first-order in space and time):  
Used with adaptive time-stepping via a combined CFL + von Neumann stability condition.

**Lax-Wendroff scheme** (second-order in space and time):  
Reduces numerical diffusion compared to Euler at the cost of small oscillations near sharp gradients.

Additional functions:
- `u_0(cfg)` — constructs the initial condition array; optionally applies `time_to_ic > 0` steps of Burgers evolution to produce a smoothed-out starting condition (useful for N-wave ICs with discontinuities).
- `predict_shock_time(cfg)` — estimates $t_{shock}$ from the minimum spatial gradient of $u_0$.
- `interpolate_solution(sol, t_arr, x, t, cfg)` — bilinear interpolation of the solution grid at arbitrary $(x, t)$.
- `residual(sol, t_arr, cfg)` — computes the PDE residual $|u_t + uu_x - \nu u_{xx}|$ on the solution grid using finite differences.

#### `data.py`

- `make_observation(cfg, u_grid, t_arr)` — generates `n_obs` random $(t, x)$ pairs and interpolates noisy $u$ values from the Cole-Hopf grid.
- `make_collocation(cfg)` — generates random collocation points for the training and extrapolation domains.
- `make_validation(cfg, u_grid, t_arr)` — generates a regular `(n_grid_val × n_grid_val)` meshgrid of $(t, x)$ pairs with ground truth $u$ values for validation.
- `make_bc_points(cfg)` — generates random time points along the boundary for the boundary condition loss.
- `generate_data(cfg)` — full pipeline; also stores the Cole-Hopf grid and initial condition arrays in the returned dict.

#### `model.py`

- **`FCNet`**: maps $(t, x) \to u(t, x)$. Architecture: `2 → [hidden × n_layers] → 1` with `Tanh` activations.
- **`InverseFCNet`**: same architecture but adds a learnable scalar parameter `_nu_raw`. The physical viscosity is recovered as `nu_hat = softplus(_nu_raw)`, which ensures $\hat\nu > 0$ at all times. The softplus parameterisation avoids unconstrained optimisation of a physically positive quantity. `_nu_raw` is initialised from a random softplus-inverse of a uniform draw in $[0.5, 2.0]$.

#### `losses.py`

- **`loss_data`** — MSE between predicted and observed $u$ values.
- **`loss_physics`** — mean squared PDE residual:
$$\mathcal{L}_{physics} = \frac{1}{N}\sum_{i=1}^N (\hat{u}_t + \hat{u}\hat{u}_x - \nu\hat{u}_{xx})^2$$
All derivatives ($\hat{u}_t$, $\hat{u}_x$, $\hat{u}_{xx}$) are computed with `torch.autograd.grad`.
- **`loss_physics_inverse`** — same but uses `model.nu_hat` in place of `cfg.nu`.
- **`loss_ic`** — MSE between the predicted and true initial condition over all $x$:
$$\mathcal{L}_{ic} = \frac{1}{N_x}\sum_i (\hat{u}(0, x_i) - u_0(x_i))^2$$
- **`loss_bc`** — MSE of predicted values at both boundaries against the Dirichlet conditions:
$$\mathcal{L}_{bc} = \frac{1}{N}\sum_i \left[(\hat{u}(t_i, 0) - u_{left})^2 + (\hat{u}(t_i, L) - u_{right})^2\right]$$

The total loss is:
$$\mathcal{L}_{tot} = \mathcal{L}_{data} + \lambda_{phys}\mathcal{L}_{physics} + \lambda_{ic}\mathcal{L}_{ic} + \lambda_{bc}\mathcal{L}_{bc}$$

#### `trainer.py`
Same structure as the DHO trainer with the following additions:

- **Inverse mode detection** — checks for a `nu_hat` attribute on the model.
- **Boundary condition loss** — `loss_bc` is computed every epoch (BC points are randomised if `cfg.randomise_bc_points` is `True`).
- The best model state is loaded back into the model before the function returns, so the returned model is already at its best weights.

#### `utils.py`
Identical in structure to the DHO utils. The `load_best_cfg` function reads an Optuna JSON output and populates a `Config` object, mapping the `regime` key to the `ic` field.

## Running the demo scripts

Both projects ship ready-to-run demo scripts under their respective `scripts/` folders. All scripts share the same CLI interface.

---

### Damped Harmonic Oscillator

#### Forward problem

```bash
python Damped_Oscillator/scripts/pinn_demo.py
python Damped_Oscillator/scripts/pinn_demo.py --underdamped
python Damped_Oscillator/scripts/pinn_demo.py --zeta 0.5 --omega_0 3.0
python Damped_Oscillator/scripts/pinn_demo.py --overdamped --output outputs/my_run
```

#### Inverse problem

```bash
python Damped_Oscillator/scripts/inversepinn_demo.py
python Damped_Oscillator/scripts/inversepinn_demo.py --underdamped
python Damped_Oscillator/scripts/inversepinn_demo.py --zeta 0.1 --omega_0 2.0
python Damped_Oscillator/scripts/inversepinn_demo.py --overdamped --output outputs/my_run
```

#### Shared CLI arguments — Damped Harmonic Oscillator

| Argument | Default | Description |
|---|---|---|
| `--zeta` | — | Fix damping ratio ζ; skips randomisation |
| `--omega_0` | — | Fix natural frequency ω₀; skips randomisation |
| `--underdamped` | — | Use underdamped regime (ζ < 1). Forward: fixed at 0.125; inverse: randomised in [0.02, 0.2] |
| `--critically-damped` | — | Use critically damped case (ζ = 1) |
| `--overdamped` | — | Use overdamped regime (ζ > 1). Forward: fixed at 1.5; inverse: randomised in [1.5, 2.0] |
| `--show` | `False` | Display plots interactively after training |
| `--output` | `outputs/<script>_<regime>` | Override the output directory |

If no damping flag is given, the forward script defaults to ζ = 0.125 and the inverse script randomises in the underdamped regime.

---

### Burgers' Equation

#### Forward problem

```bash
python Burgers_Equation/scripts/pinn_demo.py
python Burgers_Equation/scripts/pinn_demo.py --ic N_wave --nu 0.05
python Burgers_Equation/scripts/pinn_demo.py --ic Step_up --output outputs/my_run
```

#### Inverse problem

```bash
python Burgers_Equation/scripts/inversepinn_demo.py
python Burgers_Equation/scripts/inversepinn_demo.py --nu 0.3 --ic Gauss
python Burgers_Equation/scripts/inversepinn_demo.py --nu-regime low --output outputs/inverse_low
```

#### Shared CLI arguments — Burgers' Equation

| Argument | Default | Description |
|---|---|---|
| `--ic` | `Gauss` | Initial condition type: `Gauss`, `N_wave`, or `Step_up` |
| `--nu` | — | Fix viscosity to a specific value; skips randomisation |
| `--nu-regime` | `high` | Viscosity regime to sample from when `--nu` is not set: `low` (0.005–0.015) or `high` (0.1–0.5). Inverse scripts only. |
| `--show` | `False` | Display plots interactively after training |
| `--output` | `outputs/<script>_<ic>` | Override the output directory |

---

### What happens when you run a script

**Damped Harmonic Oscillator:**

1. ζ and ω₀ are resolved from CLI flags or their defaults and converted to physical constants m, c, k.
2. Training and validation data are generated from the closed-form analytic solution.
3. The network is trained with early stopping. Progress is printed every `log_every` epochs.
4. The best model, loss history, snapshots, and config are saved to the output directory.
5. Plots are generated and saved automatically via `save_plots_from_file`.

**Burgers' Equation:**

1. The initial condition and viscosity are resolved from CLI arguments or their defaults.
2. The shock formation time is estimated analytically. If it falls in the expected range (0.1–5 s), the training domain is automatically capped at the shock time, with a short extrapolation window beyond it.
3. Training data (observations, collocation points, boundary conditions) is generated from the Cole-Hopf solution.
4. The network is trained with early stopping. Progress is printed every `log_every` epochs.
5. The best model, loss history, snapshots, and config are saved to the output directory (see [Saving and loading](#saving-and-loading)).
6. Plots are generated and saved automatically via `save_plots_from_file`.

---

### Output directory layout

```
outputs/pinn_underdamped/
├── best_model.pt
├── history.csv
├── snapshots.pt
└── config.json
```

Plots are written alongside these files. To regenerate plots from a previous run without retraining:

```python
from Damped_Oscillator.plot import save_plots_from_file
save_plots_from_file("outputs/pinn_underdamped")
```
---

## Training a model

For more customisation, it is alo possible to create your own scripts using the provided code instead of using the demo files. A minimal training script follows this pattern:

```python
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device, save_model

device = get_device()
cfg    = Config(nu=0.05, ic="Gauss", n_epochs=20000)
data   = generate_data(cfg)
model  = FCNet(cfg)

history, snapshots, best_state = train(model, data, cfg, device)

save_model(best_state, history, snapshots, cfg, output_path="outputs/my_run")
```

For the inverse problem, replace `FCNet` with `InverseFCNet` and set `cfg.use_data = True`. The trainer detects the model type automatically and uses a separate learning rate (`cfg.lr_inverse`) for the physical parameter.

---

## Saving and loading

`save_model` writes four files to `output_path/`:

| File | Contents |
|---|---|
| `best_model.pt` | State dict with the lowest validation loss |
| `history.csv` | Loss curves, one row per `log_every` epochs |
| `snapshots.pt` | State dicts at each `snapshot_epochs` checkpoint |
| `config.json` | Full `Config` as JSON for reproducibility |

`load_model(model, folder_path)` restores all four. `save_plots_from_file(folder_path)` regenerates ground truth data and saves all plots without requiring a re-run of training.

---

## Hyperparameter tuning with Optuna

Both projects support Optuna-based hyperparameter search. Pass an `optuna.Trial` to `train()` to enable pruning. A minimal study looks like:

```python
import optuna
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device

def objective(trial):
    cfg = Config(
        hidden        = trial.suggest_int("hidden", 16, 128, step=16),
        n_layers      = trial.suggest_int("n_layers", 2, 6),
        lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        lambda_phys   = trial.suggest_float("lambda_phys", 1e-3, 10, log=True),
    )
    data   = generate_data(cfg)
    model  = FCNet(cfg)
    device = get_device()
    history, _, _ = train(model, data, cfg, device, verbatim=False, optuna_trial=trial)
    return min(history["loss_val"])

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
```

---
