"""
train.py
========
Training script for the PINN Burgers' equation.

Every tunable value is exposed as a CLI argument and collected into a
BurgerConfig before anything else runs.  The script:

  1. Resolves the compute device (CUDA > MPS > CPU).
  2. Generates synthetic observations.
  3. Trains a Standard-ML model (data loss only).
  4. Trains a PINN model       (data + physics + IC loss).
  5. Saves the best checkpoint after every improvement
    so a crash never loses more than one log_every interval.
  6. Applies early stopping based on validation MSE.
  7. Serialises everything needed by plot.py into an output file.

Usage examples
--------------
  python train.py                          # all defaults
  python train.py --n_epochs 4000 --lr 5e-4
  python train.py --situation "Gaussian" --viscosity 1 
  python train.py --out_dir ./run_01 --patience 50
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import BurgerConfig, FCNet, Predictor
from Burger_PDE import BurgersSolver

# =============================================================================
# 0.  DEVICE SELECTION
# =============================================================================


def get_device() -> torch.device:
    """Resolve compute device with priority order.

    Returns the first available device from the priority order:
    CUDA (NVIDIA GPU) → Apple MPS (Metal Performance Shaders) → CPU.

    Returns:
        torch.device: The selected device for training and inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1.  ANALYTIC GROUND TRUTH
# =============================================================================

def analytic(x: np.ndarray, t: np.ndarray, solver: BurgersSolver) -> np.ndarray:
    """Compute analytic solution of the Burgers equation at given points.

    Queries the BurgersSolver for solution values at each (x, t) pair.
    Supports three initial conditions: N-wave, Gaussian, and Step.

    Args:
        x: Spatial coordinates array of shape (N,).
        t: Temporal coordinates array of shape (N,).
        solver: BurgersSolver instance for evaluating the analytic solution.

    Returns:
        Array of solution values at the requested points of shape (N,).
    """

    return np.array([solver.solution_at(x_query=x_i, t_query=t_i) for x_i, t_i in zip(x, t)])


# =============================================================================
# 2.  DATA GENERATION
# =============================================================================


def stratified_time_split(t, t_begin, t_end, p_val, n_bins, seed):
    """Stratified train/validation split along time axis.

    Ensures validation points are spread across time bins to prevent clustering
    at one end. Divides the time interval into n_bins and samples from each bin.

    Args:
        t: Array of time values to split.
        t_begin: Start of time interval.
        t_end: End of time interval.
        p_val: Fraction of data to hold for validation.
        n_bins: Number of time bins for stratification.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices) for the stratified split.
    """
    rng = np.random.default_rng(seed)
    # 1. Sort by time
    idx = np.argsort(t)
    t_sorted = t[idx]

    # 2. Create bins
    bins = np.linspace(t_begin, t_end, n_bins + 1)
    bin_ids = np.digitize(t_sorted, bins)

    train_idx, val_idx = [], []

    # 3. Sample inside each bin
    for b in range(1, n_bins + 1):
        in_bin = np.where(bin_ids == b)[0]
        rng.shuffle(in_bin)

        n = len(in_bin)
        n_val = int(p_val * n)

        val_idx.extend(in_bin[:n_val])
        train_idx.extend(in_bin[n_val:])

    # 4. Map back to original indices
    return idx[train_idx], idx[val_idx]


def generate_data(cfg: BurgerConfig, device: torch.device) -> dict:
    """Build all data arrays and tensors needed for training and plotting.

    Generates observations for train, perfect validation points, collocation
    points for physics loss, and dense evaluation grids. Returns both numpy
    arrays (for plotting) and device tensors (for training).

    * Observations (training) randomly drawn from (t0, t_train]
    * Initial condition enforced via IC loss
    * Validation chosen as a large number of random points without noise
    * Collocation points placed randomly over FULL domain [t0, t_extrap] (ext. phys.)
    or over the training domain [t0,t_train] (blind)
    * Dense evaluation grids (CPU-only) for post-hoc evaluation and plotting

    Args:
        cfg: BurgerConfig instance with all data generation parameters.
        device: Torch device for placing tensors (CPU or GPU).

    Returns:
        Dictionary containing observation arrays, validation arrays, plotting grids,
        and device tensors for training.
    """

    def to_tensor(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(device)
        if requires_grad:
            t.requires_grad_(True)
        return t

    # Spatial grid for the solver (double as the dense grid for plotting)
    x_solver = np.linspace(2*cfg.x_begin, 2*cfg.x_end, 4*cfg.n_plot_x)
    # Initial condition values at t0 (for IC loss)
    u_ic_true = cfg.ic_func(to_tensor(x_solver)).cpu().numpy().reshape(-1)
    # Time step for the solver, half as big as the plotting time step for accuracy
    dt = (cfg.t_extrap - cfg.t0) / (cfg.n_plot_t * 2)

    analytical_path_str = f"{cfg.dir_analytic}/{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_analytic_pt}"
    analytical_path = Path(analytical_path_str)

    solver = BurgersSolver(
        u0=u_ic_true,
        t_start=cfg.t0,
        t_end=cfg.t_extrap,
        dt=dt,
        x_bounds=(2*cfg.x_begin, 2*cfg.x_end),
        viscosity=cfg.v
    )
    # Ensure the directory exists for saving/loading the analytic solution snapshots
    Path(cfg.dir_analytic).mkdir(parents=True, exist_ok=True)

    if analytical_path.exists():
        print(
            f"Loading precomputed analytic solution snapshots from {analytical_path_str}...")
        solver = solver.load(analytical_path_str)
    else:
        print(f"\nSolving the Burgers equation with the {cfg.situation} initial condition"
              f"to generate the analytic solution snapshots for interpolation...")
        # Precompute the solution snapshots for interpolation in the analytic function
        solver.solve()
        # Save the precomputed solution for future runs
        solver.save(analytical_path_str)

    # ── Initial condition point ───────────────────────────────────────────────
    x_samples_ic = np.linspace(cfg.x_begin, cfg.x_end, cfg.n_ic_samples_x)
    t_ic = np.full_like(x_samples_ic, cfg.t0)
    u_ic_true = analytic(x_samples_ic, t_ic, solver)

    print(
        f"\nGenerating observation data for the {cfg.situation} initial condition...")
    # ── All M observations ────────────────────────────────────────────────────
    x_obs_train = np.random.uniform(cfg.x_begin, cfg.x_end, cfg.n_obs_total)
    t_obs_train = np.random.uniform(cfg.t0, cfg.t_train, cfg.n_obs_total)
    u_obs_train = analytic(x_obs_train, t_obs_train, solver) + \
        np.random.normal(0.0, cfg.sigma, cfg.n_obs_total)

    # ── Train / validation split  (stratified) ────── We were told to use the ground truth,
    # instead of stratified time split, but this is more realistic, so we will keep it in the code.
    # train_idx, val_idx = stratified_time_split(t_all, cfg.t0, cfg.t_train,
    #                                            cfg.val_fraction, cfg.n_bins, seed=cfg.seed)

    # t_obs_train, u_obs_train, x_obs_train = t_all[train_idx], u_all[train_idx], x_full[train_idx]
    # t_obs_val,   u_obs_val,   x_obs_val   = t_all[val_idx],   u_all[val_idx],   x_full[val_idx]

    x_obs_vec = np.linspace(cfg.x_begin, cfg.x_end, cfg.n_val_x)
    t_obs_vec = np.linspace(cfg.t0, cfg.t_train, cfg.n_val_t)

    t_obs_val_mat, x_obs_val_mat = np.meshgrid(
        t_obs_vec, x_obs_vec, indexing="ij")
    t_obs_val = t_obs_val_mat.reshape(-1)
    x_obs_val = x_obs_val_mat.reshape(-1)
    u_obs_val = analytic(x_obs_val, t_obs_val, solver)

    # ── Dense grids for post-hoc evaluation and plotting (CPU numpy only) ────

    t_plot_full = np.linspace(cfg.t0, cfg.t_extrap, cfg.n_plot_t)
    t_plot_train = t_plot_full[t_plot_full <= cfg.t_train]
    x_plot_full = np.linspace(cfg.x_begin, cfg.x_end, cfg.n_plot_x)

    t_plotmat_train, x_plotmat_train = np.meshgrid(
        t_plot_train, x_plot_full, indexing="ij")
    t_plotmat_full,  x_plotmat_full = np.meshgrid(
        t_plot_full,  x_plot_full,  indexing="ij")
    t_flattened_train = t_plotmat_train.reshape(-1)
    x_flattened_train = x_plotmat_train.reshape(-1)
    t_flattened_full = t_plotmat_full.reshape(-1)
    x_flattened_full = x_plotmat_full.reshape(-1)

    u_true_train = analytic(x_flattened_train, t_flattened_train, solver)
    u_true_full = analytic(x_flattened_full, t_flattened_full, solver)

    return {
        # numpy — train split
        "t_obs_train": t_obs_train,
        "u_obs_train": u_obs_train,
        # numpy — validation split
        "t_obs_val":   t_obs_val,
        "u_obs_val":   u_obs_val,
        # numpy — dense evaluation grids
        "t_flattened_train": t_flattened_train,
        "x_flattened_train": x_flattened_train,
        "t_flattened_full":  t_flattened_full,
        "x_flattened_full":  x_flattened_full,
        "u_true_train": u_true_train,
        "u_true_full":  u_true_full,
        # tensors on DEVICE — training observations
        "t_obs_train_t": to_tensor(t_obs_train),
        "x_obs_train_t": to_tensor(x_obs_train),
        "u_obs_train_t": to_tensor(u_obs_train),
        # tensors on DEVICE — validation observations
        "t_obs_val_t":   to_tensor(t_obs_val),
        "x_obs_val_t":   to_tensor(x_obs_val),
        "u_obs_val_t":   to_tensor(u_obs_val),
        # tensor on DEVICE — IC point (requires_grad for y'(0))
        "t_ic_t":  to_tensor(t_ic),
        "x_samples_ic_t": to_tensor(x_samples_ic),
    }


# =============================================================================
# 3.  LOSS COMPONENTS
# =============================================================================

def loss_physics(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: BurgerConfig,
) -> torch.Tensor:
    """Compute physics residual loss for the Burgers equation.

    L_physics = mean(r(x,t)²) over collocation points, where r is the residual
    of the PDE: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0.

    Both spatial and temporal derivatives computed via automatic differentiation.
    Uses create_graph=True to preserve computation graph for second derivatives.

    Args:
        model: Neural network model u_hat(x, t).
        x: Spatial coordinates of collocation points, shape (n_col, 1).
        t: Temporal coordinates of collocation points, shape (n_col, 1).
        cfg: BurgerConfig with viscosity parameter ν.

    Returns:
        Physics residual loss as a scalar tensor.
    """
    u_hat = model(x, t)

    du, dotu = torch.autograd.grad(
        u_hat, (x, t),
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True,
        retain_graph=True
    )

    # Compute d²u/dx²
    d2u = torch.autograd.grad(
        du, x,
        grad_outputs=torch.ones_like(du),
        create_graph=True
    )[0]

    residual = dotu + u_hat * du - cfg.v * d2u
    return torch.mean(residual**2)


def loss_ic(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: BurgerConfig,
) -> torch.Tensor:
    """Compute initial condition loss at t=t0.

    L_ic = mean((ŷ(x, t0) - y0(x))²) enforces the initial condition.
    Weighted heavily (λ_ic >> λ_phys) because errors at t=0 propagate
    and distort the entire downstream trajectory.

    Args:
        model: Neural network model u_hat(x, t).
        x: Spatial coordinates where IC is enforced, shape (n_ic, 1).
        t: Time coordinates all equal to t0, shape (n_ic, 1).
        cfg: BurgerConfig with initial condition function.

    Returns:
        Initial condition loss as a scalar tensor.
    """
    u_hat_0 = model(x, t)

    ic_conditions_func = cfg.ic_func

    return torch.mean((u_hat_0 - ic_conditions_func(x)) ** 2)


# =============================================================================
# 4.  TRAINING LOOP
# =============================================================================

def train_model(
    model:       nn.Module,
    data:        dict,
    cfg:         BurgerConfig,
    device:      torch.device,
    use_physics: bool,
    extrapolated_physics: bool,
    label:       str,
    ckpt_path:   Path,
    use_ic:     bool = True,
) -> tuple[dict, dict]:
    """
    Train *model* in-place on *device* and return (history, snapshots).

    Best checkpoint
    ---------------
    Validation MSE is computed at every *log_every* epoch.  Whenever it
    improves by more than *min_delta*, the current weights are written to
    *ckpt_path* — even if the run is interrupted afterwards the best weights
    on disk are never more than *log_every* epochs stale.

    Early stopping
    --------------
    If validation MSE does not improve for *patience* consecutive log
    intervals the loop exits early.  This prevents over-fitting on the
    sparse training observations and saves wall-clock time.

    Snapshots
    ---------
    At every epoch listed in cfg.snapshot_epochs the weights are copied to
    CPU and stored in the *snapshots* dict for later use by plot.py.

    Returns
    -------
    history   : dict of lists logged every cfg.log_every epochs
    snapshots : dict { epoch: CPU state_dict }
    """
    model.to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=cfg.lr_step, gamma=cfg.lr_gamma
    )

    t_obs_train_t = data["t_obs_train_t"]
    x_obs_train_t = data["x_obs_train_t"]
    u_obs_train_t = data["u_obs_train_t"]
    x_obs_val_t = data["x_obs_val_t"]
    t_obs_val_t = data["t_obs_val_t"]
    u_obs_val_t = data["u_obs_val_t"]
    t_ic_t = data["t_ic_t"]
    x_samples_ic_t = data["x_samples_ic_t"]

    history: dict[str, list] = {
        "epoch":         [],
        "loss_data":     [],
        "loss_physics":  [],
        "loss_ic":       [],
        "loss_total":    [],
        "loss_val":      [],
    }
    snapshots:    dict[int, dict] = {}
    snapshot_set: set[int] = set(cfg.snapshot_epochs)

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    stopped_early = False

    t0_wall = time.perf_counter()

    x_selected_epoch = x_obs_train_t
    t_selected_epoch = t_obs_train_t
    u_selected_epoch = u_obs_train_t

    x_pool = torch.rand(cfg.n_col_pool, 1, device=device) * \
        (cfg.x_end - cfg.x_begin) + cfg.x_begin

    t_pool_ext_physics = torch.rand(
        cfg.n_col_pool, 1, device=device)*(cfg.t_extrap - cfg.t0) + cfg.t0

    t_pool_blind = torch.rand(
        cfg.n_col_pool, 1, device=device)*(cfg.t_train - cfg.t0) + cfg.t0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        # Data loss — MSE on training observations only (not validation)
        # Randomly select a subset of the training observations for this epoch to
        # speed up training and add noise robustness.  This is a form of stochastic mini-batching.
        # epoch_random_selection = torch.randperm(t_obs_train_t.shape[0])[:cfg.n_obs_per_epoch]
        # x_selected_epoch = x_obs_train_t[epoch_random_selection]
        # t_selected_epoch = t_obs_train_t[epoch_random_selection]
        # u_selected_epoch = u_obs_train_t[epoch_random_selection]
        # We found that mini batching is worse than full batching (optimal n per batch = all)

        u_pred = model(x_selected_epoch, t_selected_epoch)
        l_data = torch.mean((u_pred - u_selected_epoch) ** 2)
        l_total = l_data
        l_phys = torch.zeros(1, device=device)
        l_ic = torch.zeros(1, device=device)

        if use_physics:
            idx = torch.randint(0, cfg.n_col_pool, (cfg.n_col,))
            x_col_t = x_pool[idx].clone().detach().requires_grad_(True)

            if extrapolated_physics:
                # ── Collocation points (randomly chosen) ───────────────────────────────────
                t_col_t = t_pool_ext_physics[idx].clone(
                ).detach().requires_grad_(True)
            else:
                t_col_t = t_pool_blind[idx].clone(
                ).detach().requires_grad_(True)

            l_phys = loss_physics(model, x_col_t, t_col_t, cfg)

        if use_ic:
            l_ic = loss_ic(model, x_samples_ic_t, t_ic_t, cfg)

        l_total = l_data + cfg.lambda_phys * l_phys + cfg.lambda_ic * l_ic
        l_total.backward()
        optimiser.step()
        scheduler.step()

        # ── Snapshot: pull weights to CPU before deep-copying ─────────────────
        # .cpu() clones every tensor in state_dict to CPU without touching
        # the live model on DEVICE.  deepcopy gives us an independent dict.
        if epoch in snapshot_set:
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            snapshots[epoch] = copy.deepcopy(cpu_state)

        # ── Periodic logging + validation ─────────────────────────────────────
        if epoch % cfg.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                u_val_pred = model(x_obs_val_t, t_obs_val_t)
                l_val = torch.mean((u_val_pred - u_obs_val_t) ** 2).item()

            history["epoch"].append(epoch)
            history["loss_data"].append(l_data.item())
            history["loss_physics"].append(l_phys.item())
            history["loss_ic"].append(l_ic.item())
            history["loss_total"].append(l_total.item())
            history["loss_val"].append(l_val)

            # ── Save best checkpoint ──────────────────────────────────────────
            if l_val < best_val_loss - cfg.min_delta:
                best_val_loss = l_val
                patience_counter = 0
                cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "epoch":            epoch,
                        "model_state_dict": copy.deepcopy(cpu_state),
                        "val_loss":         best_val_loss,
                        "config":           cfg,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1

            # ── Console log ───────────────────────────────────────────────────
            if epoch % cfg.print_every == 0 or epoch == 1:
                elapsed = time.perf_counter() - t0_wall
                print(
                    f"  [{label}] epoch {epoch:>6d}  "
                    f"L_data={l_data.item():.5f}  "
                    f"L_phys={l_phys.item():.5f}  "
                    f"L_ic={l_ic.item():.5f}  "
                    f"L_total={l_total.item():.5f}  "
                    f"L_val={l_val:.5f}  "
                    f"patience={patience_counter}/{cfg.patience}  "
                    f"({elapsed:.1f}s)"
                )

            # ── Early stopping check ──────────────────────────────────────────
            if patience_counter >= cfg.patience:
                print(
                    f"\n  [{label}] Early stopping triggered at epoch {epoch}. "
                    f"Best val loss: {best_val_loss:.6f}"
                )
                stopped_early = True
                break

    total_time = time.perf_counter() - t0_wall
    actual_epochs = epoch
    print(
        f"  [{label}] {'early stop' if stopped_early else 'complete'} "
        f"after {actual_epochs} epochs in {total_time:.1f}s  "
        f"({total_time / actual_epochs * 1000:.2f} ms/epoch)  "
        f"best val loss: {best_val_loss:.6f}"
    )

    return history, snapshots


# =============================================================================
# 5.  ARGUMENT PARSING  — every BurgerConfig field is a CLI flag
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training the Burgers equation PINN.

    Defines arguments across eight groups — physical parameters, initial
    conditions, time domain, space domain, data, PINN loss weights,
    architecture, optimiser, training loop, early stopping, and output —
    with defaults drawn from ``BurgerConfig``.

    Returns:
        argparse.Namespace: Parsed arguments, with one attribute per flag
            (e.g. ``args.viscosity``, ``args.situation``, ``args.n_epochs``).
    """
    p = argparse.ArgumentParser(
        description="Train PINN model for a damped"
        "spring-mass system with the constants as unknowns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physical parameters
    g = p.add_argument_group("Physical parameters")
    g.add_argument("--viscosity", type=float,
                   default=BurgerConfig.v, help="Viscosity coefficient")

    # Initial conditions
    g = p.add_argument_group("Initial conditions")
    g.add_argument("--situation", type=str, default=BurgerConfig.situation,
                   help="Initial condition scenario: 'N-wave', 'Gaussian', or 'Step'")

    # Time domain
    g = p.add_argument_group("Time domain")
    g.add_argument("--t0",      type=float,
                   default=BurgerConfig.t0, help="Initial time [s]")
    g.add_argument("--t_train",  type=float,
                   default=BurgerConfig.t_train,  help="End of training window [s]")
    g.add_argument("--t_extrap", type=float, default=BurgerConfig.t_extrap,
                   help="End of extrapolation window [s]")

    # Space domain
    g = p.add_argument_group("Space domain")
    g.add_argument("--x_begin",  type=float,
                   default=BurgerConfig.x_begin,  help="End of training window [m]")
    g.add_argument("--x_end", type=float, default=BurgerConfig.x_end,
                   help="End of extrapolation window [m]")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--n_obs",        type=int,
                   default=BurgerConfig.n_obs_total,   help="Total noisy observations")
    g.add_argument("--sigma",        type=float,
                   default=BurgerConfig.sigma, help="Measurement noise std dev")
    g.add_argument("--n_col",      type=int,   default=BurgerConfig.n_col,
                   help="Collocation points for physics loss")
    g.add_argument("--seed",         type=int,
                   default=BurgerConfig.seed,   help="RNG seed")

    # Loss weights
    g = p.add_argument_group("PINN loss weights")
    g.add_argument("--lambda_phys", type=float,
                   default=BurgerConfig.lambda_phys, help="Physics residual weight")
    g.add_argument("--lambda_ic",   type=float,
                   default=BurgerConfig.lambda_ic, help="Initial-condition weight")

    # Architecture
    g = p.add_argument_group("Architecture")
    g.add_argument("--hidden",   type=int,
                   default=BurgerConfig.hidden, help="Neurons per hidden layer")
    g.add_argument("--n_layers", type=int,
                   default=BurgerConfig.n_layers,  help="Number of hidden layers")

    # Optimiser
    g = p.add_argument_group("Optimiser")
    g.add_argument("--lr",       type=float,
                   default=BurgerConfig.lr, help="Initial Adam learning rate")
    g.add_argument("--lr_step",  type=int,   default=BurgerConfig.lr_step,
                   help="StepLR decay interval [epochs]")
    g.add_argument("--lr_gamma", type=float,
                   default=BurgerConfig.lr_gamma,  help="StepLR decay factor")

    # Training loop
    g = p.add_argument_group("Training loop")
    g.add_argument("--n_epochs",    type=int,
                   default=BurgerConfig.n_epochs, help="Maximum training epochs")
    g.add_argument("--print_every", type=int, default=BurgerConfig.print_every,
                   help="Console log interval [epochs]")
    g.add_argument("--log_every",   type=int, default=BurgerConfig.log_every,
                   help="History log interval [epochs]")

    # Early stopping
    g = p.add_argument_group("Early stopping")
    g.add_argument("--patience",  type=int,   default=BurgerConfig.patience,
                   help="Patience in log_every units")
    g.add_argument("--min_delta", type=float, default=BurgerConfig.min_delta,
                   help="Min improvement to reset counter")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--out_dir", type=str, default=BurgerConfig.out_dir,
                   help="Output directory for all saved files")

    return p.parse_args()

# =============================================================================
# 6.  MAIN
# =============================================================================


def main() -> None:
    """
    Entry point for training and evaluating the Burgers equation PINN models.

    Orchestrates the full pipeline:

    1. Parses CLI arguments via :func:`parse_args`.
    2. Detects and reports the available compute device (CPU / CUDA), printing
       GPU name and VRAM if applicable.
    3. Summarises the run configuration (viscosity, scenario, observation
       count, noise level, collocation grid) to stdout.
    4. Trains all three models (standard ML, PINN with extended physics,
       PINN blind) by delegating to :func:`train_and_save_three`.
    5. Evaluates and compares all three models via :func:`evaluate_model`.
    6. Prints the command needed to generate plots for the completed run.

    Side effects:
        - Writes model checkpoints and result files to ``args.out_dir``.
        - Prints a structured run summary and results table to stdout.
    """
    args = parse_args()

    # ── Convert CLI arguments to kwargs for train_and_save_both ───────────────
    kwargs = {
        'viscosity': args.viscosity,
        'situation': args.situation,
        'x_begin': args.x_begin,
        'x_end': args.x_end,
        't0': args.t0,
        't_train': args.t_train,
        't_extrap': args.t_extrap,
        'n_obs_total': args.n_obs,
        'sigma': args.sigma,
        'n_col': args.n_col,
        'seed': args.seed,
        'lambda_phys': args.lambda_phys,
        'lambda_ic': args.lambda_ic,
        'hidden': args.hidden,
        'n_layers': args.n_layers,
        'lr': args.lr,
        'lr_step': args.lr_step,
        'lr_gamma': args.lr_gamma,
        'n_epochs': args.n_epochs,
        'print_every': args.print_every,
        'log_every': args.log_every,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'out_dir': args.out_dir,
    }
    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device()
    print("=" * 70)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram_gb:.1f} GB")
    print(f"  ν    : {args.viscosity:.4f}    Situation : {args.situation}")
    print("=" * 70)
    print(
        f"\n  Observations: {args.n_obs} for training (with noise σ={args.sigma})")
    print(
        f"  Collocation : {args.n_col} pts over [{args.x_begin}, {args.x_end}] x"
        f" [{args.t0}, {args.t_extrap} (phys. ext.) / {args.t_train} (blind)]")
    print(f"  IC enforced at t={args.t0} via L_ic (not in observations)\n")

    train_and_save_three(**kwargs)
    evaluate_model(situation=args.situation, v=args.viscosity)
    print(f"Run BurgersPINN/plot.py --situation {args.situation}"
          f"--viscosity {args.viscosity} to generate plots")


def train_and_save_three(**kwargs) -> None:
    """
    Train both PINN models (extended physics and blind) with provided config kwargs.

    Accepts keyword arguments matching BurgerConfig fields:
      situation, viscosity (v), x_begin, x_end, t0, t_train, t_extrap,
      n_obs_total, sigma, n_col, seed, lambda_phys, lambda_ic, hidden,
      n_layers, lr, lr_step, lr_gamma, n_epochs, print_every,
      log_every, patience, min_delta, out_dir
    """
    # Map kwargs to BurgerConfig parameter names
    config_kwargs = {
        'v': kwargs.get('viscosity', BurgerConfig.v),
        'situation': kwargs.get('situation', BurgerConfig.situation),
        'x_begin': kwargs.get('x_begin', BurgerConfig.x_begin),
        'x_end': kwargs.get('x_end', BurgerConfig.x_end),
        't0': kwargs.get('t0', BurgerConfig.t0),
        't_train': kwargs.get('t_train', BurgerConfig.t_train),
        't_extrap': kwargs.get('t_extrap', BurgerConfig.t_extrap),
        'n_obs_total': kwargs.get('n_obs_total', BurgerConfig.n_obs_total),
        'sigma': kwargs.get('sigma', BurgerConfig.sigma),
        'n_col': kwargs.get('n_col', BurgerConfig.n_col),
        'seed': kwargs.get('seed', BurgerConfig.seed),
        'lambda_phys': kwargs.get('lambda_phys', BurgerConfig.lambda_phys),
        'lambda_ic': kwargs.get('lambda_ic', BurgerConfig.lambda_ic),
        'hidden': kwargs.get('hidden', BurgerConfig.hidden),
        'n_layers': kwargs.get('n_layers', BurgerConfig.n_layers),
        'lr': kwargs.get('lr', BurgerConfig.lr),
        'lr_step': kwargs.get('lr_step', BurgerConfig.lr_step),
        'lr_gamma': kwargs.get('lr_gamma', BurgerConfig.lr_gamma),
        'n_epochs': kwargs.get('n_epochs', BurgerConfig.n_epochs),
        'print_every': kwargs.get('print_every', BurgerConfig.print_every),
        'log_every': kwargs.get('log_every', BurgerConfig.log_every),
        'patience': kwargs.get('patience', BurgerConfig.patience),
        'min_delta': kwargs.get('min_delta', BurgerConfig.min_delta),
        'out_dir': kwargs.get('out_dir', BurgerConfig.out_dir),
    }
    cfg = BurgerConfig(**config_kwargs)

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device()

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_pinn_ext_phys = out_dir / \
        f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_ckpt_pinn_ext_phys}"
    ckpt_pinn_blind = out_dir / \
        f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_ckpt_pinn_blind}"
    ckpt_ml = out_dir / f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_ckpt_ml}"
    results_path = out_dir / \
        f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_results_pt}"

    # ── Data ──────────────────────────────────────────────────────────────────
    data = generate_data(cfg, device)

    # ── Standard ML model ─────────────────────────────────────────────────────
    print("=" * 70)
    print("Training STANDARD ML  (data loss only)")
    print("=" * 70)
    model_ml = FCNet.from_config(cfg)
    print(f"  Parameters: {model_ml.param_count()}")
    hist_ml, snaps_ml = train_model(
        model_ml, data, cfg, device,
        use_physics=False,
        extrapolated_physics=False,
        label="StdML",
        ckpt_path=ckpt_ml,
    )

    # ── PINN model (extended physics) ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Training PINN  (data + physics (both training and extrapolated region) + IC loss)")
    print("=" * 70)
    model_pinn_ext_phys = FCNet.from_config(cfg)
    print(f"  Parameters: {model_pinn_ext_phys.param_count()}")
    hist_pinn_ext_phys, snaps_pinn_ext_phys = train_model(
        model_pinn_ext_phys, data, cfg, device,
        use_physics=True,
        extrapolated_physics=True,
        label="PINN ext. phy.",
        ckpt_path=ckpt_pinn_ext_phys,
    )

    # ── PINN model (blind) ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Training PINN  (data + physics (only training region) + IC loss)")
    print("=" * 70)
    model_pinn_blind = FCNet.from_config(cfg)
    print(f"  Parameters: {model_pinn_blind.param_count()}")
    hist_pinn_blind, snaps_pinn_blind = train_model(
        model_pinn_blind, data, cfg, device,
        use_physics=True,
        extrapolated_physics=False,
        label="PINN blind",
        ckpt_path=ckpt_pinn_blind,
    )
    pred_ml = Predictor(cfg, checkpoint_path=str(ckpt_ml))
    pred_pinn_ext_phys = Predictor(
        cfg, checkpoint_path=str(ckpt_pinn_ext_phys))
    pred_pinn_blind = Predictor(cfg, checkpoint_path=str(ckpt_pinn_blind))

    t_flattened_full = data["t_flattened_full"]
    x_flattened_full = data["x_flattened_full"]
    u_true_full = data["u_true_full"]

    u_ml_full = pred_ml.predict(x_flattened_full, t_flattened_full)
    u_pinn_ext_phys_full = pred_pinn_ext_phys.predict(
        x_flattened_full, t_flattened_full)
    u_pinn_blind_full = pred_pinn_blind.predict(
        x_flattened_full, t_flattened_full)

    mask_train = t_flattened_full <= cfg.t_train

    mask_extrap = t_flattened_full > cfg.t_train

    rmse_ml_train = Predictor.rmse(
        u_ml_full[mask_train],   u_true_full[mask_train])
    rmse_pinn_ext_phys_train = Predictor.rmse(
        u_pinn_ext_phys_full[mask_train],  u_true_full[mask_train])
    rmse_pinn_blind_train = Predictor.rmse(
        u_pinn_blind_full[mask_train],  u_true_full[mask_train])
    rmse_ml_extrap = Predictor.rmse(
        u_ml_full[mask_extrap],  u_true_full[mask_extrap])
    rmse_pinn_ext_phys_extrap = Predictor.rmse(
        u_pinn_ext_phys_full[mask_extrap], u_true_full[mask_extrap])
    rmse_pinn_blind_extrap = Predictor.rmse(
        u_pinn_blind_full[mask_extrap], u_true_full[mask_extrap])
    phys_ml = Predictor.physics_residual(
        u_ml_full, x_flattened_full, t_flattened_full, cfg)
    phys_pinn_ext_phys = Predictor.physics_residual(
        u_pinn_ext_phys_full, x_flattened_full, t_flattened_full, cfg)
    phys_pinn_blind = Predictor.physics_residual(
        u_pinn_blind_full, x_flattened_full, t_flattened_full, cfg)

    # ── Save results bundle for plot.py ───────────────────────────────────────
    metrics = {
        "rmse_ml_train":   rmse_ml_train,
        "rmse_pinn_ext_phys_train": rmse_pinn_ext_phys_train,
        "rmse_pinn_blind_train": rmse_pinn_blind_train,
        "rmse_ml_extrap":     rmse_ml_extrap,
        "rmse_pinn_ext_phys_extrap":   rmse_pinn_ext_phys_extrap,
        "rmse_pinn_blind_extrap":   rmse_pinn_blind_extrap,
        "phys_ml":         phys_ml,
        "phys_pinn_ext_phys":       phys_pinn_ext_phys,
        "phys_pinn_blind":       phys_pinn_blind,
    }

    torch.save(
        {
            "config":       cfg,
            "device_str":   str(device),
            "data":         {k: v for k, v in data.items()
                             if isinstance(v, np.ndarray)},   # numpy only
            "hist_ml":      hist_ml,
            "hist_pinn_ext_phys":    hist_pinn_ext_phys,
            "hist_pinn_blind":    hist_pinn_blind,
            "snaps_ml":     snaps_ml,
            "snaps_pinn_ext_phys":   snaps_pinn_ext_phys,
            "snaps_pinn_blind":   snaps_pinn_blind,
            "u_ml_full":    u_ml_full,
            "u_pinn_ext_phys_full":  u_pinn_ext_phys_full,
            "u_pinn_blind_full":  u_pinn_blind_full,
            "metrics":      metrics,
            "time_shockwave": cfg.inviscid_shockwave_time,
        },
        results_path,
    )


def evaluate_model(situation: str = BurgerConfig.situation,
                   v: float = BurgerConfig.v) -> tuple[float, float, float]:
    """
    Evaluate and compare all three models — standard ML, PINN (extended
    physics), and PINN (blind) — against ground-truth data, then print a
    side-by-side results summary.

    Loads the best-validation checkpoints for all three models from disk, runs
    inference over the full spatiotemporal grid, and computes RMSE on the
    training interval, RMSE on the extrapolation interval, and physics residual
    for each model.

    Args:
        situation (str): Identifier for the physical scenario to evaluate
            (e.g. ``"shock"``). Defaults to ``BurgerConfig.situation``.
        v (float): True kinematic viscosity used when locating the saved
            results and checkpoint files. Defaults to ``BurgerConfig.v``.

    Returns:
        tuple[float, float, float]: A
            ``(rmse_ml_train, rmse_pinn_blind_train, rmse_pinn_ext_phys_train)``
            triple containing the training-interval RMSE for the standard ML
            model, blind PINN, and extended-physics PINN, respectively.

    Side effects:
        Prints a formatted results table to stdout comparing all three models
        across all metrics.
    """
    out_dir = Path(BurgerConfig.out_dir)
    results_path = out_dir / \
        f"{situation}_{v:.1e}_{BurgerConfig.suffix_results_pt}"
    raw = torch.load(results_path, map_location="cpu", weights_only=False)
    config = raw["config"]
    data = raw["data"]
    ckpt_ml = out_dir / \
        f"{config.situation}_{config.v:.1e}_{config.suffix_ckpt_ml}"
    ckpt_pinn_ext_phys = out_dir / \
        f"{config.situation}_{config.v:.1e}_{config.suffix_ckpt_pinn_ext_phys}"
    ckpt_pinn_blind = out_dir / \
        f"{config.situation}_{config.v:.1e}_{config.suffix_ckpt_pinn_blind}"

    pred_ml = Predictor(config, checkpoint_path=str(ckpt_ml))
    pred_pinn_ext_phys = Predictor(
        config, checkpoint_path=str(ckpt_pinn_ext_phys))
    pred_pinn_blind = Predictor(config, checkpoint_path=str(ckpt_pinn_blind))

    t_flattened_full = data["t_flattened_full"]
    x_flattened_full = data["x_flattened_full"]
    u_true_full = data["u_true_full"]

    u_ml_full = pred_ml.predict(x_flattened_full, t_flattened_full)
    u_pinn_ext_phys_full = pred_pinn_ext_phys.predict(
        x_flattened_full, t_flattened_full)
    u_pinn_blind_full = pred_pinn_blind.predict(
        x_flattened_full, t_flattened_full)

    mask_train = t_flattened_full <= config.t_train

    mask_extrap = t_flattened_full > config.t_train

    rmse_ml_train = Predictor.rmse(
        u_ml_full[mask_train],   u_true_full[mask_train])
    rmse_pinn_ext_phys_train = Predictor.rmse(
        u_pinn_ext_phys_full[mask_train],  u_true_full[mask_train])
    rmse_pinn_blind_train = Predictor.rmse(
        u_pinn_blind_full[mask_train],  u_true_full[mask_train])
    rmse_ml_extrap = Predictor.rmse(
        u_ml_full[mask_extrap],  u_true_full[mask_extrap])
    rmse_pinn_ext_phys_extrap = Predictor.rmse(
        u_pinn_ext_phys_full[mask_extrap], u_true_full[mask_extrap])
    rmse_pinn_blind_extrap = Predictor.rmse(
        u_pinn_blind_full[mask_extrap], u_true_full[mask_extrap])
    phys_ml = Predictor.physics_residual(
        u_ml_full, x_flattened_full, t_flattened_full, config)
    phys_pinn_ext_phys = Predictor.physics_residual(
        u_pinn_ext_phys_full, x_flattened_full, t_flattened_full, config)
    phys_pinn_blind = Predictor.physics_residual(
        u_pinn_blind_full, x_flattened_full, t_flattened_full, config)

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY  (best-val checkpoint)")
    print("=" * 70)
    print(f"  {'Metric':<38}  {'Std ML':>10}  {'PINN (extended physics)':>10} {'PINN (blind)':>10}")
    print("  " + "-" * 62)
    print(f"  {'RMSE  (training interval)':38}  {rmse_ml_train:>10.4f}  {rmse_pinn_ext_phys_train:>10.4f}  {rmse_pinn_blind_train:>10.4f}")
    print(f"  {'RMSE  (extrapolation)':38}  {rmse_ml_extrap:>10.4f}  {rmse_pinn_ext_phys_extrap:>10.4f}  {rmse_pinn_blind_extrap:>10.4f}")
    print(f"  {'Physics Residual':38}  {phys_ml:>10.4f}  {phys_pinn_ext_phys:>10.4f}  {phys_pinn_blind:>10.4f}")

    return rmse_ml_train, rmse_pinn_blind_train, rmse_pinn_ext_phys_train


def evaluate_blind(situation: str = BurgerConfig.situation,
                   v: float = BurgerConfig.v) -> float:
    """
    Evaluate only the PINN (blind) model against ground-truth data and print
    a single-model results summary.

    Loads the best-validation checkpoint for the blind PINN from disk, runs
    inference over the full spatiotemporal grid, and computes RMSE on the
    training interval, RMSE on the extrapolation interval, and physics residual.

    Args:
        situation (str): Identifier for the physical scenario to evaluate
            (e.g. ``"shock"``). Defaults to ``BurgerConfig.situation``.
        v (float): True kinematic viscosity used when locating the saved
            results and checkpoint files. Defaults to ``BurgerConfig.v``.

    Returns:
        float: Training-interval RMSE of the blind PINN against the
            ground-truth solution.

    Side effects:
        Prints a formatted results table to stdout for the blind model.
    """
    out_dir = Path(BurgerConfig.out_dir)
    results_path = out_dir / \
        f"{situation}_{v:.1e}_{BurgerConfig.suffix_results_pt}"
    raw = torch.load(results_path, map_location="cpu", weights_only=False)
    config: BurgerConfig = raw["config"]
    data = raw["data"]
    ckpt_pinn_blind = out_dir / \
        f"{config.situation}_{config.v:.1e}_{config.suffix_ckpt_pinn_blind}"
    pred_pinn_blind = Predictor(config, checkpoint_path=str(ckpt_pinn_blind))

    t_flattened_full = data["t_flattened_full"]
    x_flattened_full = data["x_flattened_full"]
    u_true_full = data["u_true_full"]

    u_pinn_blind_full = pred_pinn_blind.predict(
        x_flattened_full, t_flattened_full)

    mask_train = t_flattened_full <= config.t_train

    mask_extrap = t_flattened_full > config.t_train

    rmse_pinn_blind_train = Predictor.rmse(
        u_pinn_blind_full[mask_train],  u_true_full[mask_train])

    rmse_pinn_blind_extrap = Predictor.rmse(
        u_pinn_blind_full[mask_extrap], u_true_full[mask_extrap])

    phys_pinn_blind = Predictor.physics_residual(
        u_pinn_blind_full, x_flattened_full, t_flattened_full, config)

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY  (best-val checkpoint)")
    print("=" * 70)
    print(f"  {'Metric':<38} {'PINN (blind)':>10}")
    print("  " + "-" * 62)
    print(f"  {'RMSE  (training interval)':38}  {rmse_pinn_blind_train:>10.4f}")
    print(f"  {'RMSE  (extrapolation)':38}  {rmse_pinn_blind_extrap:>10.4f}")
    print(f"  {'Physics Residual':38}  {phys_pinn_blind:>10.4f}")

    return rmse_pinn_blind_train


if __name__ == "__main__":
    main()
