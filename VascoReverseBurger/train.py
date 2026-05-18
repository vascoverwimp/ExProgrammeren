"""
train.py
========
Training script for the PINN damped-spring-mass system.

Every tunable value is exposed as a CLI argument and collected into a
BurgerConfig before anything else runs.  The script:

  1. Resolves the compute device (CUDA > MPS > CPU).
  2. Generates synthetic observations with a train / validation split.
  3. Trains a Standard-ML model (data loss only).
  4. Trains a PINN model       (data + physics + IC loss).
  5. Saves the best checkpoint after every improvement (best_ml.pt /
     best_pinn.pt) so a crash never loses more than one log_every interval.
  6. Applies early stopping based on validation MSE.
  7. Serialises everything needed by plot.py into training_results.pt.

Usage examples
--------------
  python train.py                          # all defaults
  python train.py --n_epochs 4000 --lr 5e-4
  python train.py --mass 2.0 --damping 0.8 --stiffness 6.0
  python train.py --out_dir ./run_01 --patience 50
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from scipy.special import erfc
from model import BurgerConfig, FCNet, Predictor
from Burger_PDE import BurgersSolver

# =============================================================================
# 0.  DEVICE SELECTION
# =============================================================================

def get_device() -> torch.device:
    """Priority: CUDA → Apple MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1.  ANALYTIC GROUND TRUTH
# =============================================================================

def analytic(x: np.ndarray, t: np.ndarray, cfg: BurgerConfig, solver: BurgersSolver) -> np.ndarray:
    """
    Analytic solution of the Burger's equation for three different regimes:
    N-wave: starting with u(x,t0) = exp(-(x-1)**2/2) - exp(-(x+1)**2/2) the solution evolves into a characteristic N-wave shape.   
    """

    return np.array([solver.solution_at(x_query=x_i, t_query=t_i) for x_i, t_i in zip(x, t)])
    # shifted_t = t - cfg.t0
    # mask = shifted_t > 1e-10  # Only apply the viscous formula where time has actually progressed beyond t0
    # if cfg.situation == "Step":
        
    #     # Initialize output array with the initial condition (t <= t0)
    #     # Defaulting to the step function logic
    #     u = np.where(x > 0, 1.0, 0.0)
        
    #     # Only calculate the complex viscous formula where time has actually progressed
    #     if np.any(mask):
    #         # Extract only the points that need the viscous calculation
    #         tm = shifted_t[mask]
    #         xm = x[mask]

    #         sqrt_4nut = np.sqrt(4 * cfg.v * tm)
            
    #         # Term A: erfc(-x / sqrt)
    #         term_a = erfc(-xm / sqrt_4nut)
            
    #         # Term B: exp(exponent) * erfc((t-x)/sqrt)
    #         exponent = (xm - 0.5 * tm) / (2 * cfg.v)
    #         # Standard float64 max exponent is ~709. 
    #         # Clipping at 500 is safe and effectively "infinite" for a ratio.
    #         exponent = np.clip(exponent, -1e7, 1e7)
            
    #         term_b = np.exp(exponent) * erfc((tm - xm) / sqrt_4nut)
            
    #         u_viscous = term_b / (term_a + term_b)
            
    #         # Place the calculated values back into the main array
    #         u_viscous_nan = np.isnan(u_viscous)
    #         u_viscous[u_viscous_nan] = 0.0  # Assign a default value (e.g., 0) to NaNs resulting from 0/0
    #         u[mask] = u_viscous

    #     return u
    
    # elif cfg.situation == "N-wave":
    #     u = np.where(np.abs(x) < 1.0, -x, 0.0) # Defaulting to the N-wave logic (negative Gaussian) for the Gaussian situation
    #     # Only calculate the complex viscous formula where time has actually progressed
    #     if np.any(mask):
    #         # Extract only the points that need the viscous calculation
    #         tm = shifted_t[mask]
    #         xm = x[mask]
            
    #         x_integrating = np.linspace(0, np.max(np.abs([cfg.x_end,cfg.x_begin])), 1000)
    #         ic_func = lambda x: np.where(np.abs(x) < 1.0, -x, 0.0)
    #         area_pos = np.trapz(ic_func(x_integrating), x_integrating)
    #         Re0 = area_pos / (2*cfg.v)
    #         denominator = 1 + np.exp(xm**2/(4*cfg.v*tm) - Re0)
    #         u_viscous = xm/(tm+1e-14) * 1/(denominator+1e-14)
            
    #         # Place the calculated values back into the main array
    #         u[mask] = u_viscous
    #     return u
    
    # elif cfg.situation == "N-wave2":
    #     u = np.where(np.abs(x) < 1.0, -x, 0.0) # Defaulting to the N-wave logic (negative Gaussian) for the Gaussian situation
    #     # Only calculate the complex viscous formula where time has actually progressed
    #     if np.any(mask):
    #         # Extract only the points that need the viscous calculation
    #         tm = shifted_t[mask]
    #         xm = x[mask]
            
    #         x_integrating = np.linspace(0, np.max(np.abs([cfg.x_end,cfg.x_begin])), 1000)
    #         ic_func = lambda x: np.where(np.abs(x) < 1.0, -x, 0.0)
    #         area_pos = np.trapz(ic_func(x_integrating), x_integrating)
    #         Re0 = area_pos / (2*cfg.v)
    #         tau = cfg.t0*(np.exp(Re0) - 1)**2
    #         Re = np.log(1 + np.sqrt(tau/tm))
    #         denominator = (1+1/(np.exp(Re0-1))*np.sqrt(tm/tau)*np.exp(-Re*xm**2/(4*cfg.v*tm*Re0)))
    #         u_viscous = xm/(tm+1e-14) * 1/(denominator+1e-14)
            
    #         # Place the calculated values back into the main array
    #         u[mask] = u_viscous
    #     return u
    
    # return x*t


# Stratified time split to ensure validation points are spread across the time axis rather than clustered at one end.

def stratified_time_split(t,t_begin,t_end, p_val, n_bins, seed):
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
        n_val   = int(p_val   * n)

        val_idx.extend(in_bin[:n_val])
        train_idx.extend(in_bin[n_val:])

    # 4. Map back to original indices
    return idx[train_idx], idx[val_idx]


# =============================================================================
# 2.  DATA GENERATION  (consistent with real-world analysis)
# =============================================================================

def generate_data(cfg: BurgerConfig, device: torch.device) -> dict:
    """
    Build all data arrays and tensors needed for training and plotting.

    Real-world workflow followed here
    ----------------------------------
    * Observations are randomly drawn from (t0, t_train] so t0 is never
      in the observation set — its initial condition is enforced via L_ic.
    * Observations are split into a TRAINING set and a VALIDATION set
      (val_fraction controls the size).  Training loss sees only the
      training observations; the validation loss drives early stopping.
    * Collocation points are placed uniformly over the FULL domain
      [0, t_extrap], extending into the extrapolation region.  No
      measurement is required at collocation points — only the ODE
      residual is evaluated there.
    * Dense evaluation grids (t_plot_*) are CPU-only; they are never used
      during training, only during post-hoc evaluation and plotting.

    Returns a dict with both numpy arrays (for plotting) and device tensors
    (for training).  All tensors that need autograd have requires_grad=True.
    """

    def to_tensor(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(device)
        if requires_grad:
            t.requires_grad_(True)
        return t
    
    x_solver = np.linspace(2*cfg.x_begin, 2*cfg.x_end, 4*cfg.n_plot_x)  # Spatial grid for the solver (double as the dense grid for plotting)
    u_ic_true = cfg.ic_func(to_tensor(x_solver)).cpu().numpy().reshape(-1)  # Initial condition values at t0 (for IC loss)
    dt = (cfg.t_extrap - cfg.t0) / (cfg.n_plot_t * 2)  # Time step for the solver, half as big as the plotting time step for accuracy

    analytical_path_str = f"{cfg.dir_analytic}/{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_analytic_pt}"
    analytical_path = Path(analytical_path_str)


    solver = BurgersSolver(
        u0=u_ic_true,
        t_start=cfg.t0,
        t_end=cfg.t_extrap,
        dt = dt,
        x_bounds=(2*cfg.x_begin, 2*cfg.x_end),
        viscosity=cfg.v
    )
    Path(cfg.dir_analytic).mkdir(parents=True, exist_ok=True)  # Ensure the directory exists for saving/loading the analytic solution snapshots

    if analytical_path.exists():
        print(f"Loading precomputed analytic solution snapshots from {analytical_path_str}...")
        solver = solver.load(analytical_path_str)
    else:
        print(f"\nSolving the Burgers equation with the {cfg.situation} initial condition to generate the analytic solution snapshots for interpolation...")
        solver.solve()  # Precompute the solution snapshots for interpolation in the analytic function
        solver.save(analytical_path_str)  # Save the precomputed solution for future runs

    # ── Initial condition point ───────────────────────────────────────────────
    x_samples_ic = np.linspace(cfg.x_begin, cfg.x_end, cfg.n_ic_samples_x)
    t_ic = np.full_like(x_samples_ic, cfg.t0)
    u_ic_true = analytic(x_samples_ic, t_ic, cfg, solver)

    print(f"\nGenerating observation data for the {cfg.situation} initial condition...")
    # ── All M observations ────────────────────────────────────────────────────
    x_obs_train, t_obs_train = np.random.uniform(cfg.x_begin, cfg.x_end, cfg.n_obs_total), np.random.uniform(cfg.t0, cfg.t_train, cfg.n_obs_total)  # spatial and temporal locations for observations
    u_obs_train = analytic(x_obs_train, t_obs_train, cfg, solver) + np.random.normal(0.0, cfg.sigma, cfg.n_obs_total)

    # ── Train / validation split  (stratified) ────── We found were told to use the ground truth,
    # instead of stratified time split, but this is more realistic, so we will keep it in the code.
    # train_idx, val_idx = stratified_time_split(t_all, cfg.t0, cfg.t_train, cfg.val_fraction, cfg.n_bins, seed=cfg.seed)

    # t_obs_train, u_obs_train, x_obs_train = t_all[train_idx], u_all[train_idx], x_full[train_idx]
    # t_obs_val,   u_obs_val,   x_obs_val   = t_all[val_idx],   u_all[val_idx],   x_full[val_idx]

    x_obs_vec = np.linspace(cfg.x_begin, cfg.x_end, cfg.n_val_x)
    t_obs_vec = np.linspace(cfg.t0, cfg.t_train, cfg.n_val_t)

    t_obs_val_mat, x_obs_val_mat = np.meshgrid(t_obs_vec, x_obs_vec, indexing="ij")
    t_obs_val = t_obs_val_mat.reshape(-1)
    x_obs_val = x_obs_val_mat.reshape(-1)
    u_obs_val = analytic(x_obs_val, t_obs_val, cfg, solver)

    # ── Dense grids for post-hoc evaluation and plotting (CPU numpy only) ────
    t_plot_train = np.linspace(cfg.t0, cfg.t_train,  np.round(cfg.n_plot_t * (cfg.t_train - cfg.t0) / (cfg.t_extrap - cfg.t0)).astype(int))
    t_plot_full  = np.linspace(cfg.t0, cfg.t_extrap, cfg.n_plot_t)
    x_plot_full = np.linspace(cfg.x_begin, cfg.x_end, cfg.n_plot_x)

    t_plotmat_train, x_plotmat_train = np.meshgrid(t_plot_train, x_plot_full, indexing="ij")
    t_plotmat_full,  x_plotmat_full  = np.meshgrid(t_plot_full,  x_plot_full,  indexing="ij")
    t_flattened_train = t_plotmat_train.reshape(-1)
    x_flattened_train = x_plotmat_train.reshape(-1)
    t_flattened_full  = t_plotmat_full.reshape(-1)
    x_flattened_full  = x_plotmat_full.reshape(-1)

    u_true_train = analytic(x_flattened_train, t_flattened_train, cfg, solver)
    u_true_full  = analytic(x_flattened_full, t_flattened_full,  cfg, solver)

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
        "t_ic_t":  to_tensor(t_ic,  requires_grad=True),
        "x_samples_ic_t": to_tensor(x_samples_ic, requires_grad=False),
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
    """
    L_physics = mean( r(tᵢ)² )  over all collocation points.

    Both derivatives are computed via automatic differentiation.
    create_graph=True on the first grad call keeps the computation graph
    alive so the backward pass through the second derivative works.
    """
    u_hat = model(x,t)

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

    residual = dotu + u_hat * du - model.v_hat * d2u
    return torch.mean(residual**2)


def loss_ic(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: BurgerConfig,
) -> torch.Tensor:
    """
    L_ic = ( ŷ(0) - y0 )² + ( ŷ'(0) - dy0 )²

    λ_ic >> λ_phys because an error at t=0 propagates and distorts the
    entire downstream trajectory.
    """
    u_hat_0 = model(x, t)
    du_0 = torch.autograd.grad(
        u_hat_0, t,
        grad_outputs=torch.ones_like(u_hat_0),
        create_graph=True,
    )[0]
    ic_conditions_func = cfg.ic_func
    ic_der_conditions_func = cfg.ic_der_func
    return torch.mean((u_hat_0 - ic_conditions_func(x)) ** 2 + (du_0 - ic_der_conditions_func(x)) ** 2)


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
    model.v_hat.data = model.v_hat.data.to(device)
    optimiser = torch.optim.Adam([{'params': model.net.parameters(), 'lr': cfg.lr, 'betas': (cfg.beta1, cfg.beta2)}, {'params': [model.v_hat], 'lr': cfg.lr_param}])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=cfg.lr_step, gamma=cfg.lr_gamma
    )

    t_obs_train_t = data["t_obs_train_t"]
    x_obs_train_t = data["x_obs_train_t"]
    u_obs_train_t = data["u_obs_train_t"]
    x_obs_val_t   = data["x_obs_val_t"]
    t_obs_val_t   = data["t_obs_val_t"]
    u_obs_val_t   = data["u_obs_val_t"]
    t_ic_t        = data["t_ic_t"]
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
    snapshot_set: set[int]        = set(cfg.snapshot_epochs)

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_loss   = float("inf")
    patience_counter = 0
    stopped_early   = False

    t0_wall = time.perf_counter()

    x_selected_epoch = x_obs_train_t
    t_selected_epoch = t_obs_train_t
    u_selected_epoch = u_obs_train_t

    x_pool = torch.rand(cfg.n_col_pool, 1, device=device)*(cfg.x_end - cfg.x_begin) + cfg.x_begin
    x_pool.requires_grad_(True)

    t_pool_ext_physics = torch.rand(cfg.n_col_pool, 1, device=device)*(cfg.t_extrap - cfg.t0) + cfg.t0
    t_pool_ext_physics.requires_grad_(True)

    t_pool_blind = torch.rand(cfg.n_col_pool, 1, device=device)*(cfg.t_train - cfg.t0) + cfg.t0
    t_pool_blind.requires_grad_(True)

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        # Data loss — MSE on training observations only (not validation)
        # Randomly select a subset of the training observations for this epoch to speed up training and add noise robustness.  This is a form of stochastic mini-batching.
        # epoch_random_selection = torch.randperm(t_obs_train_t.shape[0])[:cfg.n_obs_per_epoch]
        # x_selected_epoch = x_obs_train_t[epoch_random_selection]
        # t_selected_epoch = t_obs_train_t[epoch_random_selection]
        # u_selected_epoch = u_obs_train_t[epoch_random_selection]
        # We found that mini batching is worse than full batching
        
        u_pred   = model(x_selected_epoch, t_selected_epoch)
        l_data   = torch.mean((u_pred - u_selected_epoch) ** 2)
        l_total = l_data
        l_phys  = torch.zeros(1, device=device)
        l_ic    = torch.zeros(1, device=device)

        if use_physics:
            idx = torch.randint(0, cfg.n_col_pool, (cfg.n_col,))
            x_col_t = x_pool[idx]

            if extrapolated_physics:
            # ── Collocation points (randomly chosen) ───────────────────────────────────
                t_col_t = t_pool_ext_physics[idx]
            else:
                t_col_t = t_pool_blind[idx]
        
            l_phys = loss_physics(model, x_col_t, t_col_t, cfg)

            l_total += l_phys*cfg.lambda_phys

        if use_ic:
            l_ic   = loss_ic(model, x_samples_ic_t, t_ic_t, cfg)
            l_total += l_ic*cfg.lambda_ic

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
                best_val_loss    = l_val
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
    p = argparse.ArgumentParser(
        description="Train PINN model for a damped spring-mass system with the constants as unknowns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physical parameters
    g = p.add_argument_group("Physical parameters")
    g.add_argument("--viscosity", type=float, default=BurgerConfig.v, help="Viscosity coefficient")

    # Initial conditions
    g = p.add_argument_group("Initial conditions")
    g.add_argument("--situation", type=str, default=BurgerConfig.situation, help="Initial condition scenario: 'N-wave', 'Gaussian', or 'Step'")

    # Time domain
    g = p.add_argument_group("Time domain")
    g.add_argument("--t0",      type=float, default=BurgerConfig.t0, help="Initial time [s]")
    g.add_argument("--t_train",  type=float, default=BurgerConfig.t_train,  help="End of training window [s]")
    g.add_argument("--t_extrap", type=float, default=BurgerConfig.t_extrap, help="End of extrapolation window [s]")

    # Space domain
    g=p.add_argument_group("Space domain")
    g.add_argument("--x_begin",  type=float, default=BurgerConfig.x_begin,  help="End of training window [m]")
    g.add_argument("--x_end", type=float, default=BurgerConfig.x_end, help="End of extrapolation window [m]")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--n_obs",        type=int,   default=BurgerConfig.n_obs_total,   help="Total noisy observations")
    g.add_argument("--sigma",        type=float, default=BurgerConfig.sigma, help="Measurement noise std dev")
    g.add_argument("--n_col",      type=int,   default=BurgerConfig.n_col,  help="Collocation points for physics loss")
    g.add_argument("--seed",         type=int,   default=BurgerConfig.seed,   help="RNG seed")

    # Initial guess viscosity
    g.add_argument("--v_init", type=float, default=BurgerConfig.ini_guess_v, help="Initial guess for viscosity (for PINN training only, not data generation)")

    # Loss weights
    g = p.add_argument_group("PINN loss weights")
    g.add_argument("--lambda_phys", type=float, default=BurgerConfig.lambda_phys, help="Physics residual weight")
    g.add_argument("--lambda_ic",   type=float, default=BurgerConfig.lambda_ic, help="Initial-condition weight")

    # Architecture
    g = p.add_argument_group("Architecture")
    g.add_argument("--hidden",   type=int, default=BurgerConfig.hidden, help="Neurons per hidden layer")
    g.add_argument("--n_layers", type=int, default=BurgerConfig.n_layers,  help="Number of hidden layers")

    # Optimiser
    g = p.add_argument_group("Optimiser")
    g.add_argument("--lr",       type=float, default=BurgerConfig.lr, help="Initial Adam learning rate")
    g.add_argument("--lr_step",  type=int,   default=BurgerConfig.lr_step, help="StepLR decay interval [epochs]")
    g.add_argument("--lr_gamma", type=float, default=BurgerConfig.lr_gamma,  help="StepLR decay factor")
    g.add_argument("--lr_param", type=float, default=BurgerConfig.lr_param, help="Adam learning rate for learned physical parameters")

    # Training loop
    g = p.add_argument_group("Training loop")
    g.add_argument("--n_epochs",    type=int, default=BurgerConfig.n_epochs, help="Maximum training epochs")
    g.add_argument("--print_every", type=int, default=BurgerConfig.print_every, help="Console log interval [epochs]")
    g.add_argument("--log_every",   type=int, default=BurgerConfig.log_every,   help="History log interval [epochs]")

    # Early stopping
    g = p.add_argument_group("Early stopping")
    g.add_argument("--patience",  type=int,   default=BurgerConfig.patience,   help="Patience in log_every units")
    g.add_argument("--min_delta", type=float, default=BurgerConfig.min_delta, help="Min improvement to reset counter")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--out_dir", type=str, default=BurgerConfig.out_dir, help="Output directory for all saved files")

    return p.parse_args()


# =============================================================================
# 6.  MAIN
# =============================================================================

def main_training() -> None:
    args = parse_args()

    # ── Build config from CLI arguments ───────────────────────────────────────
    cfg = BurgerConfig(
        v            = args.viscosity,
        situation    = args.situation,
        x_begin      = args.x_begin,
        x_end        = args.x_end,
        t0           = args.t0,
        t_train      = args.t_train,
        t_extrap     = args.t_extrap,
        n_obs_total  = args.n_obs,
        sigma        = args.sigma,
        n_col        = args.n_col,
        seed         = args.seed,
        lambda_phys  = args.lambda_phys,
        lambda_ic    = args.lambda_ic,
        hidden       = args.hidden,
        n_layers     = args.n_layers,
        lr           = args.lr,
        lr_step      = args.lr_step,
        lr_gamma     = args.lr_gamma,
        lr_param     = args.lr_param,
        n_epochs     = args.n_epochs,
        print_every  = args.print_every,
        log_every    = args.log_every,
        patience     = args.patience,
        min_delta    = args.min_delta,
        out_dir      = args.out_dir,
        ini_guess_v  = args.v_init,
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device()
    print("=" * 70)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram_gb:.1f} GB")
    print(f"  ν    : {cfg.v:.4f}    Situation : {cfg.situation}")
    print(f"  ν_init : {cfg.ini_guess_v:.4f}")
    print("=" * 70)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_pinn_ext_phys = out_dir / f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_ckpt_pinn_ext_phys}"
    ckpt_pinn_blind    = out_dir / f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_ckpt_pinn_blind}"
    results_path = out_dir / f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_results_pt}"

    # ── Data ──────────────────────────────────────────────────────────────────
    data = generate_data(cfg, device)
    n_train = len(data["t_obs_train"])
    n_val   = len(data["t_obs_val"])
    print(f"\n  Observations: {cfg.n_obs_total} total  →  {n_train} train / {n_val} val")
    print(f"  Collocation : {cfg.n_col} pts over [{cfg.x_begin}, {cfg.x_end}] × [{cfg.t0}, {cfg.t_extrap}]")
    print(f"  IC enforced at t={cfg.t0} via L_ic (not in observations)\n")

    # ── PINN model (extended physics) ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Training PINN  (data + physics (normal + extrapolated) + IC loss)")
    print("=" * 70)
    model_pinn = FCNet.from_config(cfg)
    print(f"  Parameters: {model_pinn.param_count()}")
    hist_pinn_ext_phys, snaps_pinn_ext_phys = train_model(
        model_pinn, data, cfg, device,
        use_physics = True,
        extrapolated_physics = True,
        label       = "PINN (ext. phy.)",
        ckpt_path   = ckpt_pinn_ext_phys,
    )
        # ── PINN model (blind) ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Training PINN  (data + physics (only normal region) + IC loss)")
    print("=" * 70)
    model_pinn = FCNet.from_config(cfg)
    print(f"  Parameters: {model_pinn.param_count()}")
    hist_pinn_blind, snaps_pinn_blind = train_model(
        model_pinn, data, cfg, device,
        use_physics = True,
        extrapolated_physics = False,
        label       = "PINN (blind)",
        ckpt_path   = ckpt_pinn_blind,
    )

    pred_pinn_ext_phys = Predictor(cfg, checkpoint_path=str(ckpt_pinn_ext_phys))
    pred_pinn_blind = Predictor(cfg, checkpoint_path=str(ckpt_pinn_blind))

    t_flattened_full  = data["t_flattened_full"]
    x_flattened_full  = data["x_flattened_full"]
    u_true_full  = data["u_true_full"]

    t_flattened_full  = data["t_flattened_full"]
    x_flattened_full  = data["x_flattened_full"]
    u_true_full  = data["u_true_full"]

    u_pinn_ext_phys_full = pred_pinn_ext_phys.predict(x_flattened_full, t_flattened_full)
    u_pinn_blind_full = pred_pinn_blind.predict(x_flattened_full, t_flattened_full)

    nu_hat_ext_phys = pred_pinn_ext_phys.predict_params()["v_hat"]
    nu_hat_blind = pred_pinn_blind.predict_params()["v_hat"]
    mask_train  = t_flattened_full <= cfg.t_train

    mask_extrap = t_flattened_full >  cfg.t_train

    rmse_pinn_ext_phys_train = Predictor.rmse(u_pinn_ext_phys_full[mask_train],  u_true_full[mask_train])
    rmse_pinn_blind_train = Predictor.rmse(u_pinn_blind_full[mask_train],  u_true_full[mask_train])
    rmse_pinn_ext_phys_extrap   = Predictor.rmse(u_pinn_ext_phys_full[mask_extrap], u_true_full[mask_extrap])
    rmse_pinn_blind_extrap = Predictor.rmse(u_pinn_blind_full[mask_extrap], u_true_full[mask_extrap])
    phys_pinn_ext_phys       = Predictor.physics_residual(u_pinn_ext_phys_full, x_flattened_full, t_flattened_full, cfg)
    phys_pinn_blind       = Predictor.physics_residual(u_pinn_blind_full, x_flattened_full, t_flattened_full, cfg)
    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY  (best-val checkpoint)")
    print("=" * 70)
    print(f"  Device : {device}")
    print(f"  {'Metric':<38} {'PINN (extended physics)':>10} {'PINN (blind)':>10}")
    print(f"  {'RMSE  (training interval)':38}   {rmse_pinn_ext_phys_train:>10.4f}  {rmse_pinn_blind_train:>10.4f}")
    print(f"  {'RMSE  (extrapolation)':38}   {rmse_pinn_ext_phys_extrap:>10.4f}  {rmse_pinn_blind_extrap:>10.4f}")
    print(f"  {'Physics Residual':38}   {phys_pinn_ext_phys:>10.4f}  {phys_pinn_blind:>10.4f}")
    print(f"  {'True nu':38}  {cfg.v:>10.4f}")
    print(f"  {'nu_hat':38}  {nu_hat_ext_phys:>10.4f} {nu_hat_blind:>10.4f}")
    print("  " + "-" * 62)


    # ── Save results bundle for plot.py ───────────────────────────────────────
    metrics = {
        "rmse_pinn_ext_phys_train": rmse_pinn_ext_phys_train,
        "rmse_pinn_blind_train": rmse_pinn_blind_train,
        "rmse_pinn_ext_phys_extrap": rmse_pinn_ext_phys_extrap,
        "rmse_pinn_blind_extrap": rmse_pinn_blind_extrap,
        "phys_pinn_ext_phys": phys_pinn_ext_phys,
        "phys_pinn_blind": phys_pinn_blind,
        "nu_hat_ext_phys": nu_hat_ext_phys,
        "nu_hat_blind": nu_hat_blind,
    }

    torch.save(
        {
            "config":       cfg,
            "device_str":   str(device),
            "data":         {k: v for k, v in data.items()
                             if isinstance(v, np.ndarray)},   # numpy only
            "hist_pinn_ext_phys":    hist_pinn_ext_phys,
            "hist_pinn_blind":    hist_pinn_blind,
            "snaps_pinn_ext_phys":   snaps_pinn_ext_phys,
            "snaps_pinn_blind":   snaps_pinn_blind,
            "u_pinn_ext_phys_full":  u_pinn_ext_phys_full,
            "u_pinn_blind_full":  u_pinn_blind_full,
            "metrics":      metrics,
            "time_shockwave": cfg.inviscid_shockwave_time,
        },
        results_path,
    )
    print(f"\n  Results saved to  {results_path}")
    print(f"  Best PINN (extended physics) checkpoint : {ckpt_pinn_ext_phys}")
    print(f"  Best PINN (blind) checkpoint : {ckpt_pinn_blind}")
    print("\n  Run  python plot.py  to generate figures.\n")

def evaluate_model() -> None:
    
    # ── Load best checkpoints for final evaluation ────────────────────────────
    # Using best-val checkpoints rather than last-epoch weights ensures
    # the saved result reflects the model at its generalisation peak.
    args = parse_args()
    # ── Build config from CLI arguments ───────────────────────────────────────
    cfg = BurgerConfig(
        v            = args.viscosity,
        situation    = args.situation,
        x_begin      = args.x_begin,
        x_end        = args.x_end,
        t0           = args.t0,
        t_train      = args.t_train,
        t_extrap     = args.t_extrap,
        n_obs_total  = args.n_obs,
        sigma        = args.sigma,
        n_col        = args.n_col,
        seed         = args.seed,
        lambda_phys  = args.lambda_phys,
        lambda_ic    = args.lambda_ic,
        hidden       = args.hidden,
        n_layers     = args.n_layers,
        lr           = args.lr,
        lr_step      = args.lr_step,
        lr_gamma     = args.lr_gamma,
        n_epochs     = args.n_epochs,
        print_every  = args.print_every,
        log_every    = args.log_every,
        patience     = args.patience,
        min_delta    = args.min_delta,
        out_dir      = args.out_dir,
    )
    out_dir = Path(cfg.out_dir)
    results_path = out_dir / f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_results_pt}"
    raw = torch.load(results_path, map_location="cpu", weights_only=False)
    config = raw["config"]
    data = raw["data"]
    ckpt_pinn_ext_phys = out_dir / f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_ckpt_pinn_ext_phys}"
    ckpt_pinn_blind    = out_dir / f"{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_ckpt_pinn_blind}"

    pred_pinn_ext_phys = Predictor(config, checkpoint_path=str(ckpt_pinn_ext_phys))
    pred_pinn_blind    = Predictor(config, checkpoint_path=str(ckpt_pinn_blind))

    t_flattened_full  = data["t_flattened_full"]
    x_flattened_full  = data["x_flattened_full"]
    u_true_full  = data["u_true_full"]

    u_pinn_ext_phys_full = pred_pinn_ext_phys.predict(x_flattened_full, t_flattened_full)
    u_pinn_blind_full    = pred_pinn_blind.predict(x_flattened_full, t_flattened_full)

    nu_hat_ext_phys = pred_pinn_ext_phys.predict_params()["v_hat"]
    nu_hat_blind = pred_pinn_blind.predict_params()["v_hat"]

    mask_train  = t_flattened_full <= config.t_train

    mask_extrap = t_flattened_full >  config.t_train
    mask_extrap = t_flattened_full >  cfg.t_train

    rmse_pinn_ext_phys_train = Predictor.rmse(u_pinn_ext_phys_full[mask_train],  u_true_full[mask_train])
    rmse_pinn_blind_train = Predictor.rmse(u_pinn_blind_full[mask_train],  u_true_full[mask_train])
    rmse_pinn_ext_phys_extrap   = Predictor.rmse(u_pinn_ext_phys_full[mask_extrap], u_true_full[mask_extrap])
    rmse_pinn_blind_extrap = Predictor.rmse(u_pinn_blind_full[mask_extrap], u_true_full[mask_extrap])
    phys_pinn_ext_phys       = Predictor.physics_residual(u_pinn_ext_phys_full, x_flattened_full, t_flattened_full, cfg)
    phys_pinn_blind       = Predictor.physics_residual(u_pinn_blind_full, x_flattened_full, t_flattened_full, cfg)
    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY  (best-val checkpoint)")
    print("=" * 70)
    print(f"  {'Metric':<38} {'PINN (extended physics)':>10} {'PINN (blind)':>10}")
    print(f"  {'RMSE  (training interval)':38}   {rmse_pinn_ext_phys_train:>10.4f}  {rmse_pinn_blind_train:>10.4f}")
    print(f"  {'RMSE  (extrapolation)':38}   {rmse_pinn_ext_phys_extrap:>10.4f}  {rmse_pinn_blind_extrap:>10.4f}")
    print(f"  {'Physics Residual':38}   {phys_pinn_ext_phys:>10.4f}  {phys_pinn_blind:>10.4f}")
    print(f"  {'True nu':38}  {cfg.v:>10.4f}")
    print(f"  {'nu_hat':38}  {nu_hat_ext_phys:>10.4f} {nu_hat_blind:>10.4f}")
    print("  " + "-" * 62)


if __name__ == "__main__":
    main_training()
    evaluate_model()