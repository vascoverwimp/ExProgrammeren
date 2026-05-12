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
from scipy.special import erfc
from model_claude import BurgerConfig, FCNet, Predictor


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

def analytic(x: np.ndarray, t: np.ndarray, cfg: BurgerConfig) -> np.ndarray:
    """
    Analytic solution of the Burger's equation for three different regimes:
    N-wave: starting with u(x,t0) = exp(-(x-1)**2/2) - exp(-(x+1)**2/2) the solution evolves into a characteristic N-wave shape.   
    """
    shifted_t = t - cfg.t0
    if cfg.situation == "Step":
        
        mask = shifted_t > 1e-10 
    
        # Initialize output array with the initial condition (t <= t0)
        # Defaulting to the step function logic
        u = np.where(x > 0, 1.0, 0.0)
        
        # Only calculate the complex viscous formula where time has actually progressed
        if np.any(mask):
            # Extract only the points that need the viscous calculation
            tm = shifted_t[mask]
            xm = x[mask]
            
            sqrt_4nut = np.sqrt(4 * cfg.v * tm)
            
            # Term A: erfc(-x / sqrt)
            term_a = erfc(-xm / sqrt_4nut)
            
            # Term B: exp(exponent) * erfc((t-x)/sqrt)
            exponent = (xm - 0.5 * tm) / (2 * cfg.v)
            # Standard float64 max exponent is ~709. 
            # Clipping at 500 is safe and effectively "infinite" for a ratio.
            exponent = np.clip(exponent, -1e7, 1e7)
            
            term_b = np.exp(exponent) * erfc((tm - xm) / sqrt_4nut)
            
            # Calculate ratio safely
            # Adding a tiny epsilon to the denominator prevents 0/0
            u_viscous = term_b / (term_a + term_b + 1e-14)
            
            # Place the calculated values back into the main array
            u[mask] = u_viscous

        return u
    
    return x*t




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

    # ── All M observations ────────────────────────────────────────────────────
    x_full, t_all = np.random.uniform(cfg.x_begin, cfg.x_end, cfg.n_obs), np.random.uniform(cfg.t0, cfg.t_train, cfg.n_obs)  # spatial and temporal locations for observations
    u_all = analytic(x_full, t_all, cfg) + np.random.normal(0.0, cfg.sigma, cfg.n_obs)

    # ── Train / validation split  (stratified: sorted time, interleaved) ──────
    n_val   = max(1, int(np.round(cfg.n_obs * cfg.val_fraction)))
    
    # Every k-th index goes to validation to spread validation points evenly
    # across the time axis rather than bunching them at one end.
    val_idx   = np.round(np.linspace(0, cfg.n_obs - 1, n_val)).astype(int)
    train_idx = np.array([i for i in range(cfg.n_obs) if i not in val_idx])

    t_obs_train, u_obs_train, x_obs_train = t_all[train_idx], u_all[train_idx], x_full[train_idx]
    t_obs_val,   u_obs_val,   x_obs_val   = t_all[val_idx],   u_all[val_idx],   x_full[val_idx]
    # ── Collocation points (randomly chosen) ───────────────────────────────────
    t_col      = np.random.uniform(cfg.t0, cfg.t_extrap, cfg.n_col_t)
    x_col      = np.random.uniform(cfg.x_begin, cfg.x_end, cfg.n_col_x)
    u_col_true = analytic(x_col, t_col, cfg)       # used only in plots

    # ── Initial condition point ───────────────────────────────────────────────
    x_samples_ic = np.linspace(cfg.x_begin, cfg.x_end, cfg.n_ic_samples_x)
    t_ic = np.full_like(x_samples_ic, cfg.t0)

    # ── Dense grids for post-hoc evaluation and plotting (CPU numpy only) ────
    t_plot_train = np.linspace(cfg.t0, cfg.t_train,  300)
    t_plot_full  = np.linspace(cfg.t0, cfg.t_extrap, 500)
    x_plot_full = np.linspace(cfg.x_begin, cfg.x_end, 500)

    t_plotmat_train, x_plotmat_train = np.meshgrid(t_plot_train, x_plot_full, indexing="ij")
    t_plotmat_full,  x_plotmat_full  = np.meshgrid(t_plot_full,  x_plot_full,  indexing="ij")
    t_flattened_train = t_plotmat_train.reshape(-1)
    x_flattened_train = x_plotmat_train.reshape(-1)
    t_flattened_full  = t_plotmat_full.reshape(-1)
    x_flattened_full  = x_plotmat_full.reshape(-1)

    u_true_train = analytic(x_flattened_train, t_flattened_train, cfg)
    u_true_full  = analytic(x_flattened_full, t_flattened_full,  cfg)

    return {
        # numpy — full observation set (used only in plots)
        "t_obs":       t_all,
        "u_obs":       u_all,
        # numpy — train split
        "t_obs_train": t_obs_train,
        "u_obs_train": u_obs_train,
        # numpy — validation split
        "t_obs_val":   t_obs_val,
        "u_obs_val":   u_obs_val,
        # numpy — collocation & IC (also converted to tensors below)
        "t_col":       t_col,
        "x_col":       x_col,
        "u_col_true":  u_col_true,
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
        # tensors on DEVICE — collocation (requires_grad for ODE residual)
        "t_col_t": to_tensor(t_col, requires_grad=True),
        "x_col_t": to_tensor(x_col, requires_grad=True),
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

    du = torch.autograd.grad(
        u_hat, x,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True,
    )[0]

    d2u = torch.autograd.grad(
        du, x,
        grad_outputs=torch.ones_like(du),
        create_graph=True,
    )[0]

    dotu = torch.autograd.grad(
        u_hat, t,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True,
    )[0]

    residual = dotu + u_hat * du - cfg.v * d2u
    return torch.mean(residual ** 2)


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
    label:       str,
    ckpt_path:   Path,
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
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=cfg.lr_step, gamma=cfg.lr_gamma
    )

    t_obs_train_t = data["t_obs_train_t"]
    x_obs_train_t = data["x_obs_train_t"]
    u_obs_train_t = data["u_obs_train_t"]
    x_obs_val_t   = data["x_obs_val_t"]
    t_obs_val_t   = data["t_obs_val_t"]
    u_obs_val_t   = data["u_obs_val_t"]
    t_col_t       = data["t_col_t"]
    x_col_t       = data["x_col_t"]
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

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        # Data loss — MSE on training observations only (not validation)
        u_pred   = model(x_obs_train_t, t_obs_train_t)
        l_data   = torch.mean((u_pred - u_obs_train_t) ** 2)

        if use_physics:
            l_phys = loss_physics(model, x_col_t, t_col_t, cfg)
            l_ic   = loss_ic(model, x_samples_ic_t, t_ic_t, cfg)
            l_total = l_data + cfg.lambda_phys * l_phys + cfg.lambda_ic * l_ic
        else:
            l_phys  = torch.zeros(1, device=device)
            l_ic    = torch.zeros(1, device=device)
            l_total = l_data

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
    g.add_argument("--n_obs",        type=int,   default=BurgerConfig.n_obs,   help="Total noisy observations")
    g.add_argument("--val_fraction", type=float, default=BurgerConfig.val_fraction,  help="Fraction of obs for validation")
    g.add_argument("--sigma",        type=float, default=BurgerConfig.sigma, help="Measurement noise std dev")
    g.add_argument("--n_col_x",      type=int,   default=BurgerConfig.n_col_x,  help="Collocation points in x")
    g.add_argument("--n_col_t",      type=int,   default=BurgerConfig.n_col_t,  help="Collocation points in t")
    g.add_argument("--seed",         type=int,   default=BurgerConfig.seed,   help="RNG seed")

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
        n_obs        = args.n_obs,
        val_fraction = args.val_fraction,
        sigma        = args.sigma,
        n_col_t      = args.n_col_t,
        n_col_x      = args.n_col_x,
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
    print("=" * 70)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_pinn = out_dir / cfg.ckpt_pinn
    ckpt_ml   = out_dir / cfg.ckpt_ml
    results_path = out_dir / cfg.results_pt

    # ── Data ──────────────────────────────────────────────────────────────────
    data = generate_data(cfg, device)
    n_train = len(data["t_obs_train"])
    n_val   = len(data["t_obs_val"])
    print(f"\n  Observations: {cfg.n_obs} total  →  {n_train} train / {n_val} val")
    print(f"  Collocation : {cfg.n_col_t * cfg.n_col_x} pts over [{cfg.x_begin}, {cfg.x_end}] × [{cfg.t0}, {cfg.t_extrap}]")
    print(f"  IC enforced at t={cfg.t0} via L_ic (not in observations)\n")

    # ── Standard ML model ─────────────────────────────────────────────────────
    print("=" * 70)
    print("Training STANDARD ML  (data loss only)")
    print("=" * 70)
    model_ml = FCNet.from_config(cfg)
    print(f"  Parameters: {model_ml.param_count()}")
    hist_ml, snaps_ml = train_model(
        model_ml, data, cfg, device,
        use_physics = False,
        label       = "StdML",
        ckpt_path   = ckpt_ml,
    )

    # ── PINN model ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Training PINN  (data + physics + IC loss)")
    print("=" * 70)
    model_pinn = FCNet.from_config(cfg)
    print(f"  Parameters: {model_pinn.param_count()}")
    hist_pinn, snaps_pinn = train_model(
        model_pinn, data, cfg, device,
        use_physics = True,
        label       = "PINN ",
        ckpt_path   = ckpt_pinn,
    )
    pred_ml   = Predictor(cfg, checkpoint_path=str(ckpt_ml))
    pred_pinn = Predictor(cfg, checkpoint_path=str(ckpt_pinn))

    t_flattened_full  = data["t_flattened_full"]
    x_flattened_full  = data["x_flattened_full"]
    u_true_full  = data["u_true_full"]

    u_ml_full   = pred_ml.predict(x_flattened_full, t_flattened_full)
    u_pinn_full = pred_pinn.predict(x_flattened_full, t_flattened_full)

    mask_train  = t_flattened_full <= cfg.t_train
    print(mask_train.shape)
    print(mask_train)
    mask_extrap = t_flattened_full >  cfg.t_train

    rmse_ml_train   = Predictor.rmse(u_ml_full[mask_train],   u_true_full[mask_train])
    rmse_pinn_train = Predictor.rmse(u_pinn_full[mask_train],  u_true_full[mask_train])
    rmse_ml_ext     = Predictor.rmse(u_ml_full[mask_extrap],  u_true_full[mask_extrap])
    rmse_pinn_ext   = Predictor.rmse(u_pinn_full[mask_extrap], u_true_full[mask_extrap])
    phys_ml         = Predictor.physics_residual(u_ml_full, x_flattened_full, t_flattened_full, cfg)
    phys_pinn       = Predictor.physics_residual(u_pinn_full, x_flattened_full, t_flattened_full, cfg)
    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY  (best-val checkpoint)")
    print("=" * 70)
    print(f"  Device : {device}")
    print(f"  {'Metric':<38}  {'Std ML':>10}  {'PINN':>10}")
    print("  " + "-" * 62)
    print(f"  {'RMSE  (training interval)':38}  {rmse_ml_train:>10.4f}  {rmse_pinn_train:>10.4f}")
    print(f"  {'RMSE  (extrapolation)':38}  {rmse_ml_ext:>10.4f}  {rmse_pinn_ext:>10.4f}")
    print(f"  {'Physics Residual':38}  {phys_ml:>10.4f}  {phys_pinn:>10.4f}")

    # ── Save results bundle for plot.py ───────────────────────────────────────
    metrics = {
        "rmse_ml_train":   rmse_ml_train,
        "rmse_pinn_train": rmse_pinn_train,
        "rmse_ml_ext":     rmse_ml_ext,
        "rmse_pinn_ext":   rmse_pinn_ext,
        "phys_ml":         phys_ml,
        "phys_pinn":       phys_pinn,
    }

    torch.save(
        {
            "config":       cfg,
            "device_str":   str(device),
            "data":         {k: v for k, v in data.items()
                             if isinstance(v, np.ndarray)},   # numpy only
            "hist_ml":      hist_ml,
            "hist_pinn":    hist_pinn,
            "snaps_ml":     snaps_ml,
            "snaps_pinn":   snaps_pinn,
            "u_ml_full":    u_ml_full,
            "u_pinn_full":  u_pinn_full,
            "metrics":      metrics,
        },
        results_path,
    )
    print(f"\n  Results saved to  {results_path}")
    print(f"  Best ML   checkpoint : {ckpt_ml}")
    print(f"  Best PINN checkpoint : {ckpt_pinn}")
    print("\n  Run  python plot.py  to generate figures.\n")

def evaluate_model() -> None:
    
    # ── Load best checkpoints for final evaluation ────────────────────────────
    # Using best-val checkpoints rather than last-epoch weights ensures
    # the saved result reflects the model at its generalisation peak.
    
    out_dir = Path(BurgerConfig.out_dir)
    results_path = out_dir / BurgerConfig.results_pt
    raw = torch.load(results_path, map_location="cpu", weights_only=False)
    config = raw["config"]
    data = generate_data(config, torch.device("cpu"))  # CPU-only for evaluation and plotting
    ckpt_pinn = out_dir / config.ckpt_pinn
    ckpt_ml   = out_dir / config.ckpt_ml
    pred_ml   = Predictor(config, checkpoint_path=str(ckpt_ml))
    pred_pinn = Predictor(config, checkpoint_path=str(ckpt_pinn))

    t_flattened_full  = data["t_flattened_full"]
    x_flattened_full  = data["x_flattened_full"]
    u_true_full  = analytic(x_flattened_full, t_flattened_full, config)

    u_ml_full   = pred_ml.predict(x_flattened_full, t_flattened_full)
    u_pinn_full = pred_pinn.predict(x_flattened_full, t_flattened_full)

    mask_train  = t_flattened_full <= config.t_train
    print(mask_train.shape)
    print(np.sum(np.isnan(u_true_full[mask_train])))
    mask_extrap = t_flattened_full >  config.t_train

    rmse_ml_train   = Predictor.rmse(u_ml_full[mask_train],   u_true_full[mask_train])
    rmse_pinn_train = Predictor.rmse(u_pinn_full[mask_train],  u_true_full[mask_train])
    rmse_ml_ext     = Predictor.rmse(u_ml_full[mask_extrap],  u_true_full[mask_extrap])
    rmse_pinn_ext   = Predictor.rmse(u_pinn_full[mask_extrap], u_true_full[mask_extrap])
    phys_ml         = Predictor.physics_residual(u_ml_full, x_flattened_full, t_flattened_full, config)
    phys_pinn       = Predictor.physics_residual(u_pinn_full, x_flattened_full, t_flattened_full, config)
    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY  (best-val checkpoint)")
    print("=" * 70)
    print(f"  {'Metric':<38}  {'Std ML':>10}  {'PINN':>10}")
    print("  " + "-" * 62)
    print(f"  {'RMSE  (training interval)':38}  {rmse_ml_train:>10.4f}  {rmse_pinn_train:>10.4f}")
    print(f"  {'RMSE  (extrapolation)':38}  {rmse_ml_ext:>10.4f}  {rmse_pinn_ext:>10.4f}")
    print(f"  {'Physics Residual':38}  {phys_ml:>10.4f}  {phys_pinn:>10.4f}")



if __name__ == "__main__":
    main_training()
    # evaluate_model()