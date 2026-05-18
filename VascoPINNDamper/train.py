"""
train.py
========
Training script for the PINN damped-spring-mass system.

Every tunable value is exposed as a CLI argument and collected into a
DamperConfig before anything else runs.  The script:

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

from model import DamperConfig, FCNet, Predictor


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

def analytic(t: np.ndarray, cfg: DamperConfig) -> np.ndarray:
    zeta, w0, wd, y0, dy0 = cfg.zeta, cfg.omega_0, cfg.omega_d, cfg.y0, cfg.dy0
    t = np.asarray(t)
    
    if zeta < 1:  # Underdamped
        return np.exp(-zeta * w0 * t) * (
            y0 * np.cos(wd * t) + (dy0 + zeta * w0 * y0) / wd * np.sin(wd * t)
        )
    
    elif zeta == 1:  # Critically damped
        return (y0 + (dy0 + w0 * y0) * t) * np.exp(-w0 * t)
    
    else:  # Overdamped
        r1 = -w0 * (zeta - np.sqrt(zeta**2 - 1))
        r2 = -w0 * (zeta + np.sqrt(zeta**2 - 1))
        C1 = (dy0 - r2 * y0) / (r1 - r2)
        C2 = y0 - C1
        return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

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

def generate_data(cfg: DamperConfig, device: torch.device) -> dict:
    """
    Build all data arrays and tensors needed for training and plotting.

    Real-world workflow followed here
    ----------------------------------
    * Observations are randomly drawn from (0, t_train] so t=0 is never
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
    t_obs_train = np.random.uniform(0, cfg.t_train, cfg.n_obs)
    y_obs_train = analytic(t_obs_train, cfg) + np.random.normal(0.0, cfg.sigma, cfg.n_obs)

    # ── Train / validation split  (stratified) ────── We found were told to use the ground truth,
    # instead of stratified time split, but this is more realistic, so we will keep it in the code.
    # train_idx, val_idx = stratified_time_split(t_all, 0, cfg.t_train, cfg.val_fraction, cfg.n_bins, cfg.seed)

    # t_obs_train, y_obs_train = t_all[train_idx], y_all[train_idx]
    # t_obs_val,   y_obs_val   = t_all[val_idx],   y_all[val_idx]

    t_obs_val = np.linspace(0, cfg.t_train, cfg.n_val)
    y_obs_val = analytic(t_obs_val, cfg)

    # ── Initial condition point ───────────────────────────────────────────────
    t_ic = np.array([0.0])

    # ── Dense grids for post-hoc evaluation and plotting (CPU numpy only) ────
    t_plot_full = np.linspace(0.0, cfg.t_extrap,  cfg.n_plot_t)
    t_plot_train  = t_plot_full[t_plot_full <= cfg.t_train]
    y_true_train = analytic(t_plot_train, cfg)
    y_true_full  = analytic(t_plot_full,  cfg)

    return {
        # numpy — train split
        "t_obs_train": t_obs_train,
        "y_obs_train": y_obs_train,
        # numpy — validation split
        "t_obs_val":   t_obs_val,
        "y_obs_val":   y_obs_val,
        # numpy — dense evaluation grids
        "t_plot_train": t_plot_train,
        "t_plot_full":  t_plot_full,
        "y_true_train": y_true_train,
        "y_true_full":  y_true_full,
        # tensors on DEVICE — training observations
        "t_obs_train_t": to_tensor(t_obs_train),
        "y_obs_train_t": to_tensor(y_obs_train),
        # tensors on DEVICE — validation observations
        "t_obs_val_t":   to_tensor(t_obs_val),
        "y_obs_val_t":   to_tensor(y_obs_val),
        # tensor on DEVICE — IC point (requires_grad for y'(0))
        "t_ic_t":  to_tensor(t_ic,  requires_grad=True),
    }


# =============================================================================
# 3.  LOSS COMPONENTS
# =============================================================================

def loss_physics(
    model: nn.Module,
    t: torch.Tensor,
    cfg: DamperConfig,
) -> torch.Tensor:
    """
    L_physics = mean( r(tᵢ)² )  over all collocation points.

    r(t) = m·ŷ''(t) + c·ŷ'(t) + k·ŷ(t)

    Both derivatives are computed via automatic differentiation.
    create_graph=True on the first grad call keeps the computation graph
    alive so the backward pass through the second derivative works.
    """
    y_hat = model(t)

    dy = torch.autograd.grad(
        y_hat, t,
        grad_outputs=torch.ones_like(y_hat),
        create_graph=True,
    )[0]

    d2y = torch.autograd.grad(
        dy, t,
        grad_outputs=torch.ones_like(dy),
        create_graph=True,
    )[0]

    residual = cfg.mass * d2y + cfg.damping * dy + cfg.stiffness * y_hat
    return torch.mean(residual ** 2)


def loss_ic(
    model: nn.Module,
    t: torch.Tensor,
    cfg: DamperConfig,
) -> torch.Tensor:
    """
    L_ic = ( ŷ(0) - y0 )² + ( ŷ'(0) - dy0 )²

    λ_ic >> λ_phys because an error at t=0 propagates and distorts the
    entire downstream trajectory.
    """
    y_hat_0 = model(t)
    dy_0 = torch.autograd.grad(
        y_hat_0, t,
        grad_outputs=torch.ones_like(y_hat_0),
        create_graph=True,
    )[0]
    return torch.mean((y_hat_0 - cfg.y0) ** 2 + (dy_0 - cfg.dy0) ** 2)


# =============================================================================
# 4.  TRAINING LOOP
# =============================================================================

def train_model(
    model:       nn.Module,
    data:        dict,
    cfg:         DamperConfig,
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

    torch.manual_seed(cfg.seed)

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=cfg.lr_step, gamma=cfg.lr_gamma
    )

    t_obs_train_t = data["t_obs_train_t"]
    y_obs_train_t = data["y_obs_train_t"]
    t_obs_val_t   = data["t_obs_val_t"]
    y_obs_val_t   = data["y_obs_val_t"]
    t_ic_t        = data["t_ic_t"]

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

    t_col_pool_extrap = torch.rand(cfg.n_col_pool, device=device, requires_grad=True)*(cfg.t_extrap - 0.0) + 0.0
    t_col_pool_train  = torch.rand(cfg.n_col_pool, device=device, requires_grad=True)*(cfg.t_train - 0.0) + 0.0

    t_selected_epoch = t_obs_train_t
    y_selected_epoch = y_obs_train_t
    
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimiser.zero_grad()
        
        # # Randomly select a subset of the training observations for this epoch to speed up training and add noise robustness.  This is a form of stochastic mini-batching.
        # epoch_random_selection = torch.randperm(t_obs_train_t.shape[0])[:cfg.n_obs_per_epoch]
        # t_selected_epoch = t_obs_train_t[epoch_random_selection]
        # y_selected_epoch = y_obs_train_t[epoch_random_selection]
        # We found that mini batching is worse than full batching


        # Data loss — MSE on training observations only (not validation)
        y_pred   = model(t_selected_epoch)
        l_data   = torch.mean((y_pred - y_selected_epoch) ** 2)
        l_total = l_data
        l_phys  = torch.zeros(1, device=device)
        l_ic    = torch.zeros(1, device=device)
        if use_physics:
            idx = torch.randint(0, cfg.n_col_pool, (cfg.n_col,))
            if extrapolated_physics:
                t_col = t_col_pool_extrap[idx]
            else:
                t_col = t_col_pool_train[idx]

            l_phys = loss_physics(model, t_col, cfg)

            l_total += cfg.lambda_phys * l_phys
        if use_ic:
            l_ic   = loss_ic(model, t_ic_t, cfg)
            l_total += cfg.lambda_ic * l_ic


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
                y_val_pred = model(t_obs_val_t)
                l_val = torch.mean((y_val_pred - y_obs_val_t) ** 2).item()

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
# 5.  ARGUMENT PARSING  — every DamperConfig field is a CLI flag
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PINN model for a damped spring-mass system with the constants as unknowns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physical parameters
    g = p.add_argument_group("Physical parameters")
    g.add_argument("--mass",      type=float, default=DamperConfig.mass,  help="Mass m [kg]")
    g.add_argument("--damping",   type=float, default=DamperConfig.damping,  help="Damping coefficient c")
    g.add_argument("--stiffness", type=float, default=DamperConfig.stiffness,  help="Spring stiffness k [N/m]")
    g.add_argument("--y0",        type=float, default=DamperConfig.y0,  help="Initial displacement y(0)")
    g.add_argument("--dy0",       type=float, default=DamperConfig.dy0,  help="Initial velocity y'(0)")

    # Time domain
    g = p.add_argument_group("Time domain")
    g.add_argument("--t_train",  type=float, default=DamperConfig.t_train,  help="End of training window [s]")
    g.add_argument("--t_extrap", type=float, default=DamperConfig.t_extrap, help="End of extrapolation window [s]")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--n_obs",        type=int,   default=DamperConfig.n_obs,   help="Total noisy observations")
    g.add_argument("--sigma",        type=float, default=DamperConfig.sigma, help="Measurement noise std dev")
    g.add_argument("--n_col",        type=int,   default=DamperConfig.n_col,  help="Collocation points")
    g.add_argument("--seed",         type=int,   default=DamperConfig.seed,   help="RNG seed")

    # Loss weights
    g = p.add_argument_group("PINN loss weights")
    g.add_argument("--lambda_phys", type=float, default=DamperConfig.lambda_phys, help="Physics residual weight")
    g.add_argument("--lambda_ic",   type=float, default=DamperConfig.lambda_ic, help="Initial-condition weight")

    # Architecture
    g = p.add_argument_group("Architecture")
    g.add_argument("--hidden",   type=int, default=DamperConfig.hidden, help="Neurons per hidden layer")
    g.add_argument("--n_layers", type=int, default=DamperConfig.n_layers,  help="Number of hidden layers")

    # Optimiser
    g = p.add_argument_group("Optimiser")
    g.add_argument("--lr",       type=float, default=DamperConfig.lr, help="Initial Adam learning rate")
    g.add_argument("--lr_step",  type=int,   default=DamperConfig.lr_step, help="StepLR decay interval [epochs]")
    g.add_argument("--lr_gamma", type=float, default=DamperConfig.lr_gamma,  help="StepLR decay factor")

    # Training loop
    g = p.add_argument_group("Training loop")
    g.add_argument("--n_epochs",    type=int, default=DamperConfig.n_epochs, help="Maximum training epochs")
    g.add_argument("--print_every", type=int, default=DamperConfig.print_every, help="Console log interval [epochs]")
    g.add_argument("--log_every",   type=int, default=DamperConfig.log_every,   help="History log interval [epochs]")

    # Early stopping
    g = p.add_argument_group("Early stopping")
    g.add_argument("--patience",  type=int,   default=DamperConfig.patience,   help="Patience in log_every units")
    g.add_argument("--min_delta", type=float, default=DamperConfig.min_delta, help="Min improvement to reset counter")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--out_dir", type=str, default=DamperConfig.out_dir, help="Output directory for all saved files")

    return p.parse_args()


# =============================================================================
# 6.  MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── Build config from CLI arguments ───────────────────────────────────────
    cfg = DamperConfig(
        mass         = args.mass,
        damping      = args.damping,
        stiffness    = args.stiffness,
        y0           = args.y0,
        dy0          = args.dy0,
        t_train      = args.t_train,
        t_extrap     = args.t_extrap,
        n_obs        = args.n_obs,
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
    print(f"  ζ      : {cfg.zeta:.4f}   ω₀ = {cfg.omega_0:.4f}   ωd = {cfg.omega_d:.4f}")
    print("=" * 70)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_ml   = out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_{cfg.suffix_ckpt_ml}"
    ckpt_pinn_ext_phys = out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_{cfg.suffix_ckpt_pinn_ext_phys}"
    ckpt_pinn_blind = out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_{cfg.suffix_ckpt_pinn_blind}"
    results_path = out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_{cfg.suffix_results_pt}"

    # ── Data ──────────────────────────────────────────────────────────────────
    data = generate_data(cfg, device)
    n_train = len(data["t_obs_train"])
    n_val   = len(data["t_obs_val"])
    print(f"\n  Noisy observations: {n_train} to train") 
    print(f" Perfect observations: {n_val} to validate (for early stopping and best checkpoint selection)")
    print(f"  Collocation : {cfg.n_col} pts over [0, {cfg.t_extrap}]")
    print(f"  IC enforced at t=0 via L_ic (not in observations)\n")

    # ── Standard ML model ─────────────────────────────────────────────────────
    print("=" * 70)
    print("Training STANDARD ML  (data loss only)")
    print("=" * 70)
    model_ml = FCNet.from_config(cfg)
    print(f"  Parameters: {model_ml.param_count()}")
    hist_ml, snaps_ml = train_model(
        model_ml, data, cfg, device,
        use_physics = False,
        extrapolated_physics=False,  # irrelevant when use_physics=False
        label       = "StdML",
        ckpt_path   = ckpt_ml,
    )

    # ── PINN model (normal + extrapolated) ──────────────────────────────────
    print()
    print("=" * 70)
    print("Training PINN  (data + physics (normal + extrapolated) + IC loss)")
    print("=" * 70)
    model_pinn = FCNet.from_config(cfg)
    print(f"  Parameters: {model_pinn.param_count()}")
    hist_pinn_ext_phys, snaps_pinn_ext_phys = train_model(
        model_pinn, data, cfg, device,
        use_physics = True,
        extrapolated_physics=True,  # physics loss on collocation points in extrapolation region
        label       = "PINN (phys. ext.)",
        ckpt_path   = ckpt_pinn_ext_phys,
    )
    # ── PINN model (blind) ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Training PINN  (data + physics (normal + extrapolated) + IC loss)")
    print("=" * 70)
    model_pinn = FCNet.from_config(cfg)
    print(f"  Parameters: {model_pinn.param_count()}")
    hist_pinn_blind, snaps_pinn_blind = train_model(
        model_pinn, data, cfg, device,
        use_physics = True,
        extrapolated_physics=False,  # physics loss on collocation points in extrapolation region
        label       = "PINN (blind)",
        ckpt_path   = ckpt_pinn_blind,
    )
    # ── Load best checkpoints for final evaluation ────────────────────────────
    # Using best-val checkpoints rather than last-epoch weights ensures
    # the saved result reflects the model at its generalisation peak.
    pred_ml   = Predictor(cfg, checkpoint_path=str(ckpt_ml))
    pred_pinn_ext_phys = Predictor(cfg, checkpoint_path=str(ckpt_pinn_ext_phys))
    pred_pinn_blind = Predictor(cfg, checkpoint_path=str(ckpt_pinn_blind))

    t_plot_full  = data["t_plot_full"]
    y_true_full  = data["y_true_full"]

    y_ml_full   = pred_ml.predict(t_plot_full)
    y_pinn_ext_phys_full = pred_pinn_ext_phys.predict(t_plot_full)
    y_pinn_blind_full = pred_pinn_blind.predict(t_plot_full)

    mask_train  = t_plot_full <= cfg.t_train
    mask_extrap = t_plot_full >  cfg.t_train

    rmse_ml_train   = Predictor.rmse(y_ml_full[mask_train],   y_true_full[mask_train])
    rmse_pinn_ext_phys_train = Predictor.rmse(y_pinn_ext_phys_full[mask_train],  y_true_full[mask_train])
    rmse_pinn_blind_train = Predictor.rmse(y_pinn_blind_full[mask_train],  y_true_full[mask_train])
    rmse_ml_extrap     = Predictor.rmse(y_ml_full[mask_extrap],  y_true_full[mask_extrap])
    rmse_pinn_ext_phys_extrap   = Predictor.rmse(y_pinn_ext_phys_full[mask_extrap], y_true_full[mask_extrap])
    rmse_pinn_blind_extrap   = Predictor.rmse(y_pinn_blind_full[mask_extrap], y_true_full[mask_extrap])
    phys_ml         = Predictor.physics_residual(y_ml_full,   t_plot_full, cfg)
    phys_pinn_ext_phys       = Predictor.physics_residual(y_pinn_ext_phys_full, t_plot_full, cfg)
    phys_pinn_blind          = Predictor.physics_residual(y_pinn_blind_full, t_plot_full, cfg)
    y0_ml                       = float(pred_ml.predict(np.array([0.0])))
    y0_pinn_ext_phys         = float(pred_pinn_ext_phys.predict(np.array([0.0])))
    y0_pinn_blind            = float(pred_pinn_blind.predict(np.array([0.0])))

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY  (best-val checkpoint)")
    print("=" * 70)
    print(f"  Device : {device}")
    print(f"  {'Metric':<38}  {'Std ML':>10}  {'PINN (phys. ext.)':>10}  {'PINN (blind)':>10}")
    print("  " + "-" * 62)
    print(f"  {'RMSE  (training interval)':38}  {rmse_ml_train:>10.4f}  {rmse_pinn_ext_phys_train:>10.4f}  {rmse_pinn_blind_train:>10.4f}")
    print(f"  {'RMSE  (extrapolation)':38}  {rmse_ml_extrap:>10.4f}  {rmse_pinn_ext_phys_extrap:>10.4f}  {rmse_pinn_blind_extrap:>10.4f}")
    print(f"  {'Physics residual  (full domain)':38}  {phys_ml:>10.4f}  {phys_pinn_ext_phys:>10.4f}  {phys_pinn_blind:>10.4f}")
    print(f"  {'ŷ(0)':38}  {y0_ml:>10.4f}  {y0_pinn_ext_phys:>10.4f}  {y0_pinn_blind:>10.4f}")

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
        "phys_pinn_blind":          phys_pinn_blind,
        "y0_ml":           y0_ml,
        "y0_pinn_ext_phys":         y0_pinn_ext_phys,
        "y0_pinn_blind":            y0_pinn_blind,
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
            "y_ml_full":    y_ml_full,
            "y_pinn_ext_phys_full":  y_pinn_ext_phys_full,
            "y_pinn_blind_full":   y_pinn_blind_full,
            "metrics":      metrics,
        },
        results_path,
    )
    print(f"\n  Results saved to  {results_path}")
    print(f"  Best ML   checkpoint : {ckpt_ml}")
    print(f"  Best PINN (phys. ext.) checkpoint : {ckpt_pinn_ext_phys}")
    print(f"  Best PINN (blind) checkpoint : {ckpt_pinn_blind}")
    print("\n  Run  python plot.py  to generate figures.\n")


def evaluate_blind(w0: float, zeta: float, suffix_results_pt=DamperConfig.suffix_results_pt) -> float:





    results_path = Path(f"{DamperConfig.out_dir}/w0{w0:.1e}_zeta{zeta:.1e}_{suffix_results_pt}")

    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: DamperConfig = raw["config"]
    data = raw["data"]

    out_dir = Path(cfg.out_dir)
    ckpt_pinn_blind = out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_{cfg.suffix_ckpt_pinn_blind}"

    pred_pinn_blind = Predictor(cfg, checkpoint_path=str(ckpt_pinn_blind))



    t_plot_full  = data["t_plot_full"]
    y_true_full  = data["y_true_full"]

    y_pinn_blind_full = pred_pinn_blind.predict(t_plot_full)

    mask_train  = t_plot_full <= cfg.t_train
    mask_extrap = t_plot_full >  cfg.t_train

    rmse_pinn_blind_train = Predictor.rmse(y_pinn_blind_full[mask_train],  y_true_full[mask_train])
    rmse_pinn_blind_extrap   = Predictor.rmse(y_pinn_blind_full[mask_extrap], y_true_full[mask_extrap])
    phys_pinn_blind          = Predictor.physics_residual(y_pinn_blind_full, t_plot_full, cfg)
    y0_pinn_blind            = float(pred_pinn_blind.predict(np.array([0.0])))

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY  (best-val checkpoint)")
    print("=" * 70)
    print(f"  {'Metric':<38}  {'PINN (blind)':>10}")
    print("  " + "-" * 62)
    print(f"  {'RMSE  (training interval)':38}  {rmse_pinn_blind_train:>10.4f}")
    print(f"  {'RMSE  (extrapolation)':38}  {rmse_pinn_blind_extrap:>10.4f}")
    print(f"  {'Physics residual  (full domain)':38}  {phys_pinn_blind:>10.4f}")
    print(f"  {'ŷ(0)':38}  {y0_pinn_blind:>10.4f}")

    return rmse_pinn_blind_train

if __name__ == "__main__":
    main()
    default_cfg = DamperConfig()
    default_w0 = default_cfg.omega_0
    default_zeta = default_cfg.zeta
    # evaluate_blind(default_w0, default_zeta)