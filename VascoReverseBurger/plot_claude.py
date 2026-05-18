"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
figures for the Burgers' equation dual PINN comparison:

  Fig 1  burger_fig1_summary.png          — 2×3 heatmap overview + loss curves
  Fig 2  burger_fig2_slices.png           — u(x) profiles at fixed time slices
  Fig 3  burger_fig3_nu_evolution.png     — ν (viscosity) parameter evolution
  Fig 4  burger_fig4_pinn_blind_epochs.png    — PINN_blind field snapshots per epoch
  Fig 5  burger_fig5_pinn_ext_phys_epochs.png — PINN_ext_phys field snapshots per epoch

Because the solution is a 2-D field u(x, t), visualisation uses:
  · pcolormesh heatmaps  (x on horizontal axis, t on vertical axis)
  · line plots at fixed t (time slices) so pointwise accuracy is legible

All flat prediction arrays are reshaped to (n_t, n_x) grids via
_to_grid(); the shape is inferred from the unique values stored in
the data bundle — no hard-coded grid sizes.

Usage
-----
  python plot.py                                       # default results file
  python plot.py --results ./run_01/training_results.pt
  python plot.py --results training_results.pt --out_dir ./figs
  python plot.py --no_show                             # save only, no plt.show()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

from model import BurgerConfig, Predictor


# =============================================================================
# COLOUR PALETTE
# =============================================================================

TEAL_BLIND  = "#1D9E75"   # PINN blind prediction
ORANGE_EXT  = "#EF9F27"   # PINN ext_phys prediction
GRAY_TRUE   = "#888780"   # true solution / neutral lines
PURPLE_IC   = "#7F77DD"   # initial conditions
LGRAY       = "#D3D1C7"   # spines
BG          = "#FAFAF8"   # figure background
PANEL       = "#F1EFE8"   # axes background

CMAP_FIELD = "RdBu_r"    # diverging — for signed velocity field u
CMAP_ERROR = "Oranges"   # sequential — for |error|


# =============================================================================
# SHARED HELPERS
# =============================================================================

def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def _to_grid(flat: np.ndarray, data: dict, which: str = "full") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape a flat prediction array back to a 2-D (n_t, n_x) grid.

    The meshgrid was built with indexing="ij":
        t_mat, x_mat = meshgrid(t_plot, x_plot, indexing="ij")
    so the flat layout is: row-major over (t, x), i.e. for a given t index
    all x values appear consecutively.

    Returns
    -------
    t_vals  : 1-D array of unique t values  (length n_t)
    x_vals  : 1-D array of unique x values  (length n_x)
    grid    : 2-D array of shape (n_t, n_x)
    """
    key_t = f"t_flattened_{which}"
    key_x = f"x_flattened_{which}"
    t_flat = data[key_t]
    x_flat = data[key_x]

    t_vals = np.unique(t_flat)   # sorted ascending
    x_vals = np.unique(x_flat)
    n_t, n_x = len(t_vals), len(x_vals)
    return t_vals, x_vals, flat.reshape(n_t, n_x)


def _colorbar(ax: plt.Axes, im, label: str = "") -> None:
    """Attach a slim, labelled colorbar to *ax* without resizing it."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def _heatmap(
    ax: plt.Axes,
    t_vals: np.ndarray,
    x_vals: np.ndarray,
    grid: np.ndarray,
    title: str,
    cfg: BurgerConfig,
    vmin=None, vmax=None,
    cmap=CMAP_FIELD,
    cbar_label: str = "u(x, t)",
    show_shockwave: bool = True,
) -> None:
    """
    Plot a pcolormesh heatmap with:
      · x on the horizontal axis
      · t on the vertical axis
      · dashed horizontal line at t = t_train marking the extrapolation boundary
      · dashed line at t = t_shockwave marking the predicted shockwave formation
    """
    style_ax(ax)
    im = ax.pcolormesh(
        x_vals, t_vals, grid,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", rasterized=True,
    )
    _colorbar(ax, im, label=cbar_label)
    
    # Training boundary
    ax.axhline(cfg.t_train, color="white", lw=1.2, ls="--", alpha=0.9)
    ax.text(
        x_vals[int(len(x_vals) * 0.02)], cfg.t_train + 0.02 * (t_vals[-1] - t_vals[0]),
        "extrap ↑", color="white", fontsize=7, va="bottom",
    )
    
    # Shockwave formation time (inviscid case)
    if show_shockwave:
        t_shock = cfg.inviscid_shockwave_time
        if t_vals[0] <= t_shock <= t_vals[-1]:
            ax.axhline(t_shock, color="#FF6B6B", lw=1.0, ls=":", alpha=0.8)
            ax.text(
                x_vals[int(len(x_vals) * 0.98)], t_shock + 0.01 * (t_vals[-1] - t_vals[0]),
                f"shock→", color="#FF6B6B", fontsize=7, va="bottom", ha="right", fontweight="bold"
            )
    
    ax.set_xlabel("x  [m]",    fontsize=8)
    ax.set_ylabel("time  [s]", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", pad=4, color="#444441")
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(t_vals[0],  t_vals[-1])


# =============================================================================
# FIGURE 1 — 2×3 summary  (heatmaps + loss curves for two PINNs)
# =============================================================================
#
#  ┌──────────────────┬──────────────────┬──────────────────┐
#  │    True u(x,t)   │  PINN blind      │  PINN ext_phys   │
#  ├──────────────────┼──────────────────┼──────────────────┤
#  │  blind |error│   │  ext_phys |error│  Loss curves     │
#  └──────────────────┴──────────────────┴──────────────────┘

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    cfg: BurgerConfig = bundle["cfg"]
    data              = bundle["data"]
    hist_blind        = bundle["hist_pinn_blind"]
    hist_ext_phys     = bundle["hist_pinn_ext_phys"]
    u_blind_flat      = bundle["u_pinn_blind_full"]
    u_ext_phys_flat   = bundle["u_pinn_ext_phys_full"]
    m                 = bundle["metrics"]
    device_str        = bundle["device_str"]

    u_true_flat = data["u_true_full"]

    t_vals, x_vals, grid_true = _to_grid(u_true_flat, data, "full")
    _,      _,      grid_blind = _to_grid(u_blind_flat, data, "full")
    _,      _,      grid_ext_phys = _to_grid(u_ext_phys_flat, data, "full")

    err_blind = np.abs(grid_blind - grid_true)
    err_ext_phys = np.abs(grid_ext_phys - grid_true)

    # Shared colour limits so the three field maps are directly comparable
    u_abs   = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u = -u_abs, u_abs
    vmax_err = max(err_blind.max(), err_ext_phys.max())

    fig = plt.figure(figsize=(17, 10))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.55, wspace=0.42,
        left=0.06, right=0.97, top=0.92, bottom=0.07,
    )
    ax_true  = fig.add_subplot(gs[0, 0])
    ax_blind = fig.add_subplot(gs[0, 1])
    ax_ext_phys = fig.add_subplot(gs[0, 2])
    ax_eb    = fig.add_subplot(gs[1, 0])
    ax_ee    = fig.add_subplot(gs[1, 1])
    ax_loss  = fig.add_subplot(gs[1, 2])

    # ── Row 0: field heatmaps ─────────────────────────────────────────────────
    _heatmap(ax_true, t_vals, x_vals, grid_true, "True  u(x, t)",
             cfg, vmin_u, vmax_u)
    _heatmap(ax_blind, t_vals, x_vals, grid_blind,
             f"PINN blind  (train RMSE={m['rmse_pinn_blind_train']:.4f},"
             f"  extrap RMSE={m['rmse_pinn_blind_extrap']:.4f})",
             cfg, vmin_u, vmax_u)
    _heatmap(ax_ext_phys, t_vals, x_vals, grid_ext_phys,
             f"PINN ext_phys  (train RMSE={m['rmse_pinn_ext_phys_train']:.4f},"
             f"  extrap RMSE={m['rmse_pinn_ext_phys_extrap']:.4f})",
             cfg, vmin_u, vmax_u)

    # ── Row 1: error maps ─────────────────────────────────────────────────────
    _heatmap(ax_eb, t_vals, x_vals, err_blind,
             f"PINN blind  |error|   (mean={err_blind.mean():.4f},"
             f"  max={err_blind.max():.4f})",
             cfg, vmin=0, vmax=vmax_err, cmap=CMAP_ERROR, cbar_label="|u_pred − u_true|")
    _heatmap(ax_ee, t_vals, x_vals, err_ext_phys,
             f"PINN ext_phys  |error|  (mean={err_ext_phys.mean():.4f},"
             f"  max={err_ext_phys.max():.4f})",
             cfg, vmin=0, vmax=vmax_err, cmap=CMAP_ERROR, cbar_label="|u_pred − u_true|")

    # ── Row 1 right: loss curves ───────────────────────────────────────────────
    style_ax(ax_loss)
    ax_loss.set_title("Training loss  (log scale)",
                      fontsize=9, loc="left", pad=4, color="#444441")
    ax_loss.semilogy(hist_blind["epoch"],   hist_blind["loss_data"],
                     color=TEAL_BLIND,   lw=1.5, ls="--",  label="Blind L_data")
    ax_loss.semilogy(hist_blind["epoch"],   hist_blind["loss_val"],
                     color=TEAL_BLIND,   lw=1.5, ls=":",   label="Blind L_val")
    ax_loss.semilogy(hist_ext_phys["epoch"], hist_ext_phys["loss_data"],
                     color=ORANGE_EXT,   lw=1.5,            label="Ext_phys L_data")
    ax_loss.semilogy(hist_ext_phys["epoch"], hist_ext_phys["loss_val"],
                     color=ORANGE_EXT,   lw=1.5, ls=":",   label="Ext_phys L_val")
    ax_loss.semilogy(hist_ext_phys["epoch"], hist_ext_phys["loss_physics"],
                     color=PURPLE_IC, lw=1.2, ls=":",   label="Ext_phys L_phys")
    ax_loss.semilogy(hist_ext_phys["epoch"], hist_ext_phys["loss_ic"],
                     color=PURPLE_IC, lw=1.2, ls="-.",  label="Ext_phys L_ic")
    ax_loss.semilogy(hist_ext_phys["epoch"], hist_ext_phys["loss_total"],
                     color=ORANGE_EXT,   lw=2.5, alpha=0.25, label="Ext_phys L_total")
    for ep in cfg.snapshot_epochs[:-1]:
        ax_loss.axvline(ep, color=GRAY_TRUE, lw=0.5, ls=":", alpha=0.4)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=6.5, framealpha=0.5)

    fig.suptitle(
        f"Burgers' PINN Dual Model Comparison  |  device={device_str}  |  ν_true={cfg.v}  "
        f"situation={cfg.situation}  |  "
        f"x∈[{cfg.x_begin}, {cfg.x_end}]  t∈[{cfg.t0}, {cfg.t_extrap}]\n"
        f"{cfg.n_obs_total} obs  σ={cfg.sigma}  |  "
        f"{cfg.n_col} random collocation pts every epoch |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"training window: t ≤ {cfg.t_train}  |  shockwave (inviscid): t ≈ {cfg.inviscid_shockwave_time:.3f}",
        fontsize=9, y=0.975, color="#2C2C2A",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 1 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 2 — time-slice profiles  u(x) at fixed t values (two PINNs)
# =============================================================================

def make_slice_figure(bundle: dict, out_path: Path,
                      n_slices: int = 8) -> plt.Figure:
    """
    Show u(x) profiles at *n_slices* fixed times spanning [t0, t_extrap].

    Compares true vs PINN_blind vs PINN_ext_phys.
    """
    cfg: BurgerConfig = bundle["cfg"]
    data              = bundle["data"]
    u_blind_flat      = bundle["u_pinn_blind_full"]
    u_ext_phys_flat   = bundle["u_pinn_ext_phys_full"]
    m                 = bundle["metrics"]

    u_true_flat = data["u_true_full"]

    t_vals, x_vals, grid_true = _to_grid(u_true_flat, data, "full")
    _,      _,      grid_blind = _to_grid(u_blind_flat, data, "full")
    _,      _,      grid_ext_phys = _to_grid(u_ext_phys_flat, data, "full")

    # Choose slice times: half from training, half from extrapolation
    t_train_vals  = t_vals[t_vals <= cfg.t_train]
    t_extrap_vals = t_vals[t_vals >  cfg.t_train]
    slice_times = np.linspace(t_train_vals[1], t_extrap_vals[-1], n_slices)


    n_cols = 3
    n_rows = (n_slices + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 2.8),
                             sharex=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    for idx, t_target in enumerate(slice_times):
        ax = axes_flat[idx]
        style_ax(ax)

        # Find the closest stored t index
        i_t = int(np.argmin(np.abs(t_vals - t_target)))
        t_actual = t_vals[i_t]
        in_extrap = t_actual > cfg.t_train
        near_shockwave = abs(t_actual - cfg.inviscid_shockwave_time) < 0.2

        u_true_s = grid_true[i_t, :]
        u_blind_s = grid_blind[i_t, :]
        u_ext_phys_s = grid_ext_phys[i_t, :]

        rmse_b = float(np.sqrt(np.mean((u_blind_s - u_true_s) ** 2)))
        rmse_e = float(np.sqrt(np.mean((u_ext_phys_s - u_true_s) ** 2)))

        ax.plot(x_vals, u_true_s, color=GRAY_TRUE, lw=1.8,
                label="True", zorder=5)
        ax.plot(x_vals, u_blind_s, color=TEAL_BLIND, lw=2.0, ls="-",
                label=f"PINN blind  (RMSE={rmse_b:.4f})", zorder=4)
        ax.plot(x_vals, u_ext_phys_s, color=ORANGE_EXT, lw=1.6, ls="--",
                label=f"PINN ext_phys  (RMSE={rmse_e:.4f})", zorder=3)

        region_tag = "extrap" if in_extrap else "train"
        color_tag  = ORANGE_EXT if in_extrap else TEAL_BLIND
        shock_marker = "close to shockwave" if near_shockwave else ""
        ax.set_title(
            f"t = {t_actual:.3f} s  [{region_tag}] {shock_marker}",
            fontsize=9, loc="left", pad=4, color=color_tag,
        )
        if in_extrap:
            ax.set_facecolor("#FFF5E6")
        if near_shockwave:
            ax.set_facecolor("#FFE0E0")  # light red for shockwave region
        ax.set_ylabel("u(x, t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("x  [m]", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.5)

    # Hide any unused panels
    for idx in range(n_slices, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Burgers' PINN Dual Model — u(x) profiles at fixed times\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"Training window: t ≤ {cfg.t_train}  |  shockwave (inviscid): t ≈ {cfg.inviscid_shockwave_time:.3f} (⚡ red panels)",
        fontsize=9, y=1.01, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 2 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 3 — Parameter Evolution: ν (viscosity) over epochs
# =============================================================================

def make_nu_evolution_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Show how the learned viscosity parameter ν (v_hat) evolves during training
    for both PINN_blind and PINN_ext_phys models.
    """
    cfg: BurgerConfig = bundle["cfg"]
    hist_blind = bundle["hist_pinn_blind"]
    hist_ext_phys = bundle["hist_pinn_ext_phys"]
    snaps_blind = bundle["snaps_pinn_blind"]
    snaps_ext_phys = bundle["snaps_pinn_ext_phys"]
    
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.12)
    
    ax_nu = fig.add_subplot(gs[0, 0])
    ax_comparison = fig.add_subplot(gs[0, 1])
    
    # ── Left panel: ν evolution over epochs ─────────────────────────────────────
    style_ax(ax_nu)
    
    # Extract v_hat from snapshots at snapshot epochs
    epochs_blind = sorted(snaps_blind.keys())
    epochs_ext_phys = sorted(snaps_ext_phys.keys())
    
    v_hat_blind = []
    v_hat_ext_phys = []
    
    pred_blind = Predictor(cfg)
    pred_ext_phys = Predictor(cfg)
    
    for ep in epochs_blind:
        pred_blind.load_state_dict(snaps_blind[ep])
        v_hat_blind.append(pred_blind.predict_params()["v_hat"])
    
    for ep in epochs_ext_phys:
        pred_ext_phys.load_state_dict(snaps_ext_phys[ep])
        v_hat_ext_phys.append(pred_ext_phys.predict_params()["v_hat"])
    
    ax_nu.semilogx(epochs_blind, v_hat_blind, color=TEAL_BLIND, lw=2.5, marker="o",
                   markersize=6, label="PINN blind", zorder=3)
    ax_nu.semilogx(epochs_ext_phys, v_hat_ext_phys, color=ORANGE_EXT, lw=2.5, marker="s",
                   markersize=6, label="PINN ext_phys", zorder=3)
    
    # True viscosity line
    ax_nu.axhline(cfg.v, color=GRAY_TRUE, lw=2.0, ls="--", alpha=0.7, label=f"True ν = {cfg.v}", zorder=2)
    
    ax_nu.set_xlabel("epoch  (log scale)", fontsize=10)
    ax_nu.set_ylabel("learned ν (v_hat)", fontsize=10)
    ax_nu.set_title("Viscosity Parameter Evolution", fontsize=11, loc="left", pad=8, color="#444441")
    ax_nu.grid(True, alpha=0.2)
    ax_nu.legend(fontsize=9, framealpha=0.7)
    
    # ── Right panel: Final vs True ─────────────────────────────────────────────
    style_ax(ax_comparison)
    
    final_v_blind = v_hat_blind[-1]
    final_v_ext_phys = v_hat_ext_phys[-1]
    
    models = ["PINN blind", "PINN ext_phys", "True value"]
    v_values = [final_v_blind, final_v_ext_phys, cfg.v]
    colors = [TEAL_BLIND, ORANGE_EXT, GRAY_TRUE]
    
    bars = ax_comparison.bar(models, v_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, v_values):
        height = bar.get_height()
        ax_comparison.text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add error bars (relative error)
    errors_blind = abs(final_v_blind - cfg.v) / cfg.v * 100
    errors_ext_phys = abs(final_v_ext_phys - cfg.v) / cfg.v * 100
    
    ax_comparison.set_ylabel("Viscosity ν", fontsize=10)
    ax_comparison.set_title(f"Final Estimates  (Blind error: {errors_blind:.1f}%, Ext_phys error: {errors_ext_phys:.1f}%)",
                           fontsize=11, loc="left", pad=8, color="#444441")
    ax_comparison.grid(True, alpha=0.2, axis="y")
    
    fig.suptitle(
        f"Burgers' PINN — Discovered Viscosity Parameter\n"
        f"True ν = {cfg.v:.2e}  |  situation={cfg.situation}  |  "
        f"Initial guess = {cfg.ini_guess_v}  |  λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}",
        fontsize=11, y=0.98, color="#2C2C2A", fontweight="bold"
    )
    
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 3 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 4 — PINN_blind epoch snapshots
# =============================================================================

def make_epoch_figure_blind(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Two heatmaps per snapshot epoch for PINN_blind: the predicted field and its error map.
    """
    cfg: BurgerConfig = bundle["cfg"]
    snaps = bundle["snaps_pinn_blind"]
    data = bundle["data"]
    pred: Predictor = bundle["pred_blind"]

    u_true_flat = data["u_true_full"]
    t_vals, x_vals, grid_true = _to_grid(u_true_flat, data, "full")
    x_flat = data["x_flattened_full"]
    t_flat = data["t_flattened_full"]

    # Shared colour limits across all epochs for fair comparison
    u_abs    = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u = -u_abs, u_abs

    epochs_sorted = sorted(snaps.keys())
    n_snap = len(epochs_sorted)

    fig, axes = plt.subplots(
        n_snap, 2,
        figsize=(13, n_snap * 3.2),
    )
    fig.patch.set_facecolor(BG)
    if n_snap == 1:
        axes = axes[np.newaxis, :]

    # Compute global max error across all epochs
    vmax_err_global = 0.0
    for epoch in epochs_sorted:
        u_ep = pred.predict_from_state(snaps[epoch], x_flat, t_flat)
        err_ep = np.abs(u_ep - u_true_flat)
        vmax_err_global = max(vmax_err_global, err_ep.max())

    for row, epoch in enumerate(epochs_sorted):
        ax_pred = axes[row, 0]
        ax_err  = axes[row, 1]

        u_ep_flat = pred.predict_from_state(snaps[epoch], x_flat, t_flat)
        t_vals_ep, x_vals_ep, grid_pred = _to_grid(u_ep_flat, data, "full")
        err_ep = np.abs(grid_pred - grid_true)

        # Check if this epoch is close to shockwave time (use max_epoch ~= t_shock as proxy)
        is_near_shock = abs(epoch - cfg.inviscid_shockwave_time) < 500  # within 500 epochs if thinking about time
        shock_marker = " ⚡" if is_near_shock else ""

        _heatmap(ax_pred, t_vals_ep, x_vals_ep, grid_pred,
                f"PINN_blind epoch {epoch}{shock_marker}",
                cfg, vmin_u, vmax_u, cbar_label="u(x, t)")
        _heatmap(ax_err, t_vals_ep, x_vals_ep, err_ep,
                f"|error|  epoch {epoch}{shock_marker}",
                cfg, vmin=0, vmax=vmax_err_global, cmap=CMAP_ERROR, cbar_label="|error|")
        
        if is_near_shock:
            ax_pred.set_facecolor("#FFE0E0")
            ax_err.set_facecolor("#FFE0E0")

    fig.suptitle(
        f"PINN_blind Solution Field Evolution\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"Training window: t ≤ {cfg.t_train}  |  shockwave: t ≈ {cfg.inviscid_shockwave_time:.3f} (dashed line)",
        fontsize=9.5, y=1.005, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 4 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 5 — PINN_ext_phys epoch snapshots
# =============================================================================

def make_epoch_figure_ext_phys(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Two heatmaps per snapshot epoch for PINN_ext_phys: the predicted field and its error map.
    """
    cfg: BurgerConfig = bundle["cfg"]
    snaps = bundle["snaps_pinn_ext_phys"]
    data = bundle["data"]
    pred: Predictor = bundle["pred_ext_phys"]

    u_true_flat = data["u_true_full"]
    t_vals, x_vals, grid_true = _to_grid(u_true_flat, data, "full")
    x_flat = data["x_flattened_full"]
    t_flat = data["t_flattened_full"]

    # Shared colour limits across all epochs for fair comparison
    u_abs    = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u = -u_abs, u_abs

    epochs_sorted = sorted(snaps.keys())
    n_snap = len(epochs_sorted)

    fig, axes = plt.subplots(
        n_snap, 2,
        figsize=(13, n_snap * 3.2),
    )
    fig.patch.set_facecolor(BG)
    if n_snap == 1:
        axes = axes[np.newaxis, :]

    # Compute global max error across all epochs
    vmax_err_global = 0.0
    for epoch in epochs_sorted:
        u_ep = pred.predict_from_state(snaps[epoch], x_flat, t_flat)
        err_ep = np.abs(u_ep - u_true_flat)
        vmax_err_global = max(vmax_err_global, err_ep.max())

    for row, epoch in enumerate(epochs_sorted):
        ax_pred = axes[row, 0]
        ax_err  = axes[row, 1]

        u_ep_flat = pred.predict_from_state(snaps[epoch], x_flat, t_flat)
        t_vals_ep, x_vals_ep, grid_pred = _to_grid(u_ep_flat, data, "full")
        err_ep = np.abs(grid_pred - grid_true)

        _heatmap(ax_pred, t_vals_ep, x_vals_ep, grid_pred,
                f"PINN_ext_phys epoch {epoch}",
                cfg, vmin_u, vmax_u, cbar_label="u(x, t)")
        _heatmap(ax_err, t_vals_ep, x_vals_ep, err_ep,
                f"|error|  epoch {epoch}",
                cfg, vmin=0, vmax=vmax_err_global, cmap=CMAP_ERROR, cbar_label="|error|")

    fig.suptitle(
        f"PINN_ext_phys Solution Field Evolution\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"Training window: t ≤ {cfg.t_train}  |  shockwave: t ≈ {cfg.inviscid_shockwave_time:.3f} (dashed line)",
        fontsize=9.5, y=1.005, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 5 saved  →  {out_path}")
    return fig


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Burgers dual PINN results from training_results.pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument_group("Input / output")
    p.add_argument(
        "--results", type=str, default=None,
        help="Path to training_results.pt bundle (defaults to out_dir/situation_viscosity_training_results.pt)",
    )
    p.add_argument(
        "--out_dir", type=str, default=BurgerConfig.out_dir,
        help="Directory for figures (defaults to same folder as results file)",
    )
    p.add_argument(
        "--n_slices", type=int, default=9,
        help="Number of time-slice panels in Figure 2",
    )
    p.add_argument(
        "--no_show", action="store_true",
        help="Save only; do not call plt.show()",
    )
    p.add_argument_group("Selection of situation and viscosity")
    p.add_argument(
        "--situation", type=str, choices=["Step", "Gaussian", "N-wave"], default=BurgerConfig.situation,
        help="Which Burgers' situation to plot (only used if no --results specified; defaults to config value)",
    )
    p.add_argument(
        "--viscosity", type=float, default=BurgerConfig.v,
        help="Viscosity parameter (only used if no --results specified; defaults to config value)",
    )
    p.add_argument(
        "--input_dir", type=str, default=BurgerConfig.out_dir,

        help="Directory where training_results.pt files are located (only used if no --results specified;" \
        " defaults to config output directory)",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    
    if args.results is None:
        results_string = (
            f"{args.out_dir}/{args.situation}_"
            f"{args.viscosity:.1e}_{BurgerConfig.suffix_results_pt}"
        )
    else:
        results_string = args.results
    
    results_path = Path(results_string)
    if not results_path.exists():
        print(f"ERROR: results file not found:  {results_path}")
        print(f"       Did you run train.py first?")
        sys.exit(1)

    out_dir = Path(args.out_dir) 
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load bundle ───────────────────────────────────────────────────────────
    print(f"\n  Loading  {results_path} …")
    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: BurgerConfig = raw["config"]
    snaps_blind = raw.get("snaps_pinn_blind", {})
    snaps_ext_phys = raw.get("snaps_pinn_ext_phys", {})

    # Recover grid dimensions from stored flat arrays
    t_flat = raw["data"]["t_flattened_full"]
    x_flat = raw["data"]["x_flattened_full"]
    n_t = len(np.unique(t_flat))
    n_x = len(np.unique(x_flat))

    print(f"  Config   : ν={cfg.v}  situation={cfg.situation}  "
          f"x∈[{cfg.x_begin},{cfg.x_end}]  t∈[{cfg.t0},{cfg.t_extrap}]")
    print(f"  Grid     : {n_t} × {n_x}  = {n_t * n_x} points")
    print(f"  Snapshots (blind):     {sorted(snaps_blind.keys())}")
    print(f"  Snapshots (ext_phys):  {sorted(snaps_ext_phys.keys())}\n")

    # Single shared Predictors
    pred_blind = Predictor(cfg)
    pred_ext_phys = Predictor(cfg)

    bundle = {
        "cfg":        cfg,
        "device_str": raw["device_str"],
        "data":       raw["data"],
        "hist_pinn_blind":    raw.get("hist_pinn_blind", {}),
        "hist_pinn_ext_phys": raw.get("hist_pinn_ext_phys", {}),
        "snaps_pinn_blind":   snaps_blind,
        "snaps_pinn_ext_phys": snaps_ext_phys,
        "u_pinn_blind_full":  raw.get("u_pinn_blind_full", np.array([])),
        "u_pinn_ext_phys_full": raw.get("u_pinn_ext_phys_full", np.array([])),
        "metrics":    raw.get("metrics", {}),
        "pred_blind": pred_blind,
        "pred_ext_phys": pred_ext_phys,
    }

    print("Generating Figure 1 — heatmap summary …")
    make_summary_figure(bundle, out_path=out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig1_summary.png")

    print(f"Generating Figure 2 — time-slice profiles ({args.n_slices} slices) …")
    make_slice_figure(bundle, out_path=out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig2_slices.png",
                      n_slices=args.n_slices)

    print("Generating Figure 3 — ν parameter evolution …")
    make_nu_evolution_figure(bundle, out_path=out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig3_nu_evolution.png")

    print("Generating Figure 4 — PINN_blind epoch snapshots …")
    make_epoch_figure_blind(bundle, out_path=out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig4_pinn_blind_epochs.png")

    print("Generating Figure 5 — PINN_ext_phys epoch snapshots …")
    make_epoch_figure_ext_phys(bundle, out_path=out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig5_pinn_ext_phys_epochs.png")

    print(f"\n  All figures written to  {out_dir}/")

    if not args.no_show:
        plt.show()

    print("  Done.\n")


if __name__ == "__main__":
    main()