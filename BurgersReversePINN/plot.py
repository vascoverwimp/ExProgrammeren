"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
figures for the Burgers' equation dual PINN (reverse - learning viscosity):

  Fig 1  burger_fig1_loss.png             — Loss progression (all types)
  Fig 2  burger_fig2_nu_evolution.png     — ν (viscosity) parameter evolution
  Fig 3  burger_fig3_slices.png           — u(x) profiles at fixed time slices
  Fig 4  burger_fig4_heatmaps.png         — 2×2 heatmap overview (final models)
  Fig 5  burger_fig5_blind_epochs.png     — PINN_blind field snapshots per epoch
  Fig 6  burger_fig6_ext_phys_epochs.png  — PINN_ext_phys field snapshots per epoch

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
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

from BurgersReversePINN.model import BurgerConfig, Predictor


# =============================================================================
# COLOUR PALETTE
# =============================================================================

TEAL_BLIND = "#1D9E75"   # PINN blind prediction
ORANGE_EXT = "#EF9F27"   # PINN ext_phys prediction
GRAY_TRUE = "#888780"   # true solution / neutral lines
PURPLE_IC = "#7F77DD"   # initial conditions
LGRAY = "#D3D1C7"   # spines
BG = "#FAFAF8"   # figure background
PANEL = "#F1EFE8"   # axes background

CMAP_FIELD = "RdBu_r"    # diverging — for signed velocity field u
CMAP_ERROR = "Oranges"   # sequential — for |error|


# =============================================================================
# SHARED HELPERS
# =============================================================================

def style_ax(ax: plt.Axes) -> None:
    """Style axes with custom background color and spine styling.

    Args:
        ax: Matplotlib axes object to style.
    """
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def _to_grid(flat: np.ndarray,
             data: dict,
             which: str = "full") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a flat prediction array to a 2-D grid.

    Converts flat (n_t*n_x,) arrays back to (n_t, n_x) grids using stored
    coordinate arrays. Layout assumes row-major order over (t, x) from meshgrid.

    Args:
        flat: Flattened array of shape (n_t*n_x,).
        data: Dictionary containing t_flattened_* and x_flattened_* keys.
        which: Grid type ('full' for entire domain or 'train' for training window).

    Returns:
        Tuple of (t_vals, x_vals, grid) where:
        - t_vals: Unique time values (length n_t)
        - x_vals: Unique spatial values (length n_x)
        - grid: Reshaped 2D array of shape (n_t, n_x)
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
    """Attach a slim colorbar to axes without resizing them.

    Args:
        ax: Matplotlib axes object to attach colorbar to.
        im: Image/collection object from pcolormesh or similar.
        label: Optional label text for the colorbar.
    """
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
    """Plot a pcolormesh heatmap with training boundary and shockwave marker.

    Args:
        ax: Matplotlib axes to plot on.
        t_vals: 1-D array of unique time values.
        x_vals: 1-D array of unique spatial values.
        grid: 2-D array of shape (n_t, n_x) to visualize.
        title: Title for the axes.
        cfg: BurgerConfig with training time and domain bounds.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
        cmap: Matplotlib colormap name.
        cbar_label: Label for the colorbar.
        show_shockwave: Whether to show shockwave formation line.
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
        x_vals[int(len(x_vals) * 0.02)], cfg.t_train +
        0.02 * (t_vals[-1] - t_vals[0]),
        "extrap ↑", color="white", fontsize=7, va="bottom",
    )

    # Shockwave formation time (inviscid case)
    if show_shockwave:
        t_shock = cfg.inviscid_shockwave_time
        if not np.isinf(t_shock) and t_vals[0] <= t_shock <= t_vals[-1]:
            ax.axhline(t_shock, color="#FF6B6B", lw=1.0, ls=":", alpha=0.8)
            ax.text(
                x_vals[int(len(x_vals) * 0.98)], t_shock +
                0.01 * (t_vals[-1] - t_vals[0]),
                "shock→", color="#FF6B6B", fontsize=7, va="bottom", ha="right", fontweight="bold"
            )

    ax.set_xlabel("x  [m]",    fontsize=8)
    ax.set_ylabel("time  [s]", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", pad=4, color="#444441")
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(t_vals[0],  t_vals[-1])


# =============================================================================
# FIGURE 1 — Loss progression (all loss types)
# =============================================================================

def make_loss_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """Create loss progression figure for both PINN models.

    Single-axes plot showing training, validation, physics, and IC losses
    for PINN blind and PINN extended physics models. Line style distinguishes
    models, color distinguishes loss types.

    Args:
        bundle: Dictionary containing training results and histories.
        out_path: Path where the figure is saved.

    Returns:
        The created matplotlib Figure object.
    """
    cfg = bundle["cfg"]
    h_blind = bundle["hist_pinn_blind"]
    h_ext = bundle["hist_pinn_ext_phys"]

    # One colour per loss category, shared across both models
    color_dict = {
        "loss_data":    "#2271B2",   # blue    — data
        "loss_val":     "#E6533C",   # red     — validation
        "loss_physics": "#3DAA6A",   # green   — physics residual
        "loss_ic":      "#9B5EBF",   # purple  — initial condition
    }
    label_dict = {
        "loss_data":    "data",
        "loss_val":     "val",
        "loss_physics": "physics",
        "loss_ic":      "IC",
    }
    marker_dict = {
        "loss_data":    "o",
        "loss_val":     "s",
        "loss_physics": "^",
        "loss_ic":      "v",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    for key in ("loss_data", "loss_val", "loss_physics", "loss_ic"):
        c = color_dict[key]
        mk = marker_dict[key]
        lbl = label_dict[key]
        ax.semilogy(h_blind["epoch"], h_blind[key],
                    color=c, lw=2.0, ls="-", marker=mk,
                    markersize=3, markevery=5,
                    label=f"blind — {lbl}")
        ax.semilogy(h_ext["epoch"], h_ext[key],
                    color=c, lw=1.5, ls="--", marker=mk,
                    markersize=3, markevery=5, alpha=0.8,
                    label=f"ext_phys — {lbl}")

    for ep in cfg.snapshot_epochs[:-1]:
        ax.axvline(ep, color=GRAY_TRUE, lw=0.6, ls=":", alpha=0.35, zorder=0)

    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("loss  (log scale)", fontsize=10)
    ax.set_title("All losses — blind (solid) vs ext_phys (dashed)",
                 fontsize=10, loc="left", pad=6, color="#444441")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.7, ncol=2)

    fig.suptitle(
        f"Burgers' equation (Reverse) — Training Loss Progression\n"
        f"situation={cfg.situation}  True ν={cfg.v}",
        fontsize=11, y=1.01, color="#2C2C2A", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 1 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 2 — Parameter Evolution: ν (viscosity) over epochs
# =============================================================================

def make_nu_evolution_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """Create figure showing viscosity parameter evolution during training.

    Semi-log plot of learned viscosity ν_hat versus training epoch for both
    PINN blind and PINN extended physics models. Shows how each model learns
    to estimate the true viscosity over time.

    Args:
        bundle: Dictionary containing training snapshots and config.
        out_path: Path where the figure is saved.

    Returns:
        The created matplotlib Figure object.
    """
    cfg: BurgerConfig = bundle["cfg"]
    snaps_blind = bundle["snaps_pinn_blind"]
    snaps_ext_phys = bundle["snaps_pinn_ext_phys"]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

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

    ax.semilogx(epochs_blind, v_hat_blind, color=TEAL_BLIND, lw=2.5, marker="o",
                markersize=6, label="PINN blind", zorder=3)
    ax.semilogx(epochs_ext_phys, v_hat_ext_phys, color=ORANGE_EXT, lw=2.5, marker="s",
                markersize=6, label="PINN ext_phys", zorder=3)

    ax.axhline(cfg.v, color=GRAY_TRUE, lw=2.0, ls="--", alpha=0.7,
               label=f"True ν = {cfg.v}", zorder=2)

    ax.set_xlabel("epoch  (log scale)", fontsize=10)
    ax.set_ylabel("learned ν (v_hat)", fontsize=10)
    ax.set_title("Viscosity Parameter Evolution", fontsize=11,
                 loc="left", pad=8, color="#444441")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, framealpha=0.7)

    fig.suptitle(
        f"Burgers' equation (Reverse) — Discovered Viscosity Parameter\n"
        f"True ν = {cfg.v:.2e}  |  situation={cfg.situation}  "
        f"Initial guess = {cfg.ini_guess_v}",
        fontsize=11, y=1.02, color="#2C2C2A", fontweight="bold"
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 2 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 3 — time-slice profiles  u(x) at fixed t values (two PINNs)
# =============================================================================

def make_slice_figure(bundle: dict, out_path: Path,
                      n_slices: int = 8) -> plt.Figure:
    """Create u(x) profile plots at fixed time slices.

    Generates line plots of u(x) at multiple fixed times spanning the
    full time domain for both PINN models compared to ground truth.

    Args:
        bundle: Dictionary containing predictions and ground truth data.
        out_path: Path where the figure is saved.
        n_slices: Number of time slices to plot (default 8).

    Returns:
        The created matplotlib Figure object.
    """
    cfg: BurgerConfig = bundle["cfg"]
    data = bundle["data"]
    u_blind_flat = bundle["u_pinn_blind_full"]
    u_ext_phys_flat = bundle["u_pinn_ext_phys_full"]

    u_true_flat = data["u_true_full"]

    t_vals, x_vals, grid_true = _to_grid(u_true_flat,    data, "full")
    _,      _,      grid_blind = _to_grid(u_blind_flat,   data, "full")
    _,      _,      grid_ext_phys = _to_grid(u_ext_phys_flat, data, "full")

    t_train_vals = t_vals[t_vals <= cfg.t_train]
    t_extrap_vals = t_vals[t_vals > cfg.t_train]
    slice_times = np.linspace(t_train_vals[1], t_extrap_vals[-1], n_slices)

    n_cols = 3
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 2.8),
                             sharex=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    for idx, t_target in enumerate(slice_times):
        ax = axes_flat[idx]
        style_ax(ax)

        i_t = int(np.argmin(np.abs(t_vals - t_target)))
        t_actual = t_vals[i_t]
        in_extrap = t_actual > cfg.t_train
        near_shockwave = (not np.isinf(cfg.inviscid_shockwave_time) and
                          abs(t_actual - cfg.inviscid_shockwave_time) < 0.2)

        u_true_s = grid_true[i_t, :]
        u_blind_s = grid_blind[i_t, :]
        u_ext_phys_s = grid_ext_phys[i_t, :]

        rmse_b = float(np.sqrt(np.mean((u_blind_s - u_true_s) ** 2)))
        rmse_e = float(np.sqrt(np.mean((u_ext_phys_s - u_true_s) ** 2)))

        ax.plot(x_vals, u_true_s,     color=GRAY_TRUE,  lw=1.8,
                label="True",                                  zorder=5)
        ax.plot(x_vals, u_blind_s,    color=TEAL_BLIND, lw=2.0,
                label=f"PINN blind  (RMSE={rmse_b:.4f})",      zorder=4)
        ax.plot(x_vals, u_ext_phys_s, color=ORANGE_EXT, lw=1.6, ls="--",
                label=f"PINN ext_phys  (RMSE={rmse_e:.4f})", zorder=3)

        region_tag = "extrap" if in_extrap else "train"
        color_tag = ORANGE_EXT if in_extrap else TEAL_BLIND
        shock_marker = "  ⚡ near shockwave" if near_shockwave else ""
        ax.set_title(
            f"t = {t_actual:.3f} s  [{region_tag}]{shock_marker}",
            fontsize=9, loc="left", pad=4, color=color_tag,
        )
        if in_extrap:
            ax.set_facecolor("#FFF5E6")
        if near_shockwave:
            ax.set_facecolor("#FFE0E0")
        ax.set_ylabel("u(x, t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("x  [m]", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.5)

    for idx in range(n_slices, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    shock_info = (
        f"shockwave (inviscid): t ≈ {cfg.inviscid_shockwave_time:.3f} (⚡ red panels)"
        if not np.isinf(cfg.inviscid_shockwave_time) else "no shockwave"
    )
    fig.suptitle(
        f"Burgers' PINN Dual Model — u(x) profiles at fixed times\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"Training window: t ≤ {cfg.t_train}  |  {shock_info}",
        fontsize=9, y=1.01, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 3 saved  →  {out_path}")   # FIX: was "Figure 2"
    return fig


# =============================================================================
# FIGURE 4 — Final predictions: True vs PINN_blind vs PINN_ext_phys (heatmaps)
# =============================================================================

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """Create a 2×3 heatmap summary comparing both final PINN models.

    Row 0 shows the true u(x, t) field alongside both PINN predictions;
    Row 1 shows the signed error for each model. Field panels share a
    common colormap range; error panels share a symmetric scale set by
    the largest absolute error across both models.

    Layout::

        Row 0: True u | PINN blind | PINN ext_phys
        Row 1: (empty)| PINN blind error | PINN ext_phys error

    Args:
        bundle: Dictionary containing ``cfg``, ``data``, ``metrics``,
            and the two ``u_pinn_*_full`` flat prediction arrays.
        out_path: Path where the figure is saved.

    Returns:
        The created matplotlib Figure object.

    Side effects:
        Saves the figure to ``out_path`` and prints the save path
        to stdout.
    """
    cfg: BurgerConfig = bundle["cfg"]
    data = bundle["data"]

    # FIX: u_true lives inside data, not at the top level of the bundle
    u_true_flat = data["u_true_full"]
    u_blind_flat = bundle["u_pinn_blind_full"]
    u_ext_phys_flat = bundle["u_pinn_ext_phys_full"]

    # FIX: reshape flat arrays to 2-D grids for _heatmap
    t_vals, x_vals, grid_true = _to_grid(u_true_flat,     data, "full")
    _,      _,      grid_blind = _to_grid(u_blind_flat,    data, "full")
    _,      _,      grid_ext_phys = _to_grid(u_ext_phys_flat, data, "full")

    err_blind = grid_blind - grid_true
    err_ext_phys = grid_ext_phys - grid_true

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.96, top=0.92, bottom=0.08)

    ax_true = fig.add_subplot(gs[0, 0])
    ax_blind = fig.add_subplot(gs[0, 1])
    ax_ext_phys = fig.add_subplot(gs[0, 2])
    ax_err_blind = fig.add_subplot(gs[1, 1])
    ax_err_ext_phys = fig.add_subplot(gs[1, 2])
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.set_visible(False)

    vmin_pred = min(grid_true.min(), grid_blind.min(), grid_ext_phys.min())
    vmax_pred = max(grid_true.max(), grid_blind.max(), grid_ext_phys.max())

    vmax_err = max(np.abs(err_blind).max(), np.abs(err_ext_phys).max())
    vmin_err = -vmax_err

    # FIX: _heatmap signature is (ax, t_vals, x_vals, grid, title, cfg, ...)
    # and it already draws its own colorbar internally — no return value needed.
    _heatmap(ax_true, t_vals, x_vals, grid_true,
             "True u(x,t)", cfg,
             vmin=vmin_pred, vmax=vmax_pred, cmap="RdBu_r")

    # FIX: metrics key is nu_hat_blind, not v_hat_blind (matches train.py)
    nu_blind = bundle["metrics"].get("nu_hat_blind",    0.0)
    nu_ext = bundle["metrics"].get("nu_hat_ext_phys", 0.0)

    _heatmap(ax_blind, t_vals, x_vals, grid_blind,
             f"PINN blind  (ν̂={nu_blind:.3e})", cfg,
             vmin=vmin_pred, vmax=vmax_pred, cmap="RdBu_r")

    _heatmap(ax_ext_phys, t_vals, x_vals, grid_ext_phys,
             f"PINN ext_phys  (ν̂={nu_ext:.3e})", cfg,
             vmin=vmin_pred, vmax=vmax_pred, cmap="RdBu_r")

    _heatmap(ax_err_blind, t_vals, x_vals, err_blind,
             "PINN blind error", cfg,
             vmin=vmin_err, vmax=vmax_err, cmap="seismic", cbar_label="error")

    _heatmap(ax_err_ext_phys, t_vals, x_vals, err_ext_phys,
             "PINN ext_phys error", cfg,
             vmin=vmin_err, vmax=vmax_err, cmap="seismic", cbar_label="error")

    # Apply title colours after _heatmap (it sets its own grey title)
    ax_true.title.set_color("#2C2C2A")
    ax_blind.title.set_color(TEAL_BLIND)
    ax_ext_phys.title.set_color(ORANGE_EXT)
    ax_err_blind.title.set_color(TEAL_BLIND)
    ax_err_ext_phys.title.set_color(ORANGE_EXT)

    fig.suptitle(
        f"Burgers' PINN — Final Model Predictions (Reverse Problem)\n"
        f"ν={cfg.v:.2e}  situation={cfg.situation}  Training: t ≤ {cfg.t_train}",
        fontsize=12, y=0.98, color="#2C2C2A", fontweight="bold"
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 4 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 5 — PINN_blind epoch snapshots
# =============================================================================

def make_epoch_figure_blind(bundle: dict, out_path: Path) -> plt.Figure:
    """Create per-epoch field evolution figure for the blind PINN.

    Produces one row per snapshot epoch with two columns: the predicted
    u(x, t) field and the absolute pointwise error. The error colormap
    is shared across all epochs, scaled to the global maximum absolute
    error.

    Args:
        bundle: Dictionary containing ``cfg``, ``data``,
            ``snaps_pinn_blind``, and ``pred_blind``.
        out_path: Path where the figure is saved.

    Returns:
        The created matplotlib Figure object.

    Side effects:
        Mutates ``pred_blind`` by calling ``load_state_dict`` on
        each snapshot in turn. Saves the figure to ``out_path``
        and prints the save path to stdout.
    """
    cfg: BurgerConfig = bundle["cfg"]
    snaps = bundle["snaps_pinn_blind"]
    data = bundle["data"]
    pred: Predictor = bundle["pred_blind"]

    u_true_flat = data["u_true_full"]
    *_, grid_true = _to_grid(u_true_flat, data, "full")
    x_flat = data["x_flattened_full"]
    t_flat = data["t_flattened_full"]

    u_abs = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u = -u_abs, u_abs

    epochs_sorted = sorted(snaps.keys())
    n_snap = len(epochs_sorted)

    fig, axes = plt.subplots(n_snap, 2, figsize=(13, n_snap * 3.2))
    fig.patch.set_facecolor(BG)
    if n_snap == 1:
        axes = axes[np.newaxis, :]

    # FIX: predict_from_state does not exist; use load_state_dict + predict
    vmax_err_global = 0.0
    for epoch in epochs_sorted:
        pred.load_state_dict(snaps[epoch])
        u_ep = pred.predict(x_flat, t_flat)
        err_ep = np.abs(u_ep - u_true_flat)
        vmax_err_global = max(vmax_err_global, err_ep.max())

    for row, epoch in enumerate(epochs_sorted):
        ax_pred = axes[row, 0]
        ax_err = axes[row, 1]

        pred.load_state_dict(snaps[epoch])
        u_ep_flat = pred.predict(x_flat, t_flat)
        t_vals_ep, x_vals_ep, grid_pred = _to_grid(u_ep_flat, data, "full")
        err_ep = np.abs(grid_pred - grid_true)

        is_near_shock = (not np.isinf(cfg.inviscid_shockwave_time) and
                         abs(epoch - cfg.inviscid_shockwave_time) < 500)
        shock_marker = r"\lightning" if is_near_shock else ""

        _heatmap(ax_pred, t_vals_ep, x_vals_ep, grid_pred,
                 f"PINN_blind epoch {epoch}{shock_marker}",
                 cfg, vmin_u, vmax_u, cbar_label="u(x, t)")
        _heatmap(ax_err, t_vals_ep, x_vals_ep, err_ep,
                 f"|error|  epoch {epoch}{shock_marker}",
                 cfg, vmin=0, vmax=vmax_err_global, cmap=CMAP_ERROR, cbar_label="|error|")

    shock_info = (
        f"shockwave: t ≈ {cfg.inviscid_shockwave_time:.3f} (dashed line)"
        if not np.isinf(cfg.inviscid_shockwave_time) else "no shockwave"
    )
    fig.suptitle(
        f"PINN_blind Solution Field Evolution\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"Training window: t ≤ {cfg.t_train}  |  {shock_info}",
        fontsize=9.5, y=1.005, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 5 saved  →  {out_path}")   # FIX: was "Figure 4"
    return fig


# =============================================================================
# FIGURE 6 — PINN_ext_phys epoch snapshots
# =============================================================================

def make_epoch_figure_ext_phys(bundle: dict, out_path: Path) -> plt.Figure:
    """Create per-epoch field evolution figure for the ext_phys PINN.

    Produces one row per snapshot epoch with two columns: the predicted
    u(x, t) field and the absolute pointwise error. The error colormap
    is shared across all epochs, scaled to the global maximum absolute
    error.

    Args:
        bundle: Dictionary containing ``cfg``, ``data``,
            ``snaps_pinn_ext_phys``, and ``pred_ext_phys``.
        out_path: Path where the figure is saved.

    Returns:
        The created matplotlib Figure object.

    Side effects:
        Mutates ``pred_ext_phys`` by calling ``load_state_dict`` on
        each snapshot in turn. Saves the figure to ``out_path``
        and prints the save path to stdout.
    """
    cfg: BurgerConfig = bundle["cfg"]
    snaps = bundle["snaps_pinn_ext_phys"]
    data = bundle["data"]
    pred: Predictor = bundle["pred_ext_phys"]

    u_true_flat = data["u_true_full"]
    *_, grid_true = _to_grid(u_true_flat, data, "full")
    x_flat = data["x_flattened_full"]
    t_flat = data["t_flattened_full"]

    u_abs = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u = -u_abs, u_abs

    epochs_sorted = sorted(snaps.keys())
    n_snap = len(epochs_sorted)

    fig, axes = plt.subplots(n_snap, 2, figsize=(13, n_snap * 3.2))
    fig.patch.set_facecolor(BG)
    if n_snap == 1:
        axes = axes[np.newaxis, :]

    # FIX: predict_from_state does not exist; use load_state_dict + predict
    vmax_err_global = 0.0
    for epoch in epochs_sorted:
        pred.load_state_dict(snaps[epoch])
        u_ep = pred.predict(x_flat, t_flat)
        err_ep = np.abs(u_ep - u_true_flat)
        vmax_err_global = max(vmax_err_global, err_ep.max())

    for row, epoch in enumerate(epochs_sorted):
        ax_pred = axes[row, 0]
        ax_err = axes[row, 1]

        pred.load_state_dict(snaps[epoch])
        u_ep_flat = pred.predict(x_flat, t_flat)
        t_vals_ep, x_vals_ep, grid_pred = _to_grid(u_ep_flat, data, "full")
        err_ep = np.abs(grid_pred - grid_true)

        _heatmap(ax_pred, t_vals_ep, x_vals_ep, grid_pred,
                 f"PINN_ext_phys epoch {epoch}",
                 cfg, vmin_u, vmax_u, cbar_label="u(x, t)")
        _heatmap(ax_err, t_vals_ep, x_vals_ep, err_ep,
                 f"|error|  epoch {epoch}",
                 cfg, vmin=0, vmax=vmax_err_global, cmap=CMAP_ERROR, cbar_label="|error|")

    shock_info = (
        f"shockwave: t ≈ {cfg.inviscid_shockwave_time:.3f} (dashed line)"
        if not np.isinf(cfg.inviscid_shockwave_time) else "no shockwave"
    )
    fig.suptitle(
        f"PINN_ext_phys Solution Field Evolution\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"Training window: t ≤ {cfg.t_train}  |  {shock_info}",
        fontsize=9.5, y=1.005, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 6 saved  →  {out_path}")
    return fig


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the reverse Burgers PINN plotter.

    Accepts an explicit path to a ``training_results.pt`` bundle via
    ``--results``, or infers it from ``--out_dir``, ``--situation``,
    and ``--viscosity``. Also controls the output directory, the
    number of time-slice panels in Figure 3, and whether to call
    ``plt.show()``.

    Returns:
        argparse.Namespace: Parsed arguments, including ``results``,
            ``input_dir``, ``out_dir``, ``situation``, ``viscosity``,
            ``n_slices``, and ``no_show``.
    """
    p = argparse.ArgumentParser(
        description="Plot Burgers dual PINN results from training_results.pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument_group("Input / output")
    p.add_argument(
        "--results", type=str, default=None,
        help="Path to training_results.pt bundle"
             "(defaults to out_dir/situation_viscosity_training_results.pt)",
    )
    p.add_argument(
        "--out_dir", type=str, default=BurgerConfig.out_dir,
        help="Directory for figures (defaults to same folder as results file)",
    )
    p.add_argument(
        "--n_slices", type=int, default=9,
        help="Number of time-slice panels in Figure 3",
    )
    p.add_argument(
        "--no_show", action="store_true",
        help="Save only; do not call plt.show()",
    )
    p.add_argument_group("Selection of situation and viscosity")
    p.add_argument(
        "--situation", type=str, choices=["Step", "Gaussian", "N-wave"],
        default=BurgerConfig.situation, help="Which Burgers' situation to plot",
    )
    p.add_argument(
        "--viscosity", type=float, default=BurgerConfig.v,
        help="Viscosity parameter",
    )
    p.add_argument(
        "--input_dir", type=str, default=BurgerConfig.out_dir,
        help="Directory where training_results.pt files are located",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Entry point for generating all six Burgers reverse PINN figures.

    Orchestrates the full plotting pipeline:

    1. Parses CLI arguments via :func:`parse_args` and resolves the
       path to the ``training_results.pt`` bundle, exiting with an
       error if not found.
    2. Loads the bundle and prints a summary of the config, grid
       dimensions, and available snapshot epochs.
    3. Instantiates ``Predictor`` objects for both PINN variants.
    4. Assembles a ``bundle`` dict and calls each figure function:

       - :func:`make_loss_figure`           → ``*_fig1_loss.png``
       - :func:`make_nu_evolution_figure`   → ``*_fig2_nu_evolution.png``
       - :func:`make_slice_figure`          → ``*_fig3_slices.png``
       - :func:`make_summary_figure`        → ``*_fig4_heatmaps.png``
       - :func:`make_epoch_figure_blind`    → ``*_fig5_blind_epochs.png``
       - :func:`make_epoch_figure_ext_phys` → ``*_fig6_ext_phys_epochs.png``

    5. Optionally displays all figures via ``plt.show()`` unless
       ``--no_show`` is set.

    Side effects:
        Writes six PNG files to ``args.out_dir`` and prints progress
        messages to stdout.
    """
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
        print("        Did you run train.py first?")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading  {results_path} …")
    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: BurgerConfig = raw["config"]
    snaps_blind = raw.get("snaps_pinn_blind",    {})
    snaps_ext_phys = raw.get("snaps_pinn_ext_phys", {})

    t_flat = raw["data"]["t_flattened_full"]
    x_flat = raw["data"]["x_flattened_full"]
    n_t = len(np.unique(t_flat))
    n_x = len(np.unique(x_flat))

    print(f"  Config   : ν={cfg.v}  situation={cfg.situation}  "
          f"x∈[{cfg.x_begin},{cfg.x_end}]  t∈[{cfg.t0},{cfg.t_extrap}]")
    print(f"  Grid     : {n_t} × {n_x}  = {n_t * n_x} points")
    print(f"  Snapshots (blind):     {sorted(snaps_blind.keys())}")
    print(f"  Snapshots (ext_phys):  {sorted(snaps_ext_phys.keys())}\n")

    pred_blind = Predictor(cfg)
    pred_ext_phys = Predictor(cfg)

    bundle = {
        "cfg":        cfg,
        "device_str": raw["device_str"],
        # numpy arrays from generate_data
        "data":       raw["data"],
        "hist_pinn_blind":     raw.get("hist_pinn_blind",    {}),
        "hist_pinn_ext_phys":  raw.get("hist_pinn_ext_phys", {}),
        "snaps_pinn_blind":    snaps_blind,
        "snaps_pinn_ext_phys": snaps_ext_phys,
        # FIX: u_true_full is inside raw["data"], not at the top level of the bundle
        "u_true_full":          raw["data"].get("u_true_full",         np.array([])),
        # these two ARE saved at top level by train.py
        "u_pinn_blind_full":    raw.get("u_pinn_blind_full",    np.array([])),
        "u_pinn_ext_phys_full": raw.get("u_pinn_ext_phys_full", np.array([])),
        "metrics":    raw.get("metrics", {}),
        "pred_blind":    pred_blind,
        "pred_ext_phys": pred_ext_phys,
    }

    print("Generating Figure 1 — loss evolution …")
    make_loss_figure(bundle, out_path=out_dir /
                     f"burger_{cfg.situation}_{cfg.v:.1e}_fig1_loss.png")

    print("Generating Figure 2 — ν parameter evolution …")
    make_nu_evolution_figure(bundle, out_path=out_dir /
                             f"burger_{cfg.situation}_{cfg.v:.1e}_fig2_nu_evolution.png")

    print(
        f"Generating Figure 3 — time-slice profiles ({args.n_slices} slices) …")
    make_slice_figure(
        bundle, out_path=out_dir /
        f"burger_{cfg.situation}_{cfg.v:.1e}_fig3_slices.png",
        n_slices=args.n_slices)

    print("Generating Figure 4 — final predictions (heatmaps) …")
    make_summary_figure(bundle, out_path=out_dir /
                        f"burger_{cfg.situation}_{cfg.v:.1e}_fig4_heatmaps.png")

    print("Generating Figure 5 — PINN_blind epoch snapshots …")
    make_epoch_figure_blind(bundle, out_path=out_dir /
                            f"burger_{cfg.situation}_{cfg.v:.1e}_fig5_blind_epochs.png")

    print("Generating Figure 6 — PINN_ext_phys epoch snapshots …")
    make_epoch_figure_ext_phys(
        bundle, out_path=out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig6_ext_phys_epochs.png")

    print(f"\n  All figures written to  {out_dir}/")

    if not args.no_show:
        plt.show()

    print("  Done.\n")


if __name__ == "__main__":
    main()
