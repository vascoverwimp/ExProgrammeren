"""
Plotting functions for the damped spring-mass PINN results.

Contains:
    plot_summary        -- five-panel overview (data, fit, loss, extrapolation,
                           ODE residual)
    plot_epoch_snapshots -- grid of trajectory panels at chosen training epochs

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from pathlib import Path
import torch
import torch.nn as nn
from config import Config
from data import generate_data
from model import predict, FCNet, InverseFCNet
from utils import save_show, load_model
from analytic import interpolate_solution_arr, u_0

# -- LaTeX font ------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# -- Color palette ------------------------------------------------
BLUE = "#378ADD"   # noisy training observations
RED = "#E24B4A"   # noisy test/validation observations
GREEN = "#1D9E75"   # PINN
ORANGE = "#EF9F27"   # collocation points
PURPLE = "#7F77DD"   # initial condition marker
GRAY = "#888780"   # true solution / neutral
LGRAY = "#D3D1C7"   # spine colour
BG = "#FAFAF8"   # figure background
PANEL = "#F1EFE8"   # axes background


def style_ax(ax: Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def plot_gt_1D(
    cfg:         Config,
    u_grid:      np.ndarray,
    t_arr:       np.ndarray,
    times:       list[float] = None,
    output_path: Path = None,
    show:        bool = True
) -> None:
    """
    Plot the ground truth solution u(x, t) at four different times.

    Parameters
    ----------
    cfg         : Config
    u_grid      : solution grid from lax_wendroff
    t_arr       : time array from lax_wendroff
    times       : list of 4 times to plot; defaults to 4 evenly spaced values in [0, t_dom]  # noqa:E501
    output_path : path to save the figure
    show        : whether to display the figure
    """
    if times is None:
        times = np.linspace(0, cfg.t_extrap, 4, endpoint=False).tolist()
    if len(times) != 4:
        raise ValueError(f"Expected 4 times, got {len(times)}")

    x = np.linspace(0, cfg.L, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    for ax, t_val in zip(axes_flat, times):
        t_arr_plot = np.full_like(x, t_val)
        u = interpolate_solution_arr(u_grid, t_arr, x, t_arr_plot, cfg)
        style_ax(ax)

        ax.plot(x, u, lw=2, color=GRAY, label="Ground truth $u(x, t)$")
        ax.set_xlabel("$x$",                         fontsize=11)
        ax.set_ylabel(rf"$u(x,\ t={t_val:.3f})$",   fontsize=11)
        ax.set_title(
            rf"$t = {t_val:.3f}$",
            fontsize=10, loc="left", pad=4, color="#444441"
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8, framealpha=0.5)

    fig.suptitle(
        rf"Ground truth solution $u(x, t)$ — $\nu = {cfg.nu:.4f}$",
        fontsize=13, color="#2C2C2A"
    )
    fig.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_pred_1D(
    model:       nn.Module,
    cfg:         Config,
    snapshots:   dict,
    u_grid:      np.ndarray,
    t_arr:       np.ndarray,
    n_times:     int = 8,
    times:       list[float] = None,    # ← added
    model_color: str = GREEN,
    model_label: str = "PINN",
    clip_u:      bool = False,
    output_path: Path = None,
    show:        bool = True
) -> None:
    # -- detect inverse mode ----------------------------------------------
    inverse = any("_nu_raw" in k for v in snapshots.values() for k in v.keys())

    # -- time slices -------------------------------------------------------
    if times is not None:
        times = list(times)
        n_times = len(times)
    else:
        times = np.linspace(0, cfg.t_extrap, n_times).tolist()
    x_plot = np.linspace(0, cfg.L, 300)

    n_cols = 2
    n_rows = (n_times + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(12, n_rows * 3.2),
                             sharex=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    for idx, t_val in enumerate(times):
        ax = axes_flat[idx]
        style_ax(ax)

        # -- numerical solution --------------------------------------------
        u_true = interpolate_solution_arr(
            u_grid, t_arr,
            x_plot, np.full_like(x_plot, t_val),
            cfg
        )

        # -- model prediction ----------------------------------------------
        u_pred = predict(model, np.full_like(x_plot, t_val), x_plot)
        if clip_u:
            u_pred = np.clip(u_pred, -3, 3)

        err = np.sqrt(np.mean((u_pred - u_true) ** 2))

        ax.plot(x_plot, u_true,
                color=GRAY, lw=1.4, alpha=0.9)
        ax.plot(x_plot, u_pred,
                color=model_color, lw=2.0)

        # -- extrapolation shading -----------------------------------------
        if not cfg.train_extrap:
            if t_val > cfg.t_dom:
                ax.set_title(
                    rf"$t = {t_val:.3f}$ [extrap]   |   RMSE = {err:.4f}",
                    fontsize=9, loc="left", pad=4, color="#444441"
                )
            else:
                ax.set_title(
                    rf"$t = {t_val:.3f}$   |   RMSE = {err:.4f}",
                    fontsize=9, loc="left", pad=4, color="#444441"
                )

        ax.set_xlim(0, cfg.L)
        ax.set_ylabel(f"$u(x, t={t_val:.3f})$", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("$x$", fontsize=8)

    for idx in range(n_times, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # -- shared legend -----------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color=GRAY,        lw=1.4,
               label="ground truth solution $u(x, t)$"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=rf"{model_label} prediction $\hat{{u}}(x, t)$"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))
    if inverse:
        fig.suptitle(
            rf"Ground truth and predicted solution $u(x, t)$ — $\nu = {cfg.nu:.4f} \ \hat\nu = {model.nu_hat.item():.4f}$",  # noqa:E501
            fontsize=16, color="#2C2C2A"
        )
    else:
        fig.suptitle(
            rf"Ground truth and predicted solution $u(x, t)$ — $\nu = {cfg.nu:.4f}$",  # noqa:E501
            fontsize=16, color="#2C2C2A"
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_show(output_path=output_path, show=show)


def plot_solution_grid(
    model:       nn.Module,
    data:        dict,
    cfg:         Config,
    snapshots:   dict,
    u_grid:      np.ndarray,
    t_arr:       np.ndarray,
    n_times:     int = 8,
    times:       list[float] = None,
    model_color: str = GREEN,
    model_label: str = "PINN",
    fig_title:   str = r"PINN -- spatial solution at different times",
    clip_y:      bool = False,
    output_path: Path = None,
    show:        bool = True
) -> None:
    """
    Build a 2-column grid of panels, one per time slice.

    Each panel shows:
        - numerical solution u(x, t)     (gray line)
        - model prediction u_hat(x, t)   (coloured line)
        - noisy observations at that t   (blue dots)
        - RMSE annotated in panel title

    Parameters
    ----------
    model       : trained PINN on CPU
    data        : output of generate_data()
    cfg         : Config
    u_grid      : numerical solution, shape (n_t, n_x)
    t_arr       : time array from lax_wendroff
    n_times     : number of time slices (ignored if times is provided)
    times       : optional explicit list of times to plot
    """
    # -- detect inverse mode ----------------------------------------------
    inverse = any("_nu_raw" in k for v in snapshots.values() for k in v.keys())

    if times is not None:
        times = list(times)
        n_times = len(times)
    else:
        times = np.linspace(0, cfg.t_extrap, n_times).tolist()

    x_plot = np.linspace(0, cfg.L, 300)
    n_cols = 2
    n_rows = (n_times + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 3.2),
                             sharex=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    for idx, t_val in enumerate(times):
        ax = axes_flat[idx]
        style_ax(ax)

        # -- numerical solution --------------------------------------------
        u_true = interpolate_solution_arr(
            u_grid, t_arr,
            x_plot, np.full_like(x_plot, t_val),
            cfg
        )

        # -- model prediction ----------------------------------------------
        u_pred = predict(model, np.full_like(x_plot, t_val), x_plot)
        if clip_y:
            u_pred = np.clip(u_pred, -3, 3)

        err = np.sqrt(np.mean((u_pred - u_true) ** 2))

        ax.plot(x_plot, u_true,  color=GRAY,        lw=1.4, alpha=0.9)
        ax.plot(x_plot, u_pred,  color=model_color,  lw=2.0)

        # # -- observations near this time slice -----------------------------
        # dt_window = cfg.t_extrap / (2 * n_times)
        # mask = np.abs(data["t_obs"] - t_val) < dt_window
        # if mask.any():
        #     ax.scatter(data["x_obs"][mask], data["u_obs"][mask],
        #                color=BLUE, s=28, marker="o", zorder=5, alpha=0.85)

        extrap_tag = (
            "  [extrap]"
            if t_val > cfg.t_dom and not cfg.train_extrap
            else ""
            )
        ax.set_title(rf"$t = {t_val:.3f}${extrap_tag}   |   RMSE = {err:.4f}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xlim(0, cfg.L)
        ax.set_ylabel("$u(x, t)$", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("$x$", fontsize=8)

    for idx in range(n_times, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # -- shared legend -----------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color=GRAY,        lw=1.4,
               label="Numerical solution $u(x, t)$"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=rf"{model_label} prediction $\hat{{u}}(x, t)$"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=12, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    if inverse:
        fig.suptitle(
            rf"Ground truth and predicted solution $u(x, t)$ — $\nu = {cfg.nu:.4f} \ \hat\nu = {model.nu_hat.item():.4f}$",  # noqa:E501
            fontsize=16, color="#2C2C2A"
        )
    else:
        fig.suptitle(
            rf"Ground truth and predicted solution $u(x, t)$ — $\nu = {cfg.nu:.4f}$",  # noqa:E501
            fontsize=16, color="#2C2C2A"
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_show(output_path=output_path, show=show)


def plot_losses(
    history:     dict,
    cfg:         Config,
    output_path: Path = None,
    show: bool = True
) -> None:
    """
    Plot all training loss curves on a single log-scale figure.

    Parameters
    ----------
    history   : dict from train(), keys: epoch, loss_data, loss_phys,
                loss_ic, loss_val, loss_total
    cfg       : Config, used for snapshot epoch markers
    output_path : Path to save the figure
    show : Whether to display the figure
    """
    loss_style = {
        "loss_data":  (BLUE,   "-",  r"$\mathcal{L}_{data}$"),
        "loss_phys":  (GREEN,  "-",  r"$\mathcal{L}_{phys}$"),
        "loss_ic":    (PURPLE, "-",  r"$\mathcal{L}_{ic}$"),
        "loss_bc":    (ORANGE, "-",  r"$\mathcal{L}_{bc}$"),
        "loss_val":   (RED,    "--", r"$\mathcal{L}_{val}$"),
        "loss_total": (GRAY,   ":",  r"$\mathcal{L}_{total}$"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    style_ax(ax)
    fig.patch.set_facecolor(BG)

    epochs = history["epoch"]
    for key, (color, ls, label) in loss_style.items():
        if key in history and any(v > 0 for v in history[key]):
            ax.semilogy(epochs, history[key],
                        color=color, ls=ls, lw=1.8, label=label)

    # snapshot epoch markers
    for ep in cfg.snapshot_epochs:
        if ep <= max(epochs):
            ax.axvline(ep, color=LGRAY, lw=0.8, ls=":", zorder=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training loss curves", fontsize=11,
                 loc="left", color="#444441")
    ax.legend(fontsize=9, framealpha=0.5)
    ax.set_xlim(min(epochs), max(epochs))

    plt.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_gt_2D(
    cfg:         Config,
    u_grid:      np.ndarray,
    t_arr:       np.ndarray,
    output_path: Path = None,
    show:        bool = True
) -> None:
    x = np.linspace(0, cfg.L, 500)
    t = np.linspace(0, cfg.t_extrap, 500)
    x_matrix, t_matrix = np.meshgrid(x, t)

    u_flat = interpolate_solution_arr(
        u_grid, t_arr,
        x_matrix.ravel(),
        t_matrix.ravel(),
        cfg
    )
    u_matrix = u_flat.reshape(x_matrix.shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)

    im = ax.imshow(
        u_matrix,
        origin="lower",
        extent=[0, cfg.L, 0, cfg.t_extrap],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.max(np.abs(u_matrix)),
        vmax=np.max(np.abs(u_matrix)),
    )

    if cfg.t_shock is not None:
        ax.axhline(cfg.t_shock, color=ORANGE, lw=1.5, ls="--")
        ax.annotate(
            "Inviscous shock formation time",
            xy=(0.02, cfg.t_shock / cfg.t_extrap - 0.04),
            xycoords="axes fraction", fontsize=12, color=ORANGE
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("$u(x, t)$", fontsize=16)

    ax.set_xlabel("$x$",  fontsize=14)
    ax.set_ylabel("$t$",  fontsize=14)
    ax.set_xticks(np.linspace(0, cfg.L,        6))
    ax.set_yticks(np.linspace(0, cfg.t_extrap, 6))
    ax.set_title(
        rf"Ground truth $u(x, t)$  --  $\nu = {cfg.nu:.4f}$",
        fontsize=16, loc="left", pad=6, color="#444441"
    )

    fig.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_pred_2D(
    model:       nn.Module,
    cfg:         Config,
    snapshots:   dict,
    output_path: Path = None,
    show:        bool = True
) -> None:
    # -- detect inverse mode ----------------------------------------------
    inverse = any("_nu_raw" in k for v in snapshots.values() for k in v.keys())

    x = np.linspace(0, cfg.L, 500)
    t = np.linspace(0, cfg.t_extrap, 500)
    x_matrix, t_matrix = np.meshgrid(x, t)

    u_flat = predict(model, t_matrix.ravel(), x_matrix.ravel())

    u_matrix = u_flat.reshape(x_matrix.shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)

    im = ax.imshow(
        u_matrix,
        origin="lower",
        extent=[0, cfg.L, 0, cfg.t_extrap],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.max(np.abs(u_matrix)),
        vmax=np.max(np.abs(u_matrix)),
    )

    if cfg.t_shock is not None:
        ax.axhline(cfg.t_shock, color=GREEN, lw=1.5, ls="--",
                   label="Inviscous shock formation time")
        ax.annotate(
            "Inviscous shock formation time",
            xy=(0.02, cfg.t_shock / cfg.t_extrap - 0.04),
            xycoords="axes fraction", fontsize=12, color=GREEN
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("$u(x, t)$", fontsize=16)

    ax.set_xlabel("$x$",  fontsize=14)
    ax.set_ylabel("$t$",  fontsize=14)
    ax.set_xticks(np.linspace(0, cfg.L,        6))
    ax.set_yticks(np.linspace(0, cfg.t_extrap, 6))
    if inverse:
        ax.set_title(
            rf"Predicted $u(x, t)$  --  $\nu = {cfg.nu:.4f} \ \hat\nu = {model.nu_hat.item():.4f}$",  # noqa:E501
            fontsize=16, loc="left", pad=6, color="#444441"
        )
    else:
        ax.set_title(
            rf"Predicted $u(x, t)$  --  $\nu = {cfg.nu:.4f}$",
            fontsize=16, loc="left", pad=6, color="#444441"
        )

    fig.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_epoch_figure_2D(
    cfg:         Config,
    snapshots:   dict,
    model_color: str = GREEN,
    model_label: str = "PINN",
    fig_title:   str = r"PINN -- predicted $u(x, t)$ after training epochs",
    output_path: Path = None,
    show:        bool = True
) -> None:
    """
    Grid of 2D heatmaps of the predicted solution, one per snapshot epoch.

    Parameters
    ----------
    cfg         : Config
    snapshots   : dict mapping epoch -> CPU state_dict
    model_color : unused, kept for API consistency
    model_label : label prefix in panel titles
    fig_title   : figure suptitle
    output_path : path to save the figure
    show        : whether to display the figure
    """
    epochs = sorted(snapshots.keys())
    n_snap = len(epochs)
    n_cols = 2
    n_rows = (n_snap + 1) // n_cols

    x = np.linspace(0, cfg.L,        200)
    t = np.linspace(0, cfg.t_extrap, 200)
    x_matrix, t_matrix = np.meshgrid(x, t)

    # detect model type once
    inverse = any("_nu_raw" in k for k in snapshots[epochs[0]].keys())

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 4.5),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    # shared colour scale across all panels
    v = None

    # first pass — compute all predictions and find global vmax
    u_matrices = {}
    for epoch in epochs:
        tmp = InverseFCNet(cfg) if inverse else FCNet(cfg)
        tmp.load_state_dict(snapshots[epoch])
        tmp.eval()
        u_flat = predict(tmp, t_matrix.ravel(), x_matrix.ravel())
        u_matrices[epoch] = u_flat.reshape(x_matrix.shape)

    v = max(np.max(np.abs(m)) for m in u_matrices.values())

    # second pass — plot
    for idx, epoch in enumerate(epochs):
        ax = axes_flat[idx]
        u_matrix = u_matrices[epoch]

        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(LGRAY)

        im = ax.imshow(
            u_matrix,
            origin="lower",
            extent=[0, cfg.L, 0, cfg.t_extrap],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )

        if cfg.t_shock is not None:
            ax.axhline(cfg.t_shock, color=GREEN, lw=1.5, ls="--")

        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="$u(x,t)$")

        # training domain boundary
        if not cfg.train_extrap:
            ax.axhline(cfg.t_dom, color=GRAY, lw=1.0, ls="--", alpha=0.7)

        ax.set_title(rf"Epoch = {epoch}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xticks(np.linspace(0, cfg.L,        5))
        ax.set_yticks(np.linspace(0, cfg.t_extrap, 5))

        if idx % n_cols == 0:
            ax.set_ylabel("$t$", fontsize=10)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("$x$", fontsize=10)

        if idx == 0:
            if not cfg.train_extrap:
                ax.annotate(
                    "training",
                    xy=(0.02, cfg.t_dom / cfg.t_extrap - 0.04),
                    xycoords="axes fraction",
                    fontsize=7,
                    color=GRAY
                    )
                ax.annotate(
                    "extrap",
                    xy=(0.02, cfg.t_dom / cfg.t_extrap + 0.01),
                    xycoords="axes fraction",
                    fontsize=7,
                    color=GRAY
                    )
            if cfg.t_shock is not None:
                ax.annotate(
                    "shock",
                    xy=(0.02, cfg.t_shock / cfg.t_extrap - 0.04),
                    xycoords="axes fraction",
                    fontsize=7,
                    color=GREEN
                    )

    for idx in range(n_snap, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(fig_title, fontsize=12, color="#2C2C2A")
    fig.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_summary_2D(
    model:       nn.Module,
    history:     dict,
    cfg:         Config,
    snapshots:   dict,
    data:        dict,
    u_grid:      np.ndarray,
    t_arr:       np.ndarray,
    output_path: Path = None,
    show:        bool = True
) -> None:
    # -- detect inverse mode ----------------------------------------------
    inverse = any("_nu_raw" in k for v in snapshots.values() for k in v.keys())

    # -- grid for 2D panels ------------------------------------------------
    x = np.linspace(0, cfg.L,        200)
    t = np.linspace(0, cfg.t_extrap, 200)
    x_mat, t_mat = np.meshgrid(x, t)

    u_true_flat = interpolate_solution_arr(
        u_grid, t_arr, x_mat.ravel(), t_mat.ravel(), cfg
    )
    u_true = u_true_flat.reshape(x_mat.shape)

    u_pred = predict(model, t_mat.ravel(), x_mat.ravel()).reshape(x_mat.shape)

    # -- physics residual (finite difference on prediction grid) -----------
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    u_t = np.gradient(u_pred, dt, axis=0)
    u_x = np.gradient(u_pred, dx, axis=1)
    u_xx = np.gradient(u_x,    dx, axis=1)
    phys_res = np.abs(u_t + u_pred * u_x - cfg.nu * u_xx)**2

    # -- data residual -----------------------------------------------------
    data_res = (u_pred - u_true) ** 2

    # -- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG)

    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    extent = [0, cfg.L, 0, cfg.t_extrap]
    v = np.max(np.abs(u_true))

    def style_2d(ax):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(LGRAY)
        ax.set_xlabel("$x$",  fontsize=10)
        ax.set_ylabel("$t$",  fontsize=10)
        if not cfg.train_extrap:
            ax.axhline(cfg.t_dom, color=GRAY, lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(np.linspace(0, cfg.L,        5))
        ax.set_yticks(np.linspace(0, cfg.t_extrap, 5))

    def add_cbar(fig, im, ax, label):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=9)

    # -- panel 1: ground truth ---------------------------------------------
    style_2d(ax1)
    im1 = ax1.imshow(u_true, origin="lower", extent=extent,
                     aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    add_cbar(fig, im1, ax1, "$u$")
    ax1.set_title(r"Panel 1 -- Ground truth $u(x,t)$",
                  fontsize=9, loc="left", color="#444441")
    if cfg.t_shock is not None:
        ax1.axhline(cfg.t_shock, color=GREEN, lw=1.5, ls="--")
        ax1.annotate("Inviscous shock formation time",
                     xy=(0.02, cfg.t_shock / cfg.t_extrap - 0.04),
                     xycoords="axes fraction", fontsize=7, color=GREEN)

    # -- panel 2: prediction -----------------------------------------------
    style_2d(ax2)
    im2 = ax2.imshow(u_pred, origin="lower", extent=extent,
                     aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    add_cbar(fig, im2, ax2, r"$\hat{u}$")
    ax2.set_title(rf"Panel 2 -- PINN prediction $\hat{{u}}(x,t)$  |  $\nu={cfg.nu:.4f}$",  # noqa:E501
                  fontsize=9, loc="left", color="#444441")
    if cfg.t_shock is not None:
        ax2.axhline(cfg.t_shock, color=GREEN, lw=1.5, ls="--")
        ax2.annotate("Inviscous shock formation time",
                     xy=(0.02, cfg.t_shock / cfg.t_extrap - 0.04),
                     xycoords="axes fraction", fontsize=7, color=GREEN)

    # -- panel 3: physics residual -----------------------------------------
    style_2d(ax3)
    im3 = ax3.imshow(phys_res, origin="lower", extent=extent,
                     aspect="auto", cmap="Reds", vmin=0)
    add_cbar(fig, im3, ax3, "$|r|$")
    mean_phys = np.sqrt(np.mean(phys_res))
    ax3.set_title(rf"Panel 3 -- Physics residual $|u_t + u\,u_x - \nu u_{{xx}}|^2 -- MSR = {mean_phys:.4f}$",  # noqa:E501
                  fontsize=9, loc="left", color="#444441")

    # -- panel 4: 3D surface (prediction) + observation scatter -----------
    ax4.plot_surface(x_mat, t_mat, u_pred,
                     cmap="RdBu_r", alpha=0.7, linewidth=0, antialiased=True,
                     vmin=-v, vmax=v)
    if cfg.use_data:
        ax4.scatter(data["x_obs"], data["t_obs"], data["u_obs"],
                    color=BLUE, s=8, zorder=5, alpha=0.6, label="Observations")
        ax4.legend()

    ax4.set_xlabel("$x$",    fontsize=9, labelpad=4)
    ax4.set_ylabel("$t$",    fontsize=9, labelpad=4)
    ax4.set_zlabel("$u$",    fontsize=9, labelpad=4)
    ax4.set_title("Panel 4 -- PINN prediction in 3D",
                  fontsize=9, loc="left", color="#444441")
    ax4.tick_params(labelsize=7)

    # -- panel 5: loss curves ----------------------------------------------
    ax5.set_facecolor(PANEL)
    for spine in ax5.spines.values():
        spine.set_edgecolor(LGRAY)

    LOSS_STYLE = {
        "loss_data":  (BLUE,   "-",  r"$\mathcal{L}_{data}$"),
        "loss_phys":  (GREEN,  "-",  r"$\mathcal{L}_{phys}$"),
        "loss_ic":    (PURPLE, "-",  r"$\mathcal{L}_{ic}$"),
        "loss_bc":    (ORANGE, "-",  r"$\mathcal{L}_{bc}$"),
        "loss_val":   (RED,    "--", r"$\mathcal{L}_{val}$"),
        "loss_total": (GRAY,   ":",  r"$\mathcal{L}_{total}$"),
    }
    epochs = history["epoch"]
    for key, (color, ls, label) in LOSS_STYLE.items():
        if key in history and any(v > 0 for v in history[key]):
            ax5.semilogy(epochs, history[key],
                         color=color, ls=ls, lw=1.6, label=label)

    for ep in cfg.snapshot_epochs:
        if ep <= max(epochs):
            ax5.axvline(ep, color=LGRAY, lw=0.6, ls=":", zorder=0)

    # -- panel 6: data residual --------------------------------------------
    style_2d(ax6)
    im6 = ax6.imshow(data_res, origin="lower", extent=extent,
                     aspect="auto", cmap="Reds", vmin=0)
    add_cbar(fig, im6, ax6, r"$|\hat{u}-u|^2$")
    mean_data = np.sqrt(np.mean(data_res))
    ax6.set_title(rf"Panel 6 -- Data residual $|\hat{{u}}(x,t) - u(x,t)|^2 -- RMSE = {mean_data:.4f}$",  # noqa:E501
                  fontsize=9, loc="left", color="#444441")

    # -- early stopping marker on loss plot (ax5) -------------------------
    final_epoch = max(epochs)
    if final_epoch < cfg.n_epochs:
        ax5.axvline(final_epoch, color=RED, lw=1.2, ls="--", zorder=1)
        ax5.annotate(f"Stopped\n@ {final_epoch:.0f}",
                     xy=(final_epoch, 1),
                     xycoords=("data", "axes fraction"),
                     xytext=(-28, -20), textcoords="offset points",
                     fontsize=7, color=RED, va="top")

    ax5.set_xlabel("Epoch",      fontsize=10)
    ax5.set_ylabel("Loss (log)", fontsize=10)
    ax5.set_xlim(min(epochs), max(epochs))
    ax5.legend(fontsize=8, framealpha=0.5, ncol=2)

    stopped_str = (
        f"stopped @ {final_epoch}"
        if final_epoch < cfg.n_epochs
        else f"ran full {cfg.n_epochs} epochs"
        )
    ax5.set_title(f"Panel 5 -- Training loss curves  |  {stopped_str}",
                  fontsize=9, loc="left", color="#444441")

    # -- suptitle ----------------------------------------------------------
    if inverse:
        fig.suptitle(
            rf"Burgers PINN summary  |  $\nu={cfg.nu:.3f} \ \hat\nu={model.nu_hat.item():.3f}$  |  "  # noqa:E501
            rf"IC: {cfg.ic}  |  $t_{{dom}}={cfg.t_extrap:.3f}$",
            fontsize=11, color="#2C2C2A"
        )
    elif not cfg.train_extrap:
        fig.suptitle(
            rf"Burgers PINN summary  |  $\nu={cfg.nu:.3f}$  |  "
            rf"IC: {cfg.ic}  |  $t_{{dom}}={cfg.t_dom:.3f}$  $t_{{extrap}}={cfg.t_extrap:.3f}$",  # noqa:E501
            fontsize=11, color="#2C2C2A"
        )
    else:
        fig.suptitle(
            rf"Burgers PINN summary  |  $\nu={cfg.nu:.3f}$  |  "
            rf"IC: {cfg.ic}  |  $t_{{dom}}={cfg.t_extrap:.3f}$",
            fontsize=11, color="#2C2C2A"
        )

    fig.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_predicted_parameter_convergence(
        history: dict,
        cfg: Config,
        output_path: Path = None,
        show: bool = True
) -> None:
    """
    Plot the convergence of the predicted parameter (e.g. viscosity) over epochs.  # noqa:E501

    Parameters
    ----------
    history : dict from train(), must contain "nu_hat" key with list of predicted nu values per epoch  # noqa:E501
    cfg     : Config, used for true nu value and snapshot epoch markers
    output_path : Path to save the figure
    show : Whether to display the figure
    """

    nu_hat_history = history["nu_hat"]
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(10, 5))
    style_ax(ax)
    fig.patch.set_facecolor(BG)

    ax.plot(epochs, nu_hat_history, color=GREEN,
            lw=2.0, label=r"Predicted $\hat{\nu}$")
    ax.axhline(cfg.nu, color=GRAY, lw=1.5, ls="--", label=r"True $\nu$")

    # snapshot epoch markers
    for ep in cfg.snapshot_epochs:
        if ep <= max(epochs):
            ax.axvline(ep, color=LGRAY, lw=0.8, ls=":", zorder=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Viscosity $\nu$", fontsize=12)
    ax.set_title(r"Convergence of $\hat{\nu}$",
                 fontsize=14, loc="left", color="#444441")
    ax.legend(fontsize=10, framealpha=0.5)
    ax.set_xlim(min(epochs), max(epochs))
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_method_of_characteristics(
        cfg: Config,
        samples: int = 50,
        output_path: Path = None,
        show: bool = True
) -> None:
    """
    Plot the method of characteristics for Burgers' equation.
    Each characteristic is a straight line x(t) = x0 + u0 * t,
    since u is constant along characteristics (inviscid assumption).
    """

    u = u_0(cfg)
    index_samples = np.round(np.linspace(0, cfg.n_x - 1, samples)).astype(int)
    u_samples = u[index_samples]
    x_samples = index_samples * cfg.delta_x

    t_end = cfg.t_extrap
    t_line = np.linspace(0, t_end, 200)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (x0, u0) in enumerate(zip(x_samples, u_samples)):
        x_char = x0 + u0 * t_line
        ax.plot(x_char, t_line, lw=1.2, color='blue',
                label="Characteristics" if i == 0 else None)
    if cfg.t_shock is None:
        print("Cannot plot shock time, shock time is too small/large")
    else:
        ax.axhline(cfg.t_shock, color='red',
                   linestyle='--', label="Shockwave time")

    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$t$", fontsize=14)
    ax.set_title("Method of Characteristics", fontsize=16)
    ax.set_xlim(0, cfg.L)
    ax.set_ylim(0, t_end)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_show(output_path=output_path, show=show)


def plot_three_times(
    model:       nn.Module,
    cfg:         Config,
    snapshots:   dict,
    u_grid:      np.ndarray,
    t_arr:       np.ndarray,
    times:       list[float] = None,
    model_color: str = GREEN,
    model_label: str = "PINN",
    clip_y:      bool = False,
    output_path: Path = None,
    show:        bool = True
) -> None:
    """
    Makt three plots at three times;
        -One at t=0
        -One at a time before the shock time
        -One at the shock time

    Each panel shows:
        - numerical solution u(x, t)     (gray line)
        - model prediction u_hat(x, t)   (coloured line)
        - RMSE annotated in panel title

    Parameters
    ----------
    model       : trained PINN on CPU
    data        : output of generate_data()
    cfg         : Config
    u_grid      : numerical solution, shape (n_t, n_x)
    t_arr       : time array from lax_wendroff
    n_times     : number of time slices (ignored if times is provided)
    times       : optional explicit list of times to plot
    """
    # -- detect inverse mode ----------------------------------------------
    inverse = any("_nu_raw" in k for v in snapshots.values() for k in v.keys())

    if not cfg.t_shock:
        print("Shock time not defined! Not making plot.")
    else:
        times = [0, cfg.t_shock / 2, cfg.t_shock]

    x_plot = np.linspace(0, cfg.L, 300)
    n_cols = 2

    fig, axes = plt.subplots(3, 1,
                             figsize=(6, 10),
                             sharex=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    plot_titles = [
        "Initial condition",
        "Before shock time",
        "At shock time"
    ]
    for idx, t_val in enumerate(times):
        ax = axes_flat[idx]
        style_ax(ax)
        ax.grid(True, linestyle="--", alpha=0.6)

        # -- numerical solution --------------------------------------------
        u_true = interpolate_solution_arr(
            u_grid, t_arr,
            x_plot, np.full_like(x_plot, t_val),
            cfg
        )

        # -- model prediction ----------------------------------------------
        u_pred = predict(model, np.full_like(x_plot, t_val), x_plot)
        if clip_y:
            u_pred = np.clip(u_pred, -3, 3)

        err = np.sqrt(np.mean((u_pred - u_true) ** 2))

        ax.plot(x_plot, u_true, color=GRAY,        lw=1.4, alpha=0.9)
        ax.plot(x_plot, u_pred, color=model_color,  lw=2.0)
        ax.set_xlabel("$x$", fontsize=8)

        # -- annotate time and RMSE ----------------------------------------
        ax.set_title(rf"{plot_titles[idx]}  -  $t = {t_val:.3f}$ s  -  RMSE $= {err:.4f}$",  # noqa:E501
                     fontsize=10, color="#2C2C2A")
    # -- shared legend -----------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color=GRAY,        lw=1.4,
               label="Numerical solution $u(x, t)$"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=rf"{model_label} prediction $\hat{{u}}(x, t)$"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=12, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    if inverse:
        fig.suptitle(
            rf"Ground truth and predicted solution $u(x, t)$ — $\nu = {cfg.nu:.4f} \ \hat\nu = {model.nu_hat.item():.4f}$",  # noqa:E501
            fontsize=12, color="#2C2C2A"
        )
    else:
        fig.suptitle(
            rf"Ground truth and predicted solution $u(x, t)$ — $\nu = {cfg.nu:.4f}$",  # noqa:E501
            fontsize=16, color="#2C2C2A"
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_show(output_path=output_path, show=show)


def save_model_plots(
    model:       nn.Module,
    history:     dict,
    snapshots:   dict,
    data:        dict,
    cfg:         Config,
    inverse:     bool,
    output_path: Path
) -> None:
    """
    Generate and save all summary and epoch figures for a trained model.
    """
    model.to('cpu')

    plot_gt_1D(
        cfg=cfg,
        u_grid=data["u_grid"],
        t_arr=data["t_arr"],
        times=[0, 2, 4, 6],
        output_path=output_path / "gt_solution.png",
        show=False
    )

    plot_pred_1D(
        model=model,
        cfg=cfg,
        snapshots=snapshots,
        u_grid=data["u_grid"],
        t_arr=data["t_arr"],
        times=[0, 2, 4, 6],
        output_path=output_path / "pred_solution.png",
        show=False
    )

    plot_solution_grid(
        model=model,
        data=data,
        cfg=cfg,
        snapshots=snapshots,
        u_grid=data["u_grid"],
        t_arr=data["t_arr"],
        output_path=output_path / "pred_grid.png",
        show=False
    )

    if cfg.t_shock is not None:
        plot_three_times(
            model=model,
            cfg=cfg,
            snapshots=snapshots,
            u_grid=data["u_grid"],
            t_arr=data["t_arr"],
            output_path=output_path / "3_times.png",
            show=False
        )

    plot_losses(
        history=history,
        cfg=cfg,
        output_path=output_path / "losses.png",
        show=False
    )

    plot_gt_2D(
        cfg=cfg,
        u_grid=data["u_grid"],
        t_arr=data["t_arr"],
        output_path=output_path / "gt_2D.png",
        show=False
    )

    plot_pred_2D(
        model=model,
        cfg=cfg,
        snapshots=snapshots,
        output_path=output_path / "pred_2D.png",
        show=False
    )

    plot_epoch_figure_2D(
        cfg=cfg,
        snapshots=snapshots,
        output_path=output_path / "epochs_2D.png",
        show=False
    )

    plot_summary_2D(
        model=model,
        history=history,
        cfg=cfg,
        snapshots=snapshots,
        data=data,
        u_grid=data["u_grid"],
        t_arr=data["t_arr"],
        output_path=output_path / "summary.png",
        show=False
    )

    plot_method_of_characteristics(
        cfg=cfg,
        output_path=output_path / "method_of_characteristics.png",
        show=False
    )

    if inverse:
        plot_predicted_parameter_convergence(
            history=history,
            cfg=cfg,
            output_path=output_path / "parameter_convergence.png",
            show=False
        )


def save_plots_from_file(
        folder_path: str
) -> None:
    """
    Load a model als export all plots.
    """
    folder_path = Path(folder_path)

    # -- detect model type before loading ----------------------------------
    state_dict = torch.load(folder_path / "best_model.pt", map_location="cpu")
    inverse = any("_nu_raw" in k for k in state_dict.keys())

    with open(folder_path / "config.json") as f:
        cfg = Config(**json.load(f))

    model = InverseFCNet(cfg) if inverse else FCNet(cfg)
    model, history, snapshots, cfg = load_model(model, folder_path)

    data = generate_data(cfg)
    save_model_plots(model, history, snapshots,
                     data, cfg, inverse, folder_path)
    print(f"Plots saved to {folder_path}")
