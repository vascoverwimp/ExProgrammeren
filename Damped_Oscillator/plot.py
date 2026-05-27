"""
Plotting functions for the damped spring-mass PINN results.

Contains:
    plot_summary        -- five-panel overview (data, fit, loss, extrapolation,
                           ODE residual)
    plot_epoch_snapshots -- grid of trajectory panels at chosen training epochs

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from config import Config
from data import analytic, generate_data
from model import FCNet, InverseFCNet, predict, predict_from_state
from utils import (
    convert_to_mck,
    get_device,
    load_model,
    pointwise_residual,
    rmse,
    save_show,
)

# -- LaTeX font ------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# -- Color palette ------------------------------------------------
BLUE = "#378ADD"    # noisy training observations
RED = "#E24B4A"     # noisy test/validation observations
GREEN = "#1D9E75"   # PINN
ORANGE = "#EF9F27"  # collocation points
PURPLE = "#7F77DD"  # initial condition marker
GRAY = "#888780"    # true solution / neutral
LGRAY = "#D3D1C7"   # spine colour
BG = "#FAFAF8"      # figure background
PANEL = "#F1EFE8"   # axes background


def shade_extrap(cfg: Config, ax: Axes) -> None:
    """Shade extrapolated region."""
    ax.axvspan(cfg.t_dom, cfg.t_extrap, color=GRAY, alpha=0.12)
    ax.axvline(cfg.t_dom, color=GRAY, lw=0.8, ls="--", alpha=0.6)


def style_ax(ax: Axes) -> None:
    """Style axis."""
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def plot_analytic(
        cfg: Config,
        output_path: Path = None,
        show: bool = True
) -> None:
    """Plot the analytic solution to the differential equation."""
    t = np.linspace(0, cfg.t_extrap, 500)
    y = analytic(t, cfg)

    _, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(PANEL)
    shade_extrap(cfg, ax)

    ax.plot(t, y, linewidth=2, label="Analytic solution", color=GRAY)
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Solution $y(t)$", fontsize=12)
    ax.set_title(
        "Analytic Solution of the Differential Equation"
        "\n"
        rf"$m={cfg.m:.3f}\  c={cfg.c:.3f}\  k={cfg.k:.3f}   |$"
        "\n"
        rf"$\zeta={cfg.zeta:.3f}\  \omega_0={cfg.omega_0:.3f} rad/s$",
        fontsize=12
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_predicted(
        model: nn.Module,
        snapshots: dict,
        cfg: Config,
        y_pinn_full: np.ndarray,
        t_plot_full: np.ndarray,
        output_path: Path = None,
        show: bool = True
) -> None:
    """Plot the analytic and predicted solution to the ODE."""
    # -- detect inverse mode ----------------------------------------------
    inverse = any(
        "zeta_hat" in k for v in snapshots.values() for k in v.keys()
    )

    t = np.linspace(0, cfg.t_extrap, 500)
    y = analytic(t, cfg)

    y_true = analytic(t_plot_full, cfg)
    rmse_pinn = rmse(y_pinn_full, y_true)

    _, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(PANEL)
    shade_extrap(cfg, ax)

    ax.plot(t_plot_full, y_pinn_full, linewidth=2,
            label="Predicted solution", color=GREEN, zorder=5)
    ax.plot(t, y, linewidth=2, label="Analytic solution", color=GRAY)
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Solution $y(t)$", fontsize=12)
    if inverse:
        ax.set_title(
            "Predicted Solution of the Differential Equation"
            "\n"
            rf"$m={cfg.m:.3f}\  c={cfg.c:.3f}\  k={cfg.k:.3f}   |$"
            "\n"
            rf"$\zeta={cfg.zeta:.3f}\  \omega_0={cfg.omega_0:.3f} rad/s$"
            "\n"
            rf"$\hat\zeta={model.zeta_hat.item():.3f}"
            rf"\  \hat\omega_0={model.omega_0_hat.item():.3f} rad/s$"
            "\n"
            rf"$RMSE = {rmse_pinn:.5f}$",
            fontsize=12
        )
    else:
        ax.set_title(
            "Predicted Solution of the Differential Equation"
            "\n"
            rf"$m={cfg.m:.3f}\  c={cfg.c:.3f}\  k={cfg.k:.3f}   |$"
            "\n"
            rf"$\zeta={cfg.zeta:.3f}\  \omega_0={cfg.omega_0:.3f} rad/s$"
            "\n"
            rf"$RMSE = {rmse_pinn:.5f}$",
            fontsize=12
        )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_summary(
    model: nn.Module,
    hist: dict,
    data: dict,
    cfg: Config,
    snapshots: dict,
    y_pinn_full: np.ndarray,
    t_plot_full: np.ndarray,
    device: torch.device,
    output_path: Path = None,
    show: bool = True,
    model_label: str = "PINN",
) -> None:
    """
    Five-panel summary figure for a PINN.

    Panels
    ------
    1. Training data: observations, true trajectory, collocation points
    2. Fit on training interval with RMSE
    3. Training loss curves on log scale
    4. Extrapolation beyond training window
    5. Squared error over full domain
    6. Pointwise ODE residual |r(t)|

    Parameters
    ----------
    hist : dict
        Training history from train() for the PINN.
    data : dict
        Output of generate_data().
    cfg : Config
        Physical constants and domain settings.
    y_pinn_full : np.ndarray
        PINN predictions on t_plot_full.
    t_plot_full : np.ndarray
        Dense time grid over [0, t_extrap].
    device : torch.device
        Device used during training, for display purposes only.
    output_path : str
        Path to save the figure.
    show : bool
        Whether to display the figure.
    model_label : str
        Display name for the model used in legends and suptitle.
        Defaults to "PINN".
    """
    # -- detect standard ML algorithm --------------------------------------
    model_color = RED if model_label == "ML" else GREEN

    # -- detect inverse mode ----------------------------------------------
    inverse = any("zeta_hat" in k for v in snapshots.values()
                  for k in v.keys())

    # -- derived quantities ------------------------------------------------
    y_true_full = analytic(t_plot_full, cfg)
    mask_train = t_plot_full <= cfg.t_dom
    t_plot_obs = t_plot_full[mask_train]
    y_true_train = y_true_full[mask_train]

    rmse_pinn_train = rmse(y_pinn_full[mask_train], y_true_full[mask_train])
    rmse_pinn_ext = rmse(y_pinn_full[~mask_train], y_true_full[~mask_train])
    phys_res_pinn = float(
        np.mean(pointwise_residual(y_pinn_full, t_plot_full, cfg)))

    # -- layout ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_data, ax_train, ax_loss = axes[0]
    ax_extrap, ax_error, ax_phys = axes[1]

    # -- panel 1: training data --------------------------------------------
    _ic_label = (
        rf"Initial condition  $y(0)={cfg.y0}$"
        r"  [enforced via $\mathcal{L}_{ic}$]"
    )
    _panel1_title = (
        "Panel 1 -- Training data: final randomised noisy observations"
        " vs true trajectory\n"
        r"$t=0$ excluded from random observations; "
        r"$y(0)$ and $y'(0)$ enforced via $\mathcal{L}_{ic}$"
    )
    ax_data.set_title(_panel1_title,
                      fontsize=9, loc="left", pad=6, color="#444441")
    ax_data.plot(t_plot_obs, y_true_train, color=GRAY, lw=1.5,
                 label="Analytic solution  $y(t)$")
    ax_data.scatter(
        data["t_obs"], data["y_obs"],
        color=BLUE, s=50, zorder=5, marker="o",
        label=f"Noisy observations  (N={cfg.n_obs}, sigma={cfg.sigma})"
        )
    ax_data.scatter([0], [cfg.y0], color=PURPLE, s=120, marker="*", zorder=7,
                    label=_ic_label)
    ax_data.scatter(
        data["t_col_dom"][data["t_col_dom"] <= cfg.t_dom],
        np.zeros(np.sum(data["t_col_dom"] <= cfg.t_dom)) - 1.08,
        color=ORANGE, s=8, marker="|", zorder=4, alpha=0.7,
        label="Collocation points (no measurement needed)"
    )
    ax_data.set_xlabel("Time $[s]$")
    ax_data.set_ylabel("Displacement  $y(t)$")
    ax_data.legend(fontsize=8, framealpha=0.5)
    ax_data.set_xlim(0, cfg.t_dom)

    # -- panel 2: fit on training interval ---------------------------------
    ax_train.set_title(
        f"Panel 2 -- Fit on training interval  $[0, {cfg.t_dom} s]$",
        fontsize=10, loc="left", pad=6, color="#444441"
    )
    ax_train.plot(t_plot_obs, y_true_train, color=GRAY, lw=1.5, label="True")
    ax_train.plot(
        t_plot_obs, y_pinn_full[mask_train],
        color=model_color, lw=2,
        label=f"{model_label}         ($RMSE={rmse_pinn_train:.4f}$)"
        )
    ax_train.scatter(data["t_obs"], data["y_obs"],
                     color=BLUE, s=30, zorder=5, marker="o", alpha=0.6,
                     label="Train observations")
    ax_train.scatter([0], [cfg.y0], color=PURPLE, s=120, marker="*", zorder=7,
                     label=f"IC  $y(0)={cfg.y0}$")
    ax_train.set_xlabel("Time $[s]$")
    ax_train.set_ylabel("Displacement  $y(t)$")
    ax_train.legend(fontsize=7.5, framealpha=0.5)
    ax_train.set_xlim(0, cfg.t_dom)

    # -- panel 3: loss curves ----------------------------------------------
    ax_loss.set_title("Panel 3 -- Training loss  (log scale)",
                      fontsize=10, loc="left", pad=6, color="#444441")
    ax_loss.semilogy(hist["epoch"], hist["loss_data"],
                     color=model_color, lw=1.5, label="L_data")
    ax_loss.semilogy(hist["epoch"], hist["loss_phys"],
                     color=model_color, lw=1.5, label="L_physics", linestyle="--")  # noqa: E501
    ax_loss.semilogy(hist["epoch"], hist["loss_ic"],
                     color=model_color, lw=1.5, label="L_ic", linestyle="-.")
    for ep in cfg.snapshot_epochs[:-1]:
        ax_loss.axvline(ep, color=GRAY, lw=0.5, ls=":", alpha=0.5)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7, framealpha=0.5)

    # -- panel 4: extrapolation --------------------------------------------
    ax_extrap.set_title(
        f"Panel 4 -- Extrapolation beyond training window  [{cfg.t_dom}, {cfg.t_extrap} s]",  # noqa: E501
        fontsize=10, loc="left", pad=6, color="#444441"
    )
    shade_extrap(cfg, ax_extrap)
    ax_extrap.plot(t_plot_full, y_true_full, color=GRAY, lw=1.5, label="True")
    ax_extrap.plot(t_plot_full, y_pinn_full,
                   color=model_color, lw=2,
                   label=f"{model_label}         (extrap RMSE={rmse_pinn_ext:.4f})")  # noqa: E501
    ax_extrap.scatter(data["t_obs"], data["y_obs"],
                      color=BLUE, s=30, zorder=5, marker="o", alpha=0.5,
                      label="Observations")
    ax_extrap.set_xlabel("Time $[s]$")
    ax_extrap.set_ylabel("Displacement  $y(t)$")
    ax_extrap.set_ylim(-2.5, 2.5)
    ax_extrap.set_xlim(0, cfg.t_extrap)
    ax_extrap.legend(fontsize=7.5, framealpha=0.5)
    ax_extrap.annotate("Training region", xy=(0.05, 0.93),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)
    ax_extrap.annotate("Extrapolation", xy=(0.72, 0.93),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)

    # -- panel 5: pointwise data residual ----------------------------------
    ax_error.set_title(
        r"Panel 5 -- Pointwise data residual  $|y(t) - \hat{y}(t)|^2$",
        fontsize=10, loc="left", pad=6, color="#444441"
    )
    shade_extrap(cfg, ax_error)
    r_pinn = np.abs(y_true_full - y_pinn_full) ** 2
    ax_error.plot(t_plot_full, r_pinn, color=model_color, lw=1.5,
                  label=f"{model_label}  data residual  (mean={np.mean(r_pinn):.3f})")  # noqa: E501
    ax_error.set_xlabel("Time $[s]$")
    ax_error.set_ylabel(r"$|y(t) - \hat{y}(t)|²$")
    ax_error.set_xlim(0, cfg.t_extrap)
    ax_error.legend(fontsize=7.5, framealpha=0.5)

    # -- panel 6: pointwise ODE residual -----------------------------------
    ax_phys.set_title("Panel 6 -- Pointwise ODE residual  $|r(t)|^2$",
                      fontsize=10, loc="left", pad=6, color="#444441")
    shade_extrap(cfg, ax_phys)
    r_pinn = pointwise_residual(y_pinn_full, t_plot_full, cfg) ** 2
    ax_phys.plot(t_plot_full, r_pinn, color=model_color, lw=1.5,
                 label=f"{model_label}  (mean={phys_res_pinn:.3f})")
    ax_phys.set_xlabel("Time $[s]$")
    ax_phys.set_ylabel(r"$|my'' + cy' + ky|^2$")
    ax_phys.set_xlim(0, cfg.t_extrap)
    ax_phys.legend(fontsize=7.5, framealpha=0.5)

    # -- suptitle ----------------------------------------------------------
    if inverse:
        fig.suptitle(
            rf"{model_label}  |  device={device}  |  "
            rf"$m={cfg.m:.3f}\  c={cfg.c:.3f}\  k={cfg.k:.3f}   |$"
            rf"$\zeta={cfg.zeta:.3f}\  \omega_0={cfg.omega_0:.3f} rad/s$"
            "\n"
            rf"$\hat{{\zeta}}={model.zeta_hat.item():.3f}"
            rf"\  \hat{{\omega_0}}={model.omega_0_hat.item():.3f} rad/s$"
            "\n"
            rf"{cfg.n_obs} observations ($t>0$)  $\sigma={cfg.sigma}$  |  "
            rf"{cfg.n_col_dom} collocation pts  |  "
            rf"$\lambda_{{phys}}={cfg.lambda_phys}  \lambda_{{ic}}={cfg.lambda_ic}$  |  "  # noqa: E501
            rf"IC: $y(0)={cfg.y0}\  y'(0)={cfg.dy0}$",
            fontsize=12, y=0.975, color="#131313"
        )
    else:
        fig.suptitle(
            rf"{model_label}  |  device={device}  |  "
            rf"$m={cfg.m:.3f}\  c={cfg.c:.3f}\  k={cfg.k:.3f}   |$"
            rf"$\zeta={cfg.zeta:.3f}\  \omega_0={cfg.omega_0:.3f} rad/s$"
            "\n"
            rf"{cfg.n_obs} observations ($t>0$)  $\sigma={cfg.sigma}$  |  "
            rf"{cfg.n_col_dom} collocation pts  |  "
            rf"$\lambda_{{phys}}={cfg.lambda_phys}  \lambda_{{ic}}={cfg.lambda_ic}$  |  "  # noqa: E501
            rf"IC: $y(0)={cfg.y0}\  y'(0)={cfg.dy0}$",
            fontsize=12, y=0.975, color="#131313"
        )

    plt.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_epoch_figure(
    model: nn.Module,
    data: dict,
    cfg: Config,
    snapshots: dict,
    t_plot_full: np.ndarray,
    model_label: str = "PINN",
    clip_y: bool = False,
    output_path: Path = None,
    show: bool = True,
) -> None:
    """Plot a grid of trajectory panels at chosen training epochs."""
    # -- detect standard ML algorithm --------------------------------------
    model_color = RED if model_label == "ML" else GREEN

    # -- detect inverse mode ----------------------------------------------
    inverse = any("zeta_hat" in k for v in snapshots.values()
                  for k in v.keys())

    epochs = sorted(snapshots.keys())
    n_snap = len(epochs)
    n_cols = 2
    n_rows = (n_snap + 1) // n_cols

    # -- derived quantities ------------------------------------------------
    y_true_full = analytic(t_plot_full, cfg)
    t_col_full = np.concatenate([data["t_col_dom"], data["t_col_extrap"]])

    # -- layout ------------------------------------------------------------
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 3.2),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    y_lo, y_hi = -2.2, 1.6
    col_y = y_lo + 0.08 * (y_hi - y_lo)

    for idx, epoch in enumerate(epochs):
        ax = axes_flat[idx]
        style_ax(ax)

        y_pred = predict_from_state(snapshots[epoch], t_plot_full, cfg)
        if clip_y:
            y_pred = np.clip(y_pred, -3, 3)

        err = rmse(y_pred, y_true_full)

        ax.plot(t_plot_full, y_true_full, color=GRAY, lw=1.4, alpha=0.9)
        ax.plot(t_plot_full, y_pred, color=model_color, lw=2.0)
        shade_extrap(cfg, ax)
        ax.scatter(t_col_full, np.full_like(t_col_full, col_y),
                   color=ORANGE, s=6, marker="|", alpha=0.6, zorder=3)
        ax.scatter(data["t_obs"], data["y_obs"],
                   color=BLUE, s=28, marker="o", zorder=5, alpha=0.85)
        ax.scatter([0], [cfg.y0], color=PURPLE, s=130, marker="*", zorder=7)

        ax.set_title(rf"Epoch = {epoch}   |   RMSE = {err:.4f}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xlim(0, cfg.t_extrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("$y(t)$", fontsize=8)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time $[s]$", fontsize=8)

        if idx == 0:
            ax.text(cfg.t_dom / 2, y_hi - 0.15, "training",
                    ha="center", va="top", fontsize=7, color=GRAY)
            ax.text(cfg.t_dom + (cfg.t_extrap - cfg.t_dom) / 2,
                    y_hi - 0.15, "extrapolation",
                    ha="center", va="top", fontsize=7, color=GRAY)

    for idx in range(n_snap, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # -- shared legend -----------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color=GRAY, lw=1.4,
               label="True solution  $y(t)$"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=rf"{model_label} prediction  $\hat{{y}}(t)$"),
        Line2D([0], [0], color=BLUE, lw=0, marker="o", markersize=5,
               label=rf"Train observations  (N={len(data['t_obs'])}, $\sigma={cfg.sigma}$)"),  # noqa: E501
        Line2D([0], [0], color=PURPLE, lw=0, marker="*", markersize=9,
               label=rf"IC  $y(0)={cfg.y0}$,  $y'(0)={cfg.dy0}$"),
        Line2D([0], [0], color=ORANGE, lw=0, marker="|", markersize=7,
               label=rf"Collocation pts  (N={len(t_col_full)}, $[0, {cfg.t_extrap}]$)"),  # noqa: E501
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    if inverse:
        fig.suptitle(
            "Predicted Solution of the Differential Equation"
            "\n"
            rf"$m={cfg.m:.3f}\  c={cfg.c:.3f}\  k={cfg.k:.3f}   |$"
            "\n"
            rf"$\zeta={cfg.zeta:.3f}\  \omega_0={cfg.omega_0:.3f} rad/s$"
            "\n"
            rf"$\hat\zeta={model.zeta_hat.item():.3f}"
            rf"\  \hat\omega_0={model.omega_0_hat.item():.3f} rad/s$",
            fontsize=12
        )
    else:
        fig.suptitle(
            "Predicted Solution of the Differential Equation"
            "\n"
            rf"$m={cfg.m:.3f}\  c={cfg.c:.3f}\  k={cfg.k:.3f}   |$"
            "\n"
            rf"$\zeta={cfg.zeta:.3f}\  \omega_0={cfg.omega_0:.3f} rad/s$",
            fontsize=12
        )

    plt.tight_layout()
    save_show(output_path=output_path, show=show)


def plot_losses(
    history: dict,
    cfg: Config,
    output_path: Path = None,
    show: bool = True,
) -> None:
    """
    Plot all training loss curves on a single log-scale figure.

    Parameters
    ----------
    history : dict
        From train(), keys: epoch, loss_data, loss_phys,
        loss_ic, loss_val, loss_total.
    cfg : Config
        Used for snapshot epoch markers.
    output_path : Path
        Path to save the figure.
    show : bool
        Whether to display the figure.
    """
    loss_style = {
        "loss_data":  (BLUE,   "-",  r"$\mathcal{L}_{data}$"),
        "loss_phys":  (GREEN,  "-",  r"$\mathcal{L}_{phys}$"),
        "loss_ic":    (PURPLE, "-",  r"$\mathcal{L}_{ic}$"),
        "loss_val":   (RED,    "--", r"$\mathcal{L}_{val}$"),
        "loss_total": (GRAY,   ":",  r"$\mathcal{L}_{total}$"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    epochs = history["epoch"]
    for key, (color, ls, label) in loss_style.items():
        if key in history and any(v > 0 for v in history[key]):
            ax.semilogy(epochs, history[key],
                        color=color, ls=ls, lw=1.8, label=label)

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


def plot_predicted_parameter_convergence(
        history: dict,
        cfg: Config,
        output_path: Path = None,
        show: bool = True,
) -> None:
    """
    Plot convergence of the predicted parameters over epochs.

    Parameters
    ----------
    history : dict
        From train(), must contain "zeta_hat" and "omega_0_hat" keys.
    cfg : Config
        Used for true parameter values and snapshot epoch markers.
    output_path : Path
        Path to save the figure.
    show : bool
        Whether to display the figure.
    """
    zeta_hat_history = history["zeta_hat"]
    omega_0_hat_history = history["omega_0_hat"]
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(10, 5))
    style_ax(ax)
    fig.patch.set_facecolor(BG)

    ax.plot(epochs, zeta_hat_history, color=GREEN, lw=2.0,
            label=r"Predicted $\hat{\zeta}$")
    ax.axhline(cfg.zeta, color=GRAY, lw=1.5, ls="--", label=r"True $\zeta$")
    ax.plot(epochs, omega_0_hat_history, color=ORANGE, lw=2.0,
            label=r"Predicted $\hat{\omega}_0$")
    ax.axhline(cfg.omega_0, color=GRAY, lw=1.5,
               ls=":", label=r"True $\omega_0$")

    for ep in cfg.snapshot_epochs:
        if ep <= max(epochs):
            ax.axvline(ep, color=LGRAY, lw=0.8, ls=":", zorder=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Parameters", fontsize=12)
    ax.set_title(r"Convergence of $\hat{\zeta}$ and $\hat{\omega}_0$",
                 fontsize=14, loc="left", color="#444441")
    ax.legend(fontsize=10, framealpha=0.5)
    ax.set_xlim(min(epochs), max(epochs))
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_show(output_path=output_path, show=show)


def save_model_plots(
    model: torch.nn.Module,
    history: dict,
    snapshots: dict,
    data: dict,
    cfg: Config,
    device: torch.device,
    inverse: bool,
    output_path: Path,
    model_label: str = "PINN",
) -> None:
    """Generate and save all summary and epoch figures for a trained model."""
    model.to("cpu")
    t_plot = np.linspace(0.0, cfg.t_extrap, 500)
    y_pinn = predict(model, t_plot)

    plot_analytic(
        cfg=cfg,
        output_path=output_path / "analytic_solution.png",
        show=False,
    )
    plot_predicted(
        model=model,
        cfg=cfg,
        snapshots=snapshots,
        y_pinn_full=y_pinn,
        t_plot_full=t_plot,
        output_path=output_path / "predicted_solution.png",
        show=False,
    )
    plot_summary(
        model=model,
        hist=history,
        data=data,
        cfg=cfg,
        snapshots=snapshots,
        y_pinn_full=y_pinn,
        t_plot_full=t_plot,
        device=device,
        model_label=model_label,
        output_path=output_path / "summary.png",
        show=False,
    )
    plot_epoch_figure(
        model=model,
        data=data,
        cfg=cfg,
        snapshots=snapshots,
        t_plot_full=t_plot,
        model_label=model_label,
        clip_y=False,
        output_path=output_path / "epoch_snapshots.png",
        show=False,
    )
    plot_losses(
        history=history,
        cfg=cfg,
        output_path=output_path / "losses.png",
        show=False,
    )
    if inverse:
        plot_predicted_parameter_convergence(
            history=history,
            cfg=cfg,
            output_path=output_path / "parameter_convergence.png",
            show=False,
        )


def save_plots_from_file(folder_path: str, verbatim: bool = True) -> None:
    """Load a model and export all plots."""
    folder_path = Path(folder_path)
    device = get_device()

    # -- detect model type before loading ----------------------------------
    state_dict = torch.load(folder_path / "best_model.pt", map_location="cpu")
    inverse = any(
        "nu_hat" in k or "zeta_hat" in k or "omega_0_hat" in k
        for k in state_dict.keys()
    )

    with open(folder_path / "config.json", encoding='utf-8') as f:
        cfg = Config(**json.load(f))

    model = InverseFCNet(cfg) if inverse else FCNet(cfg)
    model, history, snapshots, cfg = load_model(model, folder_path)

    data = generate_data(cfg)
    save_model_plots(model, history, snapshots, data,
                     cfg, device, inverse, folder_path)
    if verbatim:
        print(f"Plots saved to {folder_path}")


if __name__ == "__main__":
    m, c, k = convert_to_mck(zeta=0.2, omega_0=3)
    cfg_example = Config(m=m, c=c, k=k)
    plot_analytic(cfg_example, 10)
