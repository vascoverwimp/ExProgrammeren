"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
three figures:

  Fig 1  pinn_fig1_summary.png        — five-panel trajectory + loss summary
  Fig 2  pinn_fig2_param_conv.png     — ω₀ and ζ convergence over epochs
  Fig 3  pinn_fig3_pinn_epochs.png    — epoch-by-epoch trajectory snapshots

Parameter values (ω₀_hat, ζ_hat) are extracted directly from the snapshot
state_dicts saved by train.py — no re-training, no device dependency.

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
import torch

from model import DamperConfig, Predictor


# =============================================================================
# COLOUR PALETTE
# =============================================================================

BLUE   = "#378ADD"   # noisy observations / train split
TEAL   = "#1D9E75"   # PINN prediction
ORANGE = "#EF9F27"   # collocation points
PURPLE = "#7F77DD"   # IC / validation split
RED    = "#E24B4A"   # w0 parameter trace
GREEN  = "#3AAA5E"   # zeta parameter trace
GRAY   = "#888780"   # true solution / neutral
LGRAY  = "#D3D1C7"   # spines
BG     = "#FAFAF8"   # figure background
PANEL  = "#F1EFE8"   # axes background


# =============================================================================
# SHARED AXIS HELPERS
# =============================================================================

def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def shade_extrap(ax: plt.Axes, cfg: DamperConfig) -> None:
    ax.axvspan(cfg.t_train, cfg.t_extrap, color=GRAY, alpha=0.12)
    ax.axvline(cfg.t_train, color=GRAY, lw=0.8, ls="--", alpha=0.6)


# =============================================================================
# PARAMETER EXTRACTION FROM SNAPSHOTS
# =============================================================================

def extract_params(snapshots: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull zeta_hat and w0_hat scalars from every snapshot state_dict.

    Both are registered as nn.Parameter (0-d tensors) so they live in the
    state_dict under plain attribute names.  .item() converts to Python float.

    Returns
    -------
    epochs    : sorted epoch indices
    w0_vals   : w0_hat at each epoch
    zeta_vals : zeta_hat at each epoch
    """
    epochs_sorted = sorted(snapshots.keys())
    w0_vals   = np.array([snapshots[e]["w0_hat"].item()   for e in epochs_sorted])
    zeta_vals = np.array([snapshots[e]["zeta_hat"].item() for e in epochs_sorted])
    return np.array(epochs_sorted), w0_vals, zeta_vals


# =============================================================================
# POINTWISE ODE RESIDUAL  (numpy finite-difference approximation)
# =============================================================================

def pointwise_residual(
    y: np.ndarray, t: np.ndarray, cfg: DamperConfig
) -> np.ndarray:
    dy  = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(cfg.mass * d2y + cfg.damping * dy + cfg.stiffness * y)


# =============================================================================
# FIGURE 1 — summary
# =============================================================================
#
#  ┌─────────────────────────────────────────────────────┐
#  │  P1  Training data        (full width)              │
#  ├──────────────────────────┬──────────────────────────┤
#  │  P2  Fit [0, t_train]    │  P3  Loss curves         │
#  ├──────────────────────────┼──────────────────────────┤
#  │  P4  Extrapolation       │  P5  ODE residual        │
#  └──────────────────────────┴──────────────────────────┘

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    cfg: DamperConfig = bundle["cfg"]
    data              = bundle["data"]
    hist              = bundle["hist_pinn"]
    y_pinn            = bundle["y_pinn_full"]
    m                 = bundle["metrics"]
    device_str        = bundle["device_str"]

    t_obs_train  = data["t_obs_train"]
    y_obs_train  = data["y_obs_train"]
    t_obs_val    = data["t_obs_val"]
    y_obs_val    = data["y_obs_val"]
    t_obs        = data["t_obs"]
    y_obs        = data["y_obs"]
    t_col        = data["t_col"]
    t_plot_train = data["t_plot_train"]
    t_plot_full  = data["t_plot_full"]
    y_true_train = data["y_true_train"]
    y_true_full  = data["y_true_full"]

    mask_train = t_plot_full <= cfg.t_train

    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )
    ax_data   = fig.add_subplot(gs[0, :])
    ax_train  = fig.add_subplot(gs[1, 0])
    ax_loss   = fig.add_subplot(gs[1, 1])
    ax_extrap = fig.add_subplot(gs[2, 0])
    ax_phys   = fig.add_subplot(gs[2, 1])

    for ax in [ax_data, ax_train, ax_loss, ax_extrap, ax_phys]:
        style_ax(ax)

    # ── P1: training data ─────────────────────────────────────────────────────
    ax_data.set_title(
        "Panel 1 — Training data: sparse noisy observations vs true trajectory\n"
        f"t=0 excluded from observations; y(0) & y'(0) enforced via L_ic  |  "
        f"train N={len(t_obs_train)}   val N={len(t_obs_val)}",
        fontsize=9, loc="left", pad=6, color="#444441",
    )
    ax_data.plot(t_plot_train, y_true_train, color=GRAY, lw=1.5,
                 label="True trajectory  y(t)")
    ax_data.scatter(t_obs_train, y_obs_train, color=BLUE, s=55, zorder=5,
                    label=f"Train obs  (N={len(t_obs_train)}, σ={cfg.sigma})")
    ax_data.scatter(t_obs_val, y_obs_val, color=PURPLE, s=55, marker="D",
                    zorder=6, alpha=0.85,
                    label=f"Val obs  (N={len(t_obs_val)})")
    ax_data.scatter([0], [cfg.y0], color=PURPLE, s=180, marker="*", zorder=7,
                    label=f"IC  y(0)={cfg.y0}  [via L_ic]")
    col_vis = t_col[t_col <= cfg.t_train]
    # y-position for collocation ticks: just below the axis minimum
    y_min_data = min(y_obs.min(), y_true_train.min()) - 0.08
    ax_data.scatter(col_vis, np.full_like(col_vis, y_min_data),
                    color=ORANGE, s=8, marker="|", zorder=3, alpha=0.7,
                    label="Collocation pts")
    ax_data.set_xlabel("time  [s]")
    ax_data.set_ylabel("displacement  y(t)")
    ax_data.legend(fontsize=8, framealpha=0.5)
    ax_data.set_xlim(0, cfg.t_train)

    # ── P2: fit on training interval ──────────────────────────────────────────
    ax_train.set_title(
        f"Panel 2 — Fit  [0, {cfg.t_train} s]",
        fontsize=10, loc="left", pad=6, color="#444441",
    )
    ax_train.plot(t_plot_full[mask_train], y_true_full[mask_train],
                  color=GRAY, lw=1.5, label="True")
    ax_train.plot(t_plot_full[mask_train], y_pinn[mask_train],
                  color=TEAL, lw=2,
                  label=f"PINN  (RMSE={m['rmse_pinn_train']:.4f})")
    ax_train.scatter(t_obs_train, y_obs_train,
                     color=BLUE, s=30, zorder=5, alpha=0.7, label="Train obs")
    ax_train.scatter(t_obs_val, y_obs_val,
                     color=PURPLE, s=30, marker="D", zorder=6, alpha=0.7,
                     label="Val obs")
    ax_train.scatter([0], [cfg.y0],
                     color=PURPLE, s=120, marker="*", zorder=7)
    ax_train.set_xlabel("time  [s]")
    ax_train.set_ylabel("displacement  y(t)")
    ax_train.legend(fontsize=7.5, framealpha=0.5)
    ax_train.set_xlim(0, cfg.t_train)

    # ── P3: loss curves ────────────────────────────────────────────────────────
    ax_loss.set_title("Panel 3 — Training loss  (log scale)",
                      fontsize=10, loc="left", pad=6, color="#444441")
    ax_loss.semilogy(hist["epoch"], hist["loss_data"],
                     color=BLUE,   lw=1.5,           label="L_data  (train)")
    ax_loss.semilogy(hist["epoch"], hist["loss_val"],
                     color=BLUE,   lw=1.5, ls=":",   label="L_val")
    ax_loss.semilogy(hist["epoch"], hist["loss_physics"],
                     color=ORANGE, lw=1.2, ls=":",   label="L_phys")
    ax_loss.semilogy(hist["epoch"], hist["loss_ic"],
                     color=PURPLE, lw=1.2, ls="-.",  label="L_ic")
    ax_loss.semilogy(hist["epoch"], hist["loss_total"],
                     color=TEAL,   lw=2.5, alpha=0.3, label="L_total")
    for ep in cfg.snapshot_epochs[:-1]:
        ax_loss.axvline(ep, color=GRAY, lw=0.5, ls=":", alpha=0.4)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=7, framealpha=0.5)

    # ── P4: extrapolation ─────────────────────────────────────────────────────
    ax_extrap.set_title(
        f"Panel 4 — Extrapolation  [{cfg.t_train}, {cfg.t_extrap} s]",
        fontsize=10, loc="left", pad=6, color="#444441",
    )
    shade_extrap(ax_extrap, cfg)
    ax_extrap.plot(t_plot_full, y_true_full, color=GRAY, lw=1.5, label="True")
    ax_extrap.plot(t_plot_full, y_pinn,      color=TEAL, lw=2,
                   label=f"PINN  (extrap RMSE={m['rmse_pinn_ext']:.4f})")
    ax_extrap.scatter(t_obs, y_obs, color=BLUE, s=25, zorder=5,
                      alpha=0.5, label="Observations")
    ax_extrap.set_xlabel("time  [s]")
    ax_extrap.set_ylabel("displacement  y(t)")
    ax_extrap.set_ylim(-2.5, 2.5)
    ax_extrap.set_xlim(0, cfg.t_extrap)
    ax_extrap.legend(fontsize=7.5, framealpha=0.5)
    ax_extrap.annotate("training", xy=(0.14, 0.93),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)
    ax_extrap.annotate("extrapolation", xy=(0.67, 0.93),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)

    # ── P5: pointwise ODE residual ────────────────────────────────────────────
    ax_phys.set_title("Panel 5 — Pointwise ODE residual  |r(t)|",
                      fontsize=10, loc="left", pad=6, color="#444441")
    shade_extrap(ax_phys, cfg)
    r_pinn = pointwise_residual(y_pinn, t_plot_full, cfg)
    ax_phys.plot(t_plot_full, r_pinn, color=TEAL, lw=1.5,
                 label=f"PINN  (mean={m['phys_pinn']:.3f})")
    ax_phys.set_xlabel("time  [s]")
    ax_phys.set_ylabel("|m·y'' + c·y' + k·y|")
    ax_phys.set_xlim(0, cfg.t_extrap)
    ax_phys.legend(fontsize=7.5, framealpha=0.5)

    fig.suptitle(
        f"PINN parameter estimation  |  device={device_str}  |  "
        f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
        f"ζ_true={cfg.zeta:.3f}  ω₀_true={cfg.omega_0:.3f} rad/s\n"
        f"{cfg.n_obs} obs (t>0)  σ={cfg.sigma}  |  "
        f"{cfg.n_col} collocation pts  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  "
        f"|  IC: y(0)={cfg.y0}  y'(0)={cfg.dy0}",
        fontsize=9, y=0.975, color="#2C2C2A",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 1 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 2 — parameter convergence
# =============================================================================
#
#  ┌──────────────────────────┬──────────────────────────┐
#  │  ω₀_hat vs epoch         │  ζ_hat  vs epoch         │
#  │  (true ω₀ dashed)        │  (true ζ  dashed)        │
#  ├──────────────────────────┼──────────────────────────┤
#  │  |Δω₀| / ω₀  (log)       │  |Δζ|  / ζ  (log)        │
#  └──────────────────────────┴──────────────────────────┘

def make_param_convergence_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Visualise how ω₀_hat and ζ_hat evolve from their initial guesses toward
    the true values over the snapshot epochs.

    Top row : absolute value of each parameter over epochs, with the true
              value as a dashed horizontal reference and a ±5 % band.
    Bottom  : relative errors on a log scale, with a 1 % target line.

    Note: resolution is limited to cfg.snapshot_epochs.  For a denser trace
    also log the scalar values in train.py's history dict each log_every step.
    """
    cfg: DamperConfig = bundle["cfg"]
    snaps             = bundle["snaps_pinn"]

    epochs, w0_vals, zeta_vals = extract_params(snaps)

    true_w0   = cfg.omega_0
    true_zeta = cfg.zeta
    rel_err_w0   = np.abs(w0_vals   - true_w0)   / true_w0
    rel_err_zeta = np.abs(zeta_vals - true_zeta)  / true_zeta

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8),
        gridspec_kw={"height_ratios": [1.6, 1]},
    )
    fig.patch.set_facecolor(BG)
    ax_w0, ax_zeta, ax_rel_w0, ax_rel_zeta = axes.flatten()

    for ax in [ax_w0, ax_zeta, ax_rel_w0, ax_rel_zeta]:
        style_ax(ax)

    ms = 7   # marker size for snapshot dots

    # ── Top-left: ω₀_hat ──────────────────────────────────────────────────────
    ax_w0.set_title("ω₀ estimation over training",
                    fontsize=10, loc="left", pad=6, color="#444441")
    ax_w0.axhline(true_w0, color=GRAY, lw=1.2, ls="--",
                  label=f"True ω₀ = {true_w0:.4f} rad/s")
    ax_w0.fill_between(
        [epochs[0], epochs[-1]],
        true_w0 * 0.95, true_w0 * 1.05,
        color=RED, alpha=0.07, label="±5 % band",
    )
    ax_w0.plot(epochs, w0_vals, color=RED, lw=2, marker="o", ms=ms,
               label="ω₀_hat  (snapshots)")
    ax_w0.set_xlabel("epoch")
    ax_w0.set_ylabel("ω₀_hat  [rad/s]")
    ax_w0.legend(fontsize=8, framealpha=0.5)
    ax_w0.set_xlim(left=0)
    _annotate_final(ax_w0, epochs, w0_vals, true_w0, unit="rad/s")

    # ── Top-right: ζ_hat ──────────────────────────────────────────────────────
    ax_zeta.set_title("ζ estimation over training",
                      fontsize=10, loc="left", pad=6, color="#444441")
    ax_zeta.axhline(true_zeta, color=GRAY, lw=1.2, ls="--",
                    label=f"True ζ = {true_zeta:.4f}")
    ax_zeta.fill_between(
        [epochs[0], epochs[-1]],
        true_zeta * 0.95, true_zeta * 1.05,
        color=GREEN, alpha=0.07, label="±5 % band",
    )
    ax_zeta.plot(epochs, zeta_vals, color=GREEN, lw=2, marker="o", ms=ms,
                 label="ζ_hat  (snapshots)")
    ax_zeta.set_xlabel("epoch")
    ax_zeta.set_ylabel("ζ_hat  [—]")
    ax_zeta.legend(fontsize=8, framealpha=0.5)
    ax_zeta.set_xlim(left=0)
    _annotate_final(ax_zeta, epochs, zeta_vals, true_zeta, unit="")

    # ── Bottom-left: relative error ω₀ (log scale) ────────────────────────────
    ax_rel_w0.set_title("|Δω₀| / ω₀  — relative error  (log scale)",
                        fontsize=9, loc="left", pad=6, color="#444441")
    ax_rel_w0.semilogy(epochs, rel_err_w0, color=RED, lw=2, marker="o", ms=ms)
    ax_rel_w0.axhline(0.01, color=GRAY, lw=0.8, ls=":", alpha=0.7,
                      label="1 % target")
    ax_rel_w0.set_xlabel("epoch")
    ax_rel_w0.set_ylabel("|Δω₀| / ω₀")
    ax_rel_w0.legend(fontsize=8, framealpha=0.5)
    ax_rel_w0.set_xlim(left=0)

    # ── Bottom-right: relative error ζ (log scale) ────────────────────────────
    ax_rel_zeta.set_title("|Δζ| / ζ  — relative error  (log scale)",
                          fontsize=9, loc="left", pad=6, color="#444441")
    ax_rel_zeta.semilogy(epochs, rel_err_zeta, color=GREEN, lw=2,
                         marker="o", ms=ms)
    ax_rel_zeta.axhline(0.01, color=GRAY, lw=0.8, ls=":", alpha=0.7,
                        label="1 % target")
    ax_rel_zeta.set_xlabel("epoch")
    ax_rel_zeta.set_ylabel("|Δζ| / ζ")
    ax_rel_zeta.legend(fontsize=8, framealpha=0.5)
    ax_rel_zeta.set_xlim(left=0)

    fig.suptitle(
        f"Physical parameter convergence  |  "
        f"ω₀_true={true_w0:.4f} rad/s   ζ_true={true_zeta:.4f}\n"
        f"lr_param={cfg.lr_param}  |  snapshots at epochs {list(epochs)}",
        fontsize=9, y=0.995, color="#2C2C2A",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 2 saved  →  {out_path}")
    return fig


def _annotate_final(
    ax: plt.Axes,
    epochs: np.ndarray,
    vals: np.ndarray,
    true_val: float,
    unit: str,
) -> None:
    """Annotate the final snapshot point with its value and absolute error."""
    final_val = vals[-1]
    err_abs   = abs(final_val - true_val)
    suffix    = f" {unit}".rstrip()
    label     = f"final = {final_val:.4f}{suffix}  (err {err_abs:.4f}{suffix})"
    ax.annotate(
        label,
        xy=(epochs[-1], final_val),
        xytext=(-90, 14),
        textcoords="offset points",
        fontsize=7.5,
        color="#333333",
        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8),
    )


# =============================================================================
# FIGURE 3 — epoch-by-epoch trajectory snapshots
# =============================================================================

def make_epoch_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """
    2-column grid of panels, one per snapshot epoch.

    Each panel title shows:
      epoch  |  full-domain RMSE  |  ω₀_hat (true)  |  ζ_hat (true)

    This makes it easy to correlate trajectory quality with how well the
    physical parameters have converged at that point in training.
    """
    cfg: DamperConfig = bundle["cfg"]
    snaps             = bundle["snaps_pinn"]
    data              = bundle["data"]
    pred: Predictor   = bundle["pred"]

    t_obs_train  = data["t_obs_train"]
    y_obs_train  = data["y_obs_train"]
    t_obs_val    = data["t_obs_val"]
    y_obs_val    = data["y_obs_val"]
    t_col        = data["t_col"]
    t_plot_full  = data["t_plot_full"]
    y_true_full  = data["y_true_full"]

    epochs_sorted              = sorted(snaps.keys())
    snap_epochs, w0_vals, z_vals = extract_params(snaps)
    param_at = {ep: (w0_vals[i], z_vals[i]) for i, ep in enumerate(snap_epochs)}

    n_snap = len(epochs_sorted)
    n_cols = 2
    n_rows = (n_snap + 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, n_rows * 3.4),
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    y_lo, y_hi = -2.2, 1.6
    col_y = y_lo + 0.07 * (y_hi - y_lo)

    for idx, epoch in enumerate(epochs_sorted):
        ax = axes_flat[idx]
        style_ax(ax)

        # Predict from CPU snapshot without allocating a new model
        y_pred = pred.predict_from_state(snaps[epoch], t_plot_full)
        err    = Predictor.rmse(y_pred, y_true_full)
        w0_h, z_h = param_at.get(epoch, (float("nan"), float("nan")))

        # True solution
        ax.plot(t_plot_full, y_true_full,
                color=GRAY, lw=1.4, alpha=0.9)

        # PINN prediction
        ax.plot(t_plot_full, y_pred, color=TEAL, lw=2.0)

        # Extrapolation shading
        ax.axvspan(cfg.t_train, cfg.t_extrap, color=GRAY, alpha=0.10, zorder=0)
        ax.axvline(cfg.t_train, color=GRAY, lw=0.8, ls="--", alpha=0.5)

        # Collocation ticks along the bottom
        ax.scatter(t_col, np.full_like(t_col, col_y),
                   color=ORANGE, s=6, marker="|", alpha=0.6, zorder=3)

        # Observations
        ax.scatter(t_obs_train, y_obs_train,
                   color=BLUE, s=28, zorder=5, alpha=0.85)
        ax.scatter(t_obs_val, y_obs_val,
                   color=PURPLE, s=28, marker="D", zorder=6, alpha=0.75)

        # IC marker
        ax.scatter([0], [cfg.y0], color=PURPLE, s=130, marker="*", zorder=7)

        ax.set_title(
            f"epoch={epoch}   RMSE={err:.4f}   "
            f"ω₀_hat={w0_h:.3f} (true {cfg.omega_0:.3f})   "
            f"ζ_hat={z_h:.3f} (true {cfg.zeta:.3f})",
            fontsize=8.5, loc="left", pad=4, color="#444441",
        )
        ax.set_xlim(0, cfg.t_extrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y(t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("time  [s]", fontsize=8)

        # Region labels on first panel only
        if idx == 0:
            ax.text(cfg.t_train / 2, y_hi - 0.15, "training",
                    ha="center", va="top", fontsize=7, color=GRAY)
            ax.text(cfg.t_train + (cfg.t_extrap - cfg.t_train) / 2,
                    y_hi - 0.15, "extrapolation",
                    ha="center", va="top", fontsize=7, color=GRAY)

    # Hide unused grid cells
    for idx in range(n_snap, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend below all panels
    legend_elements = [
        Line2D([0], [0], color=GRAY,   lw=1.4,
               label="True solution  y(t)"),
        Line2D([0], [0], color=TEAL,   lw=2.0,
               label="PINN prediction  ŷ(t)"),
        Line2D([0], [0], color=BLUE,   lw=0, marker="o", markersize=5,
               label=f"Train obs  (N={len(t_obs_train)}, σ={cfg.sigma})"),
        Line2D([0], [0], color=PURPLE, lw=0, marker="D", markersize=5,
               label=f"Val obs  (N={len(t_obs_val)})"),
        Line2D([0], [0], color=PURPLE, lw=0, marker="*", markersize=9,
               label=f"IC  y(0)={cfg.y0},  y'(0)={cfg.dy0}"),
        Line2D([0], [0], color=ORANGE, lw=0, marker="|", markersize=7,
               label=f"Collocation pts  (N={cfg.n_col}, [0, {cfg.t_extrap}])"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"PINN trajectory evolution  |  device={bundle['device_str']}\n"
        f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
        f"{cfg.n_obs} obs  σ={cfg.sigma}  |  "
        f"{cfg.n_col} collocation pts  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"ω₀ and ζ learned simultaneously",
        fontsize=9.5, y=1.01, color="#2C2C2A",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 3 saved  →  {out_path}")
    return fig


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot PINN parameter-estimation results from training_results.pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results", type=str, default=DamperConfig.out_dir + "/training_results.pt",
        help="Path to training_results.pt written by train.py",
    )
    p.add_argument(
        "--out_dir", type=str, default=DamperConfig.out_dir,
        help="Directory for figures (defaults to same folder as results file)",
    )
    p.add_argument(
        "--no_show", action="store_true",
        help="Save only; do not call plt.show()",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(
            f"[plot.py] ERROR: results file not found: {results_path}\n"
            "  Run  python train.py  first."
        )
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load bundle ───────────────────────────────────────────────────────────
    print(f"\n  Loading  {results_path} …")
    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: DamperConfig = raw["config"]
    snaps = raw["snaps_pinn"]

    # Report what was recovered
    final_epoch = max(snaps.keys())
    w0_final   = snaps[final_epoch]["w0_hat"].item()
    zeta_final = snaps[final_epoch]["zeta_hat"].item()
    print(f"  Config   : m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
          f"ζ_true={cfg.zeta:.4f}  ω₀_true={cfg.omega_0:.4f}")
    print(f"  Snapshots: {sorted(snaps.keys())}")
    print(f"  Final ω₀_hat  = {w0_final:.4f}  (true {cfg.omega_0:.4f}  "
          f"err {abs(w0_final - cfg.omega_0):.4f})")
    print(f"  Final ζ_hat   = {zeta_final:.4f}  (true {cfg.zeta:.4f}  "
          f"err {abs(zeta_final - cfg.zeta):.4f})\n")

    # Single shared Predictor — predict_from_state swaps weights in-place
    pred = Predictor(cfg)

    bundle = {
        "cfg":        cfg,
        "device_str": raw["device_str"],
        "data":       raw["data"],
        "hist_pinn":  raw["hist_pinn"],
        "snaps_pinn": snaps,
        "y_pinn_full":raw["y_pinn_full"],
        "metrics":    raw["metrics"],
        "pred":       pred,
    }

    print("Generating Figure 1 — trajectory summary …")
    make_summary_figure(bundle, out_path=out_dir / "pinn_fig1_summary.png")

    print("Generating Figure 2 — parameter convergence …")
    make_param_convergence_figure(bundle, out_path=out_dir / "pinn_fig2_param_conv.png")

    print("Generating Figure 3 — epoch trajectory snapshots …")
    make_epoch_figure(bundle, out_path=out_dir / "pinn_fig3_pinn_epochs.png")

    print(f"\n  All figures written to  {out_dir}/")

    if not args.no_show:
        plt.show()

    print("  Done.\n")


if __name__ == "__main__":
    main()