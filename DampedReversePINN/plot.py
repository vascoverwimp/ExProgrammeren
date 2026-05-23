"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
five figures for the reverse damped-oscillator PINN (learning ω₀ and ζ):

  Fig 1  fig1_loss.png              — Loss progression (all types, single plot)
  Fig 2  fig2_summary.png           — Four-panel combined comparison overview
  Fig 3  fig3_param_conv.png        — ω₀ and ζ convergence (both models)
  Fig 4  fig4_pinn_ext_epochs.png   — PINN (phys. ext.) trajectory snapshots
  Fig 5  fig5_pinn_blind_epochs.png — PINN (blind)      trajectory snapshots

Two models are compared throughout:
  · PINN ext. physics (data + physics over full t-domain + IC)
  · PINN blind        (data + physics over training window only + IC)

Both models learn ω₀ and ζ as trainable log-scale parameters.
Parameter values are extracted from snapshot state_dicts via the
log_w0_hat / log_zeta_hat keys (10^x gives the physical value).

Usage
-----
  python plot.py
  python plot.py --w0 2.0 --zeta 0.125
  python plot.py --results path/to/training_results.pt --out_dir ./figs --no_show
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

TEAL   = "#1D9E75"   # PINN ext. physics
AMBER  = "#D4820A"   # PINN blind
BLUE   = "#378ADD"   # train observations
PURPLE = "#7F77DD"   # IC marker
RED    = "#E24B4A"   # ω₀ parameter traces
GREEN  = "#3AAA5E"   # ζ parameter traces
GRAY   = "#888780"   # true solution / neutral
LGRAY  = "#D3D1C7"   # spine colour
BG     = "#FAFAF8"   # figure background
PANEL  = "#F1EFE8"   # axes background


# =============================================================================
# SHARED HELPERS
# =============================================================================

def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def shade_extrap(ax: plt.Axes, cfg: DamperConfig) -> None:
    ax.axvspan(cfg.t_train, cfg.t_extrap, color=GRAY, alpha=0.12)
    ax.axvline(cfg.t_train, color=GRAY, lw=0.8, ls="--", alpha=0.6)


def pointwise_residual(y: np.ndarray, t: np.ndarray,
                        cfg: DamperConfig) -> np.ndarray:
    dy  = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(cfg.mass * d2y + cfg.damping * dy + cfg.stiffness * y)


# =============================================================================
# PARAMETER EXTRACTION FROM SNAPSHOTS
# =============================================================================

def extract_params(snapshots: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull ω₀_hat and ζ_hat from every snapshot state_dict.

    The model stores log10-scale parameters (log_w0_hat, log_zeta_hat)
    as nn.Parameters, so the physical values are 10^(stored scalar).

    Returns
    -------
    epochs    : sorted epoch indices (int array)
    w0_vals   : ω₀_hat at each snapshot (linear scale)
    zeta_vals : ζ_hat  at each snapshot (linear scale)
    """
    epochs_sorted = sorted(snapshots.keys())
    w0_vals   = np.array([10 ** snapshots[e]["log_w0_hat"].item()
                          for e in epochs_sorted])
    zeta_vals = np.array([10 ** snapshots[e]["log_zeta_hat"].item()
                          for e in epochs_sorted])
    return np.array(epochs_sorted), w0_vals, zeta_vals


# =============================================================================
# FIGURE 1  —  Loss progression (single plot, both PINNs, all loss types)
# =============================================================================

def make_loss_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Single-axes figure.

    Encoding:
      colour    = loss category  (shared across models)
      linestyle = model          (solid = ext_phys, dashed = blind)
      marker    = loss category
    """
    cfg   = bundle["cfg"]
    h_ext = bundle["hist_pinn_ext_phys"]
    h_bl  = bundle["hist_pinn_blind"]

    COLORS = {
        "loss_data":    "#2271B2",
        "loss_val":     "#E6533C",
        "loss_physics": "#3DAA6A",
        "loss_ic":      "#9B5EBF",
    }
    LABELS  = {"loss_data": "data", "loss_val": "val",
               "loss_physics": "physics", "loss_ic": "IC"}
    MARKERS = {"loss_data": "o", "loss_val": "s",
               "loss_physics": "^", "loss_ic": "v"}
    MODEL_LS    = {"ext": ("-", 2.0, 1.0), "bl": ("--", 1.6, 0.8)}
    MODEL_LABEL = {"ext": "ext_phys", "bl": "blind"}

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    for hist_key, model_key in (("hist_pinn_ext_phys", "ext"),
                                ("hist_pinn_blind",    "bl")):
        h = bundle[hist_key]
        ls, lw, alpha = MODEL_LS[model_key]
        mlbl = MODEL_LABEL[model_key]
        for key in ("loss_data", "loss_val", "loss_physics", "loss_ic"):
            ax.semilogy(h["epoch"], h[key],
                        color=COLORS[key], lw=lw, ls=ls, alpha=alpha,
                        marker=MARKERS[key], markersize=3, markevery=5,
                        label=f"{mlbl} — {LABELS[key]}")

    for ep in cfg.snapshot_epochs[:-1]:
        ax.axvline(ep, color=GRAY, lw=0.6, ls=":", alpha=0.35, zorder=0)

    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("loss  (log scale)", fontsize=10)
    ax.set_title("All losses — ext_phys (solid) | blind (dashed)",
                 fontsize=10, loc="left", pad=6, color="#444441")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, framealpha=0.7, ncol=2)

    fig.suptitle(
        f"Damped oscillator (Reverse) — Training Loss Progression\n"
        f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
        f"ζ_true={cfg.zeta:.3f}  ω₀_true={cfg.omega_0:.3f}",
        fontsize=11, y=1.02, color="#2C2C2A", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 1 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 2  —  Four-panel combined summary (both models)
# =============================================================================
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  P1 (full width)  Full domain: both PINNs + obs + IC + extrap shading│
#  ├─────────────────────────┬───────────────────┬────────────────────────┤
#  │  P2  Training zoom      │  P3  IC detail    │  P4  ODE residual      │
#  └─────────────────────────┴───────────────────┴────────────────────────┘

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    cfg        = bundle["cfg"]
    data       = bundle["data"]
    m          = bundle["metrics"]
    device_str = bundle["device_str"]

    t_obs_train  = data["t_obs_train"]
    y_obs_train  = data["y_obs_train"]
    t_plot_train = data["t_plot_train"]
    t_plot_full  = data["t_plot_full"]
    y_true_train = data["y_true_train"]
    y_true_full  = data["y_true_full"]

    y_ext   = bundle["y_pinn_ext_phys_full"]
    y_blind = bundle["y_pinn_blind_full"]

    mask_train = t_plot_full <= cfg.t_train

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.91, bottom=0.08,
    )
    ax_full  = fig.add_subplot(gs[0, :])
    ax_train = fig.add_subplot(gs[1, 0])
    ax_ic    = fig.add_subplot(gs[1, 1])
    ax_phys  = fig.add_subplot(gs[1, 2])

    for ax in (ax_full, ax_train, ax_ic, ax_phys):
        style_ax(ax)

    # ── P1: Full domain overview ───────────────────────────────────────────────
    shade_extrap(ax_full, cfg)
    ax_full.plot(t_plot_full, y_true_full,
                 color=GRAY,  lw=1.8, zorder=4, label="True  y(t)")
    ax_full.plot(t_plot_full, y_ext,
                 color=TEAL,  lw=2.0, zorder=3,
                 label=f"PINN ext.   train RMSE={m['rmse_pinn_ext_phys_train']:.4f}  "
                       f"extrap={m['rmse_pinn_ext_phys_extrap']:.4f}")
    ax_full.plot(t_plot_full, y_blind,
                 color=AMBER, lw=2.0, ls="--", zorder=3,
                 label=f"PINN blind  train RMSE={m['rmse_pinn_blind_train']:.4f}  "
                       f"extrap={m['rmse_pinn_blind_extrap']:.4f}")
    ax_full.scatter(t_obs_train, y_obs_train,
                    color=BLUE,   s=35, zorder=6, alpha=0.75,
                    label=f"Train obs  (N={len(t_obs_train)}, σ={cfg.sigma})")
    ax_full.scatter([0], [cfg.y0],
                    color=PURPLE, s=200, marker="*", zorder=8,
                    label=f"IC  y(0)={cfg.y0},  y'(0)={cfg.dy0}  [enforced via L_ic]")
    ax_full.annotate("training",
                     xy=(cfg.t_train * 0.5, 1), xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=8, color=GRAY,
                     xytext=(0, -6), textcoords="offset points")
    ax_full.annotate("extrapolation",
                     xy=((cfg.t_train + cfg.t_extrap) * 0.5, 1),
                     xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=8, color=GRAY,
                     xytext=(0, -6), textcoords="offset points")
    ax_full.set_xlim(0, cfg.t_extrap)
    ax_full.set_ylim(-2.5, 2.5)
    ax_full.set_xlabel("time  [s]", fontsize=9)
    ax_full.set_ylabel("displacement  y(t)", fontsize=9)
    ax_full.set_title(
        f"Panel 1 — Both PINNs on full domain [0, {cfg.t_extrap} s]  |  "
        f"True ω₀={cfg.omega_0:.4f}  ζ={cfg.zeta:.4f}  |  "
        f"ω₀_hat: ext={m.get('w0_hat_ext_phys', float('nan')):.4f}  "
        f"blind={m.get('w0_hat_blind', float('nan')):.4f}  |  "
        f"ζ_hat:  ext={m.get('zeta_hat_ext_phys', float('nan')):.4f}  "
        f"blind={m.get('zeta_hat_blind', float('nan')):.4f}",
        fontsize=9, loc="left", pad=6, color="#444441"
    )
    ax_full.legend(fontsize=8, framealpha=0.6, ncol=2)

    # ── P2: Training window zoom ───────────────────────────────────────────────
    ax_train.plot(t_plot_full[mask_train], y_true_full[mask_train],
                  color=GRAY,  lw=1.8, label="True")
    ax_train.plot(t_plot_full[mask_train], y_ext[mask_train],
                  color=TEAL,  lw=2.0,
                  label=f"Ext.  (RMSE={m['rmse_pinn_ext_phys_train']:.4f})")
    ax_train.plot(t_plot_full[mask_train], y_blind[mask_train],
                  color=AMBER, lw=2.0, ls="--",
                  label=f"Blind (RMSE={m['rmse_pinn_blind_train']:.4f})")
    ax_train.scatter(t_obs_train, y_obs_train,
                     color=BLUE, s=28, zorder=5, alpha=0.65,
                     label=f"Obs  (N={len(t_obs_train)})")
    ax_train.scatter([0], [cfg.y0],
                     color=PURPLE, s=130, marker="*", zorder=7)
    ax_train.set_xlim(0, cfg.t_train)
    ax_train.set_xlabel("time  [s]", fontsize=9)
    ax_train.set_ylabel("displacement  y(t)", fontsize=9)
    ax_train.set_title(f"Panel 2 — Training window  [0, {cfg.t_train} s]",
                       fontsize=10, loc="left", pad=6, color="#444441")
    ax_train.legend(fontsize=7.5, framealpha=0.5)

    # ── P3: IC enforcement detail ──────────────────────────────────────────────
    ic_window = min(cfg.t_train * 0.25, 1.0)
    mask_ic   = t_plot_full <= ic_window
    ax_ic.plot(t_plot_full[mask_ic], y_true_full[mask_ic],
               color=GRAY,  lw=1.8, label="True")
    ax_ic.plot(t_plot_full[mask_ic], y_ext[mask_ic],
               color=TEAL,  lw=2.0,
               label=f"Ext.  ŷ(0)={m['y0_pinn_ext_phys']:.3f}")
    ax_ic.plot(t_plot_full[mask_ic], y_blind[mask_ic],
               color=AMBER, lw=2.0, ls="--",
               label=f"Blind ŷ(0)={m['y0_pinn_blind']:.3f}")
    mask_obs_ic = t_obs_train <= ic_window
    ax_ic.scatter(t_obs_train[mask_obs_ic], y_obs_train[mask_obs_ic],
                  color=BLUE, s=50, zorder=5, alpha=0.8,
                  label=f"Obs near t=0  (N={mask_obs_ic.sum()})")
    ax_ic.scatter([0], [cfg.y0],
                  color=PURPLE, s=250, marker="*", zorder=9,
                  label=f"True IC  y(0)={cfg.y0}")
    ax_ic.axhline(cfg.y0, color=PURPLE, lw=0.8, ls=":", alpha=0.5)
    ax_ic.axvline(0,       color=PURPLE, lw=0.8, ls=":", alpha=0.3)
    ax_ic.set_xlim(-0.02 * ic_window, ic_window)
    ax_ic.set_xlabel("time  [s]", fontsize=9)
    ax_ic.set_ylabel("displacement  y(t)", fontsize=9)
    ax_ic.set_title(
        f"Panel 3 — IC enforcement detail  [0, {ic_window:.2f} s]\n"
        f"True y(0)={cfg.y0}  y'(0)={cfg.dy0}",
        fontsize=9, loc="left", pad=6, color="#444441"
    )
    ax_ic.legend(fontsize=7.5, framealpha=0.5)

    # ── P4: ODE residual ──────────────────────────────────────────────────────
    shade_extrap(ax_phys, cfg)
    r_ext   = pointwise_residual(y_ext,   t_plot_full, cfg)
    r_blind = pointwise_residual(y_blind, t_plot_full, cfg)
    ax_phys.plot(t_plot_full, r_ext,
                 color=TEAL,  lw=1.8,
                 label=f"PINN ext.   ({m['phys_pinn_ext_phys']:.4f})")
    ax_phys.plot(t_plot_full, r_blind,
                 color=AMBER, lw=1.8, ls="--",
                 label=f"PINN blind  ({m['phys_pinn_blind']:.4f})")
    ax_phys.set_xlabel("time  [s]", fontsize=9)
    ax_phys.set_ylabel("|m·y'' + c·y' + k·y|", fontsize=9)
    ax_phys.set_xlim(0, cfg.t_extrap)
    ax_phys.set_title("Panel 4 — Pointwise ODE residual  |r(t)|\n"
                      "(legend: mean residual over full domain)",
                      fontsize=9, loc="left", pad=6, color="#444441")
    ax_phys.legend(fontsize=7.5, framealpha=0.5)

    fig.suptitle(
        f"Damped oscillator (Reverse) — PINN ext. vs PINN blind  |  "
        f"device={device_str}\n"
        f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
        f"ζ_true={cfg.zeta:.3f}  ω₀_true={cfg.omega_0:.3f} rad/s  |  "
        f"IC: y(0)={cfg.y0}  y'(0)={cfg.dy0}  |  "
        f"ini guess: ω₀={cfg.ini_guess_w0:.3f}  ζ={cfg.ini_guess_zeta:.3f}",
        fontsize=9, y=0.975, color="#2C2C2A",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 2 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 3  —  Parameter convergence (both models on same axes)
# =============================================================================
#
#  Row 0: ω₀_hat over epochs (both models + true dashed + ±5% band)
#  Row 1: ζ_hat  over epochs (both models + true dashed + ±5% band)
#  Row 2: |Δω₀|/ω₀ log scale
#  Row 3: |Δζ|/ζ   log scale

def make_param_convergence_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Both models shown together on each axis so convergence speed and
    final accuracy can be compared directly.
    """
    cfg = bundle["cfg"]

    epochs_ext,  w0_ext,  z_ext  = extract_params(bundle["snaps_pinn_ext_phys"])
    epochs_blind, w0_blind, z_blind = extract_params(bundle["snaps_pinn_blind"])

    true_w0   = cfg.omega_0
    true_zeta = cfg.zeta

    rel_w0_ext    = np.abs(w0_ext   - true_w0)   / true_w0
    rel_w0_blind  = np.abs(w0_blind - true_w0)   / true_w0
    rel_z_ext     = np.abs(z_ext    - true_zeta) / true_zeta
    rel_z_blind   = np.abs(z_blind  - true_zeta) / true_zeta

    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [1.6, 1]})
    fig.patch.set_facecolor(BG)
    ax_w0, ax_zeta, ax_rel_w0, ax_rel_zeta = axes.flatten()

    for ax in axes.flatten():
        style_ax(ax)

    ms = 7

    # Shared epoch range for fill_between
    all_epochs = np.concatenate([epochs_ext, epochs_blind])
    e_lo, e_hi = all_epochs.min(), all_epochs.max()

    # ── Top-left: ω₀_hat ──────────────────────────────────────────────────────
    ax_w0.set_title("ω₀ estimation — ext_phys vs blind",
                    fontsize=10, loc="left", pad=6, color="#444441")
    ax_w0.axhline(true_w0, color=GRAY, lw=1.2, ls="--",
                  label=f"True ω₀ = {true_w0:.4f} rad/s")
    ax_w0.fill_between([e_lo, e_hi], true_w0 * 0.95, true_w0 * 1.05,
                        color=GRAY, alpha=0.08, label="±5 % band")
    ax_w0.plot(epochs_ext,   w0_ext,   color=TEAL,  lw=2, marker="o", ms=ms,
               label=f"ext_phys  (final={w0_ext[-1]:.4f})")
    ax_w0.plot(epochs_blind, w0_blind, color=AMBER, lw=2, marker="s", ms=ms,
               ls="--", label=f"blind     (final={w0_blind[-1]:.4f})")
    ax_w0.set_xlabel("epoch")
    ax_w0.set_ylabel("ω₀_hat  [rad/s]")
    ax_w0.legend(fontsize=8, framealpha=0.6)
    ax_w0.set_xlim(left=0)

    # ── Top-right: ζ_hat ──────────────────────────────────────────────────────
    ax_zeta.set_title("ζ estimation — ext_phys vs blind",
                      fontsize=10, loc="left", pad=6, color="#444441")
    ax_zeta.axhline(true_zeta, color=GRAY, lw=1.2, ls="--",
                    label=f"True ζ = {true_zeta:.4f}")
    ax_zeta.fill_between([e_lo, e_hi], true_zeta * 0.95, true_zeta * 1.05,
                          color=GRAY, alpha=0.08, label="±5 % band")
    ax_zeta.plot(epochs_ext,   z_ext,   color=TEAL,  lw=2, marker="o", ms=ms,
                 label=f"ext_phys  (final={z_ext[-1]:.4f})")
    ax_zeta.plot(epochs_blind, z_blind, color=AMBER, lw=2, marker="s", ms=ms,
                 ls="--", label=f"blind     (final={z_blind[-1]:.4f})")
    ax_zeta.set_xlabel("epoch")
    ax_zeta.set_ylabel("ζ_hat  [—]")
    ax_zeta.legend(fontsize=8, framealpha=0.6)
    ax_zeta.set_xlim(left=0)

    # ── Bottom-left: |Δω₀|/ω₀ log ─────────────────────────────────────────────
    ax_rel_w0.set_title("|Δω₀| / ω₀  — relative error  (log scale)",
                        fontsize=9, loc="left", pad=6, color="#444441")
    ax_rel_w0.semilogy(epochs_ext,   rel_w0_ext,   color=TEAL,  lw=2,
                       marker="o", ms=ms, label="ext_phys")
    ax_rel_w0.semilogy(epochs_blind, rel_w0_blind, color=AMBER, lw=2,
                       marker="s", ms=ms, ls="--", label="blind")
    ax_rel_w0.axhline(0.01, color=GRAY, lw=0.8, ls=":", alpha=0.7,
                      label="1 % target")
    ax_rel_w0.set_xlabel("epoch")
    ax_rel_w0.set_ylabel("|Δω₀| / ω₀")
    ax_rel_w0.legend(fontsize=8, framealpha=0.6)
    ax_rel_w0.set_xlim(left=0)

    # ── Bottom-right: |Δζ|/ζ log ──────────────────────────────────────────────
    ax_rel_zeta.set_title("|Δζ| / ζ  — relative error  (log scale)",
                          fontsize=9, loc="left", pad=6, color="#444441")
    ax_rel_zeta.semilogy(epochs_ext,   rel_z_ext,   color=TEAL,  lw=2,
                         marker="o", ms=ms, label="ext_phys")
    ax_rel_zeta.semilogy(epochs_blind, rel_z_blind, color=AMBER, lw=2,
                         marker="s", ms=ms, ls="--", label="blind")
    ax_rel_zeta.axhline(0.01, color=GRAY, lw=0.8, ls=":", alpha=0.7,
                        label="1 % target")
    ax_rel_zeta.set_xlabel("epoch")
    ax_rel_zeta.set_ylabel("|Δζ| / ζ")
    ax_rel_zeta.legend(fontsize=8, framealpha=0.6)
    ax_rel_zeta.set_xlim(left=0)

    fig.suptitle(
        f"Physical parameter convergence  |  "
        f"ω₀_true={true_w0:.4f} rad/s   ζ_true={true_zeta:.4f}\n"
        f"Initial guess: ω₀={bundle['cfg'].ini_guess_w0:.4f}  "
        f"ζ={bundle['cfg'].ini_guess_zeta:.4f}  |  "
        f"snapshots at epochs {list(epochs_ext)}",
        fontsize=9, y=0.998, color="#2C2C2A",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 3 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURES 4–5  —  Epoch-by-epoch trajectory snapshots (one model per figure)
# =============================================================================

def make_epoch_figure(
    snapshots:   dict,
    cfg:         DamperConfig,
    data:        dict,
    pred:        Predictor,
    model_color: str,
    model_label: str,
    fig_title:   str,
    out_path:    Path,
    fig_num:     int = 4,
) -> plt.Figure:
    """
    2-column grid of panels, one per snapshot epoch.

    Each panel title shows epoch, RMSE, ω₀_hat, and ζ_hat so trajectory
    quality can be correlated with parameter convergence at a glance.
    """
    t_obs_train = data["t_obs_train"]
    y_obs_train = data["y_obs_train"]
    t_plot_full = data["t_plot_full"]
    y_true_full = data["y_true_full"]

    # FIX: extract_params uses log_w0_hat / log_zeta_hat → 10^x
    snap_epochs, w0_vals, z_vals = extract_params(snapshots)
    param_at = {ep: (w0_vals[i], z_vals[i])
                for i, ep in enumerate(snap_epochs)}

    epochs   = sorted(snapshots.keys())
    n_snap   = len(epochs)
    n_cols   = 2
    n_rows   = (n_snap + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 3.4),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    y_lo, y_hi = -2.2, 1.6

    for idx, epoch in enumerate(epochs):
        ax = axes_flat[idx]
        style_ax(ax)

        # FIX: predict_from_state does not exist; use load_state_dict + predict
        pred.load_state_dict(snapshots[epoch])
        y_pred = pred.predict(t_plot_full)
        err    = Predictor.rmse(y_pred, y_true_full)
        w0_h, z_h = param_at.get(epoch, (float("nan"), float("nan")))

        ax.plot(t_plot_full, y_true_full, color=GRAY,        lw=1.4, alpha=0.9)
        ax.plot(t_plot_full, y_pred,      color=model_color, lw=2.0)
        ax.axvspan(cfg.t_train, cfg.t_extrap, color=GRAY, alpha=0.10, zorder=0)
        ax.axvline(cfg.t_train,               color=GRAY, lw=0.8, ls="--", alpha=0.5)

        ax.scatter(t_obs_train, y_obs_train,
                   color=BLUE,   s=28, zorder=5, alpha=0.85)
        ax.scatter([0], [cfg.y0],
                   color=PURPLE, s=180, marker="*", zorder=7)

        ax.set_title(
            f"epoch={epoch}   RMSE={err:.4f}   "
            f"ω₀_hat={w0_h:.4f} (true {cfg.omega_0:.4f})   "
            f"ζ_hat={z_h:.4f} (true {cfg.zeta:.4f})",
            fontsize=8.5, loc="left", pad=4, color="#444441",
        )
        ax.set_xlim(0, cfg.t_extrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y(t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("time  [s]", fontsize=8)

        if idx == 0:
            ax.text(cfg.t_train / 2, y_hi - 0.12, "training",
                    ha="center", va="top", fontsize=7, color=GRAY)
            ax.text(cfg.t_train + (cfg.t_extrap - cfg.t_train) / 2,
                    y_hi - 0.12, "extrapolation",
                    ha="center", va="top", fontsize=7, color=GRAY)

    for idx in range(n_snap, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=GRAY,        lw=1.4,
               label="True solution  y(t)"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=f"{model_label} prediction  ŷ(t)"),
        Line2D([0], [0], color=BLUE,   lw=0, marker="o", markersize=5,
               label=f"Train obs  (N={len(t_obs_train)}, σ={cfg.sigma})"),
        Line2D([0], [0], color=PURPLE, lw=0, marker="*", markersize=9,
               label=f"IC  y(0)={cfg.y0},  y'(0)={cfg.dy0}"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(fig_title, fontsize=10, y=1.01, color="#2C2C2A")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure {fig_num} saved  →  {out_path}")
    return fig


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot reverse-PINN damped-oscillator results from training_results.pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results",   type=str, default="",
                   help="Path to training_results.pt. "
                        "If omitted, reconstructed from --input_dir, --w0, --zeta.")
    p.add_argument("--input_dir", type=str, default=DamperConfig.out_dir,
                   help="Directory containing the results file.")
    p.add_argument("--w0",   type=float, default=DamperConfig().omega_0,
                   help="ω₀ [rad/s] for filename reconstruction.")
    p.add_argument("--zeta", type=float, default=DamperConfig().zeta,
                   help="ζ for filename reconstruction.")
    p.add_argument("--out_dir",  type=str, default=DamperConfig.out_dir,
                   help="Output directory for figures.")
    p.add_argument("--no_show",  action="store_true",
                   help="Save only; do not call plt.show().")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.results:
        results_path = Path(args.results)
    else:
        results_path = (Path(args.input_dir) /
                        f"w0{args.w0:.1e}_zeta{args.zeta:.1e}_"
                        f"{DamperConfig.suffix_results_pt}")

    if not results_path.exists():
        print(f"[plot.py] ERROR: results file not found: {results_path}\n"
              "  Run  python train.py  first.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading  {results_path} …")
    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: DamperConfig = raw["config"]
    snaps_ext   = raw["snaps_pinn_ext_phys"]
    snaps_blind = raw["snaps_pinn_blind"]

    # Report recovered parameters
    def _final_params(snaps):
        ep = max(snaps.keys())
        w0   = 10 ** snaps[ep]["log_w0_hat"].item()
        zeta = 10 ** snaps[ep]["log_zeta_hat"].item()
        return w0, zeta

    w0_ext,   z_ext   = _final_params(snaps_ext)
    w0_blind, z_blind = _final_params(snaps_blind)

    print(f"  Config  : m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
          f"ζ_true={cfg.zeta:.4f}  ω₀_true={cfg.omega_0:.4f}")
    print(f"  Ini guess: ω₀={cfg.ini_guess_w0:.4f}  ζ={cfg.ini_guess_zeta:.4f}")
    print(f"  Snapshots ext  : {sorted(snaps_ext.keys())}")
    print(f"  Snapshots blind: {sorted(snaps_blind.keys())}")
    print(f"  Final ω₀_hat : ext={w0_ext:.4f}  blind={w0_blind:.4f}  "
          f"(true {cfg.omega_0:.4f})")
    print(f"  Final ζ_hat  : ext={z_ext:.4f}  blind={z_blind:.4f}  "
          f"(true {cfg.zeta:.4f})\n")

    # Predictors — weights loaded per-snapshot inside make_epoch_figure
    pred_ext   = Predictor(cfg)
    pred_blind = Predictor(cfg)

    # Augment metrics with final parameter values so the summary panel can
    # show them without re-extracting from snapshots.
    metrics = dict(raw["metrics"])
    metrics.setdefault("w0_hat_ext_phys",   w0_ext)
    metrics.setdefault("w0_hat_blind",       w0_blind)
    metrics.setdefault("zeta_hat_ext_phys",  z_ext)
    metrics.setdefault("zeta_hat_blind",     z_blind)

    bundle = {
        "cfg":                  cfg,
        "device_str":           raw["device_str"],
        "data":                 raw["data"],
        "hist_pinn_ext_phys":   raw["hist_pinn_ext_phys"],
        "hist_pinn_blind":      raw["hist_pinn_blind"],
        "snaps_pinn_ext_phys":  snaps_ext,
        "snaps_pinn_blind":     snaps_blind,
        "y_pinn_ext_phys_full": raw["y_pinn_ext_phys_full"],
        "y_pinn_blind_full":    raw["y_pinn_blind_full"],
        "metrics":              metrics,
        "pred_ext_phys":        pred_ext,
        "pred_blind":           pred_blind,
    }

    tag = f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}"

    print("Generating Figure 1 — loss progression …")
    make_loss_figure(bundle, out_dir / f"{tag}_fig1_loss.png")

    print("Generating Figure 2 — combined summary …")
    make_summary_figure(bundle, out_dir / f"{tag}_fig2_summary.png")

    print("Generating Figure 3 — parameter convergence (both models) …")
    make_param_convergence_figure(bundle, out_dir / f"{tag}_fig3_param_conv.png")

    print("Generating Figure 4 — PINN (ext. physics) epoch snapshots …")
    make_epoch_figure(
        snapshots   = snaps_ext,
        cfg         = cfg,
        data        = raw["data"],
        pred        = pred_ext,
        model_color = TEAL,
        model_label = "PINN ext.",
        fig_title   = (
            f"PINN (ext. physics) — trajectory evolution across epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            f"Collocation extends to t_extrap={cfg.t_extrap}  |  "
            f"learning ω₀ and ζ simultaneously"
        ),
        out_path    = out_dir / f"{tag}_fig4_pinn_ext_epochs.png",
        fig_num     = 4,
    )

    print("Generating Figure 5 — PINN (blind) epoch snapshots …")
    make_epoch_figure(
        snapshots   = snaps_blind,
        cfg         = cfg,
        data        = raw["data"],
        pred        = pred_blind,
        model_color = AMBER,
        model_label = "PINN blind",
        fig_title   = (
            f"PINN (blind) — trajectory evolution across epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            f"Collocation only in training window [0, {cfg.t_train}]  |  "
            f"learning ω₀ and ζ simultaneously"
        ),
        out_path    = out_dir / f"{tag}_fig5_pinn_blind_epochs.png",
        fig_num     = 5,
    )

    print(f"\n  All figures written to  {out_dir}/")
    if not args.no_show:
        plt.show()
    print("  Done.\n")


if __name__ == "__main__":
    main()