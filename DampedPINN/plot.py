"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
five figures for the damped spring-mass PINN:

  Fig 1  fig1_loss.png              — Loss progression (all types, single plot)
  Fig 2  fig2_summary.png           — Four-panel comparison overview
  Fig 3  fig3_ml_epochs.png         — Standard-ML trajectory snapshots
  Fig 4  fig4_pinn_ext_epochs.png   — PINN (phys. ext.) trajectory snapshots
  Fig 5  fig5_pinn_blind_epochs.png — PINN (blind) trajectory snapshots

Three models are compared throughout:
  · Std ML            (data loss only)
  · PINN ext. physics (data + physics over full t-domain + IC)
  · PINN blind        (data + physics over training window only + IC)

Validation set design note
--------------------------
The validation set is a dense, noise-free grid over [0, t_train] used
purely for early-stopping.  It is NOT plotted anywhere: with cfg.n_val >> 1
the val points would simply retrace the true-solution curve, adding
visual clutter without information.  Only the sparse, noisy training
observations (the "measurements") appear as scatter.

Usage
-----
  python plot.py --results ./run_01/w0X_zetaY_training_results.pt
  python plot.py --input_dir ./run_01 --w0 2.0 --zeta 0.125
  python plot.py --results training_results.pt --out_dir ./figs --no_show
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

BLUE = "#378ADD"   # train observations / data loss
RED = "#E24B4A"   # standard ML
GREEN = "#1D9E75"   # PINN ext. physics
AMBER = "#D4820A"   # PINN blind
PURPLE = "#7F77DD"   # initial condition marker
GRAY = "#888780"   # true solution / neutral
LGRAY = "#D3D1C7"   # spine colour
BG = "#FAFAF8"   # figure background
PANEL = "#F1EFE8"   # axes background


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
# POINTWISE ODE RESIDUAL
# =============================================================================

def pointwise_residual(y: np.ndarray, t: np.ndarray,
                       cfg: DamperConfig) -> np.ndarray:
    dy = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(cfg.mass * d2y + cfg.damping * dy + cfg.stiffness * y)


# =============================================================================
# FIGURE 1  —  Loss progression (single plot, all models + loss types)
# =============================================================================

def make_loss_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Single-axes figure showing all loss types for all three models.

    Encoding:
      colour   = loss category  (same hue across models for easy cross-model tracking)
      linestyle = model          (solid = ML, dashed = ext_phys, dash-dot = blind)
      marker   = loss category

    ML only plots data + val (no physics or IC loss).
    """
    cfg = bundle["cfg"]
    h_ml = bundle["hist_ml"]
    h_ext = bundle["hist_pinn_ext_phys"]
    h_bl = bundle["hist_pinn_blind"]

    COLORS = {
        "loss_data":    "#2271B2",   # blue
        "loss_val":     "#E6533C",   # red
        "loss_physics": "#3DAA6A",   # green
        "loss_ic":      "#9B5EBF",   # purple
    }
    LABELS = {
        "loss_data":    "data",
        "loss_val":     "val",
        "loss_physics": "physics",
        "loss_ic":      "IC",
    }
    MARKERS = {
        "loss_data":    "o",
        "loss_val":     "s",
        "loss_physics": "^",
        "loss_ic":      "v",
    }
    # (linestyle, linewidth, alpha) per model
    MODEL_LS = {"ml": ("-", 2.0, 1.0), "ext": ("--",
                                               1.8, 0.9), "bl": ("-.", 1.6, 0.8)}
    MODEL_LABEL = {"ml": "ML", "ext": "ext_phys", "bl": "blind"}

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    # ML: only data + val
    for key in ("loss_data", "loss_val"):
        ls, lw, alpha = MODEL_LS["ml"]
        ax.semilogy(h_ml["epoch"], h_ml[key],
                    color=COLORS[key], lw=lw, ls=ls, alpha=alpha,
                    marker=MARKERS[key], markersize=3, markevery=5,
                    label=f"ML — {LABELS[key]}")

    # PINNs: all four loss types
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
    ax.set_title("All losses — ML (solid) | ext_phys (dashed) | blind (dash-dot)",
                 fontsize=10, loc="left", pad=6, color="#444441")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, framealpha=0.7, ncol=3)

    fig.suptitle(
        f"Damped oscillator — Training Loss Progression\n"
        f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
        f"ζ={cfg.zeta:.3f}  ω₀={cfg.omega_0:.3f}",
        fontsize=11, y=1.02, color="#2C2C2A", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 1 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 2  —  Four-panel summary
# =============================================================================
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  P1 (full width)  Full domain [0, t_extrap]: all model predictions  │
#  │                   + obs scatter + IC star + extrap shading          │
#  ├──────────────────────────────┬──────────────────┬───────────────────┤
#  │  P2  Training zoom [0,t_tr]  │  P3  IC detail   │  P4  ODE residual │
#  └──────────────────────────────┴──────────────────┴───────────────────┘

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    cfg = bundle["cfg"]
    data = bundle["data"]
    m = bundle["metrics"]
    device_str = bundle["device_str"]

    t_obs_train = data["t_obs_train"]
    y_obs_train = data["y_obs_train"]
    t_plot_train = data["t_plot_train"]
    t_plot_full = data["t_plot_full"]
    y_true_train = data["y_true_train"]
    y_true_full = data["y_true_full"]

    y_ml = bundle["y_ml_full"]
    y_ext = bundle["y_pinn_ext_phys_full"]
    y_blind = bundle["y_pinn_blind_full"]

    mask_train = t_plot_full <= cfg.t_train
    mask_extrap = t_plot_full > cfg.t_train

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.91, bottom=0.08,
    )
    ax_full = fig.add_subplot(gs[0, :])       # P1: full domain, all models
    ax_train = fig.add_subplot(gs[1, 0])       # P2: training window zoom
    ax_ic = fig.add_subplot(gs[1, 1])       # P3: IC enforcement detail
    ax_phys = fig.add_subplot(gs[1, 2])       # P4: ODE residual

    for ax in (ax_full, ax_train, ax_ic, ax_phys):
        style_ax(ax)

    # ── P1: Full domain overview ───────────────────────────────────────────────
    shade_extrap(ax_full, cfg)
    ax_full.plot(t_plot_full, y_true_full,
                 color=GRAY, lw=1.8, zorder=4, label="True  y(t)")
    ax_full.plot(t_plot_full, np.clip(y_ml, -3, 3),
                 color=RED,   lw=2.0, ls="--", zorder=3,
                 label=f"Std ML          train RMSE={m['rmse_ml_train']:.4f}  "
                 f"extrap={m['rmse_ml_extrap']:.4f}")
    ax_full.plot(t_plot_full, y_ext,
                 color=GREEN, lw=2.0, zorder=3,
                 label=f"PINN ext.       train RMSE={m['rmse_pinn_ext_phys_train']:.4f}  "
                 f"extrap={m['rmse_pinn_ext_phys_extrap']:.4f}")
    ax_full.plot(t_plot_full, y_blind,
                 color=AMBER, lw=2.0, ls="-.", zorder=3,
                 label=f"PINN blind      train RMSE={m['rmse_pinn_blind_train']:.4f}  "
                 f"extrap={m['rmse_pinn_blind_extrap']:.4f}")
    # Observations and IC — shown prominently
    ax_full.scatter(t_obs_train, y_obs_train,
                    color=BLUE, s=35, zorder=6, alpha=0.75,
                    label=f"Train obs  (N={len(t_obs_train)}, σ={cfg.sigma})")
    ax_full.scatter([0], [cfg.y0],
                    color=PURPLE, s=200, marker="*", zorder=8,
                    label=f"IC  y(0)={cfg.y0},  y'(0)={cfg.dy0}")
    ax_full.annotate("training", xy=(cfg.t_train * 0.5, 1),
                     xycoords=("data", "axes fraction"),
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
        f"Panel 1 — All predictions on full domain [0, {cfg.t_extrap} s]",
        fontsize=10, loc="left", pad=6, color="#444441"
    )
    ax_full.legend(fontsize=8, framealpha=0.6, ncol=2)

    # ── P2: Training window zoom ───────────────────────────────────────────────
    ax_train.plot(t_plot_full[mask_train], y_true_full[mask_train],
                  color=GRAY,  lw=1.8, label="True")
    ax_train.plot(t_plot_full[mask_train], y_ml[mask_train],
                  color=RED,   lw=2.0, ls="--",
                  label=f"ML    (RMSE={m['rmse_ml_train']:.4f})")
    ax_train.plot(t_plot_full[mask_train], y_ext[mask_train],
                  color=GREEN, lw=2.0,
                  label=f"Ext.  (RMSE={m['rmse_pinn_ext_phys_train']:.4f})")
    ax_train.plot(t_plot_full[mask_train], y_blind[mask_train],
                  color=AMBER, lw=2.0, ls="-.",
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

    # ── P3: Initial condition enforcement detail ───────────────────────────────
    # Zoom into t ≈ 0 so the IC star and early-time divergence are visible.
    ic_window = min(cfg.t_train * 0.25, 1.0)
    mask_ic = t_plot_full <= ic_window
    ax_ic.plot(t_plot_full[mask_ic], y_true_full[mask_ic],
               color=GRAY,  lw=1.8, label="True")
    ax_ic.plot(t_plot_full[mask_ic], y_ml[mask_ic],
               color=RED,   lw=2.0, ls="--",
               label=f"ML    ŷ(0)={m['y0_ml']:.3f}")
    ax_ic.plot(t_plot_full[mask_ic], y_ext[mask_ic],
               color=GREEN, lw=2.0,
               label=f"Ext.  ŷ(0)={m['y0_pinn_ext_phys']:.3f}")
    ax_ic.plot(t_plot_full[mask_ic], y_blind[mask_ic],
               color=AMBER, lw=2.0, ls="-.",
               label=f"Blind ŷ(0)={m['y0_pinn_blind']:.3f}")
    # Observations near t=0
    mask_obs_ic = t_obs_train <= ic_window
    ax_ic.scatter(t_obs_train[mask_obs_ic], y_obs_train[mask_obs_ic],
                  color=BLUE, s=50, zorder=5, alpha=0.8,
                  label=f"Obs near t=0  (N={mask_obs_ic.sum()})")
    # IC star — true value and each model's prediction at t=0
    ax_ic.scatter([0], [cfg.y0],
                  color=PURPLE, s=250, marker="*", zorder=9,
                  label=f"True IC  y(0)={cfg.y0}")
    # Horizontal reference at true y0
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
    r_ml = pointwise_residual(y_ml,    t_plot_full, cfg)
    r_ext = pointwise_residual(y_ext,   t_plot_full, cfg)
    r_blind = pointwise_residual(y_blind, t_plot_full, cfg)
    ax_phys.plot(t_plot_full, r_ml,    color=RED,   lw=1.8, ls="--",
                 label=f"ML          ({m['phys_ml']:.4f})")
    ax_phys.plot(t_plot_full, r_ext,   color=GREEN, lw=1.8,
                 label=f"PINN ext.   ({m['phys_pinn_ext_phys']:.4f})")
    ax_phys.plot(t_plot_full, r_blind, color=AMBER, lw=1.8, ls="-.",
                 label=f"PINN blind  ({m['phys_pinn_blind']:.4f})")
    ax_phys.axvline(cfg.t_train, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax_phys.set_xlabel("time  [s]", fontsize=9)
    ax_phys.set_ylabel("|m·y'' + c·y' + k·y|", fontsize=9)
    ax_phys.set_xlim(0, cfg.t_extrap)
    ax_phys.set_title("Panel 4 — Pointwise ODE residual  |r(t)|\n"
                      "(legend: mean residual over full domain)",
                      fontsize=9, loc="left", pad=6, color="#444441")
    ax_phys.legend(fontsize=7.5, framealpha=0.5)

    fig.suptitle(
        f"Std ML vs PINN (ext.) vs PINN (blind)  |  device={device_str}\n"
        f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
        f"ζ={cfg.zeta:.3f}  ω₀={cfg.omega_0:.3f}  ωd={cfg.omega_d:.3f} rad/s  |  "
        f"y(0): true={cfg.y0}  ML={m['y0_ml']:.3f}  "
        f"ext={m['y0_pinn_ext_phys']:.3f}  blind={m['y0_pinn_blind']:.3f}",
        fontsize=9, y=0.975, color="#2C2C2A",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 2 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURES 3–5 — epoch-by-epoch trajectory snapshots (one model per figure)
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
    clip_y:      bool = False,
    fig_num:     int = 3,
) -> plt.Figure:
    """
    2-column grid of panels, one per snapshot epoch.
    Shows true solution, model prediction, train obs, IC star,
    extrapolation shading, and RMSE per epoch.
    """
    t_obs_train = data["t_obs_train"]
    y_obs_train = data["y_obs_train"]
    t_plot_full = data["t_plot_full"]
    y_true_full = data["y_true_full"]

    epochs = sorted(snapshots.keys())
    n_snap = len(epochs)
    n_cols = 2
    n_rows = (n_snap + 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, n_rows * 3.2),
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    y_lo, y_hi = -2.2, 1.6

    for idx, epoch in enumerate(epochs):
        ax = axes_flat[idx]
        style_ax(ax)

        # FIX: predict_from_state does not exist; use load_state_dict + predict
        pred.load_state_dict(snapshots[epoch])
        y_pred = pred.predict(t_plot_full)
        if clip_y:
            y_pred = np.clip(y_pred, -3, 3)
        err = Predictor.rmse(y_pred, y_true_full)

        ax.plot(t_plot_full, y_true_full, color=GRAY,        lw=1.4, alpha=0.9,
                label="True")
        ax.plot(t_plot_full, y_pred,      color=model_color, lw=2.0,
                label=f"{model_label}  RMSE={err:.4f}")
        ax.axvspan(cfg.t_train, cfg.t_extrap, color=GRAY, alpha=0.10, zorder=0)
        ax.axvline(cfg.t_train,               color=GRAY,
                   lw=0.8, ls="--", alpha=0.5)

        # Train observations (noisy measurements only — val grid not plotted)
        ax.scatter(t_obs_train, y_obs_train,
                   color=BLUE, s=28, zorder=5, alpha=0.85,
                   label=f"Obs  (N={len(t_obs_train)}, σ={cfg.sigma})")
        # IC star
        ax.scatter([0], [cfg.y0],
                   color=PURPLE, s=180, marker="*", zorder=7,
                   label=f"True IC  y(0)={cfg.y0}")

        ax.set_title(f"epoch = {epoch}   |   RMSE = {err:.4f}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xlim(0, cfg.t_extrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y(t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("time  [s]", fontsize=8)

        if idx == 0:
            ax.text(cfg.t_train / 2,
                    y_hi - 0.12, "training",
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
        description="Generate plots from a training_results.pt bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results", type=str, default=None,
        help="Direct path to training_results.pt.  "
             "If omitted, reconstructed from --input_dir, --w0, --zeta.",
    )
    p.add_argument(
        "--input_dir", type=str, default=DamperConfig.out_dir,
        help="Directory containing the results file.",
    )
    p.add_argument(
        "--w0",   type=float, default=DamperConfig().omega_0,
        help="Natural frequency ω₀ [rad/s] for filename reconstruction.",
    )
    p.add_argument(
        "--zeta", type=float, default=DamperConfig().zeta,
        help="Damping ratio ζ for filename reconstruction.",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for figures (defaults to same folder as results file).",
    )
    p.add_argument(
        "--no_show", action="store_true",
        help="Save figures to disk only; do not call plt.show().",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.results is not None:
        results_path = Path(args.results)
    else:
        cfg_defaults = DamperConfig()
        filename = (f"w0{args.w0:.1e}_zeta{args.zeta:.1e}_"
                    f"{cfg_defaults.suffix_results_pt}")
        results_path = Path(args.input_dir) / filename

    if not results_path.exists():
        print(f"[plot.py] ERROR: results file not found: {results_path}\n"
              "  Run  python train.py  first, or supply --results <path>.")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading  {results_path} …")
    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: DamperConfig = raw["config"]
    data = raw["data"]

    print(f"  Config   : m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
          f"ζ={cfg.zeta:.4f}  ω₀={cfg.omega_0:.4f}  ωd={cfg.omega_d:.4f}")
    print(f"  Train obs: N={len(data['t_obs_train'])}  "
          f"Val grid : N={len(data['t_obs_val'])} "
          f"(dense, noiseless — early-stopping only, not plotted)")
    print(f"  Snapshots ML        : {sorted(raw['snaps_ml'].keys())}")
    print(
        f"  Snapshots PINN ext  : {sorted(raw['snaps_pinn_ext_phys'].keys())}")
    print(
        f"  Snapshots PINN blind: {sorted(raw['snaps_pinn_blind'].keys())}\n")

    bundle = {
        "cfg":                  cfg,
        "device_str":           raw["device_str"],
        "data":                 data,
        "hist_ml":              raw["hist_ml"],
        "hist_pinn_ext_phys":   raw["hist_pinn_ext_phys"],
        "hist_pinn_blind":      raw["hist_pinn_blind"],
        "snaps_ml":             raw["snaps_ml"],
        "snaps_pinn_ext_phys":  raw["snaps_pinn_ext_phys"],
        "snaps_pinn_blind":     raw["snaps_pinn_blind"],
        "y_ml_full":            raw["y_ml_full"],
        "y_pinn_ext_phys_full": raw["y_pinn_ext_phys_full"],
        "y_pinn_blind_full":    raw["y_pinn_blind_full"],
        "metrics":              raw["metrics"],
    }

    # Predictors — weights loaded per-snapshot inside make_epoch_figure
    pred_ml = Predictor(cfg)
    pred_ext = Predictor(cfg)
    pred_blind = Predictor(cfg)

    tag = f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}"

    print("Generating Figure 1 — loss progression …")
    make_loss_figure(bundle, out_dir / f"{tag}_fig1_loss.png")

    print("Generating Figure 2 — summary (full domain + training zoom + IC detail + ODE residual) …")
    make_summary_figure(bundle, out_dir / f"{tag}_fig2_summary.png")

    print("Generating Figure 3 — Standard ML epoch snapshots …")
    make_epoch_figure(
        snapshots=bundle["snaps_ml"],
        cfg=cfg,
        data=data,
        pred=pred_ml,
        model_color=RED,
        model_label="Std ML",
        fig_title=(
            f"Standard ML — trajectory evolution across epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            "Data loss only — no physics enforcement, no IC constraint"
        ),
        out_path=out_dir / f"{tag}_fig3_ml_epochs.png",
        clip_y=True,
        fig_num=3,
    )

    print("Generating Figure 4 — PINN (ext. physics) epoch snapshots …")
    make_epoch_figure(
        snapshots=bundle["snaps_pinn_ext_phys"],
        cfg=cfg,
        data=data,
        pred=pred_ext,
        model_color=GREEN,
        model_label="PINN ext.",
        fig_title=(
            f"PINN (ext. physics) — trajectory evolution across epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            f"Collocation extends to t_extrap={cfg.t_extrap}"
        ),
        out_path=out_dir / f"{tag}_fig4_pinn_ext_epochs.png",
        clip_y=False,
        fig_num=4,
    )

    print("Generating Figure 5 — PINN (blind) epoch snapshots …")
    make_epoch_figure(
        snapshots=bundle["snaps_pinn_blind"],
        cfg=cfg,
        data=data,
        pred=pred_blind,
        model_color=AMBER,
        model_label="PINN blind",
        fig_title=(
            f"PINN (blind) — trajectory evolution across epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            f"Collocation only in training window [0, {cfg.t_train}]"
        ),
        out_path=out_dir / f"{tag}_fig5_pinn_blind_epochs.png",
        clip_y=False,
        fig_num=5,
    )

    print(f"\n  All figures written to  {out_dir}/")
    if not args.no_show:
        plt.show()
    print("  Done.\n")


if __name__ == "__main__":
    main()
