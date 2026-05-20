"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
four publication-quality figures:

  Fig 1  fig1_summary.png           — five-panel comparison overview
                                       (ML | PINN-ext | PINN-blind)
  Fig 2  fig2_pinn_ext_epochs.png   — PINN (phys. ext.) trajectory snapshots
  Fig 3  fig3_pinn_blind_epochs.png — PINN (blind)      trajectory snapshots
  Fig 4  fig4_ml_epochs.png         — Standard-ML        trajectory snapshots

Validation set design note
--------------------------
The validation set is a dense, noise-free grid over [0, t_train] used
purely for early-stopping signal.  It is NOT plotted anywhere: with
cfg.n_val >> 1 the val points would simply retrace the true-solution
curve, adding visual clutter without any information.  Only the sparse,
noisy training observations (the actual "measurements") appear as scatter.

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

BLUE      = "#378ADD"   # train observations
RED       = "#E24B4A"   # standard ML
GREEN     = "#1D9E75"   # PINN ext. physics
AMBER     = "#D4820A"   # PINN blind
ORANGE    = "#EF9F27"   # collocation ticks — ext. (reaches t_extrap)
ORANGE_BL = "#B0B0B0"   # collocation ticks — blind (stays in training window)
PURPLE    = "#7F77DD"   # initial condition marker
GRAY      = "#888780"   # true solution / neutral
LGRAY     = "#D3D1C7"   # spine colour
BG        = "#FAFAF8"   # figure background
PANEL     = "#F1EFE8"   # axes background


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

def pointwise_residual(y: np.ndarray, t: np.ndarray, cfg: DamperConfig) -> np.ndarray:
    dy  = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(cfg.mass * d2y + cfg.damping * dy + cfg.stiffness * y)


# =============================================================================
# FIGURE 1 — five-panel summary
# =============================================================================
#
#  ┌───────────────────────────────────────────────────────────┐
#  │  P1  Training data — sparse noisy obs + collocation sets  │
#  ├────────────────────────┬──────────────┬───────────────────┤
#  │  P2  Fit [0, t_train]  │  P3  Losses  │  P4  Extrap.      │
#  ├────────────────────────┴──────────────┴───────────────────┤
#  │  P5  ODE residual |r(t)|  (all three models)              │
#  └───────────────────────────────────────────────────────────┘

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    cfg: DamperConfig = bundle["cfg"]
    data              = bundle["data"]
    hist_ml           = bundle["hist_ml"]
    hist_ext          = bundle["hist_pinn_ext_phys"]
    hist_blind        = bundle["hist_pinn_blind"]
    y_ml              = bundle["y_ml_full"]
    y_ext             = bundle["y_pinn_ext_phys_full"]
    y_blind           = bundle["y_pinn_blind_full"]
    m                 = bundle["metrics"]
    device_str        = bundle["device_str"]

    t_obs_train  = data["t_obs_train"]
    y_obs_train  = data["y_obs_train"]
    t_plot_train = data["t_plot_train"]
    t_plot_full  = data["t_plot_full"]
    y_true_train = data["y_true_train"]
    y_true_full  = data["y_true_full"]

    mask_train  = t_plot_full <= cfg.t_train
    mask_extrap = t_plot_full >  cfg.t_train

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.55, wspace=0.38,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )
    ax_data   = fig.add_subplot(gs[0, :])
    ax_train  = fig.add_subplot(gs[1, :2])
    ax_loss   = fig.add_subplot(gs[1, 2])
    ax_extrap = fig.add_subplot(gs[2, :2])
    ax_phys   = fig.add_subplot(gs[2, 2])

    for ax in [ax_data, ax_train, ax_loss, ax_extrap, ax_phys]:
        style_ax(ax)

    # ── P1: training data ─────────────────────────────────────────────────────
    # Val set (dense noiseless grid) intentionally omitted — it traces the
    # true solution and carries no additional visual information.
    # Collocation points omitted — they are randomized each epoch.
    ax_data.set_title(
        "Panel 1 — Training data: sparse noisy observations vs true trajectory\n"
        f"N_train={len(t_obs_train)}  σ={cfg.sigma}",
        fontsize=9, loc="left", pad=6, color="#444441",
    )
    ax_data.plot(t_plot_train, y_true_train,
                 color=GRAY, lw=1.5, label="True trajectory  y(t)")
    ax_data.scatter(t_obs_train, y_obs_train,
                    color=BLUE, s=50, zorder=5,
                    label=f"Train obs  (N={len(t_obs_train)}, σ={cfg.sigma})")
    ax_data.scatter([0], [cfg.y0],
                    color=PURPLE, s=160, marker="*", zorder=7,
                    label=f"IC  y(0)={cfg.y0}  [enforced via L_ic]")

    ax_data.set_xlabel("time  [s]")
    ax_data.set_ylabel("displacement  y(t)")
    ax_data.legend(fontsize=7.5, framealpha=0.5, ncol=2)
    ax_data.set_xlim(0, cfg.t_train)

    # ── P2: fit on training interval ──────────────────────────────────────────
    ax_train.set_title(
        f"Panel 2 — Fit  [0, {cfg.t_train} s]",
        fontsize=10, loc="left", pad=6, color="#444441",
    )
    ax_train.plot(t_plot_full[mask_train], y_true_full[mask_train],
                  color=GRAY,  lw=1.5, label="True")
    ax_train.plot(t_plot_full[mask_train], y_ml[mask_train],
                  color=RED,   lw=2, ls="--",
                  label=f"ML          (RMSE={m['rmse_ml_train']:.4f})")
    ax_train.plot(t_plot_full[mask_train], y_ext[mask_train],
                  color=GREEN, lw=2,
                  label=f"PINN ext.   (RMSE={m['rmse_pinn_ext_phys_train']:.4f})")
    ax_train.plot(t_plot_full[mask_train], y_blind[mask_train],
                  color=AMBER, lw=2, ls="-.",
                  label=f"PINN blind  (RMSE={m['rmse_pinn_blind_train']:.4f})")
    ax_train.scatter(t_obs_train, y_obs_train,
                     color=BLUE, s=28, zorder=5, alpha=0.6,
                     label=f"Train obs  (N={len(t_obs_train)})")
    ax_train.scatter([0], [cfg.y0],
                     color=PURPLE, s=120, marker="*", zorder=7)
    ax_train.set_xlabel("time  [s]")
    ax_train.set_ylabel("displacement  y(t)")
    ax_train.legend(fontsize=7.5, framealpha=0.5)
    ax_train.set_xlim(0, cfg.t_train)

    # ── P3: loss curves ────────────────────────────────────────────────────────
    ax_loss.set_title("Panel 3 — Training loss  (log scale)",
                      fontsize=10, loc="left", pad=6, color="#444441")
    # ML
    ax_loss.semilogy(hist_ml["epoch"],    hist_ml["loss_data"],
                     color=RED,   lw=1.5, ls="--", label="ML  L_data")
    ax_loss.semilogy(hist_ml["epoch"],    hist_ml["loss_val"],
                     color=RED,   lw=1.2, ls=":",  label="ML  L_val")
    # PINN ext
    ax_loss.semilogy(hist_ext["epoch"],   hist_ext["loss_data"],
                     color=GREEN, lw=1.5,            label="Ext  L_data")
    ax_loss.semilogy(hist_ext["epoch"],   hist_ext["loss_val"],
                     color=GREEN, lw=1.2, ls=":",   label="Ext  L_val")
    ax_loss.semilogy(hist_ext["epoch"],   hist_ext["loss_physics"],
                     color=GREEN, lw=1.0, ls=":",   alpha=0.5, label="Ext  L_phys")
    ax_loss.semilogy(hist_ext["epoch"],   hist_ext["loss_ic"],
                     color=GREEN, lw=1.0, ls="-.",  alpha=0.5, label="Ext  L_ic")
    # PINN blind
    ax_loss.semilogy(hist_blind["epoch"], hist_blind["loss_data"],
                     color=AMBER, lw=1.5,            label="Blind L_data")
    ax_loss.semilogy(hist_blind["epoch"], hist_blind["loss_val"],
                     color=AMBER, lw=1.2, ls=":",   label="Blind L_val")
    ax_loss.semilogy(hist_blind["epoch"], hist_blind["loss_physics"],
                     color=AMBER, lw=1.0, ls=":",   alpha=0.5, label="Blind L_phys")
    ax_loss.semilogy(hist_blind["epoch"], hist_blind["loss_ic"],
                     color=AMBER, lw=1.0, ls="-.",  alpha=0.5, label="Blind L_ic")
    for ep in cfg.snapshot_epochs[:-1]:
        ax_loss.axvline(ep, color=GRAY, lw=0.5, ls=":", alpha=0.4)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=5.5, framealpha=0.5, ncol=2)

    # ── P4: extrapolation ─────────────────────────────────────────────────────
    ax_extrap.set_title(
        f"Panel 4 — Extrapolation  [{cfg.t_train}, {cfg.t_extrap} s]",
        fontsize=10, loc="left", pad=6, color="#444441",
    )
    shade_extrap(ax_extrap, cfg)
    ax_extrap.plot(t_plot_full, y_true_full,
                   color=GRAY,  lw=1.5, label="True")
    ax_extrap.plot(t_plot_full, np.clip(y_ml, -3, 3),
                   color=RED,   lw=2, ls="--",
                   label=f"ML          (extrap RMSE={m['rmse_ml_extrap']:.4f})")
    ax_extrap.plot(t_plot_full, y_ext,
                   color=GREEN, lw=2,
                   label=f"PINN ext.   (extrap RMSE={m['rmse_pinn_ext_phys_extrap']:.4f})")
    ax_extrap.plot(t_plot_full, y_blind,
                   color=AMBER, lw=2, ls="-.",
                   label=f"PINN blind  (extrap RMSE={m['rmse_pinn_blind_extrap']:.4f})")
    # Only the real noisy measurements are shown; dense val grid not plotted.
    ax_extrap.scatter(t_obs_train, y_obs_train,
                      color=BLUE, s=25, zorder=5, alpha=0.5,
                      label=f"Train obs  (N={len(t_obs_train)})")
    ax_extrap.set_xlabel("time  [s]")
    ax_extrap.set_ylabel("displacement  y(t)")
    ax_extrap.set_ylim(-2.5, 2.5)
    ax_extrap.set_xlim(0, cfg.t_extrap)
    ax_extrap.legend(fontsize=7.5, framealpha=0.5)
    ax_extrap.annotate("training", xy=(0.12, 0.94),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)
    ax_extrap.annotate("extrapolation", xy=(0.68, 0.94),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)

    # ── P5: ODE residual ──────────────────────────────────────────────────────
    ax_phys.set_title("Panel 5 — Pointwise ODE residual  |r(t)|",
                      fontsize=10, loc="left", pad=6, color="#444441")
    shade_extrap(ax_phys, cfg)
    r_ml    = pointwise_residual(y_ml,    t_plot_full, cfg)
    r_ext   = pointwise_residual(y_ext,   t_plot_full, cfg)
    r_blind = pointwise_residual(y_blind, t_plot_full, cfg)
    ax_phys.plot(t_plot_full, r_ml,    color=RED,   lw=1.5, ls="--",
                 label=f"ML          ({m['phys_ml']:.3f})")
    ax_phys.plot(t_plot_full, r_ext,   color=GREEN, lw=1.5,
                 label=f"PINN ext.   ({m['phys_pinn_ext_phys']:.3f})")
    ax_phys.plot(t_plot_full, r_blind, color=AMBER, lw=1.5, ls="-.",
                 label=f"PINN blind  ({m['phys_pinn_blind']:.3f})")
    ax_phys.set_xlabel("time  [s]")
    ax_phys.set_ylabel("|m·y'' + c·y' + k·y|")
    ax_phys.set_xlim(0, cfg.t_extrap)
    ax_phys.legend(fontsize=7.5, framealpha=0.5)

    fig.suptitle(
        f"Std ML vs PINN (ext.) vs PINN (blind)  |  device={device_str}\n"
        f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
        f"ζ={cfg.zeta:.3f}  ω₀={cfg.omega_0:.3f}  ωd={cfg.omega_d:.3f} rad/s  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"y(0): ML={m['y0_ml']:.3f}  ext={m['y0_pinn_ext_phys']:.3f}  "
        f"blind={m['y0_pinn_blind']:.3f}  [true {cfg.y0}]",
        fontsize=9, y=0.975, color="#2C2C2A",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 1 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURES 2–4 — epoch-by-epoch trajectory snapshots  (one model per figure)
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
) -> plt.Figure:
    """
    2-column grid of panels, one per snapshot epoch.

    Shows: true solution, model prediction, train obs only (val not plotted),
    IC star, both collocation tick rows, extrapolation shading + RMSE.
    """
    t_obs_train = data["t_obs_train"]
    y_obs_train = data["y_obs_train"]
    t_plot_full = data["t_plot_full"]
    y_true_full = data["y_true_full"]

    epochs  = sorted(snapshots.keys())
    n_snap  = len(epochs)
    n_cols  = 2
    n_rows  = (n_snap + 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, n_rows * 3.2),
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    y_lo, y_hi  = -2.2, 1.6

    for idx, epoch in enumerate(epochs):
        ax = axes_flat[idx]
        style_ax(ax)

        y_pred = pred.predict_from_state(snapshots[epoch], t_plot_full)
        if clip_y:
            y_pred = np.clip(y_pred, -3, 3)
        err = Predictor.rmse(y_pred, y_true_full)

        ax.plot(t_plot_full, y_true_full,
                color=GRAY, lw=1.4, alpha=0.9)
        ax.plot(t_plot_full, y_pred,
                color=model_color, lw=2.0)

        ax.axvspan(cfg.t_train, cfg.t_extrap, color=GRAY, alpha=0.10, zorder=0)
        ax.axvline(cfg.t_train, color=GRAY, lw=0.8, ls="--", alpha=0.5)

        # Train obs only — val grid not plotted
        ax.scatter(t_obs_train, y_obs_train,
                   color=BLUE, s=28, zorder=5, alpha=0.85)

        ax.scatter([0], [cfg.y0],
                   color=PURPLE, s=130, marker="*", zorder=7)

        ax.set_title(f"epoch = {epoch}   |   RMSE = {err:.4f}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xlim(0, cfg.t_extrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y(t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("time  [s]", fontsize=8)

        if idx == 0:
            ax.text(cfg.t_train / 2, y_hi - 0.15, "training",
                    ha="center", va="top", fontsize=7, color=GRAY)
            ax.text(cfg.t_train + (cfg.t_extrap - cfg.t_train) / 2,
                    y_hi - 0.15, "extrapolation",
                    ha="center", va="top", fontsize=7, color=GRAY)

    for idx in range(n_snap, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=GRAY,        lw=1.4,
               label="True solution  y(t)"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=f"{model_label} prediction  ŷ(t)"),
        Line2D([0], [0], color=BLUE,        lw=0, marker="o", markersize=5,
               label=f"Train obs  (N={len(t_obs_train)}, σ={cfg.sigma})"),
        Line2D([0], [0], color=PURPLE,      lw=0, marker="*", markersize=9,
               label=f"IC  y(0)={cfg.y0},  y'(0)={cfg.dy0}"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(fig_title, fontsize=10, y=1.01, color="#2C2C2A")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure saved     →  {out_path}")
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
        help=(
            "Direct path to a training_results.pt file.  "
            "If omitted, path is reconstructed from --input_dir, --w0, --zeta."
        ),
    )
    p.add_argument(
        "--input_dir", type=str, default=DamperConfig.out_dir,
        help="Directory containing the results file (used when --results is not set).",
    )
    p.add_argument(
        "--w0", type=float, default=DamperConfig().omega_0,
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

    # ── Resolve results file path ─────────────────────────────────────────────
    if args.results is not None:
        results_path = Path(args.results)
    else:
        cfg_defaults = DamperConfig()
        w0   = args.w0
        zeta = args.zeta
        filename = f"w0{w0:.1e}_zeta{zeta:.1e}_{cfg_defaults.suffix_results_pt}"
        results_path = Path(args.input_dir) / filename

    if not results_path.exists():
        print(
            f"[plot.py] ERROR: results file not found: {results_path}\n"
            "  Run  python train.py  first, or supply --results <path>."
        )
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load bundle ───────────────────────────────────────────────────────────
    print(f"\n  Loading  {results_path} …")
    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: DamperConfig = raw["config"]
    data              = raw["data"]

    print(f"  Config   : m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
          f"ζ={cfg.zeta:.4f}  ω₀={cfg.omega_0:.4f}  ωd={cfg.omega_d:.4f}")
    print(f"  Train obs: N={len(data['t_obs_train'])}  "
          f"Val grid : N={len(data['t_obs_val'])} "
          f"(dense, noiseless — used for early stopping only, not plotted)")
    print(f"  Snapshots ML        : {sorted(raw['snaps_ml'].keys())}")
    print(f"  Snapshots PINN ext  : {sorted(raw['snaps_pinn_ext_phys'].keys())}")
    print(f"  Snapshots PINN blind: {sorted(raw['snaps_pinn_blind'].keys())}\n")

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

    pred_ml    = Predictor(cfg)
    pred_ext   = Predictor(cfg)
    pred_blind = Predictor(cfg)

    print("Generating Figure 1 — summary …")
    make_summary_figure(bundle, out_path=out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_fig1_summary.png")

    print("Generating Figure 2 — Standard ML epoch snapshots …")
    make_epoch_figure(
        snapshots   = bundle["snaps_ml"],
        cfg         = cfg,
        data        = data,
        pred        = pred_ml,
        model_color = RED,
        model_label = "Std ML",
        fig_title   = (
            f"Standard ML — trajectory evolution across epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            "Data loss only — no physics enforcement, no IC constraint"
        ),
        out_path    = out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_fig2_ml_epochs.png",
        clip_y      = True,
    )

    print("Generating Figure 3 — PINN (ext. physics) epoch snapshots …")
    make_epoch_figure(
        snapshots   = bundle["snaps_pinn_ext_phys"],
        cfg         = cfg,
        data        = data,
        pred        = pred_ext,
        model_color = GREEN,
        model_label = "PINN ext.",
        fig_title   = (
            f"PINN (ext. physics) — trajectory evolution across epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            f"Collocation extends to t_extrap={cfg.t_extrap}  |  "
            f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}"
        ),
        out_path    = out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_fig3_pinn_ext_epochs.png",
        clip_y      = False,
    )

    print("Generating Figure 4 — PINN (blind) epoch snapshots …")
    make_epoch_figure(
        snapshots   = bundle["snaps_pinn_blind"],
        cfg         = cfg,
        data        = data,
        pred        = pred_blind,
        model_color = AMBER,
        model_label = "PINN blind",
        fig_title   = (
            f"PINN (blind) — trajectory evolution across epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            f"Collocation only in training window [0, {cfg.t_train}]  |  "
            f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}"
        ),
        out_path    = out_dir / f"w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_fig4_pinn_blind_epochs.png",
        clip_y      = False,
    )

    print(f"\n  All figures written to  {out_dir}/")
    if not args.no_show:
        plt.show()
    print("  Done.\n")


if __name__ == "__main__":
    main()