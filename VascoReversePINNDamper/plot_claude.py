"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
three publication-quality figures:

  Fig 1  pinn_fig1_summary.png     — five-panel comparison overview
  Fig 2  pinn_fig2_pinn_epochs.png — PINN trajectory snapshots per epoch
  Fig 3  pinn_fig3_ml_epochs.png   — Standard-ML snapshots per epoch

Nothing here trains, nothing imports torch tensors from DEVICE — all
snapshot state_dicts were saved as CPU dicts in train.py, so this script
is fully CPU-bound and has no device dependencies.

Usage
-----
  python plot.py                                 # default: training_results.pt in cwd
  python plot.py --results  ./run_01/training_results.pt
  python plot.py --results  training_results.pt  --out_dir ./figs
  python plot.py --no_show                       # save only, do not call plt.show()
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

from model_claude import DamperConfig, Predictor


# =============================================================================
# COLOUR PALETTE  (single source of truth for all three figures)
# =============================================================================

BLUE   = "#378ADD"   # noisy observations
RED    = "#E24B4A"   # standard ML
GREEN  = "#1D9E75"   # PINN
ORANGE = "#EF9F27"   # collocation points
PURPLE = "#7F77DD"   # initial condition marker
GRAY   = "#888780"   # true solution / neutral
LGRAY  = "#D3D1C7"   # spine colour
BG     = "#FAFAF8"   # figure background
PANEL  = "#F1EFE8"   # axes background


# =============================================================================
# SHARED AXIS HELPERS
# =============================================================================

def style_ax(ax: plt.Axes) -> None:
    """Apply the common panel background and spine style."""
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def shade_extrap(ax: plt.Axes, cfg: DamperConfig) -> None:
    """Grey band + dashed vertical line marking the extrapolation region."""
    ax.axvspan(cfg.t_train, cfg.t_extrap, color=GRAY, alpha=0.12)
    ax.axvline(cfg.t_train, color=GRAY, lw=0.8, ls="--", alpha=0.6)


# =============================================================================
# POINTWISE ODE RESIDUAL  (numpy finite-difference approximation)
# =============================================================================

def pointwise_residual(
    y: np.ndarray, t: np.ndarray, cfg: DamperConfig
) -> np.ndarray:
    """
    Returns |m·y'' + c·y' + k·y| at every grid point via central
    finite differences.  Less accurate near boundaries but sufficient
    for a qualitative residual plot over a dense grid.
    """
    dy  = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(cfg.mass * d2y + cfg.damping * dy + cfg.stiffness * y)


# =============================================================================
# FIGURE 1 — five-panel summary comparison
# =============================================================================

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Panel 1  Training data: true trajectory + observations + collocation pts
    Panel 2  Fit on training interval  [0, t_train]
    Panel 3  Loss curves (log scale)
    Panel 4  Extrapolation  [0, t_extrap]  with shaded extrap region
    Panel 5  Pointwise ODE residual  |r(t)|
    """
    cfg: DamperConfig = bundle["cfg"]
    data              = bundle["data"]
    hist_ml           = bundle["hist_ml"]
    hist_pinn         = bundle["hist_pinn"]
    y_ml_full         = bundle["y_ml_full"]
    y_pinn_full       = bundle["y_pinn_full"]
    m                 = bundle["metrics"]
    device_str        = bundle["device_str"]

    t_obs        = data["t_obs"]
    y_obs        = data["y_obs"]
    t_obs_train  = data["t_obs_train"]
    y_obs_train  = data["y_obs_train"]
    t_obs_val    = data["t_obs_val"]
    y_obs_val    = data["y_obs_val"]
    t_col        = data["t_col"]
    t_plot_train = data["t_plot_train"]
    t_plot_full  = data["t_plot_full"]
    y_true_train = data["y_true_train"]
    y_true_full  = data["y_true_full"]

    mask_train  = t_plot_full <= cfg.t_train
    mask_extrap = t_plot_full >  cfg.t_train

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 12))
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

    # ── Panel 1: training data ────────────────────────────────────────────────
    ax_data.set_title(
        "Panel 1 — Training data: sparse noisy observations vs true trajectory\n"
        "t=0 excluded from observations; y(0) and y'(0) enforced via L_ic  |  "
        f"train split = {len(t_obs_train)}  val split = {len(t_obs_val)}",
        fontsize=9, loc="left", pad=6, color="#444441",
    )
    ax_data.plot(t_plot_train, y_true_train, color=GRAY, lw=1.5,
                 label="True trajectory  y(t)")
    ax_data.scatter(t_obs_train, y_obs_train, color=BLUE, s=50, zorder=5,
                    label=f"Train observations  (N={len(t_obs_train)}, σ={cfg.sigma})")
    ax_data.scatter(t_obs_val, y_obs_val, color=PURPLE, s=50, zorder=6,
                    marker="D", alpha=0.85,
                    label=f"Val observations  (N={len(t_obs_val)})")
    ax_data.scatter([0], [cfg.y0], color=PURPLE, s=160, marker="*", zorder=7,
                    label=f"Initial condition  y(0)={cfg.y0}  [enforced via L_ic]")
    col_in_train = t_col[t_col <= cfg.t_train]
    ax_data.scatter(col_in_train,
                    np.zeros_like(col_in_train) - 1.08,
                    color=ORANGE, s=8, marker="|", zorder=4, alpha=0.7,
                    label="Collocation points (no measurement needed)")
    ax_data.set_xlabel("time  [s]")
    ax_data.set_ylabel("displacement  y(t)")
    ax_data.legend(fontsize=8, framealpha=0.5)
    ax_data.set_xlim(0, cfg.t_train)

    # ── Panel 2: fit on training interval ─────────────────────────────────────
    ax_train.set_title(
        f"Panel 2 — Fit on training interval  [0, {cfg.t_train} s]",
        fontsize=10, loc="left", pad=6, color="#444441",
    )
    ax_train.plot(t_plot_full[mask_train], y_true_full[mask_train],
                  color=GRAY, lw=1.5, label="True")
    ax_train.plot(t_plot_full[mask_train], y_ml_full[mask_train],
                  color=RED, lw=2, ls="--",
                  label=f"Standard ML  (RMSE={m['rmse_ml_train']:.4f})")
    ax_train.plot(t_plot_full[mask_train], y_pinn_full[mask_train],
                  color=GREEN, lw=2,
                  label=f"PINN         (RMSE={m['rmse_pinn_train']:.4f})")
    ax_train.scatter(t_obs_train, y_obs_train,
                     color=BLUE, s=30, zorder=5, alpha=0.6, label="Train obs")
    ax_train.scatter(t_obs_val, y_obs_val,
                     color=PURPLE, s=30, marker="D", zorder=6, alpha=0.7,
                     label="Val obs")
    ax_train.scatter([0], [cfg.y0],
                     color=PURPLE, s=120, marker="*", zorder=7,
                     label=f"IC  y(0)={cfg.y0}")
    ax_train.set_xlabel("time  [s]")
    ax_train.set_ylabel("displacement  y(t)")
    ax_train.legend(fontsize=7.5, framealpha=0.5)
    ax_train.set_xlim(0, cfg.t_train)

    # ── Panel 3: loss curves ──────────────────────────────────────────────────
    ax_loss.set_title("Panel 3 — Training loss  (log scale)",
                      fontsize=10, loc="left", pad=6, color="#444441")
    ax_loss.semilogy(hist_ml["epoch"],   hist_ml["loss_data"],
                     color=RED,   lw=1.5, ls="--",  label="ML  L_data")
    ax_loss.semilogy(hist_ml["epoch"],   hist_ml["loss_val"],
                     color=RED,   lw=1.5, ls=":",   label="ML  L_val")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_data"],
                     color=GREEN, lw=1.5,            label="PINN  L_data")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_val"],
                     color=GREEN, lw=1.5, ls=":",    label="PINN  L_val")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_physics"],
                     color=ORANGE, lw=1.2, ls=":",   label="PINN  L_phys")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_ic"],
                     color=BLUE,  lw=1.2, ls="-.",   label="PINN  L_ic")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_total"],
                     color=GREEN, lw=2.5, alpha=0.25, label="PINN  L_total")
    for ep in cfg.snapshot_epochs[:-1]:
        ax_loss.axvline(ep, color=GRAY, lw=0.5, ls=":", alpha=0.5)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=6.5, framealpha=0.5)

    # ── Panel 4: extrapolation ─────────────────────────────────────────────────
    ax_extrap.set_title(
        f"Panel 4 — Extrapolation beyond training window  "
        f"[{cfg.t_train}, {cfg.t_extrap} s]",
        fontsize=10, loc="left", pad=6, color="#444441",
    )
    shade_extrap(ax_extrap, cfg)
    ax_extrap.plot(t_plot_full, y_true_full,
                   color=GRAY, lw=1.5, label="True")
    ax_extrap.plot(t_plot_full, np.clip(y_ml_full, -3, 3),
                   color=RED,  lw=2, ls="--",
                   label=f"Standard ML  (extrap RMSE={m['rmse_ml_ext']:.4f})")
    ax_extrap.plot(t_plot_full, y_pinn_full,
                   color=GREEN, lw=2,
                   label=f"PINN         (extrap RMSE={m['rmse_pinn_ext']:.4f})")
    ax_extrap.scatter(t_obs, y_obs,
                      color=BLUE, s=30, zorder=5, alpha=0.5,
                      label="Observations")
    ax_extrap.set_xlabel("time  [s]")
    ax_extrap.set_ylabel("displacement  y(t)")
    ax_extrap.set_ylim(-2.5, 2.5)
    ax_extrap.set_xlim(0, cfg.t_extrap)
    ax_extrap.legend(fontsize=7.5, framealpha=0.5)
    ax_extrap.annotate("training region", xy=(0.15, 0.93),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)
    ax_extrap.annotate("extrapolation",   xy=(0.70, 0.93),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)

    # ── Panel 5: pointwise ODE residual ───────────────────────────────────────
    ax_phys.set_title("Panel 5 — Pointwise ODE residual  |r(t)|",
                      fontsize=10, loc="left", pad=6, color="#444441")
    shade_extrap(ax_phys, cfg)
    r_ml   = pointwise_residual(y_ml_full,   t_plot_full, cfg)
    r_pinn = pointwise_residual(y_pinn_full, t_plot_full, cfg)
    ax_phys.plot(t_plot_full, r_ml,   color=RED,   lw=1.5, ls="--",
                 label=f"ML    (mean={m['phys_ml']:.3f})")
    ax_phys.plot(t_plot_full, r_pinn, color=GREEN, lw=1.5,
                 label=f"PINN  (mean={m['phys_pinn']:.3f})")
    ax_phys.set_xlabel("time  [s]")
    ax_phys.set_ylabel("|m·y'' + c·y' + k·y|")
    ax_phys.set_xlim(0, cfg.t_extrap)
    ax_phys.legend(fontsize=7.5, framealpha=0.5)

    # ── Supertitle ────────────────────────────────────────────────────────────
    fig.suptitle(
        f"PINN vs Standard ML  |  device={device_str}  |  "
        f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
        f"ζ={cfg.zeta:.3f}  ωd={cfg.omega_d:.3f} rad/s\n"
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
# FIGURES 2 & 3 — epoch-by-epoch trajectory snapshots
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
    One panel per snapshot epoch, arranged in a 2-column grid.

    Each panel shows:
      · true analytic solution             (gray line,    full domain)
      · model prediction at that epoch     (coloured line, full domain)
      · training observations              (blue dots)
      · validation observations            (purple diamonds)
      · initial condition                  (purple star,  t=0)
      · collocation points                 (orange ticks, along bottom)
      · training / extrapolation boundary  (dashed vertical line)
      · full-domain RMSE in the panel title

    Snapshots are CPU state_dicts — no device transfer required.
    Predictor.predict_from_state swaps weights in-place and restores them
    after each prediction so only one FCNet instance is allocated.
    """
    t_obs_train  = data["t_obs_train"]
    y_obs_train  = data["y_obs_train"]
    t_obs_val    = data["t_obs_val"]
    y_obs_val    = data["y_obs_val"]
    t_col        = data["t_col"]
    t_plot_full  = data["t_plot_full"]
    y_true_full  = data["y_true_full"]

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

    y_lo, y_hi = -2.2, 1.6
    col_y = y_lo + 0.08 * (y_hi - y_lo)   # y-position for collocation ticks

    for idx, epoch in enumerate(epochs):
        ax = axes_flat[idx]
        style_ax(ax)

        y_pred = pred.predict_from_state(snapshots[epoch], t_plot_full)
        if clip_y:
            y_pred = np.clip(y_pred, -3, 3)

        err = Predictor.rmse(y_pred, y_true_full)

        # True solution
        ax.plot(t_plot_full, y_true_full,
                color=GRAY, lw=1.4, alpha=0.9, label="True  y(t)")

        # Model prediction
        ax.plot(t_plot_full, y_pred,
                color=model_color, lw=2.0, label=f"{model_label}  ŷ(t)")

        # Extrapolation region
        ax.axvspan(cfg.t_train, cfg.t_extrap, color=GRAY, alpha=0.10, zorder=0)
        ax.axvline(cfg.t_train, color=GRAY, lw=0.8, ls="--", alpha=0.5)

        # Collocation points (orange ticks along the bottom)
        ax.scatter(t_col, np.full_like(t_col, col_y),
                   color=ORANGE, s=6, marker="|",
                   alpha=0.6, zorder=3, label="Collocation pts")

        # Training observations
        ax.scatter(t_obs_train, y_obs_train,
                   color=BLUE, s=28, zorder=5, alpha=0.85,
                   label=f"Train obs (N={len(t_obs_train)})")

        # Validation observations
        ax.scatter(t_obs_val, y_obs_val,
                   color=PURPLE, s=28, marker="D", zorder=6, alpha=0.7,
                   label=f"Val obs (N={len(t_obs_val)})")

        # Initial condition
        ax.scatter([0], [cfg.y0],
                   color=PURPLE, s=130, marker="*", zorder=7,
                   label=f"IC  y(0)={cfg.y0}")

        ax.set_title(f"epoch = {epoch}   |   RMSE = {err:.4f}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xlim(0, cfg.t_extrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y(t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("time  [s]", fontsize=8)

        # Region labels on the first panel only
        if idx == 0:
            ax.text(cfg.t_train / 2, y_hi - 0.15, "training",
                    ha="center", va="top", fontsize=7, color=GRAY)
            ax.text(cfg.t_train + (cfg.t_extrap - cfg.t_train) / 2,
                    y_hi - 0.15, "extrapolation",
                    ha="center", va="top", fontsize=7, color=GRAY)

    # Hide any unused panels in the grid
    for idx in range(n_snap, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared figure legend below all panels
    legend_elements = [
        Line2D([0], [0], color=GRAY,        lw=1.4,
               label="True solution  y(t)"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=f"{model_label} prediction  ŷ(t)"),
        Line2D([0], [0], color=BLUE,   lw=0, marker="o",  markersize=5,
               label=f"Train obs  (N={len(t_obs_train)}, σ={cfg.sigma})"),
        Line2D([0], [0], color=PURPLE, lw=0, marker="D",  markersize=5,
               label=f"Val obs  (N={len(t_obs_val)})"),
        Line2D([0], [0], color=PURPLE, lw=0, marker="*",  markersize=9,
               label=f"IC  y(0)={cfg.y0},  y'(0)={cfg.dy0}"),
        Line2D([0], [0], color=ORANGE, lw=0, marker="|",  markersize=7,
               label=f"Collocation pts  (N={cfg.n_col}, [0, {cfg.t_extrap}])"),
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
        "--results", type=str, default="training_results.pt",
        help="Path to the training_results.pt file written by train.py",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help=(
            "Directory for saved figures.  Defaults to the same directory "
            "as the results file."
        ),
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

    results_path = Path(args.results)
    if not results_path.exists():
        print(
            f"[plot.py] ERROR: results file not found: {results_path}\n"
            "  Run  python train.py  first."
        )
        sys.exit(1)

    # ── Resolve output directory ───────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load bundle ───────────────────────────────────────────────────────────
    print(f"\n  Loading  {results_path} …")
    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: DamperConfig = raw["config"]

    # Assemble the bundle dict that drawing functions expect
    bundle = {
        "cfg":        cfg,
        "device_str": raw["device_str"],
        "data":       raw["data"],
        "hist_ml":    raw["hist_ml"],
        "hist_pinn":  raw["hist_pinn"],
        "snaps_ml":   raw["snaps_ml"],
        "snaps_pinn": raw["snaps_pinn"],
        "y_ml_full":  raw["y_ml_full"],
        "y_pinn_full":raw["y_pinn_full"],
        "metrics":    raw["metrics"],
    }

    data = bundle["data"]

    print(f"  Config : m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  "
          f"ζ={cfg.zeta:.3f}  ωd={cfg.omega_d:.3f}")
    print(f"  Snapshots — PINN : {sorted(bundle['snaps_pinn'].keys())}")
    print(f"  Snapshots — ML   : {sorted(bundle['snaps_ml'].keys())}")
    print()

    # ── One Predictor instance per model (CPU only) ────────────────────────────
    # The Predictor swaps weights in-place for each snapshot query so we
    # avoid instantiating a new FCNet for every epoch panel.
    pred_ml   = Predictor(cfg)
    pred_pinn = Predictor(cfg)

    # ── Figure 1: summary ─────────────────────────────────────────────────────
    print("Generating Figure 1 — summary …")
    make_summary_figure(bundle, out_path=out_dir / "pinn_fig1_summary.png")

    # ── Figure 2: PINN epoch snapshots ────────────────────────────────────────
    print("Generating Figure 2 — PINN epoch snapshots …")
    make_epoch_figure(
        snapshots   = bundle["snaps_pinn"],
        cfg         = cfg,
        data        = data,
        pred        = pred_pinn,
        model_color = GREEN,
        model_label = "PINN",
        fig_title   = (
            f"PINN — trajectory evolution across training epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            f"{cfg.n_obs} obs  σ={cfg.sigma}  |  "
            f"{cfg.n_col} collocation pts  |  "
            f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}"
        ),
        out_path    = out_dir / "pinn_fig2_pinn_epochs.png",
        clip_y      = False,
    )

    # ── Figure 3: Standard-ML epoch snapshots ────────────────────────────────
    print("Generating Figure 3 — Standard-ML epoch snapshots …")
    make_epoch_figure(
        snapshots   = bundle["snaps_ml"],
        cfg         = cfg,
        data        = data,
        pred        = pred_ml,
        model_color = RED,
        model_label = "Std ML",
        fig_title   = (
            f"Standard ML — trajectory evolution across training epochs  "
            f"[device={bundle['device_str']}]\n"
            f"m={cfg.mass}  c={cfg.damping}  k={cfg.stiffness}  |  "
            f"{cfg.n_obs} obs  σ={cfg.sigma}  |  "
            "data loss only  (no physics, no IC enforcement)"
        ),
        out_path    = out_dir / "pinn_fig3_ml_epochs.png",
        clip_y      = True,
    )

    print(f"\n  All figures written to  {out_dir}/")

    if not args.no_show:
        plt.show()

    print("  Done.\n")


if __name__ == "__main__":
    main()