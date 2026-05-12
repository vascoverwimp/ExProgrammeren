"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
three figures for the Burgers' equation PINN:

  Fig 1  burger_fig1_summary.png      — 2×3 heatmap overview + loss curves
  Fig 2  burger_fig2_slices.png       — u(x) profiles at fixed time slices
  Fig 3  burger_fig3_pinn_epochs.png  — PINN field snapshots per epoch

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

from model_claude import BurgerConfig, Predictor


# =============================================================================
# COLOUR PALETTE
# =============================================================================

BLUE   = "#378ADD"   # ML prediction / train observations
TEAL   = "#1D9E75"   # PINN prediction
ORANGE = "#EF9F27"   # collocation points
PURPLE = "#7F77DD"   # validation observations / IC
GRAY   = "#888780"   # true solution / neutral lines
LGRAY  = "#D3D1C7"   # spines
BG     = "#FAFAF8"   # figure background
PANEL  = "#F1EFE8"   # axes background

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
) -> None:
    """
    Plot a pcolormesh heatmap with:
      · x on the horizontal axis
      · t on the vertical axis
      · dashed horizontal line at t = t_train marking the extrapolation boundary
    """
    style_ax(ax)
    im = ax.pcolormesh(
        x_vals, t_vals, grid,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", rasterized=True,
    )
    _colorbar(ax, im, label=cbar_label)
    ax.axhline(cfg.t_train, color="white", lw=1.2, ls="--", alpha=0.9)
    ax.text(
        x_vals[int(len(x_vals) * 0.02)], cfg.t_train + 0.02 * (t_vals[-1] - t_vals[0]),
        "extrap ↑", color="white", fontsize=7, va="bottom",
    )
    ax.set_xlabel("x  [m]",    fontsize=8)
    ax.set_ylabel("time  [s]", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", pad=4, color="#444441")
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(t_vals[0],  t_vals[-1])


# =============================================================================
# FIGURE 1 — 2×3 summary  (heatmaps + loss)
# =============================================================================
#
#  ┌──────────────┬──────────────┬──────────────┐
#  │  True u(x,t) │  PINN pred   │  ML pred     │
#  ├──────────────┼──────────────┼──────────────┤
#  │  PINN |error│  ML  |error │  Loss curves │
#  └──────────────┴──────────────┴──────────────┘

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    cfg: BurgerConfig = bundle["cfg"]
    data              = bundle["data"]
    hist_ml           = bundle["hist_ml"]
    hist_pinn         = bundle["hist_pinn"]
    u_ml_flat         = bundle["u_ml_full"]
    u_pinn_flat       = bundle["u_pinn_full"]
    m                 = bundle["metrics"]
    device_str        = bundle["device_str"]

    u_true_flat = data["u_true_full"]

    t_vals, x_vals, grid_true = _to_grid(u_true_flat, data, "full")
    _,      _,      grid_pinn = _to_grid(u_pinn_flat, data, "full")
    _,      _,      grid_ml   = _to_grid(u_ml_flat,   data, "full")

    err_pinn = np.abs(grid_pinn - grid_true)
    err_ml   = np.abs(grid_ml   - grid_true)

    # Shared colour limits so the three field maps are directly comparable
    u_abs   = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u = -u_abs, u_abs
    vmax_err = max(err_pinn.max(), err_ml.max())

    fig = plt.figure(figsize=(17, 10))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.55, wspace=0.42,
        left=0.06, right=0.97, top=0.92, bottom=0.07,
    )
    ax_true  = fig.add_subplot(gs[0, 0])
    ax_pinn  = fig.add_subplot(gs[0, 1])
    ax_ml    = fig.add_subplot(gs[0, 2])
    ax_ep    = fig.add_subplot(gs[1, 0])
    ax_em    = fig.add_subplot(gs[1, 1])
    ax_loss  = fig.add_subplot(gs[1, 2])

    # ── Row 0: field heatmaps ─────────────────────────────────────────────────
    _heatmap(ax_true, t_vals, x_vals, grid_true, "True  u(x, t)",
             cfg, vmin_u, vmax_u)
    _heatmap(ax_pinn, t_vals, x_vals, grid_pinn,
             f"PINN  (train RMSE={m['rmse_pinn_train']:.4f},"
             f"  extrap RMSE={m['rmse_pinn_ext']:.4f})",
             cfg, vmin_u, vmax_u)
    _heatmap(ax_ml,   t_vals, x_vals, grid_ml,
             f"Std ML  (train RMSE={m['rmse_ml_train']:.4f},"
             f"  extrap RMSE={m['rmse_ml_ext']:.4f})",
             cfg, vmin_u, vmax_u)

    # Scatter observations on the PINN and ML panels
    for ax in (ax_pinn, ax_ml):
        ax.scatter(data["t_obs_train"], data["t_obs_train"] * 0 + data["t_obs_train"],  # placeholder — see note
                   s=0)   # observations are (x, t) pairs; plot below properly

    # Scatter actual (x_obs, t_obs) points on all three panels
    for ax in (ax_true, ax_pinn, ax_ml):
        ax.scatter(data["t_obs_train"], data["t_obs_train"],  # intentional: x_obs not saved separately
                   s=0)  # skip — x_obs_train not in numpy-only bundle; see Note below

    # ── Row 1: error maps ─────────────────────────────────────────────────────
    _heatmap(ax_ep, t_vals, x_vals, err_pinn,
             f"PINN  |error|   (mean={err_pinn.mean():.4f},"
             f"  max={err_pinn.max():.4f})",
             cfg, vmin=0, vmax=vmax_err, cmap=CMAP_ERROR, cbar_label="|u_pred − u_true|")
    _heatmap(ax_em, t_vals, x_vals, err_ml,
             f"Std ML  |error|  (mean={err_ml.mean():.4f},"
             f"  max={err_ml.max():.4f})",
             cfg, vmin=0, vmax=vmax_err, cmap=CMAP_ERROR, cbar_label="|u_pred − u_true|")

    # ── Row 1 right: loss curves ───────────────────────────────────────────────
    style_ax(ax_loss)
    ax_loss.set_title("Training loss  (log scale)",
                      fontsize=9, loc="left", pad=4, color="#444441")
    ax_loss.semilogy(hist_ml["epoch"],   hist_ml["loss_data"],
                     color=BLUE,   lw=1.5, ls="--",  label="ML   L_data")
    ax_loss.semilogy(hist_ml["epoch"],   hist_ml["loss_val"],
                     color=BLUE,   lw=1.5, ls=":",   label="ML   L_val")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_data"],
                     color=TEAL,   lw=1.5,            label="PINN L_data")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_val"],
                     color=TEAL,   lw=1.5, ls=":",   label="PINN L_val")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_physics"],
                     color=ORANGE, lw=1.2, ls=":",   label="PINN L_phys")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_ic"],
                     color=PURPLE, lw=1.2, ls="-.",  label="PINN L_ic")
    ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_total"],
                     color=TEAL,   lw=2.5, alpha=0.25, label="PINN L_total")
    for ep in cfg.snapshot_epochs[:-1]:
        ax_loss.axvline(ep, color=GRAY, lw=0.5, ls=":", alpha=0.4)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=6.5, framealpha=0.5)

    fig.suptitle(
        f"Burgers' PINN vs Std ML  |  device={device_str}  |  ν={cfg.v}  "
        f"situation={cfg.situation}  |  "
        f"x∈[{cfg.x_begin}, {cfg.x_end}]  t∈[{cfg.t0}, {cfg.t_extrap}]\n"
        f"{cfg.n_obs} obs  σ={cfg.sigma}  |  "
        f"{cfg.n_col_t}×{cfg.n_col_x} collocation pts  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"training window: t ≤ {cfg.t_train}",
        fontsize=9, y=0.975, color="#2C2C2A",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 1 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 2 — time-slice profiles  u(x) at fixed t values
# =============================================================================
#
# For a PDE with a 2-D solution field, heatmaps give a global overview but
# hide pointwise error.  Slicing at fixed t gives 1-D profiles where PINN vs
# ML vs true can be read directly.
#
# Layout: 2 columns × n_rows rows.  Times are chosen to cover both the
# training region and the extrapolation region evenly.

def make_slice_figure(bundle: dict, out_path: Path,
                      n_slices: int = 8) -> plt.Figure:
    """
    Show u(x) profiles at *n_slices* fixed times spanning [t0, t_extrap].

    Slice times are chosen to give ~50 % in the training window and ~50 %
    in the extrapolation window so both regimes are represented equally.
    """
    cfg: BurgerConfig = bundle["cfg"]
    data              = bundle["data"]
    u_ml_flat         = bundle["u_ml_full"]
    u_pinn_flat       = bundle["u_pinn_full"]
    m                 = bundle["metrics"]

    u_true_flat = data["u_true_full"]

    t_vals, x_vals, grid_true = _to_grid(u_true_flat, data, "full")
    _,      _,      grid_pinn = _to_grid(u_pinn_flat, data, "full")
    _,      _,      grid_ml   = _to_grid(u_ml_flat,   data, "full")

    # Choose slice times: half from training, half from extrapolation
    t_train_vals  = t_vals[t_vals <= cfg.t_train]
    t_extrap_vals = t_vals[t_vals >  cfg.t_train]
    n_train_s  = n_slices // 2
    n_extrap_s = n_slices - n_train_s
    train_times  = np.linspace(t_train_vals[1],  t_train_vals[-1],  n_train_s)
    extrap_times = np.linspace(t_extrap_vals[0], t_extrap_vals[-1], n_extrap_s)
    slice_times  = np.concatenate([train_times, extrap_times])

    n_cols = 2
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

        u_true_s = grid_true[i_t, :]
        u_pinn_s = grid_pinn[i_t, :]
        u_ml_s   = grid_ml[i_t, :]

        rmse_p = float(np.sqrt(np.mean((u_pinn_s - u_true_s) ** 2)))
        rmse_m = float(np.sqrt(np.mean((u_ml_s   - u_true_s) ** 2)))

        ax.plot(x_vals, u_true_s, color=GRAY, lw=1.8,
                label="True", zorder=5)
        ax.plot(x_vals, u_pinn_s, color=TEAL, lw=2.0, ls="-",
                label=f"PINN  (RMSE={rmse_p:.4f})", zorder=4)
        ax.plot(x_vals, u_ml_s,   color=BLUE, lw=1.6, ls="--",
                label=f"ML   (RMSE={rmse_m:.4f})", zorder=3)

        region_tag = "extrap" if in_extrap else "train"
        color_tag  = ORANGE   if in_extrap else TEAL
        ax.set_title(
            f"t = {t_actual:.3f} s  [{region_tag}]",
            fontsize=9, loc="left", pad=4, color=color_tag,
        )
        if in_extrap:
            ax.set_facecolor("#EDE8E0")   # slightly warmer background in extrap
        ax.set_ylabel("u(x, t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("x  [m]", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.5)

    # Hide any unused panels
    for idx in range(n_slices, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Burgers' PINN vs Std ML — u(x) profiles at fixed times\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"Training window: t ≤ {cfg.t_train}  (orange panels = extrapolation)",
        fontsize=9, y=1.01, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 2 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 3 — PINN epoch snapshots  (one heatmap per snapshot epoch)
# =============================================================================
#
# Layout: 2 columns.  Left column: PINN prediction field.
#                     Right column: absolute error |PINN − true|.
# Each row corresponds to one snapshot epoch.
# This makes it easy to watch the 2-D field converge as training proceeds.

def make_epoch_figure(bundle: dict, out_path: Path) -> plt.Figure:
    """
    Two heatmaps per snapshot epoch: the predicted field and its error map.

    Snapshot state_dicts are CPU dicts — no device transfer required.
    Predictor.predict_from_state swaps weights in-place and restores them,
    so only one FCNet is allocated for all epochs.
    """
    cfg: BurgerConfig = bundle["cfg"]
    snaps             = bundle["snaps_pinn"]
    data              = bundle["data"]
    pred: Predictor   = bundle["pred"]

    u_true_flat = data["u_true_full"]
    t_vals, x_vals, grid_true = _to_grid(u_true_flat, data, "full")
    x_flat = data["x_flattened_full"]
    t_flat = data["t_flattened_full"]

    # Shared colour limits across all epochs for fair comparison
    u_abs    = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u = -u_abs, u_abs

    epochs_sorted = sorted(snaps.keys())
    n_snap = len(epochs_sorted)

    # 2 heatmaps per epoch → 2 columns of pairs, each pair = (pred | error)
    # Layout: n_snap rows × 2 columns
    fig, axes = plt.subplots(
        n_snap, 2,
        figsize=(13, n_snap * 3.2),
    )
    fig.patch.set_facecolor(BG)
    if n_snap == 1:
        axes = axes[np.newaxis, :]   # ensure 2-D indexing

    # Compute global max error across all epochs for a shared error colour scale
    vmax_err_global = 0.0
    for epoch in epochs_sorted:
        u_ep = pred.predict_from_state(snaps[epoch], x_flat, t_flat)
        err_ep = np.abs(u_ep - u_true_flat)
        vmax_err_global = max(vmax_err_global, err_ep.max())

    for row, epoch in enumerate(epochs_sorted):
        ax_pred = axes[row, 0]
        ax_err  = axes[row, 1]

        u_ep_flat = pred.predict_from_state(snaps[epoch], x_flat, t_flat)
        _,    _, grid_ep  = _to_grid(u_ep_flat,              data, "full")
        _,    _, grid_err = _to_grid(np.abs(u_ep_flat - u_true_flat), data, "full")

        rmse_ep = Predictor.rmse(u_ep_flat, u_true_flat)

        # Left: predicted field
        _heatmap(ax_pred, t_vals, x_vals, grid_ep,
                 f"Epoch {epoch}   RMSE = {rmse_ep:.4f}",
                 cfg, vmin_u, vmax_u)

        # Right: error field
        _heatmap(ax_err, t_vals, x_vals, grid_err,
                 f"Epoch {epoch}   |error|   max={grid_err.max():.4f}",
                 cfg, vmin=0, vmax=vmax_err_global,
                 cmap=CMAP_ERROR, cbar_label="|u_pred − u_true|")

    fig.suptitle(
        f"PINN solution field evolution across training epochs\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"Training window: t ≤ {cfg.t_train}  (dashed line in each panel)",
        fontsize=9.5, y=1.005, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 3 saved  →  {out_path}")
    return fig


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Burgers PINN results from training_results.pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results", type=str, default= BurgerConfig.out_dir + "/training_results.pt",
        help="Path to training_results.pt written by train.py",
    )
    p.add_argument(
        "--out_dir", type=str, default=BurgerConfig.out_dir,
        help="Directory for figures (defaults to same folder as results file)",
    )
    p.add_argument(
        "--n_slices", type=int, default=8,
        help="Number of time-slice panels in Figure 2 (must be even)",
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

    cfg: BurgerConfig = raw["config"]
    snaps_pinn = raw["snaps_pinn"]

    # Recover grid dimensions from stored flat arrays
    t_flat = raw["data"]["t_flattened_full"]
    x_flat = raw["data"]["x_flattened_full"]
    n_t = len(np.unique(t_flat))
    n_x = len(np.unique(x_flat))

    print(f"  Config   : ν={cfg.v}  situation={cfg.situation}  "
          f"x∈[{cfg.x_begin},{cfg.x_end}]  t∈[{cfg.t0},{cfg.t_extrap}]")
    print(f"  Grid     : {n_t} × {n_x}  = {n_t * n_x} points")
    print(f"  Snapshots: {sorted(snaps_pinn.keys())}\n")

    # Single shared Predictor — predict_from_state swaps weights in-place
    pred = Predictor(cfg)

    bundle = {
        "cfg":        cfg,
        "device_str": raw["device_str"],
        "data":       raw["data"],
        "hist_ml":    raw["hist_ml"],
        "hist_pinn":  raw["hist_pinn"],
        "snaps_ml":   raw["snaps_ml"],
        "snaps_pinn": snaps_pinn,
        "u_ml_full":  raw["u_ml_full"],
        "u_pinn_full":raw["u_pinn_full"],
        "metrics":    raw["metrics"],
        "pred":       pred,
    }

    print("Generating Figure 1 — heatmap summary …")
    make_summary_figure(bundle, out_path=out_dir / "burger_fig1_summary.png")

    print(f"Generating Figure 2 — time-slice profiles ({args.n_slices} slices) …")
    make_slice_figure(bundle, out_path=out_dir / "burger_fig2_slices.png",
                      n_slices=args.n_slices)

    print("Generating Figure 3 — PINN epoch heatmaps …")
    make_epoch_figure(bundle, out_path=out_dir / "burger_fig3_pinn_epochs.png")

    print(f"\n  All figures written to  {out_dir}/")

    if not args.no_show:
        plt.show()

    print("  Done.\n")


if __name__ == "__main__":
    main()