"""
plot.py
=======
Loads the training_results.pt bundle written by train.py and produces
three figures for the Burgers' equation PINN:

  Fig 1  burger_fig1_summary.png      — 2×4 heatmap overview + loss curves
  Fig 2  burger_fig2_slices.png       — u(x) profiles at fixed time slices
  Fig 3  burger_fig3_pinn_epochs.png  — PINN field snapshots per epoch

Three models are compared throughout:
  · Std ML               (data loss only)
  · PINN ext. physics    (data + physics over full domain + IC)
  · PINN blind           (data + physics over training window only + IC)

Usage
-----
  python plot.py
  python plot.py --results ./run_01/training_results.pt
  python plot.py --out_dir ./figs --no_show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

from model import BurgerConfig, Predictor


# =============================================================================
# COLOUR PALETTE
# =============================================================================

BLUE   = "#378ADD"   # Std ML
TEAL   = "#1D9E75"   # PINN ext. physics
ORANGE = "#EF9F27"   # collocation / extrapolation region
RED    = "#FF2D2D"   # PINN blind
GRAY   = "#888780"   # true solution
LGRAY  = "#D3D1C7"   # spines
BG     = "#FAFAF8"   # figure background
PANEL  = "#F1EFE8"   # axes background
SHOCK  = "#FF4C4C"   # shockwave marker


CMAP_FIELD = "RdBu_r"
CMAP_ERROR = "Oranges"


# =============================================================================
# SHARED HELPERS
# =============================================================================

def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)


def _to_grid(flat: np.ndarray, data: dict,
             which: str = "full") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_flat = data[f"t_flattened_{which}"]
    x_flat = data[f"x_flattened_{which}"]
    t_vals = np.unique(t_flat)
    x_vals = np.unique(x_flat)
    return t_vals, x_vals, flat.reshape(len(t_vals), len(x_vals))


def _colorbar(ax: plt.Axes, im, label: str = "") -> None:
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
    time_shockwave: float | None = None,
) -> None:
    style_ax(ax)
    im = ax.pcolormesh(x_vals, t_vals, grid, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
    _colorbar(ax, im, label=cbar_label)

    t_span      = t_vals[-1] - t_vals[0]
    x_label_pos = x_vals[int(len(x_vals) * 0.02)]

    ax.axhline(cfg.t_train, color="white", lw=1.2, ls="--", alpha=0.9)
    ax.text(x_label_pos, cfg.t_train + 0.02 * t_span,
            "extrap ↑", color="white", fontsize=7, va="bottom")

    if time_shockwave is not None:
        ax.axhline(time_shockwave, color=SHOCK, lw=1.2, ls="--", alpha=0.9)
        ax.text(x_label_pos, time_shockwave + 0.02 * t_span,
                "shock", color=SHOCK, fontsize=7, va="bottom")

    ax.set_xlabel("x  [m]",    fontsize=8)
    ax.set_ylabel("time  [s]", fontsize=8)
    ax.set_title(title, fontsize=8, loc="left", pad=4, color="#444441")
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(t_vals[0], t_vals[-1])


# =============================================================================
# FIGURE 1 — 2×4 summary
# =============================================================================
#
#  ┌──────────┬──────────────┬──────────────┬──────────┐
#  │ True u   │ PINN ext     │ PINN blind   │ Std ML   │
#  ├──────────┼──────────────┼──────────────┼──────────┤
#  │ ext err  │ blind err    │ ML err       │ Loss     │
#  └──────────┴──────────────┴──────────────┴──────────┘

def make_summary_figure(bundle: dict, out_path: Path) -> plt.Figure:
    cfg            = bundle["cfg"]
    data           = bundle["data"]
    m              = bundle["metrics"]
    device_str     = bundle["device_str"]
    time_shockwave = bundle.get("time_shockwave")

    t_vals, x_vals, grid_true     = _to_grid(data["u_true_full"],           data)
    _,      _,      grid_pinn_ext = _to_grid(bundle["u_pinn_ext_phys_full"], data)
    _,      _,      grid_pinn_bl  = _to_grid(bundle["u_pinn_blind_full"],    data)
    _,      _,      grid_ml       = _to_grid(bundle["u_ml_full"],            data)

    err_ext = np.abs(grid_pinn_ext - grid_true)
    err_bl  = np.abs(grid_pinn_bl  - grid_true)
    err_ml  = np.abs(grid_ml       - grid_true)

    u_abs             = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u    = -u_abs, u_abs
    vmax_err          = max(err_ext.max(), err_bl.max(), err_ml.max())

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.55, wspace=0.45,
                           left=0.05, right=0.97, top=0.91, bottom=0.07)

    ax_true  = fig.add_subplot(gs[0, 0])
    ax_ext   = fig.add_subplot(gs[0, 1])
    ax_bl    = fig.add_subplot(gs[0, 2])
    ax_ml    = fig.add_subplot(gs[0, 3])
    ax_eerr  = fig.add_subplot(gs[1, 0])
    ax_berr  = fig.add_subplot(gs[1, 1])
    ax_merr  = fig.add_subplot(gs[1, 2])
    ax_loss  = fig.add_subplot(gs[1, 3])

    # ── Row 0: field heatmaps ─────────────────────────────────────────────────
    kw = dict(cfg=cfg, vmin=vmin_u, vmax=vmax_u, time_shockwave=time_shockwave)
    _heatmap(ax_true, t_vals, x_vals, grid_true, "True  u(x, t)", **kw)
    _heatmap(ax_ext,  t_vals, x_vals, grid_pinn_ext,
             f"PINN ext.  train={m['rmse_pinn_ext_phys_train']:.4f}"
             f"  extrap={m['rmse_pinn_ext_phys_extrap']:.4f}", **kw)
    _heatmap(ax_bl,   t_vals, x_vals, grid_pinn_bl,
             f"PINN blind  train={m['rmse_pinn_blind_train']:.4f}"
             f"  extrap={m['rmse_pinn_blind_extrap']:.4f}", **kw)
    _heatmap(ax_ml,   t_vals, x_vals, grid_ml,
             f"Std ML  train={m['rmse_ml_train']:.4f}"
             f"  extrap={m['rmse_ml_extrap']:.4f}", **kw)

    # ── Row 1: error heatmaps ─────────────────────────────────────────────────
    ekw = dict(cfg=cfg, vmin=0, vmax=vmax_err, cmap=CMAP_ERROR,
               cbar_label="|u_pred − u_true|", time_shockwave=time_shockwave)
    _heatmap(ax_eerr, t_vals, x_vals, err_ext,
             f"PINN ext. |err|  mean={err_ext.mean():.4f}  max={err_ext.max():.4f}", **ekw)
    _heatmap(ax_berr, t_vals, x_vals, err_bl,
             f"PINN blind |err|  mean={err_bl.mean():.4f}  max={err_bl.max():.4f}", **ekw)
    _heatmap(ax_merr, t_vals, x_vals, err_ml,
             f"Std ML |err|  mean={err_ml.mean():.4f}  max={err_ml.max():.4f}", **ekw)

    # ── Row 1 right: loss curves ───────────────────────────────────────────────
    style_ax(ax_loss)
    ax_loss.set_title("Training loss  (log scale)", fontsize=9,
                      loc="left", pad=4, color="#444441")

    h_ml  = bundle["hist_ml"]
    h_ext = bundle["hist_pinn_ext_phys"]
    h_bl  = bundle["hist_pinn_blind"]

    ax_loss.semilogy(h_ml["epoch"],  h_ml["loss_data"],  color=BLUE,   lw=1.5, ls="--", label="ML  L_data")
    ax_loss.semilogy(h_ml["epoch"],  h_ml["loss_val"],   color=BLUE,   lw=1.5, ls=":",  label="ML  L_val")
    ax_loss.semilogy(h_ext["epoch"], h_ext["loss_data"], color=TEAL,   lw=1.5,          label="PINN-ext  L_data")
    ax_loss.semilogy(h_ext["epoch"], h_ext["loss_val"],  color=TEAL,   lw=1.5, ls=":",  label="PINN-ext  L_val")
    ax_loss.semilogy(h_ext["epoch"], h_ext["loss_physics"], color=TEAL, lw=1.0, ls="-.", label="PINN-ext  L_phys")
    ax_loss.semilogy(h_bl["epoch"],  h_bl["loss_data"],  color=RED, lw=1.5,          label="PINN-bl  L_data")
    ax_loss.semilogy(h_bl["epoch"],  h_bl["loss_val"],   color=RED, lw=1.5, ls=":",  label="PINN-bl  L_val")
    ax_loss.semilogy(h_bl["epoch"],  h_bl["loss_physics"], color=RED, lw=1.0, ls="-.", label="PINN-bl  L_phys")
    for ep in cfg.snapshot_epochs[:-1]:
        ax_loss.axvline(ep, color=GRAY, lw=0.5, ls=":", alpha=0.4)
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=6, framealpha=0.5, ncol=2)

    shock_note = f"  |  shock: t={time_shockwave:.4g} s" if time_shockwave is not None else ""
    fig.suptitle(
        f"Burgers' — Std ML vs PINN ext. vs PINN blind  |  device={device_str}"
        f"  |  ν={cfg.v}  situation={cfg.situation}"
        f"  |  x∈[{cfg.x_begin}, {cfg.x_end}]  t∈[{cfg.t0}, {cfg.t_extrap}]{shock_note}\n"
        f"{cfg.n_obs_total} obs  σ={cfg.sigma}  |  "
        f"{cfg.n_col_t}×{cfg.n_col_x} col. pts  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"train window: t ≤ {cfg.t_train}",
        fontsize=8.5, y=0.975, color="#2C2C2A",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 1 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 2 — time-slice profiles
# =============================================================================

def make_slice_figure(bundle: dict, out_path: Path,
                      n_slices: int = 8) -> plt.Figure:
    cfg            = bundle["cfg"]
    data           = bundle["data"]
    time_shockwave = bundle.get("time_shockwave")

    t_vals, x_vals, grid_true     = _to_grid(data["u_true_full"],           data)
    _,      _,      grid_pinn_ext = _to_grid(bundle["u_pinn_ext_phys_full"], data)
    _,      _,      grid_pinn_bl  = _to_grid(bundle["u_pinn_blind_full"],    data)
    _,      _,      grid_ml       = _to_grid(bundle["u_ml_full"],            data)

    # Slice time selection: half train, half extrap
    t_tr = t_vals[t_vals <= cfg.t_train]
    t_ex = t_vals[t_vals >  cfg.t_train]
    slice_times = np.linspace(t_tr[1], t_ex[-1], n_slices)
    n_tr = np.sum(slice_times <= cfg.t_train)
    # Ensure shock time is represented
    if time_shockwave is not None:
        i_shock = int(np.argmin(np.abs(t_vals - time_shockwave)))
        t_snap  = t_vals[i_shock]
        if not np.any(np.isclose(slice_times, t_snap)):
            diffs = np.abs(slice_times[:n_tr] - t_snap)
            slice_times[int(np.argmin(diffs))] = t_snap

    n_cols = 3
    n_rows = (n_slices + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.0), sharex=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    for idx, t_target in enumerate(slice_times):
        ax      = axes_flat[idx]
        style_ax(ax)
        i_t     = int(np.argmin(np.abs(t_vals - t_target)))
        t_act   = t_vals[i_t]
        in_ext  = t_act > cfg.t_train
        is_shock = time_shockwave is not None and np.isclose(t_act, time_shockwave, atol=1e-6)

        u_tr = grid_true[i_t, :]
        u_ep = grid_pinn_ext[i_t, :]
        u_bp = grid_pinn_bl[i_t, :]
        u_mp = grid_ml[i_t, :]

        rmse_e = float(np.sqrt(np.mean((u_ep - u_tr) ** 2)))
        rmse_b = float(np.sqrt(np.mean((u_bp - u_tr) ** 2)))
        rmse_m = float(np.sqrt(np.mean((u_mp - u_tr) ** 2)))

        ax.plot(x_vals, u_tr, color=GRAY,   lw=2.0, label="True",                         zorder=5)
        ax.plot(x_vals, u_ep, color=TEAL,   lw=1.8, label=f"PINN ext.  RMSE={rmse_e:.4f}", zorder=4)
        ax.plot(x_vals, u_bp, color=RED, lw=1.8, label=f"PINN blind RMSE={rmse_b:.4f}", zorder=3)
        ax.plot(x_vals, u_mp, color=BLUE,   lw=1.5, ls=":",  label=f"Std ML     RMSE={rmse_m:.4f}", zorder=2)

        if is_shock:
            i_steep = int(np.argmax(np.abs(np.gradient(u_tr, x_vals))))
            ax.axvline(x_vals[i_steep], color=SHOCK, lw=1.2, ls="--",
                       alpha=0.8, label="shock front")

        region   = "extrap" if in_ext  else "train"
        shock_lbl = "  ← shock" if is_shock else ""
        title_col = SHOCK if is_shock else (ORANGE if in_ext else TEAL)

        ax.set_title(f"t = {t_act:.3f} s  [{region}]{shock_lbl}",
                     fontsize=9, loc="left", pad=4, color=title_col)
        if in_ext:
            ax.set_facecolor("#EDE8E0")
        if is_shock:
            ax.set_facecolor("#FFE8E8")

        ax.set_ylabel("u(x, t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("x  [m]", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.5)

    for idx in range(n_slices, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    shock_note = f"  |  shock: t={time_shockwave:.4g} s  (red)" if time_shockwave is not None else ""
    fig.suptitle(
        f"Burgers' — u(x) profiles at fixed times\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"train window: t ≤ {cfg.t_train}  (orange = extrap){shock_note}",
        fontsize=9, y=1.01, color="#2C2C2A",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Figure 2 saved  →  {out_path}")
    return fig


# =============================================================================
# FIGURE 3 — epoch snapshots for both PINNs
# =============================================================================
#
# Layout per epoch row:  4 columns
#   [PINN ext pred] [PINN ext |err|] [PINN blind pred] [PINN blind |err|]

def make_epoch_figure(bundle: dict, out_path: Path) -> plt.Figure:
    cfg            = bundle["cfg"]
    data           = bundle["data"]
    pred_ext: Predictor  = bundle["pred_ext"]
    pred_bl:  Predictor  = bundle["pred_bl"]
    snaps_ext  = bundle["snaps_pinn_ext_phys"]
    snaps_bl   = bundle["snaps_pinn_blind"]
    time_shockwave = bundle.get("time_shockwave")

    t_vals, x_vals, grid_true = _to_grid(data["u_true_full"], data)
    x_flat = data["x_flattened_full"]
    t_flat = data["t_flattened_full"]

    u_abs          = np.nanmax(np.abs(grid_true))
    vmin_u, vmax_u = -u_abs, u_abs

    # Use epochs present in both snapshots
    epochs = sorted(set(snaps_ext.keys()) | set(snaps_bl.keys()))
    n_snap = len(epochs)

    # Global max error across both models and all epochs for a shared scale
    vmax_err = 0.0
    for ep in epochs:
        for snaps, pred in ((snaps_ext, pred_ext), (snaps_bl, pred_bl)):
            if ep not in snaps:
                continue
            u_ep  = pred.predict_from_state(snaps[ep], x_flat, t_flat)
            vmax_err = max(vmax_err, np.abs(u_ep - data["u_true_full"]).max())

    fig, axes = plt.subplots(n_snap, 4, figsize=(22, n_snap * 3.2))
    fig.patch.set_facecolor(BG)
    if n_snap == 1:
        axes = axes[np.newaxis, :]

    for row, epoch in enumerate(epochs):
        ax_ep, ax_ee, ax_bp, ax_be = axes[row]

        field_kw = dict(cfg=cfg, vmin=vmin_u, vmax=vmax_u,
                        time_shockwave=time_shockwave)
        err_kw   = dict(cfg=cfg, vmin=0, vmax=vmax_err, cmap=CMAP_ERROR,
                        cbar_label="|error|", time_shockwave=time_shockwave)

        # PINN ext
        if epoch in snaps_ext:
            u_ep_flat = pred_ext.predict_from_state(snaps_ext[epoch], x_flat, t_flat)
            _, _, grid_ep  = _to_grid(u_ep_flat, data)
            _, _, grid_err = _to_grid(np.abs(u_ep_flat - data["u_true_full"]), data)
            rmse_ep = Predictor.rmse(u_ep_flat, data["u_true_full"])
            _heatmap(ax_ep, t_vals, x_vals, grid_ep,
                     f"PINN ext — epoch {epoch}  RMSE={rmse_ep:.4f}", **field_kw)
            _heatmap(ax_ee, t_vals, x_vals, grid_err,
                     f"PINN ext — epoch {epoch}  |err| max={grid_err.max():.4f}", **err_kw)
        else:
            ax_ep.set_visible(False); ax_ee.set_visible(False)

        # PINN blind
        if epoch in snaps_bl:
            u_bp_flat = pred_bl.predict_from_state(snaps_bl[epoch], x_flat, t_flat)
            _, _, grid_bp  = _to_grid(u_bp_flat, data)
            _, _, grid_ber = _to_grid(np.abs(u_bp_flat - data["u_true_full"]), data)
            rmse_bp = Predictor.rmse(u_bp_flat, data["u_true_full"])
            _heatmap(ax_bp, t_vals, x_vals, grid_bp,
                     f"PINN blind — epoch {epoch}  RMSE={rmse_bp:.4f}", **field_kw)
            _heatmap(ax_be, t_vals, x_vals, grid_ber,
                     f"PINN blind — epoch {epoch}  |err| max={grid_ber.max():.4f}", **err_kw)
        else:
            ax_bp.set_visible(False); ax_be.set_visible(False)

    # Column headers
    for ax, label in zip(axes[0], ["PINN ext. — field",
                                    "PINN ext. — |error|",
                                    "PINN blind — field",
                                    "PINN blind — |error|"]):
        ax.set_title(f"{label}\n{ax.get_title()}", fontsize=8,
                     loc="left", pad=4, color="#444441")

    shock_note = f"  |  shock: t={time_shockwave:.4g} s  (red dashed)" if time_shockwave is not None else ""
    fig.suptitle(
        f"PINN ext. vs PINN blind — field evolution across epochs\n"
        f"ν={cfg.v}  situation={cfg.situation}  |  "
        f"λ_phys={cfg.lambda_phys}  λ_ic={cfg.lambda_ic}  |  "
        f"train window: t ≤ {cfg.t_train}  (white dashed){shock_note}",
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
    p.add_argument("--results",   type=str, default=None, help="Path to training_results.pt (written by train.py).")
    p.add_argument("--out_dir",   type=str, default=BurgerConfig.out_dir)
    p.add_argument("--n_slices",  type=int, default=9)
    p.add_argument("--no_show",   action="store_true")
    p.add_argument("--input_dir", type=str, default=BurgerConfig.out_dir, help="Directory where training_results.pt files are located. Used if --results not set.")
    p.add_argument("--situation", type=str, choices=["Step", "Gaussian", "N-wave"], default=BurgerConfig.situation, help="Burgers' initial condition scenario.(only used if results not set)")
    p.add_argument("--viscosity", type=float, default=BurgerConfig.v, help="Burgers' viscosity ν.(only used if results not set)")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.results is None:
        results_path = Path(f"{args.input_dir}/{args.situation}_"
                            f"{args.viscosity:.1e}_{BurgerConfig.suffix_results_pt}")
    else:
        results_path = Path(args.results)
    if not results_path.exists():
        print(f"[plot.py] ERROR: results file not found: {results_path}\n"
              "  Run  python train.py  first.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading  {results_path} …")
    raw = torch.load(results_path, map_location="cpu", weights_only=False)

    cfg: BurgerConfig = raw["config"]
    time_shockwave    = raw.get("time_shockwave")

    t_flat = raw["data"]["t_flattened_full"]
    x_flat = raw["data"]["x_flattened_full"]
    print(f"  Config  : ν={cfg.v}  situation={cfg.situation}  "
          f"x∈[{cfg.x_begin},{cfg.x_end}]  t∈[{cfg.t0},{cfg.t_extrap}]")
    print(f"  Grid    : {len(np.unique(t_flat))} × {len(np.unique(x_flat))} points")
    print(f"  Snaps   : ext={sorted(raw['snaps_pinn_ext_phys'].keys())}  "
          f"blind={sorted(raw['snaps_pinn_blind'].keys())}")
    if time_shockwave is not None:
        print(f"  Shock   : t = {time_shockwave:.6g} s")
    print()

    pred_ext = Predictor(cfg)
    pred_bl  = Predictor(cfg)

    bundle = {
        "cfg":                  cfg,
        "device_str":           raw["device_str"],
        "data":                 raw["data"],
        "hist_ml":              raw["hist_ml"],
        "hist_pinn_ext_phys":   raw["hist_pinn_ext_phys"],
        "hist_pinn_blind":      raw["hist_pinn_blind"],
        "snaps_ml":             raw["snaps_ml"],
        "snaps_pinn_ext_phys":  raw["snaps_pinn_ext_phys"],
        "snaps_pinn_blind":     raw["snaps_pinn_blind"],
        "u_ml_full":            raw["u_ml_full"],
        "u_pinn_ext_phys_full": raw["u_pinn_ext_phys_full"],
        "u_pinn_blind_full":    raw["u_pinn_blind_full"],
        "metrics":              raw["metrics"],
        "pred_ext":             pred_ext,
        "pred_bl":              pred_bl,
        "time_shockwave":       time_shockwave,
    }

    print("Generating Figure 1 — heatmap summary …")
    make_summary_figure(
        bundle,
        out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig1_summary.png")

    print(f"Generating Figure 2 — time-slice profiles ({args.n_slices} slices) …")
    make_slice_figure(
        bundle,
        out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig2_slices.png",
        n_slices=args.n_slices)

    print("Generating Figure 3 — PINN epoch heatmaps …")
    make_epoch_figure(
        bundle,
        out_dir / f"burger_{cfg.situation}_{cfg.v:.1e}_fig3_pinn_epochs.png")

    print(f"\n  All figures written to  {out_dir}/")
    if not args.no_show:
        plt.show()
    print("  Done.\n")


if __name__ == "__main__":
    main()