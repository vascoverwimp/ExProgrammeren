"""
Script to continuously generate true-predicted zeta and omega_0 pairs for the
Burgers equation inverse problem and save them to a CSV file.

Runs until KeyboardInterrupt. Each iteration randomises nu, trains an
InverseFCNet, and appends the result to a CSV.

Usage:
    python inversepinn_data_generation.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import csv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np

from config import Config
from data import generate_data
from model import InverseFCNet
from plot import (
    plot_epoch_figure,
    plot_predicted_parameter_convergence,
)
from trainer import train
from utils import convert_to_mck, get_device


SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/parameter_estimation"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_PATH / "parameter_pairs.csv"

CSV_FIELDS = [
    "run",
    "zeta_true",
    "zeta_pred",
    "omega_0_true",
    "omega_0_pred",
    "abs_error_zeta",
    "rel_error_zeta_pct",
    "abs_error_omega_0",
    "rel_error_omega_0_pct",
    "best_val_loss",
    "damped_case",
]


def run_single(run_idx: int, device) -> dict | None:
    """Randomise parameters, train model, return result row."""

    damped_case = np.random.choice(
        ["underdamped", "overdamped", "critically_damped"],
        p=[0.45, 0.45, 0.1],
    )

    if damped_case == "underdamped":
        zeta = np.random.uniform(0.02, 0.98)
    elif damped_case == "overdamped":
        zeta = np.random.uniform(1.02, 2.0)
    else:
        zeta = 1.0

    omega_0 = np.random.uniform(0.3, 4)

    m, c, k = convert_to_mck(zeta, omega_0)

    print(f"----- Randomised zeta: {zeta:.4f} -----")
    print(f"----- Randomised omega_0: {omega_0:.4f} -----")

    cfg = Config(
        n_layers=5,
        hidden=96,
        lambda_phys=0.5,
        lambda_ic=10,
        lr=0.002,
        lr_inverse=0.015,
        scheduler_gamma=0.8,
        scheduler_step=3000,
        patience=3000,
    )

    cfg.m = m
    cfg.c = c
    cfg.k = k

    data = generate_data(cfg)
    model = InverseFCNet(cfg)

    history, snapshots, _ = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label=f"run {run_idx}",
        verbatim=False,
    )

    zeta_pred = float(model.zeta_hat.item())
    omega_0_pred = float(model.omega_0_hat.item())

    abs_error_zeta = abs(zeta_pred - zeta)
    rel_error_zeta = abs_error_zeta / zeta * 100

    abs_error_omega_0 = abs(omega_0_pred - omega_0)
    rel_error_omega_0 = abs_error_omega_0 / omega_0 * 100

    best_val = min(history["loss_val"])

    print(
        f"  [run {run_idx}] "
        f"zeta_pred={zeta_pred:.4f}  "
        f"omega_0_pred={omega_0_pred:.4f}  "
        f"zeta err={abs_error_zeta:.4f} ({rel_error_zeta:.1f}%)  "
        f"omega_0 err={abs_error_omega_0:.4f} ({rel_error_omega_0:.1f}%)  "
        f"val={best_val:.5f}"
    )

    model.to("cpu")

    t_plot = np.linspace(0.0, cfg.t_extrap, 500)

    plot_predicted_parameter_convergence(
        history=history,
        cfg=cfg,
        output_path=(
            f"{OUTPUT_PATH}/convergence_zeta_{zeta:.4f}_"
            f"omega_0_{omega_0:.4f}.png"
        ),
        show=False,
    )

    plot_epoch_figure(
        model=model,
        data=data,
        cfg=cfg,
        snapshots=snapshots,
        t_plot_full=t_plot,
        output_path=(
            f"{OUTPUT_PATH}/epoch_snapshots_zeta_{zeta:.4f}_"
            f"omega_0_{omega_0:.4f}.png"
        ),
        show=False,
    )

    return {
        "run": run_idx,
        "zeta_true": zeta,
        "zeta_pred": zeta_pred,
        "omega_0_true": omega_0,
        "omega_0_pred": omega_0_pred,
        "abs_error_zeta": abs_error_zeta,
        "rel_error_zeta_pct": rel_error_zeta,
        "abs_error_omega_0": abs_error_omega_0,
        "rel_error_omega_0_pct": rel_error_omega_0,
        "best_val_loss": best_val,
        "damped_case": damped_case,
    }


def main() -> None:
    """Main loop."""
    device = get_device()

    print(f"Device     : {device}")
    print(f"Output CSV : {CSV_PATH}")
    print("Press Ctrl+C to stop.\n")

    write_header = not CSV_PATH.exists()
    run_idx = 0

    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

            if write_header:
                writer.writeheader()

            while True:
                run_idx += 1
                print(f"--- Run {run_idx} ---")

                try:
                    row = run_single(run_idx, device)
                except (RuntimeError, ValueError) as e:
                    # Known/expected failures from training or data generation:
                    print(f"  [run {run_idx}] failed: {e} — skipping.")
                    continue

                if row is not None:
                    writer.writerow(row)
                    f.flush()

    except KeyboardInterrupt:
        print(
            f"\nStopped after {run_idx} runs. "
            f"Results saved to {CSV_PATH}"
        )


if __name__ == "__main__":
    main()
