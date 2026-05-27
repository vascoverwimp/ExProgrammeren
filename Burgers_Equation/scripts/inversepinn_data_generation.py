"""
Script to continuously generate true-predicted nu pairs for the Burgers
inverse problem and save them to a CSV file.

Runs until KeyboardInterrupt. Each iteration randomises nu, trains an
InverseFCNet, and appends the result to a CSV.

Usage:
    python inversepinn_data_generation.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import csv
import numpy as np
from plot import plot_predicted_parameter_convergence, plot_solution_grid
from analytic import predict_shock_time
from config import Config
from data import generate_data
from model import InverseFCNet
from trainer import train
from utils import get_device

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/nu_estimation_2"
OUTPUT_PATH.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_PATH / "nu_pairs.csv"
CSV_FIELDS = ["run", "nu_true", "nu_pred", "abs_error",
              "rel_error_pct", "best_val_loss", "ic", "nu_class"]


def run_single(run_idx: int, device) -> dict | None:
    """
    Randomise nu, train an InverseFCNet, return result row.
    Returns None if training is skipped (e.g. shock too early).
    """
    nu_class = np.random.choice(["low", "high"])
    if nu_class == "low":
        nu = float(np.random.uniform(0.005, 0.05))
    else:
        nu = float(np.random.uniform(0.05, 0.5))
    ic = np.random.choice(["Gauss", "N_wave", "Step_up"], p=[1, 0, 0])

    print(f"-----Randomised viscosity : {nu:.4f}-----")
    print(f"-----Randomised initial condition : {ic}-----")
    cfg = Config(
        ic=ic,
        nu=nu,
        hidden=64,
        n_layers=6,
        lambda_phys=1e1,
        lambda_ic=1e2,
        lr=0.003,
        scheduler_gamma=0.6,
        scheduler_step=3000,
        patience_threshold=1e-5,
        use_data=True
    )

    time_to_shock = predict_shock_time(cfg)
    if 0.1 <= time_to_shock <= 5:
        print(
            f"  [run {run_idx}] Shock at {time_to_shock:.3f}s — adjusting domain.")  # noqa:E501
        cfg.t_dom = time_to_shock + 2.0
        cfg.t_extrap = time_to_shock + 4.0
    elif time_to_shock < 0.1:
        print(
            f"  [run {run_idx}] Shock too early ({time_to_shock:.3f}s) — setting default times.")  # noqa:E501

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

    nu_pred = float(model.nu_hat.item())
    abs_error = abs(nu_pred - nu)
    rel_error = abs_error / nu * 100

    best_val = min(history["loss_val"])

    print(f"  [run {run_idx}] nu_true={nu:.4f}  nu_pred={nu_pred:.4f}  "
          f"err={abs_error:.4f} ({rel_error:.1f}%)  val={best_val:.5f}  ic={cfg.ic}")  # noqa:E501

    model.to('cpu')
    plot_predicted_parameter_convergence(
        history=history,
        cfg=cfg,
        output_path=f"{OUTPUT_PATH}/convergence_nu_{ic}_{nu:.4f}.png",
        show=False
    )

    plot_solution_grid(
        model=model,
        data=data,
        cfg=cfg,
        snapshots=snapshots,
        u_grid=data["u_grid"],
        t_arr=data["t_arr"],
        output_path=f"{OUTPUT_PATH}/solution_nu_{ic}_{nu:.4f}.png",
        show=False
    )

    return {
        "run":             run_idx,
        "nu_true":         nu,
        "nu_pred":         nu_pred,
        "abs_error":       abs_error,
        "rel_error_pct":   rel_error,
        "best_val_loss":   best_val,
        "ic":              ic,
        "nu_class":        nu_class
    }


def main():
    """Main loop."""
    device = get_device()
    print(f"Device      : {device}")
    print(f"Output CSV  : {CSV_PATH}")
    print("Press Ctrl+C to stop.\n")

    write_header = not CSV_PATH.exists()
    run_idx = 0

    try:
        with open(CSV_PATH, "a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()

            while True:
                run_idx += 1
                print(f"--- Run {run_idx} ---")

                try:
                    row = run_single(run_idx, device)
                except (RuntimeError, ValueError, OSError) as e:
                    print(f"  [run {run_idx}] failed: {e} — skipping.")
                    continue

                if row is not None:
                    writer.writerow(row)
                    f.flush()   # write immediately so data isn't lost on interrupt  # noqa:E501

    except KeyboardInterrupt:
        print(f"\nStopped after {run_idx} runs. Results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
