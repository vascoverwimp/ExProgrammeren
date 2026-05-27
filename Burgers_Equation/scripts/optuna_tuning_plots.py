"""
Load the best hyperparameter-tuned model for each regime,
retrain it, and save plots to the correct folders.

Usage:
    python optuna_tuning_plots.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device, save_model, load_best_cfg
from plot import save_plots_from_file

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/optuna_tuning"
OUTPUT_PATH.mkdir(exist_ok=True)

ICS = ["Gauss", "N_wave", "Step_up", "Step_down", "Slope"]


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}\n")

    for ic in ICS:
        print(f"{'='*60}")
        print(f"Regime: {ic}")
        print(f"{'='*60}")

        # -- load best hyperparameters for this damped case -------------------
        best_params_path = OUTPUT_PATH / ic / "best_params.json"
        if not best_params_path.exists():
            print(f"  No best_params.json found for {ic} — skipping.")
            continue

        cfg = load_best_cfg(best_params_path)
        cfg.ic = ic
        print(f"  Best cfg : {cfg}")

        # -- output folder for this damped case -------------------------------
        ic_output = OUTPUT_PATH / ic
        ic_output.mkdir(parents=True, exist_ok=True)

        # -- train ---------------------------------------------------------
        data = generate_data(cfg)
        model = FCNet(cfg)
        history, snapshots, best_state = train(
            model=model,
            data=data,
            cfg=cfg,
            device=device,
            label=ic,
        )

        # -- save ----------------------------------------------------------
        save_model(best_state, history, snapshots, cfg, ic_output)
        print(f"  Model saved to {ic_output}")

        # -- plots ---------------------------------------------------------
        save_plots_from_file(ic_output)
        print(f"  Plots saved to {ic_output}\n")


if __name__ == "__main__":
    main()
