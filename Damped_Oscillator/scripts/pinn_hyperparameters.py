"""
Solving the damped harmonic oscillation
with different hyperparameters.

Usage:
    python pinn_hyperparameters.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device, save_model
from plot import save_plots_from_file

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/pinn_hyperparameters/"


def train_model(cfg, label, device):
    """Train a model with one hyperparameter changed."""
    output_path = OUTPUT_PATH / label
    output_path.mkdir(exist_ok=True, parents=True)

    # -- train model ------------------------------------------------
    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label=label,
    )

    # -- save model ------------------------------------------------
    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    # -- make plots ------------------------------------------------
    save_plots_from_file(output_path)


def main():
    """Main loop."""
    device = get_device()
    print(f"Device : {device}")

    cfg_list = [
        Config(
            n_layers=2,
            hidden=4
        ),
        Config(
            lambda_phys=1e2,
        ),
        Config(
            lambda_phys=1e2,
            train_extrap=False
        ),
        Config(
            lambda_phys=1e0,
            use_data=False
        ),
        Config(
            lambda_ic=0,
        ),
        Config(
            lr=0.05
        ),
        Config(
            lr=1e-3,
            scheduler_gamma=0.3,
            scheduler_step=1000
        )
    ]

    label_list = [
        "low_complexity",
        "large_phys",
        "large_phys_no_extrap",
        "large_phys_no_data",
        "no_ic",
        "high_lr",
        "low_lr"
    ]

    for cfg, label in zip(cfg_list, label_list):
        train_model(cfg, label, device)


if __name__ == "__main__":
    main()
