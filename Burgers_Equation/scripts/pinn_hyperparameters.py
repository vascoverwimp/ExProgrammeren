"""
Solving the Burger's equation
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
from analytic import predict_shock_time
from data import generate_data
from model import FCNet
from trainer import train
from plot import save_plots_from_file
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)


def train_model(cfg: Config, output_path: str) -> None:
    """Train a model."""
    device = get_device()
    print(f"Device : {device}")

    print(cfg)

    time_to_shock = predict_shock_time(cfg)
    if time_to_shock <= 5 and time_to_shock >= 0.1:
        print("--------------------------------------------------"
              "\n"
              f"Shockwave predicted to occur at {time_to_shock:.3f}s."
              "\n"
              "Setting training domain to end at shock time."
              "\n"
              "--------------------------------------------------")
        cfg.t_shock = time_to_shock
        cfg.t_dom = time_to_shock + 2.0
        cfg.t_extrap = time_to_shock + 4.0
    else:
        print("--------------------------------------------------"
              "\n"
              f"Shockwave predicted to occur at {time_to_shock:.3f}s."
              "\n"
              "This time is not suitable for training. Returning None"
              "\n"
              f"(ic={cfg.ic}, nu={cfg.nu})"
              "--------------------------------------------------")
        return None

    # -- train model ------------------------------------------------
    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
    )

    # -- save model ------------------------------------------------
    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    # -- make plots ------------------------------------------------
    save_plots_from_file(output_path)


def main() -> None:
    """Main loop."""
    cfg_list = [
        Config(
            n_layers=2,
            hidden=4
        ),
        Config(
            lambda_phys=1e2,
        ),
        Config(
            use_bc=False
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
        "no_bc",
        "high_lr",
        "low_lr"
    ]

    for cfg, label in zip(cfg_list, label_list):
        train_model(cfg, OUTPUT_PATH / f"Gauss_nu_0.05_{label}")


if __name__ == "__main__":
    main()
