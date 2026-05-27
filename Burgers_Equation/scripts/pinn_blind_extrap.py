"""
Solving the Burger's equation
without collocation points in the extrapolated region.

Usage:
    python pinn_blind_extrap.py

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
    ics = ["Gauss"]
    nu_values = [0.05]
    for ic in ics:
        for nu in nu_values:
            cfg = Config(
                ic=ic,
                nu=nu,
                train_extrap=False
            )
            output_path = OUTPUT_PATH / f"{ic}_nu_{nu}_extrapblind"

            train_model(cfg, output_path)


if __name__ == "__main__":
    main()
