"""
Solving the damped harmonic oscillation
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
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device, save_model
from plot import save_plots_from_file

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    """Main loop."""
    device = get_device()
    print(f"Device : {device}")

    # -- Underdamped ------------------------------------------------
    output_path = SCRIPT_DIR / "outputs/pinn_blindextrap"  # noqa: E501
    output_path.mkdir(exist_ok=True)
    cfg = Config(
        m=1.0,
        c=0.5,
        k=4.0,
        train_extrap=False
    )

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


if __name__ == "__main__":
    main()
