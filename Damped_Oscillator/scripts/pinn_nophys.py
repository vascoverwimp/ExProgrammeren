"""
Solving the damped harmonic oscillation
without any physics loss terms (traditional dense network).

Usage:
    python pinn_nophys.py

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
from utils import get_device
from plot import save_model_plots

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def main():
    device = get_device()
    print(f"Device : {device}")

    # -- Underdamped ------------------------------------------------
    cfg = Config(
        m=1.0,
        c=0.5,
        k=4.0,
        use_physics=False,
        use_ic=False
    )
    data = generate_data(cfg)

    # -- train standard model ------------------------------------------------
    output_path = SCRIPT_DIR / "outputs/pinn_nophys"
    output_path.mkdir(exist_ok=True)
    model = FCNet(cfg)
    history, snapshots, _ = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
    )

    # -- make plots ------------------------------------------------
    save_model_plots(
        model=model,
        history=history,
        snapshots=snapshots,
        data=data,
        cfg=cfg,
        device=device,
        inverse=False,
        output_path=output_path,
        model_label="ML",
    )


if __name__ == "__main__":
    main()
