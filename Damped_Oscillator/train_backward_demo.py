"""
Demo script for training a PINN to solve the inverse 
problem of the damped harmonic oscillator with parameter estimation.

Usage:
    python train_forward.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import numpy as np
from pathlib import Path
from config import Config
from data import generate_data
from model import InverseFCNet
from trainer import train
from utils import get_device, save_model, convert_to_mck, set_seed
from plot import save_plots_from_file


def main():
    device = get_device()
    print(f"Device : {device}")
    cfg = Config()

    ########## PARAMETER RADOMISATION ##########

    # ---------- Underdamped ----------

    set_seed(13)
    zeta = np.random.uniform(0.02, 0.2)
    omega_0 = np.random.uniform(1, 5)
    output_path = Path.cwd() / "Damped_Oscillator/outputs/demo_backward_model_underdamped_bis"
    output_path.mkdir(exist_ok=True)

    # ---------- Critcally damped ----------

    # set_seed(13)
    # zeta = 1
    # omega_0 = np.random.uniform(0.5, 5)
    # output_path = Path.cwd() / "Damped_Oscillator/outputs/demo_backward_model_criticallydamped"
    # output_path.mkdir(exist_ok=True)

    # ---------- Overdamped ----------

    # set_seed(16)
    # zeta = np.random.uniform(1.5, 2)
    # omega_0 = np.random.uniform(2, 3)
    # output_path = Path.cwd() / "Damped_Oscillator/outputs/demo_backward_model_overdamped"
    # output_path.mkdir(exist_ok=True)

    print("------------------------------------------------"
          f"\nRandomised zeta    : {zeta:.4f}"
          f"\nRandomised omega_0 : {omega_0:.4f}"
          "\n------------------------------------------------")
    
    m, c, k = convert_to_mck(zeta, omega_0)

    cfg = Config(
        m=m,
        c=c,
        k=k
    )

    data     = generate_data(cfg)
    model    = InverseFCNet(cfg)
    history, snapshots, best_state = train(
        model       = model,
        data        = data,
        cfg         = cfg,
        device      = device,
        label       = "Model",
    )

    # -- save model ------------------------------------------------
    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    # -- make plots ------------------------------------------------
    save_plots_from_file(output_path)

if __name__ == "__main__":
    main()