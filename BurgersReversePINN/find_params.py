"""Hyperparameter search and viscosity prediction for reverse PINN.

This module searches over a range of viscosity values and trains reverse PINN
models to estimate the true viscosity from synthetic observations. Used to
evaluate model performance across different viscosity regimes.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from model import BurgerConfig
from train import train_and_save_both, evaluate_model


def viscosity_predictions(situation: str):
    """Search over viscosity values and train reverse PINN models to estimate viscosity.

    Trains PINN models for multiple viscosity values and evaluates how well
    the reverse PINN can estimate the true viscosity. Generates plots comparing
    estimated vs true viscosity for both blind and extended physics models.

    Args:
        situation: Initial condition type ('Gaussian', 'Step', or 'N-wave').

    Returns:
        None. Saves viscosity prediction plots to the output directory and
        prints viscosity estimates for each tested value.
    """
    viscosities1 = np.logspace(-5, 1, num=7)
    viscosities2 = np.linspace(0.2, 0.9, num=8)
    viscosities3 = np.linspace(0.02, 0.09, num=8)
    viscosities4 = np.linspace(2, 9, num=8)
    viscosities = np.concatenate(
        (viscosities1, viscosities2, viscosities3, viscosities4))
    plotting = []  # To store RMSE values for plotting later
    for v in viscosities:
        cfg = BurgerConfig(situation=situation, v=v)
        kwargs = dict(
            viscosity=v,
            situation=situation,
        )
        location = Path(
            f"{cfg.out_dir}/{cfg.situation}_{cfg.v:.1e}_{cfg.suffix_results_pt}")
        if not location.exists():
            train_and_save_both(**kwargs)

        nu_hat_blind, nu_hat_ext_phys = evaluate_model(situation, v)
        plotting.append((v, nu_hat_blind, nu_hat_ext_phys))
        print(f"Viscosity: {v:.1e}, Nu Hat (blind): {nu_hat_blind:.4f},"
              f" Nu Hat (ext phys): {nu_hat_ext_phys:.4f}")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.scatter([v for v, _, _ in plotting], [nu_hat_blind for _,
                nu_hat_blind, _ in plotting], marker='o', label='PINN (blind)')
    plt.scatter([v for v, _, _ in plotting], [nu_hat_ext_phys for _, _,
                nu_hat_ext_phys in plotting], marker='o', label='PINN (ext phys)')
    # add a reference line for the true viscosity => first diagonal
    plt.axline((0, 0), slope=1, color='gray',
               linestyle='--', label='True Viscosity')
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Viscosity')
    plt.ylabel('Estimated Viscosity')
    plt.title(
        f'Viscosity Estimation vs True Viscosity for {situation} initial condition')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{cfg.out_dir}/viscosity_predictions_{situation}.png")


if __name__ == "__main__":
    situations = ["Gaussian", "Step", "N-wave"]
    for situation in situations:
        viscosity_predictions(situation)
