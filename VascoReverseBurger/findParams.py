from model import BurgerConfig
from train import train_and_save_both, evaluate_model
import matplotlib.pyplot as plt
import numpy as np

def viscosity_predictions(situation:str):
    viscosities = np.logspace(-5, -1, num=20)  # Search over a range of viscosities from 10^-3 to 10^-1
    plotting = []  # To store RMSE values for plotting later
    for v in viscosities:
        cfg = BurgerConfig(situation=situation, v=v)
        kwargs = dict(
            viscosity = v,
            situation = situation,
        )
        train_and_save_both(**kwargs)
        nu_hat_blind, nu_hat_ext_phys = evaluate_model(situation, v)
        plotting.append((v, nu_hat_blind, nu_hat_ext_phys))
        print(f"Viscosity: {v:.1e}, Nu Hat (blind): {nu_hat_blind:.4f}, Nu Hat (ext phys): {nu_hat_ext_phys:.4f}")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([v for v, _, _ in plotting], [nu_hat_blind for _, nu_hat_blind, _ in plotting], marker='o', label='PINN (blind)')
    plt.plot([v for v, _, _ in plotting], [nu_hat_ext_phys for _, _, nu_hat_ext_phys in plotting], marker='o', label='PINN (ext phys)')
    plt.xscale('log')
    plt.yscale('log')
    # add a reference line for the true viscosity => first diagonal
    plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='True Viscosity')
    plt.xlabel('Viscosity')
    plt.ylabel('Estimated Viscosity (Nu Hat)')
    plt.title(f'Viscosity Estimation vs True Viscosity for {situation}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"viscosity_predictions_{situation}.png")
    plt.show()

if __name__ == "__main__":
    situation = "Step"
    viscosity_predictions(situation)