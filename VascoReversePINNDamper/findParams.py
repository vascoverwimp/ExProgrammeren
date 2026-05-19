from pathlib import Path
from model import DamperConfig
from train import train_and_save_both, evaluate_model
import matplotlib.pyplot as plt
import numpy as np

def damper_predictions():
    mass = 1.0  # kg
    dampings = np.linspace(0.5, 5.5, num=11)
    stiffnesses = np.linspace(0.5, 5.5, num=11) 

    plotting = []  # To store RMSE values for plotting later
    for damping in dampings:
        for stiffness in stiffnesses:
            cfg = DamperConfig(mass=mass, damping=damping, stiffness=stiffness)
            kwargs = dict(
                mass = mass,
                damping = damping,
                stiffness = stiffness,
            )
            true_w0 = cfg.omega_0
            true_zeta = cfg.zeta
            location = Path(f"{cfg.out_dir}/w0{cfg.omega_0:.1e}_zeta{cfg.zeta:.1e}_{cfg.suffix_results_pt}")

            if not location.exists():
                train_and_save_both(**kwargs)

            w0_hat_blind, zeta_hat_blind, w0_hat_ext_phys, zeta_hat_ext_phys = evaluate_model(true_w0, true_zeta)
            plotting.append((damping, stiffness, true_w0, true_zeta, w0_hat_blind, zeta_hat_blind, w0_hat_ext_phys, zeta_hat_ext_phys))
            print(f"Damping: {damping:.1e}, Stiffness: {stiffness:.1e}, Omega0 (true): {true_w0:.4f}, Zeta (true): {true_zeta:.4f}")
            print(f"  Omega0 Hat (blind): {w0_hat_blind:.4f}, Zeta Hat (blind): {zeta_hat_blind:.4f}")
            print(f"  Omega0 Hat (ext phys): {w0_hat_ext_phys:.4f}, Zeta Hat (ext phys): {zeta_hat_ext_phys:.4f}")
    # Plot the results (omega0)
    plt.figure(figsize=(8, 5))
    plt.scatter([omega0 for _, _, omega0, _, _, _, _, _ in plotting], [w0_hat_blind for _, _, _, _, w0_hat_blind, _, _, _ in plotting], marker='o', label='PINN (blind)')
    plt.scatter([omega0 for _, _, omega0, _, _, _, _, _ in plotting], [w0_hat_ext_phys for _, _, _, _, _, _, w0_hat_ext_phys, _ in plotting], marker='o', label='PINN (ext phys)')
    # add a reference line for the true viscosity
    plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='True Omega0')
    plt.xlabel('Omega0 (True)')
    plt.ylabel('Estimated Omega0 (Omega0 Hat)')
    plt.title(f'Omega0 Estimation vs True Omega0')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{cfg.out_dir}/omega0_predictions.png")

    # Plot the results (zeta)
    plt.figure(figsize=(8, 5))
    plt.scatter([zeta for _, _, _, zeta, _, _, _, _ in plotting], [zeta_hat_blind for _, _, _, _, _, zeta_hat_blind, _, _ in plotting], marker='o', label='PINN (blind)')
    plt.scatter([zeta for _, _, _, zeta, _, _, _, _ in plotting], [zeta_hat_ext_phys for _, _, _, _, _, _, _, zeta_hat_ext_phys in plotting], marker='o', label='PINN (ext phys)')
    # add a reference line for the true viscosity
    plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='True Zeta')
    plt.xlabel('Zeta (True)')
    plt.ylabel('Estimated Zeta (Zeta Hat)')
    plt.title(f'Zeta Estimation vs True Zeta')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{cfg.out_dir}/zeta_predictions.png")

    # Plot the results (zeta function of damping and stiffness, colormap)
    zeta_true = np.array([true_zeta for _, _, _, true_zeta, _, _, _, _ in plotting]).reshape(len(dampings), len(stiffnesses))
    zeta_blind = np.array([zeta_hat_blind for _, _, _, _, _, zeta_hat_blind, _, _ in plotting]).reshape(len(dampings), len(stiffnesses))
    zeta_ext_phys = np.array([zeta_hat_ext_phys for _, _, _, _, _, _, _, zeta_hat_ext_phys in plotting]).reshape(len(dampings), len(stiffnesses))
    
    # Use same scale for all colorbars
    vmin = min(zeta_true.min(), zeta_blind.min(), zeta_ext_phys.min())
    vmax = max(zeta_true.max(), zeta_blind.max(), zeta_ext_phys.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    im0 = axes[0].imshow(zeta_true, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('Stiffness')
    axes[0].set_ylabel('Damping')
    axes[0].set_title('True Zeta')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(zeta_blind, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('Stiffness')
    axes[1].set_ylabel('Damping')
    axes[1].set_title('PINN Blind Zeta')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(zeta_ext_phys, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_xlabel('Stiffness')
    axes[2].set_ylabel('Damping')
    axes[2].set_title('PINN Ext Phys Zeta')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/zeta_2d.png")


    # Plot the results (omega0 function of damping and stiffness, colormap)
    omega0_true = np.array([true_w0 for _, _, true_w0, _, _, _, _, _ in plotting]).reshape(len(dampings), len(stiffnesses))
    omega0_blind = np.array([w0_hat_blind for _, _, _, _, w0_hat_blind, _, _, _ in plotting]).reshape(len(dampings), len(stiffnesses))
    omega0_ext_phys = np.array([w0_hat_ext_phys for _, _, _, _, _, _, w0_hat_ext_phys, _ in plotting]).reshape(len(dampings), len(stiffnesses))
    
    # Use same scale for all colorbars
    vmin_w = min(omega0_true.min(), omega0_blind.min(), omega0_ext_phys.min())
    vmax_w = max(omega0_true.max(), omega0_blind.max(), omega0_ext_phys.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    im0 = axes[0].imshow(omega0_true, cmap='viridis', aspect='auto', origin='lower', vmin=vmin_w, vmax=vmax_w)
    axes[0].set_xlabel('Stiffness')
    axes[0].set_ylabel('Damping')
    axes[0].set_title('True Omega0')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(omega0_blind, cmap='viridis', aspect='auto', origin='lower', vmin=vmin_w, vmax=vmax_w)
    axes[1].set_xlabel('Stiffness')
    axes[1].set_ylabel('Damping')
    axes[1].set_title('PINN Blind Omega0')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(omega0_ext_phys, cmap='viridis', aspect='auto', origin='lower', vmin=vmin_w, vmax=vmax_w)
    axes[2].set_xlabel('Stiffness')
    axes[2].set_ylabel('Damping')
    axes[2].set_title('PINN Ext Phys Omega0')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/omega0_2d.png")

    plt.show()

if __name__ == "__main__":
    damper_predictions()

