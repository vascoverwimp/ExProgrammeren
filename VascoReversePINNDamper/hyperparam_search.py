import matplotlib.pyplot as plt
from pathlib import Path
from model import DamperConfig, FCNet
from train import evaluate_blind, train_model, generate_data, get_device
import numpy as np


def search_learning_rate_param():
    # Define a range of learning rates to search over
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]  # Search over a range of learning rates
    learning_rates = 10**np.linspace(-3.5, -0.5, num=25) # Search over a range of learning rates from 10^-3.5 to 10^-1.5: [3.16e-4, 5.62e-4, 1e-3, 1.78e-3, 3.16e-3, 5.62e-3, 1e-2, 1.78e-2, 3.16e-2]
    plot_RMSE = []  # To store RMSE values for plotting later
    plot_w0_diff = []  # To store w0 differences for plotting later
    plot_zeta_diff = []  # To store zeta differences for plotting later
    device = get_device()
    data = generate_data(DamperConfig(), device=device)  # Generate data once, can be reused for all learning rates
    default_cfg = DamperConfig()
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / f"w0{default_cfg.omega_0:.1e}_zeta{default_cfg.zeta:.1e}_{default_cfg.suffix_ckpt_pinn_blind}"
    best_rmse = float('inf')
    best_w0_diff = float('inf')
    best_zeta_diff = float('inf')

    best_w0_hat = None
    best_zeta_hat = None
    best_lr_rmse = None
    best_lr_w0 = None
    best_lr_zeta = None
    for lr_param in learning_rates:
        cfg = DamperConfig(lr_param = lr_param)
        

        print(f"Testing learning rate: {lr_param}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data, cfg=cfg, device=device, use_physics=True, extrapolated_physics=False,         label       = "PINN (blind)",
        ckpt_path   = ckpt_pinn_blind)
        rmse, w0_hat, zeta_hat = evaluate_blind(cfg.omega_0, cfg.zeta)
        print(f"RMSE for learning rate {lr_param}: {rmse:.4f}")
        plot_RMSE.append((lr_param, rmse))
        plot_w0_diff.append((lr_param, abs(w0_hat - cfg.omega_0)))
        plot_zeta_diff.append((lr_param, abs(zeta_hat - cfg.zeta)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lr_rmse = lr_param
        if abs(w0_hat - cfg.omega_0) < best_w0_diff:
            best_w0_diff = abs(w0_hat - cfg.omega_0)
            best_w0_hat = w0_hat
            best_lr_w0 = lr_param
        if abs(zeta_hat - cfg.zeta) < best_zeta_diff:
            best_zeta_diff = abs(zeta_hat - cfg.zeta)
            best_zeta_hat = zeta_hat
            best_lr_zeta = lr_param

    print(f"Best learning rate RMSE: {best_lr_rmse} with RMSE: {best_rmse}")
    print(f"Best learning rate w0: {best_lr_w0} with difference: {best_w0_diff} (from w0_hat: {best_w0_hat})")
    print(f"Best learning rate zeta: {best_lr_zeta} with difference: {best_zeta_diff} (from zeta_hat: {best_zeta_hat})")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([lr for lr, rmse in plot_RMSE], [rmse for lr, rmse in plot_RMSE], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE')
    plt.title('Learning Rate vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.savefig(cfg.out_dir+"/learning_rate_vs_rmse.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot([lr for lr, w0_diff in plot_w0_diff], [w0_diff for lr, w0_diff in plot_w0_diff], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('|w0 - w0_true|')
    plt.title('Learning Rate vs w0 Difference for PINN (blind)')
    plt.grid(True)
    plt.savefig(cfg.out_dir + "/learning_rate_vs_w0_diff.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot([lr for lr, zeta_diff in plot_zeta_diff], [zeta_diff for lr, zeta_diff in plot_zeta_diff], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('|zeta - zeta_true|')
    plt.title('Learning Rate vs zeta Difference for PINN (blind)')
    plt.grid(True)
    plt.savefig(cfg.out_dir+"/learning_rate_vs_zeta_diff.png")
    plt.show()



if __name__ == "__main__":
    search_learning_rate_param()
