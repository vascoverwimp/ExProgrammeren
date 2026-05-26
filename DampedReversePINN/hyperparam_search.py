from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from DampedReversePINN.model import DamperConfig, FCNet
from DampedReversePINN.train import evaluate_blind, train_model, generate_data, get_device


def search_learning_rate_param():
    # Define a range of learning rates to search over
    learning_rates = np.logspace(-5, 0, num=6)
    # Search over a range of learning rates
    learning_rates = np.logspace(-3.5, -0.5, num=31)

    plotting_list = []  # To store RMSE values for plotting later
    plot_w0_diff = []  # To store w0 differences for plotting later
    plot_zeta_diff = []  # To store zeta differences for plotting later
    device = get_device()
    # Generate data once, can be reused for all learning rates
    data = generate_data(DamperConfig(), device=device)
    default_cfg = DamperConfig()
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / (f"w0{default_cfg.omega_0:.1e}_zeta{default_cfg.zeta:.1e}"
                                 f"_{default_cfg.suffix_ckpt_pinn_blind}")
    best_rmse = float('inf')
    best_w0_diff = float('inf')
    best_zeta_diff = float('inf')

    best_w0_hat = None
    best_zeta_hat = None
    best_lr_rmse = None
    best_lr_w0 = None
    best_lr_zeta = None
    for lr_param in learning_rates:
        cfg = DamperConfig(lr_param=lr_param)

        print(f"Testing learning rate: {lr_param}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data,
                    cfg=cfg, device=device, use_physics=True,
                    extrapolated_physics=False, label="PINN (blind)",
                    ckpt_path=ckpt_pinn_blind)
        rmse, w0_hat, zeta_hat = evaluate_blind(cfg.omega_0, cfg.zeta)
        print(f"RMSE for learning rate {lr_param}: {rmse:.4f}\n")
        plotting_list.append((lr_param, rmse))
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
    print(
        f"Best learning rate w0: {best_lr_w0} with difference:"
        f"{best_w0_diff} (from w0_hat: {best_w0_hat})")
    print(
        f"Best learning rate zeta: {best_lr_zeta} with difference:"
        f"{best_zeta_diff} (from zeta_hat: {best_zeta_hat})")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([lr for lr, rmse in plotting_list], [
             rmse for lr, rmse in plotting_list], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE')
    plt.title('Learning Rate vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.savefig(cfg.out_dir+"/learning_rate_vs_rmse.png")

    plt.figure(figsize=(8, 5))
    plt.plot([lr for lr, w0_diff in plot_w0_diff], [
             w0_diff for lr, w0_diff in plot_w0_diff], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('|w0 - w0_true|')
    plt.title('Learning Rate vs w0 Difference for PINN (blind)')
    plt.grid(True)
    plt.savefig(cfg.out_dir + "/learning_rate_vs_w0_diff.png")

    plt.figure(figsize=(8, 5))
    plt.plot([lr for lr, zeta_diff in plot_zeta_diff], [
             zeta_diff for lr, zeta_diff in plot_zeta_diff], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('|zeta - zeta_true|')
    plt.title('Learning Rate vs zeta Difference for PINN (blind)')
    plt.grid(True)
    plt.savefig(cfg.out_dir+"/learning_rate_vs_zeta_diff.png")

    plt.show()


if __name__ == "__main__":
    search_learning_rate_param()
