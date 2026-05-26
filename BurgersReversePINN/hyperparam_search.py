"""Hyperparameter search utilities for reverse PINN model training.

This module provides functions to search for optimal hyperparameters (learning rate)
for training reverse Physics-Informed Neural Networks (PINNs) that learn the 
viscosity parameter along with solving the Burgers equation.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from BurgersReversePINN.model import BurgerConfig, FCNet
from BurgersReversePINN.train import evaluate_blind, train_model, generate_data, get_device
import numpy as np


def search_learning_rate_param():
    """Search for optimal learning rate for reverse PINN training.

    Systematically tests a range of learning rates and evaluates model performance
    using both RMSE and viscosity estimation error (|v_hat - v_true|). Generates
    plots comparing learning rates against both metrics.

    Returns:
        None. Prints best learning rates for RMSE and viscosity accuracy, and
        saves two plots to the output directory showing the trade-offs.
    """
    # Define a range of learning rates to search over
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]  # Search over a range of learning rates
    learning_rates = np.logspace(-1.8, -4.2, num=25)
    plotting_list = []  # To store RMSE values for plotting later
    plot_v_diff = []  # To store nu differences for plotting later
    device = get_device()
    # Generate data once, can be reused for all learning rates
    data = generate_data(BurgerConfig(), device=device)
    default_cfg = BurgerConfig()
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / \
        f"{default_cfg.situation}_{default_cfg.v:.1e}_{default_cfg.suffix_ckpt_pinn_blind}"
    best_rmse = float('inf')
    best_v_diff = float('inf')

    best_v_hat = None
    best_lr_rmse = None
    best_lr_v = None
    for lr_param in learning_rates:
        cfg = BurgerConfig(lr_param=lr_param)
        print(f"Testing learning rate: {lr_param}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data, cfg=cfg,
                    device=device, use_physics=True, extrapolated_physics=False,
                    label="PINN (blind)", ckpt_path=ckpt_pinn_blind)
        rmse, v_hat = evaluate_blind(cfg.situation, cfg.v)
        print(f"RMSE for learning rate {lr_param}: {rmse:.4f}")
        plotting_list.append((lr_param, rmse))
        plot_v_diff.append((lr_param, abs(v_hat - cfg.v)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lr_rmse = lr_param
        if abs(v_hat - cfg.v) < best_v_diff:
            best_v_diff = abs(v_hat - cfg.v)
            best_v_hat = v_hat
            best_lr_v = lr_param

    print(f"Best learning rate RMSE: {best_lr_rmse} with RMSE: {best_rmse}")
    print(f"Best learning rate v: {best_lr_v} with difference:"
          f"{best_v_diff} (from v_hat: {best_v_hat})")
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
    plt.plot([lr for lr, v_diff in plot_v_diff], [
             v_diff for lr, v_diff in plot_v_diff], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('|v - v_true|')
    plt.title('Learning Rate vs viscosity Difference for PINN (blind)')
    plt.grid(True)
    plt.savefig(cfg.out_dir+"/learning_rate_vs_v_diff.png")


if __name__ == "__main__":
    search_learning_rate_param()
