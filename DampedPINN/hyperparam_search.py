"""Hyperparameter search utilities for damped oscillator PINN.

This module provides functions to search for optimal hyperparameters including
learning rate, loss weights, number of collocation points, and training data size.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from model import DamperConfig, FCNet
from train import evaluate_blind, train_model, generate_data, get_device


def search_learning_rate():
    """Search for optimal learning rate for PINN training.

    Systematically tests a range of learning rates and evaluates model performance
    using RMSE metric. Generates plot comparing learning rates against RMSE.

    Returns:
        None. Prints best learning rate and RMSE, saves plot to output directory.
    """
    # Define a range of learning rates to search over
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]  # Search over a range of learning rates
    # Search over a range of learning rates from 10^-3.5 to 10^-1.5:
    # [3.16e-4, 5.62e-4, 1e-3, 1.78e-3, 3.16e-3, 5.62e-3, 1e-2, 1.78e-2, 3.16e-2]
    learning_rates = np.logspace(-3.5, -1.5, num=9)
    plotting_list = []  # To store RMSE values for plotting later
    device = get_device()
    # Generate data once, can be reused for all learning rates
    data = generate_data(DamperConfig(), device=device)
    default_cfg = DamperConfig()
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / (f"w0{default_cfg.omega_0:.1e}_"
                                 f"zeta{default_cfg.zeta:.1e}_{default_cfg.suffix_ckpt_pinn_blind}")
    best_rmse = float('inf')
    best_lr = None
    for lr in learning_rates:
        cfg = DamperConfig(lr=lr)

        print(f"Testing learning rate: {lr}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data,
                    cfg=cfg, device=device, use_physics=True,
                    extrapolated_physics=False, label="PINN (blind)",
                    ckpt_path=ckpt_pinn_blind)
        rmse = evaluate_blind(cfg.omega_0, cfg.zeta)
        print(f"RMSE for learning rate {lr}: {rmse:.4f}")
        plotting_list.append((lr, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lr = lr

    print(f"Best learning rate: {best_lr} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([lr for lr, rmse in plotting_list], [
             rmse for lr, rmse in plotting_list], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE')
    plt.title('Learning Rate vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.savefig(f"{cfg.out_dir}/learning_rate_smooth")


def search_weights():
    """Search for optimal initial condition and physics loss weights.

    Performs a grid search over combinations of lambda_ic (initial condition weight)
    and lambda_phys (physics weight). Evaluates each combination and generates
    a 2D scatter plot showing RMSE across the weight space.

    Returns:
        None. Prints best weights and RMSE, saves scatter plot to output directory.
    """
    ic_weights = np.logspace(1, 2, num=9)
    phys_weights = np.logspace(-1, 0, num=9)
    plotting_list = []  # To store RMSE values for plotting later
    device = get_device()
    default_cfg = DamperConfig()
    # Generate data once, can be reused for all weight combinations
    data = generate_data(default_cfg, device=device)
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / (f"w0{default_cfg.omega_0:.1e}_"
                                 f"zeta{default_cfg.zeta:.1e}_{default_cfg.suffix_ckpt_pinn_blind}")
    best_rmse = float('inf')
    best_ic = None
    best_phys = None
    for ic_weight in ic_weights:
        for phys_weight in phys_weights:
            cfg = DamperConfig(lambda_ic=ic_weight, lambda_phys=phys_weight)
            print(
                f"Testing initial condition weight: {ic_weight}, physics weight: {phys_weight}")
            model_pinn_blind = FCNet.from_config(cfg)
            train_model(model_pinn_blind, data=data,
                        cfg=cfg, device=device, use_physics=True,
                        extrapolated_physics=False, label="PINN (blind)",
                        ckpt_path=ckpt_pinn_blind)
            rmse = evaluate_blind(cfg.omega_0, cfg.zeta)
            print(
                f"RMSE for ic_weight {ic_weight}, phys_weight {phys_weight}: {rmse:.4f}")
            plotting_list.append((ic_weight, phys_weight, rmse))
            if rmse < best_rmse:
                best_rmse = rmse
                best_ic = ic_weight
                best_phys = phys_weight
    print(
        f"Best initial condition weight: {best_ic}"
        f"best physics weight: {best_phys} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.scatter([ic_weight for ic_weight, phys_weight, rmse in plotting_list],
                [phys_weight for ic_weight, phys_weight, rmse in plotting_list],
                c=[rmse for ic_weight, phys_weight, rmse in plotting_list],
                cmap='viridis', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Initial Condition Weight (lambda_ic)')
    plt.ylabel('Physics Weight (lambda_phys)')
    plt.title('Weight Search for PINN (blind)')
    plt.colorbar(label='RMSE')
    plt.grid(True)
    plt.savefig(f"{cfg.out_dir}/weights_smooth")


def search_num_collocations():
    """Search for optimal number of collocation points.

    Tests a range of collocation point counts and evaluates model performance
    via RMSE metric. Generates a plot showing how RMSE varies with the number
    of collocation points used during training.

    Returns:
        None. Prints best number of collocation points and RMSE, saves plot.
    """
    # Search over a range of numbers of collocation points
    num_collocations = np.linspace(10, 500, num=50, dtype=int)
    plotting_list = []  # To store RMSE values for plotting later
    device = get_device()
    default_cfg = DamperConfig()
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / (f"w0{default_cfg.omega_0:.1e}_"
                                 f"zeta{default_cfg.zeta:.1e}_{default_cfg.suffix_ckpt_pinn_blind}")
    best_rmse = float('inf')
    best_num_col = None
    for num_col in num_collocations:
        cfg = DamperConfig(n_col=num_col)
        data = generate_data(cfg, device=device)

        print(f"Testing number of collocation points: {num_col}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data,
                    cfg=cfg, device=device, use_physics=True,
                    extrapolated_physics=False, label="PINN (blind)",
                    ckpt_path=ckpt_pinn_blind)
        rmse = evaluate_blind(cfg.omega_0, cfg.zeta)
        print(f"RMSE for number of collocation points {num_col}: {rmse:.4f}")
        plotting_list.append((num_col, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_num_col = num_col

    print(
        f"Best number of collocation points: {best_num_col} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([num_col for num_col, rmse in plotting_list], [
             rmse for num_col, rmse in plotting_list], marker='o')
    plt.xlabel('Number of Collocation Points')
    plt.ylabel('RMSE')
    plt.title('Number of Collocation Points vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.savefig(f"{cfg.out_dir}/collocation_smooth")


def search_num_obs_per_epoch():
    """Search for optimal number of observations per training epoch.

    Tests different mini-batch sizes to evaluate impact of stochasticity
    on training convergence and generalization. Generates plot showing
    RMSE versus observations per epoch.

    Returns:
        None. Prints best batch size and RMSE, saves plot to output directory.
    """
    # Used to test if mini-batching is beneficial.
    # We have 30 training points => so we can only have at most 30 obs per epoch
    # Search over a range of numbers of observations per epoch
    num_obs_per_epoch_list = [5, 8, 12, 16, 24, 30]
    plotting_list = []  # To store RMSE values for plotting later
    device = get_device()
    default_cfg = DamperConfig()
    # Generate data once, can be reused for all weight combinations
    data = generate_data(default_cfg, device=device)
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / (f"w0{default_cfg.omega_0:.1e}_zeta{default_cfg.zeta:.1e}"
                                 f"_{default_cfg.suffix_ckpt_pinn_blind}")
    best_rmse = float('inf')
    best_num_obs = None
    for num_obs in num_obs_per_epoch_list:
        cfg = DamperConfig(n_obs_per_epoch=num_obs)

        print(f"Testing number of observations per epoch: {num_obs}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data,
                    cfg=cfg, device=device, use_physics=True,
                    extrapolated_physics=False, label="PINN (blind)",
                    ckpt_path=ckpt_pinn_blind)
        rmse = evaluate_blind(cfg.omega_0, cfg.zeta)
        print(
            f"RMSE for number of observations per epoch {num_obs}: {rmse:.4f}")
        plotting_list.append((num_obs, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_num_obs = num_obs

    print(
        f"Best number of observations per epoch: {best_num_obs} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([num_obs for num_obs, rmse in plotting_list], [
             rmse for num_obs, rmse in plotting_list], marker='o')
    plt.xlabel('Number of Observations per Epoch')
    plt.ylabel('RMSE')
    plt.title('Number of Observations per Epoch vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.savefig(f"{cfg.out_dir}/obs_per_epoch.png")


if __name__ == "__main__":
    # search_learning_rate()
    # search_weights()
    search_num_collocations()
    # search_num_obs_per_epoch()
