"""Hyperparameter search utilities for PINN model training.

This module provides functions to search for optimal hyperparameters including
learning rate, loss weights, and number of collocation points for training
Physics-Informed Neural Networks (PINNs) on the Burgers equation.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from BurgersPINN.model import BurgerConfig, FCNet
from BurgersPINN.train import evaluate_blind, train_model, generate_data, get_device


def search_learning_rate():
    """Search for optimal learning rate for PINN training.

    Systematically tests a range of learning rates and evaluates the model
    performance using RMSE metric. Generates a plot comparing learning rates
    and saves results to file.

    Returns:
        None. Prints best learning rate and RMSE, saves plot to output directory.
    """
    # Define a range of learning rates to search over
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
    learning_rates = 10**np.linspace(-3.9, -2.1, num=15)
    plotting_list = []  # To store RMSE values for plotting later
    device = get_device()
    # Generate data once, can be reused for all learning rates
    data = generate_data(BurgerConfig(), device=device)
    def_cfg = BurgerConfig()
    out_dir = Path(def_cfg.out_dir)
    ckpt_pinn_blind = Path(f"out_dir/{def_cfg.situation}_"
                           f"{def_cfg.v:.1e}_{def_cfg.suffix_ckpt_pinn_blind}")
    best_rmse = float('inf')
    best_lr = None
    for lr in learning_rates:
        cfg = BurgerConfig(lr=lr)

        print(f"Testing learning rate: {lr}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data, cfg=cfg,
                    device=device, use_physics=True,
                    extrapolated_physics=False,
                    label="PINN (blind)",
                    ckpt_path=ckpt_pinn_blind)
        rmse = evaluate_blind(cfg.situation, cfg.v)
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
    plt.savefig(out_dir / "learning_rate_search.png")


def search_weights():
    """Search for optimal initial condition and physics loss weights.

    Performs a grid search over combinations of lambda_ic (initial condition weight)
    and lambda_phys (physics weight). Evaluates each combination and generates
    a 2D scatter plot showing RMSE across the weight space.

    Returns:
        None. Prints best weights and RMSE, saves scatter plot to output directory.
    """
    ic_weights = 10**np.linspace(0.1, 0.9, num=4)
    phys_weights = 10**np.linspace(-0.9, -0.1, num=4)
    plotting_list = []  # To store RMSE values for plotting later
    device = get_device()
    def_cfg = BurgerConfig()
    # Generate data once, can be reused for all weight combinations
    data = generate_data(def_cfg, device=device)
    out_dir = Path(def_cfg.out_dir)
    ckpt_pinn_blind = Path(f"out_dir/{def_cfg.situation}_"
                           f"{def_cfg.v:.1e}_{def_cfg.suffix_ckpt_pinn_blind}")
    best_rmse = float('inf')
    best_ic = None
    best_phys = None
    for ic_weight in ic_weights:
        for phys_weight in phys_weights:
            cfg = BurgerConfig(lambda_ic=ic_weight, lambda_phys=phys_weight)
            print(
                f"Testing initial condition weight:"
                f" {ic_weight}, physics weight: {phys_weight}")
            model_pinn_blind = FCNet.from_config(cfg)
            train_model(model_pinn_blind,
                        data=data, cfg=cfg,
                        device=device,
                        use_physics=True,
                        extrapolated_physics=False,
                        label="PINN (blind)",
                        ckpt_path=ckpt_pinn_blind)
            rmse = evaluate_blind(cfg.situation, cfg.v)
            print(
                f"RMSE for ic_weight {ic_weight},"
                f"phys_weight {phys_weight}: {rmse:.4f}")
            plotting_list.append((ic_weight, phys_weight, rmse))
            if rmse < best_rmse:
                best_rmse = rmse
                best_ic = ic_weight
                best_phys = phys_weight
    print(
        f"Best initial condition weight: {best_ic},"
        f"Best physics weight: {best_phys} with RMSE: {best_rmse}")
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
    plt.savefig(out_dir / "weight_search.png")
    plt.show()


def search_num_collocations():
    """Search for optimal number of collocation points.

    Tests a range of collocation point counts and evaluates model performance
    via RMSE metric. Generates a plot showing how RMSE varies with the number
    of collocation points used during training.

    Returns:
        None. Prints best number of collocation points and RMSE, saves plot
        to output directory.
    """
    # num_collocations = [10, 20, 50, 100, 200, 300, 400, 500]
    # Search over a range of numbers of collocation points
    num_collocations = np.linspace(210, 490, 15, dtype=int)
    plotting_list = []  # To store RMSE values for plotting later
    device = get_device()
    def_cfg = BurgerConfig()

    out_dir = Path(def_cfg.out_dir)
    ckpt_pinn_blind = Path(f"out_dir/{def_cfg.situation}_"
                           f"{def_cfg.v:.1e}_{def_cfg.suffix_ckpt_pinn_blind}")
    best_rmse = float('inf')
    best_num_col = None
    for num_col in num_collocations:
        cfg = BurgerConfig(n_col=num_col)
        data = generate_data(cfg, device=device)

        print(f"Testing number of collocation points: {num_col}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data,
                    cfg=cfg, device=device,
                    use_physics=True, extrapolated_physics=False,
                    label="PINN (blind)",
                    ckpt_path=ckpt_pinn_blind)
        rmse = evaluate_blind(cfg.situation, cfg.v)
        print(f"RMSE for number of collocation points {num_col}: {rmse:.4f}")
        plotting_list.append((num_col, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_num_col = num_col

    print(
        f"Best number of collocation points:"
        f"{best_num_col} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([num_col for num_col, rmse in plotting_list], [
             rmse for num_col, rmse in plotting_list], marker='o')
    plt.xlabel('Number of Collocation Points')
    plt.ylabel('RMSE')
    plt.title('Number of Collocation Points vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.savefig(out_dir / "num_collocations_search.png")


if __name__ == "__main__":
    # search_learning_rate()
    # search_weights()
    search_num_collocations()
