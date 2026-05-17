import matplotlib.pyplot as plt
from pathlib import Path
from model import DamperConfig, FCNet
from train import evaluate_blind, train_model, generate_data, get_device
import numpy as np


def search_learning_rate():
    # Define a range of learning rates to search over
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]  # Search over a range of learning rates
    learning_rates =10**np.linspace(-3.5, -1.5, num=9) # Search over a range of learning rates from 10^-3.5 to 10^-1.5: [3.16e-4, 5.62e-4, 1e-3, 1.78e-3, 3.16e-3, 5.62e-3, 1e-2, 1.78e-2, 3.16e-2]
    plot_RMSE = []  # To store RMSE values for plotting later
    device = get_device()
    data = generate_data(DamperConfig(), device=device)  # Generate data once, can be reused for all learning rates
    default_cfg = DamperConfig()
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / f"w0{default_cfg.omega_0:.1e}_zeta{default_cfg.zeta:.1e}_{default_cfg.suffix_ckpt_pinn_blind}"
    best_rmse = float('inf')
    best_lr = None
    for lr in learning_rates:
        cfg = DamperConfig(lr = lr)
        

        print(f"Testing learning rate: {lr}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data, cfg=cfg, device=device, use_physics=True, extrapolated_physics=False,         label       = "PINN (blind)",
        ckpt_path   = ckpt_pinn_blind)
        rmse = evaluate_blind(cfg.omega_0, cfg.zeta)
        print(f"RMSE for learning rate {lr}: {rmse:.4f}")
        plot_RMSE.append((lr, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lr = lr
            
    print(f"Best learning rate: {best_lr} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([lr for lr, rmse in plot_RMSE], [rmse for lr, rmse in plot_RMSE], marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE')
    plt.title('Learning Rate vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.show()

def search_weights():
    ic_weights = 10**np.linspace(1, 2, num=5)  
    phys_weights = 10**np.linspace(-1.5, -0.5, num=5)
    plot_RMSE = []  # To store RMSE values for plotting later
    device = get_device()
    default_cfg = DamperConfig()
    data = generate_data(default_cfg, device=device)  # Generate data once, can be reused for all weight combinations
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / f"w0{default_cfg.omega_0:.1e}_zeta{default_cfg.zeta:.1e}_{default_cfg.suffix_ckpt_pinn_blind}"
    best_rmse = float('inf')

    for ic_weight in ic_weights:
        for phys_weight in phys_weights:
            cfg = DamperConfig(lambda_ic=ic_weight, lambda_phys=phys_weight)
            print(f"Testing initial condition weight: {ic_weight}, physics weight: {phys_weight}")
            model_pinn_blind = FCNet.from_config(cfg)
            train_model(model_pinn_blind, data=data, cfg=cfg, device=device, use_physics=True, extrapolated_physics=False,         label       = "PINN (blind)",
            ckpt_path   = ckpt_pinn_blind)
            rmse = evaluate_blind(cfg.omega_0, cfg.zeta)
            print(f"RMSE for ic_weight {ic_weight}, phys_weight {phys_weight}: {rmse:.4f}")
            plot_RMSE.append((ic_weight, phys_weight, rmse))
            if rmse < best_rmse:
                best_rmse = rmse
                best_ic = ic_weight
                best_phys = phys_weight
    print(f"Best initial condition weight: {best_ic}, Best physics weight: {best_phys} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.scatter([ic_weight for ic_weight, phys_weight, rmse in plot_RMSE], 
                [phys_weight for ic_weight, phys_weight, rmse in plot_RMSE], 
                c=[rmse for ic_weight, phys_weight, rmse in plot_RMSE], 
                cmap='viridis', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Initial Condition Weight (lambda_ic)')
    plt.ylabel('Physics Weight (lambda_phys)')
    plt.title('Weight Search for PINN (blind)')
    plt.colorbar(label='RMSE')
    plt.grid(True)
    plt.show()

def search_num_collocations():
    num_collocations = [10, 20, 50, 100, 200, 500]
    # num_collocations = np.linspace(10, 100, num=19)  # Search over a range of numbers of collocation points
    plot_RMSE = []  # To store RMSE values for plotting later
    device = get_device()
    default_cfg = DamperConfig()
    data = generate_data(default_cfg, device=device)  # Generate data once, can be reused for all weight combinations
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / f"w0{default_cfg.omega_0:.1e}_zeta{default_cfg.zeta:.1e}_{default_cfg.suffix_ckpt_pinn_blind}"
    best_rmse = float('inf')
    best_num_col = None
    for num_col in num_collocations:
        cfg = DamperConfig(n_col=num_col)


        print(f"Testing number of collocation points: {num_col}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data, cfg=cfg, device=device, use_physics=True, extrapolated_physics=False,         label       = "PINN (blind)",
        ckpt_path   = ckpt_pinn_blind)
        rmse = evaluate_blind(cfg.omega_0, cfg.zeta)
        print(f"RMSE for number of collocation points {num_col}: {rmse:.4f}")
        plot_RMSE.append((num_col, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_num_col = num_col
            
    print(f"Best number of collocation points: {best_num_col} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([num_col for num_col, rmse in plot_RMSE], [rmse for num_col, rmse in plot_RMSE], marker='o')
    plt.xlabel('Number of Collocation Points')
    plt.ylabel('RMSE')
    plt.title('Number of Collocation Points vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.show()

def search_num_obs_per_epoch():
    # Used to test if mini-batching is beneficial. 
    # 20 max bec val fraction is 0.2, and we have 30 training points => around 6 observations are lost to the validation set, so we can only have at most 24 obs per epoch
    num_obs_per_epoch_list = [5, 8, 12, 16, 20]  # Search over a range of numbers of observations per epoch
    plot_RMSE = []  # To store RMSE values for plotting later
    device = get_device()
    default_cfg = DamperConfig()
    data = generate_data(default_cfg, device=device)  # Generate data once, can be reused for all weight combinations
    out_dir = Path(default_cfg.out_dir)
    ckpt_pinn_blind = out_dir / f"w0{default_cfg.omega_0:.1e}_zeta{default_cfg.zeta:.1e}_{default_cfg.suffix_ckpt_pinn_blind}"
    best_rmse = float('inf')
    best_num_obs = None
    for num_obs in num_obs_per_epoch_list:
        cfg = DamperConfig(n_obs_per_epoch=num_obs)
        

        print(f"Testing number of observations per epoch: {num_obs}")
        model_pinn_blind = FCNet.from_config(cfg)
        train_model(model_pinn_blind, data=data, cfg=cfg, device=device, use_physics=True, extrapolated_physics=False,         label       = "PINN (blind)",
        ckpt_path   = ckpt_pinn_blind)
        rmse = evaluate_blind(cfg.omega_0, cfg.zeta)
        print(f"RMSE for number of observations per epoch {num_obs}: {rmse:.4f}")
        plot_RMSE.append((num_obs, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_num_obs = num_obs

    print(f"Best number of observations per epoch: {best_num_obs} with RMSE: {best_rmse}")
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot([num_obs for num_obs, rmse in plot_RMSE], [rmse for num_obs, rmse in plot_RMSE], marker='o')
    plt.xlabel('Number of Observations per Epoch')
    plt.ylabel('RMSE')
    plt.title('Number of Observations per Epoch vs RMSE for PINN (blind)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # search_learning_rate()
    # search_weights()
    # search_num_collocations()
    # search_num_obs_per_epoch()