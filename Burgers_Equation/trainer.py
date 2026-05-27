"""
Training loop for the Burger's Equation.

Contains:
    train -- run the Adam optimiser with cosine-annealing scheduler,
             early stopping, periodic snapshots, and validation logging

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import optuna
import time
import copy
from config import Config
import torch
import torch.nn as nn
from utils import to_tensor
from losses import loss_data, loss_ic, loss_physics, loss_physics_inverse, loss_bc  # noqa:E501
from data import make_collocation, make_observation, make_bc_points


def train(
    model: nn.Module,
    data: dict,
    cfg: Config,
    device: torch.device,
    label: str = "Model",
    verbatim: bool = True,
    optuna_trial: optuna.Trial = None
) -> tuple[dict, dict]:
    """
    Train the PINN or standard ML model.

    Parameters
    ----------
    model : nn.Module
        The neural network to train. Must already be instantiated on CPU;
        this function moves it to device.
    data : dict
        Output of generate_data(). Must contain keys:
        t_train, y_train, t_val, y_val, t_col, t_ic.
    cfg : Config
        All hyperparameters, physical constants, and loss terms.
    device : torch.device
        Device to train on.
    label : str
        Label used in printed output.

    Returns
    -------
    history : dict
        Loss values logged every cfg.log_every epochs. Keys:
        epoch, loss_data, loss_phys,
        loss_ic, loss_val, loss_total.
        Also saves predicted parameters in case of inverse mode.
    snapshots : dict
        Maps epoch number to CPU state_dict at cfg.snapshot_epochs.

    Saves
    -----
    best_model.pt :
        The weights that correspond to the lowest validation loss
    """
    # -- detect inverse mode -------------------------------------------------
    inverse_mode = hasattr(model, "nu_hat")

    # -- move model and tensors to device ------------------------------------
    model.to(device)

    x_val_t = to_tensor(data["x_val"])
    t_val_t = to_tensor(data["t_val"])
    u_val_t = to_tensor(data["u_val"])
    x_ic_t = to_tensor(data["x_ic"])
    t_ic_t = to_tensor(data["t_ic"])
    u_ic_t = to_tensor(data["u_ic"])

    u_grid = data["u_grid"]
    t_arr = data["t_arr"]

    x_obs_t = to_tensor(data["x_obs"])
    t_obs_t = to_tensor(data["t_obs"])
    u_obs_t = to_tensor(data["u_obs"])

    x_col_dom_t = to_tensor(data["x_col_dom"],    requires_grad=True)
    t_col_dom_t = to_tensor(data["t_col_dom"],    requires_grad=True)
    x_col_extrap_t = to_tensor(data["x_col_extrap"], requires_grad=True)
    t_col_extrap_t = to_tensor(data["t_col_extrap"], requires_grad=True)

    t_bc_t = to_tensor(data["t_bc"])

    best_state=None

    # -- initialize optimiser and scheduler -----------------------------------
    if inverse_mode:
        optimiser = torch.optim.Adam([
            {"params": model.net.parameters(),   "lr": cfg.lr},
            {"params": [model._nu_raw],
                "lr": cfg.lr_inverse}  # ← changed
        ], betas=(cfg.adam_beta1, cfg.adam_beta2))
    else:
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2)
            )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)

    # -- history and snapshot storage ---------------------------------------
    history = {
        "epoch":            [],
        "loss_data":        [],
        "loss_phys":        [],
        "loss_ic":          [],
        "loss_bc":          [],
        "loss_val":         [],
        "loss_total":       [],
    }
    if inverse_mode:
        history["nu_hat"] = []

    snapshots = {}
    best_val_loss = float("inf")
    epochs_no_improvement = 0

    # -- training loop ------------------------------------------------------
    t0 = time.perf_counter()

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        # -- data loss ------------------------------------
        if cfg.use_data:
            if cfg.randomise_observation:
                t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
                t_obs_t = to_tensor(t_obs)  # Randomise every epoch
                x_obs_t = to_tensor(x_obs)
                u_obs_t = to_tensor(u_obs)

            u_pred = model(t_obs_t, x_obs_t)
            l_data = loss_data(u_pred, u_obs_t)
        else:
            l_data = torch.zeros(1, device=device)

        # -- physics loss ------------------------------------
        if cfg.randomise_collocation:
            (
                t_col_dom,
                x_col_dom,
                t_col_extrap,
                x_col_extrap
             ) = make_collocation(cfg)
            # Randomise every epoch
            t_col_dom_t = to_tensor(t_col_dom,     requires_grad=True)
            x_col_dom_t = to_tensor(x_col_dom,     requires_grad=True)
            t_col_extrap_t = to_tensor(t_col_extrap,  requires_grad=True)
            x_col_extrap_t = to_tensor(x_col_extrap,  requires_grad=True)

        if inverse_mode:
            if cfg.train_extrap:
                l_phys = loss_physics_inverse(
                    model,
                    torch.cat(
                        [t_col_dom_t, t_col_extrap_t]).requires_grad_(True),
                    torch.cat(
                        [x_col_dom_t, x_col_extrap_t]).requires_grad_(True)
                    )
            else:
                l_phys = loss_physics_inverse(model, t_col_dom_t, x_col_dom_t)
        elif cfg.use_physics:
            if cfg.train_extrap:
                l_phys = loss_physics(
                    cfg,
                    model,
                    torch.cat(
                        [t_col_dom_t, t_col_extrap_t]).requires_grad_(True),
                    torch.cat(
                        [x_col_dom_t, x_col_extrap_t]).requires_grad_(True)
                    )
            else:
                l_phys = loss_physics(cfg, model, t_col_dom_t, x_col_dom_t)
        else:
            l_phys = torch.zeros(1, device=device)

        # -- initial condition loss ------------------------------------
        l_ic = loss_ic(model, t_ic_t, x_ic_t,
                       u_ic_t) if cfg.use_ic else torch.zeros(1, device=device)

        # -- boundary condition loss ------------------------------------
        if cfg.randomise_bc_points:
            t_bc = make_bc_points(cfg)
            t_bc_t = to_tensor(t_bc)

        l_bc = loss_bc(cfg, model, t_bc_t) if cfg.use_bc else torch.zeros(
            1, device=device)

        loss_total = (l_data + cfg.lambda_phys * l_phys +
                      cfg.lambda_ic * l_ic + cfg.lambda_bc * l_bc)

        # -- update weights ------------------------------------
        loss_total.backward()
        optimiser.step()
        scheduler.step()

        # -- validation ------------------------------------
        model.eval()
        with torch.no_grad():
            u_val_pred = model(t_val_t, x_val_t)
            l_val = loss_data(u_val_pred, u_val_t)

        # -- Optune pruning ------------------------------------------------
        if optuna_trial is not None and epoch % cfg.log_every == 0:
            optuna_trial.report(l_val.item(), epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # -- early stopping -------------------------------------------------
        if l_val.item() < best_val_loss - cfg.patience_threshold:
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if l_val.item() < best_val_loss:
            best_val_loss = l_val.item()
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epochs_no_improvement >= cfg.patience:
            if verbatim:
                print(
                    f"  [{label}] early stopping at epoch {epoch}, improvement stalled."
                    )
            break

        # -- snapshots ------------------------------------------------------
        if epoch in cfg.snapshot_epochs:
            snapshots[epoch] = copy.deepcopy(model.state_dict())

        # -- logging --------------------------------------------------------
        if verbatim:
            if epoch % cfg.log_every == 0 or epoch == 1:
                elapsed = time.perf_counter() - t0
                nu_str = (
                    f"nu_hat={model.nu_hat.item():.5f}  "
                    if inverse_mode
                    else ""
                    )
                print(f"  [{label}] epoch {epoch:5d} | "
                      f"L_data={l_data.item():.5f}  "
                      f"L_phys={l_phys.item():.5f}  "
                      f"L_ic={l_ic.item():.5f}  "
                      f"L_bc={l_bc.item():.5f}  "
                      f"L_val={l_val.item():.5f}  "
                      f"L_total={loss_total.item():.5f}  "
                      f"{nu_str}"
                      f"({elapsed:.1f}s)")

        if epoch % cfg.log_every == 0:
            history["epoch"].append(epoch)
            history["loss_data"].append(l_data.item())
            history["loss_phys"].append(l_phys.item())
            history["loss_ic"].append(l_ic.item())
            history["loss_bc"].append(l_bc.item())
            history["loss_val"].append(l_val.item())
            history["loss_total"].append(loss_total.item())
            if inverse_mode:
                history["nu_hat"].append(model.nu_hat.item())

    model.load_state_dict(best_state)
    total_time = time.perf_counter() - t0
    if verbatim:
        print(f"  [{label}] finished in {total_time:.1f}s "
              f"({total_time / epoch * 1000:.2f} ms/epoch)")
        if inverse_mode:
            print(f"{'Predicted nu:':<20} {float(model.nu_hat):.4f}")
            print(f"{'Actual nu:':<20} {cfg.nu:.4f}")

    return history, snapshots, best_state
