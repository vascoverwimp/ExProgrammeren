"""
Training loop for the damped spring-mass PINN.

Contains:
    train -- run the Adam optimiser with cosine-annealing scheduler,
             early stopping, periodic snapshots, and validation logging

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""
import time
from config import Config
import torch
import torch.nn as nn
import optuna
from utils import to_tensor
from losses import loss_data, loss_ic, loss_physics, loss_physics_inverse
from data import make_collocation, make_train_observation

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
    snapshots : dict
        Maps epoch number to CPU state_dict at cfg.snapshot_epochs.

    Saves
    -----
    best_model.pt :
        The weights that correspond to the lowest validation loss
    """
    # -- detect inverse mode -------------------------------------------------
    inverse_mode = hasattr(model, "zeta_hat")

    # -- move model and tensors to device ------------------------------------
    model.to(device)

    t_val_t = to_tensor(data["t_val"])
    y_val_t = to_tensor(data["y_val"])
    t_ic_t = to_tensor(data["t_ic"], requires_grad=True)

    t_obs_t = to_tensor(data["t_obs"])
    y_obs_t = to_tensor(data["y_obs"])

    t_col_dom_t = to_tensor(data["t_col_dom"], requires_grad=True)
    t_col_extrap_t = to_tensor(data["t_col_extrap"], requires_grad=True)

    best_state = None

    # -- initialize optimiser and scheduler ----------------------------------
    if inverse_mode:
        optimiser = torch.optim.Adam([
            {"params": model.net.parameters(), "lr": cfg.lr},
            {
                "params": [model.zeta_hat, model.omega_0_hat],
                "lr": cfg.lr_inverse
                }
        ], betas=(cfg.adam_beta1, cfg.adam_beta2))
    else:
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2)
            )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser,
        step_size=cfg.scheduler_step,
        gamma=cfg.scheduler_gamma
        )

    # -- history and snapshot storage ---------------------------------------
    history = {
        "epoch":            [],
        "loss_data":        [],
        "loss_phys":        [],
        "loss_ic":          [],
        "loss_val":         [],
        "loss_total":       [],
    }
    if inverse_mode:
        history["zeta_hat"] = []
        history["omega_0_hat"] = []

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
            if cfg.randomise_observation:  # Randomise every epoch
                t_obs, y_obs = make_train_observation(cfg)
                t_obs_t = to_tensor(t_obs)
                y_obs_t = to_tensor(y_obs)

            y_pred = model(t_obs_t)
            l_data = loss_data(y_pred, y_obs_t)
        else:
            l_data = torch.zeros(1, device=device)

        # -- physics loss ------------------------------------
        if cfg.randomise_collocation:  # Randomise every epoch
            t_col_dom, t_col_extrap = make_collocation(cfg)
            t_col_dom_t = to_tensor(t_col_dom,    requires_grad=True)
            t_col_extrap_t = to_tensor(t_col_extrap, requires_grad=True)

        if inverse_mode:
            if cfg.train_extrap:
                l_phys = loss_physics_inverse(
                    model,
                    torch.cat([t_col_dom_t, t_col_extrap_t])
                    )
            else:
                l_phys = loss_physics_inverse(model, t_col_dom_t)
        elif cfg.use_physics:
            if cfg.train_extrap:
                l_phys = loss_physics(
                    cfg,
                    model,
                    torch.cat([t_col_dom_t, t_col_extrap_t])
                    )
            else:
                l_phys = loss_physics(cfg, model, t_col_dom_t)
        else:
            l_phys = torch.zeros(1, device=device)

        # -- initial condition loss ------------------------------------
        l_ic = (
            loss_ic(cfg, model, t_ic_t)
            if cfg.use_ic
            else torch.zeros(1, device=device)
            )

        loss_total = (
            l_data
            + cfg.lambda_phys * l_phys
            + cfg.lambda_ic * l_ic
            )

        # -- update weights ------------------------------------
        loss_total.backward()
        optimiser.step()
        scheduler.step()

        # -- validation ------------------------------------
        model.eval()
        with torch.no_grad():
            y_val_pred = model(t_val_t)
            l_val = loss_data(y_val_pred, y_val_t)

        # -- optuna pruning ------------------------------------------------
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
                    f"[{label}] early stopping at epoch {epoch},"
                    " improvement stalled."
                )
            break

        # -- snapshots ------------------------------------------------------
        if epoch in cfg.snapshot_epochs:
            snapshots[epoch] = {
                k: v.cpu() for k, v in model.state_dict().items()
            }

        # -- logging --------------------------------------------------------
        if verbatim:
            if epoch % cfg.log_every == 0 or epoch == 1:
                elapsed = time.perf_counter() - t0
                zeta_str = (
                    f"zeta_hat={model.zeta_hat.item():.5f}  "
                    if inverse_mode
                    else ""
                )
                omega_0_str = (
                    f"omega_0_hat={model.omega_0_hat.item():.5f}  "
                    if inverse_mode
                    else ""
                )
                print(
                    f"  [{label}] epoch {epoch:5d} | "
                    f"L_data={l_data.item():.5f}  "
                    f"L_phys={l_phys.item():.5f}  "
                    f"L_ic={l_ic.item():.5f}  "
                    f"L_val={l_val.item():.5f}  "
                    f"L_total={loss_total.item():.5f}  "
                    f"{zeta_str}"
                    f"{omega_0_str}"
                    f"({elapsed:.1f}s)"
                    )

        if epoch % cfg.log_every == 0:
            history["epoch"].append(epoch)
            history["loss_data"].append(l_data.item())
            history["loss_phys"].append(l_phys.item())
            history["loss_ic"].append(l_ic.item())
            history["loss_val"].append(l_val.item())
            history["loss_total"].append(loss_total.item())
            if inverse_mode:
                history["zeta_hat"].append(model.zeta_hat.item())
                history["omega_0_hat"].append(model.omega_0_hat.item())

    total_time = time.perf_counter() - t0
    if verbatim:
        print(
            f"  [{label}] finished in {total_time:.1f}s "
            f"({total_time / epoch * 1000:.2f} ms/epoch)"
            )
        if inverse_mode:
            print(
                f"{'Predicted zeta:':<20} "
                f"{float(model.zeta_hat.detach()):.4f}"
                )
            print(f"{'Actual zeta:':<20} {cfg.zeta:.4f}")
            print(
                f"{'Predicted omega_0:':<20} "
                f"{float(model.omega_0_hat.detach()):.4f}"
                )
            print(f"{'Actual omega_0:':<20} {cfg.omega_0:.4f}")

    return history, snapshots, best_state
