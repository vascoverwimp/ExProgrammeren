"""
Utils file for damped spring-mass PINN.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import json
import os
import random
import dataclasses
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from config import Config


def get_device() -> torch.device:
    """Return device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(
        arr: np.ndarray,
        requires_grad: bool = False,
        unsqueeze: bool = True
        ) -> torch.Tensor:
    """Convert a 1-D numpy array to a (N, 1) float32 tensor on DEVICE."""
    device = get_device()
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(device)
    if not unsqueeze:
        t = torch.tensor(arr, dtype=torch.float32).to(device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Calculate RMSE."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def pointwise_residual(y: np.ndarray, t: np.ndarray, cfg: Config):
    """Calculate pointwise residual."""
    dy = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(cfg.m * d2y + cfg.c * dy + cfg.k * y)


def convert_to_mck(zeta, omega_0):
    """Convert zeta and omega into c and k values for inverse problem."""
    m = 1
    c = 2 * omega_0 * zeta
    k = omega_0 ** 2
    return m, c, k


def save_show(output_path: str = None, show: bool = True) -> None:
    """Save the current figure and/or show it."""
    if output_path is not None:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def load_best_cfg(json_path: str) -> Config:
    """Load best Optuna parameters from a JSON file as a Config object."""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    params = data["best_params"]

    if "m" in data:
        params["m"] = data["m"]
    if "c" in data:
        params["c"] = data["c"]
    if "k" in data:
        params["k"] = data["k"]

    return Config(**params)


def save_model(
    best_state:  dict,
    history:     dict,
    snapshots:   dict,
    cfg:         Config,
    output_path: Path,
    verbatim: bool = True,
) -> None:
    """
    Save model weights, training history, snapshots, and config.

    Saves
    -----
    <output_path>/final_model.pt       -- final model state dict
    <output_path>/history.csv          -- loss curves, one row per log step
    <output_path>/snapshots.pt         -- all snapshot state dicts
    <output_path>/config.json          -- full config for reproducibility
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # -- final model weights as .pt file ------------------------------------
    torch.save(best_state, output_path / "best_model.pt")

    # -- history as .csv file -----------------------------------------------
    import csv
    with open(output_path / "history.csv", "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()
        writer.writerows([
            {k: history[k][i] for k in history}
            for i in range(len(history["epoch"]))
        ])

    # -- snapshots as a single .pt file ------------------------------------
    torch.save(snapshots, output_path / "snapshots.pt")

    # -- config as .json file --------------------------------
    with open(output_path / "config.json", "w", encoding='utf-8') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)

    if verbatim:
        print(f"Saved model, history, snapshots and config to {output_path}/")


def load_model(
        model: nn.Module,
        folder_path: str
        ) -> tuple[nn.Module, dict, dict]:
    """
    Load model weights, history, and snapshots from a saved run.

    Parameters
    ----------
    model : nn.Module
        The model architecture to load weights into. Must be on CPU.
    folder_path : str
        Path to the folder containing the saved model files.

    Returns
    -------
    model     : nn.Module with loaded weights
    history   : dict of loss curves
    snapshots : dict mapping epoch -> state dict
    """
    import csv
    folder_path = Path(folder_path)

    # model weights
    model.load_state_dict(
        torch.load(folder_path / "best_model.pt", map_location="cpu")
        )

    # history
    with open(folder_path / "history.csv", encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    history = {k: [float(r[k]) for r in rows] for k in rows[0]}

    # config
    with open(folder_path / "config.json", encoding='utf-8') as f:
        cfg_dict = json.load(f)
    cfg = Config(**cfg_dict)

    # snapshots
    snapshots = torch.load(folder_path / "snapshots.pt", map_location="cpu")

    return model, history, snapshots, cfg
