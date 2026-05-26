"""
Optuna TPE hyperparameter search for the Burger's equation.

Usage:
    python optuna_tuning.py


Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import csv
import json
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np
import optuna
import torch

from config import Config
from data import generate_data
from model import FCNet
from trainer import train
from utils import convert_to_mck, get_device

# -- settings --------------------------------------------------------------
SEED = 42
N_TRIALS = 1000

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/optuna_tuning"

CSV_PATH = OUTPUT_PATH / "optuna_results.csv"

CSV_FIELDS = [
    "trial",
    "state",
    "rmse",
    "hidden",
    "n_layers",
    "lambda_phys",
    "lambda_ic",
    "lr",
    "scheduler_gamma",
    "scheduler_step",
]

DAMPED_CASE_CFGS = {
    "underdamped": {
        "zeta": 0.125,
        "omega_0": 2.0,
    },
    "critically": {
        "zeta": 1.0,
        "omega_0": 2.0,
    },
    "overdamped": {
        "zeta": 1.5,
        "omega_0": 2.0,
    },
}


# -- fixed seed ------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -- generate data once ----------------------------------------------------
_base_cfg = Config()

_base_m = _base_cfg.m
_base_c = _base_cfg.c
_base_k = _base_cfg.k

_base_data = generate_data(_base_cfg)


# -- objective -------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Run a single Optuna trial."""
    set_seed(SEED)

    cfg = Config(
        # -- regime parameters from _base_cfg -----------------------------
        m=_base_m,
        c=_base_c,
        k=_base_k,
        # -- architecture ------------------------------------------------
        hidden=trial.suggest_int(
            "hidden",
            16,
            128,
            step=16,
        ),
        n_layers=trial.suggest_int(
            "n_layers",
            2,
            6,
        ),
        # -- loss weights ------------------------------------------------
        lambda_phys=trial.suggest_float(
            "lambda_phys",
            1e-3,
            1e1,
            log=True,
        ),
        lambda_ic=trial.suggest_float(
            "lambda_ic",
            1e0,
            1e2,
            log=True,
        ),
        # -- learning rate and scheduler --------------------------------
        lr=trial.suggest_float(
            "lr",
            1e-4,
            1e-2,
            log=True,
        ),
        scheduler_gamma=trial.suggest_float(
            "scheduler_gamma",
            0.3,
            0.9,
        ),
        scheduler_step=trial.suggest_int(
            "scheduler_step",
            1000,
            5000,
            step=1000,
        ),
    )

    device = get_device()
    model = FCNet(cfg)

    history, _, _ = train(
        model=model,
        data=_base_data,
        cfg=cfg,
        device=device,
        verbatim=False,
        optuna_trial=trial,
    )

    best_val = min(history["loss_val"])

    # -- write to CSV ------------------------------------------------------
    row = {
        "trial": trial.number,
        "state": "complete",
        "rmse": best_val,
        "hidden": cfg.hidden,
        "n_layers": cfg.n_layers,
        "lambda_phys": cfg.lambda_phys,
        "lambda_ic": cfg.lambda_ic,
        "lr": cfg.lr,
        "scheduler_gamma": cfg.scheduler_gamma,
        "scheduler_step": cfg.scheduler_step,
    }

    write_header = not CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    return best_val


# -- callback --------------------------------------------------------------
def print_callback(
    optuna_study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    """Print Optuna trial progress."""
    print(
        f"  Trial {trial.number:>4} | "
        f"State: {trial.state.name:<10} | "
        f"RMSE: "
        f"{f'{trial.value:.6f}' if trial.value is not None else 'pruned':>12} | "  # noqa: E501
        f"Best: {optuna_study.best_value:.6f}"
    )


# -- run -------------------------------------------------------------------
if __name__ == "__main__":

    for regime_name, regime_params in DAMPED_CASE_CFGS.items():

        print(f"\n{'=' * 60}")
        print(f"Starting study: {regime_name}")
        print(f"{'=' * 60}")

        # -- generate data for this regime --------------------------------
        _base_m, _base_c, _base_k = convert_to_mck(
            regime_params["zeta"],
            regime_params["omega_0"],
        )

        _base_cfg = Config(
            m=_base_m,
            c=_base_c,
            k=_base_k,
        )

        _base_data = generate_data(_base_cfg)

        # -- per-regime output paths --------------------------------------
        regime_path = OUTPUT_PATH / regime_name
        regime_path.mkdir(parents=True, exist_ok=True)

        CSV_PATH = regime_path / "optuna_results.csv"

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=20,
            ),
            sampler=optuna.samplers.TPESampler(
                seed=SEED,
            ),
            storage=f"sqlite:///{regime_path}/optuna.db",
            study_name=f"damped_oscillator_{regime_name}",
            load_if_exists=True,
        )

        study.optimize(
            objective,
            n_trials=N_TRIALS,
            callbacks=[print_callback],
        )

        # -- summary ------------------------------------------------------
        print(f"\n[{regime_name}] Best value : {study.best_value:.6f}")
        print(f"[{regime_name}] Best params:")

        for key, value in study.best_params.items():
            print(f"  {key:<25} {value}")

        # -- save best params ---------------------------------------------
        with open(
            regime_path / "best_params.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "regime": regime_name,
                    "m": float(_base_m),
                    "c": float(_base_c),
                    "k": float(_base_k),
                    "best_val": study.best_value,
                    "best_params": study.best_params,
                },
                f,
                indent=4,
            )

        df = study.trials_dataframe()

        df.to_csv(
            regime_path / "all_trials.csv",
            index=False,
        )

        print(f"[{regime_name}] Saved to {regime_path}")
