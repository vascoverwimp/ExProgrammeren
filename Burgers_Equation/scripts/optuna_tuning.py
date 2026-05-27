"""
Optuna TPE hyperparameter search for the Burger's equation.

Usage:
    python optuna_tuning.py


Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import random
import csv
import json
import numpy as np
import torch
import optuna
from pathlib import Path

from config import Config
from model import FCNet
from trainer import train
from data import generate_data
from utils import get_device

# -- settings --------------------------------------------------------------
SEED = 42
N_TRIALS = 500

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/optuna_tuning_2"
OUTPUT_PATH.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_PATH / "optuna_results.csv"
CSV_FIELDS = [
    "trial", "state", "rmse",
    # architecture
    "hidden", "n_layers",
    # loss weights
    "lambda_phys", "lambda_ic", "lambda_bc",
    # learning rate
    "lr", "scheduler_gamma", "scheduler_step",
]

ICS = ["Gauss"]

# -- fixed seed ------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -- generate data once ----------------------------------------------------
_base_cfg = Config()
_base_data = generate_data(_base_cfg)

# -- objective -------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    set_seed(SEED)

    cfg = Config(
        # -- regime parameters from _base_cfg -----------------------------
        ic=_base_cfg.ic,
        # architecture
        hidden=trial.suggest_int("hidden",          16,   128, step=16),
        n_layers=trial.suggest_int("n_layers",         2,     10),
        # loss weights
        lambda_phys=trial.suggest_float(
            "lambda_phys",     1e-1,  1e2, log=True),
        lambda_ic=trial.suggest_float("lambda_ic",       1e1,   1e3, log=True),
        lambda_bc=trial.suggest_float("lambda_bc",       1e1,   1e3, log=True),
        # learning rate and scheduler
        lr=trial.suggest_float("lr",              1e-4,  1e-2, log=True),
        scheduler_gamma=trial.suggest_float("scheduler_gamma", 0.3,   0.9),
        scheduler_step=trial.suggest_int(
            "scheduler_step",  1000, 5000, step=1000),

        use_data=True
    )

    device = get_device()
    model = FCNet(cfg)

    try:
        history, _, _ = train(
            model=model,
            data=_base_data,
            cfg=cfg,
            device=device,
            verbatim=False,
            optuna_trial=trial,
        )
    except optuna.exceptions.TrialPruned:
        raise

    best_val = min(history["loss_val"])

    # -- write to CSV ------------------------------------------------------
    row = {
        "trial":             trial.number,
        "state":             "complete",
        "rmse":              best_val,
        "hidden":            cfg.hidden,
        "n_layers":          cfg.n_layers,
        "lambda_phys":       cfg.lambda_phys,
        "lambda_ic":         cfg.lambda_ic,
        "lambda_bc":         cfg.lambda_bc,
        "lr":                cfg.lr,
        "scheduler_gamma":   cfg.scheduler_gamma,
        "scheduler_step":    cfg.scheduler_step,
    }

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return best_val


# -- callback --------------------------------------------------------------
def print_callback(
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial
        ) -> None:
    print(
        f"  Trial {trial.number:>4} | "
        f"State: {trial.state.name:<10} | "
        f"RMSE: {f'{trial.value:.6f}' if trial.value is not None else 'pruned':>12} | "  # noqa:E501
        f"Best:  {study.best_value:.6f}"
    )


# -- run -------------------------------------------------------------------
if __name__ == "__main__":
    for ic in ICS:
        print(f"\n{'='*60}")
        print(f"Starting study: {ic}")
        print(f"{'='*60}")

        # -- generate data for this initial condition -------------------------
        _base_cfg = Config(ic=ic)
        _base_data = generate_data(_base_cfg)

        # -- per-regime output paths ---------------------------------------
        regime_path = OUTPUT_PATH / ic
        regime_path.mkdir(parents=True, exist_ok=True)
        CSV_PATH = regime_path / "optuna_results.csv"

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
            sampler=optuna.samplers.TPESampler(seed=SEED),
            storage=f"sqlite:///{regime_path}/optuna.db",
            study_name=f"Burgers_Equation_{ic}",
            load_if_exists=True,
        )

        study.optimize(
            objective,
            n_trials=N_TRIALS,
            callbacks=[print_callback],
        )

        # -- summary -------------------------------------------------------
        print(f"\n[{ic}] Best value : {study.best_value:.6f}")
        print(f"[{ic}] Best params:")
        for k, v in study.best_params.items():
            print(f"  {k:<25} {v}")

        # -- save best params ----------------------------------------------
        with open(regime_path / "best_params.json", "w") as f:
            json.dump({
                "regime":      ic,
                "best_val":    study.best_value,
                "best_params": study.best_params,
            }, f, indent=4)

        df = study.trials_dataframe()
        df.to_csv(regime_path / "all_trials.csv", index=False)
        print(f"[{ic}] Saved to {regime_path}")
