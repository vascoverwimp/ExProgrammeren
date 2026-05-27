"""
Demo script for training a PINN to solve
the inverse problem of the Burgers equation.

Usage:
    python inversepinn_demo.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import argparse
import numpy as np
from analytic import predict_shock_time
from config import Config
from data import generate_data
from model import InverseFCNet
from plot import save_plots_from_file
from trainer import train
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent

IC_CHOICES = ["Gauss", "N_wave", "Step_up"]
NU_LOW = (0.005, 0.015)
NU_HIGH = (0.1,   0.5)


def parse_args():
    """Parsing arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic",
                        type=str,
                        default=None,
                        choices=IC_CHOICES,
                        help="Initial condition type (default: Gauss)"
                        )
    parser.add_argument("--nu",
                        type=float,
                        default=None,
                        help="Fix viscosity (skips randomisation)"
                        )
    parser.add_argument("--nu-regime",
                        type=str,
                        default="high",
                        choices=["low", "high"],
                        help="Viscosity regime to sample from (default: high)"
                        )
    parser.add_argument("--show",
                        action="store_true",
                        help="Display plots interactively"
                        )
    parser.add_argument("--output",
                        type=Path,
                        default=None,
                        help="Override output directory"
                        )
    return parser.parse_args()


def resolve_parameters(args):
    """Resolve ic and nu from CLI args."""
    ic = args.ic if args.ic is not None else "Gauss"
    if args.nu is not None:
        nu = args.nu
    else:
        lo, hi = NU_LOW if args.nu_regime == "low" else NU_HIGH
        nu = np.random.uniform(lo, hi)
    return ic, nu


def resolve_output_path(args):
    """Pick an output folder based on viscosity regime."""
    if args.output is not None:
        return args.output
    return SCRIPT_DIR / f"outputs/inversepinn_{args.nu_regime()}"


def main():
    """Main loop."""
    print("=" * 50)
    print("  Inverse PINN Demo — Burgers Equation")
    print("  Training a physics-informed neural network")
    print("  to solve the inverse problem.")
    print("=" * 50)
    device = get_device()
    print(f"Device : {device}")

    args = parse_args()
    ic, nu = resolve_parameters(args)
    output_path = resolve_output_path(args, ic)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"IC     : {ic}")
    print(f"ν      : {nu:.4f}")

    cfg = Config(
        ic=ic,
        nu=nu,
        hidden=112,
        n_layers=9,
        lambda_phys=1e1,
        lambda_ic=1e2,
        lr=0.003,
        scheduler_gamma=0.6,
        scheduler_step=3000,
        patience_threshold=1e-5,
        use_data=True,
    )

    time_to_shock = predict_shock_time(cfg)
    if 0.1 <= time_to_shock <= 5:
        print("-" * 50)
        print(f"Shockwave predicted at t = {time_to_shock:.3f}s.")
        print("Setting training domain to end at shock time.")
        print("-" * 50)
        cfg.t_shock = time_to_shock
        cfg.t_dom = time_to_shock + 2.0
        cfg.t_extrap = time_to_shock + 4.0
    else:
        print(
            f"Shock time {time_to_shock:.3f}s outside expected range"
            " — using default domain."
        )

    # -- train model ------------------------------------------------
    data = generate_data(cfg)
    model = InverseFCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
    )

    # -- save model -------------------------------------------------
    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    # -- make plots -------------------------------------------------
    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
