"""
Demo script for training a PINN to solve the inverse
problem of the damped harmonic oscillator with parameter estimation.

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
from config import Config
from data import generate_data
from model import InverseFCNet
from plot import save_plots_from_file
from trainer import train
from utils import convert_to_mck, get_device, save_model, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    """Parsing arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--zeta",
                        type=float,
                        default=None,
                        help="Fix damping ratio (skips randomisation)"
                        )
    parser.add_argument("--omega_0",
                        type=float,
                        default=None,
                        help="Fix natural frequency (skips randomisation)"
                        )
    parser.add_argument("--underdamped",
                        action="store_true",
                        help="Randomise in underdamped regime (ζ < 1)"
                        )
    parser.add_argument("--critically-damped",
                        action="store_true",
                        help="Use critically damped case (ζ = 1)"
                        )
    parser.add_argument("--overdamped",
                        action="store_true",
                        help="Randomise in overdamped regime (ζ > 1)"
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
    """Resolve zeta and omega_0 from CLI args, with randomised fallbacks."""

    # -- Resolve zeta --------------------------------------------------
    if args.zeta is not None:
        zeta = args.zeta
    elif args.critically_damped:
        zeta = 1.0
    elif args.underdamped:
        zeta = np.random.uniform(0.02, 0.2)
    elif args.overdamped:
        zeta = np.random.uniform(1.5, 2.0)
    else:
        zeta = np.random.uniform(0.02, 0.2)  # default: underdamped

    # -- Resolve omega_0 -----------------------------------------------
    omega_0 = (
        args.omega_0
        if args.omega_0 is not None
        else np.random.uniform(1.0, 5.0)
        )

    return zeta, omega_0


def resolve_output_path(args, zeta):
    """Pick an output folder based on damping regime unless overridden."""
    if args.output is not None:
        return args.output

    if zeta < 1.0:
        label = "underdamped"
    elif zeta == 1.0:
        label = "criticallydamped"
    else:
        label = "overdamped"

    return SCRIPT_DIR / f"outputs/inversepinn_demo_{label}"


def main():
    print("=" * 50)
    print("  Inverse PINN Demo — Damped Harmonic Oscillator")
    print("  Training a physics-informed neural network")
    print("  to solve the inverse problem.")
    print("=" * 50)
    device = get_device()
    print(f"Device : {device}")

    set_seed(13)
    args = parse_args()
    zeta, omega_0 = resolve_parameters(args)
    output_path = resolve_output_path(args, zeta)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"ζ      : {zeta:.4f}")
    print(f"ω₀     : {omega_0:.4f}")

    m, c, k = convert_to_mck(zeta, omega_0)
    cfg = Config(
        m=m,
        c=c,
        k=k,
        hidden=96,
        n_layers=5,
        lambda_phys=0.5,
        lambda_ic=10,
        lr=2e-3,
        lr_inverse=1.5e-2,
        scheduler_gamma=0.8,
        scheduler_step=3000,
        use_data=True
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

    # -- make plots -------------------------------------------------
    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
