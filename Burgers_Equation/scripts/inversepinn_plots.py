"""
Script to plot true-predicted nu pairs for the Burgers
inverse problem read from a CSV file.

Usage:
    python InversePINN_plots.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot import style_ax
from utils import save_show

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/nu_estimation_2"
OUTPUT_PATH.mkdir(exist_ok=True)


# -- LaTeX font ------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# -- Color palette ------------------------------------------------
BLUE = "#378ADD"   # noisy training observations
RED = "#E24B4A"   # noisy test/validation observations
GREEN = "#1D9E75"   # PINN
ORANGE = "#EF9F27"   # collocation points
PURPLE = "#7F77DD"   # initial condition marker
GRAY = "#888780"   # true solution / neutral
LGRAY = "#D3D1C7"   # spine colour
BG = "#FAFAF8"   # figure background
PANEL = "#F1EFE8"   # axes background


def nu_estimation_plot(
        csv_path: Path,
        ic: str,
        nu_class: str = None,
        output_path: Path = None,
        show: bool = True
) -> None:
    """Make estimation plot."""
    nu_data = pd.read_csv(csv_path)
    nu_data = nu_data[nu_data["ic"] == ic]
    if nu_class is not None:
        nu_data = nu_data[nu_data["nu_class"] == nu_class]

    fig, ax = plt.subplots(figsize=(8, 8))
    style_ax(ax)

    ax.scatter(nu_data["nu_true"], nu_data["nu_pred"], s=5, color=RED)
    ax.plot(np.linspace(0, 0.5, 2), np.linspace(0, 0.5, 2),
            color=GRAY, linestyle="--", alpha=0.6)

    ax.set_xlabel(r"$\nu$", fontsize=20)
    ax.set_ylabel(r"$\hat\nu$", fontsize=20)

    if nu_class == "low":
        ax.set_xlim(0, 0.05)
        ax.set_ylim(0, 0.05)

    if nu_class is not None:
        ax.set_title(
            rf"$\nu$ prediction for {ic} initial condition and {nu_class} viscosity", fontsize=14)  # noqa:E501
        max = np.max(nu_data[["nu_true", "nu_pred"]])
    else:
        ax.set_title(
            rf"$\nu$ prediction for {ic} initial condition", fontsize=14)

    ax.grid()
    save_show(output_path, show)


def rmse(csv_path: str) -> tuple:
    """Calculate RMSE."""
    """Return the RMSE of every case"""
    df = pd.read_csv(csv_path)
    df_low = df[df["nu_class"] == "low"]
    df_high = df[df["nu_class"] == "high"]

    nu_full = np.sqrt(np.mean((df["nu_true"]-df["nu_pred"]) ** 2))
    nu_low = np.sqrt(np.mean((df_low["nu_true"]-df_low["nu_pred"]) ** 2))
    nu_high = np.sqrt(np.mean((df_high["nu_true"]-df_high["nu_pred"]) ** 2))

    return nu_full, nu_low, nu_high


if __name__ == "__main__":
    nu_full, nu_low, nu_high = rmse(csv_path=OUTPUT_PATH / "nu_pairs.csv")
    print(
        "----------------------------------\n"
        "RMSE for different regimes\n"
        f"all nus : {nu_full:.4f}\n"
        f"low nu: {nu_low:.4f}\n"
        f"high nu : {nu_high:.4f}\n"
        "----------------------------------"
    )

    for ic in ["Gauss"]:

        for nu_class in ["low", "high", None]:
            nu_estimation_plot(csv_path=OUTPUT_PATH / "nu_pairs.csv",
                               ic=ic,
                               nu_class=nu_class,
                               output_path=OUTPUT_PATH /
                               f"scatter_{ic}_{nu_class}.png",
                               show=False
                               )

        print(f"Saved {ic} scatter plots")
