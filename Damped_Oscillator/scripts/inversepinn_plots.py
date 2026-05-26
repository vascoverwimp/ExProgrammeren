"""
Script to read a CSV file and plot the predicted-true parameter pairs
and a 2D color coded scatter plot.

Usage:
    python InversePINN_plots.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from plot import style_ax
from utils import save_show

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/parameter_estimation"

# -- LaTeX font ------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
})

# -- Color palette ---------------------------------------------------------
BLUE = "#378ADD"
RED = "#E24B4A"
GREEN = "#1D9E75"
ORANGE = "#EF9F27"
PURPLE = "#7F77DD"
GRAY = "#888780"
LGRAY = "#D3D1C7"
BG = "#FAFAF8"
PANEL = "#F1EFE8"


def parameter_estimation_plot(
    csv_path: Path,
    parameter: str,
    damped_case: str | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    df = pd.read_csv(csv_path)

    if damped_case is not None:
        df = df[df["damped_case"] == damped_case]

    fig, ax = plt.subplots(figsize=(8, 8))
    style_ax(ax)

    ax.scatter(
        df[f"{parameter}_true"],
        df[f"{parameter}_pred"],
        s=5,
        color=RED,
    )

    max_val = np.max(df[f"{parameter}_true"])
    ax.plot(
        np.linspace(0, max_val, 2),
        np.linspace(0, max_val, 2),
        color=GRAY,
        linestyle="--",
        alpha=0.6,
    )

    damped_case_str = "" if damped_case is None else f"for {damped_case} case"

    if parameter == "zeta":
        ax.set_xlabel(r"$\zeta$", fontsize=20)
        ax.set_ylabel(r"$\hat\zeta$", fontsize=20)
        ax.set_title(rf"$\zeta$ prediction {damped_case_str}", fontsize=14)
    else:
        ax.set_xlabel(r"$\omega_0$", fontsize=20)
        ax.set_ylabel(r"$\hat\omega_0$", fontsize=20)
        ax.set_title(rf"$\omega_0$ prediction {damped_case_str}", fontsize=14)

    ax.grid()

    save_show(output_path, show)


def parameter_space_rmse_plot(
    csv_path: Path,
    damped_case: str | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    df = pd.read_csv(csv_path)

    if damped_case is not None:
        df = df[df["damped_case"] == damped_case]

    mse = np.sqrt(
        0.5
        * (
            (df["zeta_pred"] - df["zeta_true"]) ** 2
            + (df["omega_0_pred"] - df["omega_0_true"]) ** 2
        )
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    style_ax(ax)

    sc = ax.scatter(
        df["zeta_true"],
        df["omega_0_true"],
        c=mse,
        s=18,
        cmap="plasma",
        norm=LogNorm(
            vmin=mse.clip(lower=1e-10).min(),
            vmax=mse.max(),
        ),
        linewidths=0,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(
        r"RMSE $= \sqrt{\frac{1}{2}[(\hat\zeta-\zeta)^2+(\hat\omega_0-\omega_0)^2]}$",  # noqa: E501
        fontsize=12,
    )
    cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())

    damped_case_str = "" if damped_case is None else f" — {damped_case}"

    ax.set_xlabel(r"$\zeta_{\rm true}$", fontsize=18)
    ax.set_ylabel(r"$\omega_{0,\rm true}$", fontsize=18)
    ax.set_title(f"Parameter-space RMSE{damped_case_str}", fontsize=14)

    ax.grid(alpha=0.3)

    save_show(output_path, show)


def rmse(csv_path: Path) -> tuple[float, ...]:
    df = pd.read_csv(csv_path)

    df_under = df[df["damped_case"] == "underdamped"]
    df_crit = df[df["damped_case"] == "critically_damped"]
    df_over = df[df["damped_case"] == "overdamped"]

    def rmse_col(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    return (
        rmse_col(df["zeta_true"], df["zeta_pred"]),
        rmse_col(df_under["zeta_true"], df_under["zeta_pred"]),
        rmse_col(df_crit["zeta_true"], df_crit["zeta_pred"]),
        rmse_col(df_over["zeta_true"], df_over["zeta_pred"]),
        rmse_col(df["omega_0_true"], df["omega_0_pred"]),
        rmse_col(df_under["omega_0_true"], df_under["omega_0_pred"]),
        rmse_col(df_crit["omega_0_true"], df_crit["omega_0_pred"]),
        rmse_col(df_over["omega_0_true"], df_over["omega_0_pred"]),
    )


def main() -> None:
    (
        rmse_zeta_full,
        rmse_zeta_under,
        rmse_zeta_crit,
        rmse_zeta_over,
        rmse_omega_0_full,
        rmse_omega_0_under,
        rmse_omega_0_crit,
        rmse_omega_0_over,
    ) = rmse(csv_path=OUTPUT_PATH / "parameter_pairs.csv")

    print(
        "----------------------------------\n"
        "RMSE for different damping cases\n"
        f"zeta full : {rmse_zeta_full:.4f}\n"
        f"zeta underdamped: {rmse_zeta_under:.4f}\n"
        f"zeta critically damped : {rmse_zeta_crit:.4f}\n"
        f"zeta overdamped : {rmse_zeta_over:.4f}\n"
        f"omega_0 full : {rmse_omega_0_full:.4f}\n"
        f"omega_0 underdamped: {rmse_omega_0_under:.4f}\n"
        f"omega_0 critically damped : {rmse_omega_0_crit:.4f}\n"
        f"omega_0 overdamped : {rmse_omega_0_over:.4f}\n"
        "----------------------------------"
    )

    for damped_case in [
        "overdamped",
        "underdamped",
        "critically_damped",
    ]:
        for parameter in ["zeta", "omega_0"]:
            parameter_estimation_plot(
                csv_path=OUTPUT_PATH / "parameter_pairs.csv",
                parameter=parameter,
                damped_case=damped_case,
                output_path=OUTPUT_PATH
                / f"scatter_{parameter}_{damped_case}.png",
                show=False,
            )
            print(f"Saved {damped_case} scatter plot for {parameter}")

    for parameter in ["zeta", "omega_0"]:
        parameter_estimation_plot(
            csv_path=OUTPUT_PATH / "parameter_pairs.csv",
            parameter=parameter,
            output_path=OUTPUT_PATH / f"scatter_{parameter}_full.png",
            show=False,
        )
        print(f"Saved full scatter plot for {parameter}")

    parameter_space_rmse_plot(
        csv_path=OUTPUT_PATH / "parameter_pairs.csv",
        output_path=OUTPUT_PATH / "parameter_space_rmse_full.png",
        show=False,
    )

    print("Saved parameter space RMSE plot")


if __name__ == "__main__":
    main()
