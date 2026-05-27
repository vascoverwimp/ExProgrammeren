"""
Data generation and analytic solution for the Burgers equation.

Observation points, collocation points, the initial conditions, and
the validation set are all produced in this script.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from config import Config
from analytic import cole_hopf_grid, interpolate_solution


def make_observation(
        cfg: Config,
        u_grid: np.ndarray,
        t_arr: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make observation points."""
    t_obs = np.random.uniform(0.1, cfg.t_dom, cfg.n_obs)
    x_obs = np.random.uniform(0, cfg.L, cfg.n_obs)

    u_obs = []
    for x, t in zip(x_obs, t_obs):
        u_obs.append(interpolate_solution(u_grid, t_arr, x, t,
                     cfg) + np.random.normal(0.0, cfg.sigma))

    return t_obs, x_obs, np.array(u_obs)


def make_collocation(
        cfg: Config
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray
            ]:
    """Make collocation points."""
    t_obs_dom = np.linspace(0.1, cfg.t_dom, cfg.n_col_dom)
    x_obs_dom = np.random.uniform(0, cfg.L, cfg.n_col_dom)

    t_obs_extrap = np.linspace(cfg.t_dom, cfg.t_extrap, int(
        cfg.n_col_dom * (cfg.t_extrap - cfg.t_dom) / cfg.t_dom))
    x_obs_extrap = np.random.uniform(0, cfg.L, int(
        cfg.n_col_dom * (cfg.t_extrap - cfg.t_dom) / cfg.t_dom))

    return t_obs_dom, x_obs_dom, t_obs_extrap, x_obs_extrap


def make_validation(cfg: Config, u_grid: np.ndarray, t_arr: np.ndarray):
    """Make validation points."""
    t_val_1d = np.linspace(0.1, cfg.t_dom, cfg.n_grid_val)
    x_val_1d = np.linspace(0, cfg.L, cfg.n_grid_val)

    # Flatten into (N,) arrays of (x, t) pairs
    tt, xx = np.meshgrid(t_val_1d, x_val_1d)
    t_val = tt.ravel()
    x_val = xx.ravel()

    u_val = np.array([
        interpolate_solution(u_grid, t_arr, x, t, cfg)
        for x, t in zip(x_val, t_val)
    ])

    return t_val, x_val, u_val


def make_bc_points(cfg: Config) -> np.ndarray:
    """Make boundary condition points."""
    t_bc = np.random.uniform(0, cfg.t_dom, cfg.n_bc)
    return t_bc


def generate_data(cfg: Config) -> dict:
    """Generate all data and return as a dict."""
    print("Generating data...")

    u_grid, t_arr = cole_hopf_grid(cfg, pad=800 if cfg.ic in [
                                   "Step_up", "Step_down", "Slope"] else 400)

    t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
    t_col_dom, x_col_dom, t_col_extrap, x_col_extrap = make_collocation(cfg)
    t_val, x_val, u_val = make_validation(cfg, u_grid, t_arr)
    t_bc = make_bc_points(cfg)

    return {
        "u_grid":       u_grid,
        "t_arr":        t_arr,
        "t_obs":        t_obs,
        "x_obs":        x_obs,
        "u_obs":        u_obs,
        "t_val":        t_val,
        "x_val":        x_val,
        "u_val":        u_val,
        "t_col_dom":    t_col_dom,
        "x_col_dom":    x_col_dom,
        "t_col_extrap": t_col_extrap,
        "x_col_extrap": x_col_extrap,
        "t_bc":         t_bc,
        "x_ic":         np.linspace(0, cfg.L, u_grid.shape[1]),
        "t_ic":         np.zeros(u_grid.shape[1]),
        "u_ic":         u_grid[0],
    }


def plot_observations_3d(data: dict) -> None:
    """Make a 3d plot of the observation points."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data["t_obs"], data["x_obs"], data["u_obs"],
               color='red', label='Observations')

    ax.set_xlabel('Time')
    ax.set_ylabel('Space')
    ax.set_zlabel('u(t,x)')
    ax.set_title('Observation Points')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    my_cfg = Config()
    my_data = generate_data(my_cfg)
    plot_observations_3d(my_data)
