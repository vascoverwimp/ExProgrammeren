"""
Analytic and numerical solution techniques for the Burgers equation.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from config import Config


def gauss(cfg: Config) -> np.ndarray:
    """Generate a gauss curve."""
    x = np.linspace(0, cfg.L, cfg.n_x)
    mean = cfg.L / 2
    return cfg.height * np.exp(-((x - mean) ** 2) / (2 * cfg.sigma_ic ** 2))


def step_up(cfg: Config) -> np.ndarray:
    """Generate a step-up function."""
    u = np.zeros(cfg.n_x)
    mid = cfg.n_x // 2
    u[mid:] = cfg.height
    return u


def step_down(cfg: Config) -> np.ndarray:
    """Generate a step-down function."""
    u = np.zeros(cfg.n_x)
    mid = cfg.n_x // 2
    u[:mid] = cfg.height
    return u


def n_wave(cfg: Config) -> np.ndarray:
    """Generate an N-wave."""
    u = np.zeros(cfg.n_x)
    center = int(cfg.n_x // 2)
    width_rel = int(cfg.width // (2 * cfg.delta_x))
    u[int(center-width_rel):int(center+width_rel)
      ] = np.linspace(-cfg.height, cfg.height, int(2 * width_rel))
    return -u


def n_wave_chop(cfg: Config) -> np.ndarray:
    """Generate an N-wave chopped in half and rearranged."""
    u = np.zeros(cfg.n_x)

    center = int(cfg.n_x // 2)

    with_rel = int(cfg.width // (2 * cfg.delta_x))

    # Positive triangle
    start1 = center - with_rel
    peak1 = center

    # Negative triangle
    peak2 = center
    end2 = center + with_rel

    # Rising positive slope
    u[start1:peak1] = np.linspace(0, cfg.height, peak1 - start1)

    # Falling negative slope
    u[peak2:end2] = np.linspace(-cfg.height, 0, end2 - peak2)

    return -u


def negative_slope(cfg: Config) -> np.ndarray:
    """Generate a slope."""
    u = np.zeros(cfg.n_x)
    center = int(cfg.n_x // 2)
    width_rel = int(cfg.width // (2 * cfg.delta_x))
    u[:int(center-width_rel)] = cfg.height
    u[int(center-width_rel):int(center+width_rel)
      ] = np.linspace(cfg.height, 0, int(2 * width_rel))
    return u


def u_0(cfg: Config) -> np.ndarray:
    """Generate the initial condition u(x, 0) based on the config."""
    if cfg.ic == "Gauss":
        u_init = gauss(cfg)
    elif cfg.ic == "Step_up":
        u_init = step_up(cfg)
    elif cfg.ic == "N_wave":
        u_init = n_wave(cfg)
    elif cfg.ic == "N_wave_chop":
        u_init = n_wave_chop(cfg)
    elif cfg.ic == "Slope":
        u_init = negative_slope(cfg)
    elif cfg.ic == "Step_down":
        u_init = step_down(cfg)
    else:
        raise ValueError(f"Unknown initial condition type: {cfg.ic}")

    if cfg.time_to_ic > 0:
        return solve_burgers_padded(u_init, cfg.time_to_ic, cfg, 200)
    else:
        return u_init


# -- Cole-Hopf transformation ------------------------------------------------
def cole_hopf_trans(u_init: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Transform to phi using Cole-Hopf tranformation.

    phi_0 = exp(-1/2nu int^x_0 u(x)dx)
    """
    phi_0 = np.zeros_like(u_init, dtype=float)
    for i, _ in enumerate(phi_0):
        integral = np.sum(u_init[:i]) * cfg.delta_x
        phi_0[i] = np.exp(-1.0 / (2.0 * cfg.nu) * integral)
    return phi_0


def heat_convolution(
        phi_0: np.ndarray,
        t: float,
        cfg: Config,
        x_grid: np.ndarray
        ) -> np.ndarray:
    """Convolute with the heat kernel."""
    if t == 0.0:
        return phi_0.copy()

    diff = x_grid[:, None] - x_grid[None, :]
    conv = np.exp(-diff**2 / (4.0 * cfg.nu * t))
    phi = (conv @ phi_0) * cfg.delta_x / np.sqrt(4.0 * np.pi * cfg.nu * t)
    return phi


def reverse_cole_hopf_trans(phi: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Recover u from phi.

    u(x, t) = -2\nu \frac{dphi}{phi}
    """
    u = -2.0 * cfg.nu * (np.gradient(phi, cfg.delta_x) / phi)
    return u


def solve_burgers_padded(
        u_init: np.ndarray,
        t: float,
        cfg: Config,
        pad: int
        ) -> np.ndarray:
    """Solve the Burgers equation throught the Cole-Hopf transformation."""
    u_0_padded = np.pad(u_init, pad_width=pad, mode='edge')
    x_grid_padded = np.arange(len(u_0_padded)) * \
        cfg.delta_x - pad * cfg.delta_x

    phi_0 = cole_hopf_trans(u_0_padded, cfg)
    phi_t = heat_convolution(phi_0, t, cfg, x_grid_padded)
    u_full = reverse_cole_hopf_trans(phi_t, cfg)

    return u_full[pad:-pad]


def cole_hopf_grid(cfg: Config, pad: int = 200) -> tuple:
    """Generate a solution grid."""
    u_init = u_0(cfg)

    u_grid = np.zeros(shape=(cfg.n_t, cfg.n_x))
    u_grid[0] = u_init

    for i, t in tqdm(
        enumerate(np.linspace(cfg.delta_t, cfg.t_extrap, cfg.n_t - 1)),
        total=cfg.n_t - 1,
        desc="Computing ground truth solution with Cole-Hopf"
    ):
        u_grid[i + 1] = solve_burgers_padded(u_init, t, cfg, pad)

    # -- rescale rows for ICs with a known maximum -------------------------
    if cfg.ic in ("Step_up", "Step_down", "Slope"):
        u_max = np.max(np.abs(u_init))
        if u_max > 1e-12:
            for i in range(1, cfg.n_t):
                row_max = np.max(np.abs(u_grid[i]))
                if row_max > 1e-12:
                    u_grid[i] *= u_max / row_max

    t_arr = np.linspace(0, cfg.t_extrap, cfg.n_t)
    return u_grid, t_arr


# -- Numerical schemes ------------------------
def compute_stable_dt(u: np.ndarray, cfg: Config, cfl: float = 0.4) -> float:
    """
    Compute a stable time step satisfying both the advection CFL condition
    and the diffusion (von Neumann) stability condition.

    CFL (advection) : dt <= cfl * dx / max(|u|)
    Von Neumann (diffusion): dt <= dx^2 / (2 * nu)
    """
    dx = cfg.delta_x
    max_u = np.max(np.abs(u))

    dt_adv = cfl * dx / max_u if max_u > 1e-12 else np.inf
    dt_diff = 0.4 * dx**2 / cfg.nu   # safety factor 0.4 < 0.5

    return min(dt_adv, dt_diff)


def forward_euler(u: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Forward step using Euler method.
    """
    du = np.gradient(u, cfg.delta_x)
    d2u = np.gradient(du, cfg.delta_x)
    u_step = -u * du + cfg.nu * d2u

    return u + cfg.delta_t * u_step


def forward_euler_upwind(u: np.ndarray, dt: float, cfg: Config) -> np.ndarray:
    """
    One forward-Euler step with:
      - First-order upwind differencing for the advection term u*du/dx
      - Second-order central differencing for the diffusion term nu*d2u/dx2

    Periodic boundary conditions are used via np.roll.
    """
    dx = cfg.delta_x

    u_right = np.roll(u, -1)   # u[i+1]
    u_left = np.roll(u,  1)   # u[i-1]
    u_right[-1] = cfg.bc_right
    u_left[0] = cfg.bc_left

    # Upwind advection: use backward difference where u > 0, forward where u < 0  # noqa: E501
    adv_pos = u * (u - u_left) / dx   # u >= 0: backward difference
    adv_neg = u * (u_right - u) / dx   # u <  0: forward  difference

    advection = np.where(u >= 0, adv_pos, adv_neg)

    # Central diffusion
    diffusion = cfg.nu * (u_right - 2*u + u_left) / dx**2

    return u + dt * (-advection + diffusion)


def forward_lax_wendroff(u: np.ndarray, dt: float, cfg: Config) -> np.ndarray:
    """
    Lax-Wendroff scheme for Burgers' equation.
    Second-order accurate in space and time.
    Diffusion term handled with central differences.
    """
    dx = cfg.delta_x

    u_right = np.roll(u, -1)
    u_left = np.roll(u,  1)
    u_right[-1] = cfg.bc_right
    u_left[0] = cfg.bc_left

    # Lax-Wendroff for advection (second-order)
    f = 0.5 * u**2               # flux f(u) = u²/2
    f_right = 0.5 * u_right**2
    f_left = 0.5 * u_left**2

    c_right = 0.5 * (u + u_right)      # local wave speed at i+1/2
    c_left = 0.5 * (u + u_left)       # local wave speed at i-1/2

    flux_right = 0.5*(f + f_right) - 0.5*(dt/dx)*c_right**2*(u_right - u)
    flux_left = 0.5*(f + f_left) - 0.5*(dt/dx)*c_left**2 * (u - u_left)

    advection = (flux_right - flux_left) / dx

    # Central diffusion (second-order)
    diffusion = cfg.nu * (u_right - 2*u + u_left) / dx**2

    return u + dt * (-advection + diffusion)


def euler_method(
        u_init: np.ndarray,
        t_end: float,
        cfg: Config,
        auto_dt: bool = True,
        cfl: float = 0.4
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Burgers' equation using forward Euler + upwind advection.

    Parameters
    ----------
    u_init   : Initial condition u(x, 0).
    t_end    : Final simulation time.
    cfg      : Simulation configuration.
    auto_dt  : If True, recompute a stable dt every step (recommended).
               If False, use cfg.delta_t (make sure it satisfies CFL yourself).
    cfl      : CFL safety factor (default 0.4).

    Returns
    -------
    sol  : np.ndarray of shape (n_snapshots, n_x)  — solution snapshots
    t_arr: np.ndarray of shape (n_snapshots,)       — corresponding times
    """
    u = u_init.copy()
    t = 0.0
    sol = [u.copy()]
    t_arr = [0.0]

    while t < t_end:
        dt = compute_stable_dt(u, cfg, cfl) if auto_dt else cfg.delta_t
        dt = min(dt, t_end - t)
        u = forward_euler_upwind(u, dt, cfg)
        t += dt

        sol.append(u.copy())
        t_arr.append(t)

    return np.array(sol), np.array(t_arr)


def lax_wendroff(
        u_init: np.ndarray,
        t_end: float,
        cfg: Config,
        auto_dt: bool = True,
        cfl: float = 0.4
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Burgers' equation using forward Lax Wendroff.

    Parameters
    ----------
    u_init   : Initial condition u(x, 0).
    t_end    : Final simulation time.
    cfg      : Simulation configuration.
    auto_dt  : If True, recompute a stable dt every step (recommended).
               If False, use cfg.delta_t (make sure it satisfies CFL yourself).
    cfl      : CFL safety factor (default 0.4).

    Returns
    -------
    sol  : np.ndarray of shape (n_snapshots, n_x)  — solution snapshots
    t_arr: np.ndarray of shape (n_snapshots,)       — corresponding times
    """
    u = u_init.copy()
    t = 0.0
    sol = [u.copy()]
    t_arr = [0.0]

    while t < t_end:
        dt = compute_stable_dt(u, cfg, cfl) if auto_dt else cfg.delta_t
        dt = min(dt, t_end - t)
        u = forward_lax_wendroff(u, dt, cfg)
        t += dt

        sol.append(u.copy())
        t_arr.append(t)

    return np.array(sol), np.array(t_arr)


def gauss_solution_grid(cfg: Config) -> np.ndarray:
    """Lax-Wendroff gauss solution grid."""
    return lax_wendroff(gauss(cfg=cfg), t_end=cfg.t_extrap, cfg=cfg)


def step_up_solution_grid(cfg: Config) -> np.ndarray:
    """Lax-Wendroff step-up solution grid."""
    return lax_wendroff(step_up(cfg=cfg), t_end=cfg.t_extrap, cfg=cfg)


def n_wave_solution_grid(cfg: Config) -> np.ndarray:
    """Lax-Wendroff n-wave solution grid."""
    return lax_wendroff(n_wave(cfg=cfg), t_end=cfg.t_extrap, cfg=cfg)


def n_wave_chop_solution_grid(cfg: Config) -> np.ndarray:
    """Lax-Wendroff n-wave chopped solution grid."""
    return lax_wendroff(n_wave_chop(cfg=cfg), t_end=cfg.t_extrap, cfg=cfg)


def slope_solution_grid(cfg: Config) -> np.ndarray:
    """Lax-Wendroff slope solution grid."""
    return lax_wendroff(negative_slope(cfg=cfg), t_end=cfg.t_extrap, cfg=cfg)


# -- Interpolation ------------------------
def interpolate_solution(
        sol: np.ndarray,
        t_arr: np.ndarray,
        x: float,
        t: float,
        cfg: Config
        ) -> float:
    """
    Bilinear interpolation of the solution at physical coordinates (x, t).
    """
    n_t, n_x = sol.shape

    # Convert physical coords to fractional indices
    xi = x / cfg.delta_x
    ti = np.searchsorted(t_arr, t, side='right') - 1  # handles variable dt

    # Clamp
    xi = np.clip(xi, 0, n_x - 1)
    ti = np.clip(ti, 0, n_t - 2)

    # Integer neighbours
    x0, x1 = int(np.floor(xi)), min(int(np.floor(xi)) + 1, n_x - 1)
    t0, t1 = int(ti), min(int(ti) + 1, n_t - 1)

    # Fractional distances
    dx = xi - x0
    dt = (t - t_arr[t0]) / (t_arr[t1] - t_arr[t0]
                            ) if t_arr[t1] != t_arr[t0] else 0.0

    # Corner values
    f00, f10 = sol[t0, x0], sol[t0, x1]
    f01, f11 = sol[t1, x0], sol[t1, x1]

    return (
        f00 * (1 - dx) * (1 - dt) +
        f10 * dx * (1 - dt) +
        f01 * (1 - dx) * dt +
        f11 * dx * dt
    )


def interpolate_solution_arr(
        sol: np.ndarray,
        t_arr: np.ndarray,
        x: np.ndarray,
        t: np.ndarray,
        cfg: Config
        ) -> np.ndarray:
    """
    Vectorised wrapper around interpolate_solution for arrays of (x, t) pairs.
    """
    u_obs = []
    for x_i, t_i in zip(x, t):
        u_obs.append(interpolate_solution(sol, t_arr, x_i, t_i, cfg))
    return np.array(u_obs)


# -- Residuals ------------------------
def residual(sol: np.ndarray, t_arr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Compute PDE residual |ut + u*ux - nu*uxx| on the solution grid.
    Uses the time spacing from adaptive stepping.
    """
    dt_arr = np.diff(t_arr)                        # variable spacing
    # shape (n_t-1, 1) for broadcasting
    dt_col = dt_arr[:, np.newaxis]

    # Non-uniform time derivative: forward difference between consecutive steps
    ut = (sol[1:] - sol[:-1]) / dt_col            # shape (n_t-1, n_x)
    # evaluate u at midpoint in time
    u_mid = 0.5 * (sol[1:] + sol[:-1])

    ux = np.gradient(u_mid, cfg.delta_x, axis=1)
    uxx = np.gradient(ux,    cfg.delta_x, axis=1)

    return np.abs(ut + u_mid * ux - cfg.nu * uxx)


def rmse(sol: np.ndarray, t_arr: np.ndarray, cfg: Config) -> np.ndarray:
    """Pointwise RMSE"""
    return np.mean(residual(sol, t_arr, cfg) ** 2)


# -- Characteristics ------------------------
def predict_shock_time(cfg: Config) -> float:
    """
    Predict when shockwave will occur using method of characteristics.
    """
    u = u_0(cfg)
    du = np.gradient(u, cfg.delta_x)
    if np.all(du > 0):
        return float('Inf')
    else:
        return -1 / np.min(du)


# -- Plotting ------------------------
def plot_anim(sol: np.ndarray, plot_pause: float = 1) -> None:
    """
    Make an animation of the numerical solution.
    """
    n_t, n_x = sol.shape

    x = np.arange(n_x)

    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot(x, sol[0], lw=2)

    ax.set_xlim(0, n_x - 1)
    ax.set_ylim(sol.min(), sol.max())
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("Time evolution")

    ax.grid(True, alpha=0.3)

    def update(frame):
        line.set_ydata(sol[frame])
        ax.set_title(f"Time step {frame}/{n_t}")
        return line,

    _ = animation.FuncAnimation(
        fig,
        update,
        frames=n_t,
        interval=plot_pause,
        blit=True
    )

    plt.show()


if __name__ == "__main__":
    my_cfg = Config(ic="N_wave")
    my_cfg.height = 2
    u_init_example = u_0(my_cfg)
    plt.plot(u_init_example)
    plt.show()

    # # Upwind Euler
    # sol_upwind, t_arr_upwind = euler_method(u_init, cfg.t_extrap, cfg)
    # rmse_upwind = rmse(sol_upwind, t_arr_upwind, cfg)

    # # Lax-Wendroff
    # sol_lw, t_arr_lw = lax_wendroff(u_init, cfg.t_extrap, cfg)
    # rmse_lw = rmse(sol_lw, t_arr_lw, cfg)

    # # Cole-Hopf integral
    # sol_ch, t_arr_ch = cole_hopf_grid(cfg, pad=600)
    # rmse_ch = rmse(sol_ch, t_arr_ch, cfg)

    # print(f"RMSE Upwind Euler : {rmse_upwind:.6e}")
    # print(f"RMSE Lax-Wendroff : {rmse_lw:.6e}")
    # print(f"RMSE Cole-Hopf : {rmse_ch:.6e}")
