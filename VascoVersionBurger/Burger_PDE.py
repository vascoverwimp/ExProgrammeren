from __future__ import annotations
 
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class BurgersSolver:
    """
    Solver for the 1-D viscous Burgers' equation.
 
    Parameters
    ----------
    u0 : np.ndarray
        Initial condition evaluated on a uniform grid of N points that spans
        [x_start, x_end].  The length of the array fixes the spatial
        resolution.
    t_start : float
        Start time of the simulation.
    t_end : float
        End time of the simulation.
    dt: float
        Timestep of the simulation.
    x_bounds : Tuple[float, float]
        (x_start, x_end) — the left and right spatial boundaries.
    viscosity : float
        Kinematic viscosity ν ≥ 0.  Default is 0.01.
    cfl : float
        CFL safety factor used to compute the adaptive time-step.
        Values in (0, 1) keep the scheme stable.  Default is 0.4.
 
    Attributes (populated after calling solve())
    --------------------------------------------
    x : np.ndarray          Spatial grid (N points).
    t : np.ndarray          Time vector of all stored snapshots.
    u : np.ndarray          Solution array, shape (len(t), N).
    """
 
    u0: np.ndarray
    t_start: float
    t_end: float
    dt: float
    x_bounds: Tuple[float, float]
 
    # --- class properties ---
    viscosity: float = 0.01
    cfl: float = 0.4
 
    # --- populated by solve() ---
    x: np.ndarray = field(init=False, repr=False)
    t: np.ndarray = field(init=False, repr=False)
    u: np.ndarray = field(init=False, repr=False)
 
    def __post_init__(self) -> None:
        if self.viscosity < 0:
            raise ValueError("viscosity must be non-negative.")
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be strictly greater than t_start.")
        if self.x_bounds[1] <= self.x_bounds[0]:
            raise ValueError("x_bounds[1] must be strictly greater than x_bounds[0].")
        if self.u0.ndim != 1 or len(self.u0) < 3:
            raise ValueError("u0 must be a 1-D array with at least 3 points.")
 
        self.x = np.linspace(self.x_bounds[0], self.x_bounds[1], len(self.u0))
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _dx(self) -> float:
        # np.ptp = max - min; divide by number of intervals
        return np.ptp(self.x) / (len(self.x) - 1)
 
    def _rhs(self, u: np.ndarray) -> np.ndarray:
        """
        Evaluate  -u ∂u/∂x + ν ∂²u/∂x²  over the full grid.
 
        Advection  : upwind (1st-order) via np.diff with prepend / append.
                     np.diff(u, prepend=u[0])  → backward difference at every point
                     np.diff(u, append=u[-1])  → forward  difference at every point
                     np.where selects direction from the local wave speed.
        Diffusion  : np.gradient applied twice for ∂²u/∂x² (2nd-order central).
        Boundary   : Dirichlet — endpoints forced to zero tendency.
        """
        dx = self._dx()
 
        # ---- advection: upwind first-order differences ------------------
        # backward: (u[i] - u[i-1]) / dx  — stable when u > 0
        du_bwd = np.diff(u, prepend=u[0])  / dx
        # forward:  (u[i+1] - u[i]) / dx  — stable when u < 0
        du_fwd = np.diff(u, append=u[-1]) / dx
        adv = np.where(u >= 0, u * du_bwd, u * du_fwd)
 
        # ---- diffusion: two successive np.gradient calls ----------------
        # np.gradient uses 2nd-order central differences at interior points
        # and one-sided differences at boundaries (full-length output).
        d2u_dx2 = np.gradient(np.gradient(u, dx), dx)
 
        dudt = -adv + self.viscosity * d2u_dx2
 
        # Enforce Dirichlet boundaries (zero tendency at endpoints)
        dudt[np.array([0, -1])] = 0.0
        return dudt
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def solve(self, store_every: int = 1) -> "BurgersSolver":
        """
        Integrate the PDE forward in time using adaptive RK4.
 
        Parameters
        ----------
        store_every : int
            Save the solution every *store_every* time steps.
            Increase to reduce memory usage for long runs.
 
        Returns
        -------
        self  (for method chaining)
        """
        u = self.u0.copy().astype(float)
        t = float(self.t_start)
 
        snapshots_u = [u.copy()]
        snapshots_t = [t]
 
        step = 0
        while t < self.t_end:
            dt = np.minimum(self.dt, self.t_end - t)  # don't overshoot
 
            # --- classical RK4 ---
            k1 = self._rhs(u)
            k2 = self._rhs(u + 0.5 * dt * k1)
            k3 = self._rhs(u + 0.5 * dt * k2)
            k4 = self._rhs(u +       dt * k3)
            u  = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
 
            t    += dt
            step += 1
 
            if step % store_every == 0 or np.isclose(t, self.t_end):
                snapshots_u.append(u.copy())
                snapshots_t.append(t)
 
        self.u = np.array(snapshots_u)   # shape: (n_snapshots, N)
        self.t = np.array(snapshots_t)   # shape: (n_snapshots,)
        return self
 
    def solution_at(self, x_query: float, t_query: float) -> float:
        """
        Return the solution interpolated (in time) at *t_query*.
 
        Uses np.searchsorted to locate the bracketing snapshots and
        np.interp for the scalar blending weight.
 
        Parameters
        ----------
        x_query : float
            Spatial point at which the solution is requested.
        t_query : float
            Time at which the solution is requested.  Must lie within
            [t_start, t_end].
 
        Returns
        -------
        float  — the solution u(x_query, t_query) obtained by linear interpolation in time
        between the two nearest snapshots.  If x_query lies between grid points,
        the returned value is further linearly interpolated in space between the two nearest grid points.
        """
        if not hasattr(self, "u"):
            raise RuntimeError("Call solve() before querying the solution.")
        if not (self.t[0] <= t_query <= self.t[-1]):
            raise ValueError(
                f"t_query={t_query} is outside [{self.t[0]}, {self.t[-1]}]."
            )
        if not (self.x_bounds[0] <= x_query) or not (x_query <= self.x_bounds[1]):
            raise ValueError(
                f"x_query values must lie within [{self.x_bounds[0]}, {self.x_bounds[1]}]."
            )
 
        # np.searchsorted: index of first t > t_query; clamp into valid range
        idx_t = np.clip(np.searchsorted(self.t, t_query), 1, len(self.t) - 1)
        t0, t1 = self.t[idx_t - 1], self.t[idx_t]
 
        # np.interp maps t_query onto [0, 1] between t0 and t1
        alpha = float(np.interp(t_query, [t0, t1], [0.0, 1.0]))
        u_interp = np.add((1.0 - alpha) * self.u[idx_t - 1], alpha * self.u[idx_t])

        
        # np.searchsorted: index of first x > x_query; clamp into valid range
        idx_x = np.clip(np.searchsorted(self.x, x_query), 1, len(self.x) - 1)
        x0, x1 = self.x[idx_x - 1], self.x[idx_x]
        beta = float(np.interp(x_query, [x0, x1], [0.0, 1.0]))
        return np.add((1.0 - beta) * u_interp[idx_x - 1], beta * u_interp[idx_x])  # linear spatial interpolation at boundaries

 