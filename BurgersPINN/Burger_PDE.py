"""
Burgers' Equation PDE Solver — Fully Implicit Crank-Nicolson + Newton
======================================================================
Solves the viscous Burgers' equation:

    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

Time discretisation: Crank-Nicolson applied to BOTH terms simultaneously.

    u^{n+1} - u^n     1 [ R(u^{n+1}) + R(u^n) ]
    ─────────────── = ─ [─────────────────────── ]
           dt         2

where R(u) = −u ∂u/∂x + ν ∂²u/∂x²

This gives the nonlinear residual at each step:

    F(v) = v − u^n − (dt/2) · [R(v) + R(u^n)] = 0,   v ≡ u^{n+1}

which is solved with Newton's method:

    J_F(v^k) · δ = −F(v^k)
    v^{k+1}       = v^k + δ

where the analytical Jacobian is:

    J_F = I − (dt/2) · J_R(v)

and J_R is built from the exact derivatives of the upwind advection stencil
and the diffusion stencil (see _build_jac_rhs for details).

Boundary conditions: Dirichlet — endpoints held at their initial values.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BurgersSolver:
    """
    Fully implicit Crank-Nicolson solver for the 1-D viscous Burgers' equation.

    Parameters
    ----------
    u0 : np.ndarray
        Initial condition on a uniform N-point grid spanning [x_start, x_end].
    t_start : float
        Start time.
    t_end : float
        End time.
    dt : float
        Fixed time-step.  Unconditionally stable for all dt > 0; accuracy
        still requires a reasonable choice relative to the wave speed.
    x_bounds : Tuple[float, float]
        (x_start, x_end) spatial domain boundaries.
    viscosity : float
        Kinematic viscosity ν ≥ 0.  Default 0.01.
    newton_tol : float
        Infinity-norm convergence tolerance for Newton iterations.  Default 1e-10.
    newton_max_iter : int
        Maximum Newton iterations per time step.  Default 50.

    Attributes (set after solve())
    --------------------------------
    x : np.ndarray   Spatial grid, shape (N,).
    t : np.ndarray   Snapshot times, shape (n_steps+1,).
    u : np.ndarray   Solution snapshots, shape (n_steps+1, N).
    newton_iters : np.ndarray  Newton iteration count at each step.
    """

    u0:             np.ndarray
    t_start:        float
    t_end:          float
    dt:             float
    x_bounds:       Tuple[float, float]
    viscosity:       float

    newton_tol:      float = 1e-10
    newton_max_iter: int = 50

    x:            np.ndarray = field(init=False, repr=False)
    t:            np.ndarray = field(init=False, repr=False)
    u:            np.ndarray = field(init=False, repr=False)
    newton_iters: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.viscosity < 0:
            raise ValueError("viscosity must be non-negative.")
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be strictly greater than t_start.")
        if self.x_bounds[1] <= self.x_bounds[0]:
            raise ValueError(
                "x_bounds[1] must be strictly greater than x_bounds[0].")
        if self.u0.ndim != 1 or len(self.u0) < 3:
            raise ValueError("u0 must be a 1-D array with at least 3 points.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        self.x = np.linspace(self.x_bounds[0], self.x_bounds[1], len(self.u0))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dx(self) -> float:
        return np.ptp(self.x) / (len(self.x) - 1)

    def _rhs(self, u: np.ndarray) -> np.ndarray:
        """
        Full spatial RHS:  R(u) = −u ∂u/∂x  +  ν ∂²u/∂x²

        Advection : upwind (1st-order), direction set by local sign of u.
        Diffusion : 2nd-order central via np.diff(n=2).
        Boundaries: tendency forced to 0 (Dirichlet).
        """
        dx = self._dx()

        # Upwind advection
        du_bwd = np.diff(u, prepend=u[0]) / dx
        du_fwd = np.diff(u, append=u[-1]) / dx
        adv = np.where(u >= 0, u * du_bwd, u * du_fwd)

        # Central diffusion
        d2u = np.diff(u, n=2, prepend=u[0], append=u[-1]) / dx**2

        R = -adv + self.viscosity * d2u
        R[np.array([0, -1])] = 0.0
        return R

    def _residual(self, v: np.ndarray, u_n: np.ndarray,
                  rn: np.ndarray, dt: float) -> np.ndarray:
        """
        Crank-Nicolson residual:

            F(v) = v − u_n − (dt/2) · [R(v) + R(u_n)]

        Boundary rows reduce to the Dirichlet constraint  v_i − u_n_i = 0
        (which is automatically satisfied since R zeroes its boundary entries,
        but is stated explicitly for clarity).
        """
        return v - u_n - (dt / 2.0) * (self._rhs(v) + rn)

    def _build_jac_rhs(self, v: np.ndarray) -> np.ndarray:
        """
        Analytical Jacobian of R(v) w.r.t. v,  J_R = ∂R/∂v.

        Diffusion block (tridiagonal, independent of v):
            ∂(ν d²u/dx²)_i / ∂v_j  =  ν/dx² · [δ_{j,i-1} − 2δ_{ji} + δ_{j,i+1}]

        Advection block (also tridiagonal, depends on v via upwind direction):
            v_i ≥ 0  →  A_i = −v_i(v_i − v_{i-1})/dx
                ∂A_i/∂v_i   = −(2v_i − v_{i-1})/dx    (main diagonal)
                ∂A_i/∂v_{i-1} =  v_i/dx                (sub-diagonal)

            v_i < 0  →  A_i = −v_i(v_{i+1} − v_i)/dx
                ∂A_i/∂v_i   = (2v_i − v_{i+1})/dx     (main diagonal)
                ∂A_i/∂v_{i+1} = −v_i/dx               (super-diagonal)

        Boundary rows are zeroed (Dirichlet tendency is identically 0).
        """
        N = len(v)
        dx = self._dx()
        r = self.viscosity / dx**2

        # ---- diffusion Jacobian ----
        J = (np.diag(np.full(N, -2.0 * r))
             + np.diag(np.full(N - 1,  r), k=1)
             + np.diag(np.full(N - 1,  r), k=-1))

        # ---- advection Jacobian ----
        # bool mask, shape (N,)
        pos = v >= 0

        # v_{i-1}, ghost at left
        v_left = np.concatenate([[v[0]],  v[:-1]])
        # v_{i+1}, ghost at right
        v_right = np.concatenate([v[1:],  [v[-1]]])

        # Main diagonal: combines both cases via np.where
        main_adv = np.where(pos,
                            -(2.0 * v - v_left) / dx,   # pos branch
                             (2.0 * v - v_right) / dx)   # neg branch
        J += np.diag(main_adv)

        # Sub-diagonal (row i, col i-1): non-zero only when v_i >= 0
        # np.diag(arr, k=-1)[i] = arr[i] placed at J[i+1, i], so arr[j] = ∂A_{j+1}/∂v_j
        sub_adv = np.where(pos[1:],   v[1:] / dx,  0.0)
        J += np.diag(sub_adv, k=-1)

        # Super-diagonal (row i, col i+1): non-zero only when v_i < 0
        # np.diag(arr, k=+1)[i] = arr[i] placed at J[i, i+1]
        sup_adv = np.where(~pos[:-1], -v[:-1] / dx, 0.0)
        J += np.diag(sup_adv, k=1)

        # Dirichlet: zero out boundary rows (tendency is always 0 there)
        J[np.array([0, -1]), :] = 0.0
        return J

    def _build_jac_F(self, v: np.ndarray, dt: float) -> np.ndarray:
        """
        Jacobian of the CN residual F:

            J_F = I − (dt/2) · J_R(v)

        Boundary rows → identity (from the Dirichlet constraint v_i − u_n_i = 0).
        """
        N = len(v)
        J_F = np.eye(N) - (dt / 2.0) * self._build_jac_rhs(v)

        # Enforce Dirichlet rows: ∂(v_i − u_n_i)/∂v_j = δ_{ij}
        J_F[0, :] = 0.0
        J_F[0,  0] = 1.0
        J_F[-1, :] = 0.0
        J_F[-1, -1] = 1.0
        return J_F

    def _newton_solve(self, u_n: np.ndarray, dt: float) -> Tuple[np.ndarray, int]:
        """
        Solve F(v) = 0 by Newton's method, starting from v⁰ = u_n.

        Returns
        -------
        v : np.ndarray   Converged solution u^{n+1}.
        k : int          Number of iterations taken.
        """
        v = u_n.copy()
        rn = self._rhs(u_n)   # R(u^n) is constant throughout the Newton loop

        for k in range(1, self.newton_max_iter + 1):
            F = self._residual(v, u_n, rn, dt)

            if np.linalg.norm(F, np.inf) < self.newton_tol:
                return v, k

            J = self._build_jac_F(v, dt)
            dv = np.linalg.solve(J, -F)
            v = np.add(v, dv)

        raise RuntimeError(
            f"Newton failed to converge in {self.newton_max_iter} iterations "
            f"(residual = {np.linalg.norm(F, np.inf):.2e}).  "
            "Try reducing dt or increasing newton_max_iter."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> "BurgersSolver":
        """
        Advance the PDE using fully implicit Crank-Nicolson + Newton iteration.

        Stores a snapshot at t_start and after every dt step up to t_end.

        Returns
        -------
        self  (for method chaining)
        """
        u = self.u0.copy().astype(float)
        t = float(self.t_start)

        snapshots_u = [u.copy()]
        snapshots_t = [t]
        iters = []

        while t < self.t_end:
            dt_step = float(np.minimum(self.dt, self.t_end - t))

            u, n_iter = self._newton_solve(u, dt_step)
            iters.append(n_iter)
            t = round(t + dt_step, 14)

            snapshots_u.append(u.copy())
            snapshots_t.append(t)

        self.u = np.array(snapshots_u)
        self.t = np.array(snapshots_t)
        self.newton_iters = np.array(iters)
        return self

    def solution_at(self, x_query: float, t_query: float) -> float:
        """
        Bilinear interpolation: query u at any (x, t) within the solved domain.

        Parameters
        ----------
        x_query : float   Spatial location in [x_bounds[0], x_bounds[1]].
        t_query : float   Time in [t_start, t_end].

        Returns
        -------
        float — u(x_query, t_query).
        """
        if not hasattr(self, "u"):
            raise RuntimeError("Call solve() before querying the solution.")
        if not (self.t[0] <= t_query <= self.t[-1]):
            raise ValueError(
                f"t_query={t_query} outside [{self.t[0]}, {self.t[-1]}].")
        if not (self.x_bounds[0] <= x_query <= self.x_bounds[1]):
            raise ValueError(f"x_query={x_query} outside {self.x_bounds}.")

        # --- interpolate in time ---
        idx_t = np.clip(np.searchsorted(self.t, t_query), 1, len(self.t) - 1)
        alpha = float(
            np.interp(t_query, self.t[idx_t - 1: idx_t + 1], [0.0, 1.0]))
        u_snap = np.add(
            (1.0 - alpha) * self.u[idx_t - 1], alpha * self.u[idx_t])

        # --- interpolate in space ---
        idx_x = np.clip(np.searchsorted(self.x, x_query), 1, len(self.x) - 1)
        beta = float(
            np.interp(x_query, self.x[idx_x - 1: idx_x + 1], [0.0, 1.0]))
        return float(np.add((1.0 - beta) * u_snap[idx_x - 1], beta * u_snap[idx_x]))

    def save(self, filepath: str) -> None:
        """
        Save the full solver state to a compressed .npz file.

        Stores all constructor parameters (scalars and arrays) alongside
        the post-solve fields (x, t, u, newton_iters) so the object can
        be reconstructed exactly with BurgersSolver.load().

        Parameters
        ----------
        filepath : str
            Destination path.  A .npz extension is added automatically
            by np.savez_compressed if not already present.
        """
        if not hasattr(self, "u"):
            raise RuntimeError("Call solve() before saving.")

        np.savez_compressed(
            filepath,
            # --- constructor args ---
            u0=self.u0,
            t_start=self.t_start,
            t_end=self.t_end,
            dt=self.dt,
            x_bounds=self.x_bounds,
            viscosity=self.viscosity,
            newton_tol=self.newton_tol,
            newton_max_iter=self.newton_max_iter,
            # --- computed fields ---
            x=self.x,
            t=self.t,
            u=self.u,
            newton_iters=self.newton_iters,
        )

    @classmethod
    def load(cls, filepath: str) -> "BurgersSolver":
        """
        Load a solver instance previously saved with save().

        Parameters
        ----------
        filepath : str
            Path to the .npz file (with or without the .npz extension).

        Returns
        -------
        BurgersSolver with all fields restored — ready to call solution_at()
        without needing to re-run solve().
        """
        data = np.load(filepath if filepath.endswith(
            ".npz") else filepath + ".npz")

        obj = cls(
            u0=data["u0"],
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]),
            dt=float(data["dt"]),
            x_bounds=tuple(data["x_bounds"]),
            viscosity=float(data["viscosity"]),
            newton_tol=float(data["newton_tol"]),
            newton_max_iter=int(data["newton_max_iter"]),
        )

        obj.x = data["x"]
        obj.t = data["t"]
        obj.u = data["u"]
        obj.newton_iters = data["newton_iters"]
        return obj
