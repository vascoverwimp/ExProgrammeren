"""
model.py
========
Neural-network architecture, unified configuration, and inference wrapper
for the PINN damped-spring-mass system.

Import this module from both train.py and plot.py.  Nothing here does I/O,
touches files, or starts a training loop — it is pure definition.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn


# =============================================================================
# 1.  CONFIGURATION  — single source of truth for every parameter
# =============================================================================

@dataclass
class BurgerConfig:
    """
    All physical constants, data-generation settings, architecture choices,
    and training hyper-parameters in one place.

    Pass an instance of this class through the entire pipeline so that
    train.py, plot.py, and any future scripts always agree on the same
    values.  CLI arguments in train.py create a BurgerConfig and override
    individual fields before anything else runs.

    Physical model
    --------------
      udot(x,t) + u * u'(x,t) - v * u''(x,t) = 0
      u(x,t0) = y0,  u'(x,t0) = 0
      For the cases:
      N-wave: y0 = exp(-(x-1)**2/2) - exp(-(x+1)**2/2)
      Gaussian: y0 = exp(-x**2/2)
    """
    # ── Time domain ──────────────────────────────────────────────────────────
    t0:            float = 0    # start of observation window [s]
    t_train:        float = 2.0    # end of observation window   [s]
    t_extrap:       float = 3.0   # end of extrapolation window [s]

    # ── Space domain ─────────────────────────────────────────────────────────
    x_begin:  float = -7.0   # left boundary of observation window   [m]
    x_end:    float = 7.0    # right boundary of observation window  [m]

    # ── Physical parameters ──────────────────────────────────────────────────
    v:      float = 0.1      # viscosity  [m^2/s]

    # ── Initial conditions ───────────────────────────────────────────────────
    situation: str = "N-wave"  # "N-wave","Gaussian", or "Step"
    # ── Data ─────────────────────────────────────────────────────────────────
    n_obs_total:       int = 1000   # total noisy observations (before split)
    # bins for stratisfying validation split along time axis (unused)
    n_bins:            int = 20
    # fraction of observations held out for val (unused)
    val_fraction: float = 0.2
    # number of training observations to use per epoch (unused)
    n_obs_per_epoch: int = 250
    n_val: int = 200  # number of validation observations
    n_val_t: int = 100  # number of validation observations along time axis
    n_val_x: int = 200  # number of validation observations along space axis
    sigma:       float = 0.05  # measurement noise std dev
    # initial condition samples in x dimension (for IC loss)
    n_ic_samples_x: int = 200
    n_col:       int = 250   # collocation points to sample (physics residual)
    n_col_pool:  int = 10000  # pool of collocation points
    seed:        int = 42    # global RNG seed

    # ── Plotting sampling ────────────────────────────────────────────────────
    # spatial resolution for all plots (including snapshots)
    n_plot_x: int = 500
    # temporal resolution for all plots (including snapshots)
    n_plot_t: int = 200

    # ── PINN loss weights ────────────────────────────────────────────────────
    lambda_phys: float = 0.7943282347242815  # physics-residual weight
    # initial-condition weight (>> lambda_phys)
    lambda_ic:   float = 1.2589254117941673

    # ── Network architecture ─────────────────────────────────────────────────
    hidden:   int = 48   # neurons per hidden layer
    n_layers: int = 6    # number of hidden layers

    # ── Optimiser ────────────────────────────────────────────────────────────
    beta1:    float = 0.9    # Adam beta1
    beta2:    float = 0.999  # Adam beta2
    lr:       float = 0.0018077686769634343   # initial Adam learning rate
    lr_step:  int = 3000   # StepLR: decay every this many epochs
    lr_gamma: float = 0.5    # StepLR: multiplicative factor

    # ── Training loop ────────────────────────────────────────────────────────
    n_epochs:    int = 40_000   # maximum training epochs
    print_every: int = 1_000   # console log frequency (epochs)
    log_every:   int = 100     # history-dict write frequency (epochs)

    # ── Early stopping ───────────────────────────────────────────────────────
    patience:  int = 30     # patience in units of log_every
    min_delta: float = 1e-6   # minimum improvement to reset the counter

    # ── Snapshot epochs for trajectory plots ─────────────────────────────────
    snapshot_epochs: list = field(
        default_factory=lambda: [1, 50, 200, 1_000, 8_000, 20_000, 40_000]
    )

    # ── Output paths ─────────────────────────────────────────────────────────
    # directory for analytic solution data
    dir_analytic: str = "./AnalyticBurger"
    suffix_analytic_pt: str = "analytic_solution.npz"  # numpy bundle
    # directory for all saved files
    out_dir:    str = "./BurgersPINN/Output"
    suffix_results_pt: str = "training_results.pt"  # torch.save bundle
    # best-so-far extended physics PINN checkpoint
    suffix_ckpt_pinn_ext_phys:  str = "best_pinn_ext_phys.pt"
    # best-so-far blind PINN checkpoint
    suffix_ckpt_pinn_blind:    str = "best_pinn_blind.pt"
    # best-so-far standard ML checkpoint
    suffix_ckpt_ml:    str = "best_ml.pt"

    # ── Derived quantities (read-only) ───────────────────────────────────────
    @property
    def ic_func(self):
        """Initial condition function u(x,t0) based on the chosen situation.

        Returns:
            A callable that takes spatial coordinates and returns initial values.
        """
        if self.situation == "N-wave":
            return lambda x: torch.where(
                torch.abs(x) < 1.0,
                -x,
                torch.zeros_like(x)
            )

        if self.situation == "Gaussian":
            return lambda x: torch.exp(-x**2 / 2)

        if self.situation == "Step":
            return lambda x: torch.where(
                x > 0,
                torch.ones_like(x),
                torch.zeros_like(x)
            )

        raise ValueError(f"Unknown situation: {self.situation}")

    @property
    def inviscid_shockwave_time(self) -> float:
        """Time of shockwave formation in the inviscid (v=0) case.

        For the inviscid Burgers equation, shockwaves form when characteristics
        intersect at t = 1 / max(-du/dx). Returns infinity if no shockwave forms.

        Returns:
            Time of shockwave formation as a float, or infinity if no shockwave.
        """
        # For the inviscid Burgers equation, shockwaves form when
        # characteristics intersect which happens at t = 1 / max(-du/dx)
        # where du/dx is the spatial derivative of the initial condition.
        # The max of -du/dx occurs at the point where du/dx is most negative,
        # which is the steepest downward slope in the initial condition.
        # For the N-wave and Gaussian initial conditions, this can be found
        # analytically, while for the Step function, no shockwave forms since
        # it's non-decreasing.
        if self.situation == "N-wave":
            return 1.0  # Shockwave forms at t=1 (max = 1)
        if self.situation == "Gaussian":
            # 1/max(x exp(-x^2/2)) = 1/(1 * exp(-1/2)) = e^(1/2)
            # Shockwave forms at t=e^(1/2) for the Gaussian initial condition
            return float(np.exp(0.5))
        if self.situation == "Step":
            # No shockwave forms if the initial condition is non-decreasing
            return float('inf')

        raise ValueError(f"Unknown situation: {self.situation}")

    def abs_path(self, filename: str) -> Path:
        """Return an absolute Path for a file stored in out_dir.

        Args:
            filename: Name of the file to locate in the output directory.

        Returns:
            Path object pointing to the file in out_dir.
        """
        return Path(self.out_dir) / filename


# =============================================================================
# 2.  NEURAL NETWORK ARCHITECTURE
# =============================================================================

class FCNet(nn.Module):
    """
    Fully-connected network:  1  →  [hidden × Tanh] × n_layers  →  1

    Tanh activations are mandatory.  ReLU has a zero second derivative
    everywhere, which zeroes out the physics-residual gradient entirely.
    Tanh is smooth and its higher-order derivatives are non-trivial.

    The model is always constructed on CPU; move it to the target device
    with .to(device) in the training script.
    """

    def __init__(self, hidden: int = 32, n_layers: int = 4) -> None:
        """Initialize fully-connected network with Tanh activations.

        Args:
            hidden: Number of neurons per hidden layer (default 32).
            n_layers: Number of hidden layers (default 4).
        """
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """Forward pass through the network.

        Args:
            x: Spatial coordinates tensor of shape (N, 1).
            t: Temporal coordinates tensor of shape (N, 1).

        Returns:
            Network output tensor of shape (N, 1) representing u(x, t).
        """
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

    def param_count(self) -> int:
        """Count total number of trainable parameters in the network.

        Returns:
            Total parameter count as an integer.
        """
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: BurgerConfig) -> "FCNet":
        """Convenience constructor that reads architecture from a config.

        Args:
            cfg: BurgerConfig instance containing architecture parameters.

        Returns:
            FCNet instance with architecture specified in the config.
        """
        return cls(hidden=cfg.hidden, n_layers=cfg.n_layers)


# =============================================================================
# 3.  PREDICTOR  — inference wrapper
# =============================================================================

class Predictor:
    """
    Stateless CPU inference wrapper around FCNet.

    Usage
    -----
    >>> pred = Predictor(cfg, checkpoint_path="best_pinn.pt")
    >>> y = pred.predict(np.linspace(0, 10, 500))

    The Predictor always operates on CPU so that numpy interop is trivial
    (no .detach().numpy() dance for CUDA tensors).  Training happens on
    DEVICE in train.py; the Predictor is only used after training.
    """

    def __init__(
        self,
        cfg: BurgerConfig,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Initialize the inference wrapper with model and optional checkpoint.

        Args:
            cfg: BurgerConfig instance with model architecture parameters.
            checkpoint_path: Optional path to a saved checkpoint file to load weights from.
        """
        self.cfg = cfg
        self.model = FCNet.from_config(cfg)
        if checkpoint_path is not None:
            self.load(checkpoint_path)
        self.model.eval()

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self, path: str) -> None:
        """Load weights from a checkpoint file saved by train.py.

        Args:
            path: Path to the checkpoint file containing model state_dict.
        """
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        state = bundle.get("model_state_dict", bundle)
        self.model.load_state_dict(state)
        self.model.eval()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load directly from an in-memory state_dict (e.g. a snapshot).

        Args:
            state_dict: Dictionary containing model parameters to load.
        """
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Run forward pass on numpy space and time arrays (same length).

        Performs inference without gradient computation on the given spatial
        and temporal coordinates.

        Args:
            x: Spatial coordinates array of shape (N,).
            t: Temporal coordinates array of shape (N,).

        Returns:
            Network predictions of shape (N,) as a numpy array.
        """
        t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            return self.model(x_t, t_t).squeeze().numpy()

    def predict_from_state(
        self, state_dict: dict, x: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Temporarily load a snapshot state_dict and predict, then restore original weights.

        Used to replay epoch snapshots without allocating a separate Predictor per snapshot.
        Restores original weights after prediction, leaving the predictor unchanged.

        Args:
            state_dict: Dictionary containing model parameters of the snapshot.
            x: Spatial coordinates array of shape (N,).
            t: Temporal coordinates array of shape (N,).

        Returns:
            Network predictions of shape (N,) using the snapshot weights.
        """
        original = copy.deepcopy(self.model.state_dict())
        try:
            self.load_state_dict(state_dict)
            return self.predict(x, t)
        finally:
            self.model.load_state_dict(original)
            self.model.eval()

    # ── Metrics ──────────────────────────────────────────────────────────────

    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        """Compute root mean square error between predictions and true values.

        Args:
            pred: Predicted values array.
            true: Ground truth values array.

        Returns:
            RMSE value as a float.
        """
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    @staticmethod
    def physics_residual(
        u: np.ndarray, x: np.ndarray, t: np.ndarray, cfg: BurgerConfig
    ) -> float:
        """Approximate ODE residual via numpy central finite differences.

        Computes the physics residual of the Burgers equation by taking numerical
        derivatives. Trims 5 boundary points on each side where finite-diff is inaccurate.

        Args:
            u: Solution values array.
            x: Spatial coordinates array.
            t: Temporal coordinates array.
            cfg: BurgerConfig instance containing physical parameters (viscosity v).

        Returns:
            Root mean square physics residual as a float.
        """

        t_vec = np.unique(t)

        x_vec = np.unique(x)

        n_t = len(t_vec)

        n_x = len(x_vec)
        u_grid = u.reshape(n_t, n_x)
        du = np.gradient(u_grid, x_vec, axis=1)
        d2u = np.gradient(du, x_vec, axis=1)
        dotu = np.gradient(u_grid, t_vec, axis=0)

        r = dotu + u_grid * du - cfg.v * d2u
        return float(np.sqrt(np.mean(r[5:-5, 5:-5] ** 2)))
