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
from importlib.metadata import requires
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


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
    # ── Time domain ───────────────────────────────────────────────────────────
    t0 :            float = 0    # start of observation window [s]
    t_train:        float = 6.0    # end of observation window   [s]
    t_extrap:       float = 10.0   # end of extrapolation window [s]

    # ── Space domain ───────────────────────────────────────────────────────────
    x_begin:  float = -7.0   # left boundary of observation window   [m]
    x_end:    float = 7.0    # right boundary of observation window  [m]

    # ── Physical parameters ───────────────────────────────────────────────────
    v:      float = 0.1      # viscosity  [m^2/s]

    # ── Initial conditions ───────────────────────────────────────────────────
    situation: str = "Step"  # "N-wave","Gaussian", or "Step"
    # ── Data ─────────────────────────────────────────────────────────────────
    n_obs:       int    = 100   # total noisy observations (before split)
    val_fraction: float = 0.2  # fraction of observations held out for val
    sigma:       float  = 0.05  # measurement noise std dev
    n_ic_samples_x: int = 200  # initial condition samples in x dimension (for IC loss)
    n_col_x:       int  = 20   # collocation points (physics residual) in x dimension
    n_col_t:       int  = 20   # collocation points (physics residual) in t dimension
    seed:        int    = 42    # global RNG seed

    # ── PINN loss weights ─────────────────────────────────────────────────────
    lambda_phys: float = 1  # physics-residual weight
    lambda_ic:   float = 50.0  # initial-condition weight (>> lambda_phys)

    # ── Network architecture ──────────────────────────────────────────────────
    hidden:   int = 32   # neurons per hidden layer
    n_layers: int = 4    # number of hidden layers

    # ── Optimiser ────────────────────────────────────────────────────────────
    lr:       float = 1e-3   # initial Adam learning rate
    lr_step:  int   = 3000   # StepLR: decay every this many epochs
    lr_gamma: float = 0.5    # StepLR: multiplicative factor

    # ── Training loop ─────────────────────────────────────────────────────────
    n_epochs:    int = 80_000   # maximum training epochs
    print_every: int = 1_000   # console log frequency (epochs)
    log_every:   int = 100     # history-dict write frequency (epochs)

    # ── Early stopping ────────────────────────────────────────────────────────
    patience:  int   = 300     # patience in units of log_every
    min_delta: float = 1e-6   # minimum improvement to reset the counter

    # ── Snapshot epochs for trajectory plots ─────────────────────────────────
    snapshot_epochs: list = field(
        default_factory=lambda: [1, 50, 200, 500, 1_000, 2_000, 4_000, 8_000, 20_000, 40_000, 80_000]
    )

    # ── Output paths ──────────────────────────────────────────────────────────
    out_dir:    str = "./VascoVersionBurger/Output"                  # directory for all saved files
    results_pt: str = "training_results.pt"  # torch.save bundle (relative)
    ckpt_pinn:  str = "best_pinn.pt"         # best-so-far PINN checkpoint
    ckpt_ml:    str = "best_ml.pt"           # best-so-far standard ML checkpoint

    # ── Derived quantities (read-only) ────────────────────────────────────────
    @property
    def ic_func(self):
        """Initial condition function u(x,t0) based on the chosen situation."""
        if self.situation == "N-wave":
            return lambda x: torch.where(
                torch.abs(x) < 1.0,
                -x,
                torch.zeros_like(x)
            )

        elif self.situation == "Gaussian":
            return lambda x: torch.exp(-x**2 / 2)

        elif self.situation == "Step":
            return lambda x: torch.where(
                x > 0,  
                torch.ones_like(x),
                torch.zeros_like(x)
            )
        else:
            raise ValueError(f"Unknown situation: {self.situation}")

    @property
    def ic_der_func(self):
        """Initial condition function du/dt(x,t0) based on the chosen situation."""
        return lambda x: torch.zeros_like(x)

    def abs_path(self, filename: str) -> Path:
        """Return an absolute Path for a file stored in out_dir."""
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
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: BurgerConfig) -> "FCNet":
        """Convenience constructor that reads architecture from a config."""
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
        self.cfg = cfg
        self.model = FCNet.from_config(cfg)
        if checkpoint_path is not None:
            self.load(checkpoint_path)
        self.model.eval()

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, path: str) -> None:
        """Load weights from a checkpoint file saved by train.py."""
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        state = bundle.get("model_state_dict", bundle)
        self.model.load_state_dict(state)
        self.model.eval()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load directly from an in-memory state_dict (e.g. a snapshot)."""
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Run forward pass on a numpy space and time array (have to have the same length).  Returns a numpy array of
        the same length.  No gradient computation.
        """
        t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            return self.model(x_t, t_t).squeeze().numpy()

    def predict_from_state(
        self, state_dict: dict, x:np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Temporarily load a snapshot state_dict and predict, then restore
        the original weights.  Used by plot.py to replay epoch snapshots
        without allocating a separate Predictor per snapshot.
        """
        original = copy.deepcopy(self.model.state_dict())
        try:
            self.load_state_dict(state_dict)
            return self.predict(x,t)
        finally:
            self.model.load_state_dict(original)
            self.model.eval()

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    @staticmethod
    def physics_residual(
        u: np.ndarray, x:np.ndarray, t: np.ndarray, cfg: BurgerConfig
    ) -> float:
        """
        Approximate ODE residual via numpy central finite differences.
        Trims 5 boundary points on each side where finite-diff is inaccurate.
        """

        t_vec = np.unique(t)

        x_vec = np.unique(x)

        n_t = len(t_vec)
        n_x = len(x_vec)

        u_grid = u.reshape(n_t, n_x)

        du  = np.gradient(u_grid, x_vec,axis=1)
        d2u = np.gradient(du, x_vec,axis=1)
        dotu = np.gradient(u_grid, t_vec,axis=0)
        r   = dotu + u_grid * du - cfg.v * d2u
        return float(np.sqrt(np.mean(r[5:-5] ** 2)))