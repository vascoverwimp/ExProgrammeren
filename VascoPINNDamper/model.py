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
import torch.nn as nn


# =============================================================================
# 1.  CONFIGURATION  — single source of truth for every parameter
# =============================================================================

@dataclass
class DamperConfig:
    """
    All physical constants, data-generation settings, architecture choices,
    and training hyper-parameters in one place.

    Pass an instance of this class through the entire pipeline so that
    train.py, plot.py, and any future scripts always agree on the same
    values.  CLI arguments in train.py create a DamperConfig and override
    individual fields before anything else runs.

    Physical model
    --------------
      m * y''(t) + c * y'(t) + k * y(t) = 0
      y(0) = y0,  y'(0) = dy0

    Derived quantities (read-only properties)
    -----------------------------------------
      omega_0   undamped natural frequency  sqrt(k/m)
      zeta      damping ratio               c / (2 * sqrt(m*k))
      omega_d   damped natural frequency    omega_0 * sqrt(1 - zeta²)
    """

    # ── Physical parameters ───────────────────────────────────────────────────
    mass:      float = 1.0   # m  [kg]
    damping:   float = 0.5   # c  [N·s/m]
    stiffness: float = 4.0   # k  [N/m]
    y0:        float = 1.0   # initial displacement   y(0)
    dy0:       float = 0.0   # initial velocity       y'(0)

    # ── Time domain ───────────────────────────────────────────────────────────
    t_train:  float = 6.0    # end of observation window   [s]
    t_extrap: float = 10.0   # end of extrapolation window [s]

    # ── Data ─────────────────────────────────────────────────────────────────
    n_obs:       int   = 30    # total noisy observations (before split)
    n_bins:      int   = 5    # bins for stratisfying validation split along time axis (unused)
    n_obs_per_epoch: int = 10  # number of training observations to use per epoch (for stochasticity, unused)
    n_val:       int   = 200     # number of observations in the validation set, >> n_obs(for early stopping)
    val_fraction: float = 0.2  # fraction of observations held out for val (unused, we use n_val instead)
    sigma:       float = 0.05  # measurement noise std dev
    n_col:       int   = 100   # collocation points (physics residual)
    seed:        int   = 42    # global RNG seed

    # ── Plotting sampling ───────────────────────────────────────────────────────────────
    n_plot_t: int = 500  # temporal resolution for all plots (including snapshots)

    # ── PINN loss weights ─────────────────────────────────────────────────────
    lambda_phys: float = 0.3162  # physics-residual weight
    lambda_ic:   float = 10.0  # initial-condition weight (>> lambda_phys)

    # ── Network architecture ──────────────────────────────────────────────────
    hidden:   int = 32   # neurons per hidden layer
    n_layers: int = 4    # number of hidden layers

    # ── Optimiser ────────────────────────────────────────────────────────────
    beta1:    float = 0.9    # Adam beta1
    beta2:    float = 0.999  # Adam beta2
    lr:       float = 0.00316   # initial Adam learning rate
    lr_step:  int   = 3000   # StepLR: decay every this many epochs
    lr_gamma: float = 0.5    # StepLR: multiplicative factor

    # ── Training loop ─────────────────────────────────────────────────────────
    n_epochs:    int = 8_000   # maximum training epochs
    print_every: int = 1_000   # console log frequency (epochs)
    log_every:   int = 100     # history-dict write frequency (epochs)

    # ── Early stopping ────────────────────────────────────────────────────────
    patience:  int   = 30     # patience in units of log_every
    min_delta: float = 1e-6   # minimum improvement to reset the counter

    # ── Snapshot epochs for trajectory plots ─────────────────────────────────
    snapshot_epochs: list = field(
        default_factory=lambda: [1, 50, 200, 500, 1_000, 2_000, 4_000, 8_000]
    )

    # ── Output paths ──────────────────────────────────────────────────────────
    out_dir:    str = "./VascoPINNDamper/Output"                  # directory for all saved files
    suffix_results_pt: str = "training_results.pt"  # torch.save bundle 
    suffix_ckpt_pinn_ext_phys:  str = "best_pinn_ext_phys.pt"         # best-so-far extended physics PINN checkpoint
    suffix_ckpt_pinn_blind:    str = "best_pinn_blind.pt"           # best-so-far blind PINN checkpoint
    suffix_ckpt_ml:    str = "best_ml.pt"           # best-so-far standard ML checkpoint

    # ── Derived quantities (read-only) ────────────────────────────────────────
    @property
    def omega_0(self) -> float:
        return float(np.sqrt(self.stiffness / self.mass))

    @property
    def zeta(self) -> float:
        return float(self.damping / (2.0 * np.sqrt(self.mass * self.stiffness)))

    @property
    def omega_d(self) -> float:
        return float(self.omega_0 * np.sqrt(max(0.0, 1.0 - self.zeta ** 2)))

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
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: DamperConfig) -> "FCNet":
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
        cfg: DamperConfig,
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

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, t: np.ndarray) -> np.ndarray:
        """
        Run forward pass on a numpy time array.  Returns a numpy array of
        the same length.  No gradient computation.
        """
        t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            return self.model(t_t).squeeze().numpy()

    def predict_from_state(
        self, state_dict: dict, t: np.ndarray
    ) -> np.ndarray:
        """
        Temporarily load a snapshot state_dict and predict, then restore
        the original weights.  Used by plot.py to replay epoch snapshots
        without allocating a separate Predictor per snapshot.
        """
        original = copy.deepcopy(self.model.state_dict())
        try:
            self.load_state_dict(state_dict)
            return self.predict(t)
        finally:
            self.model.load_state_dict(original)
            self.model.eval()

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    @staticmethod
    def physics_residual(
        y: np.ndarray, t: np.ndarray, cfg: DamperConfig
    ) -> float:
        """
        Approximate ODE residual via numpy central finite differences.
        Trims 5 boundary points on each side where finite-diff is inaccurate.
        """
        dy  = np.gradient(y, t)
        d2y = np.gradient(dy, t)
        r   = cfg.mass * d2y + cfg.damping * dy + cfg.stiffness * y
        return float(np.sqrt(np.mean(r[5:-5] ** 2)))