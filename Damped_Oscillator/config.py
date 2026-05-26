"""
Configuration for the damped spring-mass PINN.

All physical constants, hyperparameters, and runtime settings
for the PINN are defined as a single config class.
Import and instantiate Config in every script that needs these values.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    """Config class containing all parameters for model training."""
    # Physical parameters
    m: float = 1.0       # mass  [kg]
    c: float = 0.5       # damping coefficient
    k: float = 4.0       # spring stiffness  [N/m]

    @property
    def omega_0(self) -> float:
        """Return omega_0."""
        return float(np.sqrt(self.k / self.m))

    @property
    def zeta(self) -> float:
        """Return zeta."""
        return float(self.c / (2 * np.sqrt(self.m * self.k)))

    @property
    def omega_d(self) -> float:
        """Return omega_d."""
        return float(self.omega_0 * np.sqrt(max(0.0, 1.0 - self.zeta**2)))

    # Initial conditions
    y0: float = 1.0                 # initial displacement
    dy0: float = 0.0                # initial velocity

    # Time domain
    t_dom: float = 6.0              # End time of known domain
    t_extrap: float = 10.0          # End time of extrapolated domain

    # Loss weights
    use_physics: bool = True        # Use physics loss terms
    lambda_phys: float = 1e-2       # Physics loss weight
    use_ic: bool = True             # Use initial condition terms
    lambda_ic: float = 1e1          # Initial condition loss weight
    train_extrap: bool = True       # Generate collocation points in the extrapolated domain  # noqa: E501
    use_data: bool = True           # Generate training observation points

    # Data generation parameters
    n_obs: float = 15                   # Noisy observation points
    n_val: float = 200                  # Validation points
    randomise_observation: bool = True  # Randomise observation points every epoch  # noqa: E501

    n_col_dom: float = 500              # ODE residual collocation points
    sigma: float = 0.05                 # Standard deviation for n_obs
    randomise_collocation: bool = True  # Randomise collocation points every epoch  # noqa: E501

    # Network architecture
    hidden: int = 64                    # Network layer node count
    n_layers: int = 5                   # Network layer count

    # Hyperparameters
    n_epochs: int = 30000               # Number of epochs
    lr: float = 5e-3                    # Starting learning rate
    lr_inverse: float = 2e-2            # Learning rate for inverse problem; parameter estimation  # noqa: E501
    patience: int = 1000                # Patience for model stop
    patience_threshold: float = 1e-5    # Patience threshold for new best model
    dropout_rate: float = 0.0           # Dropout rate for every layer
    adam_beta1: float = 0.9             # Adam optimiser: moment hyperparameter
    adam_beta2: float = 0.999           # Adam optimiser: RMSprop hyperparameter  # noqa: E501
    scheduler_gamma: float = 0.5        # StepRL: scheduler decay factor
    scheduler_step: int = 3000          # StepRL: decay after this many epochs

    # Epoch snapshots
    snapshot_epochs: tuple = (1, 50, 300, 1000, 2000, 4000, 10000, 20000)
    log_every: int = 50
