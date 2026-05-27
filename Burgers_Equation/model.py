"""
Neural network architecture and inference helpers for the Burgers equation PINN.  # noqa:E501

Contains:
    FCNet        -- fully-connected network with Tanh activations
    predict      -- predict y-values for array of t-values
    predict_from_state -- restore a snapshot and predict

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import torch
import torch.nn as nn
import numpy as np
from config import Config


class FCNet(nn.Module):
    """
    Fully-connected network: 2 -> [hidden]*n_layers -> 1
    Tanh activations throughout.

    Tanh is mandatory (not ReLU) because we differentiate the network output
    twice via autograd.  ReLU has zero second derivative everywhere, which
    would make the physics residual carry no gradient signal.

    The model is instantiated on CPU and moved to DEVICE via .to(DEVICE)
    after construction (see training section).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers = [nn.Linear(2, cfg.hidden), nn.Tanh(),
                  nn.Dropout(cfg.dropout_rate)]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden),
                       nn.Tanh(), nn.Dropout(cfg.dropout_rate)]
        layers += [nn.Linear(cfg.hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(torch.cat([t, x], dim=1))

    def param_count(self) -> int:
        """Return parameter count."""
        return sum(p.numel() for p in self.parameters())


class InverseFCNet(nn.Module):
    """
    Inverse Fully-connected network: 2 -> [hidden]*n_layers -> 1
    Tanh activations throughout. Can be used to predict equation parameters

    Tanh is mandatory (not ReLU) because we differentiate the network output
    twice via autograd.  ReLU has zero second derivative everywhere, which
    would make the physics residual carry no gradient signal.

    The model is instantiated on CPU and moved to DEVICE via .to(DEVICE)
    after construction (see training section).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers = [nn.Linear(2, cfg.hidden), nn.Tanh(),
                  nn.Dropout(cfg.dropout_rate)]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden),
                       nn.Tanh(), nn.Dropout(cfg.dropout_rate)]
        layers += [nn.Linear(cfg.hidden, 1)]
        self.net = nn.Sequential(*layers)

        nu_init = np.random.uniform(0.5, 2.0)   # random start
        nu_init = float(np.log(np.exp(nu_init) - 1.0))   # softplus inverse
        self._nu_raw = nn.Parameter(
            torch.tensor([nu_init], requires_grad=True))

    @property
    def nu_hat(self):
        """Positive viscosity."""
        return torch.nn.functional.softplus(self._nu_raw)  # always > 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(torch.cat([t, x], dim=1))

    def param_count(self) -> int:
        """Return parameter count."""
        return sum(p.numel() for p in self.parameters())


def predict(model: nn.Module, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Run inference on CPU.  The model must already be on CPU.
    No gradients needed -- torch.no_grad() saves memory and time.
    """
    model.eval()
    t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return model(t_t, x_t).squeeze().numpy()


def predict_from_state(
        state_dict: dict,
        t: np.ndarray,
        x: np.ndarray,
        cfg: Config
        ) -> np.ndarray:
    """
    Restore a CPU snapshot and predict.
    Snapshots were saved as CPU state_dicts in train(), so no device
    transfer is needed here.
    """
    tmp = FCNet(cfg)
    tmp.load_state_dict(state_dict)
    tmp.eval()
    return predict(tmp, t, x)
