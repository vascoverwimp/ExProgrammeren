"""
==============================================================================
Physics-Informed Neural Network (PINN) for a Damped Spring-Mass System
==============================================================================

PROBLEM
-------
We observe a few noisy measurements of the displacement y(t) of a mass on a
damped spring.  The true dynamics are governed by the second-order linear ODE

        m * y''(t)  +  c * y'(t)  +  k * y(t)  =  0

with initial conditions  y(0) = 1,  y'(0) = 0.

We compare two approaches:
  1. Standard supervised ML  – a neural network trained only to minimise the
     RMSE on the observed data points.
  2. PINN                    – the same network architecture, but the loss
     also penalises violations of the ODE at collocation points.

DEPENDENCIES
------------
    pip install torch numpy matplotlib

Run:
    python pinn_spring_damper.py

The script prints training progress and produces a single figure with six
panels that tell the complete story.
==============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1.  SYSTEM PARAMETERS
# =============================================================================
M   = 1.0       # mass  [kg]
C   = 0.5       # damping coefficient
K   = 4.0       # spring stiffness  [N/m]
Y0  = 1.0       # initial displacement
DY0 = 0.0       # initial velocity

# Derived quantities (underdamped regime assumed)
OMEGA_0 = np.sqrt(K / M)                          # natural frequency
ZETA    = C / (2.0 * np.sqrt(M * K))              # damping ratio  (< 1 here)
OMEGA_D = OMEGA_0 * np.sqrt(1.0 - ZETA**2)        # damped frequency

T_TRAIN = 6.0    # end of observation window  [s]
T_EXTRAP = 10.0  # end of extrapolation window [s]

# =============================================================================
# 2.  ANALYTIC SOLUTION  (ground truth)
# =============================================================================

def analytic(t: np.ndarray) -> np.ndarray:
    """
    Closed-form solution for the underdamped case with y(0)=1, y'(0)=0:

        y(t) = e^{-ζ ω₀ t} * [ cos(ω_d t)  +  (ζ ω₀ / ω_d) sin(ω_d t) ]
    """
    return np.exp(-ZETA * OMEGA_0 * t) * (
        np.cos(OMEGA_D * t)
        + (ZETA * OMEGA_0 / OMEGA_D) * np.sin(OMEGA_D * t)
    )

# =============================================================================
# 3.  DATA GENERATION
# =============================================================================

N_OBS   = 15       # number of noisy observations (training data)
SIGMA   = 0.05     # measurement noise standard deviation
N_COL   = 200      # collocation points for physics residual

# --- observation points: random draws from [0, T_TRAIN] ---
t_obs = np.sort(np.random.uniform(0.0, T_TRAIN, N_OBS))
y_obs = analytic(t_obs) + np.random.normal(0.0, SIGMA, N_OBS)

# --- collocation points: uniform from [0, T_EXTRAP] (extends beyond training!)
#     No measurements required – we freely place them wherever we want the
#     ODE to be satisfied, including the extrapolation region.
t_col = np.linspace(0.0, T_EXTRAP, N_COL)

# --- dense grid for evaluation / plotting ---
t_plot_train  = np.linspace(0.0, T_TRAIN,  300)
t_plot_full   = np.linspace(0.0, T_EXTRAP, 500)

# Convert to PyTorch tensors
def to_tensor(arr, requires_grad=False):
    return torch.tensor(arr, dtype=torch.float32,
                        requires_grad=requires_grad).unsqueeze(1)

t_obs_t  = to_tensor(t_obs)
y_obs_t  = to_tensor(y_obs)
t_col_t  = to_tensor(t_col, requires_grad=True)   # grad needed for ODE residual

# =============================================================================
# 4.  NEURAL NETWORK ARCHITECTURE
# =============================================================================
# Both models share the same architecture so the comparison is fair.
# A small fully-connected network with Tanh activations – smooth activations
# are important because we differentiate the network output twice.

class FCNet(nn.Module):
    """
    Simple fully-connected network: 1 → [hidden]*n_layers → 1
    Tanh activations throughout (smooth → well-defined higher derivatives).
    """
    def __init__(self, hidden: int = 32, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


# =============================================================================
# 5.  PHYSICS RESIDUAL
# =============================================================================

def ode_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """
    Evaluate  r(t) = m*y'' + c*y' + k*y  at each collocation point.

    Derivatives are computed via automatic differentiation (torch.autograd.grad),
    which differentiates through the network itself – no finite differences.

    Steps:
      1.  Forward pass:  ŷ  = model(t)
      2.  First deriv:   ŷ' = d(ŷ)/dt   via autograd
      3.  Second deriv:  ŷ'' = d(ŷ')/dt via autograd
      4.  Residual:      r = m*ŷ'' + c*ŷ' + k*ŷ
    """
    y_hat = model(t)

    # first derivative
    dy = torch.autograd.grad(
        y_hat, t,
        grad_outputs=torch.ones_like(y_hat),
        create_graph=True   # keep graph so we can differentiate again
    )[0]

    # second derivative
    d2y = torch.autograd.grad(
        dy, t,
        grad_outputs=torch.ones_like(dy),
        create_graph=True
    )[0]

    return M * d2y + C * dy + K * y_hat


# =============================================================================
# 6.  TRAINING
# =============================================================================

def train(model: nn.Module,
          lambda_physics: float,
          n_epochs: int = 8000,
          lr: float = 1e-3,
          print_every: int = 1000,
          label: str = "Model") -> dict:
    """
    Train the model.

    Loss = L_data  +  λ * L_physics

      L_data    = MSE of model output vs noisy observations
      L_physics = mean squared ODE residual at collocation points
                  (zero when lambda_physics = 0  →  standard ML)

    Returns a dict with training history.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=3000, gamma=0.5
    )

    history = {"epoch": [], "loss_data": [], "loss_physics": [], "loss_total": []}

    for epoch in range(1, n_epochs + 1):
        optimiser.zero_grad()

        # --- data loss ---
        y_pred = model(t_obs_t)
        loss_data = torch.mean((y_pred - y_obs_t) ** 2)

        # --- physics loss ---
        if lambda_physics > 0:
            residual   = ode_residual(model, t_col_t)
            loss_phys  = torch.mean(residual ** 2)
        else:
            loss_phys  = torch.tensor(0.0)

        loss_total = loss_data + lambda_physics * loss_phys
        loss_total.backward()
        optimiser.step()
        scheduler.step()

        if epoch % print_every == 0 or epoch == 1:
            print(f"  [{label}] epoch {epoch:5d} | "
                  f"L_data={loss_data.item():.5f}  "
                  f"L_phys={loss_phys.item():.5f}  "
                  f"L_total={loss_total.item():.5f}")

        if epoch % 100 == 0:
            history["epoch"].append(epoch)
            history["loss_data"].append(loss_data.item())
            history["loss_physics"].append(loss_phys.item())
            history["loss_total"].append(loss_total.item())

    return history


# --- instantiate & train ---
print("=" * 60)
print("Training standard ML model  (λ = 0, data loss only)")
print("=" * 60)
model_ml = FCNet(hidden=32, n_layers=4)
hist_ml  = train(model_ml, lambda_physics=0.0, label="Standard ML")

print()
print("=" * 60)
print("Training PINN  (λ = 1e-2, data + physics loss)")
print("=" * 60)
model_pinn = FCNet(hidden=32, n_layers=4)
hist_pinn  = train(model_pinn, lambda_physics=1e-2, label="PINN      ")

# =============================================================================
# 7.  EVALUATION
# =============================================================================

def predict(model: nn.Module, t: np.ndarray) -> np.ndarray:
    t_t = to_tensor(t)
    with torch.no_grad():
        return model(t_t).squeeze().numpy()


y_true_full  = analytic(t_plot_full)
y_true_train = analytic(t_plot_train)

y_ml_full   = predict(model_ml,   t_plot_full)
y_pinn_full = predict(model_pinn, t_plot_full)

# RMSE helper
def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))

# Training-interval RMSE
mask_train = t_plot_full <= T_TRAIN
rmse_ml_train   = rmse(y_ml_full[mask_train],   y_true_full[mask_train])
rmse_pinn_train = rmse(y_pinn_full[mask_train],  y_true_full[mask_train])

# Extrapolation-interval RMSE
mask_extrap = t_plot_full > T_TRAIN
rmse_ml_ext   = rmse(y_ml_full[mask_extrap],   y_true_full[mask_extrap])
rmse_pinn_ext = rmse(y_pinn_full[mask_extrap],  y_true_full[mask_extrap])

# Physics residual on full domain (using dense grid, finite differences for eval)
def physics_residual_np(y: np.ndarray, t: np.ndarray) -> float:
    """Approximate ODE residual via central finite differences."""
    dt  = t[1] - t[0]
    dy  = np.gradient(y, dt)
    d2y = np.gradient(dy, dt)
    r   = M * d2y + C * dy + K * y
    return float(np.sqrt(np.mean(r[5:-5] ** 2)))   # trim edges

phys_res_ml   = physics_residual_np(y_ml_full,   t_plot_full)
phys_res_pinn = physics_residual_np(y_pinn_full,  t_plot_full)

print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Metric':<35}  {'Std ML':>10}  {'PINN':>10}")
print("-" * 58)
print(f"{'RMSE  (training interval)':35}  {rmse_ml_train:>10.4f}  {rmse_pinn_train:>10.4f}")
print(f"{'RMSE  (extrapolation)':35}  {rmse_ml_ext:>10.4f}  {rmse_pinn_ext:>10.4f}")
print(f"{'Physics residual (full domain)':35}  {phys_res_ml:>10.4f}  {phys_res_pinn:>10.4f}")


# =============================================================================
# 8.  VISUALISATION
# =============================================================================

BLUE   = "#378ADD"
RED    = "#E24B4A"
GREEN  = "#1D9E75"
GRAY   = "#888780"
LGRAY  = "#D3D1C7"

fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor("#FAFAF8")

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.52, wspace=0.38,
                       left=0.07, right=0.97,
                       top=0.93, bottom=0.07)

ax_data   = fig.add_subplot(gs[0, :])   # row 0, all cols
ax_train  = fig.add_subplot(gs[1, :2])  # row 1, cols 0-1
ax_loss   = fig.add_subplot(gs[1, 2])   # row 1, col 2
ax_extrap = fig.add_subplot(gs[2, :2])  # row 2, cols 0-1
ax_phys   = fig.add_subplot(gs[2, 2])   # row 2, col 2

for ax in [ax_data, ax_train, ax_loss, ax_extrap, ax_phys]:
    ax.set_facecolor("#F1EFE8")
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)

# ── helper: shade extrapolation region ──────────────────────────────────────
def shade_extrap(ax, alpha=0.12):
    ax.axvspan(T_TRAIN, T_EXTRAP, color=GRAY, alpha=alpha, label="_nolegend_")
    ax.axvline(T_TRAIN, color=GRAY, lw=0.8, ls="--", alpha=0.6)


# ── Panel 1: training data ───────────────────────────────────────────────────
ax_data.set_title("Panel 1 — Training data: sparse noisy observations vs true trajectory",
                  fontsize=10, loc="left", pad=6, color="#444441")
ax_data.plot(t_plot_train, y_true_train, color=GRAY, lw=1.5,
             label="True trajectory  y(t)")
ax_data.scatter(t_obs, y_obs, color=BLUE, s=50, zorder=5,
                label=f"Noisy observations  (N={N_OBS}, σ={SIGMA})")
ax_data.set_xlabel("time  [s]")
ax_data.set_ylabel("displacement  y(t)")
ax_data.legend(fontsize=8, framealpha=0.5)
ax_data.set_xlim(0, T_TRAIN)

# Annotate collocation points along x-axis
ax_data.eventplot(t_col[t_col <= T_TRAIN], orientation="horizontal",
                  lineoffsets=-1.05, linelengths=0.06,
                  linewidths=0.6, color=GREEN, alpha=0.5)
ax_data.annotate("▲ collocation points (no measurement needed)",
                 xy=(0.01, 0.04), xycoords="axes fraction",
                 fontsize=7, color=GREEN)


# ── Panel 2: fit on training interval ────────────────────────────────────────
ax_train.set_title("Panel 2 — Fit on training interval  [0, 6 s]",
                   fontsize=10, loc="left", pad=6, color="#444441")
ax_train.plot(t_plot_full[mask_train], y_true_full[mask_train],
              color=GRAY, lw=1.5, label="True")
ax_train.plot(t_plot_full[mask_train], y_ml_full[mask_train],
              color=RED,   lw=2,   ls="--",
              label=f"Standard ML  (RMSE={rmse_ml_train:.4f})")
ax_train.plot(t_plot_full[mask_train], y_pinn_full[mask_train],
              color=GREEN, lw=2,
              label=f"PINN         (RMSE={rmse_pinn_train:.4f})")
ax_train.scatter(t_obs, y_obs, color=BLUE, s=30, zorder=5, alpha=0.6,
                 label="Observations")
ax_train.set_xlabel("time  [s]")
ax_train.set_ylabel("displacement  y(t)")
ax_train.legend(fontsize=7.5, framealpha=0.5)
ax_train.set_xlim(0, T_TRAIN)


# ── Panel 3: training loss curves ────────────────────────────────────────────
ax_loss.set_title("Panel 3 — Training loss",
                  fontsize=10, loc="left", pad=6, color="#444441")
ax_loss.semilogy(hist_ml["epoch"],   hist_ml["loss_data"],
                 color=RED,   lw=1.5, ls="--", label="ML  L_data")
ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_data"],
                 color=GREEN, lw=1.5, label="PINN  L_data")
ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_physics"],
                 color=GREEN, lw=1.5, ls=":", alpha=0.7,
                 label="PINN  L_physics")
ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_total"],
                 color=GREEN, lw=2.5, alpha=0.3, label="PINN  L_total")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("loss  (log scale)")
ax_loss.legend(fontsize=7, framealpha=0.5)


# ── Panel 4: extrapolation ───────────────────────────────────────────────────
ax_extrap.set_title("Panel 4 — Extrapolation beyond training window  [6, 10 s]",
                    fontsize=10, loc="left", pad=6, color="#444441")
shade_extrap(ax_extrap)
ax_extrap.plot(t_plot_full, y_true_full,
               color=GRAY,  lw=1.5, label="True")
ax_extrap.plot(t_plot_full, np.clip(y_ml_full, -3, 3),
               color=RED,   lw=2,   ls="--",
               label=f"Standard ML  (extrap RMSE={rmse_ml_ext:.4f})")
ax_extrap.plot(t_plot_full, y_pinn_full,
               color=GREEN, lw=2,
               label=f"PINN         (extrap RMSE={rmse_pinn_ext:.4f})")
ax_extrap.scatter(t_obs, y_obs, color=BLUE, s=30, zorder=5, alpha=0.5,
                  label="Observations")
ax_extrap.set_xlabel("time  [s]")
ax_extrap.set_ylabel("displacement  y(t)")
ax_extrap.set_ylim(-2.5, 2.5)
ax_extrap.set_xlim(0, T_EXTRAP)
ax_extrap.legend(fontsize=7.5, framealpha=0.5)
ax_extrap.annotate("← training region →", xy=(0.18, 0.93),
                   xycoords="axes fraction", fontsize=7.5, color=GRAY)
ax_extrap.annotate("← extrapolation →",   xy=(0.67, 0.93),
                   xycoords="axes fraction", fontsize=7.5, color=GRAY)


# ── Panel 5: physics residual profile ────────────────────────────────────────
ax_phys.set_title("Panel 5 — ODE residual  |r(t)|  along domain",
                  fontsize=10, loc="left", pad=6, color="#444441")
shade_extrap(ax_phys)

dt      = t_plot_full[1] - t_plot_full[0]

def pointwise_residual(y, t):
    dy  = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(M * d2y + C * dy + K * y)

r_ml   = pointwise_residual(y_ml_full,   t_plot_full)
r_pinn = pointwise_residual(y_pinn_full,  t_plot_full)

ax_phys.plot(t_plot_full, r_ml,   color=RED,   lw=1.5, ls="--",
             label=f"ML    (mean={phys_res_ml:.3f})")
ax_phys.plot(t_plot_full, r_pinn, color=GREEN, lw=1.5,
             label=f"PINN  (mean={phys_res_pinn:.3f})")
ax_phys.set_xlabel("time  [s]")
ax_phys.set_ylabel("|r(t)| = |mÿ + cẏ + ky|")
ax_phys.set_xlim(0, T_EXTRAP)
ax_phys.legend(fontsize=7.5, framealpha=0.5)
ax_phys.annotate("collocation pts\nextend to here",
                 xy=(T_EXTRAP - 0.2, ax_phys.get_ylim()[1] * 0.85),
                 fontsize=6.5, color=GRAY, ha="right")


# ── super-title and caption ──────────────────────────────────────────────────
fig.suptitle(
    "Physics-Informed Neural Network (PINN)  vs  Standard Supervised ML\n"
    f"Damped spring   m={M}  c={C}  k={K}  |  "
    f"ζ={ZETA:.3f}  ω_d={OMEGA_D:.3f} rad/s  |  "
    f"{N_OBS} observations  σ={SIGMA}  |  {N_COL} collocation pts",
    fontsize=11, y=0.975, color="#2C2C2A"
)

plt.savefig("pinn_spring_damper.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nFigure saved to  pinn_spring_damper.png")
plt.show()
