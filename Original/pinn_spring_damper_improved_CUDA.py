"""
==============================================================================
Physics-Informed Neural Network (PINN) for a Damped Spring-Mass System
                    — CUDA-accelerated version —
==============================================================================

PROBLEM
-------
We observe a few noisy measurements of the displacement y(t) of a mass on a
damped spring.  The true dynamics are governed by the second-order linear ODE

        m * y''(t)  +  c * y'(t)  +  k * y(t)  =  0

with initial conditions  y(0) = 1,  y'(0) = 0.

We compare two approaches:
  1. Standard supervised ML  -- a neural network trained only to minimise the
     RMSE on the observed data points.
  2. PINN                    -- the same network architecture, but the loss
     also penalises violations of the ODE at collocation points AND enforces
     the initial conditions explicitly.

LOSS FUNCTION (PINN)
--------------------
  L_total = L_data  +  lambda_phys * L_physics  +  lambda_ic * L_ic

  L_data    = (1/M) * sum_j  ( y_hat(t_j) - y_j )^2
                  MSE on the M noisy observations

  L_physics = (1/N) * sum_i  ( m*y_hat''(t_i) + c*y_hat'(t_i) + k*y_hat(t_i) )^2
                  mean squared ODE residual at N collocation points
                  derivatives obtained via automatic differentiation

  L_ic      = ( y_hat(0) - y0 )^2  +  ( y_hat'(0) - dy0 )^2
                  enforces the two initial conditions exactly
                  y_hat'(0) also obtained via automatic differentiation

  lambda_ic >> lambda_phys  because an error at t=0 propagates forward and
  corrupts the entire trajectory.

CUDA STRATEGY
-------------
  A single DEVICE constant is resolved at startup:
    - 'cuda'  if a CUDA GPU is available
    - 'mps'   if running on Apple Silicon (Metal Performance Shaders)
    - 'cpu'   otherwise
  Every model and every training tensor lives on DEVICE throughout.
  Snapshots are saved as CPU state_dicts so they don't hold GPU memory.
  Inference for plotting always runs on CPU to keep numpy interop simple.

OUTPUT
------
  Figure 1  --  summary comparison (data, fit, loss, extrapolation, residual)
  Figure 2  --  PINN trajectory snapshots at selected training epochs
  Figure 3  --  Standard ML trajectory snapshots at the same epochs

DEPENDENCIES
------------
    pip install torch numpy matplotlib

Run:
    python pinn_spring_damper.py
==============================================================================
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 0.  DEVICE SELECTION
# =============================================================================
# Priority order: CUDA GPU → Apple MPS → CPU
# The rest of the code uses DEVICE everywhere, so switching hardware requires
# no further changes.

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print(f"  Running on device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU               : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM              : "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# =============================================================================
# 1.  SYSTEM PARAMETERS
# =============================================================================
M   = 1.0       # mass  [kg]
C   = 0.5       # damping coefficient
K   = 4.0       # spring stiffness  [N/m]
Y0  = 1.0       # initial displacement  y(0)  -- enforced via L_ic
DY0 = 0.0       # initial velocity      y'(0) -- enforced via L_ic

# Derived quantities (underdamped regime: zeta < 1)
OMEGA_0 = np.sqrt(K / M)
ZETA    = C / (2.0 * np.sqrt(M * K))
OMEGA_D = OMEGA_0 * np.sqrt(1.0 - ZETA**2)

T_TRAIN  = 6.0    # end of observation window   [s]
T_EXTRAP = 10.0   # end of extrapolation window [s]

# Loss weights
LAMBDA_PHYS = 1e-2
LAMBDA_IC   = 10.0

# Epochs at which to snapshot model weights for trajectory plots
SNAPSHOT_EPOCHS = [1, 50, 200, 500, 1000, 2000, 4000, 8000]

# =============================================================================
# 2.  ANALYTIC SOLUTION  (ground truth, always on CPU / numpy)
# =============================================================================

def analytic(t: np.ndarray) -> np.ndarray:
    """
    Closed-form solution for the underdamped case with y(0)=1, y'(0)=0:

        y(t) = e^{-zeta*omega0*t} * [ cos(omega_d*t)
                                     + (zeta*omega0/omega_d) * sin(omega_d*t) ]
    """
    return np.exp(-ZETA * OMEGA_0 * t) * (
        np.cos(OMEGA_D * t)
        + (ZETA * OMEGA_0 / OMEGA_D) * np.sin(OMEGA_D * t)
    )

# =============================================================================
# 3.  DATA GENERATION
# =============================================================================

N_OBS = 15      # noisy observations
SIGMA = 0.05    # measurement noise standard deviation
N_COL = 200     # collocation points for physics residual

# Observation points: random draws from (0.1, T_TRAIN].
# t=0 deliberately excluded -- initial condition is enforced via L_ic.
t_obs = np.sort(np.random.uniform(0.1, T_TRAIN, N_OBS))
y_obs = analytic(t_obs) + np.random.normal(0.0, SIGMA, N_OBS)

# Collocation points: uniform on [0, T_EXTRAP].
# Extends into extrapolation region -- no measurement required.
t_col      = np.linspace(0.0, T_EXTRAP, N_COL)
y_col_true = analytic(t_col)   # true values for plotting only

# Initial condition point: exactly t=0
t_ic = np.array([0.0])

# Dense evaluation grids  (CPU numpy, used only for plotting)
t_plot_train = np.linspace(0.0, T_TRAIN,  300)
t_plot_full  = np.linspace(0.0, T_EXTRAP, 500)
y_true_full  = analytic(t_plot_full)
y_true_train = analytic(t_plot_train)

# ── convert to tensors and move to DEVICE ────────────────────────────────────
def to_tensor(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    """
    Convert a 1-D numpy array to a (N, 1) float32 tensor on DEVICE.

    requires_grad=True is needed for tensors that will be differentiated
    with respect to in autograd.grad (collocation and IC points).
    """
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    if requires_grad:
        t.requires_grad_(True)
    return t

# Training tensors -- all reside on DEVICE for the entire run
t_obs_t = to_tensor(t_obs)
y_obs_t = to_tensor(y_obs)
t_col_t = to_tensor(t_col, requires_grad=True)   # grad needed: ODE residual
t_ic_t  = to_tensor(t_ic,  requires_grad=True)   # grad needed: y'(0)

# =============================================================================
# 4.  NEURAL NETWORK ARCHITECTURE
# =============================================================================

class FCNet(nn.Module):
    """
    Fully-connected network: 1 -> [hidden]*n_layers -> 1
    Tanh activations throughout.

    Tanh is mandatory (not ReLU) because we differentiate the network output
    twice via autograd.  ReLU has zero second derivative everywhere, which
    would make the physics residual carry no gradient signal.

    The model is instantiated on CPU and moved to DEVICE via .to(DEVICE)
    after construction (see training section).
    """
    def __init__(self, hidden: int = 32, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# 5.  LOSS COMPONENTS
# =============================================================================
# All tensors entering these functions already live on DEVICE.
# The scalar constants M, C, K are plain Python floats -- PyTorch broadcasts
# them correctly regardless of device.

def loss_physics(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """
    L_physics = mean( r(t_i)^2 )  over collocation points

    r(t) = m*y_hat''(t) + c*y_hat'(t) + k*y_hat(t)

    Both derivatives are obtained via automatic differentiation through the
    network.  create_graph=True on the first call keeps the computation graph
    alive so the second differentiation can flow back through it.

    The [0] unpacks the single-element tuple returned by autograd.grad
    (the API always returns a tuple, one entry per input tensor).
    """
    y_hat = model(t)

    # first derivative  dy/dt
    dy = torch.autograd.grad(
        y_hat, t,
        grad_outputs=torch.ones_like(y_hat),
        create_graph=True
    )[0]

    # second derivative  d2y/dt2
    d2y = torch.autograd.grad(
        dy, t,
        grad_outputs=torch.ones_like(dy),
        create_graph=True
    )[0]

    residual = M * d2y + C * dy + K * y_hat
    return torch.mean(residual ** 2)


def loss_ic(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """
    L_ic = ( y_hat(0) - Y0 )^2  +  ( y_hat'(0) - DY0 )^2

    Both initial conditions are enforced explicitly.
    y_hat'(0) is obtained via autograd because initial velocity
    is never directly observed in the data.

    LAMBDA_IC >> LAMBDA_PHYS: an error at t=0 compounds over the
    entire trajectory, so this constraint must be tight.
    """
    y_hat_0 = model(t)

    dy_0 = torch.autograd.grad(
        y_hat_0, t,
        grad_outputs=torch.ones_like(y_hat_0),
        create_graph=True
    )[0]

    return torch.mean((y_hat_0 - Y0) ** 2 + (dy_0 - DY0) ** 2)


# =============================================================================
# 6.  TRAINING  (with weight snapshots at chosen epochs)
# =============================================================================

def train(model:            nn.Module,
          use_physics:      bool,
          n_epochs:         int   = 8000,
          lr:               float = 1e-3,
          print_every:      int   = 1000,
          label:            str   = "Model",
          snapshot_epochs:  list  = None) -> tuple:
    """
    Train the model on DEVICE and return (history_dict, snapshots_dict).

    Snapshots are saved as CPU state_dicts (via .cpu() before deepcopy) so
    they do not continue to occupy GPU memory after being saved.  This also
    means predict_from_state can run purely on CPU for plotting.

    Returns
    -------
    history   : dict with lists of loss values logged every 100 epochs
    snapshots : dict  { epoch : CPU state_dict }
    """
    # Move model to DEVICE once before the loop.
    # All subsequent forward/backward passes happen on DEVICE.
    model.to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=3000, gamma=0.5
    )

    history = {
        "epoch": [], "loss_data": [],
        "loss_physics": [], "loss_ic": [], "loss_total": []
    }
    snapshots    = {}
    snapshot_set = set(snapshot_epochs or [])

    t0_wall = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        optimiser.zero_grad()

        # --- data loss (MSE on 15 noisy observations, all on DEVICE) ---
        y_pred = model(t_obs_t)
        l_data = torch.mean((y_pred - y_obs_t) ** 2)

        if use_physics:
            l_phys = loss_physics(model, t_col_t)
            l_ic   = loss_ic(model, t_ic_t)
            loss_total = l_data + LAMBDA_PHYS * l_phys + LAMBDA_IC * l_ic
        else:
            l_phys     = torch.zeros(1, device=DEVICE)
            l_ic       = torch.zeros(1, device=DEVICE)
            loss_total = l_data

        loss_total.backward()
        optimiser.step()
        scheduler.step()

        # ── snapshot: pull weights to CPU before deep-copying ────────────────
        # .cpu() creates a CPU copy of every tensor in state_dict without
        # moving the live model off DEVICE.  deepcopy then clones those
        # CPU tensors into an independent dict.
        if epoch in snapshot_set:
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            snapshots[epoch] = copy.deepcopy(cpu_state)

        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0_wall
            print(f"  [{label}] epoch {epoch:5d} | "
                  f"L_data={l_data.item():.5f}  "
                  f"L_phys={l_phys.item():.5f}  "
                  f"L_ic={l_ic.item():.5f}  "
                  f"L_total={loss_total.item():.5f}  "
                  f"({elapsed:.1f}s)")

        if epoch % 100 == 0:
            history["epoch"].append(epoch)
            history["loss_data"].append(l_data.item())
            history["loss_physics"].append(l_phys.item())
            history["loss_ic"].append(l_ic.item())
            history["loss_total"].append(loss_total.item())

    total_time = time.perf_counter() - t0_wall
    print(f"  [{label}] training complete in {total_time:.1f}s  "
          f"({total_time / n_epochs * 1000:.2f} ms/epoch)")

    return history, snapshots


# --- instantiate models and train ---
print()
print("=" * 60)
print("Training standard ML model  (data loss only)")
print("=" * 60)
model_ml = FCNet(hidden=32, n_layers=4)
print(f"  Parameters: {model_ml.param_count()}")
hist_ml, snaps_ml = train(
    model_ml, use_physics=False,
    label="Standard ML", snapshot_epochs=SNAPSHOT_EPOCHS
)

print()
print("=" * 60)
print("Training PINN  (data + physics + initial condition loss)")
print("=" * 60)
model_pinn = FCNet(hidden=32, n_layers=4)
print(f"  Parameters: {model_pinn.param_count()}")
hist_pinn, snaps_pinn = train(
    model_pinn, use_physics=True,
    label="PINN      ", snapshot_epochs=SNAPSHOT_EPOCHS
)

# =============================================================================
# 7.  EVALUATION
# =============================================================================
# All inference for evaluation and plotting runs on CPU.
# We move the trained models back to CPU so that predict() and
# predict_from_state() work with plain numpy arrays without device checks.

model_ml.to("cpu")
model_pinn.to("cpu")

def predict(model: nn.Module, t: np.ndarray) -> np.ndarray:
    """
    Run inference on CPU.  The model must already be on CPU.
    No gradients needed -- torch.no_grad() saves memory and time.
    """
    t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return model(t_t).squeeze().numpy()

def predict_from_state(state_dict: dict, t: np.ndarray) -> np.ndarray:
    """
    Restore a CPU snapshot and predict.
    Snapshots were saved as CPU state_dicts in train(), so no device
    transfer is needed here.
    """
    tmp = FCNet(hidden=32, n_layers=4)   # always instantiated on CPU
    tmp.load_state_dict(state_dict)
    tmp.eval()
    return predict(tmp, t)

def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def physics_residual_np(y: np.ndarray, t: np.ndarray) -> float:
    """Approximate ODE residual via numpy central finite differences."""
    dy  = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    r   = M * d2y + C * dy + K * y
    return float(np.sqrt(np.mean(r[5:-5] ** 2)))

y_ml_full   = predict(model_ml,   t_plot_full)
y_pinn_full = predict(model_pinn, t_plot_full)

mask_train  = t_plot_full <= T_TRAIN
mask_extrap = t_plot_full >  T_TRAIN

rmse_ml_train   = rmse(y_ml_full[mask_train],   y_true_full[mask_train])
rmse_pinn_train = rmse(y_pinn_full[mask_train],  y_true_full[mask_train])
rmse_ml_ext     = rmse(y_ml_full[mask_extrap],  y_true_full[mask_extrap])
rmse_pinn_ext   = rmse(y_pinn_full[mask_extrap], y_true_full[mask_extrap])

phys_res_ml   = physics_residual_np(y_ml_full,  t_plot_full)
phys_res_pinn = physics_residual_np(y_pinn_full, t_plot_full)

y0_ml   = float(predict(model_ml,   np.array([0.0])))
y0_pinn = float(predict(model_pinn, np.array([0.0])))

print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  Device used : {DEVICE}")
print(f"{'Metric':<35}  {'Std ML':>10}  {'PINN':>10}")
print("-" * 58)
print(f"{'RMSE  (training interval)':35}  {rmse_ml_train:>10.4f}  {rmse_pinn_train:>10.4f}")
print(f"{'RMSE  (extrapolation)':35}  {rmse_ml_ext:>10.4f}  {rmse_pinn_ext:>10.4f}")
print(f"{'Physics residual (full domain)':35}  {phys_res_ml:>10.4f}  {phys_res_pinn:>10.4f}")
print(f"{'y_hat(0)  [true = {Y0}]':35}  {y0_ml:>10.4f}  {y0_pinn:>10.4f}")


# =============================================================================
# 8.  COLOUR PALETTE  (shared across all figures)
# =============================================================================

BLUE   = "#378ADD"   # noisy observations
RED    = "#E24B4A"   # standard ML
GREEN  = "#1D9E75"   # PINN
ORANGE = "#EF9F27"   # collocation points
PURPLE = "#7F77DD"   # initial condition marker
GRAY   = "#888780"   # true solution / neutral
LGRAY  = "#D3D1C7"   # spine colour
BG     = "#FAFAF8"   # figure background
PANEL  = "#F1EFE8"   # axes background


# =============================================================================
# 9.  FIGURE 1 — summary comparison
# =============================================================================

def shade_extrap(ax):
    ax.axvspan(T_TRAIN, T_EXTRAP, color=GRAY, alpha=0.12)
    ax.axvline(T_TRAIN, color=GRAY, lw=0.8, ls="--", alpha=0.6)

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)

fig1 = plt.figure(figsize=(15, 12))
fig1.patch.set_facecolor(BG)
gs1 = gridspec.GridSpec(3, 3, figure=fig1,
                        hspace=0.55, wspace=0.38,
                        left=0.07, right=0.97,
                        top=0.93,  bottom=0.07)

ax_data   = fig1.add_subplot(gs1[0, :])
ax_train  = fig1.add_subplot(gs1[1, :2])
ax_loss   = fig1.add_subplot(gs1[1, 2])
ax_extrap = fig1.add_subplot(gs1[2, :2])
ax_phys   = fig1.add_subplot(gs1[2, 2])

for ax in [ax_data, ax_train, ax_loss, ax_extrap, ax_phys]:
    style_ax(ax)

# ── Panel 1: training data ───────────────────────────────────────────────────
ax_data.set_title(
    "Panel 1 — Training data: sparse noisy observations vs true trajectory\n"
    "t=0 excluded from random observations; y(0) and y'(0) enforced via L_ic",
    fontsize=9, loc="left", pad=6, color="#444441")
ax_data.plot(t_plot_train, y_true_train, color=GRAY, lw=1.5,
             label="True trajectory  y(t)")
ax_data.scatter(t_obs, y_obs, color=BLUE, s=50, zorder=5,
                label=f"Noisy observations  (N={N_OBS}, sigma={SIGMA})")
ax_data.scatter([0], [Y0], color=PURPLE, s=120, marker="*", zorder=7,
                label=f"Initial condition  y(0)={Y0}  [enforced via L_ic]")
ax_data.scatter(t_col[t_col <= T_TRAIN],
                np.zeros(np.sum(t_col <= T_TRAIN)) - 1.08,
                color=ORANGE, s=8, marker="|", zorder=4, alpha=0.7,
                label="Collocation points (no measurement needed)")
ax_data.set_xlabel("time  [s]")
ax_data.set_ylabel("displacement  y(t)")
ax_data.legend(fontsize=8, framealpha=0.5)
ax_data.set_xlim(0, T_TRAIN)

# ── Panel 2: fit on training interval ────────────────────────────────────────
ax_train.set_title("Panel 2 — Fit on training interval  [0, 6 s]",
                   fontsize=10, loc="left", pad=6, color="#444441")
ax_train.plot(t_plot_full[mask_train], y_true_full[mask_train],
              color=GRAY, lw=1.5, label="True")
ax_train.plot(t_plot_full[mask_train], y_ml_full[mask_train],
              color=RED, lw=2, ls="--",
              label=f"Standard ML  (RMSE={rmse_ml_train:.4f})")
ax_train.plot(t_plot_full[mask_train], y_pinn_full[mask_train],
              color=GREEN, lw=2,
              label=f"PINN         (RMSE={rmse_pinn_train:.4f})")
ax_train.scatter(t_obs, y_obs, color=BLUE, s=30, zorder=5, alpha=0.6,
                 label="Observations")
ax_train.scatter([0], [Y0], color=PURPLE, s=120, marker="*", zorder=7,
                 label="IC  y(0)=1")
ax_train.set_xlabel("time  [s]")
ax_train.set_ylabel("displacement  y(t)")
ax_train.legend(fontsize=7.5, framealpha=0.5)
ax_train.set_xlim(0, T_TRAIN)

# ── Panel 3: training loss curves ────────────────────────────────────────────
ax_loss.set_title("Panel 3 — Training loss  (log scale)",
                  fontsize=10, loc="left", pad=6, color="#444441")
ax_loss.semilogy(hist_ml["epoch"],   hist_ml["loss_data"],
                 color=RED,   lw=1.5, ls="--", label="ML  L_data")
ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_data"],
                 color=GREEN, lw=1.5,           label="PINN  L_data")
ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_physics"],
                 color=GREEN, lw=1.5, ls=":",   label="PINN  L_physics")
ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_ic"],
                 color=BLUE,  lw=1.5, ls="-.",  label="PINN  L_ic")
ax_loss.semilogy(hist_pinn["epoch"], hist_pinn["loss_total"],
                 color=GREEN, lw=2.5, alpha=0.3, label="PINN  L_total")
for ep in SNAPSHOT_EPOCHS[:-1]:
    ax_loss.axvline(ep, color=GRAY, lw=0.5, ls=":", alpha=0.5)
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("loss")
ax_loss.legend(fontsize=7, framealpha=0.5)

# ── Panel 4: extrapolation ───────────────────────────────────────────────────
ax_extrap.set_title("Panel 4 — Extrapolation beyond training window  [6, 10 s]",
                    fontsize=10, loc="left", pad=6, color="#444441")
shade_extrap(ax_extrap)
ax_extrap.plot(t_plot_full, y_true_full,
               color=GRAY, lw=1.5, label="True")
ax_extrap.plot(t_plot_full, np.clip(y_ml_full, -3, 3),
               color=RED,  lw=2,  ls="--",
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
ax_extrap.annotate("training region", xy=(0.15, 0.93),
                   xycoords="axes fraction", fontsize=7.5, color=GRAY)
ax_extrap.annotate("extrapolation",   xy=(0.70, 0.93),
                   xycoords="axes fraction", fontsize=7.5, color=GRAY)

# ── Panel 5: pointwise ODE residual ──────────────────────────────────────────
ax_phys.set_title("Panel 5 — Pointwise ODE residual  |r(t)|",
                  fontsize=10, loc="left", pad=6, color="#444441")
shade_extrap(ax_phys)

def pointwise_residual(y, t):
    dy  = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(M * d2y + C * dy + K * y)

r_ml   = pointwise_residual(y_ml_full,  t_plot_full)
r_pinn = pointwise_residual(y_pinn_full, t_plot_full)

ax_phys.plot(t_plot_full, r_ml,   color=RED,   lw=1.5, ls="--",
             label=f"ML    (mean={phys_res_ml:.3f})")
ax_phys.plot(t_plot_full, r_pinn, color=GREEN, lw=1.5,
             label=f"PINN  (mean={phys_res_pinn:.3f})")
ax_phys.set_xlabel("time  [s]")
ax_phys.set_ylabel("|m*y'' + c*y' + k*y|")
ax_phys.set_xlim(0, T_EXTRAP)
ax_phys.legend(fontsize=7.5, framealpha=0.5)

fig1.suptitle(
    f"PINN vs Standard ML  |  device={DEVICE}  |  "
    f"m={M}  c={C}  k={K}  zeta={ZETA:.3f}  omega_d={OMEGA_D:.3f} rad/s\n"
    f"{N_OBS} observations (t>0)  sigma={SIGMA}  |  "
    f"{N_COL} collocation pts  |  "
    f"lambda_phys={LAMBDA_PHYS}  lambda_ic={LAMBDA_IC}  "
    f"|  IC: y(0)={Y0}  y'(0)={DY0}",
    fontsize=9, y=0.975, color="#2C2C2A"
)
fig1.savefig("pinn_fig1_summary.png", dpi=150, bbox_inches="tight",
             facecolor=BG)
print("Figure 1 saved to  pinn_fig1_summary.png")


# =============================================================================
# 10.  FIGURES 2 & 3 — epoch-by-epoch trajectory snapshots
# =============================================================================

def make_epoch_figure(snapshots:   dict,
                      model_color: str,
                      model_label: str,
                      fig_title:   str,
                      filename:    str,
                      clip_y:      bool = False) -> plt.Figure:
    """
    Build a 2-column grid of panels, one per snapshot epoch.

    Each panel shows:
      - true analytic solution        (gray line,    full domain)
      - model prediction at that epoch (coloured line, full domain)
      - noisy observations             (blue dots,    training window)
      - initial condition              (purple star,  t=0)
      - collocation points             (orange ticks, full domain)
      - training / extrapolation boundary (dashed vertical line)
      - RMSE over full domain annotated in title

    Snapshots were saved as CPU state_dicts, so predict_from_state
    runs entirely on CPU -- no device transfer needed here.
    """
    epochs = sorted(snapshots.keys())
    n_snap = len(epochs)
    n_cols = 2
    n_rows = (n_snap + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 3.2),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten()

    y_lo, y_hi = -2.2, 1.6
    col_y = y_lo + 0.08 * (y_hi - y_lo)   # y-position for collocation ticks

    for idx, epoch in enumerate(epochs):
        ax = axes_flat[idx]
        style_ax(ax)

        # predictions from CPU snapshot -- no .to(device) needed
        y_pred = predict_from_state(snapshots[epoch], t_plot_full)
        if clip_y:
            y_pred = np.clip(y_pred, -3, 3)

        err = rmse(y_pred, y_true_full)

        # true solution
        ax.plot(t_plot_full, y_true_full,
                color=GRAY, lw=1.4, alpha=0.9, label="True  y(t)")

        # model prediction
        ax.plot(t_plot_full, y_pred,
                color=model_color, lw=2.0, label=f"{model_label}  ŷ(t)")

        # extrapolation region
        ax.axvspan(T_TRAIN, T_EXTRAP, color=GRAY, alpha=0.10, zorder=0)
        ax.axvline(T_TRAIN, color=GRAY, lw=0.8, ls="--", alpha=0.5)

        # collocation points (orange ticks along bottom, full domain)
        ax.scatter(t_col, np.full_like(t_col, col_y),
                   color=ORANGE, s=6, marker="|",
                   alpha=0.6, zorder=3, label="Collocation pts")

        # noisy observations (training window only)
        ax.scatter(t_obs, y_obs,
                   color=BLUE, s=28, zorder=5, alpha=0.85,
                   label=f"Observations (N={N_OBS})")

        # initial condition
        ax.scatter([0], [Y0],
                   color=PURPLE, s=130, marker="*", zorder=7,
                   label=f"IC  y(0)={Y0}")

        ax.set_title(f"epoch = {epoch}   |   RMSE = {err:.4f}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xlim(0, T_EXTRAP)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y(t)", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("time  [s]", fontsize=8)

        if idx == 0:
            ax.text(T_TRAIN / 2, y_hi - 0.15, "training",
                    ha="center", va="top", fontsize=7, color=GRAY)
            ax.text(T_TRAIN + (T_EXTRAP - T_TRAIN) / 2, y_hi - 0.15,
                    "extrapolation", ha="center", va="top",
                    fontsize=7, color=GRAY)

    for idx in range(n_snap, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=GRAY,        lw=1.4,
               label="True solution  y(t)"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=f"{model_label} prediction  ŷ(t)"),
        Line2D([0], [0], color=BLUE,        lw=0, marker="o", markersize=5,
               label=f"Noisy observations  (N={N_OBS}, sigma={SIGMA})"),
        Line2D([0], [0], color=PURPLE,      lw=0, marker="*", markersize=9,
               label=f"Initial condition  y(0)={Y0},  y'(0)={DY0}"),
        Line2D([0], [0], color=ORANGE,      lw=0, marker="|", markersize=7,
               label=f"Collocation points  "
                     f"(N={N_COL}, full domain [0, {T_EXTRAP}])"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(fig_title, fontsize=10, y=1.01, color="#2C2C2A")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Figure saved to  {filename}")
    return fig


fig2 = make_epoch_figure(
    snapshots    = snaps_pinn,
    model_color  = GREEN,
    model_label  = "PINN",
    fig_title    = (
        f"PINN — trajectory evolution across training epochs  "
        f"[device={DEVICE}]\n"
        f"m={M}  c={C}  k={K}  |  {N_OBS} observations  sigma={SIGMA}  |  "
        f"{N_COL} collocation pts  |  "
        f"lambda_phys={LAMBDA_PHYS}  lambda_ic={LAMBDA_IC}"
    ),
    filename     = "pinn_fig2_pinn_epochs.png",
    clip_y       = False,
)

fig3 = make_epoch_figure(
    snapshots    = snaps_ml,
    model_color  = RED,
    model_label  = "Std ML",
    fig_title    = (
        f"Standard ML — trajectory evolution across training epochs  "
        f"[device={DEVICE}]\n"
        f"m={M}  c={C}  k={K}  |  {N_OBS} observations  sigma={SIGMA}  |  "
        f"data loss only  (no physics, no IC enforcement)"
    ),
    filename     = "pinn_fig3_ml_epochs.png",
    clip_y       = True,
)

plt.show()
print("\nDone.")
