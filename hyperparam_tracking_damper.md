**Hyperparameter tracking**
*Idea:*

First learning rate parameters, then weights (ic + physics), then number of collocations and number of obs per batch
As our test we will choose the one that minimizes the RMSE (at the plotting definition) on the training area for the blind PINN.

*Fixed:*
    # ── Network architecture ──────────────────────────────────────────────────
    hidden:   int = 32   # neurons per hidden layer
    n_layers: int = 4    # number of hidden layers

    # ── Data ─────────────────────────────────────────────────────────────────
    n_obs:       int   = 30    # total noisy observations (before split)
    val_fraction: float = 0.2  # fraction of observations held out for val
    sigma:       float = 0.05  # measurement noise std dev

We will investigate using the following physics (underdamped) system
    # ── Physical parameters ───────────────────────────────────────────────────
    mass:      float = 1.0   # m  [kg]
    damping:   float = 0.5   # c  [N·s/m]
    stiffness: float = 4.0   # k  [N/m]
    y0:        float = 1.0   # initial displacement   y(0)
    dy0:       float = 0.0   # initial velocity       y'(0)

    # ── Time domain ───────────────────────────────────────────────────────────
    t_train:  float = 6.0    # end of observation window   [s]
    t_extrap: float = 10.0   # end of extrapolation window [s]
    
    # ── Early stopping ────────────────────────────────────────────────────────
    patience:  int   = 30     # patience in units of log_every
    min_delta: float = 1e-6   # minimum improvement to reset the counter

*Base:*
    # ── Data ─────────────────────────────────────────────────────────────────
    n_obs_per_epoch: int = 10  # number of training observations to use per epoch (for stochasticity)
    val_fraction: float = 0.2  # fraction of observations held out for val
    n_col:       int   = 100   # collocation points (physics residual)


    # ── PINN loss weights ─────────────────────────────────────────────────────
    lambda_phys: float = 1e-1  # physics-residual weight
    lambda_ic:   float = 10.0  # initial-condition weight (>> lambda_phys)

    # ── Optimiser ────────────────────────────────────────────────────────────
    lr:       float = 1e-3   # initial Adam learning rate
    lr_step:  int   = 3000   # StepLR: decay every this many epochs
    lr_gamma: float = 0.5    # StepLR: multiplicative factor



Base RMSE: 0.0812

*Learning rate*
Default value: lr = 0.001
Starting search: learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
Zoomed search: learning_rates = 10**np.linspace(-3, -1, num=11)
Best lr = 0.02512, will be used for all future hyperparm optim.
Best RMSE: 0.02528657771573793

*Weights*
Default value: ic = 10 , phys = 0.1
Starting search (grid):
ic_weights = 10**np.linspace(1, 3, num=7)  
phys_weights = 10**np.linspace(-3, -1, num=7)
Best: 100, 0.01


Best initial condition weight: 100, Best physics weight: 0.01
Best RMSE: 0.0158

*Number of collocation points*
Default value: 100
Starting search:
num_collocations = [20, 50, 100, 200, 500]
Best choice: 50
Best RMSE: 0.0374807569799953
But we get 0.0158 earlier with 100 so we will include 100 in our training for zoomed

Zoomed search:
num_collocations = np.linspace(10, 100, num=19)
Best choice: 55
Best RMSE: 0.0140

*Number of observations per epoch*

Starting search:
