**Hyperparameter tracking**
*Idea:*

First adam parameters (lr, betas, but the betas don't need to be changed in most cases), then weights (ic + physics), then number of collocations
As our test we will choose the one that minimizes the RMSE (at the plotting definition) on the training area for the blind PINN.

*Fixed:*
    # ── Network architecture ──────────────────────────────────────────────────
    hidden:   int = 32   # neurons per hidden layer
    n_layers: int = 4    # number of hidden layers

    # ── Data ─────────────────────────────────────────────────────────────────
    n_obs:       int   = 30    # total noisy observations (before split)
    n_bins:      int   = 5    # bins for stratisfying validation split along time axis (unused)
    n_obs_per_epoch: int = 10  # number of training observations to use per epoch (for stochasticity, unused)
    n_val:       int   = 200     # number of observations in the validation set, >> n_obs(for early stopping)
    val_fraction: float = 0.2  # fraction of observations held out for val (unused, we use n_val instead)
    sigma:       float = 0.05  # measurement noise std dev
    seed:        int   = 42    # global RNG seed

    # ── Optimiser ────────────────────────────────────────────────────────────
    beta1:    float = 0.9    # Adam beta1
    beta2:    float = 0.999  # Adam beta2
    lr_step:  int   = 3000   # StepLR: decay every this many epochs
    lr_gamma: float = 0.5    # StepLR: multiplicative factor

    # ── Time domain ───────────────────────────────────────────────────────────
    t_train:  float = 6.0    # end of observation window   [s]
    t_extrap: float = 10.0   # end of extrapolation window [s]
    
    # ── Early stopping ────────────────────────────────────────────────────────
    patience:  int   = 30     # patience in units of log_every
    min_delta: float = 1e-6   # minimum improvement to reset the counter

We will investigate using the following physics (underdamped) system
    # ── Physical parameters ───────────────────────────────────────────────────
    mass:      float = 1.0   # m  [kg]
    damping:   float = 0.5   # c  [N·s/m]
    stiffness: float = 4.0   # k  [N/m]
    y0:        float = 1.0   # initial displacement   y(0)
    dy0:       float = 0.0   # initial velocity       y'(0)

*Base:*
    # ── Data ─────────────────────────────────────────────────────────────────
    n_col:       int   = 100   # collocation points (physics residual)

    # ── PINN loss weights ─────────────────────────────────────────────────────
    lambda_phys: float = 1e-1  # physics-residual weight
    lambda_ic:   float = 10.0  # initial-condition weight (>> lambda_phys)

    # ── Optimiser ────────────────────────────────────────────────────────────
    lr:       float = 1e-3   # initial Adam learning rate



Base RMSE: 0.0355

*Learning rate*
Default value: lr = 0.001
Starting search: learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0] 
Best lr = 0.01
Best RMSE: 0.0103
Zoomed search: learning_rates = 10**np.linspace(-3.5, -1.5, num=11)
Best lr = 0.00316, will be used for all future hyperparm optim.
Best RMSE: 0.0076

*Weights*
Default value: ic = 10 , phys = 0.1
Starting search (grid):
ic_weights = 10**np.linspace(1, 3, num=7)  
phys_weights = 10**np.linspace(-3, -1, num=7)
Best: 21.544, 0.1
Best RMSE: 0.00922638

Zoomed search (grid):
ic_weights = 10**np.linspace(1, 2, num=5)  
phys_weights = 10**np.linspace(-1.5, -0.5, num=5)
Best ic: 10.0, Best physics weight: 0.31622776601683794 
Best RMSE: 0.01045978

*Number of collocation points*
Default value: 100
Starting search:
num_collocations = [20, 50, 100, 200, 500]
Best number of collocation points: 20 with RMSE: 0.011735529299658041
But basically flatline, so will use 100 to be safe

*Number of observations per epoch*

Best = all observations => will do that



*One last one: lr_param*

For the reverse problem we have to tune lr_param
We will look at 3 things that need to be optimized: RMSE and distances to true values of w0 and zeta

Starting search: learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
Best learning rate RMSE: 0.001 with RMSE: 0.004002771307036508
Best learning rate w0: 0.01 with difference: 0.004451186567684617 (from w0_hat: 1.9955488134323154)
Best learning rate zeta: 0.001 with difference: 0.0009697452508586474 (from zeta_hat: 0.12403025474914135)
We will keep searching around 0.1-0.001:
Zoomed search: learning_rates =10**np.linspace(-3.5, -0.5, num=25)
Best learning rate RMSE: 0.01 with RMSE: 0.002715598467403243
Best learning rate w0: 0.005623413251903491 with difference: 0.00044879758739879705 (from w0_hat: 2.000448797587399)
Best learning rate zeta: 0.042169650342858224 with difference: 0.0006081193295721377 (from zeta_hat: 0.12560811932957214)

We will choose 0.01, it also has a decent zeta_hat: 0.1271 (0.0021 diff) and decent omega0_hat: 2.0013 (0.0013 diff).