**Hyperparameter tracking**
*Idea:*

First adam parameters (lr, betas, but the betas don't need to be changed in most cases), then weights (ic + physics), then number of collocations
As our test we will choose the one that minimizes the RMSE (at the plotting definition) on the training area for the blind PINN.

*Fixed:*
    # ── Time domain ───────────────────────────────────────────────────────────
    t0 :            float = 0    # start of observation window [s]
    t_train:        float = 2.0    # end of observation window   [s]
    t_extrap:       float = 3.0   # end of extrapolation window [s]

    # ── Space domain ───────────────────────────────────────────────────────────
    x_begin:  float = -7.0   # left boundary of observation window   [m]
    x_end:    float = 7.0    # right boundary of observation window  [m]


    # ── Data ─────────────────────────────────────────────────────────────────
    n_obs_total:       int    = 1000   # total noisy observations (before split)
    n_val: int = 200  # number of validation observations 
    n_val_t: int = 100  # number of validation observations along time axis
    n_val_x: int = 200  # number of validation observations along space axis
    sigma:       float  = 0.05  # measurement noise std dev
    n_ic_samples_x: int = 200  # initial condition samples in x dimension (for IC loss)

    n_col_pool:  int  = 10000 # pool of collocation points to sample from each epoch
    seed:        int    = 42    # global RNG seed

    # ── Plotting sampling ───────────────────────────────────────────────────────────────
    n_plot_x: int = 500  # spatial resolution for all plots (including snapshots)
    n_plot_t: int = 200  # temporal resolution for all plots (including snapshots)

    # ── Network architecture ──────────────────────────────────────────────────
    hidden:   int = 48   # neurons per hidden layer
    n_layers: int = 6    # number of hidden layers

    # ── Optimiser ────────────────────────────────────────────────────────────
    beta1:    float = 0.9    # Adam beta1
    beta2:    float = 0.999  # Adam beta2
    lr_step:  int   = 3000   # StepLR: decay every this many epochs
    lr_gamma: float = 0.5    # StepLR: multiplicative factor

    # ── Training loop ─────────────────────────────────────────────────────────
    n_epochs:    int = 40_000   # maximum training epochs
    print_every: int = 1_000   # console log frequency (epochs)
    log_every:   int = 100     # history-dict write frequency (epochs)

    # ── Early stopping ────────────────────────────────────────────────────────
    patience:  int   = 30     # patience in units of log_every
    min_delta: float = 1e-6   # minimum improvement to reset the counter

We will investigate using the following physics system:
    # ── Physical parameters ───────────────────────────────────────────────────
    v:      float = 0.1      # viscosity  [m^2/s]

    # ── Initial conditions ───────────────────────────────────────────────────
    situation: str = "N-wave"  # "N-wave","Gaussian", or "Step"

*Base:*
    # ── Data ─────────────────────────────────────────────────────────────────
    n_col:       int  = 400   # collocation points (physics residual)

    # ── PINN loss weights ─────────────────────────────────────────────────────
    lambda_phys: float = 0.1  # physics-residual weight
    lambda_ic:   float = 50.0  # initial-condition weight (>> lambda_phys)

    # ── Optimiser ────────────────────────────────────────────────────────────
    lr:       float = 1e-2   # initial Adam learning rate




Base RMSE: 0.1248

*Learning rate*
Default value: lr = 0.1
Starting search: learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0] 
Best learning rate: 0.001 with RMSE: 0.020584824793342654

Zoomed search: learning_rates =10**np.linspace(-3.9, -2.1, num=15)
Best learning rate: 0.0018077686769634343 , will be used for all future hyperparm optim.
Best RMSE: 0.012208526184268941

*Weights*
Default value: ic = 50 , phys = 0.1
Starting search (grid):
ic_weights = 10**np.linspace(0.5, 2.5, num=5)  
phys_weights = 10**np.linspace(-2.5, -0.5, num=5)
Best initial condition weight: 3.1622776601683795, Best physics weight: 0.31622776601683794 with RMSE: 0.005687778753226005

Zoomed search (grid):
ic_weights = 10**np.linspace(0.1, 0.9, num=4)  
phys_weights = 10**np.linspace(-0.9, -0.1, num=4)
Best initial condition weight: 1.2589254117941673, Best physics weight: 0.7943282347242815 with RMSE: 0.004674636637779943
This is still at the edge of our search, but we want ic > data > phy, so we can't optimize any further without endangering this balance.

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
Best learning rate RMSE: 0.006105402296585327 with RMSE: 0.00412373159709762
Best learning rate w0: 0.006105402296585327 with difference: 0.0006521940231323242 (from w0_hat: 1.9993478059768677)
Best learning rate zeta: 0.011787686347935873 with difference: 0.0012118816375732422 (from zeta_hat: 0.12621188163757324)

We will choose , it also has a decent zeta_hat: 0.1270 (0.0020 diff)