**Hyperparameter tracking, Burgers' equation**
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
Default value: 400
Starting search:
num_collocations = [10, 20, 50, 100, 200, 300, 400, 500]
Best number of collocation points: 300 with RMSE: 0.005644020877243901
Keep looking around 300-400
num_collocations = np.linspace(210, 490, 15)
Best number of collocation points: 250 with RMSE: 0.004315248141050536

*One last one: lr_param*

For the reverse problem we have to tune lr_param
We will look at 3 things that need to be optimized: RMSE and distances to true values of nu (0.1)

Let's try this for the third time
Starting search: learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
Best learning rate RMSE: 0.0001 with RMSE: 0.011156415551776214
Best learning rate v: 0.001 with difference: 0.04093186918169268 (from v_hat: 0.05906813081830733)

Zoomed search: learning_rates =np.logspace(-1.8, -4.2, num=25)
Best learning rate RMSE: 0.0001584893192461111 with RMSE: 0.00529069332427842
Best learning rate v: 0.00019952623149688788 with difference: 0.0009480903908367228 (from v_hat: 0.09905190960916328)
These lie very close together, we will choose 0.00019952623149688788 as it also has a decent RMSE (0.0055)