# *Section 1*

## Question 1 #

loss_total = l_data + LAMBDA_PHYS * l_phys + LAMBDA_IC * l_ic

loss_data = torch.mean((y_pred - y_obs_t) ** 2)
l_phys = loss_physics(model, t_col_t)
l_ic   = loss_ic(model, t_ic_t)

if lambda_physics > 0:
            residual   = ode_residual(model, t_col_t)
            loss_phys  = torch.mean(residual ** 2)
        else:
            loss_phys  = torch.tensor(0.0)

def loss_physics(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """
    L_physics = mean( r(t_i)^2 )  over collocation points

    r(t) = m*y_hat''(t) + c*y_hat'(t) + k*y_hat(t)

    Derivatives via automatic differentiation.
    The [0] unpacks the single-element tuple returned by autograd.grad.
    """
This evaluation is 0 for the original equation

def loss_ic(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """
    L_ic = ( y_hat(0) - Y0 )^2  +  ( y_hat'(0) - DY0 )^2

    Both initial conditions are enforced explicitly.
    y_hat'(0) is obtained via autograd because initial velocity
    is never directly observed in the data.
    """
## Question 2 #

Not sure yet

## Question 3 #

These forms return positive values that grow when the model is further away from the correct solution. In comparison with the mean absolute error, the MSE punishes outliers more harshly.

## Question 4 #

Type of neural network:
one input layer of one neuron, 4 hidden layers, each hidden layer has 32 neurons, one output layer of one neuron
We use tanh as our activation function, it has an easily computable first and second derivative, ReLU does not have well behaved derivatives (first derivative = 1 or 0 and second = 0).
Number of parameters: every hidden layer and output layer has 32 biases, first hidden layer and output layer have 32 weights, hidden layers 2, 3, and 4 have 32*32 weights.
Input = a singular value that is free to be any number (time)
Output = a singular value between -1 and 1 (y value)
TO DO: Determine if dimensions are adequate

## Question 5 #

It receives 15 randomly chosen timestamps with their corresponding y value (to which normal distributed noise has been added). These timestamps are the same for every epoch. A better way could be changing it to be a randomly chosen every epoch (but needs to be tested).
TO DO: test

## Question 6 #

TO DO: zoek gwn ff op

## Question 7 #

TO DO: same bro

## Question 8 #

8000 epochs. By looking at the graph of test accuracy and training accuracy plotted over the epochs.

## Question 9 #

Yes, these would be uniformly sampled timestamps with their y value taken from the analytic solution. This is done in the code as t_plot_full

# *Section 2* #

## Question 1 #

These make it so the random choices that are being made will always have the same seed in every run. This means the program will produce the same result every time, even though it is randomly sampling points.

## Question 2 #

They indicate what type the input t of the function analytic is expected to be and what type the output of the function analytic is expected to be.

## Question 3 #

The sorting is unnecessary but I don't think there is anything bad about it since we are giving everything as one batch.

## Question 4 #

It makes sure to equally sample the whole system, but they could end up in special periodic spots when choosing a certain choic of T_EXTRAP and/or N_COL. For example they could all fall in the zeros of the underdampened harmonic oscillator

## Question 5 #

It sets all the gradients to zero, this is important as otherwise it will continue to use the gradient from the earlier epoch.

## Question 6 #

loss_total.backward() propagates the total loss backwards, which calculates the gradient of each parameter.
optimiser.step updates all of the parameters using the learning rate and the current and past gradients (since Adam is a momentum based optimiser). 
scheduler.step changes the learning rate to the needed value for the next iteration.
As you can see, the order of these 3 statements is very important, first we need to get the current gradient, then we need to apply it using the current learning rate, then we can update the learning rate.

## Question 7 #

Because we need to calculate the derivatives of the y value at the collocation point and the initial condition for the loss calculation. We do this by using the torch autograd which requires the parameter to be defined as a torch tensor that has "requires grad" on.

## Question 8 #

The unsqueeze(1) method transforms the 2 dimensional array into a 3 dimensional tensor: a 2x2 would become a 2x1x2 tensor, the one refers to the dimension that will be the "single" dimension. For example here the y direction (0 = x, z = 2) is only one thick.
The reason why we do this is idk, probably because torch wants us to TO DO

## Question 9 #

It is set to all ones because it is expecting you to give the gradient of the function that you are deriving to, but f = t so df/dt = 1 in all cases.
The function returns a gradient function and then a bunch of variables indicating which were the options that were used to make this gradient function. Since we only want the gradient function we only take the first (so zeroth) element.

