# Phase-field simulations

Phase-field simulations are implemented in C++ using the STL and FFTW to generate training data for the neural networks, which are implemented separately in Python using PyTorch. 

## High-level choices
The phase-field model is based on the Landau–Ginzburg free-energy functional,
$$F[\phi] = \!\int_\Omega \left[f(\phi,T) + \frac{\kappa}{2} \lvert\nabla\phi\rvert^2
\right]\!dV.$$
Its local free-energy density is given by a temperature-dependent Landau potential:
$$f(\phi,T) = \frac{a(T)}{2}\phi^2 + \frac{b}{4}\phi^4\!,$$
where $a(T)=a_0(T-T_0)$. Above the critical temperature $T_0$, the potential has a single minimum at $\phi=0$. Below $T_0$, the potential becomes a symmetric double well. 

Its derivative is $\frac{\partial f}{\partial\phi} = a(T)\phi + b\phi^3$. Below $T_0$, minima are at $\pm \sqrt{a_0(T_0-T)/b}$, with a maximum at 0.

A single non-conserved order parameter is evolved using the Allen–Cahn equation:
$$\frac{\partial\phi}{\partial t} = M\!\left(\!\kappa\nabla^2\phi - \frac{\partial f}{\partial\phi}\right)\!\!.$$
Here $M$ is the mobility, it is (at least initially) temperature-independent. The Cahn–Hilliard equation (conserved) may be implemented later if and when a latent-space model is shown to learn phase-field evolution. 

During simulation, total free energy is computed every saved timestep and should decrease monotonically.


## Solver
Simulations are performed on two-dimensional periodic domains using a Fourier pseudo-spectral discretization:
$$\nabla^2 \; \longrightarrow \; -k^2,  \qquad  k^2 = k_x^2+k_y^2.$$
Time integration uses a first-order semi-implicit Fourier scheme, treating the Laplacian implicitly and the nonlinear term explicitly:
$$\frac{\phi^{n+1}-\phi^n}{\Delta t} = M \!\left(\!\kappa\nabla^2\phi^{n+1} - \left.\frac{\partial f}{\partial\phi}\right|_{\phi^n}\!\right)\!\!.$$
Applying the Fourier transform gives
$$\hat{\phi}^{\,n+1} = \frac{\hat{\phi}^{\,n} - \Delta t\,M\,\widehat{\frac{\partial f}{\partial\phi}(\phi^n)}} {1+\Delta t\,M\,\kappa\,k^2}.$$


## Parameters
The first tests are run on a 64-by-64 grid, for faster debugging. Larger grids (128×128, 256×256, perchance 512×512) will be introduced once the solver and neural surrogate are stable. While increasing the grid size merely increases the computational cost of the phase-field simulations, it also requires a deeper convolutional autoencoder, which can be trickier to train. So the decision to switch will not be rushed.

Initial conditions consist of a homogeneous state perturbed by Gaussian white noise,
$$\phi(\mathbf{r},0) = \phi_0 + A\eta(\mathbf{r}), \qquad \eta \sim \mathcal{N}(0,1).$$

Dedimensionalization: $T_0$, $M$ and $b$ are set to 1 without loss of generality.


## Input parameter file
Simulation parameters are stored in a plain text file so that they can be written by the user and read directly by both the C++ phase-field solver and the Python training pipeline. It includes:
- system size (e.g. 256 by 256),
- physical parameters (mobility, etc.),
- phase-field timestep `dt`,
- number of time steps,
- list of different temperatures to be run,
- list of different initial conditions to be run (e.g. amplitude of noise),
- snapshot schedule.


## Storing simulation results
Simulation results are stored in binary format (no need for human access) readable in Python. Half precision (`float16`) values are used to reduce storage requirements while preserving sufficient numerical accuracy for neural-network training. 

There is one subdirectory for each simulation run, containing one binary file per stored snapshot (named by time index). Subdirectories are named by system size, then by parameters (temperature, initial conditions): `./data/256x256/T980_A120/t0001100.bin`. Unlike a single file for the whole trajectory, this allows random access to individual snapshots during autoencoder training while preserving temporal ordering for training the latent surrogate model.


## On snapshot schedule
Since phase-field evolution slows drastically over time, a constant output interval is inefficient. In addition to the physical parameters, the input file specifies the output schedule. It is not uniform, so it can efficiently capture the rapid early evolution and the slower late-stage coarsening.

The LDS does not require equally spaced stored frames, it only needs to know the time increment between them, and may therefore be trained using variable timesteps: $\hat{z}(t_{i+1}) = z(t_i) + f_\theta(z(t_i), \Delta t_i)$, with $\Delta t_i = t_{i+1} - t_i$.
