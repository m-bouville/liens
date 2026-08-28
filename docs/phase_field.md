# Phase-field simulations

Phase-field simulations are implemented in C++ using the STL to generate training data for the neural networks, which are implemented separately in Python using PyTorch (see [./docs/neural_nets.md](neural_nets.md)). 

## High-level choices
The phase-field model is based on the Landau–Ginzburg free-energy functional,

$$F[\phi] = \int_\Omega \left[f(\phi,T) + \frac{\kappa}{2} \lvert\nabla\phi\rvert^2
\right]dV.$$

Its local free-energy density is given by a temperature-dependent Landau potential:

$$f(\phi,T) = \frac{a(T)}{2}\phi^2 + \frac{b}{4}\phi^4,$$

where $a(T)=a_0(T-T_0)$. Its derivative is $\frac{\partial f}{\partial\phi} = a(T)\phi + b\phi^3$. 

Above the critical temperature $T_0$, the potential has a single minimum at $\phi=0$. Below $T_0$, it becomes a symmetric double well: minima at $\pm \sqrt{a_0(T_0-T)/b}$ and a maximum at 0. The potential at the minima is $-a^2(T)/(4b)$.
Dedimensionalization: $T_0$ and $b$ are typically set to 1 without loss of generality.

A single non-conserved order parameter is evolved using the Allen–Cahn equation:

$$\frac{\partial\phi}{\partial t} = M\left(\kappa\nabla^2\phi - \frac{\partial f}{\partial\phi}\right).$$

Here $M$ is the mobility, it is (at least initially) temperature-independent. The Cahn–Hilliard equation (conserved) may be run later if and when a latent-space model is shown to learn phase-field evolution. 

Simulations are performed on two-dimensional periodic domains. The chemical potential is computed as 

$$\mu = \frac{\partial f}{\partial \phi} - \kappa \nabla^2 \phi,$$

after which the Allen–Cahn equation is integrated using an explicit forward-Euler time step. A Fourier pseudo-spectral discretization (FFTW) may be implemented later (but this is not a priority).
During simulation, total free energy is computed every saved timestep and should decrease monotonically.

The parameter sweep is parallelized using `<future>`.



## Parameters

### System size
The first tests are run on a 64-by-64 grid, for faster debugging. Larger grids (128×128, 256×256, perchance 512×512) are introduced once the solver and neural surrogate are stable. While increasing the grid size merely increases the computational cost of the phase-field simulations, it also requires a deeper convolutional autoencoder, which can be trickier to train. So the decision to switch will not be rushed.

The larger the system the longer computing a single step takes. But larger systems also require more time steps (they can accomodate larger grains, which take longer to evolve). One calls `tau_down​` the number of time steps needed for full coarsening. It is an indication of how long the runs need to be for each system size.

|  			| 32×32 | 64×64 | 128×128 | 256×256 | 512×512 |
|-----------|-------|-------|---------|---------|---------|
|`tau_down` | 80e3  | 600e3 |  2.5e6  |   10e6  |   40e6  |


### Initial conditions
Initial conditions consist of a homogeneous state perturbed by uniform white noise in [-`noise`/2, `noise`/2] (the parameter `noise` is set in the configuration file). Different seeds are also used in [./cpp/config.txt](../cpp/config.txt).


## Input parameter file
Simulation parameters are stored in a plain text file at `./config.txt`. Its format and path let it be written by the user and read directly by both the C++ phase-field solver and the Python training pipeline. It includes:
- system size (e.g. 256 by 256),
- phase-field timestep `dt`,
- number of time steps,
- parameters in Landau potential (a0, b, T0),
- other physical parameters (mobility, kappa, etc.),
- initial average value of the order parameter,
- list of different temperatures to be run,
- list of different noise amplitude and seed for random initialization,
- snapshot schedule.



## Storing simulation results
Simulation results are stored in binary format (no need for human access) readable in Python. Half precision (`float16`) values are used to reduce storage requirements while preserving sufficient numerical accuracy for neural-network training. 

There is one subdirectory for each simulation run, containing one binary file per stored snapshot (named by time index). Subdirectories are named by system size, then by parameters (temperature, initial conditions): `/datasets/256x256/T980_n050_s97/t0001100` for $T=0.980$, noise amplitude of 0.050 and 97 for seed. Unlike a single file for the whole trajectory, this allows random access to individual snapshots during autoencoder training while preserving temporal ordering for training the latent surrogate model.

Physics-based statistics are also computed (see [./docs/neural_nets.md](neural_nets.md) for more details).


### On snapshot schedule
Since phase-field evolution slows drastically over time, a constant output interval is inefficient. The output schedule in the input file can efficiently capture the rapid early evolution and the slower late-stage coarsening.

The LDS does not require equally spaced stored frames, it only needs to know the time increment between them, and may therefore be trained using variable timesteps.


## Overall architecture
```text
                            main()
                              │
                              ▼
                   Config::load("config.txt")
                              │
                              ▼
                       Simulation
                              │
                              ▼
        ┌───────────────────────────────────────────────┐
        │ run()                                         │
        │                                               │
        │ for seed                                      │
        │   for noise                                   │
        │     for T                                     │
        │         └──────────────┐                      │
        └────────────────────────│──────────────────────┘
                                 ▼
                     __runOneSimulation(T, noise, seed)
                                 │
                                 ├───────────────┐
                                 │               │
                                 ▼               ▼
                             Field φ       LandauPotential
                                 │
                                 ▼
                           OrderParameter
                                 │
                                 ▼
                             FDSolver
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             Potential::derivative()     FD operators
             (bulk derivative)           (∇²φ, |∇φ|²)
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                        μ = ∂f/∂φ − κ ∇²φ
                                 │
                                 ▼
              Allen–Cahn timestep (or Cahn–Hilliard later)
                                 │
                                 ▼
                     update OrderParameter::field
                                 │
                                 ▼
 						 if save timestep?
						 		 │
					 ┌───────────┴────────────┐
					 ▼                        ▼
			 compute total energy      write snapshot
			 compute statistics         (float16 binary)
					 │                        │
					 └───────────┬────────────┘
								 ▼
						   next timestep
```