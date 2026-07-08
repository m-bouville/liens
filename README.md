# LIENS — Latent Interface Evolution Neural Surrogate
## A latent-space neural surrogate for phase-field microstructure evolution


This project investigates neural-network surrogate models for phase-field microstructure evolution in materials science and engineering (MSE). Instead of repeatedly solving the governing partial differential equations (PDE), a neural network (NN) learns a reduced-order representation of the phase-field state together with a surrogate evolution operator acting in that latent space. The objective is to use this mapping between successive microstructures to accelerate phase-field simulations while preserving the underlying physics.



## A quick introduction to phase field
Phase-field simulates the evolution of microstructures without explicitly tracking sharp interfaces (e.g. grain boundaries). Instead the method introduces one or more continuous order parameters (OP) that vary smoothly across space, so that interfaces have a finite width. These fields take different equilibrium values in different phases. The evolution of the system is typically governed by PDEs describing gradient flow of a free-energy functional: the Cahn-Hilliard equation is used for conserved OP (e.g. composition) and Allen-Cahn if not conserved (phases).

For details (equations, implementation), see `./docs/phase_field.md`.



## Neural surrogate
The NN model will be trained on phase-field simulation results to predict the microstructure at $t + \Delta t$ based on the microstructure at $t$, i.e. it will learn a surrogate that approximates phase-field time evolution operators, without explicitly solving discretized PDEs.

The model will use an autoencoder (AE) based on convolutional neural networks (CNN). The encoder will compress the image of the microstructure, $x(t)$, as a latent representation, $z(t)$. From this the decoder will recover (approximately) the microstructure:
- encode: $z = E(x)$,
- decode: $x'= D(z)$, i.e. $D(E(x))$.

The convolutional autoencoder has a symmetric encoder–decoder architecture. The latent representation retains coarse spatial organization while reducing the dimensionality sufficiently for efficient latent-space dynamics. For details on the architecture of the autoencoder, see `./docs/neural_nets.md`.



## Latent representation 
The latent representation will also be used by a Latent Dynamics Surrogate (LDS) model to predict the microstructure at $t + \Delta t$: $z(t + \Delta t) = z(t) + f_\theta(z(t), \Delta t)$, with $\theta$ physical parameters (e.g. temperature). As the LDS does not have the stability constraints of PDEs, inference is possible at coarser effective time resolution than the phase-field solver ($\Delta t$ a multiple of the phase-field time step).

For sufficiently small time intervals, the evolution is approximately linear in the time increment: $f_\theta(z(t), \Delta t) \propto \Delta t$, so what we need to learn is the slope,
$$g_\theta(z(t)) = \dfrac{z(t + \Delta t) - z(t)}{\Delta t}.$$ 
Then, $z(t + \Delta t) = z(t) + g_\theta(z(t))\,\Delta t$.

Representing the microstructure in latent space rather than real space has two advantages:
- it can be much smaller (even a smallish 256×256 image has 65'000 degrees of freedom),
- it is customized to our purpose (a bit like using a vectorized image instead of a bitmap).
This is essentially a learned reduced-order model of a PDE flow map.

The underlying hypothesis is two-fold. Phase-field evolution occurs on a smooth low-dimensional manifold that can be learnt by a convolutional autoencoder. And the corresponding latent dynamics can be approximated by a neural surrogate operating entirely in latent space.

For phase-field systems, the latent variables are expected to behave similarly to coordinates on a reduced thermodynamic manifold, dynamics should approximate gradient flow.



## Process
The process is:
$$x(t) \xrightarrow{E} z(t) \xrightarrow{f_\theta} z(t+\Delta t) \xrightarrow{D} x(t+\Delta t).$$
In fact, the encoding occurs only once at the beginning and the decoding once at the end (plus when plots are needed):
$$x(0) \xrightarrow{E} z(0) \xrightarrow{f_\theta} z(\Delta t) \xrightarrow{f_\theta} z(2 \Delta t) \xrightarrow{f_\theta} \ldots \xrightarrow{f_\theta} z(T) \xrightarrow{D} x(T).$$


```text
        x(t)
          │
          ▼
    ┌───────────┐
    │ encoder E │
    └─────┬─────┘
          │
        z(t)
          │
          ├───────────────────┐
          │                   │
          │                   ▼
          │             ┌───────────┐
          │             │    fθ     │
          │             └─────┬─────┘
          │                   │
          │                   ▼
          │                z(t+Δt)
          │                   │
          ▼                   ▼
    ┌───────────┐       ┌───────────┐
    │ decoder D │       │ decoder D │
    └─────┬─────┘       └─────┬─────┘
          │                   │
          ▼                   ▼
       x'(t)              x'(t+Δt)
```



## Workflow
0.	Generate hundreds of phase-field simulations. 
1.	Train a CNN autoencoder on individual microstructures (no time evolution yet) to learn a latent representation that is reconstructive and physically descriptive.
2.  Validate that the latent space behaves like a smooth, structured coordinate system (see `./docs/neural_nets.md` for more details).
3.	Freeze the encoder and train the Latent Dynamics Surrogate (LDS) to predict future microstructures in latent space.
4.	Fine-tune the encoder and the LDS for prediction, including a small input from reconstruction.
5.	Fine-tune the latent representation and the LDS to both predict future microstructures and reconstruct the original microstructure.



## Repository structure

```text
.
├── cpp/               # phase-field solver
├── python/
│   ├── main.py
│   ├── params/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── checkpoints/
│   │   └── stage<N>/
│   ├── utils/
│   └── tests/
├── datasets/
│   ├── 64x64/
│   ├── 128x128/
│   └── 256x256/
├── output/
├── docs/
│   ├── phase_field.md
│   └── neural_nets.md
└── README.md
```

## Current status

- [X] C++ phase-field solver
- [x] Dataset generation
- [X] 1. CNN autoencoder
- [x] 2. Latent-space validation
- [x] 3. Latent dynamics surrogate
- [ ] 4-5. End-to-end training