# LIENS — Latent Interface Evolution Neural Surrogate
## A latent-space neural emulator for Allen–Cahn phase-field microstructure evolution


This project investigates neural-network surrogate models for phase-field microstructure evolution in materials science and engineering (MSE). Instead of repeatedly solving the governing partial differential equations (PDE), a neural network (NN) learns a reduced-order representation of the phase-field state together with a surrogate evolution operator acting in that latent space. The objective is to use this mapping between successive microstructures to accelerate phase-field simulations while preserving the underlying physics.



## A quick introduction to phase field
Phase-field simulates the evolution of microstructures without explicitly tracking sharp interfaces (e.g. grain boundaries). Instead the method introduces one or more continuous order parameters (OP) that vary smoothly across space, so that interfaces have a finite width. These fields take different equilibrium values in different phases. The evolution of the system is typically governed by PDEs describing gradient flow of a free-energy functional: the Cahn-Hilliard equation is used for conserved OP (e.g. composition) and Allen-Cahn if not conserved (phases).

For details (equations, implementation), see [./docs/phase_field.md](docs/phase_field.md).



## Neural surrogate
The NN model is trained on phase-field simulation results to predict the microstructure at $t + \delta t$ based on the microstructure at $t$, i.e. it learns a surrogate that approximates phase-field time evolution operators, without explicitly solving discretized PDEs.

The model uses an autoencoder (AE) based on convolutional neural networks (CNN). The encoder compresses the image of the microstructure, $x(t)$, as a latent representation, $z_0(t)$. From this the decoder recovers (approximately) the microstructure:
- encode: $z_0 = E(x)$,
- decode: $x'= D(z_0)$, i.e. $D(E(x_0))$.

The convolutional autoencoder has a symmetric encoder–decoder architecture. The latent representation retains coarse spatial organization while reducing the dimensionality sufficiently for efficient latent-space dynamics. For details on the architecture of the autoencoder, see [./docs/NN-neural_nets.md](docs/NN-neural_nets.md) and [./docs/NN-code_structure.md](docs/NN-code_structure.md).


## Latent representation
The latent representation is split into two streams: 
- $z_0(t)$, the "state" (from which the decoder can recover $x(t)$), 
- $z_1(t)$, an approximation of $\dot{z}_0(t)$ that exists purely in latent space and is never decoded. 

A Latent Dynamics Surrogate (LDS), $f_\theta$, predicts the next state from both:

$$z_0(t + \delta t) = z_0(t) + z_1(t)\,\delta t + f_\theta(z_0(t), z_1(t), \theta)\,\delta t^2/2,$$

with $\theta$ physical parameters (e.g. temperature). $z_1(t)\,\delta t$ is the first-order (linear) term; $f_\theta$ predicts the second-order (curvature) correction on top of it. 

As the LDS does not have the stability constraints of PDEs, inference is possible at coarser effective time resolution than the phase-field solver ($\delta t$ a multiple of the phase-field time step) — in practice the second-order term is capped for large $\delta t$, to avoid a blow-up.

Representing the microstructure in latent space rather than real space has two advantages:
- it can be much smaller (even a smallish 256×256 image has 65'000 degrees of freedom),
- it is customized to our purpose (a bit like using a vectorized image instead of a bitmap).
This is essentially a learned reduced-order model of a PDE flow map.

The underlying hypothesis is two-fold. Phase-field evolution occurs on a smooth low-dimensional manifold that can be learnt by a convolutional autoencoder. And the corresponding latent dynamics can be approximated by a neural surrogate operating entirely in latent space.

For phase-field systems, the latent variables are expected to behave similarly to coordinates on a reduced thermodynamic manifold, dynamics should approximate gradient flow.

See [./docs/NN-neural_nets.md](docs/NN-neural_nets.md) for the full derivation, including the Taylor expansions.


## Process
The process is:

$$x(t) \xrightarrow{E} z(t) \xrightarrow{f_\theta} z(t+\delta t) \xrightarrow{D} x(t+\delta t).$$

In fact, the encoding occurs only once at the beginning and the decoding once at the end (plus when plots are needed):

$$x(0) \xrightarrow{E} z(0) \xrightarrow{f_\theta} z(\delta t) \xrightarrow{f_\theta} z(2 \delta t) \xrightarrow{f_\theta} \ldots \xrightarrow{f_\theta} z(T) \xrightarrow{D} x(T).$$

(Writing $z$ loosely for the full latent state $(z_0,\ z_1)$; see [./docs/NN-neural_nets.md](docs/NN-neural_nets.md) for the actual split-stream mechanics.)


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
          │                z(t+δt)
          │                   │
          ▼                   ▼
    ┌───────────┐       ┌───────────┐
    │ decoder D │       │ decoder D │
    └─────┬─────┘       └─────┬─────┘
          │                   │
          ▼                   ▼
       x'(t)              x'(t+δt)
```



## Workflow
0.	Generate several thousands of  phase-field simulations. 
1.	Train a CNN autoencoder on individual microstructures (no time evolution yet) to learn a latent representation that is reconstructive and physically descriptive.
2.  Train a second latent stream, $z_1$, to represent $\dot{z}_0$ via a derivative loss (see [./docs/NN-neural_nets.md](docs/NN-neural_nets.md) for more details).
3.	Freeze the encoder and train the Latent Dynamics Surrogate (LDS) to predict future microstructures in latent space. (Two sub-steps: 3a and 3b.)
4.	Fine-tune the encoder and the LDS for prediction, including a small input from reconstruction.
5.	Fine-tune the latent representation and the LDS to both predict future microstructures and reconstruct the original microstructure.



## Repository structure

```text
.
├── cpp/               		# phase-field solver
├── datasets/
│   ├── 64x64/
│   ├── 128x128/
│   └── 256x256/			# eventually
├── docs/
│   ├── phase_field.md 		# C++ simulations (doc written by hand)
│   ├── neural_nets.md 		# Python NN (doc written by hand)
│   └── NN-code_structure.md #written automatically by Claude
├── figures/				# figures selected for inclusion here
├── output/					# figures generated automatically
│   └── datasets/			# statistics on phase-field runs
│   └── stage<N>/
├── python/            		# neural network
└── README.md 				# written by hand
```

For more details on the structure of the `python` directory, see [./docs/NN-code_structure.md](docs/NN-code_structure.md).



## Current status

### For those who like checklists
- [X] C++ phase-field solver
- [X] Dataset generation
- [X] 1. CNN autoencoder `C0`
- [X] 2. Derivative `C1`
- [X] 3. Latent dynamics surrogate
- [X] 4-5. End-to-end training
- [X] Obtaining satisfactory results for short times
- [ ] Obtaining satisfactory results at all temperatures
- [ ] Obtaining satisfying results overall

### For those who prefer text
- All five training stages are implemented and run end-to-end. The work currently underway is improving the accuracy of the surrogate, not making it run at all.
- The initial development was carried out using 64×64 images (32×32 for testing the code end-to-end). They were serviceable but hit their limit, with finite-size artifacts eventually dominating the results. The focus is now on 128×128 microstructures.
- There is a (physically plausible) difference of behavior above and below $T = 0.9 \times T_0$. Work is being done to smoothen out this wrinkle.
- Predicting $t + \delta t$ gives sensible results, $t + 10 \delta t$ does not.
- A set of attribution diagnostics has localized the remaining error to the latent dynamics rather than the autoencoder (stages 1 and 2), and identified that the learned second-order corrector ($f_\theta$) does not yet belong in the propagation path as trained. This may hint at the need for a redesign of stage 3 (training $f_\theta$ as a vector field under a multi-step objective) as the next step. 