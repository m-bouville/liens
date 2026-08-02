# Neural networks: autoencoder and surrogate evolution


See `./docs/NN-code_structure.md` for more details on the structure of the code.


## Autoencoder

### Architecture
The convolutional autoencoder has a symmetric encoder–decoder architecture. The encoder depth scales with the (square) system size so that the spatial bottleneck remains 8×8: three downsampling stages for 64×64 inputs (64→32→16→8), five for 256×256, and so on. Each resolution level consists of two 3×3 convolutions (circular padding to match periodic boundaries) with ReLU activations, followed by a stride-2 convolution for downsampling (mirrored by learned upsampling in the decoder), with BatchNorm/LayerNorm.

The encoder terminates with a 1×1 convolution reducing the feature dimension to 16 channels, yielding an 8×8×16 latent representation. This bottleneck retains coarse spatial organization while reducing the dimensionality sufficiently for efficient latent-space dynamics. Initial runs will use a larger latent space, to ensure that sufficient information is available for reconstruction, before shrinking it to find the lower bound for accuracy.
 

## Training Stages and loss functions

### Dataset expansion
From each snapshot, more are created through
- mirror (horizontal and vertical);
- rotation by $\pm$90 °, 180°;
- transposing;
- translation by (Nx/2, 4), (4, Ny/2), (Nx/2+4, Ny/2+4) thanks to periodic boundaries
  - +4: shifts by half the size of the latent channels, so the translations are not in phase (to reduce the 8x8 checkerboard artifact).


### Training Stages

There are five losses, which can be mixed and matched at the different stages:
- reconstruction loss: $L_\mathrm{recon}$,
- statistics loss: $L_\mathrm{stats}$,
- derivative loss: $L_\mathrm{deriv}$,
- one-step latent prediction loss: $L_\mathrm{1step}$,
- multi-step rollout loss: $L_\mathrm{rollout}$.


| \#| Stage             | Trained | Frozen | Unused | Space | Snapshots | Loss                            |
|---|-------------------|--------------|------|------|-------|-------|---------------------------------|
| 1 a| autoencoder (C$_0$)|E, D$_0$, SH$_0$|   | f    | real  | 1    | `L_recon0 + λ L_stats0`           |
| 2 | derivative (C$_1$) |E${}^*$, D$_0^*$| SH$_0$ | f |latent$^\dagger$|3| `L_recon0 + λ L_stats0 + λ₁ L_deriv` |
| 3a| LDS               | f       | E, SH$_0$ | D$_0$ | latent| 2    | `L_1step + λ₁ L_deriv`           |
| 3b| LDS               | f       | E, SH$_0$ | D$_0$ | latent| n+1  | `L_rollout + ε L_1step + λ₁ L_deriv` |
| 4 | encoder refinement| E, f    | D$_0$, SH$_0$|  | latent$^\dagger$|n+1|`L_rollout + λ L_stats0 + ε L_recon0 + λ₁ L_deriv` |
| 5 | end-to-end        | E, f, D$_0$ | SH$_0$  |      | real  | n+1 | `L_recon + λ L_stats + λ₂ L_rollout + λ₁ L_deriv` |

Notes:
- `L_deriv` requires three consecutive snapshots: adding it to stage 1 would double the computational cost (3 E + 1D instead of 1 E + 1 D);
- SH: `stats_head`, n: `n_rollout_steps;
- ${}^*$: outter layers frozen;
- ${}^\dagger$: mostly.


![structure of stages and checkpoints](/assets/docs/liens_stage_checkpoint_flow.png "structure of stages and checkpoints")


### Reconstruction loss
Compare $x' = D(E(x))$, the microstructure recovered by the AE, to $x$:
$$L_\mathrm{recon} = \left\| x' - x \right\|_2^2.$$
This is done in real space. $L_1$ may be used if sharper interfaces are desired.


### Statistics loss
Letting $s_i(x)$ denote the normalized _i_-th (out of $N_s$) microstructural statistic (measured, real), $g_i(z)$ the value of that normalized statistic in the `stats_head` (latent) and $w_i$ its weight,
$$L_\mathrm{stats} = \sum_{i=1}^{N_s} w_i \left[g_i(z) - s_i(x)\right]^2.$$
(See below for more details.)


### Derivative loss
One compares $z_1(t)$ and $[z_0(t+\Delta t) - z_0(t)] / \Delta t$.
See latent-space derivative (step 2) below. Note: in stage 1 the small dense NN for the statistics is trained along the encoder and decoder, in 2 it is frozen.


### One-step latent prediction loss
Compare the prediction to the ground truth in latent space:
$$L_\mathrm{1step} = \left\| z_1(t) - [z_0(t+\Delta t) - z_0(t)] / \Delta t \right\|_2^2.$$
Doing so in latent space avoids reliance on the decoder, cleanly separating decoder loss and prediction loss.


### Multi-step rollout loss
By having several predictions in a row, we let error accumulate:
$$z_0(t_{k}) \xrightarrow{f} \hat{z}_0(t_{k+1}) \xrightarrow{f} \hat{z}_0(t_{k+2}) \xrightarrow{f} \hat{z}_0(t_{k+3}) \xrightarrow{f} \ldots.$$
When starting from the snapshot at time $t_k$, the rollout loss is:
$$L_\mathrm{rollout} = \sum_{i=1}^{N_r} \left\| \hat{z}_0(t_{k+i}) - z_0(t_{k+i}) \right\|_2^2,$$
where $\hat{z}_0(t_{k+i+1}) = \hat{z}_0(t_{k+i}) + z_1(t_{k+i})\,\Delta t_i$ and $\Delta t_i = t_{k+i+1} - t_{k+i}$. Perhaps weigh later predictions slightly more? (The first prediction is easy, long-term stability is what matters.)



## Physics-informed statistics

### Motivation
The latent representation serves two purposes:
- recover the microstructure in real space (decoder),
- predict the microstructure at $t + \Delta t$.

The reconstruction loss alone does not constrain the latent representation to preserve physically meaningful features. Auxiliary losses in the `stats_head`, based on microstructural statistics, nudge the encoder toward latent variables that capture characteristics such as phase fraction, interface density, anisotropy and characteristic length scales. This is expected to improve latent-space organization and, consequently, the accuracy and stability of the learned surrogate dynamics.


### Statistics
- Overall metrics:
  - average(OP)
  - stdev(OP)
  - fraction(OP < 0.1) and fraction(OP > 0.9) [in the case of an order parameter between 0 and 1]
  - energy
- fraction in grain boundary (GB):
  - average norm of OP gradients
- Anisotropy (see below)
- Length scale:
  - first peak in autocorrelation: length and strength.


### No live calculations
I just have a small dense net (`stats_head`) with $N_s$ output cells and say: "the values of these must match the statistics calculated in real space", without recalculating the statistics on x' (let alone on $\hat{z}$). Statistics are auxiliary prediction targets rather than differentiable image-derived losses. The statistics head is trained in latent space, only from ground-truth statistics computed offline.

```text
  latent (1024)
      ↓
Linear(1024 → 128)
    ReLU
Linear(128 → Ns)
```


### Anisotropy
Compute, pixel by pixel, 
$$J_\sigma = G_\sigma \ast \left(\!\nabla z\ \nabla z^{\!\top}\right),$$
with $G_\sigma$ Gaussian kernel. Compute eigenvalues $\lambda_1 \ge \lambda_2$ and eigenvectors $v_1$ and $v_2$. Derive:
- Anisotropy measure: $A = (\lambda_1 - \lambda_2) / (\lambda_1 + \lambda_2 + \varepsilon)$, with $\varepsilon > 0$ a small constant for numerical stability.
  - $A \approx 0$: isotropic,
  - $A \approx 1$: strong directional structure.
- Local orientation (normal direction): $\arctan(v_{1y} / v_{1x})$.	



## Predicting in latent space

### Split latent space
One wants to get a latent representation which could reconstruct, while preserving internal logic. We introduce a split latent channel:
- C$_0$ encodes the microstructure (the "state", 0th order): $z_0(t)$ is such that $D(z_0(t));
- C$_1$ encodes the time derivative (1st order): $\mathrm{d}z_0 / \mathrm{d}t = \dot{z}_0$ (unlike C$_0$, C$_1$ is not decoded: it works purely in latent space). 

A Taylor expansion of $z_0$ gives
$$z_0(t + \Delta t) = z_0(t) + \dot{z}_0(t)\,\Delta t + \ddot{z}_0(t)\,(\Delta t^2/2) + o(\Delta t^2).$$
The idea is to replace $\dot{z}_0$ with $z_1$ by training in stage 2.


### Initial training
- stage 1: C$_0$ trained through AE (`L_recon0 + λ L_stats0`)
- stage 2: improved latent representation of the link between C$_0$ and C$_1$: $z_1(t)$ learns $\frac{z_0(t+\Delta t) - z_0(t)}{\Delta t}$.
- stage 3 starts from
  - $z_0(t+\Delta t) = z_0(t) + \Delta t z_1(t)$,
  - $z_1(t+\Delta t)$ from was it learned,
  - 3a: `L_1step`, 3b: `L_rollout`

The asymmetry between C$_0$ and C$_1$ in on the number of channels, not size, in the latent space (still true?).


### stage 2
Working directly in latent space is faster (it is smaller than real space), but _a priori_ we cannot do arithmetic in it. 

Training the AE (stage 1) ensures that the microstructure (the state) can be reconstructed in real space. This means that the latent representation $z_0$ somehow describes $x$, not that is a smooth and structured coordinate system. 

The goal of stage 2 is to have a meaningful latent representation of the relationship between C$_0$ and C$_1$, $z_1 \approx \mathrm{d}z_0 / \mathrm{d}t$, by training $z_1(t)$ against $[z_0(t+\Delta t) - z_0(t)] / \Delta t$ (*) directly (`L_deriv`). This makes latent-space arithmetic more intuitive: we can have the prediction $$\tilde{z}_0(t + \Delta t) = z_0(t) + z_1(t)\,\Delta t + \dot{z}_1(t)\,(\Delta t^2/2).$$

This involves adding `L_deriv` to the loss function of stage 1 (with a paired-window dataset for derivative calculation).

In stage 2, we are changing the latent representation, while maintaining the reconstruction of C$_0$. We can freeze the outter layers of both encoder and decoder (not those close to the latent space) for regularization.

#### (*) Calculating the derivative
In fact, we use a symmetric, second-order discretization. We expand $z_0(t+\Delta t_+)$ and $z_0(t-\Delta t_-)$. The weighted sum, $z_0(t+\Delta t_+) / {\Delta t_+}^{\!2} - z_0(t-\Delta t_-) / {\Delta t_-}^{\!2}$, removes the second-order term, and one finally obtains 
$$\dot{z}_0(t)  =
{} \frac{z_0(t+\Delta t_+) \, {\Delta t_-}^{\!2} - z_0(t-\Delta t_-) \, {\Delta t_+}^{\!2} - z_0(t) ( {\Delta t_-}^{\!2} - {\Delta t_+}^{\!2}) }{(\Delta t_+ + \Delta t_-)\Delta t_+ \, \Delta t_- }
{} + o(\Delta t_+) + o(\Delta t_-).$$
$z_1(t)$ is trained against this.


### stage 3 (LDS)
Provided with $z_0(t)$ and $z_1(t)$, one must learn $z_0(t+\Delta t)$ and $z_1(t+\Delta t)$.

#### fθ and dz1 / dt
$f_\theta$ does not need to be trained to the first-order term, it only needs to learn $\ddot{z}_0$ (curvature, $\approx \dot{z}_1$), plus the gap between $z_1$ and $\dot{z}_0$. Then $f_\theta(z_0(t), z_1(t))$ should be trained against
$$[z_0(t + \Delta t) - z_0(t) - z_1(t)\,\Delta t] / (\Delta t^2/2),$$
with $\theta$ (currently just the temperature) as further input.
Finally,
$$z_0(t + \Delta t) = z_0(t) + z_1(t)\,\Delta t + f_\theta(z_0(t), z_1(t)) \, (\Delta t^2/2) + o(\Delta t^2).$$

In practice, the second order is not multiplied by $\Delta t ^2$ but by  $\min(\Delta t, \Delta t_\mathrm{cap})^2$, to avoid the risk of a blow-up at large $\Delta t$.


#### gθ and d²z1 / dt²
A similar Taylor expansion for $z_1$ gives
$$z_1(t + \Delta t) = z_1(t) + \dot{z}_1(t)\,\Delta t + \ddot{z}_1(t)\,(\Delta t^2/2) + o(\Delta t^2).$$
Since $f_\theta(z_0(t), z_1(t))$ is trained to approximate $\dot{z}_1(t)$, we can add a $g_\theta$ approximating $\ddot{z}_1$ (i.e. $\dot{f}_\theta$), by training $g_\theta(z_0(t), z_1(t))$ against
$$[z_1(t + \Delta t) - z_1(t) - f_\theta(z_0(t), z_1(t))\,\Delta t] / (\Delta t^2/2),$$
again accounting for $\theta$.

In stage 3a, we use only one step, `L_1step`. Stage 3b will involve several consecutive steps (`L_rollout`)​.



### Interpolation and perturbation
In what follows $\mathrm{stats}(z)$ is the vector of statistics, of norm $\|\mathrm{stats}(z)\|$, calculated by the small dense NN described above. It cannot be the statistics of $x$ in the real world, since this would say nothing about latent representation.

Interpolation in latent space should preserve physical plausibility, not just visual smoothness.
One takes $t_1 < t_2 < t_3$ three successive time steps in the same simulation, with real states $x_i$ and latent representations $z_i = E(x_i)$. We compare $z_2$ to the interpolated $\tilde{z} = (1-\alpha) z_1 + \alpha z_3$, with $\alpha = (t_2-t_1) / (t_3-t_1)$: we measure and minimize $\|\mathrm{stats}(\tilde{z}) - \mathrm{stats}(z_2)\| \, / \, {\|\mathrm{stats}(z_2)\|}$.

Let $z_{\varepsilon} = z + \varepsilon \, \eta$, with $\eta \sim \mathcal{N}(0, 1)$. One expects that $\mathrm{stats}(z_{\varepsilon}) \approx \mathrm{stats}(z) + \varepsilon \Delta S$. So a linear regression of $\|\mathrm{stats}(z_{\varepsilon}) - \mathrm{stats}(z)\|$ with several values of $\varepsilon$ can nudge towards:
- intercept $\approx 0$ (no discontinuity),
- $R^2 \approx 1$ (e.g. not curvature). 
Variant: structured perturbations (directional latent shifts) instead of isotropic noise.

Interpolation and perturbation are currently used as _post-hoc_ diagnostic, not as loss functions.



## Encoder refinement (step 4) and end-to-end (stage 5)
In stage 1 the autoencoder was trained for reconstruction and stats-accuracy of the state. Stage 2 focus on the relationship between $z_1$ and $z_0$ in latent space. Stage 3 trained `f` to predict dynamics, with E and D frozen. Stage 4 is the first time the encoder must seek a latent representation balancing reconstruction (with the decoder) and dynamics prediction (along with LDS). 

D is frozen, even though `L_recon` is in the loss function: this is what distinguishes stage 4 from stage 5. D is a tether keeping E's output compatible with the existing decoder. (Since the encoder is no longer frozen, the latent representation of each sample cannot be cached, unlike in stage 3.)
