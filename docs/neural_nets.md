# Neural networks: autoencoder and surrogate evolution



## Autoencoder

### Architecture
The convolutional autoencoder has a symmetric encoder–decoder architecture. The encoder depth scales with the (square) system size so that the spatial bottleneck remains 8×8: three downsampling stages for 64×64 inputs (64→32→16→8), five for 256×256, and so on. Each resolution level consists of two 3×3 convolutions (circular padding to match periodic boundaries) with ReLU activations, followed by a stride-2 convolution for downsampling (mirrored by learned upsampling in the decoder), with BatchNorm/LayerNorm.

The encoder terminates with a 1×1 convolution reducing the feature dimension to 16 channels, yielding an 8×8×16 latent representation. This bottleneck retains coarse spatial organization while reducing the dimensionality sufficiently for efficient latent-space dynamics. Initial runs will use a larger latent space, to ensure that sufficient information is available for reconstruction, before shrinking it to find the lower bound for accuracy.


### Using a U-Net?
Using a U-Net instead of a pure AE would improve the decoding. But skip connections can weaken the usefulness of the latent representation used by the Latent Dynamics Surrogate (LDS). Since the decoder will have more information (latent representation + skips) than the LDS (only latent), the encoder-decoder pair may work even if the latent representation has little information (which would break the LDS). 

To avoid this, skip connections will be added and trained only after freezing the encoder. Step 4 (or separate step just before or after) now includes: train skip connections and retrain decoder (encoder still frozen). Possible to alternate in step 4: train encoder + decoder (+ skips) and train encoder + LDS?



## Training Stages and loss functions

### Dataset expansion
From each snapshot, more are created through
- mirror (horizontal and vertical);
- rotation by $\pm$90 °, 180°;
- transposing;
- translation by (Nx/2, 0), (0, Ny/2), (Nx/2, Ny/2) thanks to periodic boundaries.


### Training Stages

There are five losses, which can be mixed and matched at different steps:
- reconstruction loss: $L_\mathrm{recon}$,
- statistics loss: $L_\mathrm{stats}$,
- interpolation loss: $L_\mathrm{interp}$,
- one-step latent prediction loss: $L_\mathrm{1step}$,
- multi-step rollout loss: $L_\mathrm{rollout}$.

| Stage                | Train     | Space | Loss                   |
|----------------------|-----------|-------|------------------------|
| 1. autoencoder       | (E, D, SH)| real  | `L_recon + λ L_stats`  |
| 2. latent validation | (E, D)    | both  | `L_recon + λ L_stats + λ' L_interp` |
| 3. LDS               | (f)       | latent| `L_1step` (3a), then `L_rollout + ε L_1step` (3b) |
| 4. encoder refinement| (E, f)    | latent| `L_rollout + ε L_recon + λ L_stats |
| 5. end-to-end        | (E, f, D) | real  | `L_recon + λ L_stats + λ'' L_rollout` |

SH: `stats_head` 


### Reconstruction loss
Compare $x' = D(E(x))$, the microstructure recovered by the AE, to $x$:
$$L_\mathrm{recon} = \left\| x' - x \right\|_2^2.$$
This is done in real space. $L_1$ may be used if sharper interfaces are desired.


### Statistics loss
Letting $s_i(x)$ denote the normalized _i_-th (out of $N_s$) microstructural statistic (measured, real), $g_i(z)$ the value of that normalized statistic in the `stats_head` (latent) and $w_i$ its weight,
$$L_\mathrm{stats} = \sum_{i=1}^{N_s} w_i \left[g_i(z) - s_i(x)\right]^2.$$
(See below for more details.)


### Interpolation loss
See latent-space validation (step 2) below. Note: in stage 1 the small dense NN for the statistics is trained along the encoder and decoder, in 2 it is frozen.


### One-step latent prediction loss
Compare the prediction to the ground truth in latent space:
$$L_\mathrm{1step} = \left\| [z(t) + f_\theta(z(t), \Delta t)] - z(t+\Delta t) \right\|_2^2.$$
Doing so in latent space avoids reliance on the decoder, cleanly separating decoder loss and prediction loss.


### Multi-step rollout loss
By having several predictions in a row, we let error accumulate:
$$z(t_{k}) \xrightarrow{f} \hat{z}(t_{k+1}) \xrightarrow{f} \hat{z}(t_{k+2}) \xrightarrow{f} \hat{z}(t_{k+3}) \xrightarrow{f} \ldots.$$
When starting from the snapshot at time $t_k$, the rollout loss is:
$$L_\mathrm{rollout} = \sum_{i=1}^{N_r} \left\| \hat{z}(t_{k+i}) - z(t_{k+i}) \right\|_2^2,$$
where $\hat{z}(t_{k+i+1}) = \hat{z}(t_{k+i}) + f_\theta(\hat{z}(t_{k+i}), \Delta t_i)$ and $\Delta t_i = t_{k+i+1} - t_{k+i}$. Perhaps weigh later predictions slightly more? (The first prediction is easy, long-term stability is what matters.)



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



## Latent-space validation (stage 2)
Training the AE (stage 1) ensures that the initial state can be recovered. This proves that the latent representation $z$ somehow describes $x$, not that is a smooth and structured coordinate system. Doing arithmetic to predict the time evolution would be impossible in a jagged space. For that, latent representation must make sense.

Coding analogy: stage 1 creates code that works (recovers the original), and stage 2 refactors it to make it suitable for stage 3.


### How stage 2 works
In stage 1, there are two constraints: recovering the original (decoder) and getting statistics right (`stats_head`). In stage 2, we add a loss function on qualitative representation (interpolation, perturbation). 

In stage 2, we are changing the latent representation, so we need to change the encoder. And if the decoder did not change, it would no longer reconstruct properly, so we cannot freeze it either. Since neither E nor D is frozen, there could be a synchronized drift. We keep `stats_head` (with frozen coefficients) to prevent this.

We can also freeze the outter layers of both encoder and decoder (not those close to the latent space) for regularization.


### Working in latent space
We work directly in latent space to avoid relying on the decoder. But there, we cannot calculate physical statistics or measure a direct distance between microstructures pixel by pixel (the calculation $z_2 - z_1$ has no meaning). 

In what follows $\mathrm{stats}(z)$ is the vector of statistics, of norm $\|\mathrm{stats}(z)\|$, calculated by the small dense NN described above. It cannot be the statistics of $x$ in the real world, since this would say nothing about latent representation.


### Interpolation
Interpolation in latent space should preserve physical plausibility, not just visual smoothness.
One takes $t_1 < t_2 < t_3$ three successive time steps in the same simulation, with real states $x_i$ and latent representations $z_i = E(x_i)$. We compare $z_2$ to the interpolated $\tilde{z} = (1-\alpha) z_1 + \alpha z_3$, with $\alpha = (t_2-t_1) / (t_3-t_1)$: we measure and minimize $\|\mathrm{stats}(\tilde{z}) - \mathrm{stats}(z_2)\| \, / \, {\|\mathrm{stats}(z_2)\|}$.


### Perturbation
Let $z_{\varepsilon} = z + \varepsilon \, \eta$, with $\eta \sim \mathcal{N}(0, 1)$. One expects that $\mathrm{stats}(z_{\varepsilon}) \approx \mathrm{stats}(z) + \varepsilon \Delta S$. So a linear regression of $\|\mathrm{stats}(z_{\varepsilon}) - \mathrm{stats}(z)\|$ with several values of $\varepsilon$ can nudge towards:
- intercept $\approx 0$ (no discontinuity),
- $R^2 \approx 1$ (e.g. not curvature). 

Perturbation is currently used as _post-hoc_ diagnostic, not as loss function.

Variant: structured perturbations (directional latent shifts) instead of isotropic noise.



## Encoder refinement (step 4) and end-to-end (stage 5)
In stage 1 the encoder was trained for reconstruction and stats-accuracy, and in stage 2 for generic interpolation-smoothness. Stage 3 trained `f` to predict dynamics, with D frozen. Stage 4 is the first time the encoder must seek a latent representation balancing reconstruction (with the decoder) and dynamics prediction (along with LDS). 

D is frozen, even though `L_recon` is in the loss function: this is what distinguishes stage 4 from stage 5. D is a tether keeping E's output compatible with the existing decoder. (Since the encoder is no longer frozen, the latent representation of each sample cannot be cached, unlike in stage 3.)
