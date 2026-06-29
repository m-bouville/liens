# Neural networks: autoencoder and surrogate evolution




## Autoencoder

### Architecture
The convolutional autoencoder has a symmetric encoder–decoder architecture. The encoder depth scales with the (square) system size so that the spatial bottleneck remains 8×8: three downsampling stages for 64×64 inputs (64→32→16→8), five for 256×256, and so on. Each resolution level consists of two padded 3×3 convolutions with ReLU activations, followed by a stride-2 convolution for downsampling (mirrored by learned upsampling in the decoder), with BatchNorm/LayerNorm.

The encoder terminates with a 1×1 convolution reducing the feature dimension to 16 channels, yielding an 8×8×16 latent representation. This bottleneck retains coarse spatial organization while reducing the dimensionality sufficiently for efficient latent-space dynamics. Initial runs will use a smaller latent space of 4×4×8, for speed and to find the lower bound for accuracy.


### Using a U-Net?
Using a U-Net instead of a pure AE would improve the decoding. But skip connections may interfere with the LDS. Since the decoder will have more information (latent representation + skips) than the LDS (only latent), the encoder-decoder pair may work even if the latent representation has little information (which would break the LDS). 

To avoid this, skip connections will be added and trained only after freezing the encoder. Step 4 (or separate step just before or after) now includes: train skip connections and retrain decoder (encoder still frozen). Possible to alternate in step 4: train encoder + decoder (+ skips) and train encoder + LDS?



## Training Stages and loss functions

### Training Stages

There are four losses, which can be mixed and matched at different steps:
- reconstruction loss: $L_\mathrm{recon}$,
- statistics loss: $L_\mathrm{stats}$,
- one-step latent prediction loss: $L_\mathrm{1step}$,
- multi-step rollout loss: $L_\mathrm{rollout}$.

| Stage                | Train    | Space | Loss                   |
|----------------------|----------|-------|------------------------|
| 2. AE                | (E, D)   | real  | `L_recon + λ₁ L_stats`  |
| 4. LDS               | (f)      | latent| `L_rollout` (or `L_1step` initially) |
| 5. Encoder refinement| (E, f)   | latent| `L_rollout + ε L_recon + λ₁ L_stats` |
| 6. End-to-end        | (E, f, D)| real  | `L_recon + λ₁ L_stats + λ₂ L_rollout` |


### Reconstruction loss
Compare $x' = D(E(x))$, the microstructure recovered by the AE, to $x$:
$$L_\mathrm{recon} = \left\| x' - x \right\|_2^2.$$
This is done in real space. $L_1$ may be used if sharper interfaces are desired.


### Statistics loss
Letting $s_i(x)$ denote the _i_-th (out of $N_s$) microstructural statistic and $w_i$ its weight,
$$L_\mathrm{stats} = \sum_{i=1}^{N_s} w_i \left[g_i(x') - s_i(x)\right]^2$$
in real space, or $g_i(\hat{z})$ instead of $g_i(x')$ in latent space. (See below for more details.)


### One-step latent prediction loss
Compare the prediction to the ground truth in latent space:
$$L_\mathrm{1step} = \left\| [z(t) + f_\theta(z(t), \Delta t)] - z(t+\Delta t) \right\|_2^2.$$
Doing so in latent space avoids reliance on the decoder, cleanly separating decoder loss and prediction loss.


### Multi-step rollout loss
By having several predictions in a row, we let error accumulate:
$$z(t_{k}) \xrightarrow{f} \hat{z}(t_{k+1}) \xrightarrow{f} \hat{z}(t_{k+2}) \xrightarrow{f} \hat{z}(t_{k+3}) \xrightarrow{f} \ldots.$$
When starting from the snapshot at time $t_k$, the rollout loss is:
$$L_\mathrm{rollout} = \sum_{i=1}^{N_r} \left\| \hat{z}(t_{k+i}) - z(t_{k+i}) \right\|_2^2,$$
where $\hat{z}(t_{k+i+1}) = \hat{z}(t_{k+i}) + f_\theta(\hat{z}(t_{k+i}), \Delta t_i)$ and $\Delta t_i = t_{k+i+1} - t_{k+i}$. Perhaps weight later predictions slightly more? (The first prediction is easy, long-term stability is what matters.)



## Physics-informed tatistics
The latent representation serves two purposes:
- recover the microstructure in real space (decoder),
- predict the microstructure at $t + \Delta t$.

The reconstruction loss alone does not constrain the latent representation to preserve physically meaningful features. Auxiliary losses based on microstructural statistics nudge the encoder toward latent variables that capture characteristics such as phase fraction, interface density, anisotropy and characteristic length scales. This is expected to improve latent-space organization and, consequently, the accuracy and stability of the learned surrogate dynamics.

Simple microstructures will be produced by hand to check that statistics work properly:
- vertical and horizontal stripes of two or three different widths,
- stripes with sinusoidal interfaces (grains are still elongated but interfaces are not purely anisotropic),
- checkerboard.


### Possible statistics
Written for stable states at order parameter OP = 0 and 1. 
- Overall metrics:
  - average(OP) [hard mass-conservation constraint if conserved OP]
  - sum(OP < 0) and sum(OP > 1) [unphysical -> turn into penalty?]
  - stdev(OP) [proxy for evolution of phases]
  - fraction(OP < 0.1) and fraction(OP > 0.9) [volume of each phase]
  - energy
- fraction in grain boundary (GB):
  - number of (OP ∉ [0.1, 0.9])
  - number of OP gradients with norm > 0
- Anisotropy (of boundaries or grains):
  - OP gradient orientation along [01], [10], [11], [−11] (exclude those with norm ≈ 0)
  - stdev(SMA of OP along x), idem y and ±45 °
- Length scale:
  - first peak in Fourier and/or autocorrelation
    - along [01], [10], [11], [−11]

Note: gradients with norm $\approx 0$ are excluded because the orientation of a (quasi) null vector is meaningless.


### No live calculations
I just have a small dense net with $N_s$ output cells and say: "the values of these must match the statistics calculated in real space", without recalculating the statistics on x' (let alone on $\hat{z}$). Statistics are auxiliary prediction targets rather than differentiable image-derived losses. And they can be used in latent space.

```text
  latent (1024)
      ↓
Linear(1024 → 128)
    ReLU
Linear(128 → Ns)
```


### Variant
Compute, pixel by pixel, 
$$J_\sigma = G_\sigma \ast \left(\!\nabla z\ \nabla z^{\!\top}\right),$$
with $G_\sigma$ Gaussian kernel. Compute eigenvalues $\lambda_1 \ge \lambda_2$ and eigenvectors $v_1$ and $v_2$. Derive:
- Anisotropy measure: $A = (\lambda_1 - \lambda_2) / (\lambda_1 + \lambda_2 + \varepsilon)$, with $\varepsilon > 0$ a small constant for numerical stability.
  - $A \approx 0$: isotropic,
  - $A \approx 1$: strong directional structure.
- Interface density: $\lambda_1 + \lambda_2$
  - high: many interfaces, 
  - low: bulk phases.
- Local orientation (normal direction): $\arctan(v_{1y} / v_{1x})$.	

Store:
- mean($\lambda_1$)
- mean($\lambda_2$)
- mean anisotropy $A$
- stdev($A$)
- interface density
- orientation entropy
- 10-bin orientation histogram (or PCA-reduced)
- spatial histogram pooling [what exactly?]

Total: 15–20 scalars per image. This is enough to:
- constrain latent space,
- detect collapse,
- guide physics consistency.



## Latent-space validation (step 3)
Verify that latent space behaves like a smooth, structured coordinate system rather than a brittle compression code:
- Interpolation: compare $D(z_\alpha)$ to $x_\alpha = (1-\alpha) x_1 + \alpha x_2$, with $z_\alpha = (1-\alpha) z_1 + \alpha z_2$ and $\alpha \in [0, 1]$.
  - interpolation should preserve physical plausibility, not just visual smoothness.
- Perturbing the latent representation ($z_\varepsilon = z + \varepsilon \times \eta$, where $\eta \sim \mathcal{N}(0, 1)$) and decode again: $x_\varepsilon = D(z_\varepsilon)$
  - isotropic noise,
  - structured perturbations (directional latent shifts).
- Temporal consistency check:
  - Take real simulation sequences, $x(t)$ and $x(t+\Delta t)$, and encode both: $z(t) = E(x(t))$ and $z(t+\Delta t) = E(x(t+\Delta t))$. 
  - Compute latent displacement statistics, $\Delta z = z(t+\Delta t) - z(t)$, and check that the distribution of $\Delta z(t)$ is smooth, bounded and not heavy-tailed and that $\Delta z$ is correlated over time (deterministic evolution rather than stochastic latent collapse).
  - check direction consistency: Are transitions consistent? For similar states: $\Delta z$ should be similar in direction/magnitude.
- Reconstruction under partial corruption: mask random patches in input image, encode + decode, check if latent representation still reconstructs global structure (as opposed to local pixel memorization).
- Consistency over a cycle (encode → decode → re-encode): on top of the obvious $D(E(x)) \approx x$, check that $E(D(z)) \approx z$, to detect:
  - decoder hallucination,
  - encoder/decoder mismatch.
