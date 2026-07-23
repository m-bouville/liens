"""
Stage 2 direct diagnostic: is z1's own error, measured against ITS OWN
training target, temperature-dependent -- the direct test of the
hypothesis raised by evaluation/check_parameter_dependence.py's own
T<0.9 vs T>=0.9 split (there, eps/eps'/z0_ddot all flipped sign and grew
5-9x in magnitude at T>=0.9, while f_theta's own bias relative to the
true curvature stayed comparatively stable across the split -- pointing
at z1's error itself, not f_theta, as the more likely root cause).

That earlier finding was INFERRED algebraically, through f_theta's own
downstream residual (see check_parameter_dependence.py's own module
docstring for the full derivation). This script instead measures z1's
error DIRECTLY: train_stage2's own L_deriv target (see train_ae.py) is

    target_deriv(t) = (z0(t+dt) - z0(t)) / dt

-- i.e. EXACTLY the same eps/eps' decomposition applies here too, one
level up the pipeline, with NO f_theta or A(dt^3) term involved at all
(there's no z0_ddot/curvature concept at this level -- z1 vs
target_deriv is a first-order-only comparison):

    z1(t) - target_deriv(t) = eps/dt + eps'

Fits this 2-term model (via the SAME robust_polynomial_fit machinery
check_parameter_dependence.py uses) for ALL DATA, T<0.9, and T>=0.9
separately -- if THIS eps/eps' ALSO flips sign/grows at T>=0.9, that
confirms the root cause is upstream of f_theta entirely, in z1's own
training (stage 2) or in the encoder's own behavior near/above the
critical temperature this project's own metadata centers theta on.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_deriv_temperature --stage2-checkpoint checkpoints/stage2/64x64.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder, EncoderDecoderPair, MultiStreamAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_streams import (
    cross_check_stream_configs_against_state_dict, resolve_stream_configs_from_checkpoint_config,
)
from training.datasets import MicrostructureEvolutionDataset
from utils import load_datasets as load

# Same anchor rationale as check_interpolation.py/check_parameter_dependence.py's
# own _PYTHON_ROOT (see either's docstring) -- never a bare relative path.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/


def robust_polynomial_fit(x: np.ndarray, y: np.ndarray, basis_funcs: list,
                           n_iter: int = 10, huber_delta_scale: float = 1.345):
    """
    Identical to check_parameter_dependence.py's own function of the
    same name -- duplicated here (not imported) specifically so this
    script has no import-time dependency on that module's own, much
    heavier LDS-checkpoint-loading machinery, which this diagnostic
    never needs (it only ever loads a stage-2 AE checkpoint). See that
    module's own docstring for the full IRLS/Huber mechanism.
    """
    X = np.column_stack([f(x) for f in basis_funcs])
    n, p = X.shape
    weights = np.ones(n)

    def _weighted_lstsq(w):
        sqrt_w = np.sqrt(w)
        coefs, *_ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
        return coefs

    coefs = _weighted_lstsq(weights)
    for _ in range(n_iter):
        residuals = y - X @ coefs
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad if mad > 0 else np.std(residuals) + 1e-12
        huber_delta = huber_delta_scale * scale
        abs_resid = np.abs(residuals)
        weights = np.where(abs_resid <= huber_delta, 1.0, huber_delta / np.maximum(abs_resid, 1e-12))
        coefs = _weighted_lstsq(weights)

    residuals = y - X @ coefs
    XtWX = X.T @ (weights[:, None] * X)
    dof = max(n - p, 1)
    sigma2 = np.sum(weights * residuals ** 2) / dof
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
        coef_stderr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        coef_stderr = np.full(p, np.nan)
    return coefs, coef_stderr


def _load_encoder(checkpoint_path: Path, device: torch.device):
    """
    Mirrors check_interpolation.py's own checkpoint-format-detection
    logic EXACTLY (same branches, same order) -- deliberately NOT
    factored into a shared helper at this point in the project's own
    history (that refactor belongs with check_interpolation.py itself,
    not smuggled in as a side effect of this diagnostic), so this is a
    direct copy, adapted only to return just the encoder submodule
    (this script never needs a decoder or stats_head at all).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, config={model_cfg}")

    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, checkpoint["model_state"],
    )
    recon_stream = stream_configs[recon_stream_name]
    decoder_for_stream = model_cfg.get("decoder_for_stream")
    is_flat_checkpoint = any(k.startswith("encoder.") for k in checkpoint["model_state"])
    if is_flat_checkpoint:
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs)
        decoder = Decoder(output_size=model_cfg["size"], out_channels=1,
                           base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        ae = EncoderDecoderPair(encoder, decoder, stream_name=recon_stream_name,
                                 mode=recon_stream.mode).to(device)
    elif len(stream_configs) == 1:
        ae = Autoencoder(
            size=model_cfg["size"], channels=1,
            base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
            latent_spatial_size=recon_stream.spatial_size,
        ).to(device)
    elif decoder_for_stream is None:
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs)
        decoder = Decoder(output_size=model_cfg["size"], out_channels=1,
                           base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                     stream_configs=stream_configs).to(device)
    else:
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs)
        decoders = {}
        for stream_name, decoder_key in decoder_for_stream.items():
            stream_cfg = stream_configs[stream_name]
            decoders[decoder_key] = Decoder(
                output_size=model_cfg["size"], out_channels=1,
                base_channels=model_cfg["base_channels"], latent_channels=stream_cfg.channels,
                latent_spatial_size=stream_cfg.spatial_size,
            )
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders=decoders,
                                     stream_configs=stream_configs,
                                     decoder_for_stream=decoder_for_stream).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()
    ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]

    if "deriv" not in stream_configs:
        raise ValueError(
            f"{checkpoint_path} has no 'deriv' stream (stream_configs={list(stream_configs)}) -- "
            f"this diagnostic needs z1 (the deriv stream) to exist at all, which means a checkpoint "
            f"trained at or after stage 2, not stage 1/1b."
        )
    return ae_encoder, checkpoint.get("test_dirs")


def check_deriv_temperature(
    stage2_checkpoint_path: Path, min_step: int | None = None, min_stdev_phi: float | None = None,
    device: str | None = None,
) -> dict:
    """
    Measures z1(t) - target_deriv(t), target_deriv = (z0(t+dt)-z0(t))/dt
    -- EXACTLY train_stage2's own L_deriv target (see train_ae.py) --
    per window, then fits eps/dt + eps' (see this module's own
    docstring) for ALL DATA, T<0.9, and T>=0.9 separately.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder, test_dirs = _load_encoder(stage2_checkpoint_path, device)
    if not test_dirs:
        raise ValueError(f"{stage2_checkpoint_path} has no saved test_dirs")
    test_dirs = [Path(d) for d in test_dirs]

    # encode_both_streams=True gets z0 AND z1 cached together, in one
    # pass, with the SAME bounded-memory streaming/parallel-read
    # machinery MicrostructureEvolutionDataset already has (see its own
    # docstring) -- reused here rather than a bespoke encode loop.
    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=encoder, device=device, window_length=2,
        min_step=min_step or 0, min_stdev_phi=min_stdev_phi, encode_both_streams=True,
    )
    print(f"Evaluating {len(dataset)} (t, t+dt) windows...")

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    dts, temperatures, residuals_signed = [], [], []
    metadata_cache: dict[Path, object] = {}

    idx = 0
    with torch.no_grad():
        for window, window_deriv, dt_window, theta in loader:
            batch_size = window.shape[0]
            window = window.to(device)
            window_deriv = window_deriv.to(device)
            dt_window = dt_window.to(device)

            z0_t = window[:, 0]
            z0_next = window[:, 1]
            z1_t = window_deriv[:, 0]
            dt = dt_window[:, 0]

            # EXACT match to train_stage2's own L_deriv target -- see
            # this module's own docstring.
            dt_r = dt.view(-1, 1, 1, 1)
            target_deriv = (z0_next - z0_t) / dt_r
            residual = (z1_t - target_deriv).flatten(start_dim=1).mean(dim=1)  # (B,) signed, per-window

            residuals_signed.extend(residual.cpu().tolist())
            dts.extend(dt.cpu().tolist())

            for i in range(batch_size):
                run_dir, steps = dataset.window_info(idx)
                if run_dir not in metadata_cache:
                    metadata_cache[run_dir] = load.read_metadata(run_dir / "metadata.txt")
                temperatures.append(metadata_cache[run_dir].temperature)
                idx += 1

    dts = np.array(dts)
    temperatures = np.array(temperatures)
    residuals_signed = np.array(residuals_signed)

    basis_funcs = [lambda dt: 1.0 / dt, lambda dt: np.ones_like(dt)]

    def _fit_and_report(mask, label):
        n = mask.sum()
        if n < 20:
            print(f"\n  Skipping [{label}] -- only {n} windows, too few for a meaningful fit.")
            return None
        coefs, stderr = robust_polynomial_fit(dts[mask], residuals_signed[mask], basis_funcs)
        eps, eps_prime = coefs
        eps_se, eps_prime_se = stderr
        print(f"\n  [{label}] (n={n}):")
        print(f"    eps  = {eps: .6e}  (stderr {eps_se:.2e}, {abs(eps)/eps_se if eps_se > 0 else float('nan'):.1f} sigma)")
        print(f"    eps' = {eps_prime: .6e}  (stderr {eps_prime_se:.2e}, "
              f"{abs(eps_prime)/eps_prime_se if eps_prime_se > 0 else float('nan'):.1f} sigma)")
        return {"eps": eps, "eps_stderr": eps_se, "eps_prime": eps_prime, "eps_prime_stderr": eps_prime_se, "n": int(n)}

    print("\n" + "=" * 70)
    print(f"z1 direct error decomposition: z1(t) - target_deriv(t) = eps/dt + eps'  [{stage2_checkpoint_path.name}]")
    print("=" * 70)
    result_all = _fit_and_report(np.ones_like(dts, dtype=bool), "ALL DATA")
    t_mask = temperatures < 0.9
    result_lo = _fit_and_report(t_mask, "T < 0.9")
    result_hi = _fit_and_report(~t_mask, "T >= 0.9")
    print("=" * 70)

    return {"all": result_all, "T<0.9": result_lo, "T>=0.9": result_hi,
            "dts": dts, "temperatures": temperatures, "residuals_signed": residuals_signed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--min-stdev-phi", type=float, default=None)
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    check_deriv_temperature(
        stage2_checkpoint_path=args.stage2_checkpoint, min_step=args.min_step,
        min_stdev_phi=args.min_stdev_phi, device=args.device,
    )


if __name__ == "__main__":
    main()
