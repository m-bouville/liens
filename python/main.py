"""
Scan a sweep defined by config.txt and report the status of each run
directory it implies: COMPLETE, INCOMPLETE, or skipped if missing entirely.

Usage:
    python main.py --config path/to/config.txt --base path/to/datasets
"""

import argparse
from   pathlib import Path
import sys

import torch



from   utils  import load_datasets as load
from   utils  import plots

from   models import encoder, decoder, autoencoder

from   training.datasets import MicrostructureSnapshotDataset
from   training.losses import ReconLoss




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("../config.txt"),
        help="path to the sweep config.txt (default: ../config.txt)",
    )
    parser.add_argument(
        "--base", type=Path, default=Path("../datasets"),
        help="base datasets directory the run dirs live under (default: ../datasets)",
    )
    args = parser.parse_args()

    config = load.read_config(args.config)
    dirs   = load.enumerate_run_dirs(config, base=args.base)

    n_complete   = 0
    n_incomplete = 0
    n_missing    = 0



    # testing models/{encoder, decoder, autoencoder}.py
    x = torch.randn(2, 1, 64, 64)
    z = encoder.Encoder(64)(x)

    x_recon = decoder.Decoder(64)(z)
    print(z.shape, x_recon.shape)

    ae = autoencoder.Autoencoder(64)
    x_recon, z = ae(x)
    print(x_recon.shape, z.shape)  # expect [2, 1, 64, 64], [2, 16, 8, 8]
    assert x_recon.shape == torch.Size([2,  1, 64, 64])
    assert z      .shape == torch.Size([2, 16,  8,  8])
    # [2, 16, 8, 8]: (batch, latent_channels, 8, 8),


    # Testing /training/datasets.py
    config = load.read_config("../config.txt")
    ds = MicrostructureSnapshotDataset.from_sweep(config, base="../datasets")
    print(len(ds), ds[0].shape)
    assert ds[0].shape == torch.Size([1, config.nx, config.ny])


    # Testing losses
    ae = autoencoder.Autoencoder(512)
    recon_loss = ReconLoss()

    x = torch.randn(2, 1, 512, 512)
    x_recon, z = ae(x)
    loss = recon_loss(x_recon, x)
    print(loss.item())


    sys.exit()




    for d in dirs:
        if not d.exists():
            n_missing += 1
            continue  # ignore missing runs entirely, as requested

        if load.is_complete(d):
            n_complete += 1
            print(f"COMPLETE    {d}")

            metadata = load.read_metadata(d / "metadata.txt")
            check    = load.check_snapshots_saved(d, metadata)
            if check["missing"] or check["bad_size"]:
                print(f"            ! {len(check['missing'])} missing, "
                      f"{len(check['bad_size'])} bad size")

            # statistics = load.read_statistics_csv(d)

            # value_max = 0.7

            # plots.show_snapshot(d / "t0100000", config.nx, config.ny,
            #                     ax=None, cmap="RdBu", vmin=-value_max, vmax=value_max)

            # plots.make_video(d, metadata, output_path = "./my_video.mp4",
            #                  fps=5, cmap="RdBu", vmin=-value_max, vmax=value_max)

        else:
            n_incomplete += 1

            metadata = load.read_metadata(d / "metadata.txt")
            check    = load.check_snapshots_saved(d, metadata)
            print(f"INCOMPLETE  {d}  first missing step {check['missing'][0]}")

    total = len(dirs)
    print()
    print(
        f"{total} possible runs in sweep -> "
        f"{n_complete} complete, {n_incomplete} incomplete, "
        f"{n_missing} missing (ignored)"
    )


if __name__ == "__main__":
    main()
