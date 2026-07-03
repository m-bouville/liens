import matplotlib.pyplot    as plt
import matplotlib.animation as animation

from   pathlib  import Path


from   utils    import load_datasets as load


def show_snapshot(path: str | Path, nx: int, ny: int,
                   ax=None, cmap: str = "RdBu", vmin: float = -1.0, vmax: float = 1.0,
                   title: str | None = None):
    """
    Display a single raw binary snapshot (as written by save_phi_half),
    without going through the solver's PNG export.
    """
    phi = load.read_phi_half(path, nx, ny)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(phi, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title or Path(path).name)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def make_video(run_dir: str | Path, metadata: "load.RunMetadata",
               output_path: str | Path, fps: int = 10,
               cmap: str = "RdBu", vmin: float = -1.0, vmax: float = 1.0):
    """
    Build a video from a run's saved snapshots, in step order.
    Skips steps whose file is missing rather than failing outright --
    check_snapshots_saved should be used beforehand for a real completeness check.

    Output format is inferred from output_path's extension (.mp4 needs
    ffmpeg; .gif works everywhere via Pillow).
    """
    run_dir     = Path(run_dir)
    output_path = Path(output_path)
    print(f"Making video {output_path} from snapshots in {run_dir}")

    if output_path.suffix not in (".mp4", ".gif"):
        raise ValueError(
            f"output_path '{output_path}' must end in .mp4 or .gif "
            f"(got suffix '{output_path.suffix}')"
        )

    frames = []
    count  = 0
    print(f"Making a video from the time steps: {metadata.save_steps}")
    for step in metadata.save_steps:
        f = run_dir / load.snapshot_filename(step)
        if f.exists():
            frames.append((step, load.read_phi_half(f, metadata.nx, metadata.ny)))
            count += 1

    if not frames:
        raise ValueError(f"{run_dir}: no snapshot files found to build a video from")
    else:
        print(f"{count} frames")

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0][1], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    title = ax.set_title(f"step {frames[0][0]}")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        step, phi = frames[i]
        im.set_data(phi)
        title.set_text(f"step {step}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)

    writer = "ffmpeg" if output_path.suffix == ".mp4" else "pillow"
    anim.save(output_path, writer=writer, fps=fps)
    plt.close(fig)

    return output_path