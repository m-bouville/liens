"""`_cache_info.txt` makes the opaque blake2b-named latent-cache subdirs
self-describing. Static block written once; 'cache last accessed' refreshed on
every call (including cache hits); the date is NOT part of the directory key."""
from pathlib import Path

from training.latent_cache import write_cache_info


def _info():
    return {"streams cached": "z0 (state) + z1 (deriv)",
            "ae_checkpoint": "checkpoints/stage2/x.pt", "latent_channels": 8}


def test_content_and_layout(tmp_path):
    write_cache_info(tmp_path, "abc123", 128, info=_info())
    text = (tmp_path / "128x128-abc123" / "_cache_info.txt").read_text()
    for needle in ("fingerprint     = abc123", "size            = 128",
                   "streams cached  = z0 (state) + z1 (deriv)",
                   "ae_checkpoint", "latent_channels = 8"):
        assert needle in text
    # the access time is cache metadata, not a key input: separated by a blank line
    assert "\n\ncache last accessed = " in text
    assert "independent of WHEN it was built" in text


def test_timestamp_refreshes_but_identity_is_stable(tmp_path):
    write_cache_info(tmp_path, "abc123", 128, info=_info())
    p = tmp_path / "128x128-abc123" / "_cache_info.txt"
    first = p.read_text()
    write_cache_info(tmp_path, "abc123", 128, info=_info())
    second = p.read_text()
    strip = lambda t: "\n".join(l for l in t.splitlines()
                                if not l.startswith("cache last accessed"))
    assert strip(first) == strip(second)     # static block identical
    assert "cache last accessed = " in second


def test_post_hoc_on_existing_dir(tmp_path):
    """A pre-change cache subdir (latents present, no info file) gets the note on
    its next access, without touching the latents."""
    d = tmp_path / "128x128-old"; d.mkdir()
    (d / "run-x-both.pt").write_bytes(b"latents")
    write_cache_info(tmp_path, "old", 128, info=_info())
    assert (d / "_cache_info.txt").exists()
    assert (d / "run-x-both.pt").read_bytes() == b"latents"


def test_failure_is_swallowed(tmp_path):
    """A missing note must never break caching: unwritable target -> no raise."""
    blocker = tmp_path / "128x128-x"
    blocker.write_text("a FILE where the dir should be")   # mkdir will fail
    write_cache_info(tmp_path, "x", 128, info=_info())      # must not raise
