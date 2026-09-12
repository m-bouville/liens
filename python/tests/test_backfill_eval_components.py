"""
Tests for evaluation.backfill_eval_components.

Targets the bugs that RECURRED (and were twice wrongly declared fixed): a
checkpoint's components/params must come ONLY from the run that produced it,
identified by the log's `-> saved at HH:MM` annotation matching the checkpoint
stamp. Cross-RUN and cross-SIZE contamination (an 09/09 checkpoint filled from an
11/09 log; a 128x128 row filled from a 64x64 log) are the failures these guard.
"""
import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import evaluation.backfill_eval_components as bf


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_log(path, header_params, epoch_lines):
    """epoch_lines: list of (epoch, val_total, saved_at 'HHhMM')."""
    lines = [header_params, "/6000 train = rollout/1e-08 | valid ... | ema"]
    for ep, val, sv in epoch_lines:
        hh, mm = sv.split("h")
        lines.append(f" {ep}  1.0 =1.0 ( 0.5) | {val} =1.0 + 0.1 + 0.1 ( 0.5) | {val}  -> saved at {hh}:{mm}")
    path.write_text("\n".join(lines) + "\n")


def _write_ckpt(path, config, epoch, val_loss=1.0):
    torch.save({"config": config, "epoch": epoch, "val_loss": val_loss}, path)


def _eval_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch"])
        for cp, ep in rows:
            w.writerow([cp, ep])


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# log parsing
# --------------------------------------------------------------------------- #
def test_parse_log_params_extracts_training_params_and_source(tmp_path):
    lg = tmp_path / "128x128-stage3a-20260911_21h32.log"
    _write_log(lg, "min_step=2000  n_rollout_steps=2  lr=0.002  dynamics_mode='deriv_linear'",
               [(1118, "2.41", "21h29")])
    lg.write_text(lg.read_text().replace(
        "/6000 train",
        "Loaded frozen encoder from checkpoints\\stage2\\s2.pt (epoch 20, val_loss=1.0, latent_channels=4)\n/6000 train"))
    pp = bf.parse_log_params(lg)
    assert pp["n_rollout_steps"] == "2" and pp["lr"] == "0.002"
    assert pp["dynamics_mode"] == "deriv_linear"
    assert pp["source_stage2"].endswith("s2.pt")
    assert pp["latent_channels"] == "4"           # clean, no trailing paren


def test_parse_log_components_uses_saved_at(tmp_path):
    lg = tmp_path / "128x128-stage3a-20260831_04h36.log"
    _write_log(lg, "lr=0.002", [(5460, "1.229", "04h18")])
    _n, per_epoch, saved_at = bf.parse_log(lg)
    assert 5460 in per_epoch
    assert saved_at[5460] == "04h18"


# --------------------------------------------------------------------------- #
# confident_log_for: the core anti-contamination guard
# --------------------------------------------------------------------------- #
def _mklog(tmp_path, stem, epoch_lines):
    lg = tmp_path / (stem + ".log")
    _write_log(lg, "lr=0.002", epoch_lines)
    return lg


def test_matches_own_log_via_saved_at(tmp_path):
    lg = _mklog(tmp_path, "128x128-stage3a-20260831_04h36", [(5460, "1.229", "04h18")])
    parsed = lambda p: bf.parse_log(p)[1:]        # (per_epoch, saved_at)
    m = bf.confident_log_for("128x128-stage3a-20260831_04h18", [lg], 5460, parsed)
    assert m == lg


def test_NO_cross_run_contamination_different_day(tmp_path):
    """An 09/09 checkpoint must NOT be filled from an 11/09 log, even though that
    log contains the same epoch number. (The bug that was 'fixed' twice.)"""
    own = _mklog(tmp_path, "128x128-stage3a-20260909_11h20", [(476, "2.3", "11h18")])
    other = _mklog(tmp_path, "128x128-stage3a-20260911_21h32", [(476, "9.9", "21h05")])
    parsed = lambda p: bf.parse_log(p)[1:]
    # checkpoint saved at 11h18 on 09/09 -> only the 09/09 log's saved-at matches
    m = bf.confident_log_for("128x128-stage3a-20260909_11h18", [own, other], 476, parsed)
    assert m == own, "must pick the 09/09 log, never the 11/09 one"


def test_NO_cross_size_contamination(tmp_path):
    """A 128x128 checkpoint must NOT be filled from a 64x64 log."""
    log64 = _mklog(tmp_path, "64x64-stage2-20260818_07h27", [(7, "1.0", "07h20")])
    parsed = lambda p: bf.parse_log(p)[1:]
    m = bf.confident_log_for("128x128-stage2-20260911_11h25", [log64], 7, parsed)
    assert m is None, "different size prefix must never match"


def test_refuses_when_no_saved_at_matches(tmp_path):
    lg = _mklog(tmp_path, "128x128-stage3a-20260831_04h36", [(5460, "1.229", "09h99".replace("99", "59"))])
    parsed = lambda p: bf.parse_log(p)[1:]
    m = bf.confident_log_for("128x128-stage3a-20260831_04h18", [lg], 5460, parsed)
    assert m is None


# --------------------------------------------------------------------------- #
# end-to-end backfill
# --------------------------------------------------------------------------- #
def test_backfill_fills_components_and_params_from_correct_sources(tmp_path):
    sd = tmp_path / "stage3a"; sd.mkdir()
    stem = "128x128-stage3a-20260911_21h29"
    _write_ckpt(sd / (stem + ".pt"), {"latent_channels": 4, "normalize_phi": True}, 1118, 2.41)
    lg = sd / "128x128-stage3a-20260911_21h32.log"
    _write_log(lg, "n_rollout_steps=2  lr=0.002", [(1118, "2.41", "21h29")])
    lg.write_text(lg.read_text().replace(
        "/6000 train",
        "Loaded frozen encoder from checkpoints\\stage2\\s2.pt (epoch 20, val_loss=1.0, latent_channels=4)\n/6000 train"))
    ev = sd / "eval-stage3a.csv"
    _eval_csv(ev, [(f"checkpoints/stage3a/{stem}.pt", 1118)])

    bf.backfill(sd, ev, dry_run=False)
    r = _read(ev)[0]
    assert r["p_n_rollout_steps"] == "2" and r["p_lr"] == "0.002"     # training params from log
    assert r["p_latent_channels"] == "4"                              # architecture from config
    assert r["source_stage2"].endswith("s2.pt")                       # provenance from log prose
    assert r["val_loss"] == str(2.41)                                 # total from checkpoint


def test_backfill_does_not_clobber_existing_cells(tmp_path):
    sd = tmp_path / "stage3a"; sd.mkdir()
    stem = "128x128-stage3a-20260911_21h29"
    _write_ckpt(sd / (stem + ".pt"), {"latent_channels": 4}, 1118)
    lg = sd / "128x128-stage3a-20260911_21h32.log"
    _write_log(lg, "lr=0.999", [(1118, "2.4", "21h29")])              # log says lr=0.999
    ev = sd / "eval-stage3a.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["checkpoint_path", "epoch", "p_lr"])
        w.writerow([f"checkpoints/stage3a/{stem}.pt", 1118, "0.002"])  # already 0.002
    bf.backfill(sd, ev, dry_run=False)
    assert _read(ev)[0]["p_lr"] == "0.002", "existing value must not be overwritten"


def test_backfill_leaves_unmatched_row_blank_not_fabricated(tmp_path):
    sd = tmp_path / "stage3a"; sd.mkdir()
    stem = "128x128-stage3a-20260909_11h20"
    _write_ckpt(sd / (stem + ".pt"), {"latent_channels": 4}, 476)
    # a log from a DIFFERENT run/day whose saved-at won't match this checkpoint
    _mklog(sd, "128x128-stage3a-20260911_21h32", [(476, "9.9", "21h05")])
    ev = sd / "eval-stage3a.csv"
    _eval_csv(ev, [(f"checkpoints/stage3a/{stem}.pt", 476)])
    bf.backfill(sd, ev, dry_run=False)
    r = _read(ev)[0]
    # params still fill from the checkpoint config; but component/source (log-only)
    # must NOT be fabricated from the wrong log
    assert not (r.get("source_stage2") or "").strip(), "must not source from a wrong-run log"


def test_own_log_deleted_only_wrong_run_survives_refuses(tmp_path):
    """The real incident: a run's OWN log was deleted (runs thrown out), and only an
    OLDER run's log survives that happens to reach the same epoch. Must REFUSE, not
    fill from the surviving wrong-run log (which would stamp the row with the wrong
    stage-2 encoder). This is the contamination the current matcher must never make."""
    sd = tmp_path / "stage3a"; sd.mkdir()
    stem = "128x128-stage3a-20260909_11h20"
    _write_ckpt(sd / (stem + ".pt"), {"latent_channels": 4}, 476)
    # only the 04h18 run's log survives; it reaches epoch 476 but saved it at 04:11,
    # not at this checkpoint's 11h20 stamp
    lg = sd / "128x128-stage3a-20260831_04h36.log"
    lg.write_text(
        "Loaded frozen encoder from checkpoints\\stage2\\128x128-stage2-20260827_02h22.pt "
        "(epoch 4, val_loss=1, latent_channels=8)\n"
        "/6000 train = rollout/1e-08 | valid|ema\n"
        " 476  1 =1(0.5)| 2.3 =1+0.1+0.1(0.5)| 2.3 -> saved at 04:11\n")
    parsed = lambda p: bf.parse_log(p)[1:]
    assert bf.confident_log_for(stem, [lg], 476, parsed) is None


def test_all_seeds_rows_from_pt_files_and_creates_csv(tmp_path):
    """--all: with no CSV yet, seed a row per .pt (epoch read from each checkpoint),
    create the file, and fill params from each config."""
    sd = tmp_path / "stage3a"; sd.mkdir()
    _write_ckpt(sd / "128x128-stage3a-20260831_04h18.pt", {"latent_channels": 8}, 5460)
    _write_ckpt(sd / "128x128-stage3a-20260911_21h29.pt", {"latent_channels": 4}, 1118)
    ev = sd / "eval-stage3a.csv"
    assert not ev.exists()
    bf.backfill(sd, ev, dry_run=False, all_pts=True)
    assert ev.exists()
    rows = _read(ev)
    assert len(rows) == 2
    lc = {r["checkpoint_path"].split("/")[-1]: r.get("p_latent_channels") for r in rows}
    assert lc["128x128-stage3a-20260831_04h18.pt"] == "8"
    assert lc["128x128-stage3a-20260911_21h29.pt"] == "4"


def test_all_is_idempotent_no_duplicate_rows(tmp_path):
    sd = tmp_path / "stage3a"; sd.mkdir()
    _write_ckpt(sd / "128x128-stage3a-20260911_21h29.pt", {"latent_channels": 4}, 1118)
    ev = sd / "eval-stage3a.csv"
    bf.backfill(sd, ev, dry_run=False, all_pts=True)
    bf.backfill(sd, ev, dry_run=False, all_pts=True)   # re-run
    assert len(_read(ev)) == 1, "re-running --all must not duplicate rows"


def test_without_all_missing_csv_is_a_noop(tmp_path):
    """Without --all, a missing CSV is not created (backfill only fills existing rows)."""
    sd = tmp_path / "stage3a"; sd.mkdir()
    _write_ckpt(sd / "128x128-stage3a-20260911_21h29.pt", {"latent_channels": 4}, 1118)
    ev = sd / "eval-stage3a.csv"
    bf.backfill(sd, ev, dry_run=False, all_pts=False)
    assert not ev.exists()


def test_parse_log_params_ignores_prose_lists_and_parens(tmp_path):
    """A stage-2 log's 'Resuming from ... (stat_names=[...], ...)' and 'head_hidden=64)'
    prose lines must NOT corrupt the param scan with tokens like "['angle" or "64)"."""
    lg = tmp_path / "128x128-stage2-20260827_06h59.log"
    lg.write_text(
        "Resuming from checkpoints/stage2/prev.pt (stat_names=['angle', 'anisotropy'], "
        "ancestor_stats_weight=1, this stage's stats0_weight=0.25)\n"
        "upgrading the 'deriv' stream to a residual head (head_hidden=64); H is zero-init.\n"
        "trunk_from_deriv_weight=0.2: L_deriv gradient into the trunk scaled by 0.2 (1.0=full).\n"
        "other parameters:\n"
        "  size=128  lr=0.0001  stats0_scale=0.3  deriv_scale=7e-08\n"
        "/6000 train = x | valid | ema\n"
        " 7  1 =1(0.5)| 1.4 =1+0.1+0.1(0.5)| 1.4 -> saved at 07:30\n")
    pp = bf.parse_log_params(lg)
    # clean scalar params captured
    assert pp["size"] == "128" and pp["lr"] == "0.0001" and pp["stats0_scale"] == "0.3"
    # NO garbage from lists/parens/prose
    for k, v in pp.items():
        assert not any(c in k for c in "[]()';:"), f"garbage key {k!r}"
        assert not any(c in str(v) for c in "[]()';:"), f"garbage value {k}={v!r}"
    assert "stat_names" not in pp                 # a list -> never a scalar param


def test_provenance_paths_strip_trailing_prose_punctuation(tmp_path):
    """'Resuming from <path>: single-stream ...' must yield the path WITHOUT the
    trailing ':' -- a checkpoint path never ends in punctuation, and a stray ':'
    breaks anything that treats resume_from as a real path."""
    lg = tmp_path / "128x128-stage2a-20260910_10h35.log"
    lg.write_text(
        "Resuming from checkpoints/stage1/128x128-stage1.pt: single-stream (stage 1a) checkpoint\n"
        "Loaded frozen encoder from checkpoints/stage2/s2.pt) (latent_channels=8)\n"
        "/80 train = recon0/7e-05 | valid | ema\n"
        " 5  1 =1(0.5)| 2 =1+0.1(0.5)| 2 -> saved at 10:35\n")
    pp = bf.parse_log_params(lg)
    assert pp["resume_from"] == "checkpoints/stage1/128x128-stage1.pt"   # no trailing ':'
    assert not pp["resume_from"].endswith((":", ",", ";", ")"))
    assert not pp["source_stage2"].endswith((":", ",", ";", ")"))
