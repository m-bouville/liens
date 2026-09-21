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
    assert "s2.pt (epoch 20, val_loss=1.0)" == pp["source_stage2"].split("\\")[-1]
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
    assert "s2.pt (epoch 20, val_loss=1.0)" in r["source_stage2"]     # path + pin from log prose
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
    # the PATH part (before any " (epoch...)" pin) must not keep stray punctuation
    _path = pp["source_stage2"].split(" (epoch")[0]
    assert not _path.endswith((":", ",", ";", ")"))


def test_exact_stem_match_confirms_epoch_present(tmp_path):
    """The real incident: an exact-stem-matching log that does NOT yet contain the
    requested epoch (e.g. queried at an epoch beyond what the log has logged so
    far) must be REFUSED, not returned -- returning it caused a KeyError at the
    call site when the fill loop indexed per_epoch[epoch]."""
    stem = "128x128-stage2-20260912_15h34"
    lg = _mklog(tmp_path, stem, [(1, "3.38", "15h34"), (4, "2.69", "15h34")])  # only up to epoch 4
    parsed = lambda p: bf.parse_log(p)[1:]
    # exact stem match exists, but epoch 20 was never logged
    assert bf.confident_log_for(stem, [lg], 20, parsed) is None
    # an epoch that IS present still matches via exact stem
    assert bf.confident_log_for(stem, [lg], 4, parsed) == lg


def test_backfill_never_raises_on_a_matched_log_missing_the_epoch(tmp_path):
    """End-to-end: even if confident_log_for's guarantee were ever violated, the
    call site must not KeyError -- it should report and leave the row blank."""
    sd = tmp_path / "stage2"; sd.mkdir()
    stem = "128x128-stage2-20260912_15h34"
    _write_ckpt(sd / (stem + ".pt"), {"latent_channels": 8}, 20)   # epoch 20 on the checkpoint
    _mklog(sd, stem, [(1, "3.38", "15h34"), (4, "2.69", "15h34")])  # log only reaches epoch 4
    ev = sd / "eval-stage2.csv"
    _eval_csv(ev, [(f"checkpoints/stage2/{stem}.pt", 20)])
    bf.backfill(sd, ev, dry_run=False)     # must not raise
    r = _read(ev)[0]
    assert not (r.get("val_rollout") or r.get("val_recon0") or "").strip()


def test_prose_head_hidden_does_not_leak_but_param_block_kept(tmp_path):
    """The 'upgrading ... residual head (head_hidden=64); H is zero-init.' PROSE line
    must NOT leak a bare 'head_hidden' param, while the param-BLOCK's deriv_head_hidden
    is kept. (Prose with parens + sentence punctuation is not a param line.)"""
    lg = tmp_path / "128x128-stage2-20260913_15h10.log"
    lg.write_text(
        "upgrading the 'deriv' stream to a residual head (head_hidden=64); H is zero-init.\n"
        "other parameters:\n"
        "  size=128  deriv_head_hidden=64  dynamics_mode=deriv_linear  lr=2.5e-05\n"
        "/100 train = recon0/7e-05 | valid | ema\n"
        " 13  1 =1(0.5)| 2 =1+0.1(0.5)| 2 -> saved at 15:10\n")
    pp = bf.parse_log_params(lg)
    assert pp.get("deriv_head_hidden") == "64"     # from the param block -> p_deriv_head_hidden
    assert "head_hidden" not in pp                  # the bare prose one must not leak
    # clean scalars on the real param line still captured
    assert pp["size"] == "128" and pp["lr"] == "2.5e-05" and pp["dynamics_mode"] == "deriv_linear"


def test_prune_stale_baseline_rows(tmp_path):
    """A baseline row that carries a MODEL epoch is pre-fix residue (baselines are
    model-independent and now key with a blank epoch). prune removes exactly those,
    leaving blank-epoch baselines and all model rows untouched."""
    ev = tmp_path / "eval-stage3b.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "eval_variant", "rollout_median_corr_dx"])
        w.writerow(["checkpoints/stage3b/m.pt", "500", "rollout6", "0.61"])          # model row -- keep
        w.writerow(["baseline:previous_derivative:enc", "387", "previous_derivative_rollout6", "0.65"])  # stale -- drop
        w.writerow(["baseline:previous_derivative:enc", "2466", "previous_derivative_rollout6", "0.63"]) # stale -- drop
        w.writerow(["baseline:previous_derivative:enc", "", "previous_derivative_rollout6", "0.64"])     # correct -- keep
        w.writerow(["checkpoints/stage2/s2.pt", "20", "stage2_z0z1dt_rollout6", "0.22"])                 # stale stage2 -- drop
    removed = bf.prune_stale_baseline_rows(ev, dry_run=False)
    assert removed == 3
    rows = _read(ev)
    assert len(rows) == 2
    kinds = {(r["checkpoint_path"].startswith("baseline"), (r["epoch"] or "").strip()) for r in rows}
    # remaining: the model row, and the blank-epoch baseline
    assert any(r["checkpoint_path"] == "checkpoints/stage3b/m.pt" for r in rows)
    assert any(r["checkpoint_path"].startswith("baseline") and not (r["epoch"] or "").strip() for r in rows)


def test_prune_dry_run_changes_nothing(tmp_path):
    ev = tmp_path / "eval-stage3b.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "eval_variant"])
        w.writerow(["baseline:previous_derivative:enc", "387", "previous_derivative_rollout6"])
    before = ev.read_text()
    bf.prune_stale_baseline_rows(ev, dry_run=True)
    assert ev.read_text() == before      # dry run must not write


# --------------------------------------------------------------------------- #
# components are stored RAW (before weight/scale), not as printed contributions
# --------------------------------------------------------------------------- #
def test_components_are_unweighted_to_raw_using_header_weight_and_scale(tmp_path):
    """The log's breakdown prints weight*raw/scale (contributions that sum to the
    total). The ledger must store the RAW loss, recovered as contribution*scale/weight
    from the header's "[w*]name/scale" terms -- otherwise a change of rollout_scale
    alters the stored 'rollout' with no change in the model. Hand-checked against a
    real stage-3a line."""
    lg = tmp_path / "128x128-stage3a-20260911_21h32.log"
    lg.write_text(
        "/2000 train = 1*rollout/1e-08 +0.2*stats0_predict/0.001 +0.1*z0_growth/0.005 | valid | ema\n"
        "1046  3.147 = 2.803 + 0.206 + 0.138 ( 1.428) |  2.433 = 2.022 + 0.273 + 0.139 ( 0.830) | 2.456  -> saved at 22:25\n")
    _n, pe, _sv = bf.parse_log(lg)
    r = pe[1046]
    assert r["rollout"] == pytest.approx(2.022 * 1e-8 / 1.0)
    assert r["stats0_predict"] == pytest.approx(0.273 * 0.001 / 0.2)
    assert r["z0_growth"] == pytest.approx(0.139 * 0.005 / 0.1)


def test_implicit_weight_one_when_header_term_has_no_multiplier(tmp_path):
    """'recon0/7e-05' (no 'w*') means weight 1: raw = contribution*scale."""
    lg = tmp_path / "128x128-stage2-20260827_06h59.log"
    lg.write_text(
        "/100 train = recon0/7e-05 +0.25*stats0/0.3 +0.5*deriv/7e-08 | valid | ema\n"
        " 14| 3.0 = 2.0 + 0.5 + 0.5 | 3.7900 = 3.6463 + 0.1473 + 2.5771 | 3.79\n")
    _n, pe, _sv = bf.parse_log(lg)
    r = pe[14]
    assert r["recon0"] == pytest.approx(3.6463 * 7e-05 / 1.0)      # implicit weight 1
    assert r["stats0"] == pytest.approx(0.1473 * 0.3 / 0.25)
    assert r["deriv"] == pytest.approx(2.5771 * 7e-08 / 0.5)


def test_single_component_log_without_header_terms_is_left_as_is(tmp_path):
    """An old single-component log has no '[w*]name/scale' header, so there is
    nothing to un-weight with: the lone value is stored unchanged (as comp0)."""
    lg = tmp_path / "128x128-stage3a-20260831_04h36.log"
    lg.write_text("/6000  train  (1step)   valid  (1step)     ema\n"
                  "5460   1.342 ( 0.685),  1.229 ( 0.589) | 1.251234  -> saved at 04:18\n")
    _n, pe, _sv = bf.parse_log(lg)
    assert pe[5460]["comp0"] == pytest.approx(1.229)


def test_backfill_marks_log_filled_components_as_raw(tmp_path):
    sd = tmp_path / "stage3a"; sd.mkdir()
    stem = "128x128-stage3a-20260911_21h29"
    _write_ckpt(sd / (stem + ".pt"), {"latent_channels": 4}, 1118)
    (sd / "128x128-stage3a-20260911_21h32.log").write_text(
        "/2000 train = 1*rollout/1e-08 +0.2*stats0_predict/0.001 | valid | ema\n"
        " 1118  2 =1+1(0.5)| 2.41 =2.0 + 0.27 ( 0.8) | 2.5  -> saved at 21:29\n")
    ev = sd / "eval-stage3a.csv"
    _eval_csv(ev, [(f"checkpoints/stage3a/{stem}.pt", 1118)])
    bf.backfill(sd, ev, dry_run=False)
    r = _read(ev)[0]
    assert r["val_components_kind"] == "raw"
    assert float(r["val_rollout"]) == pytest.approx(2.0 * 1e-8)   # raw, not the printed 2.0


def test_inactive_term_is_stored_blank_not_zero(tmp_path):
    """A term with weight 0 (inactive) prints a contribution of 0 in the breakdown,
    which says NOTHING about its raw loss. It must be stored as unknown (blank),
    never as 0 -- a stored 0 reads as a perfect loss. Same for a contribution of
    exactly 0 under a nonzero weight (the term was off that epoch)."""
    lg = tmp_path / "128x128-stage4-20260830_15h01.log"
    lg.write_text(
        "/30 train = 0*rollout/0.5 +0.2*recon0/0.0002 +0.05*stats0/0.3 | valid | ema\n"
        " 15| 1.0 = 0 + 0.5 + 0.5 | 0.4046 = 0 + 0.1955 + 0.0435 | 0.40  -> saved at 15:01\n")
    _n, pe, _sv = bf.parse_log(lg)
    r = pe[15]
    assert r["rollout"] is None                                   # weight 0 -> unknown
    assert r["recon0"] == pytest.approx(0.1955 * 0.0002 / 0.2)     # active terms still raw
    assert r["stats0"] == pytest.approx(0.0435 * 0.3 / 0.05)


def test_zero_contribution_under_nonzero_weight_is_blank(tmp_path):
    lg = tmp_path / "128x128-stage4-20260901_07h29.log"
    lg.write_text(
        "/30 train = 1*rollout/0.5 +0.2*recon_predict/0.2 | valid | ema\n"
        " 32| 1.0 = 0.5 + 0.5 | 0.8968 = 0.3261 + 0 | 0.90  -> saved at 07:29\n")
    _n, pe, _sv = bf.parse_log(lg)
    assert pe[32]["recon_predict"] is None     # printed 0 -> term off -> unknown
    assert pe[32]["rollout"] == pytest.approx(0.3261 * 0.5 / 1.0)


def test_backfill_writes_blank_cell_for_inactive_term(tmp_path):
    sd = tmp_path / "stage4"; sd.mkdir()
    stem = "128x128-stage4-20260830_15h01"
    _write_ckpt(sd / (stem + ".pt"), {"latent_channels": 4}, 15)
    (sd / (stem + ".log")).write_text(
        "/30 train = 0*rollout/0.5 +0.2*recon0/0.0002 | valid | ema\n"
        " 15| 1 = 0 + 1 | 0.4 = 0 + 0.1955 | 0.4  -> saved at 15:01\n")
    ev = sd / "eval-stage4.csv"
    _eval_csv(ev, [(f"checkpoints/stage4/{stem}.pt", 15)])
    bf.backfill(sd, ev, dry_run=False)
    r = _read(ev)[0]
    assert (r.get("val_rollout") or "").strip() == ""      # blank, NOT "0" and NOT "None"
    assert float(r["val_recon0"]) == pytest.approx(0.1955 * 0.0002 / 0.2)


def test_collapse_redundant_empty_variant_row(tmp_path):
    """An empty-variant row whose data is a subset of a tagged sibling (float
    formatting aside) is redundant and dropped; a tagged row and unique empty
    rows are kept."""
    ev = tmp_path / "eval-stage4.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "eval_variant", "val_loss",
                    "rollout_median_corr_dx"])
        # redundant empty twin (rounded val_loss) + its tagged superset
        w.writerow(["checkpoints/stage4/a.pt", "35", "", "7.357999802", ""])
        w.writerow(["checkpoints/stage4/a.pt", "35", "rollout6", "7.357999801635742", "0.699"])
        # a unique empty row with NO tagged sibling -> keep
        w.writerow(["checkpoints/stage4/b.pt", "20", "", "0.39", ""])
    removed = bf.collapse_redundant_empty_variant_rows(ev, dry_run=False)
    assert removed == 1
    rows = _read(ev)
    assert len(rows) == 2
    a_rows = [r for r in rows if "a.pt" in r["checkpoint_path"]]
    assert len(a_rows) == 1 and a_rows[0]["eval_variant"] == "rollout6"   # kept the tagged one
    assert any("b.pt" in r["checkpoint_path"] for r in rows)              # unique empty kept


def test_legitimate_parenthetical_param_group_is_not_treated_as_prose(tmp_path):
    """'(a0=1.0, b=1.0, kappa=0.2, M=0.05, phi_max=1.5)' is a clean comma-separated
    key=value group, not prose -- it must be scanned, not dropped wholesale just
    because it has '(' and ', '. Real prose ('(head_hidden=64); H is zero-init.')
    must still be rejected."""
    lg = tmp_path / "128x128-stage4-20260908_08h25.log"
    lg.write_text(
        "allen_cahn_weight=0.25  allen_cahn_scale=0.0001  all_steps=False  "
        "(a0=1.0, b=1.0, kappa=0.2, M=0.05, phi_max=1.5)\n"
        "other parameters:\n  min_step=2000  normalize_phi=False\n"
        "/30 train = x | valid | ema\n"
        " 8  1 =1(0.5)| 2 =1(0.5)| 2  -> saved at 08:25\n")
    pp = bf.parse_log_params(lg)
    assert pp["allen_cahn_weight"] == "0.25"
    assert pp["all_steps"] == "False"
    assert pp["normalize_phi"] == "False"       # not shadowed by the paren-group fix


def test_bare_alias_suppressed_when_qualified_name_present(tmp_path):
    """A bare 'a0' inside a '(a0=1.0, b=1.0, kappa=0.2)' summary group must not
    create p_a0 when 'allen_cahn_a0=1.0' is also printed in the same block --
    same physical parameter, two names; keep only the qualified one."""
    lg = tmp_path / "x.log"
    lg.write_text(
        "  allen_cahn_weight=0.25  all_steps=False  (a0=1.0, b=1.0, kappa=0.2, M=0.05, phi_max=1.5)\n"
        "other parameters:\n"
        "  allen_cahn_a0=1.0  allen_cahn_b=1.0  allen_cahn_kappa=0.2  allen_cahn_mobility=0.05\n"
        "  allen_cahn_phi_max=1.5  lr=2e-07\n"
        "/30 train = x | valid | ema\n 8  1 =1(0.5)| 2 =1(0.5)| 2  -> saved at 04:25\n")
    pp = bf.parse_log_params(lg)
    for bare in ("a0", "b", "kappa", "M", "phi_max"):
        assert bare not in pp, f"bare {bare!r} leaked despite a qualified counterpart"
    assert pp["allen_cahn_a0"] == "1.0" and pp["allen_cahn_kappa"] == "0.2"
    assert pp["allen_cahn_weight"] == "0.25"        # unrelated key unaffected


def test_drop_aliased_param_columns(tmp_path):
    ev = tmp_path / "eval-stage4.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "p_a0", "p_allen_cahn_a0", "p_lr"])
        w.writerow(["checkpoints/stage4/x.pt", "8", "1", "1", "2e-07"])
        w.writerow(["checkpoints/stage4/y.pt", "9", "", "1", "1e-07"])   # blank alias, fine to drop
    n = bf.drop_aliased_param_columns(ev, dry_run=False)
    assert n == 1
    hdr = list(_read(ev)[0].keys())
    assert "p_a0" not in hdr and "p_allen_cahn_a0" in hdr and "p_lr" in hdr


def test_drop_aliased_param_columns_keeps_bare_when_values_disagree(tmp_path):
    """If the bare and qualified columns ever hold DIFFERENT values for the same
    row, they are not safely aliased -- do not drop (data integrity over tidiness)."""
    ev = tmp_path / "eval-stage4.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "p_a0", "p_allen_cahn_a0"])
        w.writerow(["checkpoints/stage4/x.pt", "8", "1", "2"])   # disagree
    n = bf.drop_aliased_param_columns(ev, dry_run=False)
    assert n == 0
    assert "p_a0" in list(_read(ev)[0].keys())


def test_stage4_provenance_prose_fills_both_sources(tmp_path):
    """Stage-4/5 prints a DIFFERENT provenance line than stage-3
    ('Stage 4: loaded E/D/stats_head from <path> (...), f_theta from <path2> (...)')
    -- both the stage-2 encoder and the stage-3 f_theta it refines must be captured."""
    lg = tmp_path / "x.log"
    lg.write_text(
        "Stage 4: loaded E/D/stats_head from checkpoints/stage2/128x128-stage2.pt "
        "(epoch 12, val_loss=3.29466), f_theta from checkpoints/stage3b/128x128-stage3b.pt "
        "(epoch 5671, val_loss=1.43711)\n"
        "/30 train = x | valid | ema\n 8  1 =1(0.5)| 2 =1(0.5)| 2  -> saved at 04:25\n")
    pp = bf.parse_log_params(lg)
    # sources are PINNED with (epoch, val_loss) so the generic overwritten path stays traceable
    assert "128x128-stage2.pt (epoch 12, val_loss=3.29466)" in pp["source_stage2"]
    assert "128x128-stage3b.pt (epoch 5671, val_loss=1.43711)" in pp["source_stage3"]


def test_drop_out_of_scope_columns_removes_empty_dynamics_params_from_stage4(tmp_path):
    sd = tmp_path / "stage4"; sd.mkdir()
    ev = sd / "eval-stage4.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "p_lr", "p_dynamics_mode",
                    "p_derivative_source", "p_derivative_time"])
        w.writerow(["checkpoints/stage4/x.pt", "8", "2e-07", "", "", ""])
    n = bf.drop_out_of_scope_columns(ev, dry_run=False)
    assert n == 3
    hdr = list(_read(ev)[0].keys())
    assert "p_dynamics_mode" not in hdr and "p_lr" in hdr


def test_drop_out_of_scope_columns_never_drops_a_column_holding_data(tmp_path):
    """Even if out of scope, a column with real data must never be silently removed."""
    ev = tmp_path / "eval-stage4.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "p_dynamics_mode"])
        w.writerow(["checkpoints/stage4/x.pt", "8", "deriv_linear"])   # has data
    n = bf.drop_out_of_scope_columns(ev, dry_run=False)
    assert n == 0
    assert "p_dynamics_mode" in list(_read(ev)[0].keys())


def test_drop_out_of_scope_columns_is_noop_for_a_dynamics_stage(tmp_path):
    ev = tmp_path / "eval-stage3a.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "p_dynamics_mode"])
        w.writerow(["checkpoints/stage3a/x.pt", "8", ""])
    n = bf.drop_out_of_scope_columns(ev, dry_run=False)
    assert n == 0    # stage3a IS a dynamics stage -- nothing out of scope


def test_row_missing_params_ignores_out_of_scope_columns():
    """A stage-4 row must NOT be flagged as needing backfill just because its
    out-of-scope dynamics columns are blank (they can never be filled). But an
    in-scope source_stage2 still counts."""
    filled = {"p_size": "128", "p_latent_channels": "4", "p_normalize_phi": "False",
              "p_lr": "2e-7", "p_n_rollout_steps": "6",
              "source_stage2": "a", "source_stage3": "b", "resume_from": "c",
              "p_dynamics_mode": "", "p_derivative_source": "", "p_derivative_time": ""}
    assert bf._row_missing_params(filled, "stage4") is False   # only OOS blanks remain
    # for stage3a the dynamics params ARE in scope (must be filled), and
    # source_stage3 is OUT of scope (its blank must not flag)
    stage3a = dict(filled, p_dynamics_mode="d", p_derivative_source="s",
                   p_derivative_time="t", source_stage3="")
    assert bf._row_missing_params(stage3a, "stage3a") is False
    # and a stage3a row with a dynamics param blank IS flagged
    assert bf._row_missing_params(dict(stage3a, p_dynamics_mode=""), "stage3a") is True


def test_collapse_uses_canonical_key_across_path_spellings(tmp_path):
    """collapse groups the empty twin and its tagged superset even when their
    checkpoint_path is spelled differently (absolute vs relative)."""
    ev = tmp_path / "eval-stage4.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "eval_variant", "val_loss", "rollout_median_corr_dx"])
        w.writerow([r"D:\\work\\checkpoints\\stage4\\a.pt", "35", "", "7.36", ""])       # empty twin, absolute
        w.writerow(["checkpoints/stage4/a.pt", "35", "rollout6", "7.36", "0.70"])   # tagged, relative
    removed = bf.collapse_redundant_empty_variant_rows(ev, dry_run=False)
    assert removed == 1
    assert len(_read(ev)) == 1 and _read(ev)[0]["eval_variant"] == "rollout6"


def test_source_paths_stripped_to_checkpoints_preserving_separator_and_pin(tmp_path):
    """Absolute source paths from the log are shortened to 'checkpoints...' with the
    original separator kept and any (epoch...) pin preserved."""
    lg = tmp_path / "x.log"
    lg.write_text(
        "Stage 4: loaded E/D/stats_head from D:\\work\\NN\\phase_field\\python\\checkpoints\\stage2\\128x128-stage2.pt "
        "(epoch 12, val_loss=3.29), f_theta from D:\\work\\checkpoints\\stage3b\\128x128-stage3b.pt (epoch 5671, val_loss=1.44)\n"
        "/30 train = x | valid | ema\n 8  1 =1(0.5)| 2 =1(0.5)| 2  -> saved at 04:25\n")
    pp = bf.parse_log_params(lg)
    assert pp["source_stage2"] == "checkpoints\\stage2\\128x128-stage2.pt (epoch 12, val_loss=3.29)"
    assert pp["source_stage3"] == "checkpoints\\stage3b\\128x128-stage3b.pt (epoch 5671, val_loss=1.44)"


def test_strip_source_path_prefixes_cleanup(tmp_path):
    ev = tmp_path / "eval-stage4.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "source_stage2", "source_stage3"])
        w.writerow(["checkpoints/stage4/x.pt", "8",
                    "D:\\work\\checkpoints\\stage2\\a.pt (epoch 1, val_loss=2)",
                    "D:\\work\\checkpoints\\stage3b\\b.pt"])
    n = bf.strip_source_path_prefixes(ev, dry_run=False)
    assert n == 2
    r = _read(ev)[0]
    assert r["source_stage2"] == "checkpoints\\stage2\\a.pt (epoch 1, val_loss=2)"
    assert r["source_stage3"] == "checkpoints\\stage3b\\b.pt"


def test_collapse_merges_unique_cells_before_dropping_empty_twin(tmp_path):
    """An empty-variant row carrying ch*_imp (not on the tagged sibling) is not a
    pure subset -- collapse must MERGE the importances into the tagged rollout row
    and drop the empty one, treating formatting-only differences (case, display
    rounding) as non-conflicts."""
    ev = tmp_path / "eval-stage4.csv"
    with open(ev, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_path", "epoch", "eval_variant",
                    "rollout_median_corr_dx", "val_rollout", "p_normalize_phi", "ch0_imp"])
        # empty twin: has ch0_imp + full-precision val_rollout + "False"
        w.writerow(["checkpoints/stage4/a.pt", "8", "", "", "0.9740859270", "False", "0.007"])
        # tagged: has the metric + ROUNDED val_rollout + "FALSE", no ch0_imp
        w.writerow(["checkpoints/stage4/a.pt", "8", "rollout6", "0.692", "0.9741", "FALSE", ""])
    removed = bf.collapse_redundant_empty_variant_rows(ev, dry_run=False)
    assert removed == 1
    rows = _read(ev)
    assert len(rows) == 1
    r = rows[0]
    assert r["eval_variant"] == "rollout6"          # tag preserved
    assert r["rollout_median_corr_dx"] == "0.692"   # metric kept
    assert r["ch0_imp"] == "0.007"                  # importance MERGED in


def test_scales_captured_from_loss_term_header(tmp_path):
    """The per-term SCALES live only in the loss-term header ("/N train = w*name/scale
    +...")," not the flat key=value block -- so eval-stageN.csv had weights but not
    scales. parse_log_params must emit p_*_scale from the header. Also guards the
    early-break bug: a save-schedule report line ("4114 runs (94%):") must NOT stop
    parsing before the header is reached."""
    lg = tmp_path / "128x128-stage5-20260915_02h07.log"
    lg.write_text(
        "82028 train windows, 31768 val windows\n"
        "     4114 runs (94.3%):  71 saved steps, last at 6,000,000\n"   # must NOT break here
        "rollout_weight=0.2  lr=6e-06\n"
        "/ 80 train = 0.2*rollout/10.0 +0.2*recon0/0.08 +0.5*allen_cahn/1e-06 | valid | ema\n"
        "   1| 12.6 = 1(0.5) | 13.8 = 1(0.6) | 13.8  -> saved at 02:07\n")
    pp = bf.parse_log_params(lg)
    assert pp["rollout_scale"] == "10.0"          # from the header, previously lost
    assert pp["recon0_scale"] == "0.08"
    assert pp["allen_cahn_scale"] == "1e-06"
    assert pp["lr"] == "6e-06"                     # flat-block param still captured
    assert pp["rollout_weight"] == "0.2"


def test_save_schedule_report_line_does_not_stop_param_parsing(tmp_path):
    """Regression: '<int> runs (...)' matched the epoch-line break and stopped
    parsing before the header/flat block -- masking scales for stage 4/5 logs."""
    lg = tmp_path / "x.log"
    lg.write_text(
        "      236 runs ( 5.4%):  73 saved steps\n"
        "lr=2e-07  n_rollout_steps=6\n"
        "/6000 train = 1*rollout/1e-08 | valid | ema\n"
        " 1| 1=1(0.5)|2=1(0.6)|2 -> saved at 00:00\n")
    pp = bf.parse_log_params(lg)
    assert pp["lr"] == "2e-07" and pp["n_rollout_steps"] == "6"   # reached, not cut off
    assert pp["rollout_scale"] == "1e-08"


def test_stage5_resume_from_stage4_is_captured(tmp_path):
    """Stage 5 prints its stage-4 source as '(resuming from <path>)' and
    'Stage 5: loaded resumed from <path>' -- lowercase / different wording than
    stage-2/3's 'Resuming from'. All must be captured into resume_from, else
    eval-stage5.csv has no stage-4 provenance."""
    lg = tmp_path / "128x128-stage5-20260915_02h07.log"
    lg.write_text(
        "STAGE 5: end-to-end refinement (resuming from D:\\w\\checkpoints\\stage4\\128x128-stage4-20260914_17h02.pt)\n"
        "Stage 5: loaded resumed from D:\\w\\checkpoints\\stage4\\128x128-stage4-20260914_17h02.pt\n"
        "lr=6e-06\n/80 train = 1*rollout/10 | valid | ema\n 1| 1=1(0.5)|2=1(0.6)|2 -> saved at 02:07\n")
    pp = bf.parse_log_params(lg)
    assert pp["resume_from"].endswith("128x128-stage4-20260914_17h02.pt")


def test_stage23_resume_wording_still_captured(tmp_path):
    """Regression: the stage-2/3 'Resuming from <path>' wording must still match
    after broadening the pattern for stage-4/5."""
    lg = tmp_path / "128x128-stage3b-x.log"
    lg.write_text(
        "Resuming from checkpoints/stage3a/128x128-stage3a-20260831_04h18.pt\n"
        "lr=0.002\n/6000 train = 1*rollout/1e-08 | valid | ema\n"
        " 1| 1=1(0.5)|2=1(0.6)|2 -> saved at 04:18\n")
    pp = bf.parse_log_params(lg)
    assert pp["resume_from"].endswith("128x128-stage3a-20260831_04h18.pt")


def test_importances_broadcast_to_all_horizon_rows(tmp_path):
    """Channel importances are checkpoint-level (identical across rollout horizons),
    so a checkpoint with rollout6 AND rollout8 rows must get the importances on
    BOTH -- not just the first (which left the other's ch*_imp empty)."""
    from utils.eval_log import upsert_eval_metrics
    f = tmp_path / "eval-stage5.csv"; ck = "checkpoints/stage5/x.pt"
    upsert_eval_metrics(f, ck, 8, {"rollout_median_corr_dx": 0.7, "rollout_n_steps": 6},
                        eval_variant="rollout6")
    upsert_eval_metrics(f, ck, 8, {"rollout_median_corr_dx": 0.6, "rollout_n_steps": 8},
                        eval_variant="rollout8")
    upsert_eval_metrics(f, ck, 8, {"ch0_imp": 0.6, "ch1_imp": 0.06},
                        owned_cols=["ch0_imp", "ch1_imp"], broadcast=True)
    rows = _read(f)
    assert len(rows) == 2                                  # no new row
    assert all((r.get("ch0_imp") or "").strip() == "0.6" for r in rows)   # BOTH filled
    # rollout metrics stay per-variant
    assert {r["rollout_n_steps"] for r in rows} == {"6", "8"}
