"""
The actual pipeline orchestrator: run_from_params_file() runs stages
1->2->3(a/b)->4->5 as specified by one stage-parameters file. See
main.py's own module docstring for the file format, naming convention,
and caching behavior -- this module is deliberately just the
orchestration logic itself; main.py stays a thin CLI entry point
(argument parsing, then calling this).
"""
from pathlib import Path

from evaluation.check_interpolation import check_interpolation
from evaluation.check_latent_channels import check_latent_channels
from evaluation.check_perturbation import check_perturbation
from evaluation.check_reconstruction import check_reconstruction
from evaluation.check_parameter_dependence import check_parameter_dependence
from evaluation.check_rollout import check_rollout
from orchestration.checkpoint_identification import _validate_checkpoint_stage
from orchestration.checkpoint_registry import (
    _find_matching_checkpoint, _make_checkpoint_callback, _report_checkpoint_epoch,
    _signature_kwargs, _upsert_registry,
)
from orchestration.logging_utils import _log_to_file
from orchestration.paths import _PYTHON_ROOT, _STAGE_DIRS
from orchestration.stage_params import (
    _backup_before_overwrite, _prepare_stage_kwargs, _resolve_stage_specific_ancestor,
    _strip_unrecognized_params, parse_stage_params, renamed_keys,
    report_unrecognized_global_params,
)
from training.checkpoint_components import split_joint_checkpoint_for_evaluation
from training.train_stage1 import train_autoencoder
from training.train_stage2 import train_stage2
from training.train_lds import train_lds
from training.train_refinement import train_refinement


def run_from_params_file(params_path: Path, default_base: Path,
                          device: str | None = None) -> Path:
    """
    Runs stages 1->2->3(a/b)->4->5 as specified by a stage-parameters
    file, stopping early and returning whichever checkpoint is the LAST
    one actually produced (stage 1's own if no '# Stage 2' section is
    given -- stage 2 is what builds the multi-stream (deriv-stream)
    ancestor every later stage needs, directly from stage 1's own
    checkpoint (see training/extend_encoder.py), so stage 2 onward is
    skipped entirely without it, same "stop early if not configured"
    convention already used for stage 4/5 below; stage 3's if no
    '# Stage 4' section is given, stage 4's if no '# Stage 5', stage
    5's if both are present).
    See the module docstring for the file format, naming convention, and
    caching behavior (own expected filename, then the parameter registry).
    """
    global_params, stages = parse_stage_params(params_path)
    if not stages:
        print(f"WARNING: {params_path} has no recognized '# Stage N' section headers -- "
              f"every key ended up in the global section and NONE will be used for "
              f"training (stage 1 requires at least --latent-channels, for instance). "
              f"'# Stage N' must be on its own line, exactly '# Stage' followed by a "
              f"number, nothing else on that line.")
    if "1b" in stages:
        # Stage 1b no longer exists as a separate pass -- train_stage2()
        # now builds the deriv stream itself, directly from stage 1's own
        # checkpoint (see training/extend_encoder.py's own module
        # docstring for the full rationale). A '# Stage 1b' section left
        # in an existing params file is silently ignored otherwise (this
        # module never looks it up at all anymore) -- flagged explicitly
        # here so that isn't a silent, confusing no-op.
        print(f"WARNING: {params_path} has a '# Stage 1b' section, but stage 1b no longer "
              f"exists as a separate pass -- train_stage2() now builds the deriv stream "
              f"itself, directly from stage 1's own checkpoint. This section's own keys "
              f"are ignored entirely; move any still-relevant ones (e.g. "
              f"condition_on_theta) into '# Stage 2' instead.")
    base_path = Path(global_params.pop("base", default_base))

    # Nx/Ny: the ONLY source of grid size now (config.txt is no longer
    # read at all -- see module docstring). Checked in both the global
    # section (their intended home) and Stage 1's section (an easy,
    # reasonable place to put them by mistake, since grid size is most
    # associated with the autoencoder) so either placement works.
    stage1_raw = stages.setdefault(1, {})
    nx = global_params.pop("Nx", None) or stage1_raw.pop("Nx", None)
    ny = global_params.pop("Ny", None) or stage1_raw.pop("Ny", None)
    if nx is None or ny is None:
        raise ValueError(f"{params_path}: Nx and Ny are required (config.txt is no longer "
                          f"read for grid size at all) -- give them in the global section "
                          f"or Stage 1's section.")
    nx, ny = int(nx), int(ny)
    if nx != ny:
        raise ValueError(f"{params_path}: only square grids are supported, got Nx={nx}, Ny={ny}")
    size = nx
    extra_signature = {"Nx": nx, "Ny": ny}

    # One global check, here, against the UNION of every stage's training
    # function -- see report_unrecognized_global_params' own docstring for
    # why globals must NOT be reported per-stage (a global is offered to
    # every stage by design, so most are legitimately unusable by most
    # stages). Runs after base/Nx/Ny are popped above, so those don't warn.
    report_unrecognized_global_params(
        global_params, (train_autoencoder, train_stage2, train_lds, train_refinement))

    stem = params_path.stem

    def stage_dir(stage_num: int | str) -> Path:
        """
        This stage's own checkpoint directory, created ON DEMAND rather
        than up front. Replaces an earlier loop that mkdir'd EVERY entry
        in _STAGE_DIRS on every run, which left behind directories for
        stages this run never touches -- checkpoints/stage3 on a 3a/3b
        curriculum run, and (until _STAGE_DIRS dropped the entry
        entirely) checkpoints/stage1b, long after stage 1b stopped
        existing. Empty, permanently-unused directories in a checkpoints
        tree are actively misleading: they look like a stage that ran and
        produced nothing.

        Safe to defer: resolve_checkpoint's own lookups are globs, and
        globbing a directory that doesn't exist yet returns empty rather
        than raising -- so a not-yet-created stage dir reads exactly like
        an empty one, which is what it is.
        """
        d = _STAGE_DIRS[stage_num]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_output_path(stage_num: int) -> Path:
        return stage_dir(stage_num) / f"{stem}-stage{stage_num}.pt"

    def resolve_checkpoint(stage_num: int, force: bool, signature: dict,
                           target_epochs: int | None) -> Path | None:
        """Two-tier cache check, in order: own expected filename, then
        this stage's own parameter registry. Returns an existing path
        to reuse, or None if this stage must actually be (re)trained.
        Either hit is structurally validated before being trusted --
        catches a mismatched/mislabeled file before wasting time
        training on top of it -- and reports the epoch it was actually
        saved at against the target, to flag likely-killed-early runs."""
        own_path = stage_output_path(stage_num)
        if own_path.exists():
            if force:
                print(f"WARNING: {own_path} already exists and will be OVERWRITTEN (force=True)")
            else:
                _validate_checkpoint_stage(own_path, stage_num, device)
                print(f"Stage {stage_num}: found existing {own_path}, reusing.")
                _report_checkpoint_epoch(own_path, target_epochs, device)
                return own_path
        if not force:
            registry_path = stage_dir(stage_num) / f"registry-stage{stage_num}.csv"
            match = _find_matching_checkpoint(registry_path, signature, stage_num)
            if match is not None:
                _validate_checkpoint_stage(match, stage_num, device)
                print(f"Stage {stage_num}: found matching checkpoint in registry "
                      f"({match}), reusing -- parameters are identical to an "
                      f"already-trained run under a different name.")
                _report_checkpoint_epoch(match, target_epochs, device)
                return match
        return None

    # ---- Stage 1: autoencoder ----
    stage1_kwargs = _prepare_stage_kwargs(stages.get(1, {}), global_params)
    force1 = stage1_kwargs.pop("force", False)
    stage1_kwargs = _strip_unrecognized_params(train_autoencoder, stage1_kwargs, "Stage 1",
                                            own_keys=renamed_keys(stages.get(1, {})))
    signature1 = {"base_path": str(base_path),
                  **extra_signature, **_signature_kwargs(stage1_kwargs)}
    stage1_checkpoint = resolve_checkpoint(1, force1, signature1, stage1_kwargs.get("epochs"))
    if stage1_checkpoint is None:
        with _log_to_file(stage_output_path(1).with_suffix(".log")):
            print("=" * 70)
            print("STAGE 1: training autoencoder")
            print("=" * 70)
            registry1_path = stage_dir(1) / "registry-stage1.csv"
            stage1_checkpoint = train_autoencoder(
                size=size, base_path=base_path,
                checkpoint_path=stage_output_path(1), device=device,
                loss_curve_path=_PYTHON_ROOT.parent / "output" / f"stage1/{stage_output_path(1).stem}-loss_curve.png",
                on_checkpoint_saved=_make_checkpoint_callback(registry1_path, signature1),
                **stage1_kwargs,
            )
            print(f"\nStage 1 complete: {stage1_checkpoint}\n")
            _upsert_registry(registry1_path, stage1_checkpoint, signature1)

            print("=" * 70)
            print("Sanity check: reconstruction quality (stage 1 checkpoint)")
            print("=" * 70)
            check_reconstruction(
                checkpoint_path=stage1_checkpoint, device=device,
                min_step=stage1_kwargs.get("min_step", 0),
                min_stdev_phi=stage1_kwargs.get("min_stdev_phi"),
                output_path=_PYTHON_ROOT.parent / "output" / f"stage1/{stage1_checkpoint.stem}-reconstruction.png",
            )
            print()

    # ---- Stage 2: latent-space validation ----
    # Stage 2 now resumes DIRECTLY from stage 1's own checkpoint --
    # there is no more separate stage 1b pass. Stage 1b's own former
    # role (extending the encoder with a fresh deriv bottleneck +
    # theta-conditioner, transferring stage 1's trained weights
    # unchanged) is now done in memory, once, right at the start of
    # train_stage2() itself (see training/extend_encoder.py's own
    # module docstring for the full rationale: stage 1b's own training
    # loop had been inert since it started running at epochs=0, and
    # D1 -- the one thing genuinely built only by that loop's own
    # surrounding setup -- is confirmed permanently unnecessary; deriv
    # lives purely in latent space from stage 2 onward, via L_deriv).
    #
    # Gating moved here from stage 1b's own former section: stage 1's
    # OWN output is still a complete, usable single-stream autoencoder
    # on its own, but stage 2 onward all require the deriv stream that
    # only stage 2 now builds -- so if no '# Stage 2' section is given,
    # the pipeline stops here, at stage 1's own output, same as before
    # (just gated on '2' now, not '1b').
    if 2 not in stages:
        print("No '# Stage 2' section given -- stopping after stage 1 "
              "(stage 2 onward all require stage 2's own deriv-stream "
              "construction).\n")
        return stage1_checkpoint

    stage2_kwargs = _prepare_stage_kwargs(stages.get(2, {}), global_params)
    force2 = stage2_kwargs.pop("force", False)
    stage2_kwargs = _strip_unrecognized_params(train_stage2, stage2_kwargs, "Stage 2",
                                            own_keys=renamed_keys(stages.get(2, {})))
    stage2_kwargs, stage2_resume_from, stage2_overridden = _resolve_stage_specific_ancestor(
        stage2_kwargs, stage1_checkpoint, "Stage 2")
    # Naming note: registries use a consistent "stageN_checkpoint" ancestry
    # convention across ALL stages, independent of whatever the underlying
    # function calls its own parameter (train_stage2 calls it resume_from,
    # train_lds calls it ae_checkpoint_path) -- so "stage1_checkpoint" means
    # the same thing everywhere you look, in any registry. Stage 2's direct
    # ancestor is stage 1 now (train_stage2 builds the deriv stream itself),
    # so the signature reflects THAT ancestry -- UNLESS overridden above (an
    # explicit Stage-2-section resume_from, e.g. continuing an already-
    # trained stage-2 checkpoint's own deriv_target_centered curriculum),
    # in which case the signature reflects the REAL ancestor actually used,
    # not the pipeline's own unused default -- otherwise two runs built from
    # genuinely different ancestors could collide on the same cache key.
    #
    # No more "epochs==0: skip, reuse stage 1b's output" special case --
    # that existed because stage 1b, not stage 2, used to be where the
    # deriv stream got built; skipping stage 2 entirely at epochs=0 would
    # have meant NO deriv stream at all now. train_stage2(epochs=0) still
    # does real, necessary work (building the deriv stream, matching the
    # SAME "epochs=0 ablation" pattern stage 1b/stage 3 already use) --
    # confirmed directly, not assumed: it produces a complete, correctly-
    # shaped checkpoint (PURE_LATENT deriv, no D1, stats_head1 present),
    # not a no-op.
    signature2 = {"base_path": str(base_path),
                   "stage1_checkpoint": str(stage1_checkpoint),
                   **({"resumed_from": str(stage2_resume_from)} if stage2_overridden else {}),
                   **extra_signature, **_signature_kwargs(stage2_kwargs)}
    stage2_checkpoint = resolve_checkpoint(2, force2, signature2, stage2_kwargs.get("epochs"))
    if stage2_checkpoint is None:
        if stage2_overridden:
            # Only here, not for the ordinary stage1->stage2 default
            # above -- THAT default is always a different file from
            # this stage's own output, so nothing is ever at risk of
            # being silently overwritten in the normal, automatic
            # flow. An explicit override, though, commonly points at
            # THIS SAME stage's own prior output (that's the whole
            # point of deriv_target_centered's own curriculum), which
            # force=True is about to overwrite in place -- see
            # _backup_before_overwrite's own docstring for why the
            # .log file matters here just as much as the .pt one.
            _backup_before_overwrite(stage_output_path(2))
            _backup_before_overwrite(stage_output_path(2).with_suffix(".log"))
        with _log_to_file(stage_output_path(2).with_suffix(".log")):
            print("=" * 70)
            print("STAGE 2: latent-space validation (L_deriv fine-tuning)")
            print("=" * 70)
            registry2_path = stage_dir(2) / "registry-stage2.csv"
            stage2_checkpoint = train_stage2(
                base_path=base_path, resume_from=stage2_resume_from,
                checkpoint_path=stage_output_path(2), device=device,
                loss_curve_path=_PYTHON_ROOT.parent / "output" / f"stage2/{stage_output_path(2).stem}-loss_curve.png",
                on_checkpoint_saved=_make_checkpoint_callback(registry2_path, signature2),
                **stage2_kwargs,
            )
            print(f"\nStage 2 complete: {stage2_checkpoint}\n")
            _upsert_registry(registry2_path, stage2_checkpoint, signature2)

            print("=" * 70)
            print("Sanity check: reconstruction quality (stage 2 checkpoint)")
            print("=" * 70)
            check_reconstruction(
                checkpoint_path=stage2_checkpoint, device=device,
                min_step=stage2_kwargs.get("min_step", 0),
                min_stdev_phi=stage2_kwargs.get("min_stdev_phi"),
                output_path=_PYTHON_ROOT.parent / "output" / f"stage2/{stage2_checkpoint.stem}-reconstruction.png",
            )
            print()

            print("=" * 70)
            print("Sanity check: latent channel activations (stage 2 checkpoint)")
            print("=" * 70)
            check_latent_channels(
                ae_checkpoint_path=stage2_checkpoint, device=device,
                min_step=stage2_kwargs.get("min_step", 0),
                min_stdev_phi=stage2_kwargs.get("min_stdev_phi"),
                output_path=_PYTHON_ROOT.parent / "output" / f"stage2/{stage2_checkpoint.stem}-latent_channels.png",
            )
            print()

            print("=" * 70)
            print("Sanity check: interpolation consistency (stage 2 checkpoint)")
            print("=" * 70)
            check_interpolation(
                checkpoint_path=stage2_checkpoint, device=device,
                output_path=_PYTHON_ROOT.parent / "output" / f"stage2/{stage2_checkpoint.stem}-interpolation.png",
            )
            print()

            print("=" * 70)
            print("Sanity check: perturbation response (stage 2 checkpoint)")
            print("=" * 70)
            check_perturbation(
                checkpoint_path=stage2_checkpoint, device=device,
                output_path=_PYTHON_ROOT.parent / "output" / f"stage2/{stage2_checkpoint.stem}-perturbation.png",
            )
            print()


    # ---- Stage 3: LDS -- either single-phase ('# Stage 3') or a
    # two-phase curriculum ('# Stage 3a' + '# Stage 3b': train
    # n_rollout_steps=1 first, stable and fast, then resume with the
    # target n_rollout_steps -- see train_lds()'s resume_from docstring
    # for why). Same train_lds() function and same caching/registry
    # machinery either way; only the params-file section names and
    # whether a resume_from is threaded through differ. ----
    has_3a, has_3b, has_bare_3 = "3a" in stages, "3b" in stages, 3 in stages
    if has_3a != has_3b:
        raise ValueError(f"{params_path}: Stage 3a and Stage 3b must both be given together "
                          f"(3a alone is just a warmup, not a usable final result) -- found "
                          f"only {'3a' if has_3a else '3b'}.")
    if has_bare_3 and has_3a:
        raise ValueError(f"{params_path}: give EITHER '# Stage 3' (single-phase) OR "
                          f"'# Stage 3a' + '# Stage 3b' (curriculum), not both.")

    def run_lds_stage(stage_key: int | str, resume_from: Path | None = None,
                       run_sanity_check: bool = True) -> tuple[Path, bool]:
        """Runs one train_lds() phase -- stage_key is 3 (single-phase),
        or '3a'/'3b' (curriculum). Caching/registry/logging are
        identical regardless of which phase this is; resume_from (only
        given for 3b) is recorded in the signature so a 3b result is
        correctly tied to the specific 3a checkpoint it built on, not
        just its own hyperparameters. run_sanity_check=False for the 3a/
        3b curriculum calls specifically -- their sanity checks are done
        together, with shared samples, right after both exist (see
        has_3a branch below); for the bare-3 case, run_sanity_check
        defaults True and check_rollout picks its own random samples as
        before, no comparison partner to match.

        Returns (checkpoint, freshly_trained) -- freshly_trained is
        False on a cache hit (resolve_checkpoint found an existing
        match), True if train_lds() actually ran this call. The has_3a
        branch uses this to decide whether the shared-sample comparison
        needs regenerating, matching every other stage's convention of
        not re-running sanity checks against a checkpoint that was
        already reused from the registry, not retrained this run."""
        kwargs = _prepare_stage_kwargs(stages.get(stage_key, {}), global_params)
        force = kwargs.pop("force", False)
        kwargs = _strip_unrecognized_params(train_lds, kwargs, f"Stage {stage_key}",
                                            own_keys=renamed_keys(stages.get(stage_key, {})))
        kwargs, resume_from, overridden = _resolve_stage_specific_ancestor(
            kwargs, resume_from, f"Stage {stage_key}")
        # Default to quiet (only print on save/early-stop), not train_lds()'s
        # own default of every-epoch -- stage 3 commonly runs hundreds of
        # epochs, and setdefault respects an explicit log_every_epoch in
        # the params file either way.
        kwargs.setdefault("log_every_epoch", False)
        # BOTH ancestors recorded explicitly -- since two stage-2 checkpoints
        # can share identical stage-2 parameters while differing in stage 1
        # (e.g. a quick vs a fully-trained stage 1), stage2_checkpoint alone
        # (e.g. a quick vs a fully-trained stage 1), stage2_checkpoint alone
        # already disambiguates for MATCHING purposes (it's produced by
        # exactly one stage-1 checkpoint), but recording stage1_checkpoint
        # too means you can see the full ancestry from this ONE registry,
        # without having to open stage 2's own registry and follow ITS
        # stage1_checkpoint field.
        signature = {"base_path": str(base_path),
                      "stage1_checkpoint": str(stage1_checkpoint),
                      "stage2_checkpoint": str(stage2_checkpoint),
                      **({"resumed_from": str(resume_from)} if resume_from is not None else {}),
                      **extra_signature, **_signature_kwargs(kwargs)}
        checkpoint = resolve_checkpoint(stage_key, force, signature, kwargs.get("epochs"))
        freshly_trained = checkpoint is None
        if checkpoint is None:
            if overridden:
                # Only for an EXPLICIT, same-stage override -- not the
                # normal 3b-resumes-from-3a case (resume_from is not
                # None there too, but that's always a DIFFERENT file,
                # 3a's own checkpoint, from 3b's own output, so nothing
                # is ever at risk of being silently overwritten there).
                _backup_before_overwrite(stage_output_path(stage_key))
                _backup_before_overwrite(stage_output_path(stage_key).with_suffix(".log"))
            with _log_to_file(stage_output_path(stage_key).with_suffix(".log")):
                print("=" * 70)
                resume_note = f" (resuming from {resume_from})" if resume_from is not None else ""
                print(f"STAGE {stage_key}: latent dynamics surrogate (frozen encoder){resume_note}")
                print("=" * 70)
                registry_path = stage_dir(stage_key) / f"registry-stage{stage_key}.csv"
                checkpoint = train_lds(
                    size=size, base_path=base_path, ae_checkpoint_path=stage2_checkpoint,
                    checkpoint_path=stage_output_path(stage_key), device=device,
                    loss_curve_path=(
                        _PYTHON_ROOT.parent / "output" / f"stage{stage_key}"
                        / f"{stage_output_path(stage_key).stem}-loss_curve.png"
                    ),
                    resume_from=resume_from,
                    on_checkpoint_saved=_make_checkpoint_callback(registry_path, signature),
                    **kwargs,
                )
                print(f"\nStage {stage_key} complete: {checkpoint}\n")
                _upsert_registry(registry_path, checkpoint, signature)

                if run_sanity_check:
                    print("=" * 70)
                    print(f"Sanity check: rollout quality (stage {stage_key} checkpoint)")
                    print("=" * 70)
                    check_rollout(
                        lds_checkpoint_path=checkpoint, device=device,
                        output_path=_PYTHON_ROOT.parent / "output" / f"stage{stage_key}/{checkpoint.stem}-rollout.png",
                    )
                    print()

                    print("=" * 70)
                    print("Sanity check: parameter dependence (dt, temperature, noise) "
                          f"(stage {stage_key} checkpoint)")
                    print("=" * 70)
                    check_parameter_dependence(
                        lds_checkpoint_path=checkpoint, device=device,
                        output_path=_PYTHON_ROOT.parent / "output" / f"stage{stage_key}/{checkpoint.stem}-parameter_dependence.png",
                    )
                    print()
        return checkpoint, freshly_trained

    if has_3a:
        stage3a_checkpoint, fresh_3a = run_lds_stage("3a", run_sanity_check=False)
        stage3_checkpoint, fresh_3b = run_lds_stage("3b", resume_from=stage3a_checkpoint,
                                                     run_sanity_check=False)

        # 3a and 3b sanity-checked TOGETHER with the SAME windows, so the
        # two rollout figures are a direct side-by-side comparison rather
        # than each stage picking its own independent random sample (see
        # check_rollout module docstring: "COMPARING CHECKPOINTS AT
        # DIFFERENT n_rollout_steps"). 3b runs first (random selection)
        # since it needs the longer window_length (n_rollout_steps+1);
        # check_rollout's own truncation then lets 3a's figure reuse the
        # identical (run_dir, step0) windows at its shorter length.
        #
        # Only regenerated if at least one of the two was freshly trained
        # this run -- matches every other stage's convention of not
        # re-running sanity checks against a cache-hit checkpoint. NOTE:
        # unlike the single-stage sanity check above, this isn't wrapped
        # in _log_to_file -- that helper opens in overwrite ("w") mode,
        # and both stage3a_checkpoint's and stage3_checkpoint's own
        # per-stage logs were already written (possibly just now, above)
        # under stage_output_path("3a"/"3b").with_suffix(".log");
        # reusing either path here would silently erase that training
        # log. Prints to console only for now.
        if fresh_3a or fresh_3b:
            print("=" * 70)
            print("Sanity check: rollout quality (stage 3b checkpoint)")
            print("=" * 70)
            _, shared_windows = check_rollout(
                lds_checkpoint_path=stage3_checkpoint, device=device,
                output_path=_PYTHON_ROOT.parent / "output" / f"stage3b/{stage3_checkpoint.stem}-rollout.png",
            )
            print()

            print("=" * 70)
            print("Sanity check: rollout quality (stage 3a checkpoint, same samples as 3b)")
            print("=" * 70)
            check_rollout(
                lds_checkpoint_path=stage3a_checkpoint, device=device,
                fixed_windows=shared_windows,
                output_path=_PYTHON_ROOT.parent / "output" / f"stage3a/{stage3a_checkpoint.stem}-rollout.png",
            )
            print()

            print("=" * 70)
            print("Sanity check: parameter dependence (dt, temperature, noise) "
                  "(stage 3b checkpoint)")
            print("=" * 70)
            check_parameter_dependence(
                lds_checkpoint_path=stage3_checkpoint, device=device,
                output_path=_PYTHON_ROOT.parent / "output" / f"stage3b/{stage3_checkpoint.stem}-parameter_dependence.png",
            )
            print()

            print("=" * 70)
            print("Sanity check: parameter dependence (dt, temperature, noise) "
                  "(stage 3a checkpoint)")
            print("=" * 70)
            check_parameter_dependence(
                lds_checkpoint_path=stage3a_checkpoint, device=device,
                output_path=_PYTHON_ROOT.parent / "output" / f"stage3a/{stage3a_checkpoint.stem}-parameter_dependence.png",
            )
            print()
    else:
        stage3_checkpoint, _ = run_lds_stage(3)

    # ---- Stage 4/5: encoder refinement / end-to-end -- both OPTIONAL,
    # unlike stages 1/2/3. Neither section present at all -> pipeline
    # stops at stage 3, exactly as before these stages existed. Stage 5
    # requires stage 4 to have actually run (it resumes stage 4's own
    # output, not a fresh assembly from stage 2/3 -- see
    # train_refinement()'s docstring), so stage 5 without stage 4 is an
    # error, not silently skipped. ----
    has_stage4, has_stage5 = 4 in stages, 5 in stages
    if has_stage5 and not has_stage4:
        raise ValueError(f"{params_path}: '# Stage 5' requires '# Stage 4' to also be present "
                          f"(stage 5 continues stage 4's own output, not a fresh assembly).")

    def run_refinement_stage(stage_key: int, resume_from: Path | None = None) -> Path:
        """Runs one train_refinement() phase -- stage_key is 4 or 5.
        freeze_decoder is DERIVED from stage_key (4 -> True, 5 -> False),
        not a params-file-settable option -- matching how 3a/3b's
        n_rollout_steps is a regular param, but WHICH curriculum phase
        this is comes from the section name itself, not a value inside
        it."""
        freeze_decoder = (stage_key == 4)
        kwargs = _prepare_stage_kwargs(stages.get(stage_key, {}), global_params)
        force = kwargs.pop("force", False)
        kwargs = _strip_unrecognized_params(train_refinement, kwargs, f"Stage {stage_key}",
                                            own_keys=renamed_keys(stages.get(stage_key, {})))
        kwargs, resume_from, overridden = _resolve_stage_specific_ancestor(
            kwargs, resume_from, f"Stage {stage_key}")
        if resume_from is not None:
            signature = {"base_path": str(base_path), "resumed_from": str(resume_from),
                          **extra_signature, **_signature_kwargs(kwargs)}
        else:
            # Both ancestors recorded explicitly, same rationale as
            # run_lds_stage: full ancestry visible from this one
            # registry, without following stage2_checkpoint into ITS
            # own registry to find stage1_checkpoint, etc.
            signature = {"base_path": str(base_path),
                          "stage2_checkpoint": str(stage2_checkpoint),
                          "stage3_checkpoint": str(stage3_checkpoint),
                          **extra_signature, **_signature_kwargs(kwargs)}
        checkpoint = resolve_checkpoint(stage_key, force, signature, kwargs.get("epochs"))
        if checkpoint is None:
            if overridden:
                # Only for an EXPLICIT, same-stage override -- not the
                # normal stage-5-resumes-from-stage-4 case (resume_from
                # is not None there too, but that's always stage 4's
                # own, different checkpoint, not stage 5's own output).
                _backup_before_overwrite(stage_output_path(stage_key))
                _backup_before_overwrite(stage_output_path(stage_key).with_suffix(".log"))
            with _log_to_file(stage_output_path(stage_key).with_suffix(".log")):
                print("=" * 70)
                label = "encoder refinement" if freeze_decoder else "end-to-end refinement"
                resume_note = f" (resuming from {resume_from})" if resume_from is not None else ""
                print(f"STAGE {stage_key}: {label}{resume_note}")
                print("=" * 70)
                registry_path = stage_dir(stage_key) / f"registry-stage{stage_key}.csv"
                common_args = dict(
                    base_path=base_path, freeze_decoder=freeze_decoder,
                    checkpoint_path=stage_output_path(stage_key), device=device,
                    loss_curve_path=(
                        _PYTHON_ROOT.parent / "output" / f"stage{stage_key}"
                        / f"{stage_output_path(stage_key).stem}-loss_curve.png"
                    ),
                    on_checkpoint_saved=_make_checkpoint_callback(registry_path, signature),
                    **kwargs,
                )
                if resume_from is not None:
                    checkpoint = train_refinement(resume_from=resume_from, **common_args)
                else:
                    checkpoint = train_refinement(
                        ae_checkpoint_path=stage2_checkpoint, lds_checkpoint_path=stage3_checkpoint,
                        **common_args,
                    )
                print(f"\nStage {stage_key} complete: {checkpoint}\n")
                _upsert_registry(registry_path, checkpoint, signature)

                ae_view_path, lds_view_path = split_joint_checkpoint_for_evaluation(
                    checkpoint, _STAGE_DIRS[stage_key] / "eval_views",
                )

                print("=" * 70)
                print(f"Sanity check: reconstruction quality (stage {stage_key} checkpoint)")
                print("=" * 70)
                check_reconstruction(
                    checkpoint_path=ae_view_path, device=device,
                    min_step=kwargs.get("min_step", 0),
                    min_stdev_phi=kwargs.get("min_stdev_phi"),
                    output_path=_PYTHON_ROOT.parent / "output" / f"stage{stage_key}/{checkpoint.stem}-reconstruction.png",
                )
                print()

                print("=" * 70)
                print(f"Sanity check: latent channel activations (stage {stage_key} checkpoint)")
                print("=" * 70)
                check_latent_channels(
                    ae_checkpoint_path=ae_view_path, device=device,
                    min_step=kwargs.get("min_step", 0),
                    min_stdev_phi=kwargs.get("min_stdev_phi"),
                    output_path=_PYTHON_ROOT.parent / "output" / f"stage{stage_key}/{checkpoint.stem}-latent_channels.png",
                )
                print()

                print("=" * 70)
                print(f"Sanity check: rollout quality (stage {stage_key} checkpoint)")
                print("=" * 70)
                check_rollout(
                    lds_checkpoint_path=lds_view_path, device=device,
                    output_path=_PYTHON_ROOT.parent / "output" / f"stage{stage_key}/{checkpoint.stem}-rollout.png",
                )
                print()

                # E is trainable here with stats_head frozen -- structurally
                # the same anchor pattern as stage 2's own stats0_weight
                # anchor (see this module's docstring), and the same
                # failure mode as D's checkerboard is possible in
                # principle: E could drift into a region stats_head can no
                # longer correctly interpret, without the combined loss
                # alone revealing it. These are the diagnostics built
                # specifically to check that, previously only run after
                # stage 2 -- skipped gracefully (not an error) if the
                # ancestor AE has no stats_head at all (stats_weight<=0
                # back in stage 1), matching train_refinement()'s own
                # graceful handling of that same condition.
                try:
                    print("=" * 70)
                    print(f"Sanity check: interpolation consistency (stage {stage_key} checkpoint)")
                    print("=" * 70)
                    check_interpolation(
                        checkpoint_path=ae_view_path, device=device,
                        output_path=_PYTHON_ROOT.parent / "output" / f"stage{stage_key}/{checkpoint.stem}-interpolation.png",
                    )
                    print()

                    print("=" * 70)
                    print(f"Sanity check: perturbation response (stage {stage_key} checkpoint)")
                    print("=" * 70)
                    check_perturbation(
                        checkpoint_path=ae_view_path, device=device,
                        output_path=_PYTHON_ROOT.parent / "output" / f"stage{stage_key}/{checkpoint.stem}-perturbation.png",
                    )
                    print()
                except ValueError as e:
                    if "no stats_head" not in str(e):
                        raise
                    print(f"Skipping interpolation/perturbation sanity checks: {e}\n")
        return checkpoint

    if not has_stage4:
        return stage3_checkpoint

    stage4_checkpoint = run_refinement_stage(4)
    if not has_stage5:
        return stage4_checkpoint

    stage5_checkpoint = run_refinement_stage(5, resume_from=stage4_checkpoint)
    return stage5_checkpoint

