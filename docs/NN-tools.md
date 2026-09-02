# Diagnostic tools for the LIENS neural network

## Tests
`pytest -n 4 tests/ -q`. Variant: `pytest -n 4 tests/ -q -m "not slow"` to skip the longer tests.

Note: Checkpoint binaries required for the test suite may not be in the distribution.

`python tests/_import_graph.py` maps the project's internal imports, as two dictionaries `dict_key_imports_values` and `dict_key_imported_by_values`, saved in `tests/data/`.


## datasets (not models)
Everything else is about models at various stages.

### stdev_phi_time
`python -m evaluation.check_stdev_phi_time --base-path ../datasets --size 128 --min-step 1500 --min-run-fraction 0.5`


## compare_f_theta

### compare_panels — per-window microstructure figures
- `python -m evaluation.compare_f_theta checkpoints/stage3a/128x128-stage3a.pt checkpoints/stage3b/128x128-stage3b.pt --panels-only --n-samples 6 --steps 12 --seed 0` produce an 8-column plot (state | real dx | stage-2 dx | pred A | pred B | error A | error B | B−A), with one row per window.
- add `--trajectory` for one frame-by-frame figure per window.
- add `--t0-range LO HI` to select allowed window start-step interval


### compare_statistics — loss/correlation over many windows
- `python -m evaluation.compare_f_theta checkpoints/stage3a/128x128-stage3a.pt checkpoints/stage3b/128x128-stage3b.pt --stats-only --n-stats 200 --steps 10 --seed 0`
- add `--f-scale-sweep` to try f_theta × λ, with λ in [0, 0.25, 0.5, 1] ("λ-sweep"); 0 corresponds to stage 2 and 1 is stage 3.
- add `--alpha-sweep` for the h→0 test.

The combined `compare_f_theta` (no `--*-only` flag) still runs both: `python -m evaluation.compare_f_theta checkpoints/stage3a/128x128-stage3a.pt checkpoints/stage3b/128x128-stage3b.pt --n-stats 200 --n-samples 6 --steps 10 --seed 0 --trajectory`

### Stage 2 only 
`python -m evaluation.compare_f_theta checkpoints/stage2/128x128-stage2-20260812_20h08.pt checkpoints/stage2/128x128-stage2-20260818_13h54.pt checkpoints/stage2/128x128-stage2-20260819_11h20.pt --stage2-compare --n-stats 200 --steps 10`



## Returning microstructures

Note: `python -m evaluation.compare_f_theta --trajectory` (above) also returns microstructures.

### reconstruction (returns µstructures for AE verification)
`python -m evaluation.check_reconstruction --checkpoint checkpoints/stage2/128x128-stage2.pt --size 128 --min-step 1500 --min-stdev-phi 0.01 --device cuda`

### rollout (returns µstructures for 1 stage)
`python -m evaluation.check_rollout --lds-checkpoint checkpoints/stage3a/128x128-stage3a.pt --no-z1-resync  --min-step 1500 --min-stdev-phi 0.01`
	
#### find_windows
`python -m evaluation.find_windows --base ../datasets --size 128   --dt 125 --theta0 -0.28 --min-step 2000 --min-stdev-phi 0.01   --min-passing-steps 12`
Then
`python -m evaluation.check_rollout --lds-checkpoint checkpoints/stage3b/128x128-stage3b.pt --no-z1-resync  --min-step 1500 --min-stdev-phi 0.01 --fixed-windows   ../datasets/128x128/T725_n003_s123:15000:17500:20000 ../datasets/128x128/T725_n003_s131:15000:17500:20000 ../datasets/128x128/T725_n003_s191:15000:17500:20000 ../datasets/128x128/T725_n003_s401:15000:17500:20000 ../datasets/128x128/T725_n003_s599:15000:17500:20000 ../datasets/128x128/T725_n003_s79:15000:17500:20000`



## Other diagnostics for models

## parameter_dependence
Generates several plots: one for $\delta t$ and one for other parameters (temperature, noise amplitude).
`python -m evaluation.check_parameter_dependence --lds-checkpoint checkpoints/stage4/128x128-stage4.pt --base-path ../datasets --min-step 1500 --min-stdev-phi 0.01 --min-passing-steps 12`


### dt_vs_time (returns tables, not figures)
`python -m evaluation.check_dt_vs_time --lds-checkpoint checkpoints/stage3b/128x128-stage3b.pt --min-step 2000 --min-stdev-phi 0.01 --min-passing-steps 12  --max-dt 1e9`


### z1_degeneracy (returns tables, not figures)
`python -m evaluation.check_z1_degeneracy checkpoints/stage2/128x128-stage2.pt --size 128 --n-windows 500`


### z2_measurability (encoder diag, even when run on stage-3 checkpoint)
`python -m evaluation.check_z2_measurability checkpoints/stage3a/128x128-stage3a.pt --n-windows 512`



## Low level

### substep_convergence
`python -m evaluation.check_substep_convergence checkpoints/stage3b/128x128-stage3b.pt --size 128 --n-windows 256  --max-dt 1000 --window-length 3`


### grad_spikes 
`python -m evaluation.check_grad_spikes checkpoints/stage3b/128x128-stage3b.pt --size 128 --n-windows 256`


### memory
`python -m evaluation.check_memory checkpoints/stage3b/128x128-stage3b.pt --batch-size 2048 --size 128 --device cuda --truncate-bptt 64 --fixed-n 256 --calibrate`



## Input parameters

### Parameters used to generate a checkpoint
`python -m evaluation.inspect_checkpoint checkpoints/stage3a/128x128-stage3a.pt`
Variant: `python -m evaluation.inspect_checkpoint <ckpt> --key z0_noise_scale` prints just: 0.15


### Sweep over `min_stdev_phi`, `min_std_deriv` and `min_passing_steps`
`python -m evaluation.sweep_min_stdev_phi --base-path ../datasets --size 128 --max-runs 1000`
Variant: `--normalized` for `min_normalized_stdev_phi`, i.e. as fraction of temperature-dependent ground-state value for `phi`.

`python -m evaluation.sweep_min_std_deriv --base-path ../datasets --size 128 --min-step 1000 --min-passing-steps 12 --max-runs 1000`

`python -m evaluation.sweep_min_passing_steps --base-path ../datasets --size 128 --max-runs 1000`

Optional arguments:
- `--sma 3` to average times (x-axis from t-1 to t+1) if the plot is too choppy (default 1, i.e. no moving average);
- `--output` with path (default: `../output/datasets/128x128-min_XXX_sweep.png`);
- `--min-bin-count 100` ignores steps with fewer than 100 windows (default: 10);
- `--current-value​ 0.01` includes a curve for the value of the parameter currently used in the code, e.g. 0.01 (default: no curve).
