# R2-Dreamer

Upstream: <https://github.com/NM512/r2dreamer> @ `1fadce46c40dc59cf5a7407e4a296a3c6e4d1416`
(MIT). Default model (`size12M`, `rep_loss=r2dreamer`) and the official `dmc_vision`
config; only the environment side is changed. Build with
`bash baselines/setup_baselines.sh r2dreamer` (see `../README.md`).

## What `r2dreamer.patch` changes

| File | Change |
|---|---|
| `configs/env/dmc_vision.yaml` | new keys (all off by default): `video_hard`, `video_random_frames`, `recolor_agent`, `recolor_rgba`, `pixel_video_bg`, `distracting_background_path` |
| `configs/env/noisyshape.yaml` | new: NoisyShape config (2e5 steps, 1 env, action repeat 1, 200-step episodes) |
| `envs/__init__.py` | pass the new keys to `DeepMindControl`; new suite `noisyshape` built from the `shapeenv` package |
| `envs/dmc.py` | `_small` task suffix (wider side-camera FOV, physics unchanged); `VideoHardWrapper` (VH / VHR, optional low-contrast recolouring); `pixel_video_bg`: DAVIS frames composited into all background pixels (segmentation render), for top-down Reacher |
| `envs/wrappers.py` | `TimeLimit` always sets `is_last` (False until the limit; NoisyShape does not emit it); `FiveToFourTupleStep` adapts `shapeenv`'s 5-tuple `step` |
| `buffer.py`, `configs/configs.yaml` | opt-in `buffer.clone_shifted_action` (default `False`, see below) |

`setup_baselines.sh` adds `envs/video_hard_wrapper.py`, `envs/dmcgb/` and
`envs/small_variant.py` from StarWM's `envs/`, and `run_probes.py` (this directory's
`probes/`).

## Experiments

| Script | Setting | Overrides (besides `env=dmc_vision env.task=dmc_<task> seed logdir`) |
|---|---|---|
| `scripts/dmc.sh vh` | Main DMC, VH | `env.steps=3.01e6 env.video_hard=True env.video_random_frames=False` |
| `scripts/dmc.sh vhr` | Main DMC, VHR | `env.steps=1.01e6 env.video_hard=True env.video_random_frames=True` |
| `scripts/small_objects.sh clean\|vh\|vhr` | `cheetah_run_small`, `walker_walk_small` | `env.steps=1.01e6` / `3.01e6 env.video_hard=True` / `1.01e6 env.video_hard=True env.video_random_frames=True` |
| `scripts/low_contrast.sh vh\|vhr` | Walker Walk | VH / VHR flags as above + `env.recolor_agent=True` |
| `scripts/reacher.sh vh\|vhr` | Reacher Easy | `env.pixel_video_bg=True env.video_random_frames=False\|True env.distracting_background_path=$DAVIS_DIR`, 3.01e6 / 1.01e6 steps |
| `scripts/noisyshape.sh` | NoisyShape | `env=noisyshape buffer.clone_shifted_action=True env.distracting_background_path=$DAVIS_DIR`, then the probes |

Budgets are 3M (VH) / 1M (VHR, Clean) env steps; the extra `0.01e6` follows upstream's
default `steps: 1.01e6`. Ball-in-cup Catch is `dmc_ball_in_cup_catch` here (`dmc.sh` maps
`cup_catch` to it).

**NoisyShape probes.** `scripts/noisyshape.sh probe` runs
`run_probes.py --logdir $RUNS_DIR/baselines/r2dreamer/noisyshape --run_prefix noisyshape`:
300 current episodes (seed 42) and 150 imagination episodes (seed 12345), cached in
`traj_cache_noisyshape_300.pkl`; MLP probes on `[stoch; deter]` for colour / shape / size /
position at H0 and through 15 imagined steps. Results:
`noisyshape_s<seed>/probe_results/results.json`. Pass `--force` to recompute.

## Notes

- No auto-resume: a restarted `train.py` starts from scratch.
- `train.py` writes `latest.pt` only at the end of training and then usually exits with a
  non-zero status during interpreter shutdown (environment-worker teardown, upstream
  behaviour). The launchers therefore count a run as successful iff it wrote `latest.pt`.
- `torch.compile` is on by default and needs `nvcc`; otherwise set `TORCHDYNAMO_DISABLE=1`
  or `EXTRA_ARGS=model.compile=False`.
- Upstream `envs/parallel.py` pins host memory, so training needs a CUDA device.
- `buffer.clone_shifted_action`: `Buffer.sample` writes `action[:, :-1]` into the
  overlapping view `action[:, 1:]`; without a clone this in-place copy was observed to
  corrupt a small fraction of actions on GPU. Off for all DMC runs (upstream behaviour),
  on for the NoisyShape runs.
