# Iso-Dream++

Upstream: <https://github.com/panmt/MBRL_with_Isolated_Imaginations> @
`3f60cc6bbbf4a8da4f2b48b233868d15064998b5` (MIT). Model-based RL part (`iso_rl/`) with the
official `dmc` config; only the environment side is changed. Build with
`bash baselines/setup_baselines.sh isodream` (see `../README.md`).

## What `isodream.patch` changes

| File | Change |
|---|---|
| `iso_rl/configs.yaml` | `defaults`: `use_video_hard`, `video_random_frames` (off). New `noisyshape` config = the `dmc` config with one-hot actions (`actor_dist: onehot`, `imag_gradient: reinforce`), `action_repeat: 1`, `time_limit: 200`, `steps: 200000`, `precision: 32`, plus `davis_path` (empty -> `$DAVIS_DIR`) |
| `iso_rl/dreamer.py` | `DeepMindControl` gets `seed` and the VH/VHR flags; new suite `noisyshape` (from `shapeenv`); uniform one-hot random actor for prefill with discrete actions; CARLA import moved into the `carla` branch (CARLA not needed otherwise); config loading via the `ruamel.yaml` >= 0.18 API |
| `iso_rl/wrappers.py` | `gymnasium` instead of `gym`; `DeepMindControl(seed=..., video_hard=..., video_random_frames=...)` seeds the task (`task_kwargs={'random': seed}`) and applies `VideoHardWrapper` (VH seed `seed + 2000`); `NoisyShapeAdapter` (5-tuple -> 4-tuple step, image-only observation) |

`setup_baselines.sh` adds `iso_rl/envs/video_hard_wrapper.py` and `iso_rl/envs/dmcgb/` from
StarWM's `envs/`, and `iso_rl/run_probes_isodream.py` (this directory's `probes/`).

## Experiments

| Script | Setting | Command (run in `iso_rl/`) |
|---|---|---|
| `scripts/dmc.sh vh` | Main DMC, VH | `dreamer.py --configs defaults dmc --task dmc_<task> --steps 3000000 --use_video_hard True` |
| `scripts/dmc.sh vhr` | Main DMC, VHR | `... --steps 1000000 --use_video_hard True --video_random_frames True` |
| `scripts/noisyshape.sh` | NoisyShape (Fig. 3) | `dreamer.py --configs defaults noisyshape --steps 200000 --davis_path $DAVIS_DIR`, then the probes |

All commands also pass `--logdir <run dir> --seed <seed> --device cuda:0`. Ball-in-cup
Catch is `dmc_cup_catch` here (the codebase maps the domain `cup` to `ball_in_cup`).

**NoisyShape probes.** `run_probes_isodream.py --logdir <run dir>` loads
`latest_model.pt`, collects 200 current episodes (seed 42) and 100 imagination episodes
(seed 12345) and probes two feature sets: `controllable` = `[stoch; deter]` of the
controllable branch ("Iso-Dream++ (ctrl)" in Fig. 3) and `combined` = controllable +
non-controllable branch ("Iso-Dream++ (full)"). Output:
`<run dir>/probe_results/results.json`.

## Notes

- Iso-Dream++ resumes automatically from `<run dir>/latest_model.pt` and the stored
  episodes. `--steps` counts environment steps (the code divides by `action_repeat`).
- The training environment and the evaluation environment use the same seed
  (`--seed`), as in the paper's runs.
