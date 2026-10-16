# DenoisedMDP

Upstream: <https://github.com/facebookresearch/denoised_mdp> @
`bc68709750e5a3ea375077f0da6d569bc2217665`, licensed **CC BY-NC 4.0**. Only the patch is
distributed here; `bash baselines/setup_baselines.sh denoised_mdp` fetches the official code
and applies it (see `../README.md`). Default DenoisedMDP configuration; only the
environment side and test-time logging are changed.

## What `denoised_mdp.patch` changes

| File | Change |
|---|---|
| `denoised_mdp/envs/dmc/__init__.py` | new env-spec variants `<domain>_<task>_vh` / `_vhr` (domain/task split at the last underscore, e.g. `ball_in_cup_catch_vh`) built with `SimpleDMCEnv` |
| `denoised_mdp/envs/dmc/simple_dmc.py` | new: `SimpleDMCEnv`, a minimal `EnvBase` implementation on the standard `dm_control` suite (64x64, camera 0, action repeat and episode length from the config, actions normalised to [-1, 1]) with `VideoHardWrapper` applied at the dm_control level, as in StarWM / R2-Dreamer. It replaces the local `dmc2gym` fork for these variants, which was slow in combination with the wrapper |
| `config.py`, `main.py` | `learning.test.enable_visualize` (default `True`); `False` skips test episodes and all video dumps |

`setup_baselines.sh` installs `video_hard_wrapper.py` and `dmcgb/` from StarWM's `envs/` as
the top-level package `envs/` of the checkout (next to `main.py`), which
`simple_dmc.py` imports. The `robodesk` submodule is not cloned (not needed for DMC).

## Experiments

| Script | Setting | Overrides |
|---|---|---|
| `scripts/dmc.sh vh` | Main DMC, VH | `env.kind=dmc env.spec=<task>_vh learning.exploration.total_steps=3000000` |
| `scripts/dmc.sh vhr` | Main DMC, VHR | `env.kind=dmc env.spec=<task>_vhr learning.exploration.total_steps=1000000` |

Both also pass `learning.test.num_episodes=1 learning.test.enable_visualize=False
seed=<seed> output_base_dir=<run dir>`. With `enable_visualize=False` no test episodes are
run, so the paper reports training-episode returns (TensorBoard scalar `train_reward`).
Ball-in-cup Catch is `ball_in_cup_catch` here (`dmc.sh` maps `cup_catch` to it).

## Notes

- `config.py` locates the repository with `git rev-parse`, so run from the git checkout
  created by `setup_baselines.sh`.
- The run is written to `<run dir>/dmc_<spec>/<auto-named folder>/` (TensorBoard events,
  `output.log`, checkpoints). A restarted run resumes from its last checkpoint; a finished
  run (`COMPLETE` file) is skipped.
- DenoisedMDP is not evaluated on NoisyShape: its decomposition requires a reward signal.
