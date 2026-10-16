# Baselines

The paper compares StarWM against **R2-Dreamer**, **DenoisedMDP** and **Iso-Dream++**
using their official implementations at a pinned commit, with minimal modifications:
the shared Video-Hard distractor wrapper, NoisyShape integration, the small-object and
low-contrast DMC variants and the Reacher pixel background. The agents, losses,
optimisers and hyper-parameters are untouched (DMC runs use each codebase's own
`dmc_vision` / `dmc` / default config).

This directory contains, per baseline, a patch against the pinned upstream commit,
the launch scripts with the exact overrides of the paper's runs, and (where used)
the NoisyShape probe script. No upstream code is vendored.

| Baseline | Upstream | Commit | Licence | Shipped here |
|---|---|---|---|---|
| R2-Dreamer | [NM512/r2dreamer](https://github.com/NM512/r2dreamer) | `1fadce46c40dc59cf5a7407e4a296a3c6e4d1416` | MIT | `r2dreamer/r2dreamer.patch`, scripts, probe |
| Iso-Dream++ | [panmt/MBRL_with_Isolated_Imaginations](https://github.com/panmt/MBRL_with_Isolated_Imaginations) | `3f60cc6bbbf4a8da4f2b48b233868d15064998b5` | MIT | `isodream/isodream.patch`, scripts, probe |
| DenoisedMDP | [facebookresearch/denoised_mdp](https://github.com/facebookresearch/denoised_mdp) | `bc68709750e5a3ea375077f0da6d569bc2217665` | CC BY-NC 4.0 | `denoised_mdp/denoised_mdp.patch`, scripts |

```
baselines/
  setup_baselines.sh   clone + checkout pinned commit + apply patch + install shared files
  common/              launch_lib.sh (sourced by all launchers), smoke_test.py
  r2dreamer/           r2dreamer.patch, scripts/, probes/run_probes.py
  isodream/            isodream.patch, scripts/, probes/run_probes_isodream.py
  denoised_mdp/        denoised_mdp.patch, scripts/
```

## Setup

1. **Python environment.** All baselines were run in the same environment as StarWM
   (`../requirements.txt`), with in addition (versions used):
   ```
   hydra-core==1.3.2 omegaconf==2.2.1          # R2-Dreamer, DenoisedMDP
   torchrl==0.9.2 tensordict==0.9.1 einops==0.8.2 moviepy==1.0.3   # R2-Dreamer
   attrs==25.4.0 tensorboardX==2.6.5 gym==0.26.2 scikit-video imageio   # DenoisedMDP
   scikit-learn tqdm                            # probes
   ```
   (DenoisedMDP's own `requirements.txt` pins `hydra-core==1.2.0`; 1.3.2 with
   `omegaconf==2.2.1` works for both.) NoisyShape runs and probes need the `shapeenv`
   package (`pip install git+https://github.com/zeqiang-zhang/ShapeEnv.git`).
2. **Data.**
   - `DMCGB_DATA_DIR`: the DMC-GB2 data directory containing `video_hard/video{0..99}.mp4`
     (VH / VHR distractors), as for StarWM.
   - `DAVIS_DIR`: DAVIS 2017 `JPEGImages/480p` (NoisyShape, Reacher).
3. **Build the patched checkouts.**
   ```bash
   bash baselines/setup_baselines.sh                 # all three -> baselines/upstream/
   bash baselines/setup_baselines.sh --target DIR r2dreamer   # one baseline, elsewhere
   export BASELINES_SRC=DIR                          # only if --target was used
   ```
   For each baseline the script clones the upstream repository, checks out the pinned
   commit, runs `git apply --check` and `git apply` with the patch, copies the files shared
   with StarWM from `$STARWM_ROOT/envs` (default: this repository) into the checkout's
   `envs` package, and copies the probe script:

   | Source | R2-Dreamer | Iso-Dream++ | DenoisedMDP |
   |---|---|---|---|
   | `envs/video_hard_wrapper.py`, `envs/dmcgb/` (DMC-GB2) | `envs/` | `iso_rl/envs/` | `envs/` (top level) |
   | `envs/small_variant.py` | `envs/` | – | – |
   | `baselines/<method>/probes/*.py` | `run_probes.py` | `iso_rl/run_probes_isodream.py` | – |

   These are the same files StarWM uses, so every method sees identical distractors.
4. **Check** (builds envs through each baseline's own factory and steps them on CPU):
   ```bash
   python baselines/common/smoke_test.py r2dreamer --dmc --noisyshape
   python baselines/common/smoke_test.py isodream --dmc --noisyshape
   python baselines/common/smoke_test.py denoised_mdp --dmc
   ```

## Running

All launchers are plain bash and loop over seeds. Run them from the directory that should
hold `runs/`; output goes to `$RUNS_DIR/baselines/<method>/<setting>/<task>_s<seed>`
(log: `<run dir>.log`).

| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | `0 1 2` | seeds |
| `PARALLEL` | `0` | `1` starts all seeds of a task at once (the paper packed 3 seeds on one GPU) |
| `RUNS_DIR` | `./runs` | output root |
| `BASELINES_SRC` | `baselines/upstream` | patched checkouts |
| `PYTHON` | `python` | interpreter |
| `EXTRA_ARGS` | – | appended to every training command (e.g. `model.compile=False`) |
| `CUDA_VISIBLE_DEVICES` | – | GPU selection (all commands use the first visible GPU) |

| Paper result | Command | Steps |
|---|---|---|
| Main DMC, VH (Fig. 5, Table 5; all three baselines) | `<method>/scripts/dmc.sh vh [TASK ...]` | 3M (R2: `env.steps=3.01e6`) |
| Main DMC, VHR (Fig. 5, Table 5; all three baselines) | `<method>/scripts/dmc.sh vhr [TASK ...]` | 1M (R2: `1.01e6`) |
| NoisyShape probes (Fig. 3): R2-Dreamer, Iso-Dream++ | `r2dreamer/scripts/noisyshape.sh`, `isodream/scripts/noisyshape.sh` | 2e5 |
| Small objects, Clean / VH / VHR (Fig. 20): R2-Dreamer | `r2dreamer/scripts/small_objects.sh {clean,vh,vhr}` | 1M / 3M / 1M |
| Low-contrast Walker Walk (Table 14, Fig. 25): R2-Dreamer | `r2dreamer/scripts/low_contrast.sh {vh,vhr}` | 3M / 1M |
| Reacher Easy, DAVIS background (Table 17, Fig. 29): R2-Dreamer | `r2dreamer/scripts/reacher.sh {vh,vhr}` | 3M / 1M |

`<method>` is `r2dreamer`, `isodream` or `denoised_mdp`; the DMC tasks are `walker_walk
walker_run cheetah_run cartpole_swingup finger_spin cup_catch` (Ball-in-cup Catch; mapped to
each codebase's own task name). Example:

```bash
export DMCGB_DATA_DIR=/data/dmcgb2 DAVIS_DIR=/data/DAVIS/JPEGImages/480p
PARALLEL=1 CUDA_VISIBLE_DEVICES=0 bash baselines/r2dreamer/scripts/dmc.sh vhr walker_walk
```

**Reported metric.** The DMC curves and tables use training-episode returns: R2-Dreamer
`episode/score` (`metrics.jsonl`), Iso-Dream++ `train_return` (`metrics.jsonl`), DenoisedMDP
the TensorBoard scalar `train_reward` (its test episodes are disabled, see below).

## Notes

- **Resuming.** R2-Dreamer's `train.py` does **not** resume: restarting a run starts from
  scratch in the same logdir (use a fresh logdir). Iso-Dream++ resumes from
  `latest_model.pt` and the stored episodes; DenoisedMDP resumes from its last checkpoint.
- **R2-Dreamer `torch.compile`.** On by default (`model.compile=True`) and needs a CUDA
  toolchain (`nvcc`). Without one, run eager with `TORCHDYNAMO_DISABLE=1` or
  `EXTRA_ARGS=model.compile=False`; some of the paper's appendix runs were run eager.
- **R2-Dreamer replay buffer.** Upstream shifts actions by one step with an in-place copy
  between overlapping views of the sampled batch; on GPU this can corrupt a small fraction of
  action vectors. The patch adds `buffer.clone_shifted_action` (default `False` = upstream
  behaviour, used for all DMC runs); the NoisyShape launcher sets it to `True`, as for the
  NoisyShape runs in Fig. 3.
- **DenoisedMDP evaluation.** `learning.test.enable_visualize=False` (added by the patch)
  skips test episodes and video dumps entirely; the paper's runs used it and report
  training returns. DenoisedMDP needs `output_base_dir` to exist (the launcher creates it)
  and nests its run under `dmc_<spec>/<auto-named folder>/`.
- **Rendering / versions.** Rendering goes through MuJoCo (`MUJOCO_GL=egl` by default).
  Results can shift slightly across MuJoCo / dm_control versions; compare methods only
  within one software environment.
- DenoisedMDP is not run on NoisyShape: its decomposition requires a reward signal, and
  NoisyShape's reward is pure noise (see the paper).

## Licences

- **R2-Dreamer** (MIT, © 2026 Naoki Morihira) and **Iso-Dream++** (MIT, © 2023 Pan
  Minting): the patches and probe scripts are derivative works distributed under MIT
  together with this repository; the upstream licence applies to the code they modify.
- **DenoisedMDP** is licensed **CC BY-NC 4.0** (non-commercial). We do not redistribute
  it: `setup_baselines.sh` fetches it from the official repository and the patch only
  contains our changes (plus the few context lines `git apply` needs). The patched
  checkout remains subject to CC BY-NC 4.0.
- The shared `dmcgb/dmcgb_photometric.py` is adapted from DMC-GB2 (MIT); see `../LICENSE`.
  DMC-GB2 videos and DAVIS 2017 are third-party data under their own terms.
