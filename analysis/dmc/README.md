# DMC analysis: figures and tables of the paper

Scripts that turn DMC training runs into the paper's DMC figures and tables.
Run every command from the `StarWM/` directory. Figures go to `figures/` by
default (`--out` / `--out_dir` to change it).

| Paper item | Script |
|---|---|
| Fig 5 (DMC learning curves), Table 5 (DMC returns) | `plot_main_results.py [--table]` |
| Fig 10 a/b/c (rliable aggregate metrics, performance profiles, P(A>B)) | `rliable_analysis.py`, then `plot_rliable.py` |
| Fig 1 right (IQM, VH vs VHR) | `rliable_analysis.py`, then `plot_iqm_scatter.py` |
| Fig 7, Fig 13, Table 7 (gradient isolation) | `plot_ablations.py [--table]` |
| Fig 6, Fig 11, Fig 14, Table 8 (attention maps and statistics) | `viz_attention.py --figure {cheetah,cartpole,isolation}` |
| Fig 16 (inverse-only vs full SS) | `plot_contrastive.py` |
| Fig 17, Table 10 (number of queries N) | `plot_queries.py` |
| Fig 20 (small agent) | `plot_small_objects.py` |
| Tables 9, 14, 17 (sensitivity, low-contrast, Reacher) | `stress_tables.py` |
| Fig 25 (low-contrast Walker curves) | `plot_lowcontrast.py` |
| Fig 29 (Reacher) | `render_reacher_frames.py`, then `plot_reacher.py` |
| Fig 2, left (example observations) | `render_env_frames.py` |

`common.py` holds the loaders and aggregation rules (documented in its
docstring), `layouts.py` maps experiments to run directories.

## 1. Training the runs

All runs use `train.py --configs dmc_vision` with seeds 0, 1, 2. Settings:

| Setting | Flags | `--steps` |
|---|---|---|
| Clean | (none) | `1e6` |
| VH (sequential video) | `--use_video_hard True` | `3e6` |
| VHR (random video frame each step) | `--use_video_hard True --video_random_frames True` | `1e6` |

VH/VHR need the DMC-GB2 `video_hard` clips (`$DMCGB_DATA_DIR/video_hard/video{0..99}.mp4`);
Reacher needs DAVIS 2017 (`--distracting_background_path /path/to/DAVIS/JPEGImages/480p`).

Run directories (the default `--layout release`, `--root runs`):

```
runs/dmc/<setting>/<task>/<variant>_s<seed>           setting: clean | vh | vhr
```

`<task>` is one of `cup_catch` (Ball-in-cup Catch), `cartpole_swingup`,
`cheetah_run`, `finger_spin`, `walker_run`, `walker_walk`; the training flag is
`--task dmc_<task>`. A full command, e.g. StarWM on Walker Walk VHR, seed 0:

```bash
python train.py --configs dmc_vision --task dmc_walker_walk \
    --use_video_hard True --video_random_frames True --steps 1e6 --seed 0 \
    --logdir runs/dmc/vhr/walker_walk/starwm_s0
```

| `<variant>` | Extra flags | Used by |
|---|---|---|
| `starwm` | (none) | everything |
| `dreamerv3` | `--configs dmc_vision dreamerv3` | Fig 5, Table 5, Fig 1/10 (Clean runs = normalisation) |
| `starwm_r` | `--attn_reward_scale R`, R = 100 (cup_catch), 50 (cartpole_swingup, cheetah_run), 30 (finger_spin, walker_run, walker_walk) | Fig 5, Table 5, Fig 1/10 (VH, VHR), Fig 11 (cartpole VH) |
| `inv_only` | `--configs dmc_vision inv_only` | Fig 16 (cartpole_swingup, cheetah_run, walker_run; all settings) |
| `iso_star1` | `--configs dmc_vision iso_star1` | Fig 7/13/14, Tables 7/8 (walker_walk, cheetah_run; clean, vhr) |
| `iso_star3` | `--configs dmc_vision iso_star3` | same |
| `N1`, `N2`, `N8` | `--crossattn_num_queries 1/2/8` | Fig 17, Table 10 (vh; walker_walk, cheetah_run, cartpole_swingup) |

Fig 5 / Table 5 / Fig 1 / Fig 10 use all six tasks (StarWM, StarWM-R, DreamerV3
under VH and VHR; StarWM and DreamerV3 under Clean).

Other experiments:

| Experiment | Flags | Run directory |
|---|---|---|
| Small agent (Fig 20) | `--task dmc_<task>_small` (cheetah_run, walker_walk; clean/vh/vhr; StarWM and `dreamerv3`) | `runs/dmc_small/<setting>/<task>_small/{starwm,dreamerv3}_s<seed>` |
| Low-contrast agent (Table 14, Fig 25) | `--task dmc_walker_walk --use_video_hard True [--video_random_frames True] --recolor_agent True`, VH `3e6` / VHR `1e6` | `runs/dmc_lowcontrast/{vh,vhr}/walker_walk/{starwm,dreamerv3}_s<seed>` |
| Reacher (Table 17, Fig 29) | `--task dmc_reacher_easy --pixel_video_bg True [--video_random_frames True] --distracting_background_path <DAVIS>`, VH `3e6` / VHR `1e6`; StarWM-R: `--attn_reward_scale 30` | `runs/dmc_reacher/{vh,vhr}/{starwm,starwm_r,dreamerv3}_s<seed>` |
| Sensitivity (Table 9) | Walker Walk VHR, `1e6`, plus one of `--attn_inv_scale 0.3`, `--attn_inv_scale 3.0`, `--Annealing False`, `--res_kl_scale 2.0` | `runs/sensitivity/{linv0.3,linv3.0,bg_noanneal,bg_final2.0}_s<seed>` |

Baselines come from their own code bases. Put (or link) their run directories here:

```
runs/baselines/r2dreamer/{vh,vhr}/<task>_s<seed>              metrics.jsonl: episode/score, episode/eval_score
runs/baselines/r2dreamer/{clean,vh,vhr}/<task>_small_s<seed>  (Fig 20)
runs/baselines/r2dreamer/lowcontrast_{vh,vhr}/walker_walk_s<seed>
runs/baselines/r2dreamer/{vh,vhr}/reacher_easy_s<seed>
runs/baselines/isodream/{vh,vhr}/<task>_s<seed>               metrics.jsonl: train_return
runs/baselines/denoised_mdp/{vh,vhr}/<task>_s<seed>           TensorBoard scalar train_reward (may be nested)
```

Only directories ending in `_s<digits>` are read, so archived copies such as
`starwm_s0_old` are ignored. If a run is resumed into its directory,
`metrics.jsonl` is appended to; the loaders print a warning when the logged step
goes backwards by more than 50K (a run restarted from scratch). To check a tree:
`python analysis/dmc/common.py --scan runs/`.

## 2. Producing the figures and tables

```bash
python analysis/dmc/plot_main_results.py --table          # Fig 5 + Table 5
python analysis/dmc/rliable_analysis.py                   # ~15 min, writes figures/rliable/
python analysis/dmc/plot_rliable.py                       # Fig 10 a/b/c
python analysis/dmc/plot_iqm_scatter.py                   # Fig 1 (right)
python analysis/dmc/plot_ablations.py --table             # Fig 7, Fig 13, Table 7
python analysis/dmc/plot_contrastive.py                   # Fig 16
python analysis/dmc/plot_queries.py                       # Fig 17 + Table 10
python analysis/dmc/plot_small_objects.py                 # Fig 20
python analysis/dmc/stress_tables.py                      # Tables 9, 14, 17
python analysis/dmc/plot_lowcontrast.py                   # Fig 25
python analysis/dmc/render_reacher_frames.py --davis_path <DAVIS> --out figures/reacher_frames.npz
python analysis/dmc/plot_reacher.py --frames figures/reacher_frames.npz   # Fig 29
python analysis/dmc/viz_attention.py --figure cheetah     # Fig 6   (loads checkpoints; GPU optional)
python analysis/dmc/viz_attention.py --figure cartpole    # Fig 11
python analysis/dmc/viz_attention.py --figure isolation   # Fig 14 + Table 8
python analysis/dmc/render_env_frames.py                  # Fig 2 (DMC part)
```

`viz_attention.py` and the two render scripts need MuJoCo rendering
(`MUJOCO_GL=egl`) and the background assets above, because the world model is
rebuilt from the run's environment.

Common options: `--seeds 0 1 2` restricts every cell to these seeds;
`--layout paper --root <dir>` reads the directory layout of the original
research code (`<dir>` contains `icr_dreamerv3/`, `r2dreamer/`, `denoised_mdp/`,
`MBRL_with_Isolated_Imaginations/`; the stress-test runs are found through
`--stress_subdir <their log directory>`), which is how these scripts were
checked against the paper.

## 3. Conventions (what the numbers mean)

* **Learning curves, Tables 5, 7, 10** (`common.interp_to_grid`, `final_window`):
  per seed, a 25-point trailing moving average (5 points for Fig 7/13 / Table 7)
  of the logged return, interpolated onto a 4,000-step grid; mean and
  population std (ddof 0) over seeds. A table cell is the mean of the last 10
  grid points (~40K steps) of the mean curve ± the mean of the std curve over
  the same points. For VH/VHR (Table 5) and Table 7 the window ends at the last
  step where any seed has data; for the Clean rows of Table 5, Table 10 and the
  Fig 20 printout it ends at the last step where all seeds have data. Fig 5 /
  Table 5 / Figs 16, 17, 20 use the *training*-episode return; Figs 7/13 and
  Table 7 the evaluation return. The ± in these tables is the std over seeds,
  not the standard error.
* **rliable (Fig 1, Fig 10)**: per-seed score = mean training return over the
  last 50K steps; normalised per task by the DreamerV3 Clean score (same
  statistic, mean over seeds); stratified bootstrap with 10,000 (aggregates) and
  2,000 (P(A>B)) replications. rliable draws its bootstrap indices from NumPy's
  global RNG, so CIs differ by ~0.003 between runs; `--np_seed` fixes them.
* **Stress tests (Tables 9, 14, 17; Figs 25, 29)**: evaluation return, EMA with
  weight 0.3 per seed. Tables: value at the step matched across the compared
  methods (minimum over methods of the shortest seed's last step), mean ± SE.
  Curves: mean ± SE on a 200-point grid over the steps all seeds cover.
* **Attention statistics (Table 8)**: query Q0 of the first seed with a checkpoint,
  on the most recent stored episode; entropy (uniform = log 64 = 4.16) and mean
  per-frame maximum (uniform = 1/64).
