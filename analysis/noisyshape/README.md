# NoisyShape probes and figures

Scripts behind the NoisyShape-family results of the paper: latent probes
(Appendix C.2), their figures / tables, and the two qualitative figures
(composition swap, two-object attention).

| Paper | Training runs (`runs/<benchmark>/<variant>_s<seed>`) | Probe | Plot |
|---|---|---|---|
| Fig. 3, Table 4 | `noisyshape/{dreamerv3,dualstream,ss_only,dualstream_ss,attention_only,starwm}` | `probe_noisyshape.py` | `plot_noisyshape_probes.py` |
| Fig. 4 | `noisyshape/starwm_s0` | – | `viz_composition_swap.py` |
| Fig. 12 | `noisyshape/{iso_star1,iso_star2,iso_star3,iso_star4,starwm}` | `probe_noisyshape.py` | `plot_detach_ablation.py` |
| Fig. 15 | `noisyshape/{starwm,inv_only,ctr_only}` | `probe_noisyshape.py` | `plot_ss_ablation.py` |
| Fig. 18, Table 11 | `noisyshapemotion/{starwm,dreamerv3}` | `probe_motion.py` | `plot_motion.py` |
| Fig. 19, Table 12 | `{shape,noisyshape,noisyshapeseq}/{starwm,dreamerv3}`, `shapefixed/dreamerv3` | `probe_noisyshape.py` | `plot_progressive.py` |
| Fig. 21 | `noisytwoshape/{starwm,dreamerv3}` | `probe_two_shape.py` | `plot_two_shape.py` |
| Fig. 22 | `noisytwoshape/starwm_s0` | – | `viz_two_shape_attention.py` |

All commands below are run from the `StarWM/` directory with the `shapeenv`
package installed (or on `PYTHONPATH`) and `DAVIS_DIR` pointing to
`DAVIS/JPEGImages/480p` (DAVIS 2017). Seeds are 0, 1, 2 everywhere.

## 1. Training

`--model starwm` is the default; the other variants are presets in
`configs.yaml` that are combined with the benchmark preset.

```bash
for s in 0 1 2; do
  # Fig. 3 / Table 4 (also the StarWM run of Fig. 12, 15, 19)
  python train.py --configs noisyshape                --seed $s --logdir runs/noisyshape/starwm_s$s
  for v in dreamerv3 dualstream ss_only dualstream_ss attention_only \
           iso_star1 iso_star2 iso_star3 iso_star4 inv_only ctr_only; do
    python train.py --configs noisyshape $v           --seed $s --logdir runs/noisyshape/${v}_s$s
  done
  # Fig. 19 / Table 12
  for b in shape noisyshapeseq; do
    python train.py --configs $b                      --seed $s --logdir runs/$b/starwm_s$s
    python train.py --configs $b dreamerv3            --seed $s --logdir runs/$b/dreamerv3_s$s
  done
  python train.py --configs shapefixed dreamerv3      --seed $s --logdir runs/shapefixed/dreamerv3_s$s
  # Fig. 18 / Table 11
  python train.py --configs noisyshapemotion          --seed $s --logdir runs/noisyshapemotion/starwm_s$s
  python train.py --configs noisyshapemotion dreamerv3 --seed $s --logdir runs/noisyshapemotion/dreamerv3_s$s
  # Fig. 21 / 22 (400k steps, set in the preset)
  python train.py --configs noisytwoshape             --seed $s --logdir runs/noisytwoshape/starwm_s$s
  python train.py --configs noisytwoshape dreamerv3   --seed $s --logdir runs/noisytwoshape/dreamerv3_s$s
done
```

## 2. Probes

Each probe script takes any number of run directories of one environment,
collects the probe trajectories once (cached as `.npz` in `--cache_dir`, shared
by all models) and writes `<run_dir>/probe_results/results.json`. Existing
results are skipped unless `--force`. A GPU is strongly recommended
(`--device`, default `cuda:0`).

```bash
for b in noisyshape shape noisyshapeseq shapefixed; do
  python analysis/noisyshape/probe_noisyshape.py runs/$b/*_s[012] --cache_dir runs/probe_cache
done
python analysis/noisyshape/probe_motion.py    runs/noisyshapemotion/*_s[012] --cache_dir runs/probe_cache
python analysis/noisyshape/probe_two_shape.py runs/noisytwoshape/*_s[012]   --cache_dir runs/probe_cache
```

Protocol (identical to the paper scripts): random-policy episodes of 200 steps,
500 episodes (env seed 42) for the H0 probes and 200 separate episodes (env seed
12345) for the imagination probes (NoisyTwoShape: 200 / 100); MLP probes
`Linear(1536,256)-ReLU-Dropout-Linear(256,256)-ReLU-Dropout-Linear(256,C)`, 70/30
split (stratified), Adam 1e-3 / wd 1e-4, batch 512, <= 100 epochs, early stopping
(patience 10), best checkpoint restored (`shapeenv.probe`). For imagination,
probes are re-fitted on the filtered features of the second set and applied
after h = 1..15 `img_step`s with uniform random actions; labels at h are rolled
forward with the environment's action logic (`shapeenv.labels`).
Shape-fixed has a constant position, so its position probes are skipped.

Seeding: the paper scripts used the unseeded global `np.random` for the
random policy and the imagined actions (NoisyShape family) and unseeded torch
for probe training. Here the policy uses `RandomState(env_seed)`, imagination
`RandomState(--imag_action_seed)` and probe training `torch.manual_seed(--probe_seed)`
(the two-object script was already seeded this way), so results are
reproducible but can differ slightly from the published numbers.

## 3. Figures and tables

```bash
python analysis/noisyshape/plot_noisyshape_probes.py --out fig3.pdf \
    --r2dreamer <r2dreamer results.json x3> --isodream <iso-dream++ results.json x3>
python analysis/noisyshape/plot_detach_ablation.py --out fig12.pdf
python analysis/noisyshape/plot_ss_ablation.py     --out fig15.pdf
python analysis/noisyshape/plot_motion.py          --out fig18.pdf     # + Table 11
python analysis/noisyshape/plot_progressive.py     --out fig19.pdf     # + Table 12
python analysis/noisyshape/plot_two_shape.py       --out fig21.pdf
python analysis/noisyshape/viz_two_shape_attention.py runs/noisytwoshape/starwm_s0 --out fig22.png
python analysis/noisyshape/viz_composition_swap.py   runs/noisyshape/starwm_s0    --out fig4.png
```

The plot scripts read `--runs` (default `runs`) and print the corresponding
table. All error bars / bands and all `±` values are the standard deviation
over seeds (`np.std`, ddof = 0), as in the paper scripts. Position MSE is
`MSE_x + MSE_y` of the H0 probe (chance 0.0711 for NoisyShape).

H0 conventions: each result file has two H0 numbers per attribute,
`current_probes` (500-episode set) and `imagined_probes["0"]` (probe re-fitted
on the 200-episode imagination set, the first point of every curve).
`plot_noisyshape_probes.py` prints Table 4 with both. In the paper, Table 4
uses `imagined_probes["0"]` for the DreamerV3, attention-only and StarWM rows
and `current_probes` for the dual-stream / SS rows; Tables 11 and 12 use
`imagined_probes["0"]` (Table 11 BG ID: `current_probes`). In Fig. 21 the
first point is `current_probes`.

R2-Dreamer and Iso-Dream++ (Fig. 3 / Table 4) are trained and probed in their
own code bases; pass their per-seed `results.json` files on the command line.
Iso-Dream++ files contain a `controllable` and a `combined` entry (both plotted).

### Original paper result files

`--name_map paper` maps release names onto the directory layout of the
original runs (`logdir_paper/noisyshapev2/starwm_detach_s0`, `orig_s0`,
`icr_only_s0`, ...), e.g.

```bash
python analysis/noisyshape/plot_noisyshape_probes.py --runs logdir_paper --name_map paper \
    --isodream logdir_paper/noisyshapev2/isodream_dmccfg_s{0,1,2}/probe_results/results.json
python analysis/noisyshape/plot_two_shape.py --runs logdir_paper --name_map paper \
    --results_subdir probe_results_current
```

(Fig. 21 used the probes of the final 400k-step checkpoints, stored under
`probe_results_current` in the original runs.) Paper checkpoints load into the
release model classes, so the probe scripts also run on the original run
directories.
