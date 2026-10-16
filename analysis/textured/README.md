# Textured foreground ladder (Appendix F.3 (i))

NoisyShape with the solid colour replaced by a texture ID (4 textures) at three levels:
L1 procedural textures, L2 DAVIS crops (same domain as the background), L3 the L2 crops
alpha-blended into the background (alpha 0.5, feathered edge). Environment:
`shapeenv.TexturedNoisyShapeEnv` (`--configs texnoisyshape --tex_level L`).

| Paper | Content | Script |
|---|---|---|
| Fig. 23 | the four textures at each level | `plot_textured_envs.py` |
| Table 13, Fig. 24 | texture / shape / size probe accuracy (H0, H15), position MSE (H0) | `probe_textured.py` (per run) -> `aggregate_textured.py` |
| text after Table 13 | attention mass of the best query on the object (0.82 / 0.57 / 0.37) | `viz_attention_textured.py` |

`probe_common.py` holds the model loading, `[h;z]` feature extraction and figure style shared
with `analysis/passive` and `analysis/delay`.

All commands run from the `StarWM` root, with DAVIS 2017 `JPEGImages/480p` in `$DAVIS_DIR`
(or `--davis_path`).

## Training (3 seeds x 3 levels x 2 models)

```bash
for L in 1 2 3; do for S in 0 1 2; do
  python train.py --configs texnoisyshape --tex_level $L --seed $S \
      --logdir runs/texnoisyshape/L$L/starwm_s$S
  python train.py --configs texnoisyshape dreamerv3 --tex_level $L --seed $S \
      --logdir runs/texnoisyshape/L$L/dreamerv3_s$S
done; done
```
The preset trains for `steps=4e5` (the loop stops at about 412k env steps; "400k" in the paper).

## Analysis

```bash
# model-free probe trajectories, one cache per level (500 + 200 episodes)
for L in 1 2 3; do
  python analysis/textured/probe_textured.py --collect_only --tex_level $L \
      --traj_cache runs/traj_cache/tex_L$L.npz
done
# probe every run (writes <run>/probe_textured_results.json), then Table 13 + Fig. 24
python analysis/textured/aggregate_textured.py --root runs/texnoisyshape --run \
    --cache_dir runs/traj_cache --device cuda:0 --out_dir figures
# Fig. 23
python analysis/textured/plot_textured_envs.py --out figures/textured_ladder_envs
# attention mass on the object (one run per level)
for L in 1 2 3; do
  python analysis/textured/viz_attention_textured.py --logdir runs/texnoisyshape/L$L/starwm_s0 \
      --tex_level $L --device cuda:0 --out figures/attention_textured_L$L.png
done
```

`aggregate_textured.py` without `--run` only aggregates existing result files. Runs whose
`metrics.jsonl` ends before 150k steps are excluded and listed.

## Notes

* Protocol (Appendix C.2): H0 probes on 500 episodes (seed 42); H15 probes are trained on the
  real H0 features of a separate 200-episode set (seed 12345) and applied after 15 open-loop
  imagination steps with uniform random actions. Position MSE is MSE_x + MSE_y; chance ~0.071.
* Probe training, the imagined actions and the RSSM's stochastic state are not seeded, so a
  re-run varies in the third decimal.
* The attention-mass script uses 8 consecutive frames of one rollout (env seed 777) and a disk
  of the object's radius as the object mask (about 12% of the image). It prints both the
  combined (sum over queries) and the best-query mass; the paper quotes the best query.
