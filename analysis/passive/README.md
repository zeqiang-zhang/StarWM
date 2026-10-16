# Passive object (Appendix F.4 (i))

`PassiveShapeEnv` adds an object B to NoisyShape whose colour and position are drawn once per
episode and never respond to actions (no inverse-dynamics signal). The probes test whether
StarWM's `[h;z]` still encodes B, with an i.i.d. random background frame per step
(`passiveshape`) and with a sequential video background (`passiveshapeseq`).

| Paper | Content | Script |
|---|---|---|
| Fig. 27 | five steps of one episode per background | `plot_passive_env.py` |
| Table 16, Fig. 28 | b_colour accuracy, b_x / b_y R^2 at H0 and H15 | `probe_passive.py`, `probe_passive_position.py` (per run) -> `plot_passive_probe.py` |

The probes reuse `analysis/textured/probe_common.py`. All commands run from the `StarWM` root
with DAVIS 2017 `JPEGImages/480p` in `$DAVIS_DIR` (or `--davis_path`).

## Training (StarWM, 3 seeds per background)

```bash
for S in 0 1 2; do
  python train.py --configs passiveshape    --seed $S --logdir runs/passiveshape/starwm_s$S
  python train.py --configs passiveshapeseq --seed $S --logdir runs/passiveshapeseq/starwm_s$S
done
```
`steps=2e5`; runs stop at 212.4k env steps, the step count reported with Table 16.

## Analysis

```bash
# model-free probe sets (500 + 200 episodes); colour and position probes need separate caches
for m in random seq; do
  python analysis/passive/probe_passive.py          --mode $m --collect_only \
      --traj_cache runs/traj_cache/passive_${m}_attr.npz
  python analysis/passive/probe_passive_position.py --mode $m --collect_only \
      --traj_cache runs/traj_cache/passive_${m}_pos.npz
done
for S in 0 1 2; do for m in random seq; do
  g=passiveshape; [ $m = seq ] && g=passiveshapeseq
  python analysis/passive/probe_passive.py          --logdir runs/$g/starwm_s$S --mode $m \
      --traj_cache runs/traj_cache/passive_${m}_attr.npz --device cuda:0
  python analysis/passive/probe_passive_position.py --logdir runs/$g/starwm_s$S --mode $m \
      --traj_cache runs/traj_cache/passive_${m}_pos.npz --device cuda:0
done; done
# Table 16 (printed) + Fig. 28
python analysis/passive/plot_passive_probe.py --root runs --out_dir figures
# Fig. 27
python analysis/passive/plot_passive_env.py --out figures/passive_env
```

## Notes

* `probe_passive.py` also reports the action-driven object A's colour / shape / size as
  controls. B is static, so its H15 target equals its H0 label.
* Position: MLP regressor, 70/30 split; R^2 = 1 - MSE / Var(y_test). For H15 the regressor
  is trained on the imagined set's real H0 features and applied to all features after 15
  imagined steps.
* Probe training, imagined actions and the RSSM's stochastic state are not seeded.
