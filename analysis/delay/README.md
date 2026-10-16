# Delayed action effects (Appendix F.5)

`ActionDelay(k)` (in `envs/wrappers.py`) makes NoisyShape apply the action issued k steps
earlier, so a_t changes the image only between frames t+k and t+k+1, while the replay buffer
stores the issued a_t. Two inverse-dynamics windows are compared:

* adjacent (`--inv_delay 0`): predict a_t from (e_t, e_{t+1}), the standard objective;
* shifted (`--inv_delay k`): predict a_t from (e_{t+k}, e_{t+k+1}), the frame pair a_t changes.

| Paper | Content | Script |
|---|---|---|
| Table 19, Attn. mass / xUniform | best-query attention mass inside the object disk | `viz_attention_delay.py` |
| Table 19, Shape / Size | held-out attribute decodability from `[h;z]` | `probe_delay_attr.py` |
| Table 19 | per-run records -> mean +- SE | `run_delay_sweep.sh` -> `aggregate_delay.py` |
| F.5 text: training inverse accuracy is not used | inverse head on replay vs fresh rollouts | `eval_delay_heldout.py` |

The scripts reuse `analysis/textured/probe_common.py`. All commands run from the `StarWM` root
with DAVIS 2017 `JPEGImages/480p` in `$DAVIS_DIR`.

## Training

```bash
for k in 1 2 4; do for S in 0 1 2; do
  python train.py --configs noisyshape --action_delay $k --inv_delay 0  --checkpoint_every 20000 \
      --seed $S --logdir runs/delay/k${k}_adjacent_s$S
  python train.py --configs noisyshape --action_delay $k --inv_delay $k --checkpoint_every 20000 \
      --seed $S --logdir runs/delay/k${k}_shifted_s$S
done; done
# k = 0 reference (no delay)
python train.py --configs noisyshape --action_delay 0 --inv_delay 0 --checkpoint_every 20000 \
    --seed 0 --logdir runs/delay/k0_adjacent_s0
```

## Analysis

```bash
# held-out probe trajectories (300 episodes x 200 steps per k)
python analysis/delay/collect_delay_trajs.py --k 0 1 2 4 --out_dir runs/traj_cache
# probe every run at checkpoints/agent_step180000.pt, write runs/delay_sweep/k<k>_<arm>_s<S>.txt,
# then runs/delay_sweep/aggregate.json and the Table 19 printout
bash analysis/delay/run_delay_sweep.sh runs/delay runs/delay_sweep cuda:0
# re-aggregate existing records only
python analysis/delay/aggregate_delay.py --sweep_dir runs/delay_sweep
# optional: inverse-head accuracy on replay vs fresh rollouts
python analysis/delay/eval_delay_heldout.py --logdir runs/delay/k4_adjacent_s0 \
    --checkpoint checkpoints/agent_step180000.pt --device cuda:0
```

## Notes

* Table 19 compares all delayed runs at the matched 180k checkpoint. The k = 0 row of the paper
  is a single run probed at its last checkpoint (training had stopped at about 145k steps);
  `run_delay_sweep.sh` probes k = 0 runs at `$K0_CKPT` (default `latest.pt`). Set
  `K0_CKPT=checkpoints/agent_step140000.pt` to approximate the paper's reference with a fully
  trained run, or `K0_CKPT=checkpoints/agent_step180000.pt` for a matched-step reference.
* Attention mass: 8 consecutive frames of one rollout (env seed 555, policy seed 2); the object
  mask is a disk of the object's radius; per frame the query with the largest mass inside the
  disk is taken. xUniform in Table 19 is the mean over seeds of best mass / disk area.
* Attribute probes: MLP, 70/30 stratified split, on all 300 x 201 frames. Colour stays near 1.0
  in every arm and is omitted from Table 19.
* Probe training and the RSSM's stochastic state are not seeded; the attention metric is
  deterministic.
