# CALVIN attention routing (Appendix F.3 (iii))

CALVIN `task_D_D` (teleoperated play in a PyBullet-simulated kitchen, static camera, 7-DOF
relative actions) is used to measure where StarWM's cross-attention goes when the robot,
blocks, table and kitchen share one photorealistic render. The world model is trained
offline (no environment interaction, no policy). Only attention concentration is measured
here; there is no task-performance comparison.

| Paper | Content | Script |
|---|---|---|
| data | per-frame CALVIN npz -> 500-frame replay episodes + state cache | `calvin_convert.py` |
| model | offline world-model-only training | `train_calvin_offline.py` |
| Table 15 | attention concentration on frames with a moving block (n = 204) | `attn_table.py --moving` |
| F.3 (iii) text | gripper 9.71x vs 2.84x null on all validation frames | `attn_table.py` |
| Fig. 26 | per-query attention maps on held-out frames | `viz_attention_calvin_paperstyle.py` |
| (helpers) | camera projection, masks, attention maps; single-pass summary | `calvin_attn_metric_proj.py` |

## Data

Download `task_D_D` from the CALVIN repository and convert both splits (paths are arguments):

```bash
python calvin/calvin_convert.py --src <calvin>/task_D_D/training   --out <conv>/D_train
python calvin/calvin_convert.py --src <calvin>/task_D_D/validation --out <conv>/D_val
```
This writes `D_train/`, `D_val/` (replay episodes; `action[t]` = relative action that led to
frame t, `action[0] = 0`) and `D_train_state/`, `D_val_state/` (`scene_obs`, `robot_obs` per
frame). Both splits come from the same scene; the validation split is held out.

## Training

```bash
python calvin/train_calvin_offline.py --method starwm --offline_traindir <conv>/D_train \
    --logdir runs/calvin/starwm --steps 2e5 --checkpoint_every 20000 --device cuda:0
```
Configuration: `train.py` defaults (the paper's StarWM settings) plus `task=dmc_calvin`
(selects the continuous inverse-dynamics head), batch 16 x 64, 64x64 images, 7-D actions.
Checkpoints: `runs/calvin/starwm/checkpoints/wm_step<N>.pt`. `--method dualstream` trains the
DreamerV3 + residual-VAE world model used as the CALVIN baseline in the original study.

## Analysis (held-out validation split)

```bash
CK=runs/calvin/starwm/checkpoints/wm_step180000.pt
python calvin/attn_table.py --ckpt $CK --data_dir <conv>/D_val --state_dir <conv>/D_val_state --moving  # Table 15
python calvin/attn_table.py --ckpt $CK --data_dir <conv>/D_val --state_dir <conv>/D_val_state           # all frames
python calvin/viz_attention_calvin_paperstyle.py --ckpt $CK --data_dir <conv>/D_val \
    --out figures/calvin_attention.png                                                                     # Fig. 26
```
The reported numbers were measured on the 180k-step checkpoint of the StarWM run.

## Notes

* Masks: the end-effector (`robot_obs[:3]`) and block centres (`scene_obs`) are projected
  with the CALVIN static-camera calibration (constants in `calvin_attn_metric_proj.py`);
  gripper disk r = 5 px, block disks r = 3 px on the 64x64 image. The "active" block is the
  one that moves most between t and t+1.
* Attention per frame = maximum over the 4 queries of the 8x8 maps, nearest-upsampled and
  normalised. Concentration = attention share / area share.
* Each region has its own shuffled null (the same attention map scored against another
  frame's mask of that region). Manipulation in CALVIN is spatially centred, so the nulls are
  2-3x rather than 1x; compare every concentration with its own null.
* Frames: 40 random 500-frame validation episodes, t = 20, 45, ..., 495 (800 frames).
