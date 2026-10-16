#!/usr/bin/env python3
"""Offline, world-model-only training on converted CALVIN play data
(Appendix F.3 (iii); checkpoint used for Table 15 and Fig. 26).

No environment and no actor-critic: the world model is built from fixed
observation / action spaces (64x64 image, 7-D continuous relative action) and
``wm._train(batch)`` is called on batches (16 x 64) sampled from the converted
replay buffer (calvin_convert.py). In this code base the actor-critic never
sends gradients into the world model (``_train`` returns a detached posterior
and the behaviour uses separate optimisers), so the world model receives the
same gradients as in full-agent training on the same data; with reward 0 on
play data the actor-critic would learn nothing anyway.

The configuration is train.py's defaults (the paper's StarWM settings) plus
the CALVIN overrides in ``CALVIN``; ``task='dmc_calvin'`` only selects the
continuous (MSE) inverse-dynamics head.

Example:
  python calvin/train_calvin_offline.py --method starwm --offline_traindir <conv>/D_train \
      --logdir runs/calvin/starwm --steps 2e5 --checkpoint_every 20000 --device cuda:0
Writes <logdir>/metrics.jsonl, <logdir>/wm_latest.pt (resumed from if present) and
<logdir>/checkpoints/wm_step<N>.pt = {"wm": state_dict, "step": N}.
"""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]      # StarWM root
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tools  # noqa: E402
import train  # noqa: E402

# --method -> release model. The CALVIN "DreamerV3" arm of the original study was
# the dual-stream world model (DreamerV3 + the same 128-d residual VAE as StarWM).
METHODS = {"starwm": "starwm", "dualstream": "dualstream", "dreamerv3": "dreamerv3"}

CALVIN = dict(
    task="dmc_calvin",          # only selects the continuous inverse-dynamics head
    skip_actor_critic=True,     # world model only (no actor-critic is built here)
    action_repeat=1, envs=1, compile=False, precision=32,
    batch_size=16, batch_length=64, video_pred_log=False, discount=0.997,
    num_actions=7,
)
# StarWM settings used on CALVIN (identical to the train.py defaults; listed for clarity).
STARWM = dict(
    crossattn_num_queries=4, crossattn_num_heads=4, crossattn_proj_dim=256,
    crossattn_pe_type="2d_sin", attn_inv_scale=1.0, infonce_scale=1.0,
    infonce_temp=0.1, detach_attn_for_rssm=True, attn_anneal_steps=0,
    res_dim=128, Annealing=True, res_kl_scale=0.5, res_kl_scale_init=5.0,
    res_kl_anneal_steps=15000, use_res_kl=True,
)


def build_config(method="starwm", overrides=None):
    """train.py defaults + CALVIN overrides (+ StarWM settings) + ``overrides``."""
    cfg = train.load_config()
    cfg.model = METHODS[method]
    for k, v in CALVIN.items():
        setattr(cfg, k, v)
    if method == "starwm":
        for k, v in STARWM.items():
            setattr(cfg, k, v)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    return cfg


def spaces(cfg):
    import gymnasium as gym
    H, W = cfg.size[0], cfg.size[1]
    obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (H, W, 3), np.uint8)})
    act_space = gym.spaces.Box(-1.0, 1.0, (cfg.num_actions,), np.float32)
    return obs_space, act_space


def build_wm(cfg):
    obs_space, act_space = spaces(cfg)
    return train.build_world_model(obs_space, act_space, 0, cfg).to(cfg.device)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=sorted(METHODS), default="starwm")
    ap.add_argument("--offline_traindir", required=True, help="output of calvin_convert.py")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=2e5)
    ap.add_argument("--checkpoint_every", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=200)
    a = ap.parse_args()

    cfg = build_config(a.method, dict(logdir=a.logdir, device=a.device, seed=a.seed))
    steps = int(a.steps)
    torch.set_num_threads(8)
    tools.set_seed_everywhere(cfg.seed)
    device = cfg.device
    logdir = pathlib.Path(a.logdir); logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "checkpoints").mkdir(exist_ok=True)

    # ---- offline data ----
    train_eps = tools.load_episodes(pathlib.Path(a.offline_traindir), limit=None)
    n_frames = sum(len(e["reward"]) for e in train_eps.values())
    print(f"[data] {len(train_eps)} episodes, {n_frames} frames from {a.offline_traindir}", flush=True)
    dataset = train.make_dataset(train_eps, cfg)

    # ---- world model (no env: fixed spaces) ----
    wm = build_wm(cfg)
    wm.requires_grad_(False)          # _train enables gradients itself (tools.RequiresGrad)
    n_par = sum(p.numel() for p in wm.parameters())
    print(f"[model] {cfg.model} ({type(wm).__name__}): {n_par / 1e6:.1f}M params on {device}", flush=True)

    # ---- resume ----
    start = 0
    ckpt = logdir / "wm_latest.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        wm.load_state_dict(st["wm"]); start = int(st["step"])
        print(f"[resume] from step {start}", flush=True)

    # ---- train loop (world model only) ----
    t0 = time.time(); tlog = t0
    for step in range(start, steps):
        data = next(dataset)
        _, _, mets = wm._train(data)
        if step % a.log_every == 0:
            row = {"step": step}
            for k, v in mets.items():
                try:
                    row[k] = float(np.asarray(v).mean())
                except Exception:
                    pass
            fps = (a.log_every * cfg.batch_size * cfg.batch_length) / max(1e-6, time.time() - tlog)
            row["fps"] = fps; tlog = time.time()
            with (logdir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            keys = [k for k in ["kl_loss", "image_loss", "reward_loss",
                                "attn_inv_loss", "infonce_loss", "res_kl_loss"] if k in row]
            print(f"[{step}] " + " ".join(f"{k}={row[k]:.3f}" for k in keys) + f"  fps={fps:.0f}", flush=True)
        if step > start and step % a.checkpoint_every == 0:
            torch.save({"wm": wm.state_dict(), "step": step, "config": vars(cfg)}, ckpt)
            torch.save({"wm": wm.state_dict(), "step": step},
                       logdir / "checkpoints" / f"wm_step{step}.pt")
            print(f"[ckpt] step {step} saved", flush=True)

    torch.save({"wm": wm.state_dict(), "step": steps, "config": vars(cfg)}, ckpt)
    print(f"[done] {steps} steps in {(time.time() - t0) / 60:.1f} min -> {logdir}", flush=True)


if __name__ == "__main__":
    main()
