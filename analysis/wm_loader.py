"""Load a trained world model for analysis (probes, attention maps, swaps).

Works for runs produced by ``train.py`` (``<logdir>/config.yaml``) and for the
original paper runs, whose logged configs use the research code's flag names
(``use_models_crossattn_assemble``, ``crossattn_mode``, ``algo`` ...); those are
translated by :func:`translate_legacy_config`. Parameter names are identical
between the research code and this release, so paper checkpoints load as is.
"""
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402  (StarWM/train.py)

# research-code suite name -> shapeenv / release suite name
LEGACY_SUITES = {
    "shapev2": "shape", "shapevfixed": "shapefixed", "noisyshapev2": "noisyshape",
    "noisyshapev2seq": "noisyshapeseq", "noisyshapev4": "noisyshapemotion",
}


def _unwrap(d):
    """swanlab config.yaml stores {key: {desc, value}}; plain yaml stores {key: value}."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "value" in v and set(v) <= {"value", "desc", "sort"}:
            v = v["value"]
        out[k] = v
    return out


def translate_legacy_config(cfg):
    """Map a research-code config dict onto release keys (in place, returns it)."""
    if "model" not in cfg:
        # research-code run: its "+ SS" heads are the legacy variant (see models.py)
        cfg.setdefault("legacy_ss_ablation", True)
        if cfg.get("use_models_crossattn_assemble"):
            cfg["model"] = "starwm"
        elif cfg.get("use_models_crossattn_single"):
            cfg["model"] = "attention_only"
        elif str(cfg.get("algo", "")).startswith("dreamerv3_original"):
            cfg["model"] = "dreamerv3"
        else:
            cfg["model"] = "dualstream"   # research models.py == dual-stream ablation
    task = str(cfg.get("task", ""))
    suite = task.split("_", 1)[0]
    if suite in LEGACY_SUITES:
        cfg["task"] = LEGACY_SUITES[suite] + task[len(suite):]
    # keys that did not exist when some runs were made (their behaviour == default)
    cfg.setdefault("detach_attn_for_recon", True)
    return cfg


def load_run_config(logdir, presets=(), overrides=None, device="cuda:0"):
    """Config Namespace for a run directory.

    Precedence: release defaults < ``presets`` < the run's logged config
    (``config.yaml`` or swanlab ``run-*/files/config.yaml``, if found)
    < ``overrides``.
    """
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe", pure=True)
    config = train.load_config(presets)
    logdir = pathlib.Path(logdir)
    logged = None
    if (logdir / "config.yaml").exists():
        logged = yaml.load((logdir / "config.yaml").read_text())
    else:
        runs = sorted(logdir.glob("run-*/files/config.yaml"))
        if runs:
            logged = yaml.load(runs[-1].read_text())
    if logged:
        logged = translate_legacy_config(_unwrap(logged))
        for k, v in logged.items():
            if hasattr(config, k) and k not in ("logdir", "traindir", "evaldir", "device"):
                if isinstance(v, list) and isinstance(getattr(config, k), tuple):
                    v = tuple(v)  # the command line parses sequences as tuples
                setattr(config, k, v)
    for k, v in (overrides or {}).items():
        setattr(config, k, v)
    config.logdir = str(logdir)
    config.device = device
    return config


def _spaces_for(config):
    env = train.make_env(config, "eval", 0)
    obs_space, act_space = env.observation_space, env.action_space
    try:
        env.close()
    except Exception:
        pass
    return obs_space, act_space


def load_world_model(logdir, config=None, checkpoint="latest.pt", device="cuda:0", **kw):
    """Instantiate the run's world model and load its weights (eval mode).

    ``checkpoint`` is relative to ``logdir`` (e.g. ``checkpoints/agent_step180000.pt``).
    Returns ``(wm, config)``; ``config.model`` tells which encoder path to use.
    """
    logdir = pathlib.Path(logdir)
    if config is None:
        config = load_run_config(logdir, device=device, **kw)
    config.device = device
    obs_space, act_space = _spaces_for(config)
    config.num_actions = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    wm = train.build_world_model(obs_space, act_space, 0, config)
    ckpt = torch.load(logdir / checkpoint, map_location=device, weights_only=False)
    state = {}
    for k, v in ckpt["agent_state_dict"].items():
        if "_wm." not in k:
            continue
        nk = k.split("_wm.", 1)[1]
        if nk.startswith("_orig_mod."):  # torch.compile prefix
            nk = nk[len("_orig_mod."):]
        state[nk] = v
    missing, unexpected = wm.load_state_dict(state, strict=False)
    if missing:
        print(f"[wm_loader] missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"[wm_loader] unexpected keys ({len(unexpected)}): {unexpected[:3]}")
    wm.to(device).eval()
    return wm, config


def obs_batch(images, is_first, device):
    """Images uint8 (B,H,W,3) -> preprocessed obs dict for one time step."""
    B = images.shape[0]
    return {
        "image": torch.tensor(images, device=device, dtype=torch.float32) / 255.0,
        "is_first": torch.tensor(is_first, device=device, dtype=torch.float32),
        "is_terminal": torch.zeros(B, device=device, dtype=torch.float32),
    }


def embed_obs(wm, config, op):
    """RSSM input embedding for one time step (same path as acting)."""
    if config.model in ("starwm", "attention_only"):
        # identical to train.policy_embed but without the reshape round trip
        inputs = torch.cat([op[k] for k in wm.encoder.cnn_shapes], -1)
        x = inputs.reshape((-1,) + tuple(inputs.shape[-3:])).permute(0, 3, 1, 2)
        x = x - 0.5
        f8_proj = wm.conv_1x1(wm.encoder_to_8x8(x))
        pooled_for_rssm, _, _ = wm._compute_attention(f8_proj)
        return wm.proj_dyn(pooled_for_rssm.unsqueeze(1)).squeeze(1)
    return train.policy_embed(wm, op, config.model)


@torch.no_grad()
def encode_episodes(wm, config, images, actions, is_first, batch_size=64, horizon=0,
                    action_sampler=None):
    """Filter episodes through encoder + RSSM; optionally imagine open-loop.

    images: (E, T+1, H, W, 3) uint8; actions: (E, T, A) one-hot / continuous
    (``actions[:, t]`` leads from frame t to t+1); is_first: (E, T+1).
    Returns ``feats`` (E*(T+1), F) and, if ``horizon > 0``, ``imag_feats``
    (horizon, E*(T+1), F) plus the sampled integer imagined actions
    (horizon, E*(T+1)). ``action_sampler(B) -> (onehot float32 (B,A), int (B,))``
    defaults to uniform random one-hot actions over ``config.num_actions``.
    This reproduces encode_trajectories_batch / encode_imagined_batch of the
    paper's probe script.
    """
    device = config.device
    E, Tp1 = images.shape[:2]
    T = Tp1 - 1
    if action_sampler is None:
        A = config.num_actions

        def action_sampler(B):
            ri = np.random.randint(0, A, size=B)
            ra = np.zeros((B, A), dtype=np.float32)
            ra[np.arange(B), ri] = 1.0
            return ra, ri
    feats_all, imag_all, acts_all = [], [[] for _ in range(horizon)], [[] for _ in range(horizon)]
    for bs in range(0, E, batch_size):
        be = min(bs + batch_size, E)
        B = be - bs
        latent, action = None, None
        bf, bimg, bact = [], [[] for _ in range(horizon)], [[] for _ in range(horizon)]
        for t in range(Tp1):
            op = obs_batch(images[bs:be, t], is_first[bs:be, t], device)
            embed = embed_obs(wm, config, op)
            latent, _ = wm.dynamics.obs_step(latent, action, embed, op["is_first"])
            bf.append(wm.dynamics.get_feat(latent).cpu().numpy())
            if horizon:
                state = {k: v.clone() for k, v in latent.items()}
                for h in range(horizon):
                    ra, ri = action_sampler(B)
                    state = wm.dynamics.img_step(state, torch.tensor(ra, device=device))
                    bimg[h].append(wm.dynamics.get_feat(state).cpu().numpy())
                    bact[h].append(ri.copy())
            if t < T:
                action = torch.tensor(actions[bs:be, t], device=device)
        feats_all.append(np.stack(bf, 0).transpose(1, 0, 2).reshape(-1, bf[0].shape[-1]))
        for h in range(horizon):
            imag_all[h].append(np.stack(bimg[h], 0).transpose(1, 0, 2).reshape(-1, bimg[h][0].shape[-1]))
            acts_all[h].append(np.stack(bact[h], 0).transpose(1, 0).reshape(-1))
    feats = np.concatenate(feats_all, 0)
    if not horizon:
        return feats
    return (feats,
            np.stack([np.concatenate(x, 0) for x in imag_all], 0),
            np.stack([np.concatenate(x, 0) for x in acts_all], 0))


@torch.no_grad()
def attention_maps(wm, images):
    """StarWM / attention-only attention maps for a batch of frames.

    images: (B, H, W, 3) uint8 -> numpy (B, N, 8, 8) (N=1 for attention_only).
    """
    device = next(wm.parameters()).device
    x = torch.tensor(images, device=device, dtype=torch.float32) / 255.0
    x = x.permute(0, 3, 1, 2) - 0.5
    f8_proj = wm.conv_1x1(wm.encoder_to_8x8(x))
    _, _, A = wm._compute_attention(f8_proj)
    S = wm.f8_spatial
    return A.reshape(A.shape[0], -1, S, S).cpu().numpy()
