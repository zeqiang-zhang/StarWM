"""Helpers shared by the Appendix F latent probes (textured ladder, passive
object, delayed action) and their figures.

* model loading through ``analysis/wm_loader.py``;
* ``[h;z]`` feature extraction at H0 and after open-loop imagination. This is
  a verbatim port of the probe code used for the paper: whole episodes are
  filtered with ``RSSM.observe`` in batches of 32 and imagined actions are
  drawn with the global NumPy RNG, one draw per (batch, step). The features
  equal those of ``wm_loader.encode_episodes``, but the RNG consumption order
  differs, so the original routine is kept;
* the paper's figure style (palette and fonts).

The passive and delayed-action scripts import this module from
``analysis/textured``.
"""
import json
import os
import pathlib
import sys

import numpy as np
import torch

_ANALYSIS = pathlib.Path(__file__).resolve().parents[1]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))
import wm_loader  # noqa: E402  (also puts the StarWM root on sys.path)
import train  # noqa: E402

CURR_SEED = 42      # paper protocol: "current" set -> H0 probes
IMAG_SEED = 12345   # paper protocol: "imagined" set -> H15 probes


# --------------------------------------------------------------------- inputs
def default_davis():
    return os.environ.get("DAVIS_DIR", "")


def require_davis(parser, path):
    """All probe sets of the paper were rendered on DAVIS backgrounds."""
    if not path or not os.path.isdir(path):
        parser.error("DAVIS JPEGImages/480p directory not found: "
                     "pass --davis_path or set $DAVIS_DIR")


def last_logged_step(logdir):
    """Last ``step`` in ``<logdir>/metrics.jsonl`` (0 if none)."""
    p = os.path.join(logdir, "metrics.jsonl")
    s = 0
    if os.path.exists(p):
        for line in open(p):
            try:
                s = int(json.loads(line).get("step", s))
            except Exception:
                pass
    return s


def seed_dirs(pattern_prefix):
    """``<prefix><digits>`` directories only (skips e.g. ``*_s0_old``)."""
    import glob
    out = []
    for d in sorted(glob.glob(pattern_prefix + "*")):
        tail = d[len(pattern_prefix):]
        if tail.isdigit() and os.path.isdir(d):
            out.append(d)
    return out


# ---------------------------------------------------------------------- model
def load_model(logdir, checkpoint="latest.pt", device="cuda:0", presets=(), model=None):
    """World model of a run (eval mode). ``model`` overrides the logged model type."""
    overrides = {"model": model} if model else None
    wm, cfg = wm_loader.load_world_model(logdir, checkpoint=checkpoint, device=device,
                                         presets=presets, overrides=overrides)
    print(f"loaded {cfg.model} world model from {pathlib.Path(logdir) / checkpoint}")
    return wm, cfg


@torch.no_grad()
def _prep(wm, cfg, imgs, acts, isfirst, s, e):
    dev = cfg.device
    B = e - s
    T1 = imgs.shape[1]
    data = {
        "image": torch.tensor(imgs[s:e], device=dev),
        "action": torch.tensor(acts[s:e], device=dev, dtype=torch.float32),
        "is_first": torch.tensor(isfirst[s:e], device=dev, dtype=torch.float32),
        "is_terminal": torch.zeros(B, T1, device=dev),
    }
    pad = torch.zeros(B, 1, acts.shape[-1], device=dev)   # first action unused
    data["action"] = torch.cat([pad, data["action"]], dim=1)
    return wm.preprocess(data), B, T1


@torch.no_grad()
def _embed_batch(wm, cfg, data):
    """RSSM input: cross-attention pathway (StarWM) or plain encoder (DreamerV3)."""
    return train.policy_embed(wm, data, cfg.model)


@torch.no_grad()
def encode_h0(wm, cfg, imgs, acts, isfirst, bs=32):
    """H0 features [h;z] -> (N*T1, F)."""
    N = imgs.shape[0]
    out = []
    for s in range(0, N, bs):
        e = min(s + bs, N)
        data, B, T1 = _prep(wm, cfg, imgs, acts, isfirst, s, e)
        embed = _embed_batch(wm, cfg, data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        out.append(wm.dynamics.get_feat(post).reshape(B * T1, -1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def encode_imagined(wm, cfg, imgs, acts, isfirst, horizon, bs=32, n_actions=4):
    """Real H0 feats + open-loop imagination with uniform random one-hot actions
    started from every posterior state.
    -> rf (N*T1, F), imf (horizon, N*T1, F), ima (horizon, N*T1) int actions."""
    dev = cfg.device
    N = imgs.shape[0]
    rf, imf, ima = [], [], []
    for s in range(0, N, bs):
        e = min(s + bs, N)
        data, B, T1 = _prep(wm, cfg, imgs, acts, isfirst, s, e)
        embed = _embed_batch(wm, cfg, data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        rf.append(wm.dynamics.get_feat(post).reshape(B * T1, -1).cpu().numpy())
        state = {k: v.reshape(B * T1, *v.shape[2:]) for k, v in post.items()}
        bh_f = [None] * horizon
        bh_a = [None] * horizon
        for h in range(horizon):
            ai = np.random.randint(0, n_actions, size=B * T1)
            a = torch.zeros(B * T1, n_actions, device=dev)
            a[np.arange(B * T1), ai] = 1.0
            state = wm.dynamics.img_step(state, a)
            bh_f[h] = wm.dynamics.get_feat(state).cpu().numpy()
            bh_a[h] = ai
        imf.append(np.stack(bh_f))
        ima.append(np.stack(bh_a))
    return np.concatenate(rf, 0), np.concatenate(imf, 1), np.concatenate(ima, 1)


# --------------------------------------------------------------------- figures
# Paper palette: StarWM red, DreamerV3 blue.
METHOD_COLOR = {"starwm": "#E53935", "dreamerv3": "#1565C0", "dreamer": "#1565C0"}
METHOD_LABEL = {"starwm": "StarWM", "dreamerv3": "DreamerV3", "dreamer": "DreamerV3"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "grid.linewidth": 0.6, "axes.linewidth": 0.8,
        "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
        "axes.titleweight": "600",
    })
    return plt


def savefig(fig, path):
    """Write ``path.pdf`` and ``path.png``."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print("saved", path + ".{pdf,png}")


def mean_se(vals):
    """Mean and standard error (ddof=1) over seeds; (None, None) if empty."""
    if not len(vals):
        return None, None
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
