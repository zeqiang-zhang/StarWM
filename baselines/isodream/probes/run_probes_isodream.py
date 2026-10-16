"""
NoisyShape probe analysis for Iso-Dream++ checkpoints (StarWM paper, Fig. 3).

Reports two feature settings:
  - controllable: [stoch (50) + deter (200)] = 250-d
  - combined:     [stoch + deter + stoch_free + deter_free] = 500-d

For each, runs:
  - H0 (encoded) probes: color/shape/size (classification), pos_x/pos_y (regression)
  - H1..H15 imagined probes with the NoisyShape action logic to compute ground-truth labels

Copied into <Iso-Dream checkout>/iso_rl by ``baselines/setup_baselines.sh``.
Usage (from iso_rl/):
  python run_probes_isodream.py --logdir <run dir> --device cuda:0 [--davis_path $DAVIS_DIR]
Writes <run dir>/probe_results/results.json.
"""
import argparse
import os
import json
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm
from ruamel.yaml import YAML

HERE = pathlib.Path(__file__).resolve().parent  # iso_rl/ (configs.yaml, models.py, ...)

import models
import tools
import wrappers

import shapeenv

# DAVIS 2017 JPEGImages/480p directory (set from --davis_path / $DAVIS_DIR in main()).
DAVIS_PATH = None


# ---------------------------------------------------------------------- env --
def make_env(seed=0):
    raw = shapeenv.NoisyShapeEnv(size=(64, 64), seed=seed, davis_path=DAVIS_PATH)
    return raw


def collect_trajectories(num_episodes, env_seed, ep_len=200):
    raw = make_env(seed=env_seed)
    images, actions, labels = [], [], {
        "color": [], "shape": [], "size": [], "pos_x": [], "pos_y": []}
    rng = np.random.RandomState(env_seed)
    for _ in tqdm(range(num_episodes), desc=f"Collect seed={env_seed}"):
        obs = raw.reset(seed=int(rng.randint(2**31 - 1)))
        ep_imgs = [obs["image"]]
        ep_acts = []
        ep_lab = {
            "color": [raw._color_id], "shape": [raw._shape_id],
            "size":  [raw._size_id],  "pos_x": [raw._pos[0]],
            "pos_y": [raw._pos[1]],
        }
        for _ in range(ep_len):
            ai = rng.randint(0, 4)
            a = np.zeros(4, dtype=np.float32)
            a[ai] = 1.0
            obs, _r, _d, _t, info = raw.step(ai)
            ep_imgs.append(obs["image"])
            ep_acts.append(a)
            ep_lab["color"].append(info["color_id"])
            ep_lab["shape"].append(info["shape_id"])
            ep_lab["size"].append(info["size_id"])
            ep_lab["pos_x"].append(info["position_x"])
            ep_lab["pos_y"].append(info["position_y"])
        images.append(np.stack(ep_imgs))           # (T+1, H, W, 3)
        actions.append(np.stack(ep_acts))           # (T,   4)
        for k in ep_lab:
            labels[k].append(np.array(ep_lab[k]))
    return (
        np.stack(images),
        np.stack(actions),
        {k: np.stack(v) for k, v in labels.items()},
    )


# ----------------------------------------------------------- config / model --
def load_config(logdir):
    yaml = YAML(typ="safe", pure=True)
    cfg = yaml.load((HERE / "configs.yaml").read_text())
    merged = {}
    for n in ["defaults", "noisyshape"]:
        for k, v in cfg[n].items():
            merged[k] = v
    p = argparse.ArgumentParser()
    for k, v in sorted(merged.items()):
        p.add_argument(f"--{k}", type=tools.args_type(v),
                       default=tools.args_type(v)(v))
    config = p.parse_args([])
    config.logdir = str(logdir)
    config.num_actions = 4
    return config


def load_world_model(logdir, device):
    config = load_config(logdir)
    config.device = device
    config.act = getattr(torch.nn, config.act)
    # Schedules expected callable
    config.actor_entropy = (lambda x=config.actor_entropy: tools.schedule(x, 0))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, 0))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, 0))
    wm = models.WorldModel(0, config)
    ckpt_path = pathlib.Path(logdir) / "latest_model.pt"
    state = torch.load(ckpt_path, map_location=device)
    # The checkpoint is a Dreamer agent state_dict; world model keys are prefixed "_wm."
    wm_sd = {}
    for k, v in state.items():
        if k.startswith("_wm."):
            wm_sd[k[len("_wm."):]] = v
    missing, unexpected = wm.load_state_dict(wm_sd, strict=False)
    if missing:
        print(f"  Missing ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  Unexpected ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    wm.to(device).eval()
    return wm, config


# ---------------------------------------------------------------- features --
def extract_feats(state, kind):
    """Concatenate latent fields. kind ∈ {'controllable','combined'}.
    state[k] shape: [B, T, D] (or [B, D] for single step)."""
    if kind == "controllable":
        return torch.cat([state["stoch"], state["deter"]], dim=-1)
    if kind == "combined":
        return torch.cat([state["stoch"], state["deter"],
                          state["stoch_free"], state["deter_free"]], dim=-1)
    raise ValueError(kind)


@torch.no_grad()
def encode_h0(wm, config, images, actions, batch_size=16):
    """Returns dict {'controllable': (N*T, 250), 'combined': (N*T, 500)}.
    N = num episodes, T = T+1 - 1 = length."""
    device = config.device
    Nep, Tp1 = images.shape[:2]
    feats = {"controllable": [], "combined": []}
    for bs in tqdm(range(0, Nep, batch_size), desc="H0 observe"):
        be = min(bs + batch_size, Nep)
        bi = images[bs:be]            # (B, T+1, 64, 64, 3)
        ba = actions[bs:be]           # (B, T, 4)
        # Pad action with zero at t=0 to align with embed
        zeros = np.zeros((be - bs, 1, 4), dtype=np.float32)
        ba_full = np.concatenate([zeros, ba], axis=1)        # (B, T+1, 4)
        img_t = torch.tensor(bi, device=device, dtype=torch.float32) / 255.0 - 0.5
        act_t = torch.tensor(ba_full, device=device, dtype=torch.float32)
        embed = wm.encoder({"image": img_t}, act_t)          # (B, T+1, 2*emb)
        post, _ = wm.dynamics.observe(embed, act_t)          # post[k]: (B, T+1, ...)
        for kind in feats:
            f = extract_feats(post, kind)                    # (B, T+1, D)
            feats[kind].append(f.cpu().numpy())
    return {k: np.concatenate(v, 0).reshape(-1, np.concatenate(v, 0).shape[-1])
            for k, v in feats.items()}


@torch.no_grad()
def encode_imagined(wm, config, images, actions, horizon=15, batch_size=16):
    """For each timestep t we
      (a) obs_step to get post_t,
      (b) roll out img_step h times with random actions,
      (c) record features at h=0..horizon and the actions used.
    Returns:
      real_feats: dict[kind] -> (N*T, D)
      imag_feats: dict[kind] -> (H, N*T, D)
      imag_acts:  (H, N*T) discrete action indices used during imagination.
    """
    device = config.device
    Nep, Tp1 = images.shape[:2]
    real = {"controllable": [], "combined": []}
    imag = {"controllable": [[] for _ in range(horizon)],
            "combined":     [[] for _ in range(horizon)]}
    acts_used = [[] for _ in range(horizon)]
    rng = np.random.RandomState(0)
    for bs in tqdm(range(0, Nep, batch_size), desc="Imagine"):
        be = min(bs + batch_size, Nep)
        B = be - bs
        bi = images[bs:be]
        ba = actions[bs:be]
        zeros = np.zeros((B, 1, 4), dtype=np.float32)
        ba_full = np.concatenate([zeros, ba], axis=1)
        img_t = torch.tensor(bi, device=device, dtype=torch.float32) / 255.0 - 0.5
        act_t = torch.tensor(ba_full, device=device, dtype=torch.float32)
        embed = wm.encoder({"image": img_t}, act_t)
        post, _ = wm.dynamics.observe(embed, act_t)          # (B, T+1, *)
        for kind in real:
            real[kind].append(extract_feats(post, kind).cpu().numpy())
        # Step-by-step imagination
        # Slice post into per-timestep states for img_step rollouts
        T = Tp1
        local_imag = {kind: [[] for _ in range(horizon)] for kind in imag}
        local_acts = [[] for _ in range(horizon)]
        for t in range(T):
            cur = {k: v[:, t] for k, v in post.items()}      # (B, *)
            # random imagined actions, shared across batch dim B
            for h in range(horizon):
                ai = rng.randint(0, 4, size=B)
                a = np.zeros((B, 4), dtype=np.float32)
                a[np.arange(B), ai] = 1.0
                cur = wm.dynamics.img_step(cur, torch.tensor(a, device=device))
                for kind in imag:
                    local_imag[kind][h].append(extract_feats(cur, kind).cpu().numpy())
                local_acts[h].append(ai)
            # Reset cur for the next timestep (will be overwritten anyway)
        # local_imag[kind][h] is a list of (B, D) -> stack to (T, B, D) -> (B, T, D) -> (B*T, D)
        for kind in imag:
            for h in range(horizon):
                arr = np.stack(local_imag[kind][h], axis=0)   # (T, B, D)
                imag[kind][h].append(arr.transpose(1, 0, 2).reshape(-1, arr.shape[-1]))
        for h in range(horizon):
            arr = np.stack(local_acts[h], axis=0)            # (T, B)
            acts_used[h].append(arr.transpose(1, 0).reshape(-1))

    real_feats = {kind: np.concatenate(v, 0).reshape(-1, v[0].shape[-1])
                  for kind, v in real.items()}
    imag_feats = {kind: np.stack([np.concatenate(hl, 0) for hl in imag[kind]], 0)
                  for kind in imag}
    imag_actions = np.stack([np.concatenate(al, 0) for al in acts_used], 0)
    return real_feats, imag_feats, imag_actions


# ---------------------------------------------------------- imagined labels --
def compute_imagined_labels(start_labels, imagined_actions, horizon):
    """NoisyShape action logic: act 0 cycles color-1, act 1 color+1, act 2 shape_size-1,
    act 3 shape_size+1. shape_size is shape*3 + size in [0,12)."""
    out = {0: {"color": start_labels["color"].copy(),
               "shape": start_labels["shape"].copy(),
               "size":  start_labels["size"].copy()}}
    cc = start_labels["color"].copy().astype(np.int64)
    sc = (start_labels["shape"].astype(np.int64) * 3 +
          start_labels["size"].astype(np.int64))
    for h in range(horizon):
        a = imagined_actions[h]
        cc[a == 0] -= 1
        cc[a == 1] += 1
        sc[a == 2] -= 1
        sc[a == 3] += 1
        out[h + 1] = {
            "color": (cc % 4).astype(np.int64),
            "shape": ((sc % 12) // 3).astype(np.int64),
            "size":  ((sc % 12) % 3).astype(np.int64),
        }
    return out


# ------------------------------------------------------------------- probes --
class MLPClf(nn.Module):
    def __init__(self, d, c, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, c))

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
            return self.forward(x).argmax(-1).cpu().numpy()


class MLPReg(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
            return self.forward(x).cpu().numpy().squeeze(-1)


def train_clf(feats, labels, n_cls, device, epochs=100, patience=10):
    Xtr, Xte, ytr, yte = train_test_split(
        feats, labels, test_size=0.3, random_state=42, stratify=labels)
    m = MLPClf(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    Xv = torch.tensor(Xte, dtype=torch.float32, device=device)
    yv = torch.tensor(yte, dtype=torch.long, device=device)
    ld = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True)
    bl, bs2, ni = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for xb, yb in ld:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            tl = crit(m(Xv), yv).item()
        if tl < bl:
            bl = tl; bs2 = {k: v.clone() for k, v in m.state_dict().items()}; ni = 0
        else:
            ni += 1
        if ni >= patience:
            break
    if bs2:
        m.load_state_dict(bs2)
    m.eval()
    with torch.no_grad():
        tea = (m(Xv).argmax(-1) == yv).float().mean().item()
    return tea, m


def train_reg(feats, labels, device, epochs=100, patience=10):
    Xtr, Xte, ytr, yte = train_test_split(
        feats, labels, test_size=0.3, random_state=42)
    m = MLPReg(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device).unsqueeze(-1)
    Xv = torch.tensor(Xte, dtype=torch.float32, device=device)
    yv = torch.tensor(yte, dtype=torch.float32, device=device).unsqueeze(-1)
    ld = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True)
    bl, bs2, ni = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for xb, yb in ld:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            tl = crit(m(Xv), yv).item()
        if tl < bl:
            bl = tl; bs2 = {k: v.clone() for k, v in m.state_dict().items()}; ni = 0
        else:
            ni += 1
        if ni >= patience:
            break
    if bs2:
        m.load_state_dict(bs2)
    m.eval()
    with torch.no_grad():
        tem = crit(m(Xv), yv).item()
    return tem, m


# ----------------------------------------------------------- evaluate model --
def evaluate_with_kind(kind, feats_h0, labels_curr, real_feats, imag_feats,
                       labels_imag, imag_actions, horizon, device):
    f_curr = labels_curr
    f_imag = labels_imag
    res = {"current_probes": {}, "imagined_probes": {}}

    # H0 (current)
    print(f"  [{kind}] H0 probes")
    for nm, nc in [("color", 4), ("shape", 4), ("size", 3)]:
        tea, _ = train_clf(feats_h0, f_curr[nm].reshape(-1), nc, device)
        res["current_probes"][nm] = tea
        print(f"    {nm:8s}: acc={tea:.4f}")
    for nm in ["pos_x", "pos_y"]:
        tem, _ = train_reg(feats_h0, f_curr[nm].reshape(-1), device)
        res["current_probes"][nm] = tem
        print(f"    {nm:8s}: mse={tem:.6f}")

    # Imagined: train probe on real_feats encoded from imag-data trajectories
    print(f"  [{kind}] H0..H{horizon} imagined probes")
    f_imag_flat = {k: v.reshape(-1) for k, v in f_imag.items()}
    corrected = compute_imagined_labels(f_imag_flat, imag_actions, horizon)
    clfs, regs = {}, {}
    res["imagined_probes"]["0"] = {}
    for nm, nc in [("color", 4), ("shape", 4), ("size", 3)]:
        tea, m = train_clf(real_feats, f_imag_flat[nm], nc, device)
        clfs[nm] = m; res["imagined_probes"]["0"][nm] = tea
    for nm in ["pos_x", "pos_y"]:
        tem, m = train_reg(real_feats, f_imag_flat[nm], device)
        regs[nm] = m; res["imagined_probes"]["0"][nm] = tem
    for h in range(horizon):
        hf = imag_feats[h]
        d = {}
        for nm in ["color", "shape", "size"]:
            pred = clfs[nm].predict(hf)
            d[nm] = float(accuracy_score(corrected[h + 1][nm], pred))
        for nm in ["pos_x", "pos_y"]:
            pred = regs[nm].predict(hf)
            d[nm] = float(mean_squared_error(f_imag_flat[nm], pred))
        res["imagined_probes"][str(h + 1)] = d
    return res


# ------------------------------------------------------------------- main ---
def main():
    global DAVIS_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR"),
                        help="DAVIS 2017 JPEGImages/480p directory (default: $DAVIS_DIR)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num_curr", type=int, default=200)
    parser.add_argument("--num_imag", type=int, default=100)
    parser.add_argument("--out_name", default="results.json")
    args = parser.parse_args()
    if not args.davis_path or not os.path.isdir(args.davis_path):
        raise SystemExit("Set --davis_path or $DAVIS_DIR to the DAVIS 2017 JPEGImages/480p directory.")
    DAVIS_PATH = args.davis_path

    logdir = pathlib.Path(args.logdir)
    out_dir = logdir / "probe_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading world model from {logdir}")
    wm, config = load_world_model(logdir, args.device)

    print("Collecting current trajectories...")
    ci, ca, cl = collect_trajectories(args.num_curr, env_seed=42)
    print(f"  current shape: images={ci.shape} actions={ca.shape}")

    print("Collecting imagine trajectories...")
    ii, ia, il = collect_trajectories(args.num_imag, env_seed=12345)
    print(f"  imagine shape: images={ii.shape} actions={ia.shape}")

    print("Encoding H0 features (current)...")
    feats_h0 = encode_h0(wm, config, ci, ca, batch_size=16)
    print(f"  controllable={feats_h0['controllable'].shape} "
          f"combined={feats_h0['combined'].shape}")

    print("Encoding + imagining (imagine traj)...")
    real_feats, imag_feats, imag_actions = encode_imagined(
        wm, config, ii, ia, horizon=args.horizon, batch_size=16)
    print(f"  real combined: {real_feats['combined'].shape}, "
          f"imag combined: {imag_feats['combined'].shape}, "
          f"imag_actions: {imag_actions.shape}")

    results = {}
    for kind in ["controllable", "combined"]:
        print(f"\n=== Probes: feature kind = {kind} ===")
        results[kind] = evaluate_with_kind(
            kind, feats_h0[kind], cl,
            real_feats[kind], imag_feats[kind], il, imag_actions,
            horizon=args.horizon, device=args.device)

    out_path = out_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump({str(logdir): results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
